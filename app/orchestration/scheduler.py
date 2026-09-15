import asyncio
import logging
import random
from time import perf_counter

from app.agents import AfterSalesAgent, ConfirmationRequired, OrderAgent, QAAgent
from app.errors import MissingInputError, RetryableOperationError
from app.models import (
    AfterSalesResult,
    AgentName,
    OrderResult,
    QAResult,
    RoutePlan,
    SubTask,
    TaskResult,
    TaskStatus,
)
from app.observability import Observability

DomainResult = QAResult | OrderResult | AfterSalesResult


class TaskScheduler:
    def __init__(
        self,
        *,
        qa_agent: QAAgent,
        order_agent: OrderAgent,
        after_sales_agent: AfterSalesAgent,
        observability: Observability,
        max_attempts: int = 3,
        retry_base_delay_seconds: float = 0.25,
    ) -> None:
        self._qa_agent = qa_agent
        self._order_agent = order_agent
        self._after_sales_agent = after_sales_agent
        self._observability = observability
        self._max_attempts = max_attempts
        self._retry_base_delay = retry_base_delay_seconds
        self._logger = logging.getLogger("customer_service.scheduler")

    async def execute(
        self,
        *,
        plan: RoutePlan,
        user_id: str,
        user_input: str,
        confirmation_authorized: bool,
        expected_order_id: str | None = None,
    ) -> list[TaskResult]:
        results_by_id: dict[str, TaskResult] = {}
        values: dict[str, DomainResult] = {}
        remaining = {task.id: task for task in plan.tasks}
        plan_order = {task.id: index for index, task in enumerate(plan.tasks)}
        completed: set[str] = set()
        running: dict[
            asyncio.Task[tuple[TaskResult, DomainResult | None]], SubTask
        ] = {}
        stop_scheduling = False

        try:
            while remaining or running:
                if not stop_scheduling:
                    ready = [
                        task
                        for task in remaining.values()
                        if set(task.depends_on) <= completed
                    ]
                    ready.sort(key=lambda task: task.id)
                    if ready:
                        for task in ready:
                            del remaining[task.id]
                            failed_dependency = any(
                                results_by_id[dependency].status
                                is not TaskStatus.SUCCEEDED
                                for dependency in task.depends_on
                            )
                            if failed_dependency:
                                result = TaskResult(
                                    task_id=task.id,
                                    status=TaskStatus.FAILED,
                                    error="A dependency did not complete successfully",
                                    error_code="dependency_failed",
                                    attempts=0,
                                )
                                results_by_id[task.id] = result
                                completed.add(task.id)
                                self._record_result(task, result)
                                continue
                            future = asyncio.create_task(
                                self._execute_with_retry(
                                    task=task,
                                    values=values,
                                    user_id=user_id,
                                    user_input=user_input,
                                    confirmation_authorized=confirmation_authorized,
                                    expected_order_id=expected_order_id,
                                ),
                                name=f"customer-service-{task.id}",
                            )
                            running[future] = task
                        continue
                    if not running and remaining:
                        raise ValueError("Plan cannot be scheduled")

                if not running:
                    break

                done, _ = await asyncio.wait(
                    running, return_when=asyncio.FIRST_COMPLETED
                )
                for future in sorted(done, key=lambda item: running[item].id):
                    task = running.pop(future)
                    result, value = future.result()
                    if value is not None:
                        values[task.id] = value
                    results_by_id[task.id] = result
                    completed.add(task.id)
                    self._record_result(task, result)
                    if result.status is TaskStatus.NEEDS_INPUT:
                        stop_scheduling = True
        finally:
            if running:
                for future in running:
                    future.cancel()
                await asyncio.gather(*running, return_exceptions=True)

        return sorted(
            results_by_id.values(), key=lambda result: plan_order[result.task_id]
        )

    async def _execute_with_retry(
        self,
        *,
        task: SubTask,
        values: dict[str, DomainResult],
        user_id: str,
        user_input: str,
        confirmation_authorized: bool,
        expected_order_id: str | None,
    ) -> tuple[TaskResult, DomainResult | None]:
        for attempt in range(1, self._max_attempts + 1):
            started = perf_counter()
            try:
                with self._observability.span(
                    "agent.task",
                    {
                        "task.id": task.id,
                        "agent.name": task.agent.value,
                        "task.action": task.action.value,
                        "task.attempt": attempt,
                    },
                ):
                    value = await self._dispatch(
                        task=task,
                        values=values,
                        user_id=user_id,
                        user_input=user_input,
                        confirmation_authorized=confirmation_authorized,
                        expected_order_id=expected_order_id,
                    )
                return (
                    TaskResult(
                        task_id=task.id,
                        status=TaskStatus.SUCCEEDED,
                        data=value.model_dump(mode="json"),
                        attempts=attempt,
                    ),
                    value,
                )
            except ConfirmationRequired as exc:
                return (
                    TaskResult(
                        task_id=task.id,
                        status=TaskStatus.PENDING_CONFIRMATION,
                        error=str(exc),
                        error_code="confirmation_required",
                        attempts=attempt,
                    ),
                    None,
                )
            except MissingInputError as exc:
                return (
                    TaskResult(
                        task_id=task.id,
                        status=TaskStatus.NEEDS_INPUT,
                        data={"missing_fields": exc.fields},
                        error=str(exc),
                        error_code="missing_input",
                        attempts=attempt,
                    ),
                    None,
                )
            except RetryableOperationError as exc:
                error_code = exc.error_code
                if attempt == self._max_attempts:
                    return (
                        TaskResult(
                            task_id=task.id,
                            status=TaskStatus.FAILED,
                            error=str(exc),
                            error_code=error_code,
                            retryable=True,
                            attempts=attempt,
                        ),
                        None,
                    )
                self._record_retry(task, attempt, error_code)
                delay = random.uniform(0, self._retry_base_delay * (2 ** (attempt - 1)))
                await asyncio.sleep(delay)
            except LookupError as exc:
                return (
                    TaskResult(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        error=str(exc),
                        error_code="not_found",
                        attempts=attempt,
                    ),
                    None,
                )
            except ValueError as exc:
                return (
                    TaskResult(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        error=str(exc),
                        error_code="invalid_input",
                        attempts=attempt,
                    ),
                    None,
                )
            finally:
                self._observability.task_duration.record(
                    perf_counter() - started,
                    {"agent": task.agent.value, "action": task.action.value},
                )
        raise AssertionError("retry loop must return")

    async def _dispatch(
        self,
        *,
        task: SubTask,
        values: dict[str, DomainResult],
        user_id: str,
        user_input: str,
        confirmation_authorized: bool,
        expected_order_id: str | None,
    ) -> DomainResult:
        if task.agent is AgentName.QA:
            return await self._qa_agent.run(task.instruction)
        if task.agent is AgentName.ORDER:
            order = await self._order_agent.run(
                instruction=task.instruction,
                user_input=user_input,
                user_id=user_id,
            )
            if expected_order_id is not None and order.order_id != expected_order_id:
                raise ValueError("确认操作中的订单与待确认订单不一致")
            return order
        order = self._find_order_dependency(task.depends_on, values)
        return await self._after_sales_agent.run(
            instruction=task.instruction,
            order=order,
            user_id=user_id,
            confirmation_authorized=confirmation_authorized,
        )

    def _record_retry(self, task: SubTask, attempt: int, error_code: str) -> None:
        attributes = {
            "agent": task.agent.value,
            "action": task.action.value,
            "error_code": error_code,
        }
        self._observability.retries.add(1, attributes)
        self._logger.warning(
            "retrying specialist task",
            extra={
                "event": "agent.task.retry",
                "task_id": task.id,
                "agent": task.agent.value,
            },
        )

    def _record_result(self, task: SubTask, result: TaskResult) -> None:
        self._observability.tasks.add(
            1,
            {
                "agent": task.agent.value,
                "action": task.action.value,
                "status": result.status.value,
                "error_code": result.error_code or "none",
            },
        )

    @staticmethod
    def _find_order_dependency(
        dependency_ids: list[str],
        values: dict[str, DomainResult],
    ) -> OrderResult:
        for task_id in dependency_ids:
            value = values.get(task_id)
            if isinstance(value, OrderResult):
                return value
        raise ValueError("After-sales execution requires an order result")
