import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import cast

from langgraph.graph import END, START, StateGraph

from app.infrastructure import ModelGatewayError
from app.models import RoutePlan, TaskResult, TaskStatus
from app.observability import Observability
from app.orchestration.planner import DeterministicPlanner, Planner
from app.orchestration.scheduler import TaskScheduler
from app.orchestration.state import CustomerServiceState
from app.orchestration.synthesizer import ResponseSynthesizer
from app.orchestration.validator import InvalidPlanError, PlanValidator
from app.repositories import PendingActionRepository


class CustomerServiceGraph:
    def __init__(
        self,
        *,
        planner: Planner,
        validator: PlanValidator,
        scheduler: TaskScheduler,
        synthesizer: ResponseSynthesizer,
        pending_actions: PendingActionRepository,
        observability: Observability,
        max_replans: int = 1,
        max_attempts: int = 3,
        retry_base_delay_seconds: float = 0.25,
    ) -> None:
        self._planner = planner
        self._validator = validator
        self._scheduler = scheduler
        self._synthesizer = synthesizer
        self._pending_actions = pending_actions
        self._observability = observability
        self._max_replans = max_replans
        self._max_attempts = max_attempts
        self._retry_base_delay = retry_base_delay_seconds
        self._fallback_planner = DeterministicPlanner()
        self._logger = logging.getLogger("customer_service.orchestrator")

        builder = StateGraph(CustomerServiceState)
        builder.add_node("prepare", self._prepare)
        builder.add_node("plan", self._plan)
        builder.add_node("validate", self._validate)
        builder.add_node("execute", self._execute)
        builder.add_node("replan", self._replan)
        builder.add_node("synthesize", self._synthesize)
        builder.add_edge(START, "prepare")
        builder.add_conditional_edges(
            "prepare", self._next_after_prepare, {"plan": "plan", "validate": "validate"}
        )
        builder.add_edge("plan", "validate")
        builder.add_edge("validate", "execute")
        builder.add_conditional_edges(
            "execute",
            self._next_after_execute,
            {"replan": "replan", "synthesize": "synthesize"},
        )
        builder.add_edge("replan", "validate")
        builder.add_edge("synthesize", END)
        self._graph = builder.compile()

    async def invoke(self, state: CustomerServiceState) -> CustomerServiceState:
        attributes = {
            "conversation.id": state["conversation_id"],
            "confirmation.present": "confirmation_token" in state,
        }
        self._observability.requests.add(1, {
            "confirmation": str("confirmation_token" in state).lower()
        })
        with self._observability.span("orchestration.invoke", attributes):
            result = await self._graph.ainvoke(state)
        return cast(CustomerServiceState, result)

    async def _prepare(self, state: CustomerServiceState) -> dict[str, object]:
        token = state.get("confirmation_token")
        if token is None:
            return {
                "confirmation_authorized": False,
                "execution_history": [],
                "replan_count": 0,
            }
        pending = await self._pending_actions.resolve(
            token=token,
            user_id=state["user_id"],
            conversation_id=state["conversation_id"],
        )
        return {
            "confirmation_authorized": True,
            "user_input": pending.user_input,
            "route_plan": pending.plan,
            "confirmed_order_id": pending.order_id,
            "execution_history": [],
            "replan_count": 0,
        }

    @staticmethod
    def _next_after_prepare(state: CustomerServiceState) -> str:
        return "validate" if state.get("route_plan") is not None else "plan"

    async def _plan(self, state: CustomerServiceState) -> dict[str, RoutePlan]:
        with self._observability.span("orchestration.plan"):
            plan = await self._with_model_retry(
                "planner.create", lambda: self._planner.create_plan(state["user_input"])
            )
        return {"route_plan": plan}

    async def _validate(self, state: CustomerServiceState) -> dict[str, RoutePlan]:
        with self._observability.span("orchestration.validate"):
            try:
                plan = self._validator.validate(
                    state["route_plan"], user_input=state["user_input"]
                )
            except InvalidPlanError:
                self._observability.replans.add(1, {"reason": "invalid_plan"})
                fallback = await self._fallback_planner.create_plan(state["user_input"])
                plan = self._validator.validate(fallback, user_input=state["user_input"])
        return {"route_plan": plan}

    async def _execute(
        self, state: CustomerServiceState
    ) -> dict[str, object]:
        with self._observability.span("orchestration.execute", {
            "orchestration.replan_count": state.get("replan_count", 0),
        }):
            results = await self._scheduler.execute(
                plan=state["route_plan"],
                user_id=state["user_id"],
                user_input=state["user_input"],
                confirmation_authorized=state["confirmation_authorized"],
                expected_order_id=state.get("confirmed_order_id"),
            )
        requires_confirmation = any(
            result.status is TaskStatus.PENDING_CONFIRMATION for result in results
        )
        update: dict[str, object] = {
            "task_results": results,
            "execution_history": [*state.get("execution_history", []), *results],
            "requires_confirmation": requires_confirmation,
        }
        if requires_confirmation:
            order_id = self._successful_order_id(results)
            update["issued_confirmation_token"] = await self._pending_actions.create(
                user_id=state["user_id"],
                conversation_id=state["conversation_id"],
                user_input=state["user_input"],
                plan=state["route_plan"],
                order_id=order_id,
            )
        return update

    def _next_after_execute(self, state: CustomerServiceState) -> str:
        if state["confirmation_authorized"]:
            return "synthesize"
        if any(
            result.status is TaskStatus.PENDING_CONFIRMATION
            for result in state["task_results"]
        ):
            return "synthesize"
        has_failure = any(
            result.status is TaskStatus.FAILED for result in state["task_results"]
        )
        if has_failure and state.get("replan_count", 0) < self._max_replans:
            return "replan"
        return "synthesize"

    async def _replan(self, state: CustomerServiceState) -> dict[str, object]:
        next_count = state.get("replan_count", 0) + 1
        attributes = {"orchestration.replan_count": next_count}
        with self._observability.span("orchestration.replan", attributes):
            self._observability.replans.add(1, {"reason": "execution_failure"})
            self._logger.warning(
                "replanning after execution failure",
                extra={
                    "event": "orchestration.replan",
                    "conversation_id": state["conversation_id"],
                },
            )
            try:
                plan = await self._with_model_retry(
                    "planner.replan",
                    lambda: self._planner.replan(
                        state["user_input"],
                        previous_plan=state["route_plan"],
                        results=state["task_results"],
                    ),
                )
            except ModelGatewayError:
                plan = await self._fallback_planner.replan(
                    state["user_input"],
                    previous_plan=state["route_plan"],
                    results=state["task_results"],
                )
        return {"route_plan": plan, "replan_count": next_count}

    async def _with_model_retry(
        self,
        operation: str,
        call: Callable[[], Awaitable[RoutePlan]],
    ) -> RoutePlan:
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await call()
            except ModelGatewayError:
                if attempt == self._max_attempts:
                    raise
                self._observability.retries.add(1, {
                    "component": "planner",
                    "operation": operation,
                })
                self._logger.warning(
                    "retrying planner call",
                    extra={"event": "planner.retry"},
                )
                delay = random.uniform(
                    0, self._retry_base_delay * (2 ** (attempt - 1))
                )
                await asyncio.sleep(delay)
        raise AssertionError("retry loop must return")

    @staticmethod
    def _successful_order_id(results: list[TaskResult]) -> str:
        for result in results:
            order_id = result.data.get("order_id")
            if result.status is TaskStatus.SUCCEEDED and isinstance(order_id, str):
                return order_id
        raise ValueError("Cannot issue confirmation without a verified order")

    async def _synthesize(self, state: CustomerServiceState) -> dict[str, str]:
        with self._observability.span("orchestration.synthesize"):
            text = await self._synthesizer.synthesize(
                user_input=state["user_input"], results=state["task_results"]
            )
        return {"final_response": text}
