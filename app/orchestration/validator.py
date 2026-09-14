from collections import deque

from app.models import AgentName, RoutePlan, TaskAction
from app.orchestration.planner import has_after_sales_action


class InvalidPlanError(ValueError):
    pass


class PlanValidator:
    _allowed_actions = {
        AgentName.QA: {TaskAction.ANSWER_QUESTION},
        AgentName.ORDER: {TaskAction.QUERY_ORDER},
        AgentName.AFTER_SALES: {TaskAction.APPLY_AFTER_SALES},
    }

    def __init__(self, *, max_tasks: int = 6) -> None:
        self._max_tasks = max_tasks

    def validate(self, plan: RoutePlan, *, user_input: str | None = None) -> RoutePlan:
        if len(plan.tasks) > self._max_tasks:
            raise InvalidPlanError(f"Plan exceeds the {self._max_tasks}-task limit")

        task_by_id = {task.id: task for task in plan.tasks}
        if len(task_by_id) != len(plan.tasks):
            raise InvalidPlanError("Task ids must be unique")

        indegree = {task.id: 0 for task in plan.tasks}
        dependents: dict[str, list[str]] = {task.id: [] for task in plan.tasks}
        has_after_sales = False

        for task in plan.tasks:
            if task.action not in self._allowed_actions[task.agent]:
                raise InvalidPlanError(f"Action {task.action} is not allowed for {task.agent}")
            if task.id in task.depends_on:
                raise InvalidPlanError(f"Task {task.id} cannot depend on itself")
            for dependency in task.depends_on:
                if dependency not in task_by_id:
                    raise InvalidPlanError(f"Unknown dependency: {dependency}")
                indegree[task.id] += 1
                dependents[dependency].append(task.id)
            if task.agent is AgentName.AFTER_SALES:
                has_after_sales = True
                if not any(task_by_id[dep].agent is AgentName.ORDER for dep in task.depends_on):
                    raise InvalidPlanError("After-sales tasks must depend on an order task")

        queue = deque(task_id for task_id, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            task_id = queue.popleft()
            visited += 1
            for dependent in dependents[task_id]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
        if visited != len(plan.tasks):
            raise InvalidPlanError("Task dependencies contain a cycle")
        if has_after_sales and not plan.requires_confirmation:
            raise InvalidPlanError("After-sales writes must require confirmation")
        if user_input is not None:
            self._validate_intent_coverage(plan, user_input)
        return plan

    @staticmethod
    def _validate_intent_coverage(plan: RoutePlan, user_input: str) -> None:
        normalized = user_input.casefold()
        asks_after_sales = has_after_sales_action(normalized)
        asks_order = asks_after_sales or any(
            word in normalized for word in ("订单", "物流", "order")
        )
        agents = {task.agent for task in plan.tasks}
        if asks_order and AgentName.ORDER not in agents:
            raise InvalidPlanError("Plan omitted the requested order task")
        if asks_after_sales and AgentName.AFTER_SALES not in agents:
            raise InvalidPlanError("Plan omitted the requested after-sales task")
