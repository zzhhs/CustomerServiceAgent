import re
from typing import Protocol

from app.infrastructure import LanguageModel
from app.models import AgentName, RoutePlan, SubTask, TaskAction, TaskResult


class Planner(Protocol):
    async def create_plan(self, user_input: str) -> RoutePlan: ...

    async def replan(
        self,
        user_input: str,
        *,
        previous_plan: RoutePlan,
        results: list[TaskResult],
    ) -> RoutePlan: ...


class DeterministicPlanner:
    """Safe local planner used for development, tests, and predictable fallbacks."""

    async def create_plan(self, user_input: str) -> RoutePlan:
        normalized = user_input.casefold()
        order_intent = any(word in normalized for word in ("订单", "物流", "order"))
        after_sales_intent = has_after_sales_action(normalized)

        tasks: list[SubTask] = []
        if order_intent or after_sales_intent:
            tasks.append(SubTask(
                id="task_1", agent=AgentName.ORDER, action=TaskAction.QUERY_ORDER,
                instruction=user_input,
            ))
        if after_sales_intent:
            tasks.append(SubTask(
                id=f"task_{len(tasks) + 1}", agent=AgentName.AFTER_SALES,
                action=TaskAction.APPLY_AFTER_SALES, instruction=user_input,
                depends_on=["task_1"],
            ))
        if not tasks:
            tasks.append(SubTask(
                id="task_1", agent=AgentName.QA, action=TaskAction.ANSWER_QUESTION,
                instruction=user_input,
            ))
        return RoutePlan(tasks=tasks, requires_confirmation=after_sales_intent)

    async def replan(
        self,
        user_input: str,
        *,
        previous_plan: RoutePlan,
        results: list[TaskResult],
    ) -> RoutePlan:
        return await self.create_plan(user_input)


class ModelPlanner:
    def __init__(self, language_model: LanguageModel) -> None:
        self._language_model = language_model

    async def create_plan(self, user_input: str) -> RoutePlan:
        return await self._language_model.create_plan(user_input)

    async def replan(
        self,
        user_input: str,
        *,
        previous_plan: RoutePlan,
        results: list[TaskResult],
    ) -> RoutePlan:
        return await self._language_model.replan(
            user_input, previous_plan=previous_plan, results=results
        )


def extract_order_id(text: str) -> str | None:
    match = re.search(r"\b[A-Za-z][A-Za-z0-9-]{2,31}\b", text)
    return match.group(0).upper() if match else None


def has_after_sales_action(text: str) -> bool:
    normalized = text.casefold()
    return any(
        phrase in normalized
        for phrase in (
            "申请退货",
            "申请退款",
            "我要退货",
            "我要退款",
            "帮我退",
            "办理退",
            "发起退",
            "换货",
            "apply for a return",
            "request a refund",
        )
    )
