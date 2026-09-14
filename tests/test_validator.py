import pytest

from app.models import AgentName, RoutePlan, SubTask, TaskAction
from app.orchestration.validator import InvalidPlanError, PlanValidator


def test_validator_rejects_after_sales_without_confirmation() -> None:
    plan = RoutePlan(tasks=[
        SubTask(
            id="task_1", agent=AgentName.ORDER, action=TaskAction.QUERY_ORDER,
            instruction="query",
        ),
        SubTask(
            id="task_2", agent=AgentName.AFTER_SALES,
            action=TaskAction.APPLY_AFTER_SALES, instruction="return",
            depends_on=["task_1"],
        ),
    ], requires_confirmation=False)

    with pytest.raises(InvalidPlanError, match="must require confirmation"):
        PlanValidator().validate(plan)


def test_validator_rejects_cycles() -> None:
    plan = RoutePlan(tasks=[
        SubTask(
            id="task_1", agent=AgentName.QA, action=TaskAction.ANSWER_QUESTION,
            instruction="one", depends_on=["task_2"],
        ),
        SubTask(
            id="task_2", agent=AgentName.QA, action=TaskAction.ANSWER_QUESTION,
            instruction="two", depends_on=["task_1"],
        ),
    ])

    with pytest.raises(InvalidPlanError, match="cycle"):
        PlanValidator().validate(plan)


def test_validator_rejects_plan_that_omits_requested_after_sales() -> None:
    plan = RoutePlan(tasks=[
        SubTask(
            id="task_1", agent=AgentName.ORDER, action=TaskAction.QUERY_ORDER,
            instruction="query A123",
        ),
    ])

    with pytest.raises(InvalidPlanError, match="omitted the requested after-sales"):
        PlanValidator().validate(plan, user_input="查询订单 A123 并申请退货")
