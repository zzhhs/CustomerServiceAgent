from typing import NotRequired, TypedDict

from app.models import RoutePlan, TaskResult


class CustomerServiceState(TypedDict):
    user_id: str
    conversation_id: str
    user_input: str
    confirmation_authorized: bool
    confirmation_token: NotRequired[str]
    confirmed_order_id: NotRequired[str]
    route_plan: NotRequired[RoutePlan]
    task_results: NotRequired[list[TaskResult]]
    execution_history: NotRequired[list[TaskResult]]
    replan_count: NotRequired[int]
    final_response: NotRequired[str]
    requires_confirmation: NotRequired[bool]
    issued_confirmation_token: NotRequired[str]
