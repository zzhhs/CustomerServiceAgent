from typing import NotRequired, TypedDict

from app.models import ConversationMessage, PendingUserInput, RoutePlan, TaskResult


class CustomerServiceState(TypedDict):
    user_id: str
    conversation_id: str
    user_input: str
    raw_user_input: NotRequired[str]
    conversation_messages: NotRequired[list[ConversationMessage]]
    pending_user_input: NotRequired[PendingUserInput | None]
    resumed_pending_input: NotRequired[bool]
    confirmation_authorized: bool
    confirmation_token: NotRequired[str]
    confirmed_order_id: NotRequired[str]
    route_plan: NotRequired[RoutePlan]
    task_results: NotRequired[list[TaskResult]]
    execution_history: NotRequired[list[TaskResult]]
    replan_count: NotRequired[int]
    final_response: NotRequired[str]
    requires_confirmation: NotRequired[bool]
    requires_input: NotRequired[bool]
    requested_fields: NotRequired[list[str]]
    issued_confirmation_token: NotRequired[str]
