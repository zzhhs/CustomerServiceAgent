from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentName(StrEnum):
    QA = "qa"
    ORDER = "order"
    AFTER_SALES = "after_sales"


class TaskAction(StrEnum):
    ANSWER_QUESTION = "answer_question"
    QUERY_ORDER = "query_order"
    APPLY_AFTER_SALES = "apply_after_sales"


class TaskStatus(StrEnum):
    SUCCEEDED = "succeeded"
    NEEDS_INPUT = "needs_input"
    PENDING_CONFIRMATION = "pending_confirmation"
    REJECTED = "rejected"
    FAILED = "failed"


class SubTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, Field(pattern=r"^task_[1-9][0-9]*$")]
    agent: AgentName
    action: TaskAction
    instruction: Annotated[str, Field(min_length=1, max_length=500)]
    depends_on: list[str] = Field(default_factory=list, max_length=10)


class RoutePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: Annotated[list[SubTask], Field(min_length=1, max_length=20)]
    requires_confirmation: bool = False


class QAResult(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)


class OrderResult(BaseModel):
    order_id: str
    user_id: str
    status: Literal["CREATED", "PAID", "SHIPPED", "DELIVERED", "CANCELLED"]
    refundable: bool


class AfterSalesResult(BaseModel):
    order_id: str
    eligible: bool
    ticket_id: str | None = None
    reason: str | None = None


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    error_code: str | None = None
    retryable: bool = False
    attempts: int = Field(default=1, ge=0, le=10)


class KnowledgeDocument(BaseModel):
    id: str
    title: str
    content: str
    source: str


class KnowledgeHit(BaseModel):
    document: KnowledgeDocument
    score: float = Field(ge=0, le=1)


class TicketRecord(BaseModel):
    ticket_id: str
    order_id: str
    user_id: str
    operation: str
    idempotency_key: str


class PendingAction(BaseModel):
    """Server-side record for a write operation awaiting explicit confirmation."""

    token_hash: str
    user_id: str
    conversation_id: str
    user_input: str
    plan: RoutePlan
    order_id: str
    operation: Literal["return"]
    expires_at: datetime
    consumed_at: datetime | None = None


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class PendingUserInput(BaseModel):
    original_user_input: str
    missing_fields: list[str]


class ConversationContext(BaseModel):
    user_id: str
    conversation_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    pending_input: PendingUserInput | None = None


class Eligibility(BaseModel):
    allowed: bool
    reason: str | None = None


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["tool", "finish", "need_input"]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "AgentDecision":
        if self.kind == "tool" and not self.tool_name:
            raise ValueError("tool decisions require tool_name")
        if self.kind != "tool" and self.tool_name is not None:
            raise ValueError("only tool decisions may include tool_name")
        if self.kind in {"finish", "need_input"} and not self.message:
            raise ValueError(f"{self.kind} decisions require message")
        return self
