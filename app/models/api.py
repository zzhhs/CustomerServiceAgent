from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.domain import RoutePlan, TaskResult


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    conversation_id: Annotated[str, Field(min_length=1, max_length=128)]
    message: Annotated[str, Field(min_length=1, max_length=4000)]
    confirmation_token: Annotated[str, Field(min_length=20, max_length=512)] | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    requires_confirmation: bool
    requires_input: bool = False
    requested_fields: list[str] = Field(default_factory=list)
    confirmation_token: str | None = None
    plan: RoutePlan
    results: list[TaskResult]
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
