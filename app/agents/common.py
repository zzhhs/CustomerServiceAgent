from typing import Any

from app.infrastructure import ModelGatewayError
from app.models import AgentDecision


def require_tool(decision: AgentDecision, allowed_tool: str) -> dict[str, Any]:
    if decision.kind != "tool" or decision.tool_name != allowed_tool:
        raise ModelGatewayError(f"Specialist agent must call {allowed_tool}")
    return decision.arguments


def require_finish(decision: AgentDecision) -> str:
    if decision.kind == "need_input":
        raise ValueError(decision.message or "需要补充信息。")
    if decision.kind != "finish":
        raise ModelGatewayError("Specialist agent did not finish after observing its tool result")
    return decision.message or "处理完成。"

