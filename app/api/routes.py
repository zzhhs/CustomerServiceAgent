from typing import cast

from fastapi import APIRouter, HTTPException, Request, status

from app.infrastructure import ModelGatewayError
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.orchestration.graph import CustomerServiceGraph
from app.orchestration.state import CustomerServiceState
from app.repositories import InvalidConfirmationError
from app.security import AuthenticationError, JwtBearerAuthenticator

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.3.0")


@router.post("/api/v1/chat", response_model=ChatResponse, tags=["customer-service"])
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    graph = cast(CustomerServiceGraph, request.app.state.customer_service_graph)
    try:
        user_id = _authenticated_user_id(payload.user_id, request)
        initial_state = CustomerServiceState(
            user_id=user_id,
            conversation_id=payload.conversation_id,
            user_input=payload.message,
            confirmation_authorized=False,
        )
        if payload.confirmation_token is not None:
            initial_state["confirmation_token"] = payload.confirmation_token
        result = await graph.invoke(initial_state)
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except InvalidConfirmationError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ModelGatewayError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The configured language model is temporarily unavailable",
        ) from exc
    return ChatResponse(
        conversation_id=payload.conversation_id,
        response=result["final_response"],
        requires_confirmation=result["requires_confirmation"],
        requires_input=result.get("requires_input", False),
        requested_fields=result.get("requested_fields", []),
        confirmation_token=result.get("issued_confirmation_token"),
        plan=result["route_plan"],
        results=result["task_results"],
        metadata={
            "llm_provider": request.app.state.llm_provider,
            "replan_count": result.get("replan_count", 0),
            "total_task_attempts": sum(
                task.attempts for task in result["execution_history"]
            ),
            "context_message_count": len(result.get("conversation_messages", [])),
            "resumed_pending_input": result.get("resumed_pending_input", False),
        },
    )


def _authenticated_user_id(payload_user_id: str | None, request: Request) -> str:
    if request.app.state.auth_mode == "jwt":
        authenticator = cast(JwtBearerAuthenticator, request.app.state.authenticator)
        return authenticator.authenticate(request.headers.get("Authorization"))
    if payload_user_id is None:
        raise AuthenticationError("本地模式必须提供 user_id。")
    return payload_user_id
