import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from app.config.settings import Settings
from app.main import create_app
from app.security import JwtBearerAuthenticator

app = create_app(Settings(llm_provider="deterministic"))


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as test_client:
            yield test_client


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.3.0"}


async def test_composite_request_requires_confirmation(client: AsyncClient) -> None:
    response = await client.post("/api/v1/chat", json={
        "user_id": "user-1",
        "conversation_id": "conversation-1",
        "message": "查询订单 A123，如果已签收就申请退货",
    })

    assert response.status_code == 200
    body = response.json()
    assert body["requires_confirmation"] is True
    assert body["confirmation_token"]
    assert [task["agent"] for task in body["plan"]["tasks"]] == ["order", "after_sales"]
    assert body["results"][1]["status"] == "pending_confirmation"


async def test_confirmed_request_creates_idempotent_ticket(client: AsyncClient) -> None:
    initial_payload = {
        "user_id": "user-1",
        "conversation_id": "conversation-1",
        "message": "查询订单 A123，如果已签收就申请退货",
    }
    pending = await client.post("/api/v1/chat", json=initial_payload)
    token = pending.json()["confirmation_token"]
    first = await client.post(
        "/api/v1/chat", json={**initial_payload, "confirmation_token": token}
    )

    assert first.status_code == 200
    assert first.json()["requires_confirmation"] is False
    assert first.json()["results"][1]["status"] == "succeeded"
    first_ticket = first.json()["results"][1]["data"]["ticket_id"]

    second_pending = await client.post("/api/v1/chat", json={
        **initial_payload,
        "conversation_id": "conversation-idempotency-2",
    })
    second = await client.post("/api/v1/chat", json={
        **initial_payload,
        "conversation_id": "conversation-idempotency-2",
        "confirmation_token": second_pending.json()["confirmation_token"],
    })
    assert second.json()["results"][1]["data"]["ticket_id"] == first_ticket

    reused = await client.post(
        "/api/v1/chat", json={**initial_payload, "confirmation_token": token}
    )
    assert reused.status_code == 409


async def test_client_cannot_self_confirm(client: AsyncClient) -> None:
    response = await client.post("/api/v1/chat", json={
        "user_id": "user-1",
        "conversation_id": "conversation-bypass",
        "message": "查询订单 A123 并申请退货",
        "confirmed": True,
    })
    assert response.status_code == 422


async def test_confirmation_is_bound_to_user_and_conversation(client: AsyncClient) -> None:
    pending = await client.post("/api/v1/chat", json={
        "user_id": "user-1",
        "conversation_id": "conversation-binding",
        "message": "查询订单 A123 并申请退货",
    })
    token = pending.json()["confirmation_token"]
    response = await client.post("/api/v1/chat", json={
        "user_id": "user-2",
        "conversation_id": "conversation-binding",
        "message": "查询订单 C789 并申请退货",
        "confirmation_token": token,
    })
    assert response.status_code == 409
    accepted = await client.post("/api/v1/chat", json={
        "user_id": "user-1",
        "conversation_id": "conversation-binding",
        "message": "确认",
        "confirmation_token": token,
    })
    assert accepted.status_code == 200


async def test_confirmation_token_is_atomically_single_use(client: AsyncClient) -> None:
    payload = {
        "user_id": "user-1",
        "conversation_id": "conversation-concurrent",
        "message": "查询订单 A123 并申请退货",
    }
    pending = await client.post("/api/v1/chat", json=payload)
    confirmed_payload = {
        **payload,
        "confirmation_token": pending.json()["confirmation_token"],
    }
    first, second = await asyncio.gather(
        client.post("/api/v1/chat", json=confirmed_payload),
        client.post("/api/v1/chat", json=confirmed_payload),
    )
    assert sorted((first.status_code, second.status_code)) == [200, 409]


async def test_plain_question_routes_to_qa(client: AsyncClient) -> None:
    response = await client.post("/api/v1/chat", json={
        "user_id": "user-1",
        "conversation_id": "conversation-2",
        "message": "你们的营业时间是什么？",
    })

    assert response.status_code == 200
    assert response.json()["plan"]["tasks"][0]["agent"] == "qa"


async def test_missing_order_id_can_be_supplied_in_next_turn(client: AsyncClient) -> None:
    first = await client.post("/api/v1/chat", json={
        "user_id": "user-1",
        "conversation_id": "conversation-slot-filling",
        "message": "我要申请退货",
    })

    assert first.status_code == 200
    assert first.json()["requires_input"] is True
    assert first.json()["requested_fields"] == ["order_id"]
    assert first.json()["results"][0]["status"] == "needs_input"

    resumed = await client.post("/api/v1/chat", json={
        "user_id": "user-1",
        "conversation_id": "conversation-slot-filling",
        "message": "订单号是 A123",
    })

    body = resumed.json()
    assert resumed.status_code == 200
    assert body["requires_input"] is False
    assert body["requires_confirmation"] is True
    assert body["metadata"]["resumed_pending_input"] is True
    assert body["metadata"]["context_message_count"] == 2
    assert [task["agent"] for task in body["plan"]["tasks"]] == ["order", "after_sales"]


async def test_signed_auth_ignores_body_user_id() -> None:
    secret = "test-secret-that-is-long-enough-for-hmac"
    signed_app = create_app(Settings(
        llm_provider="deterministic", auth_mode="jwt", auth_secret=secret
    ))
    token = JwtBearerAuthenticator.issue(
        user_id="user-1",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
        secret=secret,
        issuer="customer-service-auth",
        audience="customer-service-api",
    )
    async with signed_app.router.lifespan_context(signed_app):
        async with AsyncClient(
            transport=ASGITransport(app=signed_app), base_url="http://test"
        ) as signed_client:
            response = await signed_client.post(
                "/api/v1/chat",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "user_id": "user-2",
                    "conversation_id": "signed-auth",
                    "message": "查询订单 A123",
                },
            )
    assert response.status_code == 200
    assert response.json()["results"][0]["data"]["user_id"] == "user-1"


def test_production_rejects_disabled_authentication() -> None:
    with pytest.raises(ValueError, match="not allowed in production"):
        Settings(app_env="production", llm_provider="deterministic")


async def test_telemetry_adds_trace_correlation_header() -> None:
    telemetry_app = create_app(Settings(
        llm_provider="deterministic",
        telemetry_enabled=True,
        otel_exporter="none",
    ))
    async with telemetry_app.router.lifespan_context(telemetry_app):
        async with AsyncClient(
            transport=ASGITransport(app=telemetry_app), base_url="http://test"
        ) as telemetry_client:
            response = await telemetry_client.get("/health")
    assert response.status_code == 200
    assert len(response.headers["X-Trace-ID"]) == 32
