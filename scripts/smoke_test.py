import asyncio
import json
from typing import cast

from httpx import ASGITransport, AsyncClient

from app.main import app


async def request(client: AsyncClient, payload: dict[str, object]) -> dict[str, object]:
    response = await client.post("/api/v1/chat", json=payload)
    print(f"HTTP {response.status_code}")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    response.raise_for_status()
    return cast(dict[str, object], response.json())


async def main() -> None:
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://smoke-test"
        ) as client:
            await request(client, {
                "user_id": "user-1",
                "conversation_id": "live-rag-1",
                "message": "退款一般多久可以到账？",
            })
            pending = await request(client, {
                "user_id": "user-1",
                "conversation_id": "live-return-1",
                "message": "查询订单 A123，如果已经签收并且可以退货，就申请退货",
            })
            await request(client, {
                "user_id": "user-1",
                "conversation_id": "live-return-1",
                "message": "查询订单 A123，如果已经签收并且可以退货，就申请退货",
                "confirmation_token": pending["confirmation_token"],
            })


if __name__ == "__main__":
    asyncio.run(main())
