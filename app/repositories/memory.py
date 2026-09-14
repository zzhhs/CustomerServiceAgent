import hashlib
import secrets
from asyncio import Lock
from datetime import UTC, datetime, timedelta

from app.models import OrderResult, PendingAction, RoutePlan, TicketRecord


class InvalidConfirmationError(ValueError):
    pass


class InMemoryDatabase:
    """Process-local fake database seeded with representative development data."""

    def __init__(self) -> None:
        self.orders: dict[str, OrderResult] = {
            "A123": OrderResult(
                order_id="A123", user_id="user-1", status="DELIVERED", refundable=True
            ),
            "B456": OrderResult(
                order_id="B456", user_id="user-1", status="SHIPPED", refundable=False
            ),
            "C789": OrderResult(
                order_id="C789", user_id="user-2", status="DELIVERED", refundable=True
            ),
        }
        self.tickets_by_key: dict[str, TicketRecord] = {}
        self.pending_actions: dict[str, PendingAction] = {}
        self.lock = Lock()


class OrderRepository:
    def __init__(self, database: InMemoryDatabase) -> None:
        self._database = database

    async def find_for_user(self, *, order_id: str, user_id: str) -> OrderResult | None:
        order = self._database.orders.get(order_id)
        return order if order is not None and order.user_id == user_id else None


class TicketRepository:
    def __init__(self, database: InMemoryDatabase) -> None:
        self._database = database

    async def get_or_create(
        self,
        *,
        order_id: str,
        user_id: str,
        operation: str,
        idempotency_key: str,
    ) -> TicketRecord:
        async with self._database.lock:
            existing = self._database.tickets_by_key.get(idempotency_key)
            if existing is not None:
                return existing
            digest = hashlib.sha256(idempotency_key.encode()).hexdigest()[:10].upper()
            ticket = TicketRecord(
                ticket_id=f"T-{digest}",
                order_id=order_id,
                user_id=user_id,
                operation=operation,
                idempotency_key=idempotency_key,
            )
            self._database.tickets_by_key[idempotency_key] = ticket
            return ticket


class PendingActionRepository:
    def __init__(self, database: InMemoryDatabase, *, ttl_seconds: int = 600) -> None:
        self._database = database
        self._ttl = timedelta(seconds=ttl_seconds)

    async def create(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_input: str,
        plan: RoutePlan,
        order_id: str,
    ) -> str:
        token = secrets.token_urlsafe(32)
        token_hash = self._hash(token)
        record = PendingAction(
            token_hash=token_hash,
            user_id=user_id,
            conversation_id=conversation_id,
            user_input=user_input,
            plan=plan,
            order_id=order_id,
            operation="return",
            expires_at=datetime.now(UTC) + self._ttl,
        )
        async with self._database.lock:
            self._database.pending_actions[token_hash] = record
        return token

    async def resolve(
        self, *, token: str, user_id: str, conversation_id: str
    ) -> PendingAction:
        token_hash = self._hash(token)
        async with self._database.lock:
            record = self._database.pending_actions.get(token_hash)
            if record is None:
                raise InvalidConfirmationError("确认令牌无效。")
            if record.user_id != user_id or record.conversation_id != conversation_id:
                raise InvalidConfirmationError("确认令牌与当前用户或会话不匹配。")
            if record.consumed_at is not None:
                raise InvalidConfirmationError("确认令牌已经使用。")
            if record.expires_at <= datetime.now(UTC):
                raise InvalidConfirmationError("确认令牌已经过期。")
            consumed = record.model_copy(
                update={"consumed_at": datetime.now(UTC)}
            )
            self._database.pending_actions[token_hash] = consumed
            return consumed

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()
