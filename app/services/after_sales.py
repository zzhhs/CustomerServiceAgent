from app.models import Eligibility, OrderResult, TicketRecord
from app.repositories import TicketRepository


class PolicyService:
    async def check_return_eligibility(self, order: OrderResult) -> Eligibility:
        if order.status != "DELIVERED":
            return Eligibility(allowed=False, reason="订单尚未签收，当前不能申请退货。")
        if not order.refundable:
            return Eligibility(allowed=False, reason="该订单不符合当前退货规则。")
        return Eligibility(allowed=True)


class TicketService:
    def __init__(self, tickets: TicketRepository) -> None:
        self._tickets = tickets

    async def create_return_ticket(
        self, *, order: OrderResult, user_id: str, idempotency_key: str
    ) -> TicketRecord:
        return await self._tickets.get_or_create(
            order_id=order.order_id,
            user_id=user_id,
            operation="return",
            idempotency_key=idempotency_key,
        )
