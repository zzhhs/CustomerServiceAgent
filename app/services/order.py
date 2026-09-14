from app.models import OrderResult
from app.repositories import OrderRepository


class OrderNotFoundError(LookupError):
    pass


class OrderService:
    def __init__(self, repository: OrderRepository) -> None:
        self._repository = repository

    async def get_order(self, *, order_id: str, user_id: str) -> OrderResult:
        order = await self._repository.find_for_user(order_id=order_id, user_id=user_id)
        if order is None:
            raise OrderNotFoundError("Order was not found for this user")
        return order
