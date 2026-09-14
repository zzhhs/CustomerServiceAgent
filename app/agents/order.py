from app.agents.common import require_finish, require_tool
from app.infrastructure import LanguageModel
from app.models import OrderResult, ToolDefinition
from app.orchestration.planner import extract_order_id
from app.services import OrderService


class OrderAgent:
    name = "order_agent"
    goal = "通过受限订单工具获得用户请求所需的可信订单事实"

    def __init__(
        self, *, order_service: OrderService, language_model: LanguageModel | None
    ) -> None:
        self._orders = order_service
        self._language_model = language_model

    async def run(self, *, instruction: str, user_input: str, user_id: str) -> OrderResult:
        if self._language_model is None:
            order_id = extract_order_id(user_input)
            if order_id is None:
                raise ValueError("请提供订单号，例如 A123。")
            return await self._orders.get_order(order_id=order_id, user_id=user_id)

        get_order_tool = ToolDefinition(
            name="get_order",
            description="按订单号查询当前用户拥有的订单",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        )
        decision = await self._language_model.decide_agent(
            agent_name=self.name,
            goal=f"{self.goal}。用户请求：{user_input}。子任务：{instruction}",
            observations=[],
            tools=[get_order_tool],
        )
        arguments = require_tool(decision, "get_order")
        order_id = arguments.get("order_id")
        if not isinstance(order_id, str) or not order_id.strip():
            raise ValueError("Order Agent 需要有效订单号。")
        order = await self._orders.get_order(order_id=order_id.upper(), user_id=user_id)
        observation: dict[str, object] = {
            "tool": "get_order",
            "result": order.model_dump(mode="json"),
        }
        decision = await self._language_model.decide_agent(
            agent_name=self.name,
            goal=f"{self.goal}。确认订单事实已经足够完成子任务：{instruction}",
            observations=[observation],
            tools=[],
        )
        require_finish(decision)
        return order
