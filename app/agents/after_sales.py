from app.agents.common import require_tool
from app.infrastructure import LanguageModel
from app.models import AfterSalesResult, OrderResult, ToolDefinition
from app.services import PolicyService, TicketService


class ConfirmationRequired(RuntimeError):
    pass


class AfterSalesAgent:
    name = "after_sales_agent"
    goal = "在符合政策、获得用户确认且保持幂等的前提下完成售后申请"

    def __init__(
        self,
        *,
        policy_service: PolicyService,
        ticket_service: TicketService,
        language_model: LanguageModel | None,
    ) -> None:
        self._policy = policy_service
        self._tickets = ticket_service
        self._language_model = language_model

    async def run(
        self,
        *,
        instruction: str,
        order: OrderResult,
        user_id: str,
        confirmation_authorized: bool,
    ) -> AfterSalesResult:
        observations: list[dict[str, object]] = [
            {"input_order": order.model_dump(mode="json")}
        ]
        await self._choose_tool(
            goal=f"{self.goal}。子任务：{instruction}",
            observations=observations,
            tool=ToolDefinition(
                name="check_return_eligibility",
                description="使用确定性业务规则检查订单退货资格",
                input_schema={"type": "object", "properties": {}},
            ),
        )
        eligibility = await self._policy.check_return_eligibility(order)
        observations.append({
            "tool": "check_return_eligibility",
            "result": eligibility.model_dump(mode="json"),
        })
        if not eligibility.allowed:
            return AfterSalesResult(
                order_id=order.order_id,
                eligible=False,
                reason=eligibility.reason,
            )
        if not confirmation_authorized:
            raise ConfirmationRequired("创建售后工单需要用户确认")

        await self._choose_tool(
            goal=f"{self.goal}。资格已通过且用户已经确认。子任务：{instruction}",
            observations=observations,
            tool=ToolDefinition(
                name="create_return_ticket",
                description="使用幂等键创建退货工单",
                input_schema={"type": "object", "properties": {}},
            ),
        )
        ticket = await self._tickets.create_return_ticket(
            order=order,
            user_id=user_id,
            idempotency_key=f"{user_id}:{order.order_id}:return",
        )
        # The durable tool result is authoritative. Do not make success depend on a
        # second model call after the irreversible business operation has completed.
        return AfterSalesResult(
            order_id=order.order_id,
            eligible=True,
            ticket_id=ticket.ticket_id,
        )

    async def _choose_tool(
        self,
        *,
        goal: str,
        observations: list[dict[str, object]],
        tool: ToolDefinition,
    ) -> None:
        if self._language_model is None:
            return
        decision = await self._language_model.decide_agent(
            agent_name=self.name,
            goal=goal,
            observations=observations,
            tools=[tool],
        )
        require_tool(decision, tool.name)
