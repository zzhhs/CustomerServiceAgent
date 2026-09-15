from app.infrastructure import LanguageModel, ModelGatewayError
from app.models import ConversationMessage, TaskResult, TaskStatus


class ResponseSynthesizer:
    def __init__(self, language_model: LanguageModel | None) -> None:
        self._language_model = language_model

    async def synthesize(
        self,
        *,
        user_input: str,
        results: list[TaskResult],
        conversation_messages: list[ConversationMessage] | None = None,
    ) -> str:
        needs_input = [r for r in results if r.status is TaskStatus.NEEDS_INPUT]
        if needs_input:
            return needs_input[0].error or "请补充继续处理所需的信息。"
        if self._language_model is not None:
            try:
                return await self._language_model.synthesize(
                    user_input,
                    results,
                    conversation_messages=conversation_messages or [],
                )
            except ModelGatewayError:
                # Never hide a completed business operation behind a wording-model outage.
                return self._deterministic_response(results)
        return self._deterministic_response(results)

    @staticmethod
    def _deterministic_response(results: list[TaskResult]) -> str:
        pending = [r for r in results if r.status is TaskStatus.PENDING_CONFIRMATION]
        failures = [r for r in results if r.status is TaskStatus.FAILED]
        if pending:
            return (
                "订单信息已核验。创建售后申请会产生实际业务变更，请确认后继续。"
            )
        if failures:
            details = "；".join(r.error or "未知错误" for r in failures)
            return "处理未完成：" + details
        final = results[-1].data
        if final.get("eligible") is False:
            return str(final.get("reason", "该订单不符合售后规则。"))
        if final.get("ticket_id"):
            return f"售后申请已创建，工单号为 {final['ticket_id']}。"
        if final.get("order_id"):
            return f"订单 {final['order_id']} 当前状态为 {final['status']}。"
        return str(final.get("answer", "处理完成。"))
