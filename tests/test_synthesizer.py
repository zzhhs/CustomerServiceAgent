from typing import cast

from app.infrastructure import LanguageModel, ModelGatewayError
from app.models import TaskResult, TaskStatus
from app.orchestration.synthesizer import ResponseSynthesizer


class FailingSynthesisModel:
    async def synthesize(self, user_input: str, results: list[TaskResult]) -> str:
        raise ModelGatewayError("temporary outage")


async def test_completed_write_uses_deterministic_response_when_model_fails() -> None:
    model = cast(LanguageModel, FailingSynthesisModel())
    synthesizer = ResponseSynthesizer(model)
    response = await synthesizer.synthesize(
        user_input="申请退货",
        results=[TaskResult(
            task_id="task_2",
            status=TaskStatus.SUCCEEDED,
            data={"ticket_id": "T-123"},
        )],
    )
    assert response == "售后申请已创建，工单号为 T-123。"
