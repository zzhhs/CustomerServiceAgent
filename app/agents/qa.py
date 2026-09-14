from app.agents.common import require_finish, require_tool
from app.infrastructure import LanguageModel
from app.models import QAResult, ToolDefinition
from app.services import KnowledgeService


class QAAgent:
    name = "qa_agent"
    goal = "基于企业知识库证据准确回答用户问题"

    def __init__(
        self, *, knowledge_service: KnowledgeService, language_model: LanguageModel | None
    ) -> None:
        self._knowledge = knowledge_service
        self._language_model = language_model

    async def run(self, instruction: str) -> QAResult:
        if self._language_model is None:
            hits = await self._knowledge.search(instruction)
            if not hits:
                return QAResult(answer="当前知识库中没有找到相关信息。")
            return QAResult(
                answer=hits[0].document.content,
                sources=[hit.document.source for hit in hits],
            )

        search_tool = ToolDefinition(
            name="search_knowledge",
            description="对企业知识库执行语义检索",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        observations: list[dict[str, object]] = []
        decision = await self._language_model.decide_agent(
            agent_name=self.name,
            goal=f"{self.goal}。用户问题：{instruction}",
            observations=observations,
            tools=[search_tool],
        )
        arguments = require_tool(decision, "search_knowledge")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("QA Agent 没有生成有效的知识库查询。")
        hits = await self._knowledge.search(query)
        observations.append({
            "tool": "search_knowledge",
            "result": [hit.model_dump(mode="json") for hit in hits],
        })
        if not hits:
            return QAResult(answer="当前知识库中没有找到相关信息。")

        decision = await self._language_model.decide_agent(
            agent_name=self.name,
            goal=(
                f"{self.goal}。用户问题：{instruction}。只能根据检索结果作答，"
                "在 message 中给出最终答案。"
            ),
            observations=observations,
            tools=[],
        )
        answer = require_finish(decision)
        return QAResult(answer=answer, sources=[hit.document.source for hit in hits])

