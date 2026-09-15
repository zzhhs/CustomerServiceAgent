import asyncio

from app.config.settings import Settings
from app.container import build_graph
from app.infrastructure import ModelGatewayError
from app.models import (
    AgentDecision,
    AgentName,
    ConversationMessage,
    RoutePlan,
    SubTask,
    TaskAction,
    TaskResult,
    ToolDefinition,
)
from app.orchestration.state import CustomerServiceState


class FakeLanguageModel:
    def __init__(self) -> None:
        self.plan_calls = 0
        self.synthesis_calls = 0
        self.synthesis_histories: list[list[ConversationMessage]] = []
        self.decision_calls = 0
        self.replan_calls = 0

    async def create_plan(self, user_input: str) -> RoutePlan:
        self.plan_calls += 1
        return RoutePlan(tasks=[SubTask(
            id="task_1",
            agent=AgentName.QA,
            action=TaskAction.ANSWER_QUESTION,
            instruction=user_input,
        )])

    async def replan(
        self,
        user_input: str,
        *,
        previous_plan: RoutePlan,
        results: list[TaskResult],
    ) -> RoutePlan:
        self.replan_calls += 1
        return await self.create_plan(user_input)

    async def synthesize(
        self,
        user_input: str,
        results: list[TaskResult],
        *,
        conversation_messages: list[ConversationMessage],
    ) -> str:
        self.synthesis_calls += 1
        self.synthesis_histories.append(list(conversation_messages))
        return str(results[-1].data["answer"])

    async def decide_agent(
        self,
        *,
        agent_name: str,
        goal: str,
        observations: list[dict[str, object]],
        tools: list[ToolDefinition],
    ) -> AgentDecision:
        self.decision_calls += 1
        if tools:
            return AgentDecision(
                kind="tool",
                tool_name=tools[0].name,
                arguments={"query": "退款到账时间"},
            )
        return AgentDecision(
            kind="finish",
            message="退款通常在一至三个工作日原路退回。",
        )


class CompositeFakeLanguageModel(FakeLanguageModel):
    def __init__(self) -> None:
        super().__init__()
        self.agent_names: list[str] = []

    async def create_plan(self, user_input: str) -> RoutePlan:
        self.plan_calls += 1
        return RoutePlan(
            tasks=[
                SubTask(
                    id="task_1",
                    agent=AgentName.ORDER,
                    action=TaskAction.QUERY_ORDER,
                    instruction="查询订单 A123",
                ),
                SubTask(
                    id="task_2",
                    agent=AgentName.AFTER_SALES,
                    action=TaskAction.APPLY_AFTER_SALES,
                    instruction="申请退货",
                    depends_on=["task_1"],
                ),
            ],
            requires_confirmation=True,
        )

    async def decide_agent(
        self,
        *,
        agent_name: str,
        goal: str,
        observations: list[dict[str, object]],
        tools: list[ToolDefinition],
    ) -> AgentDecision:
        self.decision_calls += 1
        self.agent_names.append(agent_name)
        if not tools:
            return AgentDecision(kind="finish", message="领域目标已完成")
        arguments = {"order_id": "A123"} if tools[0].name == "get_order" else {}
        return AgentDecision(kind="tool", tool_name=tools[0].name, arguments=arguments)

    async def synthesize(
        self,
        user_input: str,
        results: list[TaskResult],
        *,
        conversation_messages: list[ConversationMessage],
    ) -> str:
        self.synthesis_calls += 1
        self.synthesis_histories.append(list(conversation_messages))
        ticket_id = results[-1].data.get("ticket_id")
        return f"工单 {ticket_id} 已创建" if ticket_id else "请确认创建售后工单"


class RetryingFakeLanguageModel(FakeLanguageModel):
    async def decide_agent(
        self,
        *,
        agent_name: str,
        goal: str,
        observations: list[dict[str, object]],
        tools: list[ToolDefinition],
    ) -> AgentDecision:
        if self.decision_calls == 0:
            self.decision_calls += 1
            raise ModelGatewayError("temporary model outage")
        return await super().decide_agent(
            agent_name=agent_name,
            goal=goal,
            observations=observations,
            tools=tools,
        )


class RetryingPlannerFakeLanguageModel(FakeLanguageModel):
    async def create_plan(self, user_input: str) -> RoutePlan:
        self.plan_calls += 1
        if self.plan_calls == 1:
            raise ModelGatewayError("temporary planner outage")
        return RoutePlan(tasks=[SubTask(
            id="task_1",
            agent=AgentName.QA,
            action=TaskAction.ANSWER_QUESTION,
            instruction=user_input,
        )])


class ReplanningFakeLanguageModel(FakeLanguageModel):
    def __init__(self) -> None:
        super().__init__()
        self.use_bad_order = True

    async def create_plan(self, user_input: str) -> RoutePlan:
        self.plan_calls += 1
        return RoutePlan(tasks=[SubTask(
            id="task_1",
            agent=AgentName.ORDER,
            action=TaskAction.QUERY_ORDER,
            instruction="查询订单",
        )])

    async def replan(
        self,
        user_input: str,
        *,
        previous_plan: RoutePlan,
        results: list[TaskResult],
    ) -> RoutePlan:
        self.replan_calls += 1
        self.use_bad_order = False
        return previous_plan

    async def decide_agent(
        self,
        *,
        agent_name: str,
        goal: str,
        observations: list[dict[str, object]],
        tools: list[ToolDefinition],
    ) -> AgentDecision:
        self.decision_calls += 1
        if tools:
            return AgentDecision(
                kind="tool",
                tool_name="get_order",
                arguments={"order_id": "Z999" if self.use_bad_order else "A123"},
            )
        return AgentDecision(kind="finish", message="订单查询完成")

    async def synthesize(
        self,
        user_input: str,
        results: list[TaskResult],
        *,
        conversation_messages: list[ConversationMessage],
    ) -> str:
        return f"订单 {results[-1].data['order_id']} 查询完成"


class DagFakeLanguageModel(FakeLanguageModel):
    def __init__(self) -> None:
        super().__init__()
        self.root_starts = 0
        self.root_finishes: set[str] = set()
        self.both_roots_started = asyncio.Event()
        self.downstream_started = asyncio.Event()
        self.downstream_started_before_slow_root_finished = False

    async def create_plan(self, user_input: str) -> RoutePlan:
        self.plan_calls += 1
        return RoutePlan(
            tasks=[
                SubTask(
                    id="task_1",
                    agent=AgentName.QA,
                    action=TaskAction.ANSWER_QUESTION,
                    instruction="并发问题一",
                ),
                SubTask(
                    id="task_2",
                    agent=AgentName.QA,
                    action=TaskAction.ANSWER_QUESTION,
                    instruction="并发问题二",
                ),
                SubTask(
                    id="task_3",
                    agent=AgentName.QA,
                    action=TaskAction.ANSWER_QUESTION,
                    instruction="问题一的下游任务",
                    depends_on=["task_1"],
                ),
            ]
        )

    async def decide_agent(
        self,
        *,
        agent_name: str,
        goal: str,
        observations: list[dict[str, object]],
        tools: list[ToolDefinition],
    ) -> AgentDecision:
        self.decision_calls += 1
        if tools:
            if "问题一的下游任务" in goal:
                self.downstream_started_before_slow_root_finished = (
                    "并发问题一" in self.root_finishes
                    and "并发问题二" not in self.root_finishes
                )
                self.downstream_started.set()
            else:
                self.root_starts += 1
                if self.root_starts == 2:
                    self.both_roots_started.set()
                await asyncio.wait_for(self.both_roots_started.wait(), timeout=0.5)
            return AgentDecision(
                kind="tool",
                tool_name="search_knowledge",
                arguments={"query": "退款到账时间"},
            )
        if "并发问题二" in goal:
            await asyncio.wait_for(self.downstream_started.wait(), timeout=0.5)
        for instruction in ("并发问题一", "并发问题二"):
            if instruction in goal:
                self.root_finishes.add(instruction)
        return AgentDecision(kind="finish", message="知识库回答完成")


async def test_model_drives_plan_rag_answer_and_synthesis() -> None:
    model = FakeLanguageModel()
    graph = build_graph(
        Settings(llm_provider="deterministic"),
        language_model_override=model,
    )

    result = await graph.invoke(CustomerServiceState(
        user_id="user-1",
        conversation_id="conversation-rag",
        user_input="退款多久到账？",
        confirmation_authorized=False,
    ))

    assert model.plan_calls == 1
    assert model.decision_calls == 2
    assert model.synthesis_calls == 1
    assert result["task_results"][0].data["sources"][0] == "mock://knowledge/refund-time"
    assert "一至三个工作日" in result["final_response"]


async def test_response_synthesis_receives_prior_conversation_history() -> None:
    model = FakeLanguageModel()
    graph = build_graph(
        Settings(llm_provider="deterministic"),
        language_model_override=model,
    )
    await graph.invoke(CustomerServiceState(
        user_id="user-1",
        conversation_id="personalization-test",
        user_input="以后请用简短方式回答。",
        confirmation_authorized=False,
    ))
    await graph.invoke(CustomerServiceState(
        user_id="user-1",
        conversation_id="personalization-test",
        user_input="退款多久到账？",
        confirmation_authorized=False,
    ))

    history = model.synthesis_histories[-1]
    assert [message.role for message in history] == ["user", "assistant"]
    assert history[0].content == "以后请用简短方式回答。"
    assert "一至三个工作日" in history[1].content


async def test_scheduler_dispatches_real_specialist_agents() -> None:
    model = CompositeFakeLanguageModel()
    graph = build_graph(
        Settings(llm_provider="deterministic"),
        language_model_override=model,
    )

    result = await graph.invoke(CustomerServiceState(
        user_id="user-1",
        conversation_id="conversation-specialists",
        user_input="查询订单 A123 并申请退货",
        confirmation_authorized=False,
    ))

    assert model.agent_names == [
        "order_agent",
        "order_agent",
        "after_sales_agent",
    ]
    assert result["task_results"][1].status.value == "pending_confirmation"
    token = result["issued_confirmation_token"]

    confirmed_result = await graph.invoke(CustomerServiceState(
        user_id="user-1",
        conversation_id="conversation-specialists",
        user_input="this input is replaced by the stored pending action",
        confirmation_authorized=False,
        confirmation_token=token,
    ))
    assert model.plan_calls == 1
    assert confirmed_result["task_results"][1].data["ticket_id"].startswith("T-")


async def test_retryable_agent_failure_uses_bounded_retry() -> None:
    model = RetryingFakeLanguageModel()
    graph = build_graph(
        Settings(
            llm_provider="deterministic",
            task_max_attempts=2,
            retry_base_delay_seconds=0,
            max_replans=0,
        ),
        language_model_override=model,
    )
    result = await graph.invoke(CustomerServiceState(
        user_id="user-1",
        conversation_id="retry-test",
        user_input="退款多久到账？",
        confirmation_authorized=False,
    ))
    assert result["task_results"][0].status.value == "succeeded"
    assert result["task_results"][0].attempts == 2


async def test_retryable_planner_failure_uses_bounded_retry() -> None:
    model = RetryingPlannerFakeLanguageModel()
    graph = build_graph(
        Settings(
            llm_provider="deterministic",
            task_max_attempts=2,
            retry_base_delay_seconds=0,
            max_replans=0,
        ),
        language_model_override=model,
    )
    result = await graph.invoke(CustomerServiceState(
        user_id="user-1",
        conversation_id="planner-retry-test",
        user_input="退款多久到账？",
        confirmation_authorized=False,
    ))
    assert model.plan_calls == 2
    assert result["task_results"][0].status.value == "succeeded"


async def test_execution_failure_replans_and_runs_revised_plan() -> None:
    model = ReplanningFakeLanguageModel()
    graph = build_graph(
        Settings(
            llm_provider="deterministic",
            task_max_attempts=1,
            retry_base_delay_seconds=0,
            max_replans=1,
        ),
        language_model_override=model,
    )
    result = await graph.invoke(CustomerServiceState(
        user_id="user-1",
        conversation_id="replan-test",
        user_input="查询订单 A123",
        confirmation_authorized=False,
    ))
    assert model.replan_calls == 1
    assert result["replan_count"] == 1
    assert [item.status.value for item in result["execution_history"]] == [
        "failed",
        "succeeded",
    ]
    assert result["task_results"][0].data["order_id"] == "A123"


async def test_dag_runs_ready_tasks_concurrently_then_releases_dependents() -> None:
    model = DagFakeLanguageModel()
    graph = build_graph(
        Settings(llm_provider="deterministic"),
        language_model_override=model,
    )

    result = await asyncio.wait_for(
        graph.invoke(
            CustomerServiceState(
                user_id="user-1",
                conversation_id="dag-concurrency-test",
                user_input="同时回答三个客服问题",
                confirmation_authorized=False,
            )
        ),
        timeout=1,
    )

    assert model.root_starts == 2
    assert model.downstream_started_before_slow_root_finished is True
    assert [item.task_id for item in result["task_results"]] == [
        "task_1",
        "task_2",
        "task_3",
    ]
    assert all(item.status.value == "succeeded" for item in result["task_results"])
