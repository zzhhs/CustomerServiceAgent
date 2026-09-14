from app.agents import AfterSalesAgent, OrderAgent, QAAgent
from app.config.settings import Settings
from app.infrastructure import DeepSeekGateway, LanguageModel
from app.observability import Observability, create_noop_observability
from app.orchestration.graph import CustomerServiceGraph
from app.orchestration.planner import DeterministicPlanner, ModelPlanner, Planner
from app.orchestration.scheduler import TaskScheduler
from app.orchestration.synthesizer import ResponseSynthesizer
from app.orchestration.validator import PlanValidator
from app.repositories import (
    InMemoryDatabase,
    InMemoryVectorStore,
    OrderRepository,
    PendingActionRepository,
    TicketRepository,
)
from app.services import KnowledgeService, OrderService, PolicyService, TicketService


def build_graph(
    settings: Settings,
    *,
    language_model_override: LanguageModel | None = None,
    observability: Observability | None = None,
) -> CustomerServiceGraph:
    active_observability = observability or create_noop_observability()
    planner: Planner
    language_model: LanguageModel | None
    if language_model_override is not None:
        language_model = language_model_override
        planner = ModelPlanner(language_model)
    elif settings.llm_provider == "deepseek":
        if settings.deepseek_api_key is None:
            raise ValueError("DeepSeek API key is missing")
        language_model = DeepSeekGateway(
            api_key=settings.deepseek_api_key.get_secret_value(),
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            timeout=settings.deepseek_timeout_seconds,
            observability=active_observability,
        )
        planner = ModelPlanner(language_model)
    else:
        language_model = None
        planner = DeterministicPlanner()

    database = InMemoryDatabase()
    order_repository = OrderRepository(database)
    ticket_repository = TicketRepository(database)
    pending_actions = PendingActionRepository(
        database, ttl_seconds=settings.confirmation_ttl_seconds
    )
    vector_store = InMemoryVectorStore()
    qa_agent = QAAgent(
        knowledge_service=KnowledgeService(vector_store),
        language_model=language_model,
    )
    order_agent = OrderAgent(
        order_service=OrderService(order_repository),
        language_model=language_model,
    )
    after_sales_agent = AfterSalesAgent(
        policy_service=PolicyService(),
        ticket_service=TicketService(ticket_repository),
        language_model=language_model,
    )
    scheduler = TaskScheduler(
        qa_agent=qa_agent,
        order_agent=order_agent,
        after_sales_agent=after_sales_agent,
        observability=active_observability,
        max_attempts=settings.task_max_attempts,
        retry_base_delay_seconds=settings.retry_base_delay_seconds,
    )
    return CustomerServiceGraph(
        planner=planner,
        validator=PlanValidator(max_tasks=settings.max_plan_tasks),
        scheduler=scheduler,
        synthesizer=ResponseSynthesizer(language_model),
        pending_actions=pending_actions,
        observability=active_observability,
        max_replans=settings.max_replans,
        max_attempts=settings.task_max_attempts,
        retry_base_delay_seconds=settings.retry_base_delay_seconds,
    )
