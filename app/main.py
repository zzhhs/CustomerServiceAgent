from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config.settings import Settings, get_settings
from app.container import build_graph
from app.observability import configure_observability, instrument_app
from app.security import JwtBearerAuthenticator


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    observability = configure_observability(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.customer_service_graph = build_graph(
            resolved_settings, observability=observability
        )
        app.state.llm_provider = resolved_settings.llm_provider
        app.state.auth_mode = resolved_settings.auth_mode
        app.state.authenticator = (
            JwtBearerAuthenticator(
                secret=resolved_settings.auth_secret.get_secret_value(),
                issuer=resolved_settings.auth_issuer,
                audience=resolved_settings.auth_audience,
            )
            if resolved_settings.auth_secret is not None
            else None
        )
        yield
        observability.shutdown()

    application = FastAPI(
        title="Customer Service Agent", version="0.3.0", lifespan=lifespan
    )
    application.include_router(router)
    instrument_app(
        application, observability, enabled=resolved_settings.telemetry_enabled
    )
    return application


app = create_app()
