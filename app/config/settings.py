from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Customer Service Agent"
    app_env: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    auth_mode: Literal["disabled", "jwt"] = "disabled"
    auth_secret: SecretStr | None = None
    auth_issuer: str = "customer-service-auth"
    auth_audience: str = "customer-service-api"
    llm_provider: Literal["deepseek", "deterministic"] = "deepseek"
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = Field(default="deepseek-v4-pro", min_length=1)
    deepseek_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    max_plan_tasks: int = Field(default=6, ge=1, le=20)
    confirmation_ttl_seconds: int = Field(default=600, ge=60, le=3600)
    task_max_attempts: int = Field(default=3, ge=1, le=5)
    retry_base_delay_seconds: float = Field(default=0.25, ge=0, le=10)
    max_replans: int = Field(default=1, ge=0, le=3)
    telemetry_enabled: bool = False
    otel_service_name: str = "customer-service-agent"
    otel_exporter: Literal["none", "console", "otlp"] = "none"
    otel_endpoint: str = "http://localhost:4318"

    @model_validator(mode="after")
    def validate_deepseek_configuration(self) -> "Settings":
        if self.llm_provider == "deepseek" and self.deepseek_api_key is None:
            raise ValueError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")
        if self.auth_mode == "jwt" and self.auth_secret is None:
            raise ValueError("AUTH_SECRET is required when AUTH_MODE=jwt")
        if self.app_env == "production" and self.auth_mode == "disabled":
            raise ValueError("AUTH_MODE=disabled is not allowed in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
