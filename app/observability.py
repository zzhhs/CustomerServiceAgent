import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.metrics import Counter, Histogram, Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from app.config.settings import Settings


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        span_context = trace.get_current_span().get_span_context()
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if span_context.is_valid:
            payload["trace_id"] = format(span_context.trace_id, "032x")
            payload["span_id"] = format(span_context.span_id, "016x")
        for key in (
            "conversation_id",
            "task_id",
            "agent",
            "event",
            "http_method",
            "http_path",
            "http_status",
            "duration_ms",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class Observability:
    def __init__(
        self,
        *,
        tracer: Tracer,
        meter: Meter,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
    ) -> None:
        self.tracer = tracer
        self.tracer_provider = tracer_provider
        self._meter_provider = meter_provider
        self.requests: Counter = meter.create_counter("customer_service.requests")
        self.tasks: Counter = meter.create_counter("customer_service.tasks")
        self.retries: Counter = meter.create_counter("customer_service.task_retries")
        self.replans: Counter = meter.create_counter("customer_service.replans")
        self.task_duration: Histogram = meter.create_histogram(
            "customer_service.task.duration", unit="s"
        )

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[Span]:
        with self.tracer.start_as_current_span(name, attributes=attributes) as span:
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def shutdown(self) -> None:
        if self.tracer_provider is not None:
            self.tracer_provider.shutdown()
        if self._meter_provider is not None:
            self._meter_provider.shutdown()


def configure_observability(settings: Settings) -> Observability:
    _configure_logging(settings.log_level, settings.log_format)
    if not settings.telemetry_enabled:
        return create_noop_observability()

    resource = Resource.create({
        "service.name": settings.otel_service_name,
        "service.version": "0.3.0",
        "deployment.environment.name": settings.app_env,
    })
    tracer_provider = TracerProvider(resource=resource)
    meter_readers: list[PeriodicExportingMetricReader] = []

    if settings.otel_exporter == "console":
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        meter_readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
    elif settings.otel_exporter == "otlp":
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint=f"{settings.otel_endpoint.rstrip('/')}/v1/traces"
        )))
        meter_readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(
            endpoint=f"{settings.otel_endpoint.rstrip('/')}/v1/metrics"
        )))

    meter_provider = MeterProvider(resource=resource, metric_readers=meter_readers)
    return Observability(
        tracer=tracer_provider.get_tracer("customer-service-agent"),
        meter=meter_provider.get_meter("customer-service-agent"),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )


def create_noop_observability() -> Observability:
    return Observability(
        tracer=trace.get_tracer("customer-service-agent"),
        meter=metrics.get_meter("customer-service-agent"),
    )


def instrument_app(app: FastAPI, observability: Observability, *, enabled: bool) -> None:
    if enabled:
        FastAPIInstrumentor.instrument_app(
            app, tracer_provider=observability.tracer_provider
        )

    logger = logging.getLogger("customer_service.http")

    @app.middleware("http")
    async def log_request(request: Request, call_next: Any) -> Any:
        started = perf_counter()
        response = await call_next(request)
        duration_ms = (perf_counter() - started) * 1000
        logger.info(
            "request completed",
            extra={
                "event": "http.request.completed",
                "http_method": request.method,
                "http_path": request.url.path,
                "http_status": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        response.headers["Server-Timing"] = f"app;dur={duration_ms:.2f}"
        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            response.headers["X-Trace-ID"] = format(span_context.trace_id, "032x")
        return response


def _configure_logging(level: str, log_format: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonLogFormatter()
        if log_format == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    app_logger = logging.getLogger("customer_service")
    app_logger.handlers.clear()
    app_logger.addHandler(handler)
    app_logger.setLevel(level.upper())
    app_logger.propagate = False
