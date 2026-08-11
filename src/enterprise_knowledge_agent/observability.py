"""Optional MLflow tracing for the enterprise agent workflow."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol


class TraceSpan(Protocol):
    """Span operations used by the agent without coupling it to MLflow."""

    def set_inputs(self, value: Any) -> None:
        """Attach structured inputs to the span."""

    def set_outputs(self, value: Any) -> None:
        """Attach structured outputs to the span."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach one span attribute."""


class Tracer(Protocol):
    """Tracing interface required by the agent."""

    def span(
        self,
        name: str,
        *,
        span_type: str,
        inputs: Any | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> AbstractContextManager[TraceSpan]:
        """Create one tracing span."""


class _NullSpan:
    def set_inputs(self, value: Any) -> None:
        return None

    def set_outputs(self, value: Any) -> None:
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        return None


class NullTracer:
    """No-op tracer used when observability is disabled."""

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str,
        inputs: Any | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan]:
        del name, span_type, inputs, attributes
        yield _NullSpan()


class _MLflowSpan:
    def __init__(self, span: Any) -> None:
        self._span = span

    def set_inputs(self, value: Any) -> None:
        self._span.set_inputs(value)

    def set_outputs(self, value: Any) -> None:
        self._span.set_outputs(value)

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)


class MLflowTracer:
    """Manual MLflow tracer with a lazy optional dependency."""

    def __init__(self, *, tracking_uri: str, experiment_name: str) -> None:
        if not tracking_uri.strip():
            raise ValueError("tracking_uri must not be empty")
        if not experiment_name.strip():
            raise ValueError("experiment_name must not be empty")

        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow tracing requires the optional ops dependencies. "
                'Install the project with: python -m pip install -e ".[dev,ops]"'
            ) from exc

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._mlflow = mlflow

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str,
        inputs: Any | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan]:
        with self._mlflow.start_span(
            name=name,
            span_type=span_type,
            attributes=attributes,
        ) as span:
            wrapped = _MLflowSpan(span)
            if inputs is not None:
                wrapped.set_inputs(inputs)
            yield wrapped


def build_tracer(
    *,
    enabled: bool,
    tracking_uri: str,
    experiment_name: str,
) -> Tracer:
    """Build the configured tracer without importing MLflow when disabled."""

    if not enabled:
        return NullTracer()
    return MLflowTracer(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
