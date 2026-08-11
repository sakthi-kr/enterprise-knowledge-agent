"""Tests for optional runtime observability."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from enterprise_knowledge_agent.observability import NullTracer, build_tracer


def test_null_tracer_accepts_span_operations() -> None:
    tracer = NullTracer()

    with tracer.span(
        "test",
        span_type="TOOL",
        inputs={"question": "hello"},
        attributes={"kind": "test"},
    ) as span:
        span.set_inputs({"a": 1})
        span.set_outputs({"b": 2})
        span.set_attribute("key", "value")


def test_build_tracer_returns_null_when_disabled() -> None:
    tracer = build_tracer(
        enabled=False,
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="enterprise-knowledge-agent",
    )

    assert isinstance(tracer, NullTracer)


class _FakeSpan:
    def __init__(self) -> None:
        self.inputs: Any = None
        self.outputs: Any = None
        self.attributes: dict[str, Any] = {}

    def set_inputs(self, value: Any) -> None:
        self.inputs = value

    def set_outputs(self, value: Any) -> None:
        self.outputs = value

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class RecordingTracer:
    """Small tracer used by agent tests to verify instrumentation."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str,
        inputs: Any | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[_FakeSpan]:
        span = _FakeSpan()
        if inputs is not None:
            span.set_inputs(inputs)
        if attributes:
            span.attributes.update(attributes)
        record = {
            "name": name,
            "span_type": span_type,
            "span": span,
        }
        self.spans.append(record)
        yield span


def test_mlflow_tracer_validates_configuration_before_optional_import() -> None:
    with pytest.raises(ValueError, match="tracking_uri"):
        build_tracer(enabled=True, tracking_uri="", experiment_name="experiment")

    with pytest.raises(ValueError, match="experiment_name"):
        build_tracer(enabled=True, tracking_uri="http://localhost:5000", experiment_name="")


def test_mlflow_tracer_uses_manual_span_api(monkeypatch) -> None:
    class FakeContext:
        def __init__(self, span: _FakeSpan) -> None:
            self.span = span

        def __enter__(self) -> _FakeSpan:
            return self.span

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

    class FakeMLflow:
        def __init__(self) -> None:
            self.tracking_uri = ""
            self.experiment_name = ""
            self.started: list[dict[str, Any]] = []
            self.span = _FakeSpan()

        def set_tracking_uri(self, value: str) -> None:
            self.tracking_uri = value

        def set_experiment(self, value: str) -> None:
            self.experiment_name = value

        def start_span(self, **kwargs: Any) -> FakeContext:
            self.started.append(kwargs)
            return FakeContext(self.span)

    fake_mlflow = FakeMLflow()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    from enterprise_knowledge_agent.observability import MLflowTracer

    tracer = MLflowTracer(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="enterprise-knowledge-agent",
    )
    with tracer.span(
        "dense_search",
        span_type="RETRIEVER",
        inputs={"query": "incident"},
        attributes={"store": "qdrant"},
    ) as span:
        span.set_outputs([{"id": "chunk-1"}])

    assert fake_mlflow.tracking_uri == "http://127.0.0.1:5000"
    assert fake_mlflow.experiment_name == "enterprise-knowledge-agent"
    assert fake_mlflow.started == [
        {
            "name": "dense_search",
            "span_type": "RETRIEVER",
            "attributes": {"store": "qdrant"},
        }
    ]
    assert fake_mlflow.span.inputs == {"query": "incident"}
    assert fake_mlflow.span.outputs == [{"id": "chunk-1"}]
