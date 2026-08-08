"""Tests for dense embedding integration."""

import sys
from types import ModuleType

from enterprise_knowledge_agent.embeddings import FastEmbedTextEncoder


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class _FakeTextEmbedding:
    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name
        self.passage_calls: list[tuple[list[str], int]] = []
        self.query_calls: list[tuple[str, int]] = []

    def passage_embed(self, texts: list[str], *, batch_size: int):
        self.passage_calls.append((list(texts), batch_size))
        for index, _ in enumerate(texts, start=1):
            yield _FakeVector([float(index), 0.0, 1.0])

    def query_embed(self, text: str, *, batch_size: int):
        self.query_calls.append((text, batch_size))
        yield _FakeVector([0.0, 1.0, 0.0])


def test_fastembed_encoder_uses_retrieval_specific_methods(monkeypatch) -> None:
    module = ModuleType("fastembed")
    module.TextEmbedding = _FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)

    encoder = FastEmbedTextEncoder(model_name="test-model", dimension=3, batch_size=8)

    passages = encoder.embed_passages(["alpha", "beta"])
    query = encoder.embed_query("alpha query")

    assert passages == [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]
    assert query == [0.0, 1.0, 0.0]
    assert encoder._model.passage_calls == [(["alpha", "beta"], 8)]
    assert encoder._model.query_calls == [("alpha query", 1)]


def test_fastembed_encoder_rejects_empty_query(monkeypatch) -> None:
    module = ModuleType("fastembed")
    module.TextEmbedding = _FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    encoder = FastEmbedTextEncoder(model_name="test-model", dimension=3)

    try:
        encoder.embed_query("   ")
    except ValueError as exc:
        assert str(exc) == "query text must not be empty"
    else:
        raise AssertionError("Expected an empty-query ValueError")
