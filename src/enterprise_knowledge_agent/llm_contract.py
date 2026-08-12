"""Shared prompts and schemas for structured planner and grounded-answer providers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from enterprise_knowledge_agent.grounded_answer import EvidenceSource

GROUNDING_SYSTEM_INSTRUCTION = """
You answer questions using only the enterprise evidence supplied by the application.
Do not use outside knowledge. If the evidence is insufficient or does not support the requested
conclusion, return status 'insufficient_evidence' and a short explanation without citation
markers. When the evidence is sufficient, return status 'answered' and support factual claims
with inline citations such as [S1] or [S2]. Use only source labels that were supplied in the
evidence. The application derives its citation metadata directly from the inline citation markers,
so do not return a separate citation list. Prefer a concise answer that resolves conflicts
explicitly instead of hiding them.
""".strip()

PLANNER_SYSTEM_INSTRUCTION = """
You route enterprise knowledge questions to retrieval tools. Return only the requested JSON.
Choose 'dense_only' for direct factual, lookup, or single-topic questions where semantic retrieval
is sufficient. Choose 'dense_plus_graph' when the question asks about relationships, connected
entities, dependencies, cross-document evidence, project context, or multiple related systems.
Do not answer the question. The reason must be a short routing explanation, not hidden reasoning,
and must contain no more than 20 words.
""".strip()

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["answered", "insufficient_evidence"],
        },
        "answer": {"type": "string"},
    },
    "required": ["status", "answer"],
    "additionalProperties": False,
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "strategy": {
            "type": "string",
            "enum": ["dense_only", "dense_plus_graph"],
        },
        "reason": {"type": "string"},
    },
    "required": ["strategy", "reason"],
    "additionalProperties": False,
}


def build_grounded_prompt(*, question: str, evidence: Sequence[EvidenceSource]) -> str:
    """Build the provider-neutral grounded-answer prompt from selected evidence."""

    source_blocks = []
    for source in evidence:
        source_blocks.append(
            "\n".join(
                [
                    f"[{source.citation_id}]",
                    f"Title: {source.title}",
                    f"Source type: {source.source_type}",
                    f"Document ID: {source.doc_id}",
                    "Evidence:",
                    source.text,
                ]
            )
        )
    joined_sources = "\n\n".join(source_blocks)
    return f"Question:\n{question.strip()}\n\nEnterprise evidence:\n{joined_sources}"
