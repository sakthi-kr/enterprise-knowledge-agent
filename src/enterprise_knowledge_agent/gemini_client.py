"""Minimal Gemini REST client for planning and structured grounded answers."""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from enterprise_knowledge_agent.agent_types import AgentPlan, AgentStrategy
from enterprise_knowledge_agent.grounded_answer import (
    AnswerStatus,
    EvidenceSource,
    LanguageModelOutput,
    TokenUsage,
)

_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_RETRYABLE_STATUS_CODES = {408, 429}

_SYSTEM_INSTRUCTION = """
You answer questions using only the enterprise evidence supplied by the application.
Do not use outside knowledge. If the evidence is insufficient or does not support the requested
conclusion, return status 'insufficient_evidence' and a short explanation without citation
markers. When the evidence is sufficient, return status 'answered' and support factual claims
with inline citations such as [S1] or [S2]. Use only source labels that were supplied in the
evidence. The application derives its citation metadata directly from the inline citation markers,
so do not return a separate citation list. Prefer a concise answer that resolves conflicts
explicitly instead of hiding them.
""".strip()

_PLANNER_SYSTEM_INSTRUCTION = """
You route enterprise knowledge questions to retrieval tools. Return only the requested JSON.
Choose 'dense_only' for direct factual, lookup, or single-topic questions where semantic retrieval
is sufficient. Choose 'dense_plus_graph' when the question asks about relationships, connected
entities, dependencies, cross-document evidence, project context, or multiple related systems.
Do not answer the question. The reason must be a short routing explanation, not hidden reasoning,
and must contain no more than 20 words.
""".strip()

_RESPONSE_SCHEMA: dict[str, Any] = {
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

_PLAN_SCHEMA: dict[str, Any] = {
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


class GeminiAPIError(RuntimeError):
    """Raised when the Gemini API cannot produce a valid structured response."""


class GeminiRestClient:
    """Call Gemini GenerateContent through a small provider-specific adapter."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = "gemini-3.6-flash",
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Gemini API key must not be empty")
        if not _MODEL_NAME_PATTERN.fullmatch(model_name):
            raise ValueError("Gemini model name contains unsupported characters")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if retry_base_delay_seconds <= 0:
            raise ValueError("retry_base_delay_seconds must be greater than zero")

        self.model_name = model_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def __enter__(self) -> GeminiRestClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client when this adapter created it."""

        if self._owns_client:
            self._client.close()

    def plan(self, *, question: str) -> AgentPlan:
        """Select the bounded retrieval strategy for an agent question."""

        if not question.strip():
            raise ValueError("question must not be empty")

        body = {
            "systemInstruction": {"parts": [{"text": _PLANNER_SYSTEM_INSTRUCTION}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"Enterprise question:\n{question.strip()}"}],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": _PLAN_SCHEMA,
                "maxOutputTokens": 200,
                "thinkingConfig": {"thinkingLevel": "minimal"},
            },
        }
        payload = self._generate_payload(body)
        text = self._extract_candidate_text(payload)
        strategy, reason = self._parse_plan_output(text)
        return AgentPlan(
            strategy=strategy,
            reason=reason,
            model_name=self.model_name,
            usage=self._parse_usage(payload),
        )

    def generate(
        self,
        *,
        question: str,
        evidence: Sequence[EvidenceSource],
    ) -> LanguageModelOutput:
        """Generate a schema-constrained grounded answer from supplied evidence."""

        if not question.strip():
            raise ValueError("question must not be empty")
        if not evidence:
            raise ValueError("evidence must not be empty")

        prompt = self._build_prompt(question=question, evidence=evidence)
        body = {
            "systemInstruction": {"parts": [{"text": _SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": _RESPONSE_SCHEMA,
                "maxOutputTokens": 1000,
                "thinkingConfig": {"thinkingLevel": "minimal"},
            },
        }
        payload = self._generate_payload(body)
        text = self._extract_candidate_text(payload)
        status, answer = self._parse_structured_output(text)
        return LanguageModelOutput(
            status=status,
            answer=answer,
            model_name=self.model_name,
            usage=self._parse_usage(payload),
        )

    def _generate_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"{self._base_url}/v1beta/models/{self.model_name}:generateContent"
        response = self._post_with_retries(endpoint=endpoint, body=body)
        try:
            payload = response.json()
        except ValueError as exc:
            raise GeminiAPIError("Gemini API returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise GeminiAPIError("Gemini API returned a non-object JSON payload")
        return payload

    def _post_with_retries(self, *, endpoint: str, body: dict[str, Any]) -> httpx.Response:
        last_transport_error: httpx.TransportError | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    endpoint,
                    headers={
                        "x-goog-api-key": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
            except httpx.TransportError as exc:
                last_transport_error = exc
                if attempt >= self._max_retries:
                    break
                self._sleep(self._retry_delay(attempt))
                continue

            if response.is_success:
                return response

            if self._is_retryable_status(response.status_code) and attempt < self._max_retries:
                self._sleep(self._retry_delay(attempt))
                continue

            message = self._extract_error_message(response)
            raise GeminiAPIError(f"Gemini API returned HTTP {response.status_code}: {message}")

        raise GeminiAPIError(
            "Could not reach the Gemini API after retries"
        ) from last_transport_error

    def _retry_delay(self, attempt: int) -> float:
        base_delay = self._retry_base_delay_seconds * (2**attempt)
        jitter = random.uniform(0.0, base_delay * 0.25)
        return base_delay + jitter

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in _RETRYABLE_STATUS_CODES or 500 <= status_code <= 599

    @staticmethod
    def _build_prompt(*, question: str, evidence: Sequence[EvidenceSource]) -> str:
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

    @staticmethod
    def _extract_candidate_text(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            feedback = payload.get("promptFeedback")
            raise GeminiAPIError(f"Gemini API returned no candidates: {feedback or 'no feedback'}")

        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise GeminiAPIError("Gemini API candidate had an invalid shape")
        finish_reason = candidate.get("finishReason")
        if finish_reason not in (None, "STOP"):
            raise GeminiAPIError(f"Gemini generation stopped with finish reason: {finish_reason}")

        content = candidate.get("content")
        if not isinstance(content, dict):
            raise GeminiAPIError("Gemini candidate contained no content")
        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise GeminiAPIError("Gemini candidate contained no text parts")

        text_parts = [part.get("text") for part in parts if isinstance(part, dict)]
        text = "".join(part for part in text_parts if isinstance(part, str)).strip()
        if not text:
            raise GeminiAPIError("Gemini candidate contained no text")
        return text

    @staticmethod
    def _parse_structured_output(text: str) -> tuple[AnswerStatus, str]:
        parsed = GeminiRestClient._parse_json_object(text)
        try:
            status = AnswerStatus(parsed["status"])
            answer = parsed["answer"]
        except (KeyError, ValueError) as exc:
            raise GeminiAPIError("Gemini structured response contained an invalid status") from exc

        if not isinstance(answer, str) or not answer.strip():
            raise GeminiAPIError("Gemini structured response contained an invalid answer")
        return status, answer.strip()

    @staticmethod
    def _parse_plan_output(text: str) -> tuple[AgentStrategy, str]:
        parsed = GeminiRestClient._parse_json_object(text)
        try:
            strategy = AgentStrategy(parsed["strategy"])
            reason = parsed["reason"]
        except (KeyError, ValueError) as exc:
            raise GeminiAPIError("Gemini planner returned an invalid strategy") from exc
        if not isinstance(reason, str) or not reason.strip():
            raise GeminiAPIError("Gemini planner returned an invalid reason")
        reason = " ".join(reason.split())
        if len(reason) > 240:
            raise GeminiAPIError("Gemini planner returned an excessively long reason")
        return strategy, reason

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiAPIError("Gemini structured response was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise GeminiAPIError("Gemini structured response was not an object")
        return parsed

    @staticmethod
    def _parse_usage(payload: dict[str, Any]) -> TokenUsage:
        usage = payload.get("usageMetadata")
        if not isinstance(usage, dict):
            return TokenUsage()

        def as_int(field: str) -> int:
            value = usage.get(field, 0)
            return value if isinstance(value, int) and value >= 0 else 0

        return TokenUsage(
            prompt_tokens=as_int("promptTokenCount"),
            output_tokens=as_int("candidatesTokenCount"),
            thinking_tokens=as_int("thoughtsTokenCount"),
            total_tokens=as_int("totalTokenCount"),
        )

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip() or "unknown error"
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        return response.text.strip() or "unknown error"
