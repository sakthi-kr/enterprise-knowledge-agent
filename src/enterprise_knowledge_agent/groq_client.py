"""Groq REST client for structured planning and grounded answer generation."""

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
from enterprise_knowledge_agent.language_model_errors import LanguageModelAPIError
from enterprise_knowledge_agent.llm_contract import (
    ANSWER_SCHEMA,
    GROUNDING_SYSTEM_INSTRUCTION,
    PLAN_SCHEMA,
    PLANNER_SYSTEM_INSTRUCTION,
    build_grounded_prompt,
)

_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_RETRYABLE_STATUS_CODES = {408, 422, 429}


class GroqAPIError(LanguageModelAPIError):
    """Raised when the Groq API cannot produce a valid structured response."""


class GroqRestClient:
    """Call Groq Chat Completions with strict JSON-schema responses."""

    provider_name = "groq"

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = "openai/gpt-oss-20b",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Groq API key must not be empty")
        if not _MODEL_NAME_PATTERN.fullmatch(model_name):
            raise ValueError("Groq model name contains unsupported characters")
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

    def __enter__(self) -> GroqRestClient:
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
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM_INSTRUCTION},
                {
                    "role": "user",
                    "content": f"Enterprise question:\n{question.strip()}",
                },
            ],
            "response_format": self._structured_response_format(
                name="enterprise_agent_plan",
                schema=PLAN_SCHEMA,
            ),
            "reasoning_effort": "low",
            "include_reasoning": False,
            "max_completion_tokens": 200,
            "citation_options": "disabled",
        }
        payload = self._chat_payload(body)
        strategy, reason = self._parse_plan_output(self._extract_message_text(payload))
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

        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": GROUNDING_SYSTEM_INSTRUCTION},
                {
                    "role": "user",
                    "content": build_grounded_prompt(question=question, evidence=evidence),
                },
            ],
            "response_format": self._structured_response_format(
                name="grounded_enterprise_answer",
                schema=ANSWER_SCHEMA,
            ),
            "reasoning_effort": "low",
            "include_reasoning": False,
            "max_completion_tokens": 1000,
            "citation_options": "disabled",
        }
        payload = self._chat_payload(body)
        status, answer = self._parse_structured_output(self._extract_message_text(payload))
        return LanguageModelOutput(
            status=status,
            answer=answer,
            model_name=self.model_name,
            usage=self._parse_usage(payload),
        )

    @staticmethod
    def _structured_response_format(*, name: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": True,
                "schema": schema,
            },
        }

    def _chat_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"{self._base_url}/chat/completions"
        response = self._post_with_retries(endpoint=endpoint, body=body)
        try:
            payload = response.json()
        except ValueError as exc:
            raise GroqAPIError("Groq API returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise GroqAPIError("Groq API returned a non-object JSON payload")
        return payload

    def _post_with_retries(self, *, endpoint: str, body: dict[str, Any]) -> httpx.Response:
        last_transport_error: httpx.TransportError | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
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
                self._sleep(self._retry_delay(attempt, response=response))
                continue

            message = self._extract_error_message(response)
            raise GroqAPIError(f"Groq API returned HTTP {response.status_code}: {message}")

        raise GroqAPIError("Could not reach the Groq API after retries") from last_transport_error

    def _retry_delay(self, attempt: int, *, response: httpx.Response | None = None) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after is not None:
                try:
                    parsed = float(retry_after)
                except ValueError:
                    pass
                else:
                    if parsed >= 0.0:
                        return parsed

        base_delay = self._retry_base_delay_seconds * (2**attempt)
        jitter = random.uniform(0.0, base_delay * 0.25)
        return base_delay + jitter

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in _RETRYABLE_STATUS_CODES or 500 <= status_code <= 599

    @staticmethod
    def _extract_message_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GroqAPIError("Groq API returned no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise GroqAPIError("Groq API choice had an invalid shape")
        finish_reason = choice.get("finish_reason")
        if finish_reason not in (None, "stop"):
            raise GroqAPIError(f"Groq generation stopped with finish reason: {finish_reason}")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise GroqAPIError("Groq API choice contained no message")
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise GroqAPIError("Groq API choice contained no text")
        return text.strip()

    @staticmethod
    def _parse_structured_output(text: str) -> tuple[AnswerStatus, str]:
        parsed = GroqRestClient._parse_json_object(text)
        try:
            status = AnswerStatus(parsed["status"])
            answer = parsed["answer"]
        except (KeyError, ValueError) as exc:
            raise GroqAPIError("Groq structured response contained an invalid status") from exc
        if not isinstance(answer, str) or not answer.strip():
            raise GroqAPIError("Groq structured response contained an invalid answer")
        return status, answer.strip()

    @staticmethod
    def _parse_plan_output(text: str) -> tuple[AgentStrategy, str]:
        parsed = GroqRestClient._parse_json_object(text)
        try:
            strategy = AgentStrategy(parsed["strategy"])
            reason = parsed["reason"]
        except (KeyError, ValueError) as exc:
            raise GroqAPIError("Groq planner returned an invalid strategy") from exc
        if not isinstance(reason, str) or not reason.strip():
            raise GroqAPIError("Groq planner returned an invalid reason")
        reason = " ".join(reason.split())
        if len(reason) > 240:
            raise GroqAPIError("Groq planner returned an excessively long reason")
        return strategy, reason

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GroqAPIError("Groq structured response was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise GroqAPIError("Groq structured response was not an object")
        return parsed

    @staticmethod
    def _parse_usage(payload: dict[str, Any]) -> TokenUsage:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return TokenUsage()

        def as_int(container: dict[str, Any], field: str) -> int:
            value = container.get(field, 0)
            return value if isinstance(value, int) and value >= 0 else 0

        prompt_tokens = as_int(usage, "prompt_tokens")
        completion_tokens = as_int(usage, "completion_tokens")
        details = usage.get("completion_tokens_details")
        reasoning_tokens = as_int(details, "reasoning_tokens") if isinstance(details, dict) else 0
        visible_output_tokens = max(0, completion_tokens - reasoning_tokens)
        total_tokens = as_int(usage, "total_tokens")
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            output_tokens=visible_output_tokens,
            thinking_tokens=reasoning_tokens,
            total_tokens=total_tokens,
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
