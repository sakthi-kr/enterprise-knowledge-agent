"""Production-oriented API request context, logging, and error primitives."""

from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi.responses import JSONResponse

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_request_id_context: ContextVar[str] = ContextVar("eka_request_id", default="-")


class ApiError(RuntimeError):
    """An expected API-facing failure with a stable public error code."""

    def __init__(self, *, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class JsonLogFormatter(logging.Formatter):
    """Render application logs as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", _request_id_context.get()),
        }
        structured = getattr(record, "structured", None)
        if isinstance(structured, dict):
            payload.update(structured)
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def configure_logging(level: str) -> None:
    """Configure process logging once with a JSON formatter."""

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {level}")

    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)


def normalize_request_id(value: str | None) -> str:
    """Reuse a safe caller request ID or generate a new opaque identifier."""

    if value and _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return uuid4().hex


def bind_request_id(request_id: str) -> Token[str]:
    """Bind one request ID to the current context."""

    return _request_id_context.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    """Restore the previous request context."""

    _request_id_context.reset(token)


def current_request_id() -> str:
    """Return the request ID bound to the current execution context."""

    return _request_id_context.get()


def api_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
) -> JSONResponse:
    """Build the stable public API error envelope."""

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )
