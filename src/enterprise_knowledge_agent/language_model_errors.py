"""Shared errors for external language-model provider adapters."""


class LanguageModelAPIError(RuntimeError):
    """Raised when an external language-model provider request cannot complete safely."""
