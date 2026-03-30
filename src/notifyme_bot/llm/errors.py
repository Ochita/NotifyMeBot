from __future__ import annotations


class LLMServiceError(RuntimeError):
    """Raised when an LLM request fails before a usable structured response."""
