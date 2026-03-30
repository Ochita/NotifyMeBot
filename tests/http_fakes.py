"""Shared stand-ins for patched ``httpx`` responses in tests."""

from __future__ import annotations


class JsonHttpResponse:
    """Minimal async-client response for ``httpx.AsyncClient.post`` mocks."""

    def __init__(self, body: dict) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body
