"""Builders for mock Gemini / OpenAI-style HTTP bodies used in parser tests."""

from __future__ import annotations

import json
from typing import Any


def gemini_choices_response(content: dict[str, Any]) -> dict:
    """Build mock body: ``choices[0].message.content`` is JSON text."""
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(content),
                }
            }
        ]
    }
