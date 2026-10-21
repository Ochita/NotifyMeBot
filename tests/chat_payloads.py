"""Mock OpenAI-style chat completion bodies for parser tests."""

from __future__ import annotations

import json
from typing import Any


def chat_completion_response(content: dict[str, Any]) -> dict:
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
