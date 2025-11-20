from __future__ import annotations

from typing import Any

from anthropic.types import ToolUnionParam

from main import submit_answer_tool
from ...rl_task_base import ToolHandler


def build_tools() -> list[ToolUnionParam]:
    return [
        {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "The final answer to submit"}},
                "required": ["answer"],
            },
        },
    ]


def build_tool_handlers() -> dict[str, ToolHandler]:
    return {
        "submit_answer": submit_answer_tool,
    }

