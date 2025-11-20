from __future__ import annotations

from anthropic.types import ToolUnionParam

from main import python_expression_tool, submit_answer_tool
from ...rl_task_base import ToolHandler


def build_tools() -> list[ToolUnionParam]:
    return [
        {
            "name": "python_expression",
            "description": "Evaluates a Python expression",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Will be passed to exec(). Use print() to output something. Returns stdout.",
                    }
                },
                "required": ["expression"],
            },
        },
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
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

