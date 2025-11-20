from __future__ import annotations

from anthropic.types import ToolUnionParam

from main import python_expression_tool, submit_answer_tool
from utils.file_tools import write_text_file_tool
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
            "name": "write_kernel_file",
            "description": "Writes CUDA kernel code to disk",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination .cu file path"},
                    "content": {"type": "string", "description": "CUDA kernel source"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "description": "JSON object containing file_path and summary of the kernel"
                    }
                },
                "required": ["answer"],
            },
        },
    ]


def build_tool_handlers() -> dict[str, ToolHandler]:
    return {
        "python_expression": python_expression_tool,
        "write_kernel_file": write_text_file_tool,
        "submit_answer": submit_answer_tool,
    }

