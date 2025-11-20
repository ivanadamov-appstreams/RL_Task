from __future__ import annotations

from anthropic.types import ToolUnionParam

from main import python_expression_tool, submit_answer_tool
from utils.file_tools import read_text_file_tool, write_text_file_tool
from ...rl_task_base import ToolHandler


def build_tools(
    *, dataset_path: str | None, output_path: str | None
) -> list[ToolUnionParam]:
    tools: list[ToolUnionParam] = [
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
                "properties": {
                    "answer": {
                        "description": "JSON object containing number, count, positions"
                    }
                },
                "required": ["answer"],
            },
        },
    ]
    if dataset_path:
        tools.append(
            {
                "name": "read_numbers_file",
                "description": "Reads a dataset of numbers from disk",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative file path to read",
                        }
                    },
                    "required": ["path"],
                },
            }
        )
    if output_path:
        tools.append(
            {
                "name": "write_result_file",
                "description": "Writes the final JSON result to disk",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Destination file path",
                        },
                        "content": {
                            "type": "string",
                            "description": "JSON payload to write to disk",
                        },
                    },
                    "required": ["path", "content"],
                },
            }
        )
    return tools


def build_tool_handlers(
    *, dataset_path: str | None, output_path: str | None
) -> dict[str, ToolHandler]:
    handlers: dict[str, ToolHandler] = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }
    if dataset_path:
        handlers["read_numbers_file"] = read_text_file_tool
    if output_path:
        handlers["write_result_file"] = write_text_file_tool
    return handlers

