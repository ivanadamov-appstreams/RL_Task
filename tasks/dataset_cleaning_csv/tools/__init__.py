from __future__ import annotations

from anthropic.types import ToolUnionParam

from main import python_expression_tool, submit_answer_tool
from utils.file_tools import read_text_file_tool, write_text_file_tool
from ...rl_task_base import ToolHandler


def build_tools() -> list[ToolUnionParam]:
    return [
        {
            "name": "read_dataset_file",
            "description": "Reads the raw CSV dataset from disk",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the CSV file that should be read",
                    }
                },
                "required": ["path"],
            },
        },
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
            "name": "write_clean_file",
            "description": "Writes the cleaned CSV back to disk",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination CSV path"},
                    "content": {"type": "string", "description": "Cleaned CSV contents"},
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
                        "description": "JSON object containing rows_kept and average_score"
                    }
                },
                "required": ["answer"],
            },
        },
    ]


def build_tool_handlers() -> dict[str, ToolHandler]:
    return {
        "read_dataset_file": read_text_file_tool,
        "python_expression": python_expression_tool,
        "write_clean_file": write_text_file_tool,
        "submit_answer": submit_answer_tool,
    }

