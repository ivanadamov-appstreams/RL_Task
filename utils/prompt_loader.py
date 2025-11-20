from __future__ import annotations

from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
TASKS_DIR = BASE_DIR / "tasks"


@lru_cache(maxsize=None)
def load_prompt(*relative_path: str) -> str:
    """
    Load a prompt markdown file located under the tasks directory.

    Example:
        load_prompt("arithmetic_expression", "prompt.md")
    """
    path = TASKS_DIR.joinpath(*relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found at {path}")
    return path.read_text().strip()

