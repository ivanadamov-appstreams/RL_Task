from __future__ import annotations

from pathlib import Path
from typing import Any


def _resolve_path(path: str) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = Path(__file__).resolve().parent.parent / path
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_text_file_tool(path: str) -> dict[str, Any]:
    target = _resolve_path(path)
    content = target.read_text()
    return {"path": str(target), "content": content}


def write_text_file_tool(path: str, content: str) -> dict[str, Any]:
    target = _resolve_path(path)
    target.write_text(content)
    return {"path": str(target), "written_bytes": len(content)}

