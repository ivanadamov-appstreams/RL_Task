from __future__ import annotations

from pathlib import Path
from typing import Any


def verify(result: Any, *, kernel_path: Path, required_function: str) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("file_path") != str(kernel_path):
        return False
    if required_function not in result.get("summary", ""):
        return False

    if not kernel_path.exists():
        return False

    source = kernel_path.read_text()
    required_tokens = [
        "__global__",
        required_function,
        "threadIdx.x",
        "blockIdx.x",
        "blockDim.x",
    ]
    return all(token in source for token in required_tokens)

