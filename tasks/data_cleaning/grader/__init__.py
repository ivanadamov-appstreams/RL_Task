from __future__ import annotations

from typing import Any, Iterable


def verify(result: Any, expected: Iterable[int]) -> bool:
    return result == list(expected)

