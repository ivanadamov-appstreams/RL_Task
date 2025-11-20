from __future__ import annotations

from math import isclose
from typing import Any, Iterable


def verify(result: Any, expected: Iterable[float], *, tolerance: float = 1e-6) -> bool:
    if not isinstance(result, list):
        return False
    if len(result) != len(expected):
        return False
    return all(isclose(a, b, rel_tol=tolerance, abs_tol=tolerance) for a, b in zip(result, expected))

