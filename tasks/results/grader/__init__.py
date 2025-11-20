from __future__ import annotations

from typing import Any


def verify(result: Any, expected: list[dict[str, Any]]) -> bool:
    """
    Verify that the result matches the expected answer.
    
    The result should be a list of match dictionaries with the same structure
    as the expected answer. Order doesn't matter, but all matches must be present.
    """
    if not isinstance(result, list):
        return False
    
    if len(result) != len(expected):
        return False
    
    # Convert both to sets of tuples for comparison (order doesn't matter)
    # Sort by key to ensure consistent comparison
    def normalize_match(match: dict[str, Any]) -> tuple:
        return tuple(sorted(match.items()))
    
    result_set = {normalize_match(m) for m in result}
    expected_set = {normalize_match(m) for m in expected}
    
    return result_set == expected_set

