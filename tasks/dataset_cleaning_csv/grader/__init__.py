from __future__ import annotations

from math import isclose
from pathlib import Path
from typing import Any


def verify(
    result: Any,
    *,
    expected_rows: int,
    expected_average: float,
    cleaned_path: Path,
    expected_csv: str,
    tolerance: float = 1e-6,
) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("rows_kept") != expected_rows:
        return False
    average = result.get("average_score")
    if not isinstance(average, (int, float)):
        return False
    if not isclose(float(average), expected_average, rel_tol=tolerance, abs_tol=tolerance):
        return False
    if not cleaned_path.exists():
        return False
    return cleaned_path.read_text().strip() == expected_csv.strip()

