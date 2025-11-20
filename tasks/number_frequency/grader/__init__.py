from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def verify(result: Any, expected: dict[str, Any], *, output_path: str | None) -> bool:
    if not isinstance(result, dict):
        return False

    success = (
        result.get("number") == expected["number"]
        and result.get("count") == expected["count"]
        and result.get("positions") == expected["positions"]
    )
    if not success:
        return False

    if output_path:
        path = Path(output_path)
        if not path.exists():
            return False
        saved = json.loads(path.read_text())
        return saved == expected

    return True

