from __future__ import annotations

from pathlib import Path
from typing import Any

from anthropic.types import ToolUnionParam

from utils.prompt_loader import load_prompt
from ..rl_task_base import RLTask, ToolHandler
from .tools import build_tools as build_task_tools
from .tools import build_tool_handlers as build_task_tool_handlers
from .grader import verify as verify_result


class TeamAwayLossTask(RLTask):
    """
    Find all matches where any of the following hold:
    - Away team won.
    - Home team won and away team scored at least one goal.
    - Home team won and total goals < 3.
    """

    def __init__(
        self,
        *,
        csv_path: str | Path | None = None,
        description: str | None = None,
        model: str = "claude-haiku-4-5",
        max_steps: int = 20,
    ) -> None:
        super().__init__(model=model, max_steps=max_steps)
        
        if csv_path is None:
            script_dir = Path(__file__).parent
            csv_path = script_dir / "data" / "combined_matches.csv"
        
        self._csv_path = Path(csv_path).resolve()
        if not self._csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._csv_path}")
        
        self._description = description or (
            "Find matches where (a) the away team wins, (b) the home team wins and the away team scored, "
            "or (c) the home team wins and total goals < 3."
        )
        self._prompt_template = load_prompt("results", "prompt.md")
        self._rows = self._load_rows()
        self._expected_answer = self._build_expected_answer()

    def _load_rows(self) -> list[dict[str, Any]]:
        import csv

        rows: list[dict[str, Any]] = []
        with open(self._csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                result = (row.get("Резултат") or "").strip()
                rows.append(
                    {
                        "date": row.get("Дата", "").strip(),
                        "home": row.get("Домакин", "").strip(),
                        "result": result,
                        "away": row.get("Гост", "").strip(),
                        "round": row.get("Кръг", "").strip(),
                        "season": row.get("Сезон", "").strip(),
                    }
                )
        return rows

    def _build_expected_answer(self) -> list[dict[str, Any]]:
        """Build the expected answer by parsing the CSV."""
        matches = []
        for row in self._rows:
            result = row["result"]
            if ":" not in result:
                continue
            try:
                home_goals, away_goals = map(int, result.split(":"))
            except ValueError:
                continue
            away_win = away_goals > home_goals
            home_win = home_goals > away_goals
            total_goals = home_goals + away_goals
            home_win_with_away_goal = home_win and away_goals > 0
            home_win_low_total = home_win and total_goals < 3
            if away_win or home_win_with_away_goal or home_win_low_total:
                matches.append(row)
        return matches

    @property
    def prompt(self) -> str:
        return self._prompt_template.format(dataset=self._format_dataset())

    def _format_dataset(self) -> str:
        lines: list[str] = []
        for row in self._rows:
            if row["result"]:
                lines.append(
                    f"- {row['date']} | {row['home']} {row['result']} {row['away']} | {row['round']} | {row['season']}"
                )
        return "\n".join(lines)

    @property
    def expected_answer(self) -> Any:
        return self._expected_answer

    def build_tools(self) -> list[ToolUnionParam]:
        return build_task_tools()

    def build_tool_handlers(self) -> dict[str, ToolHandler]:
        return build_task_tool_handlers()

    def verify(self, result: Any) -> bool:
        return verify_result(result, self._expected_answer)

