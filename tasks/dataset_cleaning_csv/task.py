from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from anthropic.types import ToolUnionParam

from utils.prompt_loader import load_prompt
from ..rl_task_base import RLTask, ToolHandler
from .tools import build_tools as build_task_tools
from .tools import build_tool_handlers as build_task_tool_handlers
from .grader import verify as verify_result


class DatasetCleaningCSVTask(RLTask):
    """
    Ask the agent to read a raw CSV, clean it according to predefined rules, and
    persist the cleaned data while reporting summary statistics.
    """

    def __init__(
        self,
        *,
        input_path: str = "tasks/dataset_cleaning_csv/data/raw_customers.csv",
        output_path: str = "tasks/dataset_cleaning_csv/data/cleaned_customers.csv",
        description: str | None = None,
        model: str = "claude-haiku-4-5",
        max_steps: int = 20,
    ) -> None:
        super().__init__(model=model, max_steps=max_steps)
        self._input_path = Path(input_path).resolve()
        self._output_path = Path(output_path).resolve()
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._description = description or "Clean the dataset by keeping only active rows with a score."
        self._prompt_template = load_prompt("dataset_cleaning_csv", "prompt.md")
        (
            self._expected_rows,
            self._expected_average,
            self._expected_csv,
        ) = self._compute_expected_outputs()
        self._expected_answer = {
            "rows_kept": self._expected_rows,
            "average_score": self._expected_average,
        }

    def _compute_expected_outputs(self) -> tuple[int, float, str]:
        with self._input_path.open() as f:
            reader = csv.DictReader(f)
            kept = [
                row
                for row in reader
                if row["status"].strip().lower() == "active" and row["score"].strip()
            ]
        scores = [float(row["score"]) for row in kept]
        average = sum(scores) / len(scores)
        header = "id,name,status,score"
        body = [f"{row['id']},{row['name']},{row['status']},{row['score']}" for row in kept]
        cleaned_csv = "\n".join([header, *body])
        return len(kept), average, cleaned_csv

    @property
    def prompt(self) -> str:
        return self._prompt_template.format(
            description=self._description,
            input_path=self._input_path,
            output_path=self._output_path,
        )

    @property
    def expected_answer(self) -> Any:
        return self._expected_answer

    def build_tools(self) -> list[ToolUnionParam]:
        return build_task_tools()

    def build_tool_handlers(self) -> dict[str, ToolHandler]:
        return build_task_tool_handlers()

    def verify(self, result: Any) -> bool:
        return verify_result(
            result,
            expected_rows=self._expected_rows,
            expected_average=self._expected_average,
            cleaned_path=self._output_path,
            expected_csv=self._expected_csv,
        )

