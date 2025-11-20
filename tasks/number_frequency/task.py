from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from anthropic.types import ToolUnionParam

from utils.prompt_loader import load_prompt
from ..rl_task_base import RLTask, ToolHandler
from .tools import build_tools as build_task_tools
from .tools import build_tool_handlers as build_task_tool_handlers
from .grader import verify as verify_result


class NumberFrequencyTask(RLTask):
    """
    Count how many times a target number appears in a dataset, returning the
    count and the 0-based positions where it occurs.
    """

    def __init__(
        self,
        *,
        numbers: Iterable[int] | None = None,
        dataset_path: str | None = None,
        output_path: str | None = None,
        target: int,
        description: str | None = None,
        model: str = "claude-haiku-4-5",
        max_steps: int = 20,
    ) -> None:
        super().__init__(model=model, max_steps=max_steps)
        if numbers is None and dataset_path is None:
            msg = "Either numbers or dataset_path must be provided."
            raise ValueError(msg)

        self._numbers = list(numbers) if numbers is not None else None
        self._resolved_numbers: list[int] | None = self._numbers
        self._dataset_path = Path(dataset_path).resolve() if dataset_path else None
        self._output_path = Path(output_path).resolve() if output_path else None
        self._target = target
        self._description = description or (
            "Analyze the dataset and describe how often the target number occurs."
        )
        self._prompt_template = load_prompt("number_frequency", "prompt.md")
        self._expected_answer = self._build_expected_answer()

    def _load_numbers(self) -> Sequence[int]:
        if self._resolved_numbers is not None:
            return self._resolved_numbers
        if not self._dataset_path:
            raise ValueError("No dataset source defined.")
        data = json.loads(self._dataset_path.read_text())
        if not isinstance(data, list):
            raise ValueError("Dataset file must contain a JSON list of integers.")
        if not all(isinstance(item, int) for item in data):
            raise ValueError("Dataset file must contain only integers.")
        self._resolved_numbers = list(data)
        return self._resolved_numbers

    def _build_expected_answer(self) -> dict[str, Any]:
        numbers = self._load_numbers()
        positions = [idx for idx, value in enumerate(numbers) if value == self._target]
        return {
            "number": self._target,
            "count": len(positions),
            "positions": positions,
        }

    @property
    def prompt(self) -> str:
        if self._dataset_path:
            dataset_instructions = (
                f"Read the dataset by calling read_numbers_file with path '{self._dataset_path}'."
            )
            if self._output_path:
                dataset_instructions += (
                    f"\nAfter computing the JSON, call write_result_file with path '{self._output_path}' "
                    "to persist the answer before calling submit_answer."
                )
        else:
            dataset_instructions = f"Dataset: {self._resolved_numbers}"

        return self._prompt_template.format(
            description=self._description,
            target=self._target,
            dataset_instructions=dataset_instructions,
        )

    @property
    def expected_answer(self) -> Any:
        return self._expected_answer

    def build_tools(self) -> list[ToolUnionParam]:
        dataset_path = str(self._dataset_path) if self._dataset_path else None
        output_path = str(self._output_path) if self._output_path else None
        return build_task_tools(dataset_path=dataset_path, output_path=output_path)

    def build_tool_handlers(self) -> dict[str, ToolHandler]:
        dataset_path = str(self._dataset_path) if self._dataset_path else None
        output_path = str(self._output_path) if self._output_path else None
        return build_task_tool_handlers(
            dataset_path=dataset_path,
            output_path=output_path,
        )

    def verify(self, result: Any) -> bool:
        output_path = str(self._output_path) if self._output_path else None
        return verify_result(result, self._expected_answer, output_path=output_path)

