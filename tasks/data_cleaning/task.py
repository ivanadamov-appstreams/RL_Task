from __future__ import annotations

from typing import Any, Iterable

from anthropic.types import ToolUnionParam

from utils.prompt_loader import load_prompt
from ..rl_task_base import RLTask, ToolHandler
from .tools import build_tools as build_task_tools
from .tools import build_tool_handlers as build_task_tool_handlers
from .grader import verify as verify_result


class EvenNumberCleaningTask(RLTask):
    """
    Remove odd numbers from the dataset and return only the even values in
    their original order.
    """

    def __init__(
        self,
        numbers: Iterable[int],
        *,
        description: str | None = None,
        model: str = "claude-haiku-4-5",
        max_steps: int = 20,
    ) -> None:
        super().__init__(model=model, max_steps=max_steps)
        self._numbers = list(numbers)
        self._description = description or "Clean the dataset by removing all odd numbers."
        self._expected_answer = [n for n in self._numbers if n % 2 == 0]
        self._prompt_template = load_prompt("data_cleaning", "prompt.md")

    @property
    def prompt(self) -> str:
        return self._prompt_template.format(
            description=self._description,
            numbers=self._numbers,
        )

    @property
    def expected_answer(self) -> Any:
        return self._expected_answer

    def build_tools(self) -> list[ToolUnionParam]:
        return build_task_tools()

    def build_tool_handlers(self) -> dict[str, ToolHandler]:
        return build_task_tool_handlers()

    def verify(self, result: Any) -> bool:
        return verify_result(result, self._expected_answer)

