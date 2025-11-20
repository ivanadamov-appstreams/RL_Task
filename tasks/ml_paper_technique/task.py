from __future__ import annotations

from typing import Any, Iterable

from anthropic.types import ToolUnionParam

from utils.prompt_loader import load_prompt
from ..rl_task_base import RLTask, ToolHandler
from .tools import build_tools as build_task_tools
from .tools import build_tool_handlers as build_task_tool_handlers
from .grader import verify as verify_result


class MLPaperTechniqueTask(RLTask):
    """
    Ask the agent to implement a concrete technique from an ML paper (RMSNorm)
    and return the normalized vector.
    """

    def __init__(
        self,
        *,
        vector: Iterable[float] | None = None,
        gamma: float = 1.0,
        epsilon: float = 1e-5,
        technique: str = "RMSNorm (Zhang et al., 2019)",
        description: str | None = None,
        model: str = "claude-haiku-4-5",
        max_steps: int = 20,
    ) -> None:
        super().__init__(model=model, max_steps=max_steps)
        self._vector = list(vector or [0.33, -0.71, 1.2, -0.05])
        self._gamma = gamma
        self._epsilon = epsilon
        self._technique = technique
        self._description = description or "Implement the RMSNorm technique from the referenced paper."
        self._prompt_template = load_prompt("ml_paper_technique", "prompt.md")
        self._expected_answer = self._compute_expected_answer()

    def _compute_expected_answer(self) -> list[float]:
        mean_square = sum(x * x for x in self._vector) / len(self._vector)
        denom = (mean_square + self._epsilon) ** 0.5
        return [self._gamma * (x / denom) for x in self._vector]

    @property
    def prompt(self) -> str:
        return self._prompt_template.format(
            description=self._description,
            technique=self._technique,
            vector=self._vector,
            gamma=self._gamma,
            epsilon=self._epsilon,
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

