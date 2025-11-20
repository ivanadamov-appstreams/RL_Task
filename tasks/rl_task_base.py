from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from anthropic import RateLimitError

from anthropic.types import ToolUnionParam

from main import run_agent_loop


ToolHandler = Callable[..., Any]


@dataclass(slots=True)
class EpisodeResult:
    run_id: int
    success: bool
    value: Any


class RLTask(ABC):
    """
    Base abstraction around an RL-style task executed by an LLM agent.

    Subclasses define the task-specific prompt, tools, handlers, and verification
    rule while this class provides shared orchestration helpers such as running a
    single episode or a batch of episodes.
    """

    def __init__(self, *, model: str = "claude-haiku-4-5", max_steps: int = 20) -> None:
        self.model = model
        self.max_steps = max_steps

    @property
    @abstractmethod
    def prompt(self) -> str:
        ...

    @property
    @abstractmethod
    def expected_answer(self) -> Any:
        ...

    @abstractmethod
    def build_tools(self) -> list[ToolUnionParam]:
        ...

    @abstractmethod
    def build_tool_handlers(self) -> dict[str, ToolHandler]:
        ...

    def verify(self, result: Any) -> bool:
        return result == self.expected_answer

    async def run_episode(self, *, verbose: bool = False) -> Any | None:
        return await run_agent_loop(
            prompt=self.prompt,
            tools=self.build_tools(),
            tool_handlers=self.build_tool_handlers(),
            max_steps=self.max_steps,
            model=self.model,
            verbose=verbose,
        )

    async def run_batch(
        self,
        *,
        num_runs: int,
        verbose: bool = False,
        delay_seconds: float = 0.0,
        initial_delay_seconds: float | None = None,
    ) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        for run_id in range(1, num_runs + 1):
            try:
                value = await self.run_episode(verbose=verbose)
                success = self.verify(value)
            except RateLimitError as err:
                if verbose:
                    print(
                        "\nRate limit error encountered. Counting run as failure and continuing."
                    )
                value = {"error": "rate_limit", "details": str(err)}
                success = False
            except Exception as err:  # noqa: BLE001
                if verbose:
                    print(
                        "\nUnexpected error encountered. Counting run as failure and continuing."
                    )
                value = {"error": "exception", "details": str(err)}
                success = False
            results.append(EpisodeResult(run_id=run_id, success=success, value=value))
            # Use initial_delay_seconds for first run, delay_seconds for subsequent runs
            if run_id == 1 and initial_delay_seconds is not None and initial_delay_seconds > 0:
                if verbose:
                    print(f"\nWaiting {initial_delay_seconds}s after first run to avoid rate limits...")
                await asyncio.sleep(initial_delay_seconds)
            elif delay_seconds > 0 and run_id != num_runs:
                if verbose:
                    print(f"\nWaiting {delay_seconds}s before next run...")
                await asyncio.sleep(delay_seconds)
        return results

