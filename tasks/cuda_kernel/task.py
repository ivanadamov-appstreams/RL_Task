from __future__ import annotations

from pathlib import Path
from typing import Any

from anthropic.types import ToolUnionParam

from utils.prompt_loader import load_prompt
from ..rl_task_base import RLTask, ToolHandler
from .tools import build_tools as build_task_tools
from .tools import build_tool_handlers as build_task_tool_handlers
from .grader import verify as verify_result


class CudaKernelTask(RLTask):
    """
    Ask the agent to implement a CUDA kernel for vector addition and persist it
    to disk before submitting a summary.
    """

    def __init__(
        self,
        *,
        vector_length: int = 1024,
        kernel_function: str = "vector_add",
        kernel_path: str = "tasks/cuda_kernel/output/vector_add.cu",
        description: str | None = None,
        model: str = "claude-haiku-4-5",
        max_steps: int = 20,
    ) -> None:
        super().__init__(model=model, max_steps=max_steps)
        self._vector_length = vector_length
        self._kernel_function = kernel_function
        self._kernel_path = Path(kernel_path).resolve()
        self._kernel_path.parent.mkdir(parents=True, exist_ok=True)
        self._operation_description = "Compute C[i] = A[i] + B[i] for float32 vectors."
        self._description = description or "Write a CUDA kernel for vector addition that supports arbitrary lengths."
        self._prompt_template = load_prompt("cuda_kernel", "prompt.md")
        self._expected_answer = {
            "file_path": str(self._kernel_path),
            "function": self._kernel_function,
        }

    @property
    def prompt(self) -> str:
        return self._prompt_template.format(
            description=self._description,
            operation_description=self._operation_description,
            vector_length=self._vector_length,
            kernel_path=self._kernel_path,
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
            kernel_path=self._kernel_path,
            required_function=self._kernel_function,
        )

