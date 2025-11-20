from __future__ import annotations

import asyncio

from .task import ArithmeticExpressionTask


async def run_demo() -> None:
    task = ArithmeticExpressionTask(
        expression="(2**10 + 3**5) * 7 - 100",
        expected_answer=8769,
        description="Calculate (2^10 + 3^5) * 7 - 100.",
    )
    result = await task.run_episode(verbose=True)
    print("Submitted answer:", result)
    print("Expected answer:", task.expected_answer)


if __name__ == "__main__":
    asyncio.run(run_demo())

