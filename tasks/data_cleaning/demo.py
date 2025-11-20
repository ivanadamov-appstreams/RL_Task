from __future__ import annotations

import asyncio
from collections.abc import Sequence

from .task import EvenNumberCleaningTask


async def run_demo(numbers: Sequence[int]) -> None:
    task = EvenNumberCleaningTask(numbers)
    results = await task.run_batch(num_runs=3, verbose=True)
    successes = sum(1 for result in results if result.success)

    print("\nSummary")
    print("-------")
    print(f"Numbers: {numbers}")
    print(f"Expected even numbers: {task.expected_answer}")
    print(f"Pass rate: {successes}/{len(results)} ({(successes/len(results))*100:.1f}%)")
    print("Episode outcomes:", [res.value for res in results])


if __name__ == "__main__":
    asyncio.run(run_demo(numbers=[1, 4, 7, 10, 12, 5, 6, 3, 8, 9]))

