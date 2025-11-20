from __future__ import annotations

import asyncio

from .task import NumberFrequencyTask


async def run_demo() -> None:
    task = NumberFrequencyTask(
        numbers=[3, 1, 7, 3, 9, 3, 4, 3, 5, 3],
        target=3,
    )
    results = await task.run_batch(num_runs=3, verbose=True)
    successes = sum(1 for result in results if result.success)

    print("\nSummary")
    print("-------")
    print(f"Expected answer: {task.expected_answer}")
    print(f"Pass rate: {successes}/{len(results)} ({(successes/len(results))*100:.1f}%)")
    print("Episode outcomes:", [res.value for res in results])


if __name__ == "__main__":
    asyncio.run(run_demo())

