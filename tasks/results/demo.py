from __future__ import annotations

import asyncio

from .task import TeamAwayLossTask


async def run_demo() -> None:
    # Find all matches where any team was away, lost, scored at least 1 goal, and total goals > 2.5
    task = TeamAwayLossTask()
    # Use longer delays to avoid rate limits (10,000 input tokens per minute)
    # CSV content is large, so we need more time between runs
    # First run uses most tokens (reads CSV), so longer initial delay
    results = await task.run_batch(
        num_runs=5, 
        verbose=True, 
        delay_seconds=60.0,
        initial_delay_seconds=60.0
    )
    successes = sum(1 for result in results if result.success)

    print("\nSummary")
    print("-------")
    print(f"Expected answer: {task.expected_answer}")
    print(f"Pass rate: {successes}/{len(results)} ({(successes/len(results))*100:.1f}%)")
    print("Episode outcomes:", [res.value for res in results])


if __name__ == "__main__":
    asyncio.run(run_demo())

