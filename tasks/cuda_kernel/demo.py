from __future__ import annotations

import asyncio

from .task import CudaKernelTask


async def run_demo() -> None:
    task = CudaKernelTask()
    results = await task.run_batch(num_runs=3, verbose=True)
    successes = sum(1 for result in results if result.success)

    print("\nSummary")
    print("-------")
    print(f"Expected file path: {task.expected_answer['file_path']}")
    print(f"Pass rate: {successes}/{len(results)} ({(successes/len(results))*100:.1f}%)")
    print("Episode outcomes:", [res.value for res in results])


if __name__ == "__main__":
    asyncio.run(run_demo())

