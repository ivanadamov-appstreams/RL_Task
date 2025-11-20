RL Task

===

Setup instructions:

1. Clone the repository:

   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:

   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:

   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run python -m tasks.results.demo
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution.

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.
