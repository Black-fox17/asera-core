"""
Example 04: Real-time streaming response from Supervisor with sub-agent routing.
"""

import asyncio
import os
import sys
from asera import SupervisorAgent, Agent


async def main():
    api_key = os.getenv("GEMINI_API_KEY", "your-api-key")

    supervisor = SupervisorAgent(
        api_key=api_key,
        model="gemini-2.0-flash"
    )

    analyst = Agent(
        name="data_analyst",
        description="Analyzes dataset summaries and computes statistical trends.",
        api_key=api_key
    )

    @analyst.tool
    def compute_growth(start_val: float, end_val: float) -> str:
        """Calculate percentage growth between two numbers."""
        growth = ((end_val - start_val) / start_val) * 100
        return f"Percentage growth: {growth:.2f}%"

    supervisor.add_agent(analyst)

    prompt = "Our active user base increased from 15,000 to 42,500. Compute the growth rate and explain what this means."
    print(f"User: {prompt}\n")
    print("Supervisor Streaming: ", end="", flush=True)

    async for chunk in supervisor.ainvoke(prompt):
        if isinstance(chunk, str):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        elif isinstance(chunk, dict):
            print(f"\n[Event: {chunk}]", flush=True)

    print("\n\nStream finished.")


if __name__ == "__main__":
    asyncio.run(main())
