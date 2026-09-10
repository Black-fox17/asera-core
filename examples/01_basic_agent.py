"""
Example 01: Creating an autonomous agent with custom tools.
"""

import asyncio
import os
from asera import Agent


async def main():
    api_key = os.getenv("GEMINI_API_KEY", "your-api-key")

    agent = Agent(
        name="math_researcher",
        description="Performs calculations and unit conversions",
        api_key=api_key,
        model="gemini-2.0-flash"
    )

    # Register custom tools using the @agent.tool decorator
    @agent.tool
    def add(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @agent.tool
    def celsius_to_fahrenheit(celsius: float) -> float:
        """Convert temperature from Celsius to Fahrenheit."""
        return (celsius * 9 / 5) + 32

    prompt = "If it is 30 degrees Celsius outside, what is that in Fahrenheit? Also add 10 to that number."
    print(f"User: {prompt}\n")

    response = await agent.invoke(prompt)
    print(f"Agent:\n{response}")


if __name__ == "__main__":
    asyncio.run(main())
