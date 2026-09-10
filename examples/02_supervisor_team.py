"""
Example 02: Multi-Agent Collaboration using Supervisor routing.
"""

import asyncio
import os
from asera import MultiAgentSystem


async def main():
    api_key = os.getenv("GEMINI_API_KEY", "your-api-key")

    system = MultiAgentSystem(
        api_key=api_key,
        model="gemini-2.0-flash"
    )

    # 1. Specialized Researcher Agent
    researcher = system.create_agent(
        name="web_researcher",
        description="Gathers technical facts, documentation summaries, and verified information."
    )

    @researcher.tool
    def search_docs(topic: str) -> str:
        """Fetch documentation on a technical topic."""
        return f"Documentation for '{topic}': AsyncIO uses an event loop to run cooperative multitasking tasks."

    # 2. Specialized Technical Writer Agent
    writer = system.create_agent(
        name="tech_writer",
        description="Transforms raw technical facts into concise, engaging documentation."
    )

    @writer.tool
    def format_markdown_callout(title: str, text: str) -> str:
        """Format content into a markdown callout note."""
        return f"> **{title}**\n> {text}"

    # Run the collaborative team
    query = "Research how Python AsyncIO handles multitasking, and write a formatted markdown tip for developers."
    print(f"Task: {query}\n")

    response = await system.run(query)
    print(f"Supervisor Response:\n{response}")


if __name__ == "__main__":
    asyncio.run(main())
