"""
Example 05: Multi-turn conversation persistence using Checkpoints.
"""

import asyncio
import os
from asera import MultiAgentSystem, InMemoryCheckpointSaver


async def main():
    api_key = os.getenv("GEMINI_API_KEY", "your-api-key")

    # Attach in-memory checkpointer (or PostgresCheckpointSaver for DB persistence)
    saver = InMemoryCheckpointSaver()

    system = MultiAgentSystem(
        api_key=api_key,
        model="gemini-2.0-flash",
        checkpoint_saver=saver
    )

    system.create_agent(
        name="assistant",
        description="Friendly, knowledgeable assistant with long-term memory."
    )

    thread_id = "session-user-blackfox-001"
    system.set_thread(thread_id)

    # Turn 1
    t1 = "Hi! My favorite programming language is Rust and I build distributed systems."
    print(f"User (Turn 1): {t1}")
    r1 = await system.run(t1)
    print(f"Agent (Turn 1): {r1}\n")

    # Turn 2 (Verifies memory recall)
    t2 = "Can you remind me what my favorite language was and suggest a library for it?"
    print(f"User (Turn 2): {t2}")
    r2 = await system.run(t2)
    print(f"Agent (Turn 2): {r2}\n")

    # Inspect persistent checkpoints
    history = await system.get_history()
    print(f"Total messages saved in thread '{thread_id}': {len(history)}")


if __name__ == "__main__":
    asyncio.run(main())
