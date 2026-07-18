"""
MultiAgentSystem: High-level orchestrator facade for rapid application building.
"""

import os
from typing import Optional, List, Dict, Any
from asera.core.agent import Agent
from asera.core.supervisor import SupervisorAgent
from asera.memory.checkpointer import BaseCheckpointSaver, Checkpoint
from asera.logging import logger


class MultiAgentSystem:
    """
    Developer-friendly high-level facade for managing multi-agent systems.
    
    Provides simple methods to:
    - Instantiate agents
    - Set conversation thread IDs
    - Persist checkpoints across turns
    - Run coordinated multi-agent tasks
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        checkpoint_saver: Optional[BaseCheckpointSaver] = None,
        enable_rlm: bool = True
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.checkpoint_saver = checkpoint_saver
        self.supervisor = SupervisorAgent(
            api_key=self.api_key,
            model=self.model,
            enable_rlm=enable_rlm
        )
        self._current_thread_id: Optional[str] = None

        logger.info("MultiAgentSystem initialized successfully.")

    def create_agent(
        self,
        name: str,
        description: str,
        system_instruction: Optional[str] = None
    ) -> Agent:
        """Create and register a specialized worker agent."""
        agent = Agent(
            name=name,
            description=description,
            api_key=self.api_key,
            model=self.model,
            system_instruction=system_instruction
        )
        self.supervisor.add_agent(agent)
        return agent

    def set_thread(self, thread_id: str) -> None:
        """Set the active conversation thread ID."""
        self._current_thread_id = thread_id

    async def run(
        self,
        prompt: str,
        context: Optional[Any] = None,
        thread_id: Optional[str] = None
    ) -> str:
        """Run the supervisor with the given prompt and return the text response."""
        active_thread = thread_id or self._current_thread_id

        # Hydrate from persistent checkpoint if needed
        if active_thread and self.checkpoint_saver:
            state = self.supervisor.state_manager.get(active_thread)
            if not state:
                history = await self.checkpoint_saver.load(active_thread)
                if history:
                    self.supervisor.state_manager.initialize_session(
                        active_thread,
                        history=history
                    )

        result = await self.supervisor.invoke(
            prompt=prompt,
            context=context,
            thread_id=active_thread
        )

        response_text = result["response"]

        # Persist to checkpointer if configured
        if active_thread and self.checkpoint_saver:
            state = self.supervisor.state_manager.get(active_thread)
            if state:
                await self.checkpoint_saver.save(
                    Checkpoint(
                        thread_id=active_thread,
                        messages=state.messages,
                        summary=state.summary,
                        metadata=state.metadata
                    )
                )

        return response_text

    async def get_history(self, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve message history for a conversation thread."""
        active_thread = thread_id or self._current_thread_id
        if not active_thread:
            return []

        state = self.supervisor.state_manager.get(active_thread)
        if state:
            return list(state.messages)

        if self.checkpoint_saver:
            loaded = await self.checkpoint_saver.load(active_thread)
            return loaded or []

        return []

    async def clear_thread(self, thread_id: Optional[str] = None) -> None:
        """Clear conversation history for a thread."""
        active_thread = thread_id or self._current_thread_id
        if active_thread:
            self.supervisor.state_manager.clear_session(active_thread)
            if self.checkpoint_saver:
                await self.checkpoint_saver.clear(active_thread)
            logger.info(f"Thread '{active_thread}' cleared.")
