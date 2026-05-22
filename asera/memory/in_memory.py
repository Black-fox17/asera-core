"""
In-memory implementation of CheckpointSaver.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict
from asera.memory.checkpointer import Checkpoint, BaseCheckpointSaver
from asera.logging import logger


class InMemoryCheckpointSaver(BaseCheckpointSaver):
    """
    Volatile in-memory checkpoint store.
    Ideal for testing, development, and ephemeral CLI agents.
    """

    def __init__(self):
        self._storage: Dict[str, List[Checkpoint]] = defaultdict(list)

    async def save(self, checkpoint: Checkpoint) -> None:
        self._storage[checkpoint.thread_id].append(checkpoint)
        logger.debug(f"Saved in-memory checkpoint for thread '{checkpoint.thread_id}'")

    async def load(self, thread_id: str) -> Optional[List[Dict[str, Any]]]:
        checkpoints = self._storage.get(thread_id)
        if not checkpoints:
            return None
        # Return the latest checkpoint's messages
        return list(checkpoints[-1].messages)

    async def get_checkpoints(self, thread_id: str) -> List[Checkpoint]:
        return list(self._storage.get(thread_id, []))

    async def clear(self, thread_id: str) -> None:
        if thread_id in self._storage:
            del self._storage[thread_id]
            logger.debug(f"Cleared in-memory checkpoints for thread '{thread_id}'")
