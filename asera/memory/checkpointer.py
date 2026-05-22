"""
Checkpointer protocol and dataclass for conversation persistence.
"""

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Checkpoint:
    """Represents a frozen snapshot of an agent or supervisor thread."""
    thread_id: str
    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class BaseCheckpointSaver(Protocol):
    """Protocol for checkpoint persistence backends."""

    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a conversation checkpoint."""
        ...

    async def load(self, thread_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load conversation messages for a thread."""
        ...

    async def clear(self, thread_id: str) -> None:
        """Delete saved checkpoints for a thread."""
        ...
