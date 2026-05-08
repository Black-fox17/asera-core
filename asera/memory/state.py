"""
In-memory session state management and caching.
"""

from typing import Dict, List, Any, Optional
import threading
from dataclasses import dataclass, field
from asera.exceptions import SessionNotFoundError


@dataclass
class SessionState:
    """Represents the in-memory state of an ongoing multi-turn agent conversation."""
    summary: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_states: Dict[str, Any] = field(default_factory=dict)
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentStateManager:
    """
    Thread-safe in-memory cache for active conversation sessions.
    Acts as the fast L1 cache before hitting persistent storage.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AgentStateManager, cls).__new__(cls)
                    cls._instance.cache: Dict[str, SessionState] = {}
                    cls._instance.lock = threading.Lock()
        return cls._instance

    def get(self, thread_id: str) -> Optional[SessionState]:
        with self.lock:
            return self.cache.get(thread_id)

    def set(self, thread_id: str, state: SessionState) -> None:
        with self.lock:
            self.cache[thread_id] = state

    def initialize_session(
        self,
        thread_id: str,
        summary: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        tool_states: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionState:
        with self.lock:
            state = SessionState(
                summary=summary,
                messages=list(history) if history is not None else [],
                tool_states=dict(tool_states) if tool_states is not None else {},
                context=context,
                metadata=dict(metadata) if metadata is not None else {}
            )
            self.cache[thread_id] = state
            return state

    def update_session(
        self,
        thread_id: str,
        new_messages: Optional[List[Dict[str, Any]]] = None,
        new_tool_state: Optional[Dict[str, Any]] = None,
        summary: Optional[str] = None,
        context: Optional[str] = None
    ) -> None:
        with self.lock:
            if thread_id not in self.cache:
                raise SessionNotFoundError(f"Session '{thread_id}' not found in state manager.")

            state = self.cache[thread_id]
            if new_messages:
                state.messages.extend(new_messages)
            if new_tool_state:
                state.tool_states.update(new_tool_state)
            if summary is not None:
                state.summary = summary
            if context is not None:
                state.context = context

    def clear_session(self, thread_id: str) -> None:
        with self.lock:
            if thread_id in self.cache:
                del self.cache[thread_id]

    def exists(self, thread_id: str) -> bool:
        with self.lock:
            return thread_id in self.cache

    def clear_all(self) -> None:
        with self.lock:
            self.cache.clear()
