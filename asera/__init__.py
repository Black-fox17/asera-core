"""
Asera Core: Modern Multi-Agent Orchestration & Recursive Context Reasoning Engine.
"""

from asera.core.agent import Agent
from asera.core.supervisor import SupervisorAgent
from asera.core.system import MultiAgentSystem
from asera.tools import FunctionNode, ToolRegistry, tool
from asera.context import RLMContext, LocalREPL
from asera.memory import (
    SessionState,
    AgentStateManager,
    Checkpoint,
    BaseCheckpointSaver,
    InMemoryCheckpointSaver,
    PostgresCheckpointSaver
)
from asera.types import AgentRole, Message, ToolExecutionResult
from asera.exceptions import (
    AseraError,
    AgentExecutionError,
    ToolExecutionError,
    ContextExhaustedError,
    SessionNotFoundError
)

__version__ = "0.2.0"

__all__ = [
    # Core
    "Agent",
    "SupervisorAgent",
    "MultiAgentSystem",
    # Tools
    "FunctionNode",
    "ToolRegistry",
    "tool",
    # Context (RLM)
    "RLMContext",
    "LocalREPL",
    # Memory
    "SessionState",
    "AgentStateManager",
    "Checkpoint",
    "BaseCheckpointSaver",
    "InMemoryCheckpointSaver",
    "PostgresCheckpointSaver",
    # Types & Exceptions
    "AgentRole",
    "Message",
    "ToolExecutionResult",
    "AseraError",
    "AgentExecutionError",
    "ToolExecutionError",
    "ContextExhaustedError",
    "SessionNotFoundError"
]
