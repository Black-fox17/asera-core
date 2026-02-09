"""
Custom exceptions for Asera Core.
"""


class AseraError(Exception):
    """Base exception for all Asera Core errors."""
    pass


class AgentExecutionError(AseraError):
    """Raised when an agent execution loop encounters an error."""
    pass


class ToolExecutionError(AseraError):
    """Raised when tool execution fails."""
    pass


class ToolNotFoundError(AseraError):
    """Raised when an agent requests an unregistered tool."""
    pass


class ContextExhaustedError(AseraError):
    """Raised when RLM context processing exceeds maximum iterations or depth."""
    pass


class SessionNotFoundError(AseraError):
    """Raised when an expected thread session is not found in memory."""
    pass
