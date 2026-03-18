"""
Tools module for Asera Core.
"""

from asera.tools.function_node import FunctionNode
from asera.tools.registry import ToolRegistry


def tool(func=None, *, name=None, description=None, return_direct=False):
    """Convenience decorator to wrap a function into a FunctionNode."""
    def decorator(f):
        return FunctionNode(f, name=name, description=description, return_direct=return_direct)

    if func is not None:
        return decorator(func)
    return decorator


__all__ = ["FunctionNode", "ToolRegistry", "tool"]
