"""
Tool registry for organizing and discovering agent tools.
"""

from typing import Dict, List, Callable, Optional, Any
from asera.tools.function_node import FunctionNode
from asera.exceptions import ToolNotFoundError


class ToolRegistry:
    """Central registry for managing FunctionNodes."""

    def __init__(self):
        self._tools: Dict[str, FunctionNode] = {}

    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False
    ) -> Callable:
        """Register a function or decorator."""
        def decorator(f: Callable) -> Callable:
            node = FunctionNode(f, name=name, description=description, return_direct=return_direct)
            self._tools[node.name] = node
            return f

        if func is not None:
            return decorator(func)
        return decorator

    def add(self, node: FunctionNode) -> None:
        self._tools[node.name] = node

    def get(self, name: str) -> FunctionNode:
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self):
        return len(self._tools)
