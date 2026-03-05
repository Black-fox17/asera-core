"""
FunctionNode: Dynamic tool schema extraction and invocation wrapper.
"""

import inspect
import asyncio
from typing import Callable, Dict, Any, get_type_hints
from asera.logging import logger
from asera.exceptions import ToolExecutionError


class FunctionNode:
    """
    Represents a callable function wrapped as an agent tool with auto-generated schema.
    Compatible with Google Gemini function declarations.
    """

    def __init__(self, func: Callable, name: str = None, description: str = None, return_direct: bool = False):
        if not callable(func):
            raise ValueError("FunctionNode requires a callable object.")

        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func) or f"Tool function {self.name}"
        self.return_direct = return_direct
        self.schema = self._create_schema()
        self.is_async = inspect.iscoroutinefunction(func)

    def _type_to_json_type(self, py_type: Any) -> str:
        """Map Python type annotation to JSON Schema type string."""
        if py_type in (int, float):
            return "number" if py_type is float else "integer"
        if py_type is bool:
            return "boolean"
        if py_type in (list, List):
            return "array"
        if py_type in (dict, Dict):
            return "object"
        return "string"

    def _create_schema(self) -> Dict[str, Any]:
        """Auto-generate JSON Schema compatible with Gemini function declaration."""
        sig = inspect.signature(self.func)
        try:
            hints = get_type_hints(self.func)
        except Exception:
            hints = {}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, param.annotation)
            json_type = self._type_to_json_type(param_type)

            properties[param_name] = {
                "type": json_type,
                "description": f"Parameter {param_name}"
            }

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    async def execute(self, **kwargs) -> Any:
        """Execute the underlying function with provided arguments (sync or async)."""
        try:
            if self.is_async:
                return await self.func(**kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: self.func(**kwargs))
        except Exception as e:
            logger.error(f"Execution error in tool '{self.name}': {e}")
            raise ToolExecutionError(f"Error executing tool '{self.name}': {e}") from e

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
