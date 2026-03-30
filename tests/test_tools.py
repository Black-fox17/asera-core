import pytest
import asyncio
from asera.tools.function_node import FunctionNode
from asera.tools.registry import ToolRegistry
from asera.exceptions import ToolNotFoundError, ToolExecutionError


def sample_sync_func(a: int, b: str = "default") -> str:
    """A test sample function."""
    return f"{b}_{a}"


async def sample_async_func(x: float, y: float) -> float:
    """Async sum function."""
    await asyncio.sleep(0.01)
    return x + y


def test_function_node_schema_extraction():
    node = FunctionNode(sample_sync_func)
    assert node.name == "sample_sync_func"
    assert "A test sample function." in node.description

    schema = node.schema
    assert schema["name"] == "sample_sync_func"
    params = schema["parameters"]
    assert params["type"] == "object"
    assert "a" in params["properties"]
    assert params["properties"]["a"]["type"] == "integer"
    assert "b" in params["properties"]
    assert params["properties"]["b"]["type"] == "string"
    assert "a" in params["required"]
    assert "b" not in params["required"]


@pytest.mark.asyncio
async def test_function_node_execution():
    sync_node = FunctionNode(sample_sync_func)
    res_sync = await sync_node.execute(a=42, b="test")
    assert res_sync == "test_42"

    async_node = FunctionNode(sample_async_func)
    assert async_node.is_async is True
    res_async = await async_node.execute(x=2.5, y=3.5)
    assert res_async == 6.0


def test_tool_registry():
    registry = ToolRegistry()
    assert len(registry) == 0

    @registry.register
    def multiply(a: int, b: int) -> int:
        return a * b

    assert registry.has("multiply")
    assert len(registry) == 1

    tool = registry.get("multiply")
    assert tool.name == "multiply"

    with pytest.raises(ToolNotFoundError):
        registry.get("non_existent_tool")
