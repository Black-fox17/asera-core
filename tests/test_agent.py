import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from asera.core.agent import Agent
from asera.exceptions import AgentExecutionError


def test_agent_initialization():
    agent = Agent(
        name="test_worker",
        description="A worker for testing",
        api_key="fake-key-1234",
        model="gemini-2.0-flash"
    )
    assert agent.name == "test_worker"
    assert agent.model == "gemini-2.0-flash"
    assert len(agent.tools) == 0


def test_agent_tool_registration():
    agent = Agent(name="worker", description="Worker", api_key="fake")

    @agent.tool
    def calculate_tax(amount: float, rate: float = 0.1) -> float:
        """Calculate tax on amount."""
        return amount * rate

    assert "calculate_tax" in agent.tools
    assert agent.tools["calculate_tax"].name == "calculate_tax"


@pytest.mark.asyncio
async def test_agent_invoke_missing_client():
    agent = Agent(name="no_key_agent", description="No key", api_key="")
    with pytest.raises(AgentExecutionError):
        await agent.invoke("Hello")
