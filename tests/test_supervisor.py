import pytest
from asera.core.supervisor import SupervisorAgent
from asera.core.agent import Agent


def test_supervisor_initialization():
    supervisor = SupervisorAgent(api_key="fake-key", model="gemini-2.0-flash")
    assert supervisor.model == "gemini-2.0-flash"
    assert len(supervisor.agents) == 0


def test_supervisor_add_agent():
    supervisor = SupervisorAgent(api_key="fake-key")
    worker = Agent(name="coder", description="Writes code", api_key="fake-key")

    supervisor.add_agent(worker)
    assert "coder" in supervisor.agents

    schema = supervisor._create_routing_schema()
    assert schema["name"] == "route_to_agent"
    assert "coder" in schema["parameters"]["properties"]["agent_name"]["enum"]


def test_supervisor_api_triggers():
    supervisor = SupervisorAgent(api_key="fake-key")

    def mock_trigger(params):
        return {"status": "ok", "params": params}

    supervisor.register_api_trigger("deploy", mock_trigger, "Trigger deployment")
    assert "deploy" in supervisor.api_triggers

    schemas = supervisor._create_trigger_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "trigger_deploy"
