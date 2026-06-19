"""
SupervisorAgent: Multi-agent coordination, routing, streaming, and context synthesis.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from google import genai
from google.genai import types

from asera.core.agent import Agent
from asera.context.rlm import RLMContext
from asera.memory.state import AgentStateManager, SessionState
from asera.logging import logger
from asera.exceptions import AgentExecutionError


class SupervisorAgent:
    """
    Supervisor Agent responsible for coordinating specialized worker agents.
    
    Key Capabilities:
    - Dynamic agent routing: auto-generates dispatch schemas from agent descriptions
    - Parallel execution: runs multiple sub-agent invocations concurrently via asyncio.gather
    - Background API Triggers: dispatches long-running workflows with state tracking
    - Real-time streaming (ainvoke): yields token chunks while resolving tool chains
    - Integrated RLM Context: delegates massive context reasoning to Recursive Language Models
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        enable_rlm: bool = True
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.agents: Dict[str, Agent] = {}
        self.api_triggers: Dict[str, Dict[str, Any]] = {}
        self.state_manager = AgentStateManager()
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.rlm = RLMContext(api_key=self.api_key, model=self.model) if enable_rlm else None

        logger.info(f"SupervisorAgent initialized with model '{self.model}' (RLM enabled={enable_rlm})")

    def add_agent(self, agent: Agent) -> None:
        """Register a worker agent with the supervisor."""
        self.agents[agent.name] = agent
        logger.debug(f"Agent '{agent.name}' registered to supervisor.")

    def register_api_trigger(self, name: str, trigger_func: Callable, description: str) -> None:
        """Register a background job / API trigger callable."""
        self.api_triggers[name] = {
            "func": trigger_func,
            "description": description
        }
        logger.debug(f"API trigger '{name}' registered to supervisor.")

    def _create_routing_schema(self) -> Dict[str, Any]:
        agent_descriptions = "\n".join([
            f"- {name}: {agent.description}" for name, agent in self.agents.items()
        ])
        return {
            "name": "route_to_agent",
            "description": f"Route a specialized subtask to the most qualified worker agent. Available agents:\n{agent_descriptions}",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "enum": list(self.agents.keys()),
                        "description": "Name of the target agent"
                    },
                    "task": {
                        "type": "string",
                        "description": "Precise task description or query for the worker agent"
                    }
                },
                "required": ["agent_name", "task"]
            }
        }

    def _create_trigger_schemas(self) -> List[Dict[str, Any]]:
        schemas = []
        for name, config in self.api_triggers.items():
            schemas.append({
                "name": f"trigger_{name}",
                "description": config["description"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "params": {
                            "type": "object",
                            "description": "Parameters for the background task"
                        }
                    },
                    "required": ["params"]
                }
            })
        return schemas

    def _create_rlm_schema(self) -> Optional[Dict[str, Any]]:
        if not self.rlm:
            return None
        return {
            "name": "query_large_context",
            "description": "Query or extract specific information from massive offloaded context using recursive decomposition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question or extraction task to perform on the context"
                    }
                },
                "required": ["query"]
            }
        }

    async def _route_to_agent(self, agent_name: str, task: str, memory_messages: List[Dict[str, Any]]) -> str:
        if agent_name not in self.agents:
            return f"Error: Agent '{agent_name}' not found."
        try:
            return await self.agents[agent_name].invoke(task, context=memory_messages)
        except Exception as e:
            logger.error(f"Error in routed agent '{agent_name}': {e}")
            return f"Agent '{agent_name}' error: {e}"

    async def invoke(
        self,
        prompt: str,
        context: Optional[Any] = None,
        thread_id: Optional[str] = None,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute synchronous-style invocation of the supervisor.
        Returns a dict containing 'response' and optional 'api_triggered'.
        """
        if not self.client:
            raise AgentExecutionError("Supervisor has no configured Gemini client or GEMINI_API_KEY.")

        if thread_id:
            state = self.state_manager.get(thread_id)
            if not state:
                state = self.state_manager.initialize_session(thread_id, context=str(context) if context else None)
        else:
            state = SessionState(context=str(context) if context else None)

        function_declarations = []
        if self.agents:
            function_declarations.append(self._create_routing_schema())
        function_declarations.extend(self._create_trigger_schemas())
        rlm_schema = self._create_rlm_schema()
        if rlm_schema:
            function_declarations.append(rlm_schema)

        tools = [types.Tool(function_declarations=function_declarations)] if function_declarations else None

        base_system = system_instruction or (
            "You are a master supervisor orchestrating specialized worker agents and operations.\n"
            "Analyze the user's intent, divide complex tasks, route them via route_to_agent, "
            "and synthesize results into a coherent final response."
        )

        messages = []
        for msg in state.messages:
            role = "model" if msg["role"] in ("assistant", "model") else "user"
            messages.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        user_content = prompt
        if context and not self.rlm:
            user_content += f"\n\nContext:\n{context}"

        messages.append(types.Content(role="user", parts=[types.Part(text=user_content)]))

        config = types.GenerateContentConfig(
            system_instruction=base_system,
            tools=tools,
            temperature=0.1
        )

        api_triggered = None
        turn = 0

        while turn < 10:
            turn += 1
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=messages,
                config=config
            )

            if not response.candidates or not response.candidates[0].content:
                break

            content = response.candidates[0].content
            function_calls = [
                part.function_call for part in content.parts
                if hasattr(part, "function_call") and part.function_call
            ]

            if not function_calls:
                final_text = response.text or ""
                if thread_id:
                    self.state_manager.update_session(
                        thread_id,
                        new_messages=[
                            {"role": "user", "content": prompt},
                            {"role": "model", "content": final_text}
                        ]
                    )
                return {"response": final_text, "api_triggered": api_triggered}

            messages.append(content)

            agent_tasks = []
            api_results = []
            rlm_tasks = []

            for call in function_calls:
                call_args = dict(call.args) if call.args else {}
                if call.name == "route_to_agent":
                    agent_tasks.append(
                        self._route_to_agent(call_args["agent_name"], call_args["task"], state.messages)
                    )
                elif call.name == "query_large_context":
                    q = call_args.get("query", "")
                    rlm_tasks.append(asyncio.to_thread(self.rlm.query, q, context or state.context))
                elif call.name.startswith("trigger_"):
                    trigger_name = call.name.replace("trigger_", "")
                    if trigger_name in self.api_triggers:
                        params = call_args.get("params", {})
                        func = self.api_triggers[trigger_name]["func"]
                        res = await func(params) if asyncio.iscoroutinefunction(func) else func(params)
                        api_triggered = res
                        api_results.append(f"Trigger '{trigger_name}' started successfully: {res}")
                    else:
                        api_results.append(f"Trigger '{trigger_name}' not found.")

            resolved_agent_results = await asyncio.gather(*agent_tasks) if agent_tasks else []
            resolved_rlm_results = await asyncio.gather(*rlm_tasks) if rlm_tasks else []

            all_results = resolved_agent_results + resolved_rlm_results + api_results
            tool_responses = [
                types.Part.from_function_response(
                    name=function_calls[i].name,
                    response={"result": all_results[i] if i < len(all_results) else "Done"}
                )
                for i in range(len(function_calls))
            ]

            messages.append(types.Content(role="user", parts=tool_responses))

        return {"response": "Supervisor loop completed.", "api_triggered": api_triggered}

    async def ainvoke(
        self,
        prompt: str,
        context: Optional[Any] = None,
        thread_id: Optional[str] = None,
        system_instruction: Optional[str] = None
    ) -> AsyncGenerator[Any, None]:
        """
        Streaming invocation: yields text token chunks in real-time,
        and yields dictionaries for internal events (e.g. {'type': 'api_triggered'}).
        """
        if not self.client:
            raise AgentExecutionError("Supervisor has no configured Gemini client.")

        if thread_id:
            state = self.state_manager.get(thread_id)
            if not state:
                state = self.state_manager.initialize_session(thread_id, context=str(context) if context else None)
        else:
            state = SessionState(context=str(context) if context else None)

        function_declarations = []
        if self.agents:
            function_declarations.append(self._create_routing_schema())
        function_declarations.extend(self._create_trigger_schemas())
        rlm_schema = self._create_rlm_schema()
        if rlm_schema:
            function_declarations.append(rlm_schema)

        tools = [types.Tool(function_declarations=function_declarations)] if function_declarations else None

        base_system = system_instruction or (
            "You are a master supervisor orchestrating specialized worker agents.\n"
            "Analyze requests, route subtasks to agents, and stream results."
        )

        messages = []
        for msg in state.messages:
            role = "model" if msg["role"] in ("assistant", "model") else "user"
            messages.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        user_content = prompt
        if context and not self.rlm:
            user_content += f"\n\nContext:\n{context}"

        messages.append(types.Content(role="user", parts=[types.Part(text=user_content)]))

        config = types.GenerateContentConfig(
            system_instruction=base_system,
            tools=tools,
            temperature=0.1
        )

        accumulated_text = ""
        turn = 0

        while turn < 10:
            turn += 1
            response_stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=messages,
                config=config
            )

            full_parts = []
            async for chunk in response_stream:
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if part.text:
                            accumulated_text += part.text
                            yield part.text
                        full_parts.append(part)

            function_calls = [part.function_call for part in full_parts if part.function_call]
            if not function_calls:
                break

            messages.append(types.Content(role="model", parts=full_parts))

            agent_tasks = []
            api_results = []
            rlm_tasks = []

            for call in function_calls:
                call_args = dict(call.args) if call.args else {}
                if call.name == "route_to_agent":
                    agent_tasks.append(
                        self._route_to_agent(call_args["agent_name"], call_args["task"], state.messages)
                    )
                elif call.name == "query_large_context":
                    q = call_args.get("query", "")
                    rlm_tasks.append(asyncio.to_thread(self.rlm.query, q, context or state.context))
                elif call.name.startswith("trigger_"):
                    trigger_name = call.name.replace("trigger_", "")
                    if trigger_name in self.api_triggers:
                        params = call_args.get("params", {})
                        func = self.api_triggers[trigger_name]["func"]
                        res = await func(params) if asyncio.iscoroutinefunction(func) else func(params)
                        yield {"type": "api_triggered", "name": trigger_name, "result": res}
                        api_results.append(f"Trigger '{trigger_name}' complete.")

            resolved_agent_results = await asyncio.gather(*agent_tasks) if agent_tasks else []
            resolved_rlm_results = await asyncio.gather(*rlm_tasks) if rlm_tasks else []
            all_results = resolved_agent_results + resolved_rlm_results + api_results

            tool_responses = [
                types.Part.from_function_response(
                    name=function_calls[i].name,
                    response={"result": all_results[i] if i < len(all_results) else "Done"}
                )
                for i in range(len(function_calls))
            ]

            messages.append(types.Content(role="user", parts=tool_responses))

        if thread_id:
            self.state_manager.update_session(
                thread_id,
                new_messages=[
                    {"role": "user", "content": prompt},
                    {"role": "model", "content": accumulated_text}
                ]
            )
