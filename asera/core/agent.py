"""
Core Agent implementation for Asera.
"""

import os
import asyncio
from typing import Optional, Any, List, Dict, Callable
from google import genai
from google.genai import types

from asera.tools.function_node import FunctionNode
from asera.memory.state import AgentStateManager, SessionState
from asera.logging import logger
from asera.exceptions import AgentExecutionError


class Agent:
    """
    Intelligent agent powered by Google Gemini 2.0.
    
    Supports:
    - Dynamic tool registration via @agent.tool
    - Automatic JSON Schema generation
    - Thought signature preservation (Gemini 2.0 Flash Thinking)
    - Conversation checkpointing & thread isolation
    """

    def __init__(
        self,
        name: str,
        description: str,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        system_instruction: Optional[str] = None
    ):
        self.name = name
        self.description = description
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.system_instruction = system_instruction
        self.tools: Dict[str, FunctionNode] = {}
        self.state_manager = AgentStateManager()
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

        logger.info(f"Agent '{self.name}' initialized with model '{self.model}'")

    def tool(self, func: Optional[Callable] = None, *, name: Optional[str] = None, description: Optional[str] = None, return_direct: bool = False) -> Callable:
        """Decorator to register a tool with the agent."""
        def decorator(f: Callable) -> Callable:
            node = FunctionNode(f, name=name, description=description, return_direct=return_direct)
            self.tools[node.name] = node
            logger.debug(f"Tool '{node.name}' registered to agent '{self.name}'")
            return f

        if func is not None:
            return decorator(func)
        return decorator

    def add_tool(self, node: FunctionNode) -> None:
        self.tools[node.name] = node

    async def invoke(
        self,
        prompt: str,
        context: Optional[Any] = None,
        thread_id: Optional[str] = None,
        max_turns: int = 10
    ) -> str:
        """
        Execute the agent on a task with full tool-calling loop support.
        """
        if not self.client:
            raise AgentExecutionError(f"Agent '{self.name}' has no configured Gemini client or GEMINI_API_KEY.")

        logger.info(f"Agent '{self.name}' invoked with: '{prompt[:70]}...'")

        if thread_id:
            state = self.state_manager.get(thread_id)
            if not state:
                state = self.state_manager.initialize_session(thread_id, context=str(context) if context else None)
        else:
            state = SessionState(context=str(context) if context else None)

        # Build Gemini function declarations
        function_declarations = [tool.schema for tool in self.tools.values()]
        tools = [types.Tool(function_declarations=function_declarations)] if function_declarations else None

        system_text = self.system_instruction or f"You are {self.name}, {self.description}."
        if state.summary:
            system_text += f"\n\nContext summary: {state.summary}"
        if context:
            system_text += f"\n\nProvided Context:\n{context}"

        messages = []
        for msg in state.messages:
            role = "model" if msg["role"] in ("assistant", "model") else "user"
            messages.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        messages.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        config = types.GenerateContentConfig(
            system_instruction=system_text,
            tools=tools,
            temperature=0.2
        )

        turn = 0
        while turn < max_turns:
            turn += 1

            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=messages,
                config=config
            )

            if not response.candidates or not response.candidates[0].content:
                return "No response generated."

            candidate = response.candidates[0]
            content = candidate.content

            # Check for function calls
            function_calls = [
                part.function_call for part in content.parts
                if hasattr(part, "function_call") and part.function_call
            ]

            if not function_calls:
                # Text response complete
                final_text = response.text or ""
                if thread_id:
                    self.state_manager.update_session(
                        thread_id,
                        new_messages=[
                            {"role": "user", "content": prompt},
                            {"role": "model", "content": final_text}
                        ]
                    )
                return final_text

            # Append model message (preserving any thought signatures)
            messages.append(content)

            # Execute tool calls
            tool_responses = []
            for call in function_calls:
                tool_name = call.name
                tool_args = dict(call.args) if call.args else {}

                if tool_name not in self.tools:
                    res_val = f"Error: Tool '{tool_name}' not found."
                else:
                    tool_node = self.tools[tool_name]
                    try:
                        res_val = await tool_node.execute(**tool_args)
                        if tool_node.return_direct:
                            return str(res_val)
                    except Exception as err:
                        res_val = f"Error executing tool '{tool_name}': {err}"

                tool_responses.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": res_val}
                    )
                )

            messages.append(types.Content(role="user", parts=tool_responses))

        raise AgentExecutionError(f"Agent '{self.name}' exceeded maximum turn limit ({max_turns}).")
