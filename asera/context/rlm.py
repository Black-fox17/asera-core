"""
RLMContext: Recursive Language Model context reasoning engine.
"""

import re
import os
import json
from typing import Optional, List, Dict, Any
from google import genai
from google.genai import types

from asera.logging import logger
from asera.exceptions import ContextExhaustedError
from asera.context.repl import LocalREPL
from asera.context.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt
)


class RLMContext:
    """
    Recursive Language Model engine designed for zero-prompt-bloat context analysis.
    Offloads massive contexts into a Python REPL and enables the LLM to recursively inspect,
    filter, and synthesize information through programmatic sub-calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        max_depth: int = 1,
        max_iterations: int = 8,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def _extract_code(self, text: str) -> Optional[str]:
        m = re.search(r"```(?:repl|python)\s*(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else None

    def _extract_final(self, text: str, repl: LocalREPL) -> Optional[str]:
        # Check for FINAL_VAR<<<var_name>>>
        m_var = re.search(r"FINAL_VAR<<<(.*?)>>>", text, re.DOTALL)
        if m_var:
            var_name = m_var.group(1).strip()
            val = repl.get(var_name)
            return str(val) if val is not None else None

        # Check for FINAL<<<answer>>>
        m_ans = re.search(r"FINAL<<<(.*?)>>>", text, re.DOTALL)
        if m_ans:
            content = m_ans.group(1).strip()
            # If model returned a variable name instead of direct content
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", content) and repl.get(content) is not None:
                return str(repl.get(content))
            return content

        return None

    def _llm_call(self, messages: List[Dict[str, str]]) -> str:
        if not self.client:
            raise ValueError("Gemini API key is not configured for RLMContext.")

        contents = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                contents.append(
                    types.Content(
                        role="model" if msg["role"] in ("assistant", "model") else "user",
                        parts=[types.Part(text=msg["content"])]
                    )
                )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        ) if system_instruction else None

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        return response.text or ""

    def query(
        self,
        prompt: str,
        context: Optional[Any] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        depth: int = 0
    ) -> str:
        """
        Execute an RLM context query.
        Recursively decomposes the problem if needed up to max_depth.
        """
        if depth >= self.max_depth:
            # Reached max recursive depth: answer directly
            return self._llm_call([
                {"role": "system", "content": "Answer the following specific query directly and concisely."},
                {"role": "user", "content": prompt}
            ])

        logger.info(f"[RLMContext] depth={depth} starting query: '{prompt[:70]}...'")

        # Sub-query callbacks exposed to the REPL
        def subcall(subtask: str) -> str:
            if subtask.strip() == prompt.strip():
                raise RuntimeError("Blocked recursive call on identical task.")
            return self.query(subtask, context=context, depth=depth + 1)

        def subcall_batched(subtasks: List[str]) -> List[str]:
            return [subcall(t) for t in subtasks]

        repl = LocalREPL(variables={
            "context": context if context is not None else "",
            "llm_query": subcall,
            "llm_query_batched": subcall_batched,
            "re": re,
            "json": json,
            "history": history or []
        })

        metadata = QueryMetadata(
            context_total_length=len(str(context)) if context else 0,
            context_type=type(context).__name__ if context else "None",
            history_count=len(history) if history else 0
        )

        messages = build_rlm_system_prompt(RLM_SYSTEM_PROMPT, metadata)
        messages.append(build_user_prompt(prompt, 0, self.max_iterations))

        for i in range(self.max_iterations):
            logger.debug(f"[RLMContext] Iteration {i + 1}/{self.max_iterations}")

            if i == self.max_iterations - 1:
                messages.append({
                    "role": "user",
                    "content": "Iteration limit reached. Return your final answer now using FINAL<<<...>>>."
                })

            response = self._llm_call(messages)

            # Check if answer is provided
            final_ans = self._extract_final(response, repl)
            if final_ans is not None:
                logger.info(f"[RLMContext] Final answer resolved on iteration {i + 1}")
                return final_ans

            # Check if Python code was emitted
            code = self._extract_code(response)
            if not code:
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Please output a ```python ... ``` block to inspect context or return FINAL<<<answer>>>."
                })
                continue

            # Execute code in REPL
            result = repl.execute(code)
            obs = result["stdout"]
            if result["stderr"]:
                obs += f"\nError: {result['stderr']}"
            if not obs.strip():
                obs = "[Executed with no stdout]"

            messages.append({"role": "assistant", "content": response})
            followup = build_user_prompt(None, i + 1, self.max_iterations)
            followup["content"] = f"REPL Observation:\n{obs}\n\n{followup['content']}"
            messages.append(followup)

        raise ContextExhaustedError(f"RLM query exhausted {self.max_iterations} iterations without resolving a final answer.")
