"""
Prompt templates and protocols for Recursive Language Model (RLM) context reasoning.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class QueryMetadata:
    context_total_length: int = 0
    context_type: str = "str"
    history_count: int = 0


RLM_SYSTEM_PROMPT = """
You are an execution-driven Recursive Language Model (RLM) operating inside an active Python REPL.

Your objective is to answer the user's task using the offloaded `context` variable. 
Rather than reading the entire context directly into your prompt, you programmatically inspect, slice, transform, and analyze the context using Python code and targeted recursive sub-LLM calls (`llm_query`).

────────────────────────────────────────────────────
AVAILABLE REPL ENVIRONMENT
────────────────────────────────────────────────────
1. `context` (str or Any)
   The raw offloaded data (document, transcript, state, or payload).
   Inspect length, structure, or chunks via Python before answering.

2. `llm_query(prompt: str) -> str`
   Spawns a recursive sub-LLM call. Use ONLY on small, focused slices (e.g., summarizing a 500-token chunk).
   Do NOT call `llm_query` with the same global task.

3. `llm_query_batched(prompts: List[str]) -> List[str]`
   Concurrent sub-LLM calls for independent chunks or tasks.

4. `re`, `json`, `math`
   Standard libraries available in the execution environment.

────────────────────────────────────────────────────
EXECUTION RULES
────────────────────────────────────────────────────
1. Output Python code enclosed in ```python ... ``` blocks to inspect or manipulate `context`.
2. Do not attempt to guess or hallucinate content when it can be verified from `context`.
3. When you have found the answer, TERMINATE IMMEDIATELY using one of the following exact formats outside of code blocks:
   • FINAL<<<your concise final answer here>>>
   • FINAL_VAR<<<variable_name_containing_answer>>>
4. Any response without either a ```python``` block or a FINAL<<<...>>> marker is invalid.
"""


def build_rlm_system_prompt(base_prompt: str, metadata: QueryMetadata) -> List[Dict[str, str]]:
    content = base_prompt.strip()
    content += f"\n\n[Context Metadata: Type={metadata.context_type}, Length={metadata.context_total_length} characters]"
    return [{"role": "system", "content": content}]


def build_user_prompt(prompt: str, iteration: int, max_iterations: int, history_count: int = 0) -> Dict[str, str]:
    if iteration == 0:
        return {
            "role": "user",
            "content": f"Task: {prompt}\n\nPlease inspect the `context` variable in your REPL and solve the task."
        }
    return {
        "role": "user",
        "content": f"[Turn {iteration + 1}/{max_iterations}] Next step or FINAL<<<...>>>:"
    }
