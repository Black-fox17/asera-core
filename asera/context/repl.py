"""
Local Python REPL execution environment for RLM context analysis.
"""

import io
import contextlib
import traceback
from typing import Dict, Any, Optional
from asera.logging import logger


class LocalREPL:
    """
    Isolated in-process Python execution environment.
    Captures stdout and stderr, preserves local state between statements,
    and exposes sub-LLM querying functions to the running script.
    """

    def __init__(self, variables: Optional[Dict[str, Any]] = None):
        self.locals: Dict[str, Any] = {}
        # Provide standard safe builtins
        if variables:
            self.locals.update(variables)

    def execute(self, code: str, env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute python code and return standard output, standard error, and updated locals.
        """
        if env:
            self.locals.update(env)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                # Execute the code block in the isolated scope
                exec(code, self.locals, self.locals)
        except Exception:
            traceback.print_exc(file=stderr_capture)

        stdout_val = stdout_capture.getvalue()
        stderr_val = stderr_capture.getvalue()

        return {
            "stdout": stdout_val,
            "stderr": stderr_val,
            "locals": self.locals
        }

    def get(self, var_name: str, default: Any = None) -> Any:
        return self.locals.get(var_name, default)

    def reset(self) -> None:
        self.locals.clear()
