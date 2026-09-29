import pytest
from asera.context.repl import LocalREPL
from asera.context.rlm import RLMContext


def test_local_repl_execution():
    repl = LocalREPL(variables={"initial_num": 10})
    result = repl.execute("x = initial_num * 5\nprint(f'Computed {x}')")

    assert "Computed 50" in result["stdout"]
    assert repl.get("x") == 50
    assert not result["stderr"]


def test_local_repl_error_capture():
    repl = LocalREPL()
    result = repl.execute("1 / 0")

    assert "ZeroDivisionError" in result["stderr"]


def test_rlm_extract_code_and_final():
    rlm = RLMContext(api_key="fake")
    repl = LocalREPL(variables={"ans": "verified_result"})

    # Test code extraction
    code_block = "```python\nx = 10\nprint(x)\n```"
    extracted = rlm._extract_code(code_block)
    assert extracted == "x = 10\nprint(x)"

    # Test direct final extraction
    final_text = "The answer is FINAL<<<42>>>"
    assert rlm._extract_final(final_text, repl) == "42"

    # Test variable final extraction
    final_var_text = "Look at variable FINAL_VAR<<<ans>>>"
    assert rlm._extract_final(final_var_text, repl) == "verified_result"
