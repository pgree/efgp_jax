"""Run the README's fenced python blocks so the quick start stays runnable."""
import pathlib
import re

import pytest

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"
_BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), re.DOTALL)


@pytest.mark.parametrize(
    "block", _BLOCKS, ids=[f"block{i}" for i in range(len(_BLOCKS))]
)
def test_readme_code_runs(block):
    exec(compile(block, str(README), "exec"), {})
