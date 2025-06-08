import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simulate import load_model, parse_ops

@pytest.mark.parametrize("model_name", [
    "sshleifer/tiny-gpt2",
    "HuggingFaceH4/tiny-random-LlamaForCausalLM",
    "deepseek-ai/DeepSeek-R1"
])
def test_parse_ops(model_name):
    try:
        model = load_model(model_name)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    ops = parse_ops(model)
    assert len(ops) > 0
    # each entry should include dependency list and shape info
    name, deps, in_shape, out_shape = ops[0]
    assert isinstance(deps, list)
    assert in_shape is None or isinstance(in_shape, (list, dict))
    assert out_shape is None or isinstance(out_shape, (list, dict))
