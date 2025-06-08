import argparse
import warnings
import contextlib
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from transformers.modeling_utils import no_init_weights


def load_model(model_name: str):
    """Load a HuggingFace model on the meta device without weights."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_cls = AutoModel
    if config.architectures:
        arch = config.architectures[0]
        if "ForCausalLM" in arch:
            model_cls = AutoModelForCausalLM
        elif "ForSequenceClassification" in arch:
            model_cls = AutoModelForSequenceClassification

    # Instantiate the model on the meta device without allocating weights
    try:
        with no_init_weights():
            model = model_cls.from_config(config, trust_remote_code=True)
    except ImportError as exc:
        raise RuntimeError(f"Failed to load model '{model_name}': {exc}")
    model.to("meta")
    model.eval()
    return model


def _gather_shapes(obj):
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [_gather_shapes(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _gather_shapes(v) for k, v in obj.items()}
    return str(type(obj).__name__)


def parse_ops(model):
    """Collect operators by executing the model on meta tensors."""
    dummy = getattr(model, "dummy_inputs", None) or {"input_ids": torch.ones(1, 1, dtype=torch.long, device="meta")}
    if "inputs_embeds" not in dummy:
        dummy["inputs_embeds"] = None

    ops = []
    deps = {}
    shapes = {}
    idx = 0

    def hook(mod, inp, out):
        nonlocal idx
        name = f"{mod.__class__.__name__}_{idx}"
        ops.append((name, mod))
        shapes[name] = (_gather_shapes(inp), _gather_shapes(out))
        for t in inp:
            if isinstance(t, torch.Tensor) and hasattr(t, "_src"):
                deps.setdefault(name, set()).add(t._src)
        if isinstance(out, torch.Tensor):
            out._src = name
        idx += 1

    handles = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    model.to("meta")
    def patched_autocast(device_type=None, **kwargs):
        if device_type == "meta":
            return contextlib.nullcontext()
        return torch.autocast(device_type=device_type, **kwargs)

    orig_autocast = torch.autocast
    torch.autocast = patched_autocast
    try:
        model(**{k: (v.to("meta") if isinstance(v, torch.Tensor) else v) for k, v in dummy.items()})
    finally:
        torch.autocast = orig_autocast
    for h in handles:
        h.remove()

    result = []
    for name, _ in ops:
        input_shape, output_shape = shapes.get(name, (None, None))
        result.append((name, sorted(deps.get(name, [])), input_shape, output_shape))
    return result


def main():
    parser = argparse.ArgumentParser(description="LLM Simulator")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--ops", action="store_true", help="List operators")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    model = load_model(args.model)

    if args.ops:
        for name, deps, in_shape, out_shape in parse_ops(model):
            deps_str = ", ".join(deps)
            print(f"{name} -> [{deps_str}]  in={in_shape} out={out_shape}")


if __name__ == "__main__":
    main()
