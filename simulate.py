import argparse
import warnings
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification


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

    # Instantiate the model from the configuration only to avoid loading weights
    model = model_cls.from_config(config, trust_remote_code=True)
    model.eval()
    return model


def parse_ops(model):
    """Collect operators by executing the model on meta tensors."""
    dummy = getattr(model, "dummy_inputs", None) or {"input_ids": torch.ones(1, 1, dtype=torch.long, device="meta")}
    if "inputs_embeds" not in dummy:
        dummy["inputs_embeds"] = None

    ops = []
    deps = {}
    idx = 0

    def hook(mod, inp, out):
        nonlocal idx
        name = f"{mod.__class__.__name__}_{idx}"
        ops.append((name, mod))
        for t in inp:
            if isinstance(t, torch.Tensor) and hasattr(t, "_src"):
                deps.setdefault(name, set()).add(t._src)
        if isinstance(out, torch.Tensor):
            out._src = name
        idx += 1

    handles = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    model.to("meta")
    model(**{k: (v.to("meta") if isinstance(v, torch.Tensor) else v) for k, v in dummy.items()})
    for h in handles:
        h.remove()

    result = []
    for name, _ in ops:
        result.append((name, sorted(deps.get(name, []))))
    return result


def main():
    parser = argparse.ArgumentParser(description="LLM Simulator")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--ops", action="store_true", help="List operators")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    model = load_model(args.model)

    if args.ops:
        for name, deps in parse_ops(model):
            deps_str = ", ".join(deps)
            print(f"{name} -> [{deps_str}]")


if __name__ == "__main__":
    main()
