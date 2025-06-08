# LLM Simulator

Given a HuggingFace model, e.g., `Qwen/Qwen3-Reranker-4B`, the LLM Simulator
loads the architecture and walks through a dummy forward pass on the `meta`
device. Leaf modules are recorded in execution order with their data
dependencies. Each operator listing also includes the shapes of its input and
output tensors.

Run the simulator with:

```
python simulate.py --model Qwen/Qwen3-Reranker-4B --ops
```

Example using DeepSeek R1:

```
python simulate.py --model deepseek-ai/DeepSeek-R1 --ops | head
```
