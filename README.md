# LLM Simulator

A tool for parsing and analyzing operators in HuggingFace models. Given a HuggingFace model, the LLM Simulator parses out all operators, their dependencies, and provides detailed analysis of the model architecture.

## Features

- **Operator Parsing**: Extract all operators/modules from any HuggingFace model
- **Dependency Analysis**: Analyze dependencies between operators based on module hierarchy
- **Model Statistics**: Get comprehensive statistics about the model
- **Multiple Output Formats**: Display operators, dependencies, or summary information

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Parse operators from a HuggingFace model:

```bash
python simulate.py --model Qwen/Qwen3-0.6B --ops
```

### Available Options

- `--model MODEL`: HuggingFace model name (required)
- `--ops`: Parse and display all operators with detailed information including input/output tensor shapes
- `--deps`: Show dependency graph between operators
- `--summary`: Show model summary with statistics

### Examples

1. **Display all operators with tensor shapes**:
   ```bash
   python simulate.py --model Qwen/Qwen3-0.6B --ops
   ```

2. **Show model summary**:
   ```bash
   python simulate.py --model distilbert-base-uncased --summary
   ```

3. **Display dependency graph**:
   ```bash
   python simulate.py --model bert-base-uncased --deps
   ```

4. **Combine multiple options**:
   ```bash
   python simulate.py --model gpt2 --ops --summary --deps
   ```

## Output Information

### Operators (`--ops`)
For each operator, the tool displays:
- Operator type (Linear, Embedding, LayerNorm, etc.)
- Parameter count
- Shape information (weight/bias shapes, input/output features)
- **Input tensor shapes** (captured during forward pass)
- **Output tensor shapes** (captured during forward pass)
- Dependencies on other operators

### Summary (`--summary`)
- Total number of operators
- Total parameters
- Number of unique operator types
- Distribution of operators by type

### Dependencies (`--deps`)
- Hierarchical dependencies between modules
- Parent-child relationships in the model architecture

## Supported Models

The tool works with any HuggingFace model that can be loaded with `AutoModel.from_pretrained()`. Tested with:
- BERT variants (bert-base-uncased, distilbert-base-uncased)
- GPT models (gpt2, gpt2-medium)
- Qwen models (Qwen/Qwen3-0.6B, Qwen/Qwen3-Reranker-4B)
- And many others

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See `requirements.txt` for full dependencies
