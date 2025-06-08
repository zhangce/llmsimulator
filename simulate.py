#!/usr/bin/env python3
"""
LLM Simulator - Parse operators and dependencies from HuggingFace models
"""

import argparse
import sys
from typing import Dict, List, Set, Tuple, Any, Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from collections import defaultdict, deque


class ModelOperatorParser:
    """Parse operators and their dependencies from a HuggingFace model."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.operators = {}
        self.dependencies = defaultdict(set)
        self.hooks = []
        self.tensor_shapes = {}
        self.execution_order = []
        self.execution_dependencies = defaultdict(set)
        self.multi_config_shapes = {}  # Store shapes for different configurations
        
    def load_model(self):
        """Load the HuggingFace model."""
        try:
            print(f"Loading model: {self.model_name}")
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print(f"Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            return False
    
    def get_operator_type(self, module: nn.Module) -> str:
        """Get the operator type from a PyTorch module."""
        module_type = type(module).__name__
        
        # Map common PyTorch modules to operator types
        operator_mapping = {
            'Linear': 'Linear',
            'Conv1d': 'Conv1D',
            'Conv2d': 'Conv2D',
            'LayerNorm': 'LayerNorm',
            'RMSNorm': 'RMSNorm',
            'Embedding': 'Embedding',
            'MultiheadAttention': 'MultiHeadAttention',
            'GELU': 'GELU',
            'ReLU': 'ReLU',
            'SiLU': 'SiLU',
            'Tanh': 'Tanh',
            'Softmax': 'Softmax',
            'Dropout': 'Dropout',
            'ModuleList': 'ModuleList',
            'Sequential': 'Sequential',
        }
        
        return operator_mapping.get(module_type, module_type)
    
    def parse_operators(self):
        """Parse all operators from the model."""
        if not hasattr(self, 'model'):
            print("Model not loaded. Call load_model() first.")
            return
        
        print("\nParsing operators...")
        
        # Walk through all named modules
        for name, module in self.model.named_modules():
            if name == '':  # Skip the root module
                continue
                
            operator_type = self.get_operator_type(module)
            
            # Get module parameters info
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            # Store operator info
            self.operators[name] = {
                'type': operator_type,
                'module': module,
                'parameters': param_count,
                'shape_info': self._get_shape_info(module)
            }
            
            # Determine dependencies based on module hierarchy
            self._analyze_dependencies(name)
        
        # Capture tensor shapes through forward pass
        self._capture_tensor_shapes()
    
    def _get_shape_info(self, module: nn.Module) -> Dict:
        """Extract shape information from a module."""
        shape_info = {}
        
        if hasattr(module, 'weight') and module.weight is not None:
            shape_info['weight_shape'] = list(module.weight.shape)
        
        if hasattr(module, 'bias') and module.bias is not None:
            shape_info['bias_shape'] = list(module.bias.shape)
            
        if hasattr(module, 'in_features'):
            shape_info['in_features'] = module.in_features
            
        if hasattr(module, 'out_features'):
            shape_info['out_features'] = module.out_features
            
        if hasattr(module, 'num_embeddings'):
            shape_info['num_embeddings'] = module.num_embeddings
            
        if hasattr(module, 'embedding_dim'):
            shape_info['embedding_dim'] = module.embedding_dim
            
        return shape_info
    
    def _analyze_dependencies(self, module_name: str):
        """Analyze dependencies between modules based on hierarchy."""
        parts = module_name.split('.')
        
        # Add dependency on parent modules
        for i in range(1, len(parts)):
            parent_name = '.'.join(parts[:i])
            if parent_name in self.operators:
                self.dependencies[module_name].add(parent_name)
        
        # For sequential modules, add dependencies on previous modules
        if len(parts) >= 2:
            parent_parts = parts[:-1]
            current_idx = parts[-1]
            
            # If the current part is a number, it might be in a sequential container
            try:
                idx = int(current_idx)
                if idx > 0:
                    prev_module = '.'.join(parent_parts + [str(idx - 1)])
                    if prev_module in self.operators:
                        self.dependencies[module_name].add(prev_module)
            except ValueError:
                pass
    
    def _get_tensor_shape(self, tensor: Any) -> List[int]:
        """Extract shape from tensor, handling various tensor types."""
        if tensor is None:
            return []
        
        if isinstance(tensor, torch.Tensor):
            return list(tensor.shape)
        elif isinstance(tensor, (tuple, list)):
            # For tuple/list of tensors, return shape of first tensor
            if len(tensor) > 0 and isinstance(tensor[0], torch.Tensor):
                return list(tensor[0].shape)
            return []
        else:
            return []
    
    def _create_forward_hook(self, name: str):
        """Create a forward hook to capture input/output tensor shapes and execution order."""
        def hook(module, input, output):
            input_shapes = []
            output_shapes = []
            
            # Track execution order
            self.execution_order.append(name)
            
            # Capture input shapes
            if isinstance(input, (tuple, list)):
                for inp in input:
                    shape = self._get_tensor_shape(inp)
                    if shape:
                        input_shapes.append(shape)
            else:
                shape = self._get_tensor_shape(input)
                if shape:
                    input_shapes.append(shape)
            
            # Capture output shapes
            if isinstance(output, (tuple, list)):
                for out in output:
                    shape = self._get_tensor_shape(out)
                    if shape:
                        output_shapes.append(shape)
            else:
                shape = self._get_tensor_shape(output)
                if shape:
                    output_shapes.append(shape)
            
            self.tensor_shapes[name] = {
                'input_shapes': input_shapes,
                'output_shapes': output_shapes
            }
        
        return hook
    
    def _register_hooks(self):
        """Register forward hooks on all modules to capture tensor shapes."""
        print("Registering hooks to capture tensor shapes...")
        
        for name, module in self.model.named_modules():
            if name == '':  # Skip root module
                continue
            
            hook = module.register_forward_hook(self._create_forward_hook(name))
            self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _create_sample_input(self) -> torch.Tensor:
        """Create a sample input tensor for the model."""
        # Use a simple sequence for text models
        sample_text = "Hello, this is a sample input for tensor shape analysis."
        
        try:
            # Try to tokenize the sample text
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                # Add pad token if it doesn't exist
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                inputs = self.tokenizer(
                    sample_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                return inputs['input_ids']
            else:
                # Fallback: create a simple input tensor based on config
                vocab_size = getattr(self.config, 'vocab_size', 50000)
                seq_length = 32
                return torch.randint(0, min(vocab_size, 1000), (1, seq_length))
        
        except Exception as e:
            print(f"Warning: Could not create proper tokenized input, using fallback: {e}")
            # Fallback: create a simple input tensor
            vocab_size = getattr(self.config, 'vocab_size', 50000)
            seq_length = 32
            return torch.randint(0, min(vocab_size, 1000), (1, seq_length))
    
    def _capture_tensor_shapes(self):
        """Run a forward pass to capture tensor shapes."""
        print("Running forward pass to capture tensor shapes...")
        
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Create sample input
            sample_input = self._create_sample_input()
            print(f"Sample input shape: {list(sample_input.shape)}")
            
            # Register hooks
            self._register_hooks()
            
            # Run forward pass
            with torch.no_grad():
                _ = self.model(sample_input)
            
            print(f"Captured tensor shapes for {len(self.tensor_shapes)} modules")
            
        except Exception as e:
            print(f"Warning: Could not capture tensor shapes: {e}")
        finally:
            # Always remove hooks
            self._remove_hooks()
            
        # Build execution dependencies based on actual execution order
        self._build_execution_dependencies()
    
    def _create_input_for_config(self, batch_size: int, context_length: int, mode: str) -> torch.Tensor:
        """Create input tensor for specific configuration."""
        try:
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                # Add pad token if it doesn't exist
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                if mode == "prefill":
                    # For prefill, use the full context length
                    sample_text = "Hello, this is a sample input for tensor shape analysis. " * (context_length // 10 + 1)
                    inputs = self.tokenizer(
                        sample_text, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=context_length
                    )
                    input_ids = inputs['input_ids']
                    # Repeat for batch size
                    if batch_size > 1:
                        input_ids = input_ids.repeat(batch_size, 1)
                    return input_ids
                else:  # decode mode
                    # For decode, model still attends to full context but processes 1 new token
                    # Create a context with the specified length to simulate KV cache state
                    sample_text = "Hello, this is a sample input for tensor shape analysis. " * (context_length // 10 + 1)
                    inputs = self.tokenizer(
                        sample_text, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=context_length
                    )
                    input_ids = inputs['input_ids']
                    # Repeat for batch size
                    if batch_size > 1:
                        input_ids = input_ids.repeat(batch_size, 1)
                    return input_ids
            else:
                # Fallback: create tensor based on config
                vocab_size = getattr(self.config, 'vocab_size', 50000)
                if mode == "prefill":
                    return torch.randint(0, min(vocab_size, 1000), (batch_size, context_length))
                else:  # decode mode
                    return torch.randint(0, min(vocab_size, 1000), (batch_size, context_length))
        
        except Exception as e:
            print(f"Warning: Could not create proper input for config, using fallback: {e}")
            vocab_size = getattr(self.config, 'vocab_size', 50000)
            if mode == "prefill":
                return torch.randint(0, min(vocab_size, 1000), (batch_size, context_length))
            else:  # decode mode
                return torch.randint(0, min(vocab_size, 1000), (batch_size, context_length))
    
    def _capture_multi_config_shapes(self):
        """Capture tensor shapes for multiple configurations."""
        print("\nCapturing tensor shapes for multiple configurations...")
        
        # Configuration parameters (reduced for efficiency)
        batch_sizes = [1, 16]
        context_lengths = [1, 128]  # Use more reasonable context lengths
        modes = ["prefill", "decode"]
        
        self.multi_config_shapes = {}
        
        for batch_size in batch_sizes:
            for context_length in context_lengths:
                for mode in modes:
                    # Skip invalid combinations
                    if mode == "decode" and context_length > 1:
                        continue
                    
                    config_key = f"bs{batch_size}_ctx{context_length}_{mode}"
                    print(f"  Testing configuration: {config_key}")
                    
                    try:
                        # Set model to evaluation mode
                        self.model.eval()
                        
                        # Create input for this configuration
                        sample_input = self._create_input_for_config(batch_size, context_length, mode)
                        
                        # Clear previous shapes
                        config_shapes = {}
                        
                        # Create hooks for this configuration
                        def create_config_hook(name: str, config_key: str):
                            def hook(module, input, output):
                                input_shapes = []
                                output_shapes = []
                                
                                # Capture input shapes
                                if isinstance(input, (tuple, list)):
                                    for inp in input:
                                        shape = self._get_tensor_shape(inp)
                                        if shape:
                                            input_shapes.append(shape)
                                else:
                                    shape = self._get_tensor_shape(input)
                                    if shape:
                                        input_shapes.append(shape)
                                
                                # Capture output shapes
                                if isinstance(output, (tuple, list)):
                                    for out in output:
                                        shape = self._get_tensor_shape(out)
                                        if shape:
                                            output_shapes.append(shape)
                                else:
                                    shape = self._get_tensor_shape(output)
                                    if shape:
                                        output_shapes.append(shape)
                                
                                config_shapes[name] = {
                                    'input_shapes': input_shapes,
                                    'output_shapes': output_shapes
                                }
                            return hook
                        
                        # Register hooks for this configuration
                        config_hooks = []
                        for name, module in self.model.named_modules():
                            if name == '':  # Skip root module
                                continue
                            hook = module.register_forward_hook(create_config_hook(name, config_key))
                            config_hooks.append(hook)
                        
                        # Run forward pass
                        with torch.no_grad():
                            _ = self.model(sample_input)
                        
                        # Store shapes for this configuration
                        self.multi_config_shapes[config_key] = config_shapes
                        
                        # Remove hooks for this configuration
                        for hook in config_hooks:
                            hook.remove()
                        
                    except Exception as e:
                        print(f"    Warning: Could not capture shapes for {config_key}: {e}")
                        continue
        
        print(f"Captured shapes for {len(self.multi_config_shapes)} configurations")
    
    def _build_execution_dependencies(self):
        """Build execution dependencies based on the actual execution order."""
        print("Building execution dependencies...")
        
        # Clear previous execution dependencies
        self.execution_dependencies.clear()
        
        # Build dependencies based on execution order
        for i, current_module in enumerate(self.execution_order):
            # Each module depends on all modules that executed before it
            for j in range(i):
                prev_module = self.execution_order[j]
                if prev_module != current_module:  # Avoid self-dependencies
                    self.execution_dependencies[current_module].add(prev_module)
        
        print(f"Built execution dependencies for {len(self.execution_dependencies)} modules")
    
    def _topological_sort(self) -> List[str]:
        """Perform topological sort based on execution dependencies."""
        # Use Kahn's algorithm for topological sorting
        
        # Calculate in-degrees
        in_degree = defaultdict(int)
        all_modules = set(self.execution_order)
        
        # Initialize in-degrees
        for module in all_modules:
            in_degree[module] = 0
        
        # Calculate in-degrees based on execution dependencies
        for module, deps in self.execution_dependencies.items():
            for dep in deps:
                if dep in all_modules:
                    in_degree[module] += 1
        
        # Initialize queue with modules having no dependencies
        queue = deque([module for module in all_modules if in_degree[module] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # For each module that depends on current module
            for module, deps in self.execution_dependencies.items():
                if current in deps:
                    in_degree[module] -= 1
                    if in_degree[module] == 0:
                        queue.append(module)
        
        # Check for cycles (shouldn't happen with execution order, but good to verify)
        if len(result) != len(all_modules):
            print("Warning: Cycle detected in execution dependencies, using execution order")
            return self.execution_order
        
        return result
    
    def display_topological_order(self):
        """Display operators in topological order based on execution."""
        if not self.execution_order:
            print("No execution order captured. Run parse_operators() first.")
            return
        
        print(f"\n{'='*80}")
        print(f"OPERATORS IN TOPOLOGICAL ORDER: {self.model_name}")
        print(f"{'='*80}")
        print(f"Total operators: {len(set(self.execution_order))}")
        print()
        
        # Get topological order
        topo_order = self._topological_sort()
        
        print("Execution order (topologically sorted):")
        print("-" * 50)
        
        for i, name in enumerate(topo_order, 1):
            if name in self.operators:
                info = self.operators[name]
                print(f"{i:3d}. {name}")
                print(f"     Type: {info['type']}")
                
                if info['parameters'] > 0:
                    print(f"     Parameters: {info['parameters']:,}")
                
                # Show input/output tensor shapes
                if name in self.tensor_shapes:
                    tensor_info = self.tensor_shapes[name]
                    if tensor_info['input_shapes']:
                        print(f"     Input shapes: {tensor_info['input_shapes']}")
                    if tensor_info['output_shapes']:
                        print(f"     Output shapes: {tensor_info['output_shapes']}")
                
                # Show execution dependencies (limited to avoid clutter)
                if name in self.execution_dependencies and self.execution_dependencies[name]:
                    deps = sorted(list(self.execution_dependencies[name]))
                    if len(deps) <= 3:
                        print(f"     Depends on: {', '.join(deps)}")
                    else:
                        print(f"     Depends on: {', '.join(deps[:3])} ... and {len(deps) - 3} more")
                
                print()
    
    def display_multi_config_shapes(self):
        """Display operators with shapes for multiple configurations."""
        if not self.multi_config_shapes:
            print("No multi-configuration shapes captured. Run parse_operators() with multi-config analysis first.")
            return
        
        print(f"\n{'='*80}")
        print(f"OPERATORS WITH MULTI-CONFIGURATION SHAPES: {self.model_name}")
        print(f"{'='*80}")
        
        # Get all unique operator names from any configuration
        all_operators = set()
        for config_shapes in self.multi_config_shapes.values():
            all_operators.update(config_shapes.keys())
        
        print(f"Total operators analyzed: {len(all_operators)}")
        print(f"Configurations tested: {len(self.multi_config_shapes)}")
        print()
        
        # Display each operator with all its configurations
        for i, op_name in enumerate(sorted(all_operators), 1):
            if op_name in self.operators:
                info = self.operators[op_name]
                print(f"{i:3d}. {op_name}")
                print(f"     Type: {info['type']}")
                
                if info['parameters'] > 0:
                    print(f"     Parameters: {info['parameters']:,}")
                
                print(f"     Shape Analysis:")
                
                # Group configurations by mode for better readability
                prefill_configs = []
                decode_configs = []
                
                for config_key in sorted(self.multi_config_shapes.keys()):
                    if op_name in self.multi_config_shapes[config_key]:
                        if 'prefill' in config_key:
                            prefill_configs.append(config_key)
                        else:
                            decode_configs.append(config_key)
                
                # Display prefill configurations
                if prefill_configs:
                    print(f"       Prefill Mode:")
                    for config_key in prefill_configs:
                        shapes = self.multi_config_shapes[config_key][op_name]
                        # Parse config key for readable format
                        parts = config_key.split('_')
                        batch_size = parts[0][2:]  # Remove 'bs'
                        context_length = parts[1][3:]  # Remove 'ctx'
                        
                        print(f"         Batch={batch_size}, Context={context_length}:")
                        if shapes['input_shapes']:
                            print(f"           Input:  {shapes['input_shapes']}")
                        if shapes['output_shapes']:
                            print(f"           Output: {shapes['output_shapes']}")
                
                # Display decode configurations
                if decode_configs:
                    print(f"       Decode Mode:")
                    for config_key in decode_configs:
                        shapes = self.multi_config_shapes[config_key][op_name]
                        # Parse config key for readable format
                        parts = config_key.split('_')
                        batch_size = parts[0][2:]  # Remove 'bs'
                        
                        print(f"         Batch={batch_size}:")
                        if shapes['input_shapes']:
                            print(f"           Input:  {shapes['input_shapes']}")
                        if shapes['output_shapes']:
                            print(f"           Output: {shapes['output_shapes']}")
                
                print()
    
    def display_operators(self):
        """Display all operators and their information."""
        if not self.operators:
            print("No operators found. Run parse_operators() first.")
            return
        
        print(f"\n{'='*80}")
        print(f"OPERATORS IN MODEL: {self.model_name}")
        print(f"{'='*80}")
        print(f"Total operators found: {len(self.operators)}")
        print()
        
        # Group operators by type
        operators_by_type = defaultdict(list)
        for name, info in self.operators.items():
            operators_by_type[info['type']].append((name, info))
        
        # Display operators grouped by type
        for op_type, ops in sorted(operators_by_type.items()):
            print(f"\n{op_type} ({len(ops)} instances):")
            print("-" * 50)
            
            for name, info in sorted(ops):
                print(f"  {name}")
                if info['parameters'] > 0:
                    print(f"    Parameters: {info['parameters']:,}")
                
                if info['shape_info']:
                    for key, value in info['shape_info'].items():
                        print(f"    {key}: {value}")
                
                # Show input/output tensor shapes
                if name in self.tensor_shapes:
                    tensor_info = self.tensor_shapes[name]
                    if tensor_info['input_shapes']:
                        print(f"    Input tensor shapes: {tensor_info['input_shapes']}")
                    if tensor_info['output_shapes']:
                        print(f"    Output tensor shapes: {tensor_info['output_shapes']}")
                
                # Show dependencies
                if name in self.dependencies and self.dependencies[name]:
                    deps = sorted(list(self.dependencies[name]))
                    print(f"    Dependencies: {', '.join(deps[:3])}")
                    if len(deps) > 3:
                        print(f"      ... and {len(deps) - 3} more")
                print()
    
    def display_dependency_graph(self):
        """Display the dependency graph."""
        print(f"\n{'='*80}")
        print("DEPENDENCY GRAPH")
        print(f"{'='*80}")
        
        if not self.dependencies:
            print("No dependencies found.")
            return
        
        for module, deps in sorted(self.dependencies.items()):
            if deps:
                print(f"{module} depends on:")
                for dep in sorted(deps):
                    print(f"  -> {dep}")
                print()
    
    def get_statistics(self):
        """Get model statistics."""
        if not self.operators:
            return {}
        
        stats = {
            'total_operators': len(self.operators),
            'total_parameters': sum(info['parameters'] for info in self.operators.values()),
            'operator_types': len(set(info['type'] for info in self.operators.values())),
            'operators_by_type': defaultdict(int)
        }
        
        for info in self.operators.values():
            stats['operators_by_type'][info['type']] += 1
        
        return stats
    
    def display_summary(self):
        """Display a summary of the model."""
        stats = self.get_statistics()
        
        print(f"\n{'='*80}")
        print("MODEL SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Total operators: {stats['total_operators']}")
        print(f"Total parameters: {stats['total_parameters']:,}")
        print(f"Unique operator types: {stats['operator_types']}")
        print()
        
        print("Operator distribution:")
        for op_type, count in sorted(stats['operators_by_type'].items()):
            print(f"  {op_type}: {count}")


def main():
    parser = argparse.ArgumentParser(description='LLM Simulator - Parse operators from HuggingFace models')
    parser.add_argument('--model', required=True, help='HuggingFace model name (e.g., Qwen/Qwen3-0.6B)')
    parser.add_argument('--ops', action='store_true', help='Parse and display operators with input/output tensor shapes')
    parser.add_argument('--deps', action='store_true', help='Show dependency graph')
    parser.add_argument('--summary', action='store_true', help='Show model summary')
    parser.add_argument('--topological', action='store_true', help='Display operators in topological order based on execution')
    parser.add_argument('--multi-config', action='store_true', help='Analyze tensor shapes across multiple batch sizes, context lengths, and prefill/decode modes')
    
    args = parser.parse_args()
    
    if not args.ops and not args.deps and not args.summary and not args.topological and not getattr(args, 'multi_config', False):
        print("Please specify at least one of: --ops, --deps, --summary, --topological, --multi-config")
        sys.exit(1)
    
    # Create parser instance
    parser_instance = ModelOperatorParser(args.model)
    
    # Load model
    if not parser_instance.load_model():
        sys.exit(1)
    
    # Parse operators
    parser_instance.parse_operators()
    
    # Run multi-config analysis if requested
    if getattr(args, 'multi_config', False):
        parser_instance._capture_multi_config_shapes()
    
    # Display requested information
    if args.summary:
        parser_instance.display_summary()
    
    if args.ops:
        parser_instance.display_operators()
    
    if args.deps:
        parser_instance.display_dependency_graph()
    
    if args.topological:
        parser_instance.display_topological_order()
    
    if getattr(args, 'multi_config', False):
        parser_instance.display_multi_config_shapes()


if __name__ == '__main__':
    main()