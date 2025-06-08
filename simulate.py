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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pathlib import Path


class ModelOperatorParser:
    """Parse operators and their dependencies from a HuggingFace model."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.operators = {}
        self.dependencies = defaultdict(set)
        self.hooks = []
        self.tensor_shapes = {}
        
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
        """Create a forward hook to capture input/output tensor shapes."""
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
    
    def create_dependency_graph(self) -> nx.DiGraph:
        """Create a NetworkX directed graph from the dependencies."""
        G = nx.DiGraph()
        
        # Add all operators as nodes
        for name, info in self.operators.items():
            G.add_node(name, 
                      operator_type=info['type'],
                      parameters=info['parameters'],
                      tensor_shapes=self.tensor_shapes.get(name, {}))
        
        # Add edges for dependencies
        for module, deps in self.dependencies.items():
            for dep in deps:
                if dep in self.operators:  # Only add edges to actual operators
                    G.add_edge(dep, module)
        
        return G
    
    def _get_operator_color(self, operator_type: str) -> str:
        """Get color for operator type."""
        color_map = {
            'Linear': '#FF6B6B',           # Red
            'Embedding': '#4ECDC4',        # Teal
            'LayerNorm': '#45B7D1',        # Blue
            'RMSNorm': '#45B7D1',          # Blue
            'Dropout': '#96CEB4',          # Light Green
            'GELU': '#FFEAA7',             # Yellow
            'ReLU': '#FFEAA7',             # Yellow
            'SiLU': '#FFEAA7',             # Yellow
            'Softmax': '#FFEAA7',          # Yellow
            'MultiHeadAttention': '#DDA0DD', # Plum
            'Conv1D': '#FFB347',           # Orange
            'Conv2D': '#FFB347',           # Orange
            'ModuleList': '#D3D3D3',       # Light Gray
            'Sequential': '#D3D3D3',       # Light Gray
        }
        
        # For model-specific operators, use purple variants
        if any(model_name in operator_type for model_name in ['Qwen', 'Bert', 'GPT', 'Distil']):
            return '#9B59B6'  # Purple
        
        return color_map.get(operator_type, '#95A5A6')  # Default gray
    
    def _create_hierarchical_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Create a hierarchical layout based on module hierarchy."""
        pos = {}
        
        # Group nodes by their hierarchy level
        levels = defaultdict(list)
        for node in G.nodes():
            level = len(node.split('.'))
            levels[level].append(node)
        
        # Position nodes level by level
        y_spacing = 2.0
        x_spacing = 1.5
        
        for level, nodes in levels.items():
            y = -level * y_spacing
            
            # Sort nodes within level for better organization
            nodes.sort()
            
            # Calculate x positions to center the level
            total_width = (len(nodes) - 1) * x_spacing
            start_x = -total_width / 2
            
            for i, node in enumerate(nodes):
                x = start_x + i * x_spacing
                pos[node] = (x, y)
        
        return pos
    
    def visualize_dependency_graph(self, output_file: str = None, max_nodes: int = 50, 
                                 layout: str = 'hierarchical', show_labels: bool = True,
                                 figsize: Tuple[int, int] = (20, 16)):
        """
        Visualize the dependency graph.
        
        Args:
            output_file: Path to save the visualization (optional)
            max_nodes: Maximum number of nodes to display (for performance)
            layout: Layout algorithm ('hierarchical', 'spring', 'circular')
            show_labels: Whether to show node labels
            figsize: Figure size (width, height)
        """
        if not self.operators:
            print("No operators found. Run parse_operators() first.")
            return
        
        print(f"Creating dependency graph visualization...")
        
        # Create the graph
        G = self.create_dependency_graph()
        
        # Limit nodes if graph is too large
        if len(G.nodes()) > max_nodes:
            print(f"Graph has {len(G.nodes())} nodes. Limiting to {max_nodes} most connected nodes.")
            
            # Get nodes with highest degree (most connections)
            node_degrees = dict(G.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_names = [node for node, _ in top_nodes]
            
            # Create subgraph with top nodes
            G = G.subgraph(top_node_names).copy()
        
        # Set up the plot
        plt.figure(figsize=figsize)
        plt.title(f"Operator Dependency Graph - {self.model_name}\n"
                 f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}", 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Choose layout
        if layout == 'hierarchical':
            pos = self._create_hierarchical_layout(G)
        elif layout == 'spring':
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Get node colors based on operator types
        node_colors = []
        operator_types = set()
        
        for node in G.nodes():
            if node in self.operators:
                op_type = self.operators[node]['type']
                operator_types.add(op_type)
                node_colors.append(self._get_operator_color(op_type))
            else:
                node_colors.append('#95A5A6')  # Gray for unknown
        
        # Calculate node sizes based on parameters
        node_sizes = []
        for node in G.nodes():
            if node in self.operators:
                params = self.operators[node]['parameters']
                # Scale node size based on parameters (log scale for better visualization)
                if params > 0:
                    size = max(100, min(1000, 100 + (params / 1000000) * 200))
                else:
                    size = 50
                node_sizes.append(size)
            else:
                node_sizes.append(100)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                              alpha=0.8, linewidths=1, edgecolors='black')
        
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                              arrowsize=20, arrowstyle='->', alpha=0.6, width=1)
        
        if show_labels:
            # Create labels (shortened for readability)
            labels = {}
            for node in G.nodes():
                if '.' in node:
                    # Show only the last part of the module name
                    labels[node] = node.split('.')[-1]
                else:
                    labels[node] = node
            
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        # Create legend for operator types
        legend_elements = []
        for op_type in sorted(operator_types):
            color = self._get_operator_color(op_type)
            legend_elements.append(mpatches.Patch(color=color, label=op_type))
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                      fontsize=10, title="Operator Types", title_fontsize=12)
        
        # Add statistics text
        stats_text = f"Total Operators: {len(self.operators)}\n"
        stats_text += f"Total Parameters: {sum(info['parameters'] for info in self.operators.values()):,}\n"
        stats_text += f"Displayed Nodes: {len(G.nodes())}\n"
        stats_text += f"Dependencies: {len(G.edges())}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save or show the plot
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Dependency graph saved to: {output_file}")
        else:
            # Save to default location
            default_file = f"{self.model_name.replace('/', '_')}_dependency_graph.png"
            plt.savefig(default_file, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Dependency graph saved to: {default_file}")
        
        plt.show()
        
        return G


def main():
    parser = argparse.ArgumentParser(description='LLM Simulator - Parse operators from HuggingFace models')
    parser.add_argument('--model', required=True, help='HuggingFace model name (e.g., Qwen/Qwen3-0.6B)')
    parser.add_argument('--ops', action='store_true', help='Parse and display operators with input/output tensor shapes')
    parser.add_argument('--deps', action='store_true', help='Show dependency graph')
    parser.add_argument('--summary', action='store_true', help='Show model summary')
    parser.add_argument('--visualize', action='store_true', help='Create graphical dependency graph visualization')
    
    # Visualization options
    parser.add_argument('--output', type=str, help='Output file for visualization (default: auto-generated)')
    parser.add_argument('--max-nodes', type=int, default=50, help='Maximum nodes to display in visualization (default: 50)')
    parser.add_argument('--layout', choices=['hierarchical', 'spring', 'circular'], default='hierarchical',
                       help='Layout algorithm for visualization (default: hierarchical)')
    parser.add_argument('--no-labels', action='store_true', help='Hide node labels in visualization')
    parser.add_argument('--figsize', type=int, nargs=2, default=[20, 16], metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size for visualization (default: 20 16)')
    
    args = parser.parse_args()
    
    if not args.ops and not args.deps and not args.summary and not args.visualize:
        print("Please specify at least one of: --ops, --deps, --summary, --visualize")
        sys.exit(1)
    
    # Create parser instance
    parser_instance = ModelOperatorParser(args.model)
    
    # Load model
    if not parser_instance.load_model():
        sys.exit(1)
    
    # Parse operators
    parser_instance.parse_operators()
    
    # Display requested information
    if args.summary:
        parser_instance.display_summary()
    
    if args.ops:
        parser_instance.display_operators()
    
    if args.deps:
        parser_instance.display_dependency_graph()
    
    if args.visualize:
        parser_instance.visualize_dependency_graph(
            output_file=args.output,
            max_nodes=args.max_nodes,
            layout=args.layout,
            show_labels=not args.no_labels,
            figsize=tuple(args.figsize)
        )


if __name__ == '__main__':
    main()