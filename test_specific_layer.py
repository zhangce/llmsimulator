#!/usr/bin/env python3
"""
Test script to show the specific layer mentioned in the issue
"""

import torch
from simulate import ModelOperatorParser

def test_specific_layer():
    """Test the specific layer mentioned in the issue"""
    parser = ModelOperatorParser("Qwen/Qwen3-0.6B")
    
    # Load the model
    if not parser.load_model():
        print("❌ FAIL: Could not load model")
        return False
    
    # Parse operators
    parser.parse_operators()
    
    # Test with smaller configuration to avoid long execution time
    batch_sizes = [1]
    context_lengths = [128]
    modes = ["prefill", "decode"]
    
    parser.multi_config_shapes = {}
    
    for batch_size in batch_sizes:
        for context_length in context_lengths:
            for mode in modes:
                config_key = f"bs{batch_size}_ctx{context_length}_{mode}"
                print(f"Testing configuration: {config_key}")
                
                try:
                    # Set model to evaluation mode
                    parser.model.eval()
                    
                    # Create input for this configuration
                    sample_input = parser._create_input_for_config(batch_size, context_length, mode)
                    print(f"  Input shape: {list(sample_input.shape)}")
                    
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
                                    shape = parser._get_tensor_shape(inp)
                                    if shape:
                                        input_shapes.append(shape)
                            else:
                                shape = parser._get_tensor_shape(input)
                                if shape:
                                    input_shapes.append(shape)
                            
                            # Capture output shapes
                            if isinstance(output, (tuple, list)):
                                for out in output:
                                    shape = parser._get_tensor_shape(out)
                                    if shape:
                                        output_shapes.append(shape)
                            else:
                                shape = parser._get_tensor_shape(output)
                                if shape:
                                    output_shapes.append(shape)
                            
                            config_shapes[name] = {
                                'input_shapes': input_shapes,
                                'output_shapes': output_shapes
                            }
                        
                        return hook
                    
                    # Register hooks only for the specific layer we're interested in
                    target_layer = "layers.9.mlp.up_proj"
                    hooks = []
                    
                    for name, module in parser.model.named_modules():
                        if name == target_layer:
                            hook = module.register_forward_hook(create_config_hook(name, config_key))
                            hooks.append(hook)
                            break
                    
                    # Run forward pass
                    with torch.no_grad():
                        _ = parser.model(sample_input)
                    
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()
                    
                    # Store results
                    parser.multi_config_shapes[config_key] = config_shapes
                    
                    # Print results for the target layer
                    if target_layer in config_shapes:
                        shapes = config_shapes[target_layer]
                        print(f"  {target_layer}:")
                        if shapes['input_shapes']:
                            print(f"    Input:  {shapes['input_shapes']}")
                        if shapes['output_shapes']:
                            print(f"    Output: {shapes['output_shapes']}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                
                print()

if __name__ == "__main__":
    print("Testing specific layer: layers.9.mlp.up_proj")
    print("=" * 60)
    test_specific_layer()
    print("=" * 60)
    print("✅ Test completed!")