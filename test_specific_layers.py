#!/usr/bin/env python3
"""
Test specific layers mentioned in the issue
"""

import torch
from simulate import ModelOperatorParser

def test_specific_layers():
    """Test the specific layers mentioned in the original issue"""
    print("Testing specific layers from the original issue...")
    
    parser = ModelOperatorParser("Qwen/Qwen3-0.6B")
    
    # Load the model
    if not parser.load_model():
        print("❌ FAIL: Could not load model")
        return False
    
    # Parse operators
    parser.parse_operators()
    
    # Test the specific layer mentioned: layers.9.mlp.up_proj
    target_layer = "layers.9.mlp.up_proj"
    
    if target_layer in parser.operators:
        module = parser.operators[target_layer]['module']
        
        # Test layer type detection
        is_mlp = parser._is_mlp_layer(target_layer, module)
        is_attn = parser._is_attention_layer(target_layer, module)
        
        print(f"Layer: {target_layer}")
        print(f"  Is MLP layer: {is_mlp}")
        print(f"  Is Attention layer: {is_attn}")
        
        if is_mlp:
            print("  ✅ Correctly identified as MLP layer")
            print("  📝 In decode mode, this should process only 1 token")
        else:
            print("  ❌ Failed to identify as MLP layer")
    else:
        print(f"❌ Layer {target_layer} not found in model")
        # Show available layers
        mlp_layers = [name for name in parser.operators.keys() if 'mlp' in name.lower()]
        print(f"Available MLP layers: {mlp_layers[:10]}")
    
    # Test an attention layer for comparison
    attn_layers = [name for name in parser.operators.keys() if 'self_attn' in name.lower() and 'q_proj' in name.lower()]
    if attn_layers:
        target_attn = attn_layers[0]
        module = parser.operators[target_attn]['module']
        
        is_mlp = parser._is_mlp_layer(target_attn, module)
        is_attn = parser._is_attention_layer(target_attn, module)
        
        print(f"\nLayer: {target_attn}")
        print(f"  Is MLP layer: {is_mlp}")
        print(f"  Is Attention layer: {is_attn}")
        
        if is_attn:
            print("  ✅ Correctly identified as Attention layer")
            print("  📝 In decode mode, this should use full KV cache context")
        else:
            print("  ❌ Failed to identify as Attention layer")

def simulate_decode_shapes():
    """Simulate what the shapes should look like in decode mode"""
    print("\n" + "="*50)
    print("SIMULATED DECODE MODE SHAPES")
    print("="*50)
    
    batch_size = 1
    context_length = 128
    hidden_dim = 1024
    
    print(f"Configuration: Batch={batch_size}, Context={context_length}")
    print()
    
    # MLP layer (like layers.9.mlp.up_proj)
    print("layers.9.mlp.up_proj (MLP layer):")
    print(f"  Batch={batch_size}, Context={context_length} (MLP: processes 1 token in decode):")
    print(f"    Input:  [[{batch_size}, 1, {hidden_dim}]]")  # Only 1 token for MLP
    print(f"    Output: [[{batch_size}, 1, 3072]]")  # up_proj typically expands to 3072
    print()
    
    # Attention layer for comparison
    print("layers.9.self_attn.q_proj (Attention layer):")
    print(f"  Batch={batch_size}, Context={context_length} (Attention: uses full KV cache):")
    print(f"    Input:  [[{batch_size}, {context_length}, {hidden_dim}]]")  # Full context for attention
    print(f"    Output: [[{batch_size}, {context_length}, {hidden_dim}]]")  # Same size for q_proj
    print()
    
    print("Key Insight:")
    print("- MLP layers: Process only the new token (1 token) in decode mode")
    print("- Attention layers: Use full context for KV cache in decode mode")

if __name__ == "__main__":
    test_specific_layers()
    simulate_decode_shapes()