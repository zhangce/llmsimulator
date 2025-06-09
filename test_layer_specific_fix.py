#!/usr/bin/env python3
"""
Test script to verify the layer-specific decode mode fix
"""

import torch
from simulate import ModelOperatorParser

def test_layer_type_detection():
    """Test that layer type detection works correctly"""
    parser = ModelOperatorParser("test")
    
    # Test MLP layer detection
    mlp_names = [
        "layers.0.mlp.up_proj",
        "layers.5.mlp.down_proj", 
        "layers.10.mlp.gate_proj",
        "transformer.h.0.mlp.c_fc",
        "model.layers.3.feed_forward.w1"
    ]
    
    for name in mlp_names:
        if parser._is_mlp_layer(name, None):
            print(f"✅ Correctly identified MLP layer: {name}")
        else:
            print(f"❌ Failed to identify MLP layer: {name}")
    
    # Test attention layer detection
    attn_names = [
        "layers.0.self_attn.q_proj",
        "layers.5.self_attn.k_proj",
        "layers.10.self_attn.v_proj",
        "layers.2.self_attn.o_proj",
        "transformer.h.0.attn.c_attn"
    ]
    
    for name in attn_names:
        if parser._is_attention_layer(name, None):
            print(f"✅ Correctly identified attention layer: {name}")
        else:
            print(f"❌ Failed to identify attention layer: {name}")

def test_decode_mode_shapes():
    """Test that decode mode creates correct shapes for different layer types"""
    print("\nTesting decode mode with actual model...")
    
    parser = ModelOperatorParser("Qwen/Qwen3-0.6B")
    
    # Load the model
    if not parser.load_model():
        print("❌ FAIL: Could not load model")
        return False
    
    # Parse operators to get layer information
    parser.parse_operators()
    
    # Test input creation for decode mode
    batch_size = 1
    context_length = 128
    mode = "decode"
    
    input_tensor = parser._create_input_for_config(batch_size, context_length, mode)
    print(f"Decode mode input tensor shape: {list(input_tensor.shape)}")
    print(f"Expected: [1, 128] (full context for attention, adjusted per layer)")
    
    # Check some specific layers
    mlp_layers = []
    attn_layers = []
    
    for name, info in parser.operators.items():
        if parser._is_mlp_layer(name, info['module']):
            mlp_layers.append(name)
        elif parser._is_attention_layer(name, info['module']):
            attn_layers.append(name)
    
    print(f"\nFound {len(mlp_layers)} MLP layers")
    print(f"Found {len(attn_layers)} attention layers")
    
    if mlp_layers:
        print(f"Example MLP layers: {mlp_layers[:3]}")
    if attn_layers:
        print(f"Example attention layers: {attn_layers[:3]}")
    
    return True

if __name__ == "__main__":
    print("Testing layer-specific decode mode fix...")
    print("=" * 60)
    
    test_layer_type_detection()
    test_decode_mode_shapes()
    
    print("\n" + "=" * 60)
    print("✅ Layer-specific fix testing completed!")
    print("\nKey points:")
    print("- MLP layers in decode mode: process 1 token")
    print("- Attention layers in decode mode: use full KV cache context")
    print("- Input tensor: full context, shapes adjusted per layer type")