#!/usr/bin/env python3
"""
Test script to verify the decode mode fix
"""

import torch
from simulate import ModelOperatorParser

def test_decode_mode_fix():
    """Test that decode mode creates correct input shapes"""
    parser = ModelOperatorParser("Qwen/Qwen3-0.6B")
    
    # Load the model to initialize config
    if not parser.load_model():
        print("❌ FAIL: Could not load model")
        return False
    
    # Test decode mode input creation
    batch_size = 1
    context_length = 128
    mode = "decode"
    
    # Create input for decode mode
    input_tensor = parser._create_input_for_config(batch_size, context_length, mode)
    
    print(f"Decode mode input shape: {list(input_tensor.shape)}")
    print(f"Expected: [1, 1] (batch_size=1, sequence_length=1)")
    
    # Verify the shape is correct
    expected_shape = [batch_size, 1]  # Should be [1, 1] for decode mode
    actual_shape = list(input_tensor.shape)
    
    if actual_shape == expected_shape:
        print("✅ PASS: Decode mode input shape is correct!")
        return True
    else:
        print(f"❌ FAIL: Expected {expected_shape}, got {actual_shape}")
        return False

def test_prefill_mode():
    """Test that prefill mode still works correctly"""
    parser = ModelOperatorParser("Qwen/Qwen3-0.6B")
    
    # Load the model to initialize config
    if not parser.load_model():
        print("❌ FAIL: Could not load model")
        return False
    
    # Test prefill mode input creation
    batch_size = 1
    context_length = 128
    mode = "prefill"
    
    # Create input for prefill mode
    input_tensor = parser._create_input_for_config(batch_size, context_length, mode)
    
    print(f"Prefill mode input shape: {list(input_tensor.shape)}")
    print(f"Expected: [1, {context_length}] (batch_size=1, sequence_length={context_length})")
    
    # Verify the shape is correct
    expected_shape = [batch_size, context_length]
    actual_shape = list(input_tensor.shape)
    
    if actual_shape == expected_shape:
        print("✅ PASS: Prefill mode input shape is correct!")
        return True
    else:
        print(f"❌ FAIL: Expected {expected_shape}, got {actual_shape}")
        return False

if __name__ == "__main__":
    print("Testing decode mode fix...")
    print("=" * 50)
    
    decode_pass = test_decode_mode_fix()
    print()
    prefill_pass = test_prefill_mode()
    
    print()
    print("=" * 50)
    if decode_pass and prefill_pass:
        print("🎉 All tests passed! The decode mode fix is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")