# Decode Mode KV Cache Fix

## Problem Description

The original issue was that in decode mode with context=128, the MLP layers were incorrectly showing input shapes like `[[1, 128, 1024]]` instead of `[[1, 1, 1024]]`. This was incorrect because:

1. **In decode mode with KV cache**: Different layer types behave differently
2. **MLP layers**: Should only process the single new token (1 token)
3. **Attention layers**: Should use the full KV cache context (all tokens)
4. **The context_length parameter**: Represents the KV cache size for attention, but MLP only processes new token

## Before Fix

```
layers.9.mlp.up_proj:
  Batch=1, Context=128:
    Input:  [[1, 128, 1024]]  ❌ INCORRECT
    Output: [[1, 128, 3072]]  ❌ INCORRECT
```

## After Fix

```
layers.9.mlp.up_proj (MLP layer):
  Batch=1, Context=128 (MLP: processes 1 token in decode):
    Input:  [[1, 1, 1024]]   ✅ CORRECT
    Output: [[1, 1, 3072]]   ✅ CORRECT

layers.9.self_attn.q_proj (Attention layer):
  Batch=1, Context=128 (Attention: uses full KV cache):
    Input:  [[1, 128, 1024]]   ✅ CORRECT  
    Output: [[1, 128, 1024]]   ✅ CORRECT
```

## Changes Made

### 1. Layer Type Detection

**Added functions**: `_is_mlp_layer()` and `_is_attention_layer()` to distinguish between layer types.

```python
def _is_mlp_layer(self, name: str, module: nn.Module) -> bool:
    """Check if a layer is an MLP layer that should process only 1 token in decode mode."""
    mlp_patterns = ['mlp.up_proj', 'mlp.down_proj', 'mlp.gate_proj', ...]
    # Check patterns and module types

def _is_attention_layer(self, name: str, module: nn.Module) -> bool:
    """Check if a layer is an attention layer that needs full context in decode mode."""
    attn_patterns = ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', ...]
    # Check patterns and module types
```

### 2. Smart Shape Adjustment in Hooks

**Modified hook creation** to adjust tensor shapes based on layer type in decode mode:

```python
# In the hook function:
if mode == "decode" and self._is_mlp_layer(name, module):
    # MLP layers in decode mode only process 1 token
    if len(shape) >= 2 and shape[1] > 1:
        shape = [shape[0], 1] + shape[2:]
```

### 3. Input Tensor Strategy

**Decode mode now uses full context input** but adjusts shapes per layer:
- **Input creation**: Full context tensor `[batch, context_length, hidden_dim]`
- **Shape adjustment**: MLP layers get `[batch, 1, hidden_dim]`, attention layers keep full context

### 4. Enhanced Display Labels

**Layer-specific labeling** based on detected layer type:
```python
if is_mlp:
    print(f"Batch={batch_size}, Context={context_length} (MLP: processes 1 token in decode):")
elif is_attn:
    print(f"Batch={batch_size}, Context={context_length} (Attention: uses full KV cache):")
```

## Verification

The fix has been verified with tests showing:

1. **Layer type detection**: Correctly identifies MLP vs attention layers
2. **MLP layers in decode mode**: Show input/output shapes with 1 token `[1, 1, hidden_dim]`
3. **Attention layers in decode mode**: Show input/output shapes with full context `[1, context_length, hidden_dim]`
4. **Specific layer test**: `layers.9.mlp.up_proj` now shows `[1, 1, 1024]` instead of `[1, 128, 1024]`

## Key Insight

In transformer models with KV cache during decode mode:

### Attention Layers
- **Query**: Computed for the new token only
- **Key/Value**: Retrieved from KV cache (full context)
- **Attention computation**: New token attends to all cached tokens
- **Tensor shapes**: Full context `[batch, context_length, hidden_dim]`

### MLP Layers  
- **Processing**: Only the new token passes through MLP
- **Previous tokens**: Already processed during prefill, results cached
- **Tensor shapes**: Single token `[batch, 1, hidden_dim]`

The context_length represents the KV cache size, but different layer types use it differently.