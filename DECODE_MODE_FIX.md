# Decode Mode KV Cache Fix

## Problem Description

The original issue was that in decode mode with context=128, the MLP layers were incorrectly showing input shapes like `[[1, 128, 1024]]` instead of `[[1, 1, 1024]]`. This was incorrect because:

1. **In decode mode with KV cache**: Only 1 new token is processed at a time
2. **The context_length parameter**: Represents the KV cache size, not the input sequence length
3. **MLP layers**: Should only process the single new token, not all cached tokens

## Before Fix

```
layers.9.mlp.up_proj:
  Batch=1, Context=128:
    Input:  [[1, 128, 1024]]  ❌ INCORRECT
    Output: [[1, 128, 3072]]  ❌ INCORRECT
```

## After Fix

```
layers.9.mlp.up_proj:
  Batch=1, Context=128 (KV cache size, input=1 token):
    Input:  [[1, 1, 1024]]   ✅ CORRECT
    Output: [[1, 1, 3072]]   ✅ CORRECT
```

## Changes Made

### 1. Fixed Input Tensor Creation for Decode Mode

**File**: `simulate.py`, lines 303-322

**Before**:
```python
else:  # decode mode
    # For decode, model still attends to full context but processes 1 new token
    # Create a context with the specified length to simulate KV cache state
    sample_text = "Hello, this is a sample input for tensor shape analysis. " * (context_length // 10 + 1)
    inputs = self.tokenizer(
        sample_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=context_length  # ❌ Wrong: using full context length
    )
    input_ids = inputs['input_ids']
    # Repeat for batch size
    if batch_size > 1:
        input_ids = input_ids.repeat(batch_size, 1)
    return input_ids
```

**After**:
```python
else:  # decode mode
    # For decode mode with KV cache, only process 1 new token
    # The context_length parameter represents the KV cache size, but input is just 1 token
    sample_text = "Hello"
    inputs = self.tokenizer(
        sample_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=1  # ✅ Correct: only 1 token
    )
    input_ids = inputs['input_ids']
    # For decode mode, we only process 1 token regardless of context_length
    # Take only the last token to simulate the new token being processed
    if input_ids.shape[1] > 1:
        input_ids = input_ids[:, -1:]
    # Repeat for batch size
    if batch_size > 1:
        input_ids = input_ids.repeat(batch_size, 1)
    return input_ids
```

### 2. Fixed Fallback Cases

**Lines 328-330 and 337-339**: Updated fallback tensor creation to use `(batch_size, 1)` instead of `(batch_size, context_length)` for decode mode.

### 3. Updated Display Labels

**Line 912**: Added clarification in the output display:
```python
print(f"         Batch={batch_size}, Context={context_length} (KV cache size, input=1 token):")
```

## Verification

The fix has been verified with tests showing:

1. **Decode mode**: Input shape is `[1, 1]` regardless of context_length
2. **Prefill mode**: Input shape is `[1, context_length]` as expected
3. **Specific layer test**: `layers.9.mlp.up_proj` now shows correct shapes

## Key Insight

In transformer models with KV cache:
- **Prefill phase**: Process the entire input sequence
- **Decode phase**: Process only 1 new token while using cached key-value pairs from previous tokens

The context_length in decode mode represents the size of the KV cache (how many previous tokens are cached), not the input sequence length being processed.