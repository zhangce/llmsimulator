# Baseline Dependency Analysis

This document describes the baseline dependency analysis implementation at commit ead4f63.

## Current State

The dependency analysis currently uses a simple hierarchical approach that has several known issues:

### Issues Identified

1. **Ignores Intra-layer Dependencies**: 
   - Layer operations depend only on parent containers
   - Missing proper execution flow within transformer layers
   - Example: `transformer.layer.1.attention.q_lin` depends on `transformer.layer.1.attention` but not on layer norm

2. **Incorrect Inter-layer Dependencies**:
   - Layer N operations don't properly depend on Layer N-1 outputs  
   - Missing connections between layers in the transformer stack
   - Example: `transformer.layer.1.sa_layer_norm` should depend on `transformer.layer.0` output operations

3. **No Transformer-Specific Logic**:
   - Treats transformer as generic hierarchical structure
   - Doesn't understand attention → layer norm → FFN flow
   - Missing proper sequential layer connections

## Example Current Output

```
transformer.layer.1.attention.q_lin depends on:
  -> transformer
  -> transformer.layer
  -> transformer.layer.1
  -> transformer.layer.1.attention

transformer.layer.1.sa_layer_norm depends on:
  -> transformer
  -> transformer.layer
  -> transformer.layer.1
```

This baseline serves as the starting point for implementing proper transformer-aware dependency analysis.