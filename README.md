# lightning-x-gpt

Modern GPT with Lightning, inspired by x-transformers and recent research.

## Modifications

### Significantly Improved
- Flash-Attention
- Fully Sharded Data Parallel (FSDP)
- torch.compile integration

### Moderate Improvements
- RMSNorm instead of LayerNorm
- No bias for QKV projections
- ReLU Squared activation instead of GELU
- No bias for all linear layers
- Removed weight tying: +8% training time, +20% params, but faster convergence

### No Improvement Yet
- QK Normalization (probable bad implementation, TODO: try different normalization)
- SageAttention (may help inference but destroys gradients)
