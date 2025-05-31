# lightning-x-gpt

Modern GPT with Lightning, inspired by x-transformers and recent research.

## Modifications

### Significantly Improved
- Flash-Attention
- torch.compile

### Moderate Improvements
- Qwen3RMSNorm > RMSNorm > LayerNorm
- No bias for QKV projections
- ReLU Squared activation instead of GELU
- No bias for all linear layers
- Weight tying
- Gated MLP

### No Improvement Yet
- QK Normalization (probable bad implementation, TODO: try different normalization)
- SageAttention (may help inference but destroys gradients)
- Laser
- SimpleRMSNorm
