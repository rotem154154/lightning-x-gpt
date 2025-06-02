# lightning-x-gpt

Modern GPT with Lightning, inspired by x-transformers and recent research.

## Modifications

### Improvements
- Flash-Attention
- torch.compile
- Qwen3RMSNorm > RMSNorm > LayerNorm
- No bias for QKV projections
- ReLU Squared activation instead of GELU
- No bias for all linear layers
- Weight tying
- Gated MLP
- Parallel Attention/MLP

### No Improvement Yet
- QK Normalization (probable bad implementation, TODO: try different normalization)
- SageAttention (may help inference but destroys gradients)
- Laser
- SimpleRMSNorm
