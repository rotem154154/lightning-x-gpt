# lightning-x-gpt

Modern GPT with Lightning, inspired by x-transformers and recent research.

## Modifications

### Improvements
- Flash-Attention 2
- torch.compile
- Qwen3RMSNorm > RMSNorm > LayerNorm
- ReLU² > GELU
- No bias for QKV projections
- No bias for all linear layers
- Weight tying
- Gated MLP
- Parallel Attention/MLP
- DyT

### No Improvement Yet
- QK Normalization (probable bad implementation, TODO: try different normalization)
- SageAttention (may help inference but destroys gradients)
- Laser
- SimpleRMSNorm
- Sliding window attention
