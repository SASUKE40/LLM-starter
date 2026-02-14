# GPT-3 (175B) Parameter Calculation

## Model Specs

| Param | Value |
|-------|-------|
| $V$ (vocab size) | 50,257 |
| $S$ (max sequence length) | 2,048 |
| $d$ (hidden dimension) | 12,288 |
| $L$ (num layers) | 96 |
| $n_h$ (num heads) | 96 |

## Formula

GPT-3 uses an **untied** LM head, so the formula is:

$$P = 2Vd + Sd + L(12d^2 + 13d) + 2d$$

Where:

- V = vocab size
- S = max sequence length
- d = hidden dimension (channels)
- L = number of layers

## Per-Layer Breakdown (×96 layers)

Each transformer layer contains:

| Component | Formula | Parameters |
|-----------|---------|------------|
| LayerNorm 1 | $2d$ | 24,576 |
| Q, K, V projections | $3(d^2 + d)$ | 455,196,672 |
| Output projection | $d^2 + d$ | 151,732,224 |
| LayerNorm 2 | $2d$ | 24,576 |
| MLP fc (expand 4×) | $d \times 4d + 4d$ | 603,930,624 |
| MLP proj (compress) | $4d \times d + d$ | 603,967,488 |
| **Per-layer total** | $12d^2 + 13d$ | **1,814,876,160** |

## Full Model Breakdown

| Component | Formula | Parameters |
|-----------|---------|------------|
| Token embedding | $V \times d$ | 617,558,016 |
| Position embedding | $S \times d$ | 25,165,824 |
| 96 transformer layers | $L(12d^2 + 13d)$ | 173,961,510,912 |
| Final LayerNorm | $2d$ | 24,576 |
| LM head (untied) | $V \times d$ | 617,558,016 |
| **Total** | | **175,221,817,344** |

## Notes

- ≈ **175.2B parameters**, matching the commonly cited 175B
- The 96 transformer layers account for **99.3%** of all parameters
- GPT-3 uses an **untied** LM head (separate from token embedding), unlike GPT-2 which uses **weight tying**
- At this scale, the embedding parameters become a rounding error, which is why untying the LM head costs relatively little