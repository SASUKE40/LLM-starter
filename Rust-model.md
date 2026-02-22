# Rust 0.6B Model (Qwen3 Architecture)

## Model Specs

| Param | Value |
|-------|-------|
| $V$ (vocab size) | 151,936 |
| $S$ (max sequence length) | 40,960 |
| $d$ (hidden dimension) | 1,024 |
| $L$ (num layers) | 28 |
| $n_q$ (query heads) | 16 |
| $n_{kv}$ (key/value heads) | 8 (GQA) |
| $h_d$ (head dim) | 128 |
| $d_i$ (intermediate size) | 3,072 |
| position encoding | RoPE (0 learnable params) |
| normalization | RMSNorm (scale only) |
| activation | SwiGLU |
| attention bias | No |
| QK-Norm | Yes |
| weight tying | Yes |

## Formula

$$P = Vd + L \cdot P_{\text{layer}} + d$$

Where per-layer params $P_{\text{layer}}$:

| Component | Formula | Parameters |
|-----------|---------|------------|
| Q projection | $d \times n_q h_d$ | 2,097,152 |
| K projection | $d \times n_{kv} h_d$ | 1,048,576 |
| V projection | $d \times n_{kv} h_d$ | 1,048,576 |
| O projection | $n_q h_d \times d$ | 2,097,152 |
| Q RMSNorm | $n_q \times h_d$ | 2,048 |
| K RMSNorm | $n_{kv} \times h_d$ | 1,024 |
| Gate proj (SwiGLU) | $d \times d_i$ | 3,145,728 |
| Up proj (SwiGLU) | $d \times d_i$ | 3,145,728 |
| Down proj (SwiGLU) | $d_i \times d$ | 3,145,728 |
| Attention RMSNorm | $d$ | 1,024 |
| MLP RMSNorm | $d$ | 1,024 |
| **Per-layer total** | | **15,733,760** |

## Full Model Breakdown

| Component | Formula | Parameters |
|-----------|---------|------------|
| Token embedding | $V \times d$ | 155,582,464 |
| 28 transformer layers | $L \times P_{\text{layer}}$ | 440,545,280 |
| Final RMSNorm | $d$ | 1,024 |
| LM head (tied) | 0 | 0 |
| **Total** | | **596,128,768 ≈ 0.6B** |

Non-embedding parameters: **440,546,304 ≈ 0.44B**

## Key Design Choices

- **GQA**: 2 Q heads share 1 KV head → halves KV cache at inference
- **SwiGLU**: 3 weight matrices (gate/up/down), no biases, better gradient flow than GELU
- **RoPE**: no learnable position params, supports length extrapolation beyond 40K
- **RMSNorm**: scale only (no bias), cheaper than LayerNorm
- **No biases**: removed from all attention & MLP projections
- **Weight tying**: LM head reuses token embedding, saves $Vd$ = 155.6M params
- **QK-Norm**: stabilizes training by normalizing Q and K per head

# Pre-Training Stages

Stage 1: General Knowledge

171GB general text tokens

English focused

https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu


Stage 2: Reasoning and Code

3.2GB rust code tokens

rust_ds = load_dataset(
    "bigcode/the-stack-dedup",
    data_dir="data/rust",
    split="train",
)

https://huggingface.co/datasets/bigcode/the-stack-dedup/viewer

Stage 3: Long Context

Context Extension to 32,768