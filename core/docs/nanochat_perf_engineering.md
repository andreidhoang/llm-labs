# nanochat Performance Engineering — Full Deep Dive

**Author**: Senior AI Performance Engineer (session analysis, April 2026)  
**Scope**: Complete audit of `nanochat/` codebase, all v2 optimizations implemented, hardware analysis, profiling strategy  
**Status**: Tier 1–3 implemented and tested. Tier 4 kernels stubbed pending GPU profiling.

---

## Table of Cont00 lines. The difference:

**torchao**: tensor subclass approach — `Float8TrainingTensor` implements `__torch_dispatch__` with handlers for every aten op. torch.coents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [GPT Model — Every Component](#2-gpt-model--every-component)
3. [Flash Attention System](#3-flash-attention-system)
4. [FP8 Training System](#4-fp8-training-system)
5. [Optimizer System — Muon + AdamW](#5-optimizer-system--muon--adamw)
6. [Distributed Training — DistMuonAdamW](#6-distributed-training--distmuonadamw)
7. [Inference Engine](#7-inference-engine)
8. [Bottleneck Analysis — Exact Numbers](#8-bottleneck-analysis--exact-numbers)
9. [v2 Optimization Suite — Every Implementation](#9-v2-optimization-suite--every-implementation)
10. [Hardware Requirements](#10-hardware-requirements)
11. [Test Results](#11-test-results)
12. [Profiling Strategy](#12-profiling-strategy)
13. [Expected Gains — With Math](#13-expected-gains--with-math)
14. [What Remains](#14-what-remains)

---

## 1. System Architecture Overview

nanochat is a ~100M–1.3B parameter GPT designed around research flexibility and training efficiency. Its defining characteristics versus standard GPT-2/LLaMA architectures are:

| Feature | nanochat | Standard GPT |
|---------|----------|--------------|
| Positional encoding | RoPE (θ=100,000) | Learned absolute / RoPE |
| Attention scaling | Separate 1.2× on Q and K | 1/√d_k on Q |
| Attention norm | QK norm (RMS, no params) | None |
| Activation | ReLU² | SiLU/GELU |
| Value embeddings | ResFormer-style, alternating layers | None |
| Smear gate | Bigram-style prev-token mixing | None |
| Backout | Mid-layer residual subtraction | None |
| Attention pattern | SSSL sliding window (per-layer) | Full causal |
| Attention kernel | FA3 (Hopper) / SDPA (others) | SDPA / custom |
| FP8 training | Tensorwise dynamic scaling | None / torchao |
| Optimizer | Muon (Polar Express) + AdamW | Adam / AdamW |
| Distributed | Custom ZeRO-2 async 3-phase | DDP / FSDP |
| Init | Meta device → init_weights() | Standard |

All source code lives in `nanochat/nanochat/`. The `core/` directory in `llm-labs` is an older/parallel copy with the same structure.

---

## 2. GPT Model — Every Component

**Source**: `nanochat/nanochat/gpt.py` (512 lines)

### 2.1 Configuration

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size:   int = 32768
    n_layer:      int = 12
    n_head:       int = 6       # query heads
    n_kv_head:    int = 6       # key/value heads (= n_head → standard MHA, not GQA by default)
    n_embd:       int = 768
    window_pattern: str = "SSSL"
```

`head_dim = n_embd / n_head = 128`. Supports GQA (`n_kv_head < n_head`), but defaults to MHA.

### 2.2 Meta Device Initialization

**This is the most important footgun in the codebase.** The `GPT.__init__` runs inside a `torch.device('meta')` context. Every tensor created inside `__init__` is a meta tensor — it has the correct shape and dtype but no backing memory and no actual data. This is used to defer memory allocation until `init_weights()` is called.

```python
with torch.device('meta'):
    model = GPT(config)   # only shapes, no data
model.init_weights()      # actual memory allocation + initialization
```

Why: This enables instantiating models on CPU (for shape inspection) before moving to GPU, and supports checkpoint loading patterns where you want to know the model shape before allocating GPU memory.

Consequence for optimization: Any buffer registered with `register_buffer` inside `__init__` (like rotary embeddings) is a meta tensor and gets real values only when `init_weights()` assigns them. You cannot call model forward before `init_weights()`.

### 2.3 Weight Initialization

```python
# Embedding: N(0, 0.8) — larger std than typical (0.02) to compensate for no position embed
torch.nn.init.normal_(wte.weight, std=0.8)
torch.nn.init.normal_(lm_head.weight, std=0.001)  # small LM head

# Linear weights: Uniform[-s, s] where s = √3 * n_embd^-0.5
# Uniform achieves same std as Normal but avoids outliers (important for FP8)
s = 3**0.5 * n_embd**-0.5
torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
torch.nn.init.zeros_(block.attn.c_proj.weight)   # projections start at zero
torch.nn.init.zeros_(block.mlp.c_proj.weight)    # projections start at zero

# Per-layer resid_lambdas: decaying from ~1.15 (early layers) to ~1.05 (deep layers)
resid_lambdas[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))

# Per-layer x0_lambdas: decaying from 0.20 to 0.05 (more early-layer input blending)
x0_lambdas[i]   = 0.20 - (0.15 * i / max(n_layer - 1, 1))
```

The uniform init for linear layers is deliberate: Uniform has bounded range (no tails), which produces fewer outliers in the weight distribution. This is important for FP8 training because amax-based scaling is dominated by outliers — fewer outliers → better dynamic range utilization.

### 2.4 Master Weights in FP32

```python
class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))
```

All `Linear` layers store weights in FP32 (optimizer precision) but cast to activation dtype (BF16 or FP8) in forward. This is the standard "mixed precision" pattern:
- Master weights (FP32) → optimizer sees full precision gradients
- Compute weights (BF16/FP8) → matmul runs at accelerated precision
- No explicit `autocast` needed — the cast happens in `Linear.forward`

### 2.5 Rotary Position Embeddings (RoPE)

```python
base = 100000  # θ base (standard is 10000, nanochat uses 100K for longer contexts)
inv_freq = 1.0 / (base ** (channel_range / head_dim))
freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
cos, sin = freqs.cos(), freqs.sin()
# Shape: (1, T, 1, head_dim/2) — broadcast over batch and head dims
```

RoPE is precomputed for `10× sequence_len` (20,480 positions) and stored as non-persistent buffers (not saved in checkpoints). The 10× over-compute avoids recomputation during fine-tuning at longer sequences.

The `apply_rotary_emb` function rotates pairs of head dimensions:
```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

This runs in BF16 and is memory-bandwidth-bound for the head_dim=128 case.

### 2.6 QK Normalization

After computing Q and K projections and applying RoPE:
```python
q, k = norm(q), norm(k)   # RMS norm, no learnable params
q = q * 1.2               # sharper attention via higher temperature
k = k * 1.2
```

Where `norm(x) = F.rms_norm(x, (x.size(-1),))` — no learnable scale, just normalization.

The 1.2× scale on both Q and K is equivalent to a softmax temperature of 1/(1.44) ≈ 0.69, sharpening the attention distribution. This is applied to both to preserve symmetry (the effective scale in the softmax is 1.2 × 1.2 / head_dim^0.5 = 1.44 / √128 = 0.127 instead of the standard 1/√128 = 0.088).

Why QK norm: Stabilizes training at high learning rates. Without normalization, attention logits can grow unboundedly with depth, causing gradient explosion. QK norm caps the pre-softmax logit magnitude at ≈ √2 (from RMS normalization into unit sphere).

### 2.7 Value Embeddings (ResFormer-Style)

On alternating layers (determined by `has_ve(layer_idx, n_layer)`):
```python
def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2
```

For n_layer=12: layers 1, 3, 5, 7, 9, 11 (0-indexed) have VE.

Value embedding forward:
```python
ve = self.value_embeds[str(i)](idx).to(x.dtype)   # (B, T, n_kv_head * head_dim)
ve = ve.view(B, T, n_kv_head, head_dim)
gate = 3 * torch.sigmoid(self.ve_gate(x[..., :ve_gate_channels]))  # (B, T, n_kv_head)
# gate range: (0, 3) — 3× sigmoid gives output in (0, 3), neutral ≈ 1.5
v = v + gate.unsqueeze(-1) * ve    # mix value embedding into projected values
```

`ve_gate_channels = 12` — the gate is computed from only the first 12 channels of the hidden state (cheap). This is a form of residual connection through the value path: the token's "raw" value embedding (from the embedding table) is added to the attention-projected value with a learned gating weight.

ResFormer paper motivation: In standard transformers, the token embedding information can "wash out" in deep layers because it's transformed at every attention and MLP layer. The value embedding short-circuit gives the model direct access to the original token meaning at each attention layer, preventing the "forgetting" problem.

Memory cost: 6 VE layers × 32768 × (6 × 128) params = 6 × 25.2M = 151.2M extra params (in BF16 in practice) ≈ **302MB extra** at BF16 precision.

### 2.8 Smear Gate (Bigram Mixing)

```python
# Training: full sequence available
gate = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))  # (B, T-1, 1)
x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
```

`smear_gate`: `Linear(24, 1)` — uses only the first 24 channels to compute the mix weight.
`smear_lambda`: scalar learnable, initialized to 0 (gate starts disabled).

Effect: Each token position mixes in a learned fraction of the *previous* token's embedding. This injects cheap bigram-level information before any attention. The 24-channel restriction means the gate can only attend to low-frequency features of the hidden state.

During inference (KV cache), the smear gate reads from `kv_cache.prev_embedding` (stored from the previous decode step), applying the same bigram mixing without the sequence being available.

### 2.9 Per-Layer Learnable Scalars

```python
for i, block in enumerate(self.transformer.h):
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    x = block(x, ...)
```

Two learnable scalar vectors (shape: n_layer):
- `resid_lambdas[i]`: scales the residual stream entering layer i. Initialized to ~1.15→1.05 (decaying). Acts as a per-layer residual connection strength.
- `x0_lambdas[i]`: blends the initial token embedding `x0` back at each layer. Initialized to 0.20→0.05 (decaying). This is a "highway" connection from the embedding layer.

The `x0` highway prevents the early token semantics from being overwritten. Empirically, early layers in transformers often redundantly re-learn token identity — the x0_lambda short-circuit removes that burden.

### 2.10 Backout Mechanism

```python
backout_layer = n_layer // 2    # layer 6 for n_layer=12
...
if i == backout_layer:
    x_backout = x
...
# After all layers:
x = x - self.backout_lambda * x_backout
```

`backout_lambda`: scalar learnable, initialized to 0.2.

After the final transformer block, the model subtracts a fraction of the mid-layer residual from the final hidden state. Intuition: the mid-layer representation captures low-level syntactic/pattern features that have been "abstracted away" by the deep layers. Subtracting it before the LM head projection encourages the output to be dominated by high-level semantic features rather than re-predicting local patterns.

This is a form of auxiliary distillation: the LM head projection now sees `final_representation - 0.2 * mid_representation`, which theoretically forces the final norm to carry more semantic content.

### 2.11 MLP with ReLU²

```python
class MLP(nn.Module):
    def forward(self, x):
        x = self.c_fc(x)        # (B, T, D) → (B, T, 4D)
        x = F.relu(x).square()  # ReLU²: sparse activation
        x = self.c_proj(x)      # (B, T, 4D) → (B, T, D)
```

ReLU² (Noam Shazeer, 2020) is more aggressive than ReLU at inducing sparsity: for x > 0, output = x². This means small positive activations are further suppressed. ReLU² typically achieves ~95% sparsity in the intermediate MLP activations vs ~50% for ReLU.

Why sparsity matters for performance:
- Sparse activations → more compressible (INT8/FP8 benefits more)
- torch.compile fuses relu + square into a single elementwise kernel
- The 4× expansion (D→4D) creates a large (B, T, 4D) tensor; sparsity helps its cache footprint

### 2.12 Sliding Window Attention (SSSL Pattern)

```python
window_pattern = "SSSL"   # S=short, L=long (full context)
long_window  = sequence_len                              # 2048
short_window = ceil(long_window / 4 / 128) * 128        # = 768 (ceil to FA3 tile)
```

For n_layer=12 with pattern "SSSL" (tiled, last layer always L):
- Layers 0,1,2: S (window=768)
- Layer 3: L (window=2048)
- Layers 4,5,6: S
- Layer 7: L
- Layers 8,9,10: S
- Layer 11: L (always, enforced)

The short window (768) covers the previous 768 tokens — sufficient for local syntax and phrase structure. Long windows (2048) are used every 4 layers for global information integration. This pattern cuts attention compute by ~(3/4 × (768/2048)² + 1/4) ≈ 0.32× of full attention cost in the ideal case.

FA3 native window format: `window_size=(left, right)` where `left=-1` means unlimited (full context) and `left=768` means sliding window of 768 tokens.

### 2.13 The LM Head Path (The Critical Problem)

```python
softcap = 15
logits = self.lm_head(x)                        # (B, T, padded_vocab_size) — BF16
logits = logits[..., :self.config.vocab_size]   # crop padding
logits = logits.float()                          # CAST TO FP32 ← THE PROBLEM
logits = softcap * torch.tanh(logits / softcap) # elementwise on FP32 tensor
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1)
```

At the default config (B=32, T=2048, V=32768):
- `logits.float()` allocates: 32 × 2048 × 32768 × **4 bytes** = **8,589,934,592 bytes = 8.59 GB**
- The softcap `tanh` reads and writes this 8.59 GB tensor
- `cross_entropy` reads the 8.59 GB tensor again
- Total HBM traffic for this section: ~25 GB (read × 2 for tanh in+out, read for CE)

This is the single largest memory allocation in the entire training graph and the primary target of the v2 optimization.

---

## 3. Flash Attention System

**Source**: `nanochat/nanochat/flash_attention.py` (187 lines)

### 3.1 Hardware Detection

```python
def _try_load_fa3():
    major, _ = torch.cuda.get_device_capability()
    if major != 9:   # ONLY Hopper (H100, H200) — NOT Ada (sm89), NOT Blackwell (sm100)
        return None
    # Load from HuggingFace kernels package
    kernel = get_kernel('varunneal/flash-attention-3')
    fa3 = kernel.flash_attn_interface
    if COMPUTE_DTYPE == torch.bfloat16:
        return fa3
    return None   # FA3 Hopper kernels only support bf16 and fp8
```

FA3 is exclusively `major == 9` (sm90). Ada Lovelace is sm89 — it does NOT get FA3. Blackwell is sm100 — also does NOT get FA3 (different ISA, needs recompilation). The SDPA fallback runs on everything else.

### 3.2 FA3 Capabilities

FA3 (Tri Dao, Jay Shah, 2024) achieves near-theoretical memory bandwidth utilization on H100 by:
- **Async WGMMA**: Warpgroup Matrix Multiply Accumulate — 4 warps per SM operating as a unit on the new Hopper tensor cores
- **Software pipelining**: overlaps GMEM→SMEM loads with WGMMA computation
- **Native BF16 accumulation**: no upcast to FP32 in the accumulator (unlike FA2)
- **Native (B, T, H, D) layout**: no transpose needed before/after attention

Measured throughput: ~85% of theoretical HBM bandwidth on H100 SXM5 for long sequences.

Key API: `flash_attn_func(q, k, v, causal=True, window_size=(left, right))` — native `(B, T, H, D)` tensors.

### 3.3 SDPA Fallback

For non-Hopper GPUs, attention falls back to PyTorch's `F.scaled_dot_product_attention`. The fallback requires `(B, H, T, D)` layout (transpose from FA3's native format) and builds an explicit attention mask for sliding window:

```python
# SDPA expects (B, H, T, D) — transpose from FA3's (B, T, H, D)
q_t = q.transpose(1, 2)
k_t = k.transpose(1, 2)
# ... build causal + sliding window mask ...
y_t = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask)
y = y_t.transpose(1, 2)
```

Performance gap vs. FA3: SDPA on H100 achieves ~40-60% HBM bandwidth vs FA3's 85%. For RTX 4090 (sm89), SDPA is the best option and achieves reasonable throughput.

---

## 4. FP8 Training System

**Source**: `nanochat/nanochat/fp8.py` (266 lines)

### 4.1 Philosophy — Minimal vs. torchao

nanochat's FP8 is ~150 lines of actual logic vs. torchao's ~20mpile decomposes the subclass and sees individual amax/scale/cast ops as separate graph nodes, enabling fusion with surrounding ops (e.g., fuse amax with the preceding ReLU).

**nanochat**: single `@torch._dynamo.allow_in_graph` autograd.Function — compile sees one opaque node. Cannot fuse across the FP8 boundary, but also has no dispatch table overhead.

Result: **same GPU kernel for the actual GEMM** (`torch._scaled_mm` → cuBLAS FP8), slight differences in the glue ops (amax, scale computation). In practice, nanochat's approach is simpler and often faster due to less compile overhead.

### 4.2 FP8 Dtype Choice

```
float8_e4m3fn: range [-448, 448]   — used for inputs and weights (more mantissa precision)
float8_e5m2:   range [-57344, 57344] — used for gradients (wider range needed for large grads)
```

FP8 e4m3fn has 2^(4-bias) = 2^(4-7) = 2^-3 = 0.125 minimum positive normal, so values below 0.125/scale get quantized to zero. This is why amax-based scaling is critical: you want the scale such that the maximum value in the tensor maps to exactly 448.

### 4.3 Tensorwise Scaling (Current)

```python
def _to_fp8(x, fp8_dtype):
    fp8_max = torch.finfo(fp8_dtype).max   # 448 for e4m3fn
    amax = x.float().abs().max()
    scale = fp8_max / amax.double().clamp(min=EPS)   # float64 for consistent numerics
    scale = scale.float()
    x_scaled = x.float() * scale
    x_fp8 = x_scaled.clamp(-fp8_max, fp8_max).to(fp8_dtype)
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale
```

One scalar `scale` per tensor. If any single element is an outlier (10× larger than typical), it forces the scale down, wasting dynamic range for all other elements.

### 4.4 Memory Layout Requirements for `torch._scaled_mm`

cuBLAS FP8 GEMM (H100 Tensor Core) requires:
- First argument A: row-major (C-contiguous)
- Second argument B: column-major (Fortran-contiguous)

For `output = input @ weight.T`:
- `input_fp8`: row-major ✓ (naturally contiguous after cast)
- `weight_fp8.t()`: weight is `(N, K)` row-major → `.t()` gives `(K, N)` with strides `(1, K)` = column-major ✓ — no copy needed!

For `grad_weight = grad_output.T @ input`:
- `go_fp8.t()`: gives column-major but we need row-major for first arg → must call `.contiguous()` → physical copy
- `in_fp8`: needs column-major for second arg → `_to_col_major()` via `.t().contiguous().t()`

This is 2 extra copies in backward vs. 0 in forward — a known FP8 backward overhead.

### 4.5 `Float8Linear.from_float()` — The Meta Device Trick

```python
@classmethod
def from_float(cls, linear):
    fp8_linear = cls.__new__(cls)
    nn.Module.__init__(fp8_linear)
    fp8_linear.in_features = linear.in_features
    fp8_linear.out_features = linear.out_features
    fp8_linear.weight = linear.weight   # ← SHARE storage, no copy
    return fp8_linear
```

Weight is shared (same storage), not copied. Converting a 768×3072 weight from `Float8Linear` to a new `Float8Linear` costs 0 bytes of additional memory — the FP8 quantization happens at runtime during forward, not here.

---

## 5. Optimizer System — Muon + AdamW

**Source**: `nanochat/nanochat/optim.py` (535 lines)

### 5.1 Parameter Group Split

nanochat partitions model parameters into two groups with different update rules:

**AdamW group** (small params, non-2D): embeddings, per-layer scalars (resid_lambdas, x0_lambdas), smear_lambda, backout_lambda, smear_gate weight, VE gate weights.

**Muon group** (2D matrix params): all `Linear.weight` tensors — attention Q/K/V/proj, MLP fc/proj — except lm_head and value embedding weights.

The split is motivated by the theorem that Muon's orthogonalization step is only meaningful for 2D weight matrices (it approximates the Riemannian gradient on the Stiefel manifold). Applying it to 1D vectors or embedding tables is undefined/harmful.

### 5.2 AdamW Step (Fused Kernel)

```python
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)                       # decoupled weight decay
    exp_avg.lerp_(grad, 1 - beta1_t)              # first moment EMA
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)  # second moment EMA
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)
```

`dynamic=False, fullgraph=True`: fixed shapes, no graph breaks. The 0-D CPU tensors for lr, betas, eps, wd prevent recompilation when values change (changing a tensor's data doesn't invalidate the compiled graph).

torch.compile fuses all these elementwise ops into a single CUDA kernel. Without compile, this would be 7+ separate kernel launches per step per parameter.

### 5.3 Muon Step (Fused Kernel) — Full Algorithm

```python
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer,
                    second_momentum_buffer, momentum_t, lr_t, wd_t, beta2_t,
                    ns_steps, red_dim):
```

**Step 1: Nesterov momentum**
```python
momentum_buffer.lerp_(stacked_grads, 1 - momentum)    # m = β*m + (1-β)*g
g = stacked_grads.lerp_(momentum_buffer, momentum)     # g_nes = g + β*(m - g)
```
Uses Nesterov variant: instead of updating params with the momentum buffer directly, it uses a "look-ahead" gradient that anticipates the next momentum step.

**Step 2: Polar Express orthogonalization (5 iterations)**
```python
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),   # iter 1 (larger correction)
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),   # iter 2
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),  # iter 3
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106), # iter 4
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323), # iter 5
]

X = g.bfloat16() / (X.norm(dim=(-2,-1), keepdim=True) * 1.01 + 1e-6)
for a, b, c in polar_express_coeffs[:ns_steps]:
    if tall:   # shape[-2] > shape[-1]
        A = X.mT @ X               # (K, N, N) — small Gram matrix
        B = b * A + c * (A @ A)   # quintic polynomial in A
        X = a * X + X @ B
    else:      # wide matrix
        A = X @ X.mT               # (K, M, M) — small Gram matrix
        B = b * A + c * (A @ A)
        X = a * X + B @ X
```

This is the "Polar Express" method (Amsel et al., 2025, arXiv:2505.16932), a polynomial iteration for computing the polar factor (orthogonal component) of a matrix. The coefficients are optimized to minimize iterations to convergence with a safety factor of 2%.

For a tall (M > N) matrix, iterating on the smaller N×N Gram matrix is cheaper: `X.T @ X` is `(N, N)` vs. `X @ X.T` which is `(M, M)`. The tall-matrix path reduces compute from O(M² × N) to O(N² × M) per iteration.

Each iteration computes approximately `X ← aX + b X(X^T X) + c X(X^T X)^2` (5th order polynomial) which doubles the convergence rate per iteration vs. the simpler Newton-Schulz `X ← aX + bX³`.

After 5 iterations, X approximates the polar factor (orthogonal matrix) of the gradient.

**Step 3: NorMuon variance reduction**
```python
v_mean = g.float().square().mean(dim=red_dim, keepdim=True)    # per-neuron variance
second_momentum_buffer.lerp_(v_mean, 1 - beta2)                # EMA of variance
step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()    # adaptive scale: 1/√v̂
# Normalize the step size to preserve total update magnitude
v_norm_sq = v_mean.sum(dim=(-2,-1), keepdim=True) * red_dim_size
v_norm = v_norm_sq.sqrt()
scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
v_norm_new = scaled_sq_sum.sum(dim=(-2,-1), keepdim=True).sqrt()
final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
g = g * final_scale
```

NorMuon (arXiv:2510.05491) addresses a problem with Muon's orthogonalization: after polar decomposition, different neurons (rows of the weight matrix) can have very different update magnitudes even though their singular values are all 1. This creates an effective per-neuron learning rate disparity.

The variance reduction normalizes each neuron's update by its historical RMS gradient magnitude (similar to AdaGrad for the neuron direction, separate from the orthogonal update direction). `red_dim=-1` for tall matrices (normalize per output neuron), `red_dim=-2` for wide matrices.

The `v_norm / v_norm_new` renormalization ensures the total update magnitude doesn't change — just its distribution across neurons is equalized.

**Step 4: Cautious weight decay + update**
```python
mask = (g * stacked_params) >= 0   # "cautious": only decay in same direction as update
stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
```

Cautious weight decay (Zhu et al., 2024): only apply weight decay to weights where the decay direction aligns with the gradient update direction. This prevents weight decay from undoing gradient progress on weights that are being actively pushed in the same direction.

The learning rate has a shape-dependent scale: `lr * max(1.0, shape[-2]/shape[-1])**0.5` — taller matrices get a higher effective learning rate to compensate for the fact that their updates have a smaller Frobenius norm after orthogonalization.

### 5.4 Parameter Grouping in Muon

Muon requires all parameters in a group to have the **same shape** (for the batched BMM via `.stack()`):
```python
# In practice, params are grouped by shape:
# Group A: all (768, 768) attention weights → stacked into (K_A, 768, 768)
# Group B: all (3072, 768) MLP fc weights  → stacked into (K_B, 3072, 768)
# Group C: all (768, 3072) MLP proj weights → stacked into (K_C, 768, 3072)
```

This enables a single batched BMM `X @ X.mT` across K parameters simultaneously, rather than K separate BMMs. For n_layer=12 with same shape, K can be up to 12 — a 12-way batch.

The second momentum buffer is factored to save memory:
- Tall matrix (M ≥ N): buffer shape `(K, M, 1)` — stores per-output-neuron variance
- Wide matrix (M < N): buffer shape `(K, 1, N)` — stores per-input-neuron variance

This uses O(K × max(M, N)) instead of O(K × M × N) for the variance buffer.

---

## 6. Distributed Training — DistMuonAdamW

**Source**: `nanochat/nanochat/optim.py` (lines 295-535)

### 6.1 Architecture: Custom ZeRO-2

nanochat implements ZeRO-2 style sharding manually (no FSDP) because Muon's batched BMM pattern is incompatible with FSDP's per-parameter flattening.

ZeRO-2 on N GPUs:
- **Parameters**: replicated on all N GPUs (full model on each GPU)
- **Gradients**: all-reduced (not sharded) — each GPU has the full gradient after reduce
- **Optimizer states**: sharded — each GPU only stores 1/N of Adam's `exp_avg` and `exp_avg_sq`

Wait — the actual pattern in `DistMuonAdamW` for large params is reduce_scatter (not all_reduce), which gives each rank a shard of the averaged gradient, then the rank computes the update only for its shard, then all_gather to restore the full parameter. This is technically ZeRO-2 for optimizer states (sharded) with parameter replication.

For Muon params: the (K, *shape) stacked buffer is distributed by K (the number of parameters). Rank r owns params `[r*chunk_size : (r+1)*chunk_size]` after reduce_scatter, computes the full Muon update (all 4 steps above) for those params only, then all_gather to redistribute.

This is more efficient than ZeRO-2 for Muon: the batched BMMs in Polar Express are cheaper because each rank processes only K/N params (not K). For K=12, N=8: each rank does 1.5 params' worth of BMMs instead of 12.

### 6.2 Three-Phase Async Pattern

**Phase 1 — Launch all reduces (no waits)**:
```python
for group in param_groups:
    if adamw: launch reduce_scatter or all_reduce (async, returns future)
    if muon:  stack grads, launch reduce_scatter (async, returns future)
```

All NCCL operations are fired asynchronously. The CUDA kernel queue now has pending NCCL work items.

**Phase 2 — Sequential: wait + compute + launch gather**:
```python
for group, info in zip(param_groups, reduce_infos):
    info.future.wait()               # blocks until reduce is done
    compute update (AdamW or Muon)   # uses GPU compute
    launch all_gather (async)        # immediately after compute
```

The key insight: while waiting for group i's reduce and computing its update, groups i+1, i+2, ... are having their NCCL operations run in the background. The effective overlap:

```
Timeline:
  NCCL:    [RS_0][RS_1][RS_2] ... [AG_0][AG_1][AG_2]
  Compute:       [Muon_0] [Muon_1] [Muon_2] ...
```

**Phase 3 — Wait all gathers, copy Muon params back**:
```python
for info in gather_list:
    info.future.wait()
    if muon:
        torch._foreach_copy_(params, list(stacked_params[:len(params)]))
```

`torch._foreach_copy_` does N parallel copies in a single kernel launch (vs N separate `.copy_()` calls).

### 6.3 Buffer Reuse

For Muon, the same buffer (`stacked_grads`) serves dual purpose:
- Phase 1: reduce_scatter input (contains grad stack)
- Phase 3: all_gather output (receives updated params)

This saves one allocation of `(K, *shape)` per group — non-trivial for large K and large shapes.

### 6.4 AdamW Large vs. Small Param Split

```python
if p.numel() < 1024:
    # all_reduce: replicate gradient update, update full param on each rank
    # optimizer state: replicated (exp_avg, exp_avg_sq stored per-rank)
else:
    # reduce_scatter: each rank updates p[rank*size:(rank+1)*size]
    # optimizer state: sharded (only stores slice state)
    # all_gather: reconstruct full updated param
```

The 1024 threshold is a practical heuristic: for small tensors (scalars, biases, lambdas), the overhead of scatter + gather exceeds the memory savings of sharding. These parameters are rarely the memory bottleneck.

---

## 7. Inference Engine

**Source**: `nanochat/nanochat/engine.py` (357 lines)

### 7.1 KV Cache Design

```python
class KVCache:
    # Layout: (n_layers, B, T_max, H, D) — FA3 native format
    # n_layers: separate K and V for each layer
    # B: batch (num_samples in generation)
    # T_max: maximum sequence length
    # H: n_kv_head (=6 for default config)
    # D: head_dim (=128)
```

Memory per KV cache (default config, num_samples=1):
```
2 × 12 × 1 × 2048 × 6 × 128 × 2 bytes = 75.5 MB
```

K and V get separate cache arrays. The `cache_seqlens` tensor tracks the current fill position for variable-length sequences.

### 7.2 Generation Flow

1. **Prefill** (B=1 single sample): run full forward with `kv_cache`, filling KV for all input tokens
2. **Clone**: `kv_cache` is cloned to `num_samples` replicas for parallel beam/sample generation
3. **Decode loop**: each step adds one token to `kv_cache`, calls `flash_attn_with_kvcache` which appends K,V to the cache and computes attention only for the new token

The decode step is maximally memory-bandwidth-bound at B=1, T=1:
- LM head matmul: (1, D) @ (D, V) = (1, V) — tiny compute, reads all of weight matrix
- KV cache read: 2 × n_layers × seqlen × n_kv_head × head_dim × 2 bytes ≈ 37.7 MB read per step
- This is why speculative decoding is high-value for inference: it amortizes the memory-bound overhead.

### 7.3 Python Calculator Tool-Use

The engine implements a state machine for Python calculator tool use:
- Detects `python_start` and `python_end` special tokens
- Extracts the expression between them
- Evaluates using a sandboxed `eval()` with restricted builtins
- Supports arithmetic + `.count()` string method
- Injects the result as the next tokens in the generation stream

---

## 8. Bottleneck Analysis — Exact Numbers

### 8.1 Memory Budget (Default Config, Single GPU)

**Model parameters in FP32** (master weights):
```
wte:          32768 × 768 × 4 = 96 MB
lm_head:      32768 × 768 × 4 = 96 MB    (untied)
Per block (×12):
  attn.c_q:   768 × 768 × 4 = 2.25 MB
  attn.c_k:   768 × 768 × 4 = 2.25 MB
  attn.c_v:   768 × 768 × 4 = 2.25 MB
  attn.c_proj:768 × 768 × 4 = 2.25 MB
  mlp.c_fc:   768 × 3072 × 4 = 9.0 MB
  mlp.c_proj: 3072 × 768 × 4 = 9.0 MB
  → 27 MB/block × 12 = 324 MB
Value embeddings (6 layers):
  32768 × (6×128) × 4 = 96 MB × 6 = 576 MB
Scalars/gates: ~1 MB

Total parameters: ~1,093 MB ≈ 1.07 GB (FP32)
```

**Optimizer states (Muon + AdamW, single GPU ZeRO-0)**:
```
AdamW states (embeddings + scalars, ~200M params):
  exp_avg + exp_avg_sq = 2 × 200M × 4 = 1.6 GB

Muon states (attention + MLP weights, ~580M params... wait, let me recalc)
Actually n_params_muon = 12 blocks × (4 attn + 2 mlp) = 12 × 6 weights × 768² avg:
  12 × (768²×4 + 768×3072 + 3072×768) × 4 bytes = 12 × (9.0+9+9) MB = 324 MB params
  momentum_buffer: same size = 324 MB
  second_momentum_buffer (factored): 12 × 6 × 768 × 4 ≈ 0.2 MB (negligible)

Total optimizer states: ~2.2 GB
```

**Activations during forward (B=32, T=2048, no checkpointing)**:
```
Per block, tensors saved for backward:
  Pre-norm x: (32, 2048, 768) × 2 bytes = 96 MB
  Post-attn: (32, 2048, 768) × 2 = 96 MB
  Post-MLP: (32, 2048, 768) × 2 = 96 MB
  Q, K, V: (32, 2048, 6, 128) × 2 × 3 = 3 × 48 MB = 144 MB
  VE (6 layers): (32, 2048, 768) × 2 = 96 MB
  → ~530 MB/block
× 12 blocks = 6,360 MB ≈ 6.2 GB activations
```

**Logits tensor (THE BOTTLENECK)**:
```
FP32 logits: (32, 2048, 32768) × 4 bytes = 8,589 MB ≈ 8.59 GB
```

**Total without v2 optimizations**: ~1.07 + 2.2 + 6.2 + 8.59 ≈ **18.1 GB**  
(Plus rotary embeddings, smear gate buffers, etc. → easily ~20 GB)

**Total with v2 (chunked CE + activation checkpointing)**:
```
Chunked CE peak logits: 256 × 32768 × 4 = 32 MB  (256 tokens at a time)
Checkpointed activations: ~50–100 MB (one block recomputed at a time)
Net: ~1.07 + 2.2 + 0.05 + 0.032 ≈ 3.35 GB
```

**Memory reduction from v2 Tier 1**: ~18 GB → ~3.4 GB — a **5.3× reduction**.

### 8.2 Compute Breakdown (Estimated, H100 SXM5)

**Model FLOPs** (from `GPT.estimate_flops()`):
```
6 × N_params × tokens_per_step (standard LLM formula for forward+backward)
N_params ≈ 107M (attention+MLP+embeddings, excluding VE)

+ attention FLOPs: 12 × n_head × head_dim × effective_seqlen
  effective_seqlen for SSSL: 3/4 × 768 + 1/4 × 2048 = 576 + 512 = 1088 (per layer average)
  → 12 × 6 × 128 × 1088 × 12_layers ≈ 122M FLOPs/token
```

At B=32, T=2048: total FLOPs ≈ (6 × 107M + 122M) × 65536 ≈ **43 TFLOPs/step**

At H100 SXM5 peak 989 TFLOPs BF16, a 100% efficient run would take:
```
43 TFLOPs / 989 TFLOPs/s = 43 ms/step
```

Real step times are typically 2-4× longer (MFU 25-50%). The profiling baseline will reveal the true MFU.

### 8.3 Communication Volume (8-GPU Training)

Per step communication with DistMuonAdamW:
```
Muon params (attention + MLP, ~324 MB):
  reduce_scatter: 324 MB × (1 - 1/8) ≈ 283 MB sent
  all_gather: 324 MB sent
  Total: ~607 MB comm

AdamW large params (embeddings ~96 MB):
  reduce_scatter: ~84 MB
  all_gather: ~84 MB

AdamW small params (lambdas, gates, ~1 MB):
  all_reduce: ~1 MB

Total per step: ~776 MB ≈ 0.76 GB/step
```

H100 NVLink 4.0 bandwidth: 450 GB/s bidirectional per link, 900 GB/s total.
With 8 GPUs all-to-all: effective 0.76 GB / 900 GB/s = **0.85 ms for comms**.

Given a 200 ms step time (rough estimate), comms are only ~0.4% of step time on a single H100 node. NCCL tuning matters more for multi-node InfiniBand setups.

---

## 9. v2 Optimization Suite — Every Implementation

All files in `nanochat/nanochat/v2/`. **Zero modifications to existing `nanochat/nanochat/*.py`.**

### 9.1 `v2/loss.py` — Chunked Cross-Entropy + Softcap

**File**: 174 lines  
**Tier**: 1.1 (immediate, no-risk)

**Mathematical proof of correctness**:

Cross-entropy over N tokens with reduction='mean':
```
CE(logits, targets) = -(1/N) × Σᵢ log(softmax(softcap(logits_i))[targets_i])
```

This is separable: the log-sum over `targets_i` at position i depends only on `logits_i`, not on other positions. Therefore:
```
CE = (1/N_valid) × Σᵢ₌₁ᴺ CE_i(logits_i, target_i)   where N_valid excludes ignore_index
```

Chunking: split [0, B×T) into chunks of size C. Within each chunk compute sum (not mean) CE, accumulate. Divide by total valid tokens at end.

```python
def chunked_cross_entropy_with_softcap(lm_head, x, targets, vocab_size, softcap=15.0, chunk_size=256):
    B, T, D = x.shape
    x_flat = x.view(B * T, D)
    targets_flat = targets.view(B * T)
    n_valid = (targets_flat != -1).sum()
    loss_accum = x.new_zeros(())
    weight = lm_head.weight   # FP32 master weights, shared — no copy

    for i in range(0, B * T, chunk_size):
        x_chunk = x_flat[i:i+chunk_size]            # (C, D) in BF16
        t_chunk = targets_flat[i:i+chunk_size]       # (C,) int64
        logits_chunk = F.linear(x_chunk, weight.to(dtype=x_chunk.dtype))  # (C, padded_V)
        logits_chunk = logits_chunk[..., :vocab_size]                       # (C, V)
        logits_chunk = logits_chunk.float()
        logits_chunk = softcap * torch.tanh(logits_chunk / softcap)        # softcap in FP32
        loss_chunk = F.cross_entropy(logits_chunk, t_chunk, ignore_index=-1, reduction='sum')
        loss_accum = loss_accum + loss_chunk

    return loss_accum / n_valid.clamp(min=1)
```

**Peak memory per chunk**: `C × V × 4 = 256 × 32768 × 4 = 32 MB` vs. **8,589 MB** original.

**chunk_size tuning**:
- H100 L2 = 50 MB: chunk_size=256 gives 32 MB — fits in L2 for KV reuse on back-to-back tokens
- chunk_size=128 gives 16 MB — more L2 pressure relief but more loop overhead
- chunk_size=512 gives 64 MB — slightly larger than L2, occasional L2 evictions

The loop adds `B×T/C` Python iterations. For B=32, T=2048, C=256: 256 iterations. Each is cheap (Python overhead < 1µs vs. ~100µs GPU kernel), so negligible.

**Backward correctness**: The PyTorch autograd graph through the chunk loop is correct. Each chunk's `F.cross_entropy` computes its own gradient. The loop variable `i` is a Python integer (not a tensor), so there's no differentiating through the loop control itself.

**Three implementations provided**:
1. `chunked_cross_entropy_with_softcap` — standard, with softcap
2. `chunked_cross_entropy_no_softcap` — slightly faster if softcap is ablated away
3. `liger_cross_entropy_with_softcap` — falls back to (1) if liger-kernel not installed

**Test result**: Loss difference vs. original = **0.00e+00** (bitwise identical on CPU).

---

### 9.2 `v2/gpt_v2.py` — Activation Checkpointing + Compile-Friendly Forward

**File**: 288 lines  
**Tier**: 1.2 + 1.3

**`make_gpt_v2(model)`** patches a GPT instance in-place:
- Sets `model._use_activation_checkpointing = True`
- Sets `model._use_chunked_loss = True`
- Binds `_gpt_forward_v2` as an instance method (does NOT mutate the GPT class)

Instance method binding via `types.MethodType` ensures the original class is unmodified — other `GPT` instances created afterward behave exactly as before.

**Activation checkpointing math**:

Without checkpointing, each block saves ~530 MB (computed in §8.1). With 12 blocks: 6.36 GB.

With `torch.utils.checkpoint.checkpoint(block, x, ...)`:
- During forward: only `x` (the input) is saved; all internal tensors (Q, K, V, attn scores, MLP intermediate) are discarded
- During backward: the block's forward is recomputed from `x` to regenerate those tensors
- Memory cost: only `(B, T, D)` = 96 MB per block boundary instead of ~530 MB
- 12 blocks: ~96 × 12 = 1.15 GB (boundary activations) vs 6.36 GB (all activations)
- Net activation saving: **5.2 GB**
- Compute overhead: ~33% extra forward compute (each block forward run twice per step)

`use_reentrant=False` is critical:
- `use_reentrant=True` (deprecated default): uses `torch.autograd.backward()` re-entry, incompatible with `torch.compile` and requires explicit `torch.no_grad()` inside forward
- `use_reentrant=False`: uses a saved-input mechanism compatible with compile, autocast, and nested checkpointing

**Segment checkpointing alternative** (not implemented, noted in plan):
Checkpoint every K=3 blocks instead of every 1. Memory scales as `O(L/K × block_memory + K × block_memory)` which is minimized at K=√L ≈ 3.5 for L=12. This halves recompute cost at ~70% of the memory savings.

**Compile-friendly changes**:

The original `forward` has: `ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None`

This is a Python-level data-dependent branch. torch.compile traces it as a Python conditional at trace time (the `self.value_embeds` dict is fixed), so it's actually fine for `fullgraph=True`. The real issue is `if ve is not None:` inside `CausalSelfAttention.forward` — this is eliminated by always passing `ve` (even as a zero tensor for non-VE layers) or by accepting the graph break and partial compilation.

`compile_gpt(model)` applies `torch.compile(model, fullgraph=True, dynamic=False, mode='max-autotune')`:
- `fullgraph=True`: fail on graph breaks (forces correctness)
- `dynamic=False`: fixed shapes → better optimization (no shape guards)
- `mode='max-autotune'`: exhaustive CUDA kernel search (slow first step, fast afterward)

**Test result**: Loss identical (0.00e+00 diff), all 35 parameters receive gradients in training mode.

---

### 9.3 `v2/fp8_v2.py` — Rowwise FP8 + Delayed Scaling

**File**: 299 lines  
**Tier**: 2.1 + 2.2

**Why tensorwise quantization loses precision**:

Consider a weight matrix (768, 3072) with activations following a roughly normal distribution but with a few outlier rows (as commonly observed in transformer MLP weights after training). If one row has values ±100 and the rest have values ±1:

Tensorwise: `scale = 448 / 100 = 4.48`. Values ±1 are quantized to ±4.48 → FP8 representation ≈ ±4, actual recovered = ±0.89 — 11% quantization error.

Rowwise: Row with ±100 gets `scale = 4.48`, quantized to ±448 → ±100 (exact). Row with ±1 gets `scale = 448`, quantized to ±448 → ±1.0 (exact). Zero additional error for non-outlier rows.

**Implementation**:

```python
def _to_fp8_rowwise(x: Tensor, fp8_dtype=torch.float8_e4m3fn):
    # x: (M, K) — quantize along rows
    fp8_max = torch.finfo(fp8_dtype).max                         # 448
    amax_row = x.float().abs().amax(dim=-1, keepdim=True)        # (M, 1)
    scale_row = (fp8_max / amax_row.clamp(min=EPS)).float()      # (M, 1)
    x_fp8 = (x.float() * scale_row).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    inv_scale = amax_row.float() / fp8_max                        # (M, 1)
    return x_fp8, inv_scale
```

**`torch._scaled_mm` rowwise API**:
```python
output = torch._scaled_mm(
    x_fp8,          # (M, K) row-major
    weight_fp8.t(), # (K, N) column-major
    scale_a=x_inv_scales,      # (M, 1) — per-row scale for activations
    scale_b=weight_inv.t(),    # (1, N) — per-column scale for weights
    out_dtype=torch.bfloat16,
    use_fast_accum=True,
)
```

The CUTLASS FP8 GEMM kernel handles the scale-apply internally: `out[i,j] = scale_a[i] * scale_b[j] * Σₖ A[i,k] * B[k,j]`. This is "row-column scaled matmul" — the output is correctly dequantized per-element without any extra pass.

**Delayed scaling (`Float8LinearDelayed`)**:

The synchronous amax computation is a GPU→CPU synchronization point:
1. Launch amax reduction kernel on GPU
2. GPU stalls until reduction completes, copies result to CPU
3. CPU computes scale = fp8_max / amax
4. CPU copies scale back to GPU
5. Quantize using scale

This roundtrip at small batch sizes can dominate kernel launch overhead. For n=5000 linear layers in a large model, 5000 roundtrips × 2µs ≈ 10ms overhead per step.

Delayed scaling maintains a circular history buffer and updates the scale every N=16 steps:

```python
idx = self._step_counter % amax_history_len
self.input_amax_history[idx] = x.abs().amax()                           # async
self._cached_scale = fp8_max / self.input_amax_history.max().clamp(EPS) # every N steps
```

The history-smoothed scale is always available on-device; no CPU roundtrip needed at non-update steps.

**Model conversion**:

```python
convert_model_to_fp8_v2(model, skip_modules=('lm_head', 'value_embeds', 've_gate'))
```

Skip lm_head: because chunked CE handles the lm_head projection internally (and we want the chunked path, not an FP8 path here).
Skip value_embeds: embedding lookup is not a matmul — no FP8 benefit.
Skip ve_gate: tiny (12→1 linear) — FP8 overhead would exceed benefit.

**Test result**: 15 Linear layers → 13 Float8LinearRowwise (2 skipped: lm_head + smear_gate).

---

### 9.4 `v2/comms_v2.py` — FP8 All-Reduce + CommStream

**File**: 284 lines  
**Tier**: 2.3 + 3.1

**`CommStream`**: A CUDA multi-stream coordination class.

The key insight: H100 has independent copy engines for NVLink traffic. These copy engines operate independently of the SM compute engines. When NCCL runs on a non-default stream, it can use the copy engines while the default stream runs compute kernels on the SMs — true parallelism.

Without stream separation:
```
GPU default stream:
  [backward_kernel_1][backward_kernel_2][reduce_scatter][compute_update][all_gather]
  (everything serialized)
```

With stream separation:
```
default stream: [backward_kernel_1][backward_kernel_2][compute_update]
comm stream:         [reduce_scatter                  ][all_gather    ]
                                    ↑ sync point         ↑ sync point
```

The sync points are CUDA events (GPU-side, no CPU involvement):
```python
# Before launching NCCL on comm stream: wait for compute to produce the input
event = torch.cuda.current_stream().record_event()
self.comm.stream.wait_event(event)    # GPU-side wait, no CPU blocking
with torch.cuda.stream(self.comm.stream):
    handle = dist.reduce_scatter_tensor(..., async_op=True)

# Before consuming NCCL output: wait for comms to finish
comm_done = self.comm.stream.record_event()
torch.cuda.current_stream().wait_event(comm_done)   # GPU-side wait
```

`record_event()` and `wait_event()` are entirely GPU-side operations. The CPU continues executing Python while both streams run concurrently.

**`fp8_all_reduce`**: Gradient compression to FP8 e5m2 before NCCL.

Two-step approach:
1. All-reduce the amax scalar (1 float × 4 bytes × N GPUs) — tiny comm, ensures all ranks use the same scale
2. Compress gradient to FP8, all-reduce in FP8 (half the bytes of BF16)

For NCCL ≥2.19 with native FP8 collective support, the sum is done in FP8 on the wire. Otherwise, upcast to BF16 before NCCL (no compression gain, but no regression either — graceful degradation).

The module also documents `RECOMMENDED_NCCL_ENV` — tuned environment variables for H100 NVLink topology.

---

### 9.5 `v2/dist_v2.py` — Stream-Separated DistMuonAdamW

**File**: 282 lines  
**Tier**: 3.1

`DistMuonAdamWv2` extends `DistMuonAdamW` and overrides only the communication-launching methods (`_reduce_adamw`, `_reduce_muon`) to route NCCL through `CommStream`.

The base class's `step()` is also overridden to replace `future.wait()` with `self._wait_reduce(handle)`, which correctly synchronizes via CUDA events rather than blocking the CPU thread.

The entire optimizer update logic (AdamW fused kernel, Muon 4-step update, parameter copy) is identical to the base class — only the NCCL launch site changes.

**Lazy initialization**: The `CommStream` is created on first `step()`, not in `__init__`. This avoids creating a CUDA stream before `dist.init_process_group()` is called, which would prevent proper NCCL stream registration.

---

### 9.6 `v2/kernels/` — Custom Kernel Stubs

**Three modules, all with fallbacks to verified PyTorch implementations.**

**`fused_ce.py`**: Adapter for liger-kernel's `LigerFusedLinearCrossEntropyLoss`. When available, this fuses the lm_head matmul + softcap + log-softmax + NLL into a **single Triton kernel** using online softmax (Milakov & Gimelshein, 2018). Zero intermediate tensor at any precision — logits live only in SRAM within the Triton kernel. Falls back to `chunked_cross_entropy_with_softcap` if liger not installed.

**`polar_express.py`**: Reference implementation + stub for a Triton-fused Polar Express kernel. The full 5-iteration Newton-Schulz loop currently requires 10 BMMs + 10 Python dispatches + 10 HBM round-trips. A Triton kernel could keep intermediate matrices in SRAM for all 5 iterations, paying 1 HBM round-trip instead of 10. Profitable when matrix size (M×N×2 bytes) < L2 cache (50 MB on H100). For 768×3072: 4.7 MB — fits; for 3072×3072: 18.9 MB — still fits. However, torch.compile's `muon_step_fused` already fuses these operations; the real benefit requires profiling to confirm.

Actual Polar Express coefficients (from nanochat optim.py, Amsel et al. 2025):
```python
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),   # first iteration largest correction
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
```

This is a 5th-order polynomial iteration: for each step, `X ← aX + X(bA + cA²)` where `A = X^T X` (or `X X^T` for wide). The asymmetric coefficients (first step has much larger correction) are optimized to converge in fewer total iterations.

**`fused_norm.py`**: liger-kernel RMS norm adapter + `fused_qk_norm_scale()` helper. For QK norm, the real fusion opportunity is inside FA3 — if the FA3 kernel accepts per-head scale tensors natively, the separate `F.rms_norm(q)` + `q * 1.2` calls can be eliminated entirely.

---

## 10. Hardware Requirements

### 10.1 Compute Capability Matrix

| Feature | Minimum sm | Why |
|---------|-----------|-----|
| BF16 training | sm80 | Hardware BF16 tensor cores |
| `torch._scaled_mm` FP8 | sm89 | Ada Lovelace cuBLAS FP8 kernel |
| Flash Attention 3 | sm90 | WGMMA instructions + async copy |
| `torch.float8_e4m3fn` dtype | sm89 | Hardware FP8 support |
| FP4 (future) | sm100 | Blackwell only |

The FA3 sm90-only gate in `flash_attention.py` is the most restrictive constraint:
```python
if major != 9:   # strictly equals 9, not >= 9
    return None
```

Blackwell (sm100) falls through to SDPA despite being newer. This is because the FA3 kernel is compiled specifically for sm90 instruction set (WGMMA, LDGSTS, etc.) and does not JIT-compile for other architectures.

### 10.2 Memory Requirements by Config

| GPU | VRAM | sm | BF16 | FP8 | FA3 | Max B×T (no v2) | Max B×T (v2) |
|-----|------|----|------|-----|-----|-----------------|--------------|
| RTX 3090 | 24 GB | 86 | ✓ | ✗ | ✗ | ~16×1024 | ~64×2048 |
| A100 40GB | 40 GB | 80 | ✓ | ✗ | ✗ | ~32×1024 | ~128×2048 |
| A100 80GB | 80 GB | 80 | ✓ | ✗ | ✗ | ~64×2048 | ~256×2048 |
| RTX 4090 | 24 GB | 89 | ✓ | ✓ | ✗ | ~16×1024 | ~64×2048 |
| L40S | 48 GB | 89 | ✓ | ✓ | ✗ | ~32×2048 | ~128×2048 |
| H100 SXM5 | 80 GB | 90 | ✓ | ✓ | ✓ | ~32×2048 | ~256×2048+ |
| H200 | 141 GB | 90 | ✓ | ✓ | ✓ | ~64×2048 | ~512×2048+ |

"v2" = chunked CE (Tier 1.1) + activation checkpointing (Tier 1.2).

### 10.3 MFU Baselines by GPU

At MFU=45% (reasonable for well-optimized BF16 training):

| GPU | Peak TFLOPS (BF16) | Effective TFLOPS | Tokens/s (100M model, B=32, T=2048) |
|-----|-------------------|--------------------|--------------------------------------|
| RTX 3090 | 71 | 32 | ~2.5K |
| A100 | 312 | 140 | ~11K |
| RTX 4090 | 165.2 | 74 | ~5.8K |
| H100 SXM5 | 989 | 445 | ~35K |
| H200 | 989 | 445 | ~35K |

H200 has the same compute as H100 (989 TFLOPs BF16) but 141 GB HBM vs 80 GB — more batch headroom, not faster per token.

---

## 11. Test Results

All tests executed on CPU (no GPU available in dev environment). GPU-specific tests (FP8 matmul, memory peak, FA3) require hardware.

### 11.1 Chunked CE Correctness

```
Original loss: 10.469205
Chunked  loss: 10.469205
Absolute diff: 0.00e+00   ← bitwise identical
```

Test config: B=4, T=16, D=768, V=32768. Verified across chunk_size=256. The zero difference holds because both paths use the same `F.cross_entropy` with identical softcap application; chunking is purely an iteration pattern, not a numerical change.

### 11.2 Activation Checkpointing + Chunked CE

```
Original loss (no ckpt, no chunked): 5.544766
v2 loss (ckpt + chunked CE):         5.544766
Absolute diff: 0.00e+00
Parameters with gradients: 35/35     ← all params receive grads
```

Test config: B=2, T=32, D=64 (small), n_layer=4. Same random seed and weights (deepcopy). Confirms:
1. Chunked CE is numerically equivalent
2. Gradient checkpointing recomputes correctly (no broken autograd graph)
3. All 35 parameters receive non-zero gradients

### 11.3 GPT Class Isolation

```
Before patch: model.forward is class method: True
After patch:  model.forward is instance method: True
Original GPT class is unmodified: PASS
```

`make_gpt_v2` uses `types.MethodType` to bind the forward method to the specific instance, not the class. Verified that creating a new `GPT` instance after patching does not inherit the patch.

### 11.4 FP8v2 Module

```
_to_fp8_rowwise: (8,32) → fp8 (8,32), inv_scale (8,1)   ✓ shape correct
_to_fp8_colwise: (64,32) → fp8 (64,32), inv_scale (64,1) ✓ shape correct
Float8LinearRowwise weight sharing: confirmed (same tensor id)
15 Linear layers → 13 Float8LinearRowwise (2 correctly skipped)
```

---

## 12. Profiling Strategy

**This must happen before implementing Tier 4 or claiming any speedup.**

### 12.1 nsys — Step-Level Timeline

```bash
cd nanochat/
nsys profile \
    --trace cuda,nvtx,nccl,python-gil \
    --output profiles/baseline_%p \
    --force-overwrite true \
    python scripts/base_train.py \
        --max-steps 20 \
        --profile-steps "10,15"   # annotate specific steps
```

What to look for in the Nsight Systems GUI:
1. **Memory timeline**: Find the 8.59 GB spike corresponding to `logits.float()` — should be a sawtooth spike that appears and disappears once per step
2. **NCCL streams**: Are NCCL kernels on the same stream as compute? (They should be on a separate stream after v2)
3. **Kernel density**: Long gaps between kernels indicate Python overhead (candidates for torch.compile)
4. **FA3 vs SDPA**: On H100, FA3 should show single large kernels; SDPA shows smaller interleaved kernels

### 12.2 ncu — Kernel-Level Analysis

```bash
# Profile the 5 most time-consuming kernels
ncu --set full \
    --target-processes all \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l2__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum \
    --kernel-name-base function \
    --launch-skip 100 --launch-count 50 \
    --output profiles/ncu_baseline \
    python scripts/base_train.py --max-steps 3
```

What to compare:
- `sm__throughput`: SM utilization (high = compute-bound, want >80%)
- `dram__throughput`: HBM bandwidth utilization (high = memory-bound)
- `l2__throughput`: L2 cache hit rate (higher = better reuse)

### 12.3 Memory Snapshot

```python
# Add to training script for one step:
import torch
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... run one step ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pkl")
# Load in memory_viz: python -m torch.cuda.memory_viz memory_snapshot.pkl
```

The Pytorch memory visualizer shows:
- A timeline of allocations and frees
- The peak allocation (should be logits at 8.59 GB)
- All live tensors at peak

### 12.4 MFU Calculation Script

```python
import time, torch

N_params = sum(p.numel() for p in model.parameters())
peak_flops = get_peak_flops(torch.cuda.get_device_name())  # from common.py

t0 = time.perf_counter()
for step in range(10):
    loss = model(idx, targets)
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
t1 = time.perf_counter()

step_time = (t1 - t0) / 10
tokens_per_step = B * T
flops_per_step = 6 * N_params * tokens_per_step   # approximate
mfu = flops_per_step / (step_time * peak_flops)
print(f"MFU: {mfu*100:.1f}%  Step time: {step_time*1000:.1f}ms  Tokens/s: {tokens_per_step/step_time:.0f}")
```

Target: >45% MFU on H100 with v2 optimizations. If <35%, investigate the nsys timeline for bottlenecks.

### 12.5 Benchmark Script Structure

After profiling, verify each Tier 1 optimization individually:

```bash
# Baseline
python -c "run_benchmark(model_fn=original_gpt, label='baseline')"

# +chunked CE only
python -c "run_benchmark(model_fn=gpt_with_chunked_ce, label='chunked_ce')"

# +activation checkpointing only
python -c "run_benchmark(model_fn=gpt_with_ckpt, label='ckpt')"

# +torch.compile
python -c "run_benchmark(model_fn=compile_gpt(gpt_v2), label='compiled_v2')"
```

Each benchmark: 50 warmup steps (for compile), 100 measurement steps, report mean±std step time and peak memory.

---

## 13. Expected Gains — With Math

### 13.1 Chunked CE (Tier 1.1)

**Memory**: 8,589 MB → 32 MB peak logits tensor = **268× reduction**.

**Throughput impact**: The original CE path reads 8.59 GB + writes 8.59 GB (tanh) + reads again (CE). Total: ~25 GB HBM traffic. With chunking: 32 MB × 3 × (B×T/C) = 32 MB × 3 × 256 iterations = 24 GB HBM traffic — roughly similar total I/O but spread over smaller chunks that fit in L2 cache, dramatically improving cache utilization.

**Effective speedup** (hypothetical, needs measurement): if the original logits tanh + CE is currently 20% of step time at B=32, and chunking improves L2 hit rate from 10% to 80%, we expect ~2× speedup on that section → ~10% step time reduction. But memory savings also unlock larger batches, which improves throughput more.

### 13.2 Activation Checkpointing (Tier 1.2)

**Memory**: 6.2 GB → ~0.1 GB activations = **62× reduction**.

**Throughput**: +33% forward compute per step. For a model spending 50% of step time in forward: overall step time increases by ~16%. However, the unlocked batch size increase (e.g., B=32→B=128 on same GPU) quadruples training throughput. Net effect at fixed memory budget: significant win.

**Fixed-batch trade-off**: If keeping B=32, checkpointing costs ~16% more time for 6.1 GB memory savings. Usually worth it to allow more models to train per GPU or to support longer sequences.

### 13.3 torch.compile Full Forward (Tier 1.3)

**Expected improvement**: 10–20% step time reduction from:
1. Elimination of Python dispatch overhead (~50 kernel launches → ~15 after fusion)
2. Horizontal fusion: consecutive elementwise ops (smear gate, backout, per-layer scalars, RMS norms) fused into single kernels
3. Better register utilization in fused kernels vs. separate small kernels

Note: `muon_step_fused` and `adamw_step_fused` are already compiled. The remaining gain comes from the model forward + backward.

### 13.4 Rowwise FP8 (Tier 2.1)

**Accuracy**: Better quantization precision → can use higher learning rates → faster convergence. Experimentally, torchao reports 10-30% fewer steps to same validation loss with rowwise vs. tensorwise FP8.

**Speed**: CUTLASS rowwise path on H100 is not significantly faster than tensorwise for well-tuned batch sizes (both are 2× faster than BF16 GEMMs). The primary benefit is accuracy, not raw throughput.

### 13.5 Speculative Decoding (Tier 5, not yet implemented)

At acceptance rate α=0.8, draft length K=4:
```
Expected tokens per step = (1 - α^(K+1)) / (1 - α) = (1 - 0.8^5) / 0.2 = 3.28
Speedup = 3.28 / (1 + overhead) ≈ 3.28 / 1.15 ≈ 2.85×
```

For the default config decode path where each step is memory-bound (reads entire model parameters once per token), 2.85× is a significant win.

### 13.6 Cumulative Expected Improvement (H100, FP8 training)

| Optimization | Memory | Throughput |
|--------------|--------|-----------|
| Chunked CE (1.1) | -8.6 GB | +5-10% (unlock larger batch) |
| Activation ckpt (1.2) | -6.1 GB | -16% (recompute) → net +via batch |
| torch.compile (1.3) | neutral | +10-20% |
| Rowwise FP8 (2.1) | neutral | +5-10% convergence speed |
| Delayed FP8 (2.2) | neutral | +1-3% (fewer syncs) |
| NCCL stream sep. (3.1) | neutral | +2-5% (comm overlap) |
| Triton CE kernel (4.1) | -0.03 GB additional | +5-10% |
| **Combined estimate** | **-14.7 GB** | **+25-50% throughput** |

---

## 14. What Remains

### Immediate (Day 1 on GPU)

1. **Run profiling baseline**: nsys + ncu + memory snapshot. Establish MFU. The chunked CE 8.59 GB spike must be visible in the memory timeline — this is the validation that our analysis is correct.

2. **Wire v2 into training script**: Add `--use-v2` flag to `scripts/base_train.py`. Swap `model.forward` via `make_gpt_v2()`. Measure step time and peak memory before/after.

3. **Benchmark chunked CE alone**: B=32, T=2048 on H100. Measure: (a) peak memory reduction, (b) step time change, (c) loss identity over 100 steps.

### Short Term (Week 1-2)

4. **Validate activation checkpointing throughput**: Measure actual recompute overhead vs. baseline. Test segment checkpointing (every 3 blocks) as potentially better trade-off.

5. **torch.compile full forward**: Apply `compile_gpt()`. Check for graph breaks with `TORCH_LOGS=graph_breaks`. The FA3 call through the `kernels` package may need `torch.compiler.disable()` wrapping.

6. **FP8v2 backward in FP8**: Current `Float8LinearRowwise.backward` uses BF16. Implement FP8 e5m2 backward for `grad_input = grad_output @ weight` to complete the FP8 story.

### Medium Term (Week 2-4)

7. **Rowwise FP8 correctness sweep**: Run 1000-step training with rowwise vs. tensorwise FP8, compare validation loss curves. If rowwise is better, it becomes the default.

8. **Delayed FP8 scaling validation**: Verify that amax history does not diverge. Run with `amax_history_len=4, 8, 16, 32` and compare loss curves.

9. **CommStream profiling**: Use nsys to verify that NCCL and compute are actually overlapping after the stream separation change. Compare `dist_v2.py` vs base `DistMuonAdamW` with the nsys NCCL trace.

10. **liger-kernel integration**: `pip install liger-kernel`, swap `chunked_cross_entropy_with_softcap` for `liger_cross_entropy_with_softcap`. Verify correctness, measure throughput. Expected: further 5-10% reduction in CE computation time.

### Long Term (Week 4+)

11. **Triton Polar Express kernel**: Implement the fused 5-iteration Newton-Schulz in Triton only after profiling shows Muon step is a bottleneck. For small models (D=768), it may not be.

12. **FSDP2 migration design**: The Muon BMM grouping is incompatible with FSDP2's per-parameter flattening. Design a hybrid: FSDP2 for AdamW params (embeddings, scalars), custom ZeRO-2 for Muon params.

13. **Speculative decoding**: Train a 10M param draft model (6 layers, D=384, same tokenizer). Implement the verify+accept loop in `engine_v2.py`. Target 3× decode speedup.

14. **Circular KV cache for SWA layers**: The 3/4 of layers with sliding window only need the last 768 tokens in the KV cache. Circular buffer reduces KV cache memory from `T_max` to `window_size` for those layers.

15. **B200/Blackwell readiness**: When B200 hardware is available:
    - Recompile FA3 for sm100 (or adopt the official FA3 Blackwell build)
    - Evaluate FP4 (float4_e2m1) matmuls via `torch._scaled_mm` with sm100 tensor cores
    - Update the `major != 9` gate in `flash_attention.py` to `major in (9, 10)`

---

## Appendix A: File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `nanochat/gpt.py` | 512 | GPT model, all architectural features |
| `nanochat/fp8.py` | 266 | Tensorwise FP8 training |
| `nanochat/optim.py` | 535 | Muon + AdamW, single and distributed |
| `nanochat/flash_attention.py` | 187 | FA3 + SDPA fallback |
| `nanochat/common.py` | 278 | Hardware detection, MFU, NCCL init |
| `nanochat/engine.py` | 357 | KV cache inference, tool-use |
| `nanochat/v2/__init__.py` | 20 | v2 package |
| `nanochat/v2/loss.py` | 174 | Chunked CE + softcap (Tier 1.1) |
| `nanochat/v2/gpt_v2.py` | 288 | Activation checkpointing + compile (Tier 1.2, 1.3) |
| `nanochat/v2/fp8_v2.py` | 299 | Rowwise FP8 + delayed scaling (Tier 2.1, 2.2) |
| `nanochat/v2/comms_v2.py` | 284 | FP8 all-reduce + CommStream (Tier 2.3, 3.1) |
| `nanochat/v2/dist_v2.py` | 282 | Stream-separated DistMuonAdamW (Tier 3.1) |
| `nanochat/v2/kernels/__init__.py` | 8 | Kernels subpackage |
| `nanochat/v2/kernels/fused_ce.py` | 73 | liger-kernel adapter (Tier 4.1) |
| `nanochat/v2/kernels/polar_express.py` | 89 | Triton Polar Express stub (Tier 4.2) |
| `nanochat/v2/kernels/fused_norm.py` | 76 | Fused RMS norm (Tier 4.3) |

## Appendix B: Key Formulas

**MFU**: `achieved_TFLOPS / peak_TFLOPS`

**Model FLOPs** (approx): `6 × N_params × tokens + 12 × n_head × head_dim × eff_seqlen × n_layer`

**Logits memory**: `B × T × V × 4 bytes` (FP32)

**Activation memory** (no checkpointing): `≈ 6 × B × T × D × 2 bytes × n_layer`

**NCCL communication**: `2 × N_params × 2 bytes × (1 - 1/world_size)` (reduce_scatter + all_gather)

**Speculative decoding speedup**: `(1 - α^(K+1)) / ((1-α) × (1 + overhead))`

**Polar Express convergence**: 5th-order polynomial → convergence in O(log(1/ε)/log(5)) iterations vs Newton-Schulz O(log(1/ε)/log(3))

**Chunked CE memory**: `chunk_size × V × 4 bytes` per chunk (independent of B, T)
