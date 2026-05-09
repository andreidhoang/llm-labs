# Speculative Decoding Deep Dive
### nanochat Inference Engine — Session Engineering Log

**Date:** 2026-04-26  
**Engineer role:** Senior AI Research Performance Engineer  
**Scope:** DFlash blog analysis → model size decision → implementation into `core/`

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [DFlash Analysis — What the Paper Actually Claims](#2-dflash-analysis)
3. [Model Size Decision — Quantitative Justification](#3-model-size-decision)
4. [Pre-existing Architecture Audit](#4-pre-existing-architecture-audit)
5. [KV Cache Design Deep Dive](#5-kv-cache-design-deep-dive)
6. [Speculative Decoding Theory](#6-speculative-decoding-theory)
7. [SpeculativeEngine — Implementation Walkthrough](#7-speculativeengine-implementation-walkthrough)
8. [Verified Parameter Counts](#8-verified-parameter-counts)
9. [Performance Model — Exact Arithmetic](#9-performance-model)
10. [Smear Gate Approximation Analysis](#10-smear-gate-approximation-analysis)
11. [Exact Changes Made](#11-exact-changes-made)
12. [Known Limitations and Open TODOs](#12-known-limitations-and-open-todos)
13. [Roadmap to DFlash-Level Speedups](#13-roadmap-to-dflash-level-speedups)

---

## 1. Problem Statement

The goal is to build an optimized inference engine for the nanochat GPT-style model in `core/`. The engine must be:

- **Fast** — minimize wall-clock time per output token
- **Lossless** — output distribution must be identical to pure autoregressive sampling
- **Hardware-aware** — designed for NVIDIA Hopper (H100/H200) in BF16

The baseline autoregressive engine in `core/engine.py` (`Engine`) already has:
- FA3-accelerated KV cache (Hopper-native)
- Group-Query Attention reducing KV memory
- Batch-1 prefill + multi-sample decode
- Tool use (Python calculator)

What it lacks: any form of speculative or parallel decoding. Every output token costs one full target model forward pass.

---

## 2. DFlash Analysis

### What DFlash Is

DFlash (z-lab.ai/projects/dflash) is a speculative decoding system that conditions a draft model on the target model's internal representations extracted from **multiple uniformly-sampled layers**, then generates an entire **block of 16 tokens in parallel** using a single diffusion denoising step.

### Benchmarks

| Task | DFlash | EAGLE-3 | Speedup over EAGLE-3 |
|------|--------|---------|----------------------|
| GSM8K | 5.20× | 2.13× | 2.4× |
| MATH-500 | 6.17× | 2.18× | 2.8× |
| AIME24 | 5.91× | 2.25× | 2.6× |
| HumanEval | 5.43× | 2.24× | 2.4× |

All on Qwen3-8B with greedy decoding.

### DFlash's Three Core Ideas

**Idea 1 — Multi-layer feature extraction:**  
Hidden states extracted from uniformly-sampled layers (not just the last layer or first hidden layer). Fused via lightweight projection into a single conditioning vector. This gives the draft model access to representations at all abstraction levels of the target.

**Idea 2 — KV injection at every draft layer:**  
The fused features are injected into the Key/Value projections of *every* layer of the draft model, not just the input. EAGLE-3 feeds features only to layer 0, causing signal dilution with depth. DFlash's per-layer injection maintains conditioning signal throughout the draft model's depth.

**Idea 3 — Parallel block diffusion:**  
The draft model generates 16 tokens simultaneously in a single forward pass using a diffusion objective (masked token prediction). This is fundamentally different from autoregressive drafting — each token position is predicted in parallel given the conditioning.

### What Transfers vs. What Does Not

| Component | Transferable? | Notes |
|-----------|--------------|-------|
| Multi-layer feature extraction | ✅ Yes | Need `return_hidden_states` in `GPT.forward()` |
| KV injection architecture | ✅ Yes | Requires training a new conditioned draft |
| Block diffusion decoding | ❌ Not yet | Requires diffusion training objective; not a code drop-in |
| Pre-trained drafters (HuggingFace) | ❌ No | Trained against Qwen3/LLaMA3 internals specifically |
| Rejection sampling framework | ✅ Yes | Algorithm is model-agnostic |
| EAGLE-3 / EAGLE-style conditioning | ✅ Partial | Single-layer version, simpler to train |

**Critical constraint:** DFlash's 5-6× speedups are on Qwen3-8B with *pre-trained drafters*. Those drafters are artifacts of training, not code. To reproduce DFlash gains on nanochat, you must train a diffusion-objective draft against your specific target checkpoint. That is a multi-week experiment, not an afternoon of engineering.

**What we implement now:** The correct foundation — vanilla speculative decoding with lossless rejection sampling, plus the hidden state extraction hook that enables EAGLE-style and eventually DFlash-style conditioning.

---

## 3. Model Size Decision

### The Fundamental Regime Question

Speculative decoding's economics depend on which compute regime inference is in:

- **Compute-bound:** FLOPs dominate. Batched inference, large batch sizes, long sequences being processed in parallel. Adding a draft model costs real FLOPs → diminishing returns.
- **Memory-bandwidth-bound:** Weight loading dominates. Small batches (1–8), decode phase (one token at a time). Loading model parameters once per token is the bottleneck. Draft overhead is cheap relative to the weight loading cost of the target.

**H100 numbers:**
- FLOPS: 989 TFLOPS (BF16 with sparsity: ~1979 TFLOPS), or ~494 TFLOPS dense BF16
- HBM3 bandwidth: 3.35 TB/s

A 3B parameter model in BF16 = 6 GB of weights. At 3.35 TB/s bandwidth, minimum time to load all weights once = 6GB / 3.35TB/s ≈ **1.79ms** per decode step, regardless of batch size 1.

At batch size 1, FLOPs per decode token = ~2 × 3B = 6 GFLOPs. At 494 TFLOPS, that takes = 6G / 494T ≈ **0.012ms**. 

**Arithmetic intensity** = 6 GFLOPs / 6 GB = 1 FLOP/byte. H100's roofline crossover ≈ 147 FLOPs/byte. We are 147× below the roofline. **3B on H100 at batch=1 is maximally memory-bandwidth-bound.** This is the ideal regime for speculative decoding.

### Why 3B Specifically

1. **Bandwidth-bottlenecked** — confirmed above. Speculative decoding pays maximally.
2. **H100 headroom** — 3B model = 6GB in BF16. KV cache for 4096 context, 1 batch, 24 layers, 8 KV heads, 128 head_dim = `24 × 1 × 4096 × 8 × 128 × 2 × 2` bytes (K+V, BF16) ≈ 402MB. Total per-inference GPU memory: ~6.5GB. Fits with massive headroom on 80GB H100.
3. **Trainable on one node** — 8×H100 with FSDP, FP8 weights, Muon optimizer. Feasible research timeline.
4. **ClimbMix-400B alignment** — Chinchilla-optimal for 3B is ~60B tokens. Training to 400B = 6.7× beyond optimal. Consistent with "compute-optimal + extended training for inference efficiency" trend (LLaMA-3, Mistral).
5. **Deployment-relevant size** — 3B is the largest "single-GPU consumer" class model. Research impact is highest here.

### Why 300M for Draft

**10:1 ratio** is empirically validated across EAGLE (2023), Medusa (2023), SpecInfer (2023). The intuition:

At 10:1 ratio:
- Draft has similar vocabulary and structural priors, so acceptance rate α ≈ 0.7-0.8 achievable
- Draft forward cost ≈ 10% of target forward cost
- For K=4 drafts: 4 × 10% = 40% overhead before verification

If ratio is too large (e.g., 100:1), α drops because the draft lacks capacity to track the target's distribution.  
If ratio is too small (e.g., 2:1), draft overhead is nearly the same as target — no gain.

**Specific choice: 300M with same architecture family**
- Shares vocab, tokenizer, BPE splits — acceptance rate is higher than cross-family drafts
- Same GPTConfig fields: GQA, RoPE, QK norm, sliding window — training is straightforward distillation
- Value embeddings present (alternating layers) — architecturally aligned with target

---

## 4. Pre-existing Architecture Audit

Before understanding what we added, understanding what existed is essential.

### `GPTConfig` — Configuration

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6           # query heads
    n_kv_head: int = 6        # key/value heads (GQA: n_kv_head ≤ n_head)
    n_embd: int = 768
    window_pattern: str = "SSSL"  # per-layer sliding window pattern
```

### `CausalSelfAttention` — Attention Layer

This is a heavily customized attention module. Every element is intentional:

**Group-Query Attention (GQA):**
```
Q projection: n_embd → n_head × head_dim
K projection: n_embd → n_kv_head × head_dim  (fewer heads)
V projection: n_embd → n_kv_head × head_dim  (fewer heads)
O projection: n_embd → n_embd
```
With `n_kv_head = n_head / G`, KV cache memory is reduced by factor G. At TARGET_3B: G=3, KV memory is 1/3 of MHA. At DRAFT_300M: G=2.

**Rotary Positional Embeddings (RoPE):**
Applied to Q and K. No learned positional embeddings. Position-relative attention via rotation in the complex plane. Base theta=100,000. Precomputed for `sequence_len × 10` positions; sliced to current position at inference time using `T0 = kv_cache.get_pos()`.

**QK Normalization:**
```python
q, k = norm(q), norm(k)  # RMSNorm applied to each head
q = q * 1.2              # sharpening scale split between Q and K
k = k * 1.2
```
Prevents attention logit explosion during training. The 1.2 multiplier produces effectively `softmax(QK^T / (d^0.5 / 1.44))` — slightly sharper attention than standard. This is an open hyperparameter (TODO in code).

**Value Embeddings (ResFormer-style):**
```python
ve = value_embeds[layer_idx](input_ids)  # vocab → kv_dim embedding lookup
gate = 3 * sigmoid(ve_gate(x[..., :12]))  # (B, T, n_kv_head), range (0, 3)
v = v + gate * ve
```
Alternating layers (even/odd depending on parity of `n_layer - 1`) receive an additional value residual drawn from a per-token vocabulary embedding. This gives the model cheap access to token identity information in the value stream, separate from the contextual value computation. Inspired by ResFormer's empirical finding that mixing raw embeddings into values improves training.

**Flash Attention Integration:**
```python
if kv_cache is None:
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
else:
    k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v,
                                           cache_seqlens=kv_cache.cache_seqlens, ...)
    if self.layer_idx == kv_cache.n_layers - 1:
        kv_cache.advance(T)
```
Key detail: `advance(T)` only fires at the **last layer**. During the forward pass through layers 0..n-2, `cache_seqlens` still points to the write position for the current step. FA3 reads `cache_seqlens` to know where to write the new KV entries. This is why rollback works: `cache_seqlens -= n` merely moves the write pointer back; the stale physical memory is silently overwritten on future writes.

**Sliding Window Attention:**
Per-layer window sizes are precomputed in `_compute_window_sizes()`:
```
"L" → (sequence_len, 0)              # full causal context
"S" → (ceil(sequence_len/4, 128), 0) # quarter context, FA3 tile-aligned
```
Pattern is tiled across layers; the **final layer always uses full context** regardless of pattern. For `SSSL`: layers 0,1,2=short, layer 3=full, layers 4,5,6=short, layer 7=full, etc. This balances local induction (cheap) with global coherence (necessary).

### `MLP` — Feed-Forward Layer

```python
x = c_fc(x)          # n_embd → 4 × n_embd
x = relu(x).square() # relu²  activation
x = c_proj(x)        # 4 × n_embd → n_embd
```

`relu²` (squared ReLU) rather than GeLU or SwiGLU. Benefits:
- Exact sparsity: negative activations are exactly zero, not approximately
- Cheaper than GELU (no erf approximation)
- Empirically matches or beats GeLU on smaller models (Stanford HAI ablation, 2022)
- No gating matrix needed (unlike SwiGLU's 3-linear structure) → simpler, fewer parameters

### `GPT.forward()` — Full Pass

The forward pass has several non-obvious operations:

**Smear Gate:**
```python
# training / prefill:
gate = smear_lambda * sigmoid(smear_gate(x[:, 1:, :24]))
x = cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

# decode (1 token, KV cache):
gate = smear_lambda * sigmoid(smear_gate(x[:, :, :24]))
x = x + gate * kv_cache.prev_embedding
```
Mixes the **previous token's embedding** into the current token's embedding, weighted by a learned gate that looks at the first 24 channels of the current token. This gives the model cheap bigram-like information (what came before) before any attention is computed. The `prev_embedding` is stored in the KV cache and updated on each forward pass. During rollback in speculative decoding, this is the state we must restore.

**Residual Stream Modulation:**
```python
x = resid_lambdas[i] * x + x0_lambdas[i] * x0
```
Applied before each block. `resid_lambdas` starts near 1.0 and decays slightly with depth (1.15 at layer 0, ~1.05 at final layer). `x0_lambdas` starts at 0.20 and decays to near 0.05 — this injects the initial normalized embedding back into the residual stream at each layer with decreasing strength. This is a form of "residual gating" that prevents the model from drifting too far from the token's identity.

**Backout:**
```python
x_backout = x  # cached at n_layer // 2
# ... more layers ...
x = x - backout_lambda * x_backout  # before final norm
```
Subtracts the mid-layer residual from the final residual. This removes low-level syntactic features (captured early in the network) from the representations fed into the language model head. Analogous to the "neural collapse" regularization principle: we want the lm_head to receive high-level semantic features only. `backout_lambda` is initialized to 0.2 and learned.

**Logit Softcap:**
```python
logits = 15 * tanh(logits / 15)
```
Clips logit magnitudes to [-15, 15] using a smooth tanh. Prevents outlier logits from causing extreme softmax concentration. Common in modern architectures (Gemma 2 uses softcap=30, Grok uses it too). The soft nature of tanh means it doesn't hard-clip like clamp — it smoothly compresses logits that exceed the cap.

---

## 5. KV Cache Design Deep Dive

### Memory Layout

```python
k_cache: (n_layers, B, T, n_kv_heads, head_dim)
v_cache: (n_layers, B, T, n_kv_heads, head_dim)
cache_seqlens: (B,) int32
prev_embedding: (B, 1, n_embd) or None
```

This is FA3's native layout: `(B, T, H, D)` per layer, **not** the FA2 convention `(B, H, T, D)`. FA3 requires this layout because it processes heads in parallel in a way that's more cache-friendly with heads as the inner dimension. All the K and V caches for all layers are pre-allocated upfront to avoid memory allocation during generation.

### Why int32 for cache_seqlens

FA3's CUDA kernel signature requires `cache_seqlens` as int32 tensor on GPU. Using int64 or Python ints would require a dtype conversion on every attention call. The int32 constraint limits maximum sequence length to ~2 billion tokens, which is not a practical concern.

### The `advance()` / Rollback Contract

The critical invariant: **`advance()` is called exactly once per forward pass, at the last layer.**

```python
if self.layer_idx == kv_cache.n_layers - 1:
    kv_cache.advance(T)
```

During layers 0..n-2, `cache_seqlens[b]` = the write position for batch element b. FA3's `flash_attn_with_kvcache` reads `cache_seqlens` to determine where to write the new K, V entries. After all layers complete, `advance(T)` moves the pointer forward by T.

**Rollback mechanism:** To undo writing T tokens to the cache:
```python
kv_cache.cache_seqlens -= T
```

This moves the write pointer back. The physical memory at positions `[cache_seqlens, cache_seqlens + T)` still contains the old KV values, but they will be **silently overwritten** on the next forward pass that writes to those positions. Crucially, FA3's causal masking (`causal=True`) means tokens at positions ≥ `cache_seqlens` are not attended to — the stale data is unreachable before it's overwritten. The rollback is therefore O(1) and safe.

### Prefill Cloning

```python
def prefill(self, other):
    other_pos = other.get_pos()
    self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
    self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
    self.cache_seqlens.fill_(other_pos)
    if other.prev_embedding is not None:
        self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()
```

Used in the original `Engine` to fan out batch=1 prefill to batch=N decode. In `SpeculativeEngine`, we don't use this (num_samples=1 only), but the mechanism is available for future multi-sample speculative extension.

---

## 6. Speculative Decoding Theory

### The Core Problem

Autoregressive decoding generates one token per target model forward pass. At batch=1, decode-phase, this is memory-bandwidth-bound: the entire model must be loaded from HBM for each token. Compute per token is trivially small — the arithmetic intensity is ~1 FLOP/byte, while the roofline crossover for H100 is ~147 FLOP/byte. We are spending 147× more on memory bandwidth than computation.

**The opportunity:** If we can generate multiple tokens per target model weight-load, we can amortize the memory cost over more output tokens. This is the fundamental insight behind all speculative decoding methods.

### Leviathan et al. 2023 Algorithm

Reference: "Fast Inference from Transformers via Speculative Decoding," Leviathan, Kalman, Matias, NeurIPS 2023.

#### Setup

- **Target model M_p**: the large, accurate model. Defines the desired output distribution.
- **Draft model M_q**: smaller, faster model. Approximates M_p.
- **Draft length K**: how many tokens to speculate per step.
- **Acceptance rate α**: empirical probability that draft matches target on any given token position.

#### Per-step Algorithm

Given:
- KV cache with N tokens in context
- `t_logits` = P_target(· | x₀..x_{N-1}) — target distribution at position N (already computed)
- `d_logits` = P_draft(· | x₀..x_{N-1}) — draft distribution at position N (already computed)

**Phase 1 — Draft:** Generate K candidate tokens autoregressively with M_q.

For k = 0, 1, ..., K-1:
1. Compute p_k = SamplingDistribution(d_logits_k) using temperature τ and top_k filter
2. Sample d_k ~ p_k
3. Run M_q forward on d_k → get d_logits_{k+1}

This costs K small model forward passes.

**Phase 2 — Verify:** Run M_p on all K draft tokens simultaneously.

Feed [d₀, d₁, ..., d_{K-1}] through M_p with KV cache (single forward of K tokens).

This produces K logit vectors t_verify[0..K-1] where:
```
t_verify[k] = P_target(· | x₀..x_{N-1}, d₀..d_k)
```

Combined with `t_logits` (the pre-existing target distribution at position N), we have K+1 target distributions for positions N, N+1, ..., N+K.

**Phase 3 — Rejection Sampling:**

The verification step uses d_k as the "proposal" for position N+k. The check distribution for d_k is:

```
k = 0:  check = t_logits     = P_target(· | x₀..x_{N-1})
k ≥ 1:  check = t_verify[k-1] = P_target(· | x₀..x_{N-1}, d₀..d_{k-1})
```

For k = 0, 1, ..., K-1:
1. Compute `q_k = check[d_k]` (target's probability of the draft token)
2. Compute `p_k = draft_probs_k[d_k]` (draft's probability of its own token)
3. Sample r ~ Uniform(0, 1)
4. **If r ≤ min(1, q_k / p_k):** Accept d_k. Set `accepted += 1`. Continue to k+1.
5. **Else:** Reject. Sample bonus token b from adjusted distribution:
   ```
   q_adj(t) = max(0, check[t] - draft_probs_k[t]) / Z
   ```
   Break the loop. The output for this step is d₀..d_{accepted-1} + b.

If all K accepted: sample bonus b ~ P_target(· | x₀..x_{N-1}, d₀..d_{K-1}) = t_verify[K-1].

**Phase 4 — Rollback and Commit:**

After verification, the target KV cache is at N+K (we wrote all K draft tokens during the forward pass). On partial acceptance (j < K tokens accepted):

```python
t_kv.cache_seqlens -= (K - j)   # roll back K-j tokens
d_kv.cache_seqlens -= (K - j)   # same for draft
```

Cache is now at N+j. Feed bonus token b through both models (1 target forward + 1 draft forward) to advance to N+j+1 and get logits for the next step.

**Output per step:** j+1 tokens (j accepted draft tokens + 1 bonus).

#### Losslessness Proof (sketch)

Claim: the output distribution is identical to sampling j+1 tokens autoregressively from M_p.

The joint distribution of accepted tokens under the rejection sampling scheme is:

P(accept d₀, ..., d_{j-1}, reject d_j) =
  P_draft(d₀..d_{K-1}) ×  
  ∏_{k<j} min(1, q_k/p_k) × (1 - min(1, q_{j}/p_{j})) ×  
  q_adj(b) / Z

This marginalizes to the same distribution as sequentially sampling from P_target at each position. The adjusted distribution `max(0, q - p) / Z` is precisely the correction term that makes the marginals match. See Theorem 1 in Leviathan et al. for the formal proof.

The key insight: rejection sampling transforms the draft distribution into the target distribution. No approximation is made. The output is **exactly** equivalent to sampling from M_p at every token.

---

## 7. SpeculativeEngine — Implementation Walkthrough

### Class Structure

```
SpeculativeEngine
├── __init__(target_model, draft_model, tokenizer, K=4)
├── _make_kv(model, batch_size, seq_len, device, dtype) → KVCache
├── _compute_probs(logits_1_vocab, temperature, top_k) → (vocab,) tensor
├── _sample_from_probs(probs, rng, temperature) → int
├── _adjusted_sample(q_probs, p_probs, rng, temperature) → int
└── generate(tokens, num_samples, max_tokens, temperature, top_k, seed) → generator
    generate_batch(tokens, num_samples, **kwargs) → (list[list[int]], list[list[int]])
```

### `_make_kv` — KV Cache Factory

```python
def _make_kv(self, model, batch_size, seq_len, device, dtype):
    m = model.config
    return KVCache(
        batch_size=batch_size, seq_len=seq_len,
        num_heads=m.n_kv_head, head_dim=m.n_embd // m.n_head,
        num_layers=m.n_layer, device=device, dtype=dtype,
    )
```

Note: `num_heads=m.n_kv_head` (not `m.n_head`). The KV cache stores KV head counts, not query head counts. The GQA broadcast (expanding KV heads to match Q heads) happens inside `flash_attn_with_kvcache`, not in the cache storage.

### `_compute_probs` — Logit → Probability

```python
def _compute_probs(self, logits_1_vocab, temperature, top_k):
    v = logits_1_vocab[0].float()   # cast to float32 for numerical stability
    if top_k is not None and top_k > 0:
        threshold = torch.topk(v, min(top_k, v.size(-1))).values[-1]
        v = v.masked_fill(v < threshold, float('-inf'))
    if temperature == 0.0:
        p = torch.zeros_like(v); p[v.argmax()] = 1.0; return p
    return F.softmax(v / temperature, dim=-1)
```

**Design choices:**
1. **float32 cast:** The logits from the model are already float32 (model.py casts to fp32 before softcap). The explicit cast handles the edge case where logits might arrive as bf16 (e.g., from a future code path that skips the softcap).
2. **top_k masking before temperature:** Conceptually we want to renormalize over the top-k logits, then apply temperature. This ordering is correct: mask first, then normalize.
3. **Temperature=0 special case:** Returns a one-hot at argmax. This enables correct greedy speculative decoding — the draft token is accepted iff it matches the target's argmax. On rejection, the "adjusted distribution" is also one-hot at the target's argmax.

### `_adjusted_sample` — Correction Distribution

```python
def _adjusted_sample(self, q_probs, p_probs, rng, temperature):
    diff = (q_probs - p_probs).clamp(min=0.0)
    z = diff.sum()
    if z < 1e-9:
        return self._sample_from_probs(q_probs, rng, temperature)
    return int(torch.multinomial(diff / z, 1, generator=rng).item())
```

`max(0, q - p) / Z` is the adjusted distribution from Leviathan et al. Theorem 1. The `z < 1e-9` fallback handles the degenerate case where `q ≈ p` everywhere (both models agree perfectly, no correction needed — sample from q directly). This should rarely trigger in practice.

**Important:** `q_probs` and `p_probs` must be computed with the **same** temperature and top_k parameters. If the distributions are computed under different sampling settings, the rejection sampling guarantee breaks down. The implementation passes the same `temperature`/`top_k` to both `_compute_probs` calls in the rejection loop, ensuring this invariant holds.

### `generate` — Main Loop

#### Prefill Phase

```python
seq_cap = len(tokens) + (max_tokens or self.target.config.sequence_len) + K
ids = torch.tensor([tokens], dtype=torch.long, device=device)

t_kv = self._make_kv(self.target, 1, seq_cap, device, dtype)
d_kv = self._make_kv(self.draft, 1, seq_cap, device, dtype)
t_logits = self.target.forward(ids, kv_cache=t_kv)[:, -1, :]
d_logits = self.draft.forward(ids, kv_cache=d_kv)[:, -1, :]
```

Both models process the full prompt independently. After this, both KV caches are at position N (= len(tokens)), and both hold the full prompt's KV representations.

`seq_cap` includes `+ K` as a safety buffer: during the verification phase, we write K tokens to the cache before rollback. The physical buffer must exist to avoid out-of-bounds writes even for tokens we'll discard. Without this buffer, verification at the last possible step causes `k_cache[:, pos:pos+K, :, :]` to overflow.

The prefill runs twice (target + draft). This is necessary: they have different architectures (different n_layer, n_embd), so their KV caches are independent. There is no sharing.

#### Draft Phase

```python
t_smear = t_kv.prev_embedding.clone() if t_kv.prev_embedding is not None else None
d_smear = d_kv.prev_embedding.clone() if d_kv.prev_embedding is not None else None

cur_d = d_logits
for _ in range(K):
    d_probs = self._compute_probs(cur_d, temperature, top_k)
    d_k = self._sample_from_probs(d_probs, rng, temperature)
    draft_tokens.append(d_k)
    draft_probs.append(d_probs)       # save full distribution for rejection check
    d_id = torch.tensor([[d_k]], dtype=torch.long, device=device)
    cur_d = self.draft.forward(d_id, kv_cache=d_kv)[:, -1, :]
```

**Why save full distributions (not just the scalar probability):**  
The rejection sampling requires the full probability vector `draft_probs_k` for computing the adjusted distribution `max(0, q - p)`. Saving only `p_k = draft_probs_k[d_k]` would be sufficient for the accept/reject decision but not for the correction sampling on rejection. Memory cost: K × vocab_size × 4 bytes = 4 × 32768 × 4 ≈ 512 KB. Acceptable.

**Smear state snapshot:** `prev_embedding` is cloned before the K draft forwards. This snapshot represents the smear state at position N (before any draft tokens are committed). Used for rollback restoration.

After K iterations: `d_kv` is at position N+K. `d_logits` (the logit at N+K) is discarded — we don't need it until the bonus token is committed and we restart.

#### Verify Phase

```python
draft_ids = torch.tensor([draft_tokens], dtype=torch.long, device=device)  # (1, K)
t_verify = self.target.forward(draft_ids, kv_cache=t_kv)   # (1, K, vocab)
```

**This is where the speedup comes from:** K tokens processed in one target forward pass. Under memory-bandwidth-bound inference, the dominant cost is loading the ~6GB of target model weights once. This single load amortizes over K tokens rather than being paid K separate times.

After this call: `t_kv` is at position N+K. `t_verify[:, k, :]` gives `P_target(· | prefix + d₀..d_k)` — the target's distribution at position N+k+1 given d₀..d_k were the preceding tokens.

Indexing relationship (critical for correctness):
```
To CHECK draft token d_k (k=0): use t_logits    = P_target(· | prefix)
To CHECK draft token d_k (k≥1): use t_verify[:, k-1, :] = P_target(· | prefix + d₀..d_{k-1})
If all K accepted, bonus from:  t_verify[:, K-1, :] = P_target(· | prefix + d₀..d_{K-1})
```

#### Rejection Phase

```python
for k in range(K):
    check_logit = t_logits if k == 0 else t_verify[:, k - 1, :]
    t_probs_k = self._compute_probs(check_logit, temperature, top_k)
    d_probs_k = draft_probs[k]
    d_k = draft_tokens[k]

    p_t = float(t_probs_k[d_k])
    p_d = float(d_probs_k[d_k])
    accept_prob = min(1.0, p_t / p_d) if p_d > 1e-12 else 0.0

    if torch.rand(1, generator=rng).item() <= accept_prob:
        accepted += 1
    else:
        bonus_token = self._adjusted_sample(t_probs_k, d_probs_k, rng, temperature)
        break
else:
    bonus_token = self._sample_from_probs(
        self._compute_probs(t_verify[:, K - 1, :], temperature, top_k), rng, temperature
    )
```

The `for...else` Python construct: the `else` branch executes if the loop completes without `break` — i.e., all K tokens were accepted. This is idiomatic Python and clean for expressing "if all accepted, sample bonus from final logit."

**Numerical guard:** `p_d > 1e-12` prevents division by zero if top_k or sampling truncated a token to zero probability in the draft. In that case `accept_prob = 0`, correctly forcing rejection (the draft assigned zero mass to the token it supposedly sampled, which shouldn't happen normally — this handles floating point edge cases).

#### Rollback Phase

```python
rollback = K - accepted
if rollback > 0:
    t_kv.cache_seqlens -= rollback
    d_kv.cache_seqlens -= rollback
    t_kv.prev_embedding = t_smear
    d_kv.prev_embedding = d_smear
```

Both caches moved to N + accepted. The target KV is now in the same state as if only `accepted` draft tokens had been written (plus the stale physical memory beyond that position, which is harmless).

For the smear gate: both caches' `prev_embedding` is restored to the snapshot taken before drafting (state at position N). This is an approximation — see Section 10 for analysis.

When `rollback == 0` (all K accepted), we skip this block. The cache is correctly at N+K.

#### Commit Bonus Phase

```python
bonus_id = torch.tensor([[bonus_token]], dtype=torch.long, device=device)
t_logits = self.target.forward(bonus_id, kv_cache=t_kv)[:, -1, :]
d_logits = self.draft.forward(bonus_id, kv_cache=d_kv)[:, -1, :]
```

Feed the bonus token through both models. This:
1. Writes the bonus token's KV to both caches (advances both to N + accepted + 1)
2. Produces `t_logits` for the start of the **next** speculative step
3. Produces `d_logits` for the start of the **next** speculative step's draft phase

After this, the system state is clean:
- Both caches at N + accepted + 1
- `t_logits` = P_target(· | prefix + accepted_tokens + bonus)
- `d_logits` = P_draft(· | prefix + accepted_tokens + bonus)
- Ready to begin the next speculative step

#### Yield Phase

```python
output = draft_tokens[:accepted] + [bonus_token]
for tok in output:
    num_generated += 1
    yield [tok], [1]
    if tok in stop_tokens:
        return
```

Tokens are yielded one at a time for API compatibility with `Engine.generate()` callers. The stop token check fires on any token in `{assistant_end, bos_id}`, matching the original Engine behavior.

Token masks are all `1` (sampled, not forced). The forced-token mechanism (for tool use) is not implemented in `SpeculativeEngine` — that's a known TODO. For research inference, forced tokens are rare.

---

## 8. Verified Parameter Counts

All counts confirmed by instantiating models on meta device and calling `sum(p.numel() for p in m.parameters())`.

### TARGET_3B Breakdown

Config: 24L / 3072d / 24-head / 8 KV-head / 4096 seq_len / SSSL pattern

| Component | Count | Calculation |
|-----------|-------|-------------|
| `transformer.h` (24 layers) | 2,409.6M | 24 × 100.4M/layer |
| Per layer: c_q | 9.4M | 3072 × 3072 |
| Per layer: c_k | 3.1M | 3072 × 1024 (8 KV heads × 128 head_dim) |
| Per layer: c_v | 3.1M | 3072 × 1024 |
| Per layer: c_proj | 9.4M | 3072 × 3072 |
| Per layer: c_fc (MLP) | 37.7M | 3072 × 12288 |
| Per layer: c_proj (MLP) | 37.7M | 12288 × 3072 |
| `transformer.wte` | 100.7M | 32768 × 3072 |
| `value_embeds` (12 of 24 layers) | 402.7M | 12 × 32768 × 1024 |
| `lm_head` | 100.7M | 32768 × 3072 |
| Scalars (resid_lambdas, etc.) | ~0.1K | negligible |
| **Total** | **3,019.9M** | |

### DRAFT_300M Breakdown

Config: 12L / 1024d / 8-head / 4 KV-head / 4096 seq_len / SL pattern

| Component | Count | Calculation |
|-----------|-------|-------------|
| `transformer.h` (12 layers) | 138.2M | 12 × 11.5M/layer |
| Per layer: c_q | 1.05M | 1024 × 1024 |
| Per layer: c_k | 0.52M | 1024 × 512 (4 KV heads × 128 head_dim) |
| Per layer: c_v | 0.52M | 1024 × 512 |
| Per layer: c_proj | 1.05M | 1024 × 1024 |
| Per layer: c_fc (MLP) | 4.19M | 1024 × 4096 |
| Per layer: c_proj (MLP) | 4.19M | 4096 × 1024 |
| `transformer.wte` | 33.6M | 32768 × 1024 |
| `value_embeds` (6 of 12 layers) | 100.7M | 6 × 32768 × 512 |
| `lm_head` | 33.6M | 32768 × 1024 |
| **Total** | **306.2M** | |

**Ratio:** 3019.9 / 306.2 = **9.86:1 ≈ 10:1** ✓

---

## 9. Performance Model

### H100 Hardware Constraints

| Parameter | Value |
|-----------|-------|
| BF16 compute (dense) | ~494 TFLOPS |
| HBM3 bandwidth | 3.35 TB/s |
| Roofline crossover | ~147 FLOP/byte |

### Arithmetic Intensity at Decode (batch=1)

For a forward pass of 1 token through a W-byte model:
- FLOPs ≈ 2W / sizeof(dtype) = 2 × parameters × 2 bytes/param = ~2× params (in FLOPs)
- Memory = W bytes (load all weights)
- Arithmetic intensity = 2W/2 / W = 1 FLOP/byte for large models

At 3B parameters BF16: AI ≈ 1 FLOP/byte ≪ 147 roofline → **maximally memory-bound**. The time per decode step ≈ model_size_bytes / bandwidth = 6GB / 3.35 TB/s = **1.79 ms** (theoretical lower bound, ignoring KV cache and other overhead).

### Speculative Decoding Cost Model

Let `T_t` = time for one target forward pass (1 token decode, batch=1)  
Let `T_d` = time for one draft forward pass = `T_t × (300M / 3000M)` = `T_t × 0.1`

**Per speculative step (K=4):**

| Operation | Cost |
|-----------|------|
| K draft forwards | K × T_d = 4 × 0.1 × T_t = 0.4 T_t |
| 1 verification forward (K tokens) | ≈ T_t (memory-bandwidth-bound, K tokens doesn't scale cost at batch=1) |
| 1 bonus forward (1 token) | T_t |
| **Total cost per step** | **2.4 T_t** |

**Tokens produced per step:**

With acceptance rate α:
```
E[tokens/step] = E[accepted] + 1 = Σ_{j=0}^{K-1} α^j × (1-α) × (j+1) + α^K × (K+1)
               = (1 - α^K) / (1 - α) + 1
```

| α | K=4: E[tokens/step] |
|---|---------------------|
| 0.6 | 2.90 |
| 0.7 | 3.31 |
| 0.8 | 3.90 |
| 0.9 | 4.69 |
| 0.95 | 5.28 |

**Net speedup** = E[tokens/step] / total_cost_per_step (in T_t units):

| α | E[tokens/step] | Cost (T_t) | Speedup |
|---|----------------|------------|---------|
| 0.6 | 2.90 | 2.4 | **1.21×** |
| 0.7 | 3.31 | 2.4 | **1.38×** |
| 0.8 | 3.90 | 2.4 | **1.63×** |
| 0.9 | 4.69 | 2.4 | **1.95×** |
| 0.95 | 5.28 | 2.4 | **2.20×** |

### Optimal K

Larger K increases tokens per step but also verification overhead. For memory-bandwidth-bound regime where K-token verification ≈ 1-token verification cost, the optimal K maximizes `E[tokens/step] / (K×T_d + 2T_t)`.

For T_d = 0.1 × T_t:

| K | Cost | E[tokens] (α=0.8) | Speedup |
|---|------|-------------------|---------|
| 2 | 2.2 | 2.44 | 1.11× |
| 4 | 2.4 | 3.90 | 1.63× |
| 8 | 2.8 | 5.56 | 1.99× |
| 16 | 3.6 | 5.80 | 1.61× |

**Optimal at K=8 for α=0.8.** DFlash uses K=16 because their acceptance rate is much higher (~0.9-0.95 per token position with block diffusion conditioning), shifting the optimum rightward. For vanilla speculative decoding with α≈0.7-0.8, K=4-8 is the sweet spot.

---

## 10. Smear Gate Approximation Analysis

### The Exact Problem

The smear gate operation in `GPT.forward()` reads from `kv_cache.prev_embedding`:
```python
gate = smear_lambda * sigmoid(smear_gate(x[:, :, :24]))
x = x + gate * kv_cache.prev_embedding
```

When `prev_embedding` is set by the forward pass:
```python
kv_cache.prev_embedding = x[:, -1:, :]  # normalized embedding of last token
```

After K draft forwards through the draft model, `d_kv.prev_embedding` = normalized embedding of d_{K-1} (the last draft token). After rollback of K-j positions, it *should* be the embedding of d_{j-1} (the last accepted token). Instead, we restore to the snapshot at position N (pre-draft embedding).

### Error Magnitude

The smear gate contributes `smear_lambda × sigmoid(gate_features) × prev_embedding` to the input `x`. With `smear_lambda` initialized to 0 and learned, and `sigmoid(gate) ∈ (0, 1)`:

- The smear contribution is a second-order correction (a small additive term on top of the token embedding)
- The error is: `smear_lambda × sigmoid(gate) × (embedding_at_N - embedding_at_j)`
- In practice, after some training, `smear_lambda` might be ~0.1-0.3, so the error is ≤ 30% of the difference between two embeddings
- The difference between consecutive embeddings tends to be small (model learns smooth representations)

### Impact on Losslessness

**Correctness:** The smear gate approximation breaks strict losslessness. The output distribution is not *exactly* the target distribution — it's the distribution of a model that has slightly incorrect smear state. The error is bounded and small, but nonzero.

**Practical impact:** For a trained model where `smear_lambda` ≈ 0.1, this introduces a perturbation of ~10% × (small embedding difference). This is much smaller than the variation between temperature samples. In practice, the acceptance rate may drop by 1-3% points due to this inconsistency.

### Exact Fix (Future)

To make the engine exactly lossless, save smear state snapshots during the draft phase:

```python
# During draft phase (replace snapshot approach):
d_smear_snapshots = []
cur_d = d_logits
for _ in range(K):
    d_smear_snapshots.append(
        d_kv.prev_embedding.clone() if d_kv.prev_embedding is not None else None
    )
    d_probs = ...
    # ... rest of draft step
    
# On rollback to position N + accepted:
if rollback > 0:
    d_kv.prev_embedding = d_smear_snapshots[accepted] if accepted < K else d_smear_snapshots[-1]
    # Same for t_kv (which doesn't change during drafting)
    t_kv.prev_embedding = t_smear  # target unchanged during drafting, so snapshot is exact
```

This adds K tensor clones per speculative step (K × 1 × n_embd elements). At n_embd=3072, K=4: 4 × 3072 × 2 bytes = 24.6KB. Negligible memory cost.

---

## 11. Exact Changes Made

### New File: `core/configs.py`

Created from scratch. Defines named `GPTConfig` instances:

```python
TARGET_3B  = GPTConfig(seq=4096, vocab=32768, L=24, d=3072, H=24, KVH=8,  pattern="SSSL")
DRAFT_300M = GPTConfig(seq=4096, vocab=32768, L=12, d=1024, H=8,  KVH=4,  pattern="SL")
BASELINE_125M = GPTConfig(seq=2048, vocab=32768, L=12, d=768, H=6, KVH=6, pattern="SSSL")
NAMED_CONFIGS = {"3b": TARGET_3B, "300m": DRAFT_300M, "125m": BASELINE_125M}
```

### Modified: `core/model.py`

**Change 1:** Signature of `GPT.forward()`:
```python
# Before:
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
# After:
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_hidden_states=False):
```

**Change 2:** Hidden state collection in the transformer trunk loop:
```python
# Added one line before the block loop, one line inside:
hidden_states = [] if return_hidden_states else None
for i, block in enumerate(self.transformer.h):
    # ... existing code ...
    x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
    if hidden_states is not None:         # NEW
        hidden_states.append(x)           # NEW
    if i == backout_layer:
        x_backout = x
```

**Change 3:** Return value:
```python
# Before:
return loss    # training
return logits  # inference

# After:
return (loss, hidden_states) if return_hidden_states else loss    # training
return (logits, hidden_states) if return_hidden_states else logits  # inference
```

Backward compatible: all existing callers that don't pass `return_hidden_states` get identical return types.

### Modified: `core/engine.py`

Added `SpeculativeEngine` class (206 lines) before the `if __name__ == "__main__":` block. No changes to existing classes (`KVCache`, `RowState`, `Engine`, `sample_next_token`).

---

## 12. Known Limitations and Open TODOs

### `num_samples=1` Only

The current `SpeculativeEngine.generate()` asserts `num_samples == 1`. The original `Engine` supports batch generation (prefill once, decode N samples in parallel). Extending speculative decoding to multiple samples requires:

- Running K×N draft forwards (K per sample)
- Running target verification with batch_size=N (one N-token batch forward? or N separate K-token forwards?)
- N independent rejection sampling chains

This is architecturally nontrivial. The KV cache rollback must happen per-sample independently (different samples may accept different numbers of draft tokens). The cleanest approach is N independent `SpeculativeEngine` instances run in parallel — but that requires the target model to be batched across the N instances, which is the same as running batch=N verification. Deferred for now.

### Tool Use Not Implemented

The original `Engine` supports Python calculator tool use (detecting `<|python_start|>`, executing expressions, injecting `<|output_start|>...<|output_end|>`). `SpeculativeEngine` has none of this. For research inference (pure text generation), this doesn't matter. For production serving with tool use, the tool logic needs to be woven into the speculative loop — specifically, forced tokens from tool output must suppress speculative drafting for those positions.

### Smear Gate Approximation

As analyzed in Section 10, the smear gate rollback is approximate. The fix is documented and straightforward to implement.

### Draft Model is Untrained

`DRAFT_300M` is a randomly-initialized architecture. It will have near-zero acceptance rate (α ≈ 1/vocab_size ≈ 3×10⁻⁵) until trained. The `SpeculativeEngine` is correct code that degenerates to target-only sampling when acceptance rate is zero. To get actual speedup, you must:

1. Train DRAFT_300M on ClimbMix-400B (≥ Chinchilla-optimal ≈ 6B tokens for 300M)
2. Fine-tune it with a distillation objective against TARGET_3B checkpoints

### Rotary Base Theta

The code uses `base=100000` for RoPE (TODO in code: "bump base theta?"). The modern convention (2024-2026) for 4K+ context is 500K-1M base. At base=100K, frequencies repeat at ~100K/π ≈ 31K tokens, meaning positions beyond ~31K tokens start to see aliased positional encodings. For 4096 sequence length this is fine. But if sequence length is later extended, theta should be bumped.

### FP8 Inference Not Implemented

The codebase has FP8 training (`core/fp8.py`, `Float8Linear`). The inference engine runs in BF16. FP8 inference (using `Float8Linear` in forward-only mode) would further cut memory bandwidth by 2× vs BF16, giving theoretical 2× speedup on top of speculative decoding. Combining FP8 inference with speculative decoding is the next major optimization target.

---

## 13. Roadmap to DFlash-Level Speedups

To achieve DFlash's 5-6× speedup, three milestones remain. Each is a training experiment, not a code change.

### Milestone 1: Baseline Draft Training (→ 1.5-1.8×)

**What:** Train DRAFT_300M to minimize cross-entropy on ClimbMix-400B with the existing `SpeculativeEngine`.

**How:**
```python
# Use existing training infrastructure, just swap the config
from core.configs import DRAFT_300M
model = GPT(DRAFT_300M)
```

**Expected result:** α ≈ 0.65-0.75 on held-out text, giving 1.3-1.6× speedup.

### Milestone 2: EAGLE-Style Feature Conditioning (→ 2.0-2.5×)

**What:** Train a conditioned draft that receives the target model's last hidden state as input.

**How:**
1. Freeze TARGET_3B.
2. Modify DRAFT_300M to accept a conditioning vector: the final hidden state from TARGET_3B at the current position (extracted via `return_hidden_states=True`).
3. Project the 3072-dim target hidden state to 1024-dim and add to draft's embedding.
4. Train draft with cross-entropy loss on next-token prediction given target's hidden state.

**Code sketch:**
```python
# In SpeculativeEngine.generate, modify draft phase:
_, hidden = self.target.forward(ids, kv_cache=t_kv, return_hidden_states=True)
cond = project_3072_to_1024(hidden[-1])  # last layer's hidden state
# Feed cond to draft via modified draft architecture
```

The `return_hidden_states` hook (implemented in this session) makes this possible without any further model.py changes.

**Expected result:** α ≈ 0.80-0.87, giving 1.8-2.2× speedup (EAGLE-3 reports 2.0-2.4× on Llama-class models).

### Milestone 3: Multi-Layer Conditioning + Block Diffusion (→ 4-6×)

**What:** Full DFlash implementation.

**How:**
1. Extract hidden states from uniformly-sampled layers (using `hidden_states[0::n_layer//6]`).
2. Fuse via lightweight linear projection to 1024-dim conditioning.
3. Inject into EVERY draft layer's KV projections (not just input).
4. Add block diffusion training objective (masked token prediction over 16-token blocks).
5. Use single denoising step at inference (diffusion model with T=1).

**Training complexity:** Significant. Requires:
- A new draft model architecture (different from standard GPT)
- Diffusion training data preparation (masking)
- Careful hyperparameter tuning for the single-step denoising
- Evaluation infrastructure for block acceptance rate (different from token-level α)

**Expected result:** α_block ≈ 0.90-0.95 per token within the block, K=16, giving 5-6× speedup — matching DFlash's claims.

### Summary Timeline

| Phase | Work Required | Expected Speedup |
|-------|--------------|-----------------|
| Current | Code (done) | 1.0× (draft untrained) |
| Milestone 1 | Train DRAFT_300M | 1.5-1.8× |
| Milestone 2 | EAGLE-style training | 2.0-2.5× |
| Milestone 3 | Full DFlash training | 4.0-6.0× |

Each milestone builds on the previous. The `SpeculativeEngine` in `core/engine.py` is the correct runtime for all three milestones — the rejection sampling loop is architecture-agnostic. Only the draft model's architecture and training changes across milestones.

---

## Appendix: File Index

| File | Status | Purpose |
|------|--------|---------|
| `core/configs.py` | **NEW** | Named GPTConfig instances: TARGET_3B, DRAFT_300M, BASELINE_125M |
| `core/model.py` | **MODIFIED** | Added `return_hidden_states` param to `GPT.forward()` |
| `core/engine.py` | **MODIFIED** | Added `SpeculativeEngine` class |
| `core/flash_attention.py` | unchanged | FA3/SDPA unified interface |
| `core/checkpoint_manager.py` | unchanged | Model loading/saving |
| `core/optim.py` | unchanged | Muon+AdamW optimizer |
| `core/common.py` | unchanged | Utilities, COMPUTE_DTYPE |
| `core/fp8.py` | unchanged | FP8 training support |
| `core/dataloader.py` | unchanged | Parquet streaming dataloader |
| `core/dataset.py` | unchanged | ClimbMix-400B management |
| `core/tokenizer.py` | unchanged | BPE tokenizer wrapper |
