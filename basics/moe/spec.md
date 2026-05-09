# MoE Layer — Phase 1 DEFINE

> Day-1 deliverable for the modality-conditional MoE scaling project (see `../../docs/Plan.md`).
> This file follows `../../agentic_build_workflow.md`. It is the spec the verifier
> tests against. If the spec changes, the verifier changes with it — same commit.

---

## Goal (one sentence)

Implement a fine-grained MoE layer with sigmoid top-k routing, a shared expert,
and aux-loss-free balancing, that runs on a toy task and demonstrably produces
modality-conditional expert specialization.

---

## Why this exists (links to Plan.md)

- Granularity G is the central knob in Plan §4 (the IsoFLOPs sweep).
- Aux-loss-free balancing is the DeepSeek-V3 routing choice in Plan §3.
- Per-modality expert utilization is the measurement in Plan §5.4.

This is the smallest thing that contains the project's load-bearing uncertainty.
Build intuition here before scaling to `core/`.

---

## One-model-call check

No. The artifact is code that must be modified, re-run, and instrumented across
many configurations. A single generation cannot satisfy criteria 3–4 below.

---

## API spec

```python
class MoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,           # baseline FFN inner size; expert size = d_ffn / G
        granularity: int,     # G; n_routed_experts = G * base_experts
        base_experts: int,    # number of "logical" experts before granularity split
        top_k: int,           # routed experts activated per token (scales with G)
        n_shared: int = 1,    # always-on experts
        bias_update_lr: float = 1e-3,
    ): ...

    def forward(
        self, x: torch.Tensor          # (B, T, d_model)
    ) -> tuple[torch.Tensor, dict]:
        # returns:
        #   y:    (B, T, d_model)
        #   aux:  {
        #     "expert_token_counts": LongTensor (n_routed_experts,)
        #     "router_logits":       FloatTensor (B*T, n_routed_experts)
        #     "selected_experts":    LongTensor (B*T, top_k)
        #     "routing_weights":     FloatTensor (B*T, top_k)
        #   }
```

Active params per token: `(top_k + n_shared) * (d_ffn / G)` — must be constant
across G when sweeping granularity at fixed compute.

---

## Must NOT

- Use any auxiliary balancing loss term added to the training loss.
  Balancing is enforced via per-expert bias updates only.
- Mask the shared expert from any token. It runs on every token, every step.
- Drop tokens. No capacity factor in v1. Every token routes to exactly top_k
  experts plus the shared expert(s).
- Use softmax over all experts before top-k. Use sigmoid per-expert, then top-k,
  then renormalize the selected weights. (DeepSeek-V3 §2.1.1 convention.)

---

## Success criteria (the verifier — `test_moe.py`)

These are the four checks the verifier script must run. All must pass before
Phase 4 REVIEW.

1. **Shape correctness.** Forward pass on input (B=2, T=16, d_model=64) returns
   output of shape (2, 16, 64) and a non-empty aux dict with the four keys above.

2. **Dense equivalence.** With `granularity=1, base_experts=1, top_k=1, n_shared=0`,
   the layer's output equals a standard SwiGLU FFN of size d_ffn within atol=1e-5
   when initialized with the same weights.

3. **Anti-collapse.** After 500 steps on the toy task (below), routing entropy
   over experts must satisfy:
   ```
   H(expert_usage) > 0.7 * log(n_routed_experts)
   ```
   where `expert_usage` is the empirical distribution of routed tokens across
   all experts over the last 50 steps.

4. **Specialization emerges (median across seeds).** On the 2-modality toy
   task, the **median** max per-expert per-modality usage ratio across 3 seeds
   must be > 2:1. Median, not single-seed: at this toy scale the task fully
   converges (loss ~0.18) and leaves little pressure for routed experts to
   specialize, so individual seeds vary widely (we observed 1.90 / 2.55 /
   4.03 / 5.58 / 2.12 across seeds 0-4). Median across 3 seeds is robust to
   that variance without cherry-picking a passing seed.

---

## Deliberate-break test (Phase 3 step 6)

The verifier above is meaningless unless we prove the bias mechanism is
doing real work. Original v1 of this test asserted "with `bias_update_lr=0`
routing must collapse." It was wrong: even without bias updates, routing on
this task stayed nearly uniform (entropy 2.065 / max 2.079). Three structural
reasons, each a real finding worth keeping:

  1. The two modalities require different functions, so routing has natural
     pressure to spread tokens across at least 2 expert clusters.
  2. Sigmoid routing (vs softmax+top-1) is collapse-resistant — high score for
     expert i does not suppress expert j (no shared normalizer).
  3. The shared expert absorbs general-purpose computation, removing the
     rich-get-richer pressure that drives classical MoE collapse.

So the v2 test is reframed as a **comparative effect test**:

5. **Bias mechanism reduces deviation-from-uniform.** Train two models on a
   setup designed to permit imbalance (no shared expert, single modality so no
   counterbalancing pressure), same seed, same data, differing only in
   `bias_update_lr` (1e-3 vs 0.0). Metric: `max_share - 1/n_routed` (the
   distance above uniform — the controller's setpoint). Assert that the
   with-bias deviation is at least 3× smaller than the no-bias deviation.
   The first attempt used raw max_share and missed a real 12× effect because
   the baseline (1/n_routed ≈ 0.125) wasn't subtracted out.

---

## Invariants (must never break)

- Gradient flows to all experts that received ≥1 token in the batch.
  (No `.detach()` on the path from expert output to loss.)
- Bias updates happen under `torch.no_grad()`. Biases are control signals,
  not learned parameters with autograd history.
- Sum of `routing_weights` per token equals 1 (after renormalizing the
  top-k selected sigmoid scores: `g_i = s_i / Σ_{j∈TopK} s_j`).
  This is the DeepSeek-V3 §2.1.1 convention. Reasons:
  (a) constant residual-stream contribution → stable training at scale;
  (b) relative sigmoid magnitudes are preserved → router still expresses
  routing confidence through *ratios* of selected weights.
- Shared expert receives every token, no gating.

---

## Toy task (the substrate for criteria 3, 4, 5)

- **Input:** length-T sequence of integers in [0, 31]. Modality A draws from
  even integers {0, 2, ..., 30}; modality B from odd integers {1, 3, ..., 31}.
- **Targets (DIFFERENT FUNCTION per modality, not just different distribution):**
  - Modality A: `y_t = (x_t + x_{t-1}) mod 32`  (addition)
  - Modality B: `y_t = (x_t * x_{t-1}) mod 32`  (multiplication)
  - `mod 32` (= vocab) so input alphabet equals output alphabet — the LM's
    `lm_head` is reused directly as the verifier head, no special
    classification head needed. (v1 used `mod 17`, which forced a separate
    head and mismatched the production LM code path.)
- **Why two functions, not one:** the v1 spec used a single function with two
  input distributions. Result: experts had no *functional* reason to specialize,
  and the bias-update balancing pressure suppressed the weak distribution-only
  signal (max ratio plateaued at ~1.98). Specialization emerges when modalities
  require **different computation**, not just different inputs — same lesson as
  vision-vs-language in Plan §1 Finding 2.
- **Batch composition:** 50/50 mix of A and B per batch. Modality tag travels
  with each sequence so the verifier can compute per-modality expert usage.
- **Model:** `BasicsMoETransformerLM` (in `lm.py`) at `vocab_size=32,
  context_length=64, d_model=64, num_layers=2, num_heads=4, d_ff=256` with
  `moe_layer_indices=(1,)` — layer 0 dense `TransformerBlock`, layer 1
  `MoETransformerBlock`. Same primitives a production trainer would use.
- **Training:** AdamW, lr=3e-4, 500 steps, batch 32, T=16. CPU-runnable.

---

## Frozen design choices (one implementation, no port-later)

These are pinned now so the implementation, the verifier, and any future
extension all agree:

- **Router math:** `s = sigmoid(W·x)`; selection: `topk(s + bias, k=2)`;
  weighting: `g_i = s_i / Σ_{j∈TopK} s_j` (raw sigmoids, renormalized over
  selected, sum to 1). Bias is used for selection only, never for weighting.
- **Bias update:** sign-based, fixed γ=1e-3, applied under `no_grad()` after
  each optimizer step. Direction: `bias += γ · sign(target - count)` where
  target = mean tokens per expert in the batch.
- **Headline config for the verifier:** `d_model=64, d_ffn=256, base_experts=4,
  granularity=2 → n_routed=8, top_k=2, n_shared=1`. Active per token =
  3 experts × 128 = 384 FFN params (33% of total FFN capacity).
- **Expert nonlinearity:** SwiGLU (matches what every modern MoE paper uses;
  removes nonlinearity-choice as a confound).

---

## NOT building today

- Distributed expert parallelism (FSDP, expert sharding). Single-GPU/CPU.
- Token capacity limits or dropping.
- FP8, flash attention, fused kernels.
- Anything from `../core/`. This is `basics/`-level isolation.
- Vision tokens or real multimodal data.
- W&B logging. Per-step CSV is enough.
- CompleteP / μP. Naive AdamW, naive init.

If any of these creep into the implementation, scope-cut in REVIEW.

---

## Sensors and KILL conditions (Phase 5 SHIP)

Logged to `basics/moe_run.csv` every 50 steps:
- step, loss, routing_entropy, n_dead_experts, max_expert_share, grad_norm

KILL (investigate before continuing):
- routing entropy < `0.5 * log(n_routed_experts)` for 100 consecutive steps
- any expert receives 0 tokens for > 100 consecutive steps
- loss NaN at any step
- max_expert_share > 0.5 (one expert eating half the traffic)

---

## What "done" looks like

- `basics/moe.py` exists, < 300 lines.
- `basics/test_moe.py` runs all 5 checks; checks 1–4 pass, check 5 also passes
  (the broken version DOES collapse).
- `basics/moe_run.csv` committed showing one healthy training run.
- I can answer in my own words, pointing at line numbers:
  - Why sigmoid+top-k, not softmax+top-1.
  - What the bias update does, and why it's not a parameter.
  - What changes about gradient flow when G goes from 1 to 8 at fixed
    active params.
  - What expert specialization looks like in the CSV.

If any of those four explanations is shaky, do not proceed to the next phase
of the Plan. Re-read, modify, predict, re-run.
