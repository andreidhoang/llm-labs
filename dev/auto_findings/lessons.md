# Lessons across autoresearch sessions

Durable findings the next session's agent should read **before** proposing
experiments. If you propose something listed here as `DO NOT RETRY` without a
specific reason it would differ this time, you're wasting compute.

Referenced from `auto/program.md` § "Prior session findings."

## Format

```
### YYYY-MM-DD (session auto/YYYY-MM-DD)
- Tried: <one-line description of the change>
- Result: <val_bpb delta vs baseline; or "crash" or "no signal">
- Lesson: <what we now believe>
- Status: KEEP | RETRY-WITH-CONDITIONS | DO-NOT-RETRY
- Optional: bears on H_X (Tier 2 hypothesis) → see dev/sweep_design.md
```

## Entries

### 2026-05-18 (session auto/2026-05-18)

- LESSON: WARMUP_RATIO=0 at d=8 hurts (+0.025). Keep ≥0.05 across small models.
  Status: DO-NOT-RETRY at d=8 with MATRIX_LR≥0.04.
- LESSON: TOTAL_BATCH_SIZE=2^19 overshoots at d=8 (+0.037 regression). 2^18 near-optimal.
  Status: DO-NOT-RETRY at d=8. [bears on H_3]
- LESSON: MATRIX_LR=0.06 marginally regresses from 0.04 at d=8 (+0.002).
  Status: 0.04 is near-optimal for d=8 dense; testing 0.05 may find slight improvement.
- LESSON: WD=0.2, EMBED=0.8, β1=0.8, SSSL window each produce small marginal wins
  (~0.001 each) on top of session 1 winner. HP space approaching saturation in
  this direction; next session should escalate to architectural variants.
- LESSON: Cross-session replication of session 1 iter4 was exact (Δ=0.0006).
  System is reproducible to noise floor.
- LESSON: Cross-session best so far = 1.0576 (iter9 of session 2). Down 5.7% from
  Karpathy's published 0.998 baseline; gap explained by FA2 vs FA3.

### 2026-05-19 (session auto/2026-05-19 — first core/ edits)

- LESSON: QK-norm in core/model.py:CausalSelfAttention.forward is LOAD-BEARING
  at d=8 (+0.017 val_bpb regression when removed). Verified Karpathy's nanochat
  choice is not arbitrary. Status: DO-NOT-RETRY-removing.
- LESSON: softcap=15 in core/model.py:GPT.forward is LOAD-BEARING
  (+0.005 regression when relaxed to 30). Status: DO-NOT-RETRY-relaxing.
- LESSON: init scheme Uniform vs Normal (same std) is NOISE-FLOOR at d=8.
  Either works; Karpathy's preference for Uniform "to avoid outliers" not
  empirically meaningful at this scale.
- LESSON: RoPE base 10K vs 100K is NOISE-FLOOR at seq_len=2048. Would matter
  at seq_len>4096. Skip for d=8 work.
- LESSON: FA3 hub kernel path (core/flash_attention.py) is currently DEAD —
  varunneal/flash-attention-3 repo 401's; kernels-community alternatives ABI-
  mismatch on NGC torch 2.7.0a0+nv25.03. Punt to human: either find replacement
  kernel or accept FA2 + delete dead code path.
- LESSON: PLATEAU CONFIRMED at val_bpb ≈ 1.058 across 3 sessions, 22 experiments,
  $13.20. HP space + nearby arch space EXHAUSTED at d=8/5min/FA2. Stop running
  Tier 1 in this configuration — ship to Tier 2 OR escalate (FA3 fix, bigger
  arch shifts).

### 2026-05-20 (session auto/2026-05-20 — FA3 attempt + FP8)

- LESSON: FA3 build from source on NGC pytorch:25.03-py3 takes 60-90 min
  (CPU-bound, 100+ sm_90 kernel specializations). NOT viable for Tier 1
  per-session install. Need pre-built Docker image OR wait for Dao-AILab
  pre-built wheels matching nv25.03 ABI. varunneal/flash-attention-3 hub
  kernel returns 401; kernels-community alternatives ABI-mismatch.
- LESSON: FP8 via core.fp8.convert_to_float8_training HURTS at d=8/50M params
  (+0.006 val_bpb regression; MFU goes DOWN). Per-step quant/dequant overhead
  > compute savings at small scale.
- LESSON: FP8 starts helping at d=10/80M params (MFU 17% vs 11% BF16). At
  d=12/120M, MFU reaches 22% but model undertrained at 5min budget.
  [bears on Tier 2 H₀: FP8-on-shared-expert speedup should be real at d=24].
- LESSON: Inter-host throughput variance on Vast.ai 1×H200 is ~25-35%. Same
  GPU spec, same image, same code → different physical machine = different
  tokens/sec. Validates sweep_design.md single-instance discipline (§11.1).
  Implication: absolute val_bpb across sessions is contaminated; use
  within-session deltas only.

### 2026-05-19 (multimodal MoE production smoke — engineering verification)

- LESSON: scripts/base_train.py --multimodal path is VERIFIED on real
  hardware. MoE-on + frozen SigLIP2-SO400M + 3D MRoPE + scatter + per-modality
  loss decomposition all work together at d=8/H200/MFU=40%. mm_bpb dropped
  3.15 → 1.91 in 25 steps. Tier 2 sweep_design.md v3 production path is ready.
- LESSON: bpb_vision_ctx < bpb_text consistently across 3 evals → vision
  context HELPS next-token prediction (correct direction). If scatter or 3D
  MRoPE were broken, would expect the opposite.
- LESSON: r_actual is the FRACTION of text tokens with vision in their
  attention window, NOT the vision pad-token fraction. At mix_ratio=0.1
  (pad-token fraction = 0.1), r_actual = 0.885 because most text tokens have
  at least one vision token within seq_len=2048. Don't conflate the two.
- LESSON: Multimodal step time is dominated by SigLIP2 forward. At mix_ratio
  =0.3 (~1200 images/step at B=32 seq=2048): ~44s/step. At mix_ratio=0.1:
  ~20s/step. Plan Tier 2 wall-clock budgets accordingly.
- LESSON: torch.compile has a graph break in core/multimodal.py:scatter_vision
  _features around line 492 — data-dependent branch can't be traced. Training
  continues with partial compilation. Potential 5-10% Tier 2 speedup if
  refactored.
- LESSON: Multimodal first-step compile overhead is ~128 sec on H200 (vs ~20
  sec steady-state). Variable image counts per batch make tracing slower.
  Negligible at Tier 2's 2-hour budgets; problematic for autoresearch-style
  5-min budgets.
