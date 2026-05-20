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

### 2026-05-19 (FA3 ecosystem diagnostic — confirmed blocked)

- LESSON: Newer NGC image (25.06 with torch 2.8.0a0+nv25.06) STILL ships only
  FA2 (flash_attn 2.7.4.post1). The requirements.txt comment "NGC pytorch
  images provide FA3" was an incorrect assumption by the original author.
  Multiple NGC versions (25.03, 25.06) confirmed FA2-only.
- LESSON: kernels-community/flash-attn3 hub kernel uploads metadata.json in
  a format that requires `kernels` package >=0.15. Latest on PyPI is 0.14.1
  (kernels 0.15 unreleased as of May 2026). All HF Hub FA3 paths blocked
  until kernels 0.15 ships.
- LESSON: The hub kernels are prebuilt for "torch28-cxx11-cu129-x86_64-linux"
  ABI. NGC's nv25.06 build (torch 2.8.0a0+nv25.06 with CUDA 12.8) may have
  minor ABI differences even if metadata parsing worked.
- LESSON: Three FA3 paths all blocked in current ecosystem (May 2026):
  (1) varunneal/flash-attention-3 hub: 401 (repo gone),
  (2) kernels-community/flash-attn3 hub: metadata parse error,
  (3) From-source `flash-attention/hopper/setup.py install`: 60-90 min CPU compile.
  Practical conclusion: ACCEPT FA2 for Tier 1 + Tier 2 work, OR commit to
  one-time Docker image build with FA3 source-built (~1 hr engineering).
  Cost: $1 of diagnostic time on NGC 25.06 H200 confirmed this.

### 2026-05-19 (FA3 ecosystem — UNBLOCKED via nanochat-exact setup)

- LESSON CORRECTION (supersedes the prior "FA3 blocked" lesson): the earlier
  401 and metadata-parse failures were specific to NGC custom torch ABI
  (torch 2.7.0a0+nv25.03, 2.8.0a0+nv25.06). Karpathy's nanochat uses VANILLA
  PyPI torch 2.9.1+cu128 installed via `uv sync --extra gpu` from
  https://download.pytorch.org/whl/cu128 — NOT NGC images. With vanilla
  torch + kernels==0.11.7, `kernels.get_kernel('varunneal/flash-attention-3')`
  loads cleanly and provides FA3. ABI requirement: torch29-cxx11-cu128-x86_64-linux.
- LESSON: Replicating nanochat's exact env (Ubuntu 22.04 + pytorch/pytorch:
  2.8.0-cuda12.8-cudnn9-runtime + apt-get build-essential nvidia-cuda-toolkit
  python3-dev + uv sync nanochat's pyproject.toml) → FA3 works.
- LESSON (surprising): FA3 ALONE does NOT close the predicted 62% of gap to
  Karpathy's 0.998 at d=8. Measured at session 3 winner config:
    - FA2 + torch.compile + grouped_mm (session 3): 195M tokens, val_bpb=1.058
    - FA3 + torch.compile + for-loop MoE     (FA3 run): 187M tokens, val_bpb=1.060
  Roughly equivalent. The Chinchilla-based prediction assumed FA3 = 2x
  throughput, but at d=8 attention is a small fraction of total compute —
  FA3's attention speedup doesn't translate to model-level 2x. The 5.7% gap
  to Karpathy is NOT primarily FA3.
- LESSON: core/moe.py:_run_experts_grouped_mm has an eager-mode bug
  (line 122, `x_bf16.new_zeros(T_padded, D)` where T_padded is a Tensor
  not int). torch.compile masks this by tensor->int conversion in tracing.
  Workaround for eager runs: DISABLE_GROUPED_MM=1 (for-loop path).
- LESSON: To use vanilla pytorch image on Vast.ai, need to manually install
  build-essential + nvidia-cuda-toolkit + python3-dev for torch.compile to
  work. NGC ships these by default. Choose your tradeoff: NGC = bigger
  image + custom torch ABI vs vanilla = smaller + works with hub kernels.

### 2026-05-20 (session 2026-05-20-A0-attempt — Vast 8×H200 + NGC pytorch:25.03-py3)

Six independent provisioning issues surfaced during the A0 launch. All fixed,
either in code (committed to `sweep_runner.py`) or in `H100_RUNBOOK.md`
"Provisioning gotchas (2026-05-20)" section. Full incident report in
`dev/auto_findings/2026-05-20-A0-attempt/findings.md`. Quick reference:

- LESSON: NGC pytorch:25.03-py3 build hash `7c8ec84dab.nv25.03` does NOT ship
  `torch._grouped_mm` (verified at ATen op level). The image tag is the same
  as builds that do. Probe before relying on the runbook's "25.03 always has
  it" claim. Status: DO-NOT-TRUST-NGC-25.03-BLINDLY.
- LESSON: `pip install -r requirements.txt` on NGC overwrites
  `pytorch-triton 3.2.0+nvinternal` with vanilla `triton 3.7.0` from PyPI,
  breaking `from triton.compiler.compiler import triton_key` that NGC's torch
  imports. Fix at install time: `grep -v -E "^triton" requirements.txt`.
  Workaround at runtime: `TORCHDYNAMO_DISABLE=1`.
- LESSON: `torchrun --run=X scripts/X.py --run=A0` errors with
  "ambiguous option: --run= could match --run-path". Insert `--` between
  torchrun args and the script. Now in `sweep_runner.build_torchrun_cmd`.
- LESSON: `base_train.py` does NOT accept `--warmup-frac` (only
  `--warmup-steps`, default 40). Old configs carrying `warmup_frac: 0.05`
  must not be propagated to the CLI. Removed from `sweep_runner`.
- LESSON: `base_train.py --run=A0` (non-"dummy") triggers `wandb.init()`
  which fails without `WANDB_API_KEY`. `sweep_runner` now defaults
  `WANDB_MODE=offline` in the launch env. For direct torchrun, export it.
- LESSON: Default `--device-batch-size=32 max_seq_len=4096` OOMs on H200
  (140 GB) at d=24 in eager mode (no torch.compile). Use `--device-batch-size=16
  --activation-ckpt`. Trade is required without `torch.compile`.
- LESSON: At 19.7% MFU (measured, this session, eager + activation-ckpt +
  for-loop MoE on 8×H200), the v3 scaling-law study budget projects to
  ~$650 instead of the spec's ~$216. Either restore a working torch stack
  (vanilla pytorch image + matched triton) OR re-budget. The spec's 35-47%
  MFU assumption is brittle — it requires both `_grouped_mm` AND
  `torch.compile` working, which is not guaranteed by image tag alone.
- LESSON: `base_train.py` requires the nanochat tokenizer at
  `/root/.cache/nanochat/tokenizer/tokenizer.pkl` (40 KB, scp from local Mac)
  and ClimbMix shards at `/root/.cache/nanochat/base_data_climbmix/` (~370 MB
  each, fetched via `python -m core.dataset -n N`). Pre-stage both before
  launch. Status: encoded in `H100_RUNBOOK.md` "Provisioning gotchas" §5.

### 2026-05-20 (later — image search + bincount bug fix)

- LESSON: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` (Docker Hub, public
  PyPI release) is the verified production image. Ships
  `torch._grouped_mm` + triton 3.4.0 with `triton_key` symbol + working
  `torch.compile`. Status: USE THIS as the default Vast image.
- LESSON: `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel` does NOT have
  `torch._grouped_mm` — only the FP8 variant `_scaled_grouped_mm`. PyTorch
  2.7 release did not include the BF16 grouped matmul kernel; 2.8 did.
  Status: DO-NOT-USE for MoE training.
- LESSON: `core/moe.py:56` used `torch.histc` to count expert assignments,
  which returns float32. That dtype propagated into `target_idx` (via
  `torch.arange(T, dtype=orig_cum.dtype)`), and torch.compile's inductor
  backend rejected the float index in `index_copy_`. Fixed by switching
  to `torch.bincount` which returns int64 naturally. Commit `8fa6236`.
  This is a GENUINE BUG FIX — bincount is the correct API for counting
  integer expert IDs and is also faster. Status: DO-NOT-REVERT.
- LESSON: When picking an image, always run a 5-min probe on a cheap
  1-GPU host BEFORE committing to an 8-GPU rental. Probe script:
  `tests/probe_image.sh` (informal — copy from
  `dev/auto_findings/2026-05-20-A0-attempt/findings.md` "Working launch
  command"). Tests `_grouped_mm` + `torch.compile` + tiny `_grouped_mm`
  call. Costs ~$0.20 per probe. Saves $5-30 of failed-rental cost per
  hit. Status: encode as `scripts/probe_image.sh` next session.
