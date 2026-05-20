# 2026-05-20 — A0 launch attempt on Vast.ai 8×H200 (aborted)

**Goal:** launch A0 (dense + text-only @ d24, C=1.5e19) as the instrument-calibration cell of the v3 scaling-law study (`dev/scaling_law_v3_minimal.md`).

**Result:** A0 reached step 23/2743 (0.84%) at MFU 19.7% before user-initiated abort. Cell did not complete; no usable val_bpb. Instance destroyed at abort.

**Cost:** ~$5 for ~10 min of compute on 8×H200 @ $30.45/hr.

**Headline:** the spec's MFU estimate (35–47%) was 1.75–2.4× too high for this hardware + image configuration. At measured 19.7% MFU, A0's projected wall-clock was 2.65 hr (vs spec 1.75 hr), and projected cost was $81 (vs spec $25). The full 5-cell v3 study would project to ~$500 rather than ~$216.

Cause is a stack of independent environment issues, all surfaced during launch. Each is fixed in code or runbook; none are blockers individually, but together they erode the iso-FLOP assumptions in the spec.

## The 6 issues, each surfaced once during launch

### Issue 1 — `torch._grouped_mm` missing from NGC pytorch:25.03-py3 build `7c8ec84dab.nv25.03`

| | |
|---|---|
| Symptom | `assert hasattr(torch, '_grouped_mm')` in `scripts/provision_h100.sh` fails on a 25.03 image that the H100_RUNBOOK said was guaranteed to have it. |
| Probe | At ATen op level: `[x for x in dir(torch.ops.aten) if 'group' in x.lower()]` returns only `native_group_norm`. No `_grouped_mm` or `_scaled_grouped_mm`. |
| Impact for A0 | None. A0 has `num_experts=1` → for-loop fallback iterates once = mathematically equivalent to a dense MLP. |
| Impact for C1/C_mid/C2 | Significant. G=2 MoE with 8 experts means 8 Python-launched matmuls per layer per step. MFU regression beyond the spec's 35% MoE budget. |
| Fix applied | `ALLOW_GROUPED_MM_FALLBACK=1` env var to `provision_h100.sh` to allow A0 launch. |
| Real fix (for C cells) | Either (a) find a host whose NGC image ships `_grouped_mm` (different build hash), or (b) build pytorch from source on the host, or (c) accept reduced MFU and re-budget. |

### Issue 2 — `pip install -r requirements.txt` overwrites NGC's `pytorch-triton`, breaks `torch.compile`

| | |
|---|---|
| Symptom | `BackendCompilerFailed: ImportError: cannot import name 'triton_key' from 'triton.compiler.compiler'` |
| Root cause | NGC ships `pytorch-triton 3.2.0+gitb2684bf3b.nvinternal` which provides `triton_key`. `pip install -r requirements.txt` pulls in `triton 3.7.0` from PyPI; the latter removed `triton_key`, and NGC's `torch 2.7.0a0` still imports it. |
| Fix applied | `TORCHDYNAMO_DISABLE=1` to skip `torch.compile`. Training runs eager mode. |
| Real fix | Modify `requirements.txt` to skip `triton` when running on NGC image (or pin `triton<3.7`). The requirements file is for portable local dev; NGC's torch already brings the right triton. |

### Issue 3 — `--warmup-frac=0.05` doesn't exist in `base_train.py`

| | |
|---|---|
| Symptom | `base_train.py: error: unrecognized arguments: --warmup-frac=0.05` |
| Root cause | `sweep_runner.build_torchrun_cmd` translated config field `warmup_frac` → `--warmup-frac` flag. `base_train.py` only accepts `--warmup-steps` (int). Config schema and CLI surface drifted. |
| Fix applied (committed in 1a67c8a) | Drop `--warmup-frac` from sweep_runner; rely on `base_train.py`'s default 40-step warmup. |
| Impact | Default 40 steps ≈ 16% warmup for A0's ~2743 iters — slightly longer than the spec's 5% but inside nanochat's validated regime. |

### Issue 4 — `torchrun --run=A0` is consumed as `--run-path`

| | |
|---|---|
| Symptom | `torchrun: error: ambiguous option: --run=A0 could match --run-path, --run_path` |
| Root cause | `torchrun`'s argparse prefix-matches its own `--run-path` against the training script's `--run`. No `--` separator between torchrun args and training-script args. |
| Fix applied (committed in 1a67c8a) | Insert `--` in `sweep_runner.build_torchrun_cmd` between `--nproc-per-node=8` and `scripts/base_train.py`. |

### Issue 5 — `wandb.init` fails without `WANDB_API_KEY`

| | |
|---|---|
| Symptom | `wandb.errors.errors.UsageError: No API key configured. Use 'wandb login' to log in.` |
| Root cause | `--run=A0` (i.e., not the literal "dummy") triggers `wandb.init(project="nanochat", name=args.run, ...)` in `base_train.py`. Fresh host has no key. |
| Fix applied (committed in 1a67c8a) | `sweep_runner.submit` sets `WANDB_MODE=offline` in the launch env by default. Override by setting `WANDB_MODE=online` in the parent. |

### Issue 6 — `--device-batch-size=32` OOMs at d24/seq=4096 on H200 (140 GB) in eager mode

| | |
|---|---|
| Symptom | `torch.OutOfMemoryError: Tried to allocate 384.00 MiB. GPU 0 has 264.94 MiB free.` Memory was already at 137/140 GB before the failing alloc. |
| Root cause | Default `--device-batch-size=32` × seq=4096 = 131k tokens/micro/GPU. d24 trunk activations × ckpt-disabled by default → exceeds 140 GB. The default was tuned with `torch.compile + activation-ckpt` both available; with neither, eager-mode activation memory is ~2× the spec assumption. |
| Fix applied | `--device-batch-size=16 --activation-ckpt`. Memory drops to ~30 GB per GPU. |
| Trade-off | Activation checkpointing recomputes during backward → ~33% more compute → MFU drop. Combined with eager mode (no torch.compile fusion) and the for-loop MoE fallback (irrelevant for A0 but loaded), explains the measured 19.7% MFU vs the spec's 47% target. |

## What worked once all six were applied

```bash
# Working launch (verified):
PYTHONPATH=. TORCHDYNAMO_DISABLE=1 WANDB_MODE=offline torchrun \
  --nproc-per-node=8 -- scripts/base_train.py \
  --depth=24 --head-dim=128 \
  --num-experts=1 --top-k=1 --num-shared-experts=0 \
  --max-seq-len=4096 --device-batch-size=16 --activation-ckpt \
  --matrix-lr=0.02 --embedding-lr=0.3 --unembedding-lr=0.008 \
  --scalar-lr=0.5 --weight-decay=0.28 --final-lr-frac=0.05 \
  --run=A0 --target-flops=1.5e+19 --target-param-data-ratio=12
```

Observed during the 23-step run before abort:
- Loss curve: 6.24 → 4.04 (textbook exponential descent during warmup; 23 steps; lrm ramped 0.03 → 0.60)
- GPU util: 99% on all 8 H200s
- Throughput: ~300k tok/sec total = ~37.5k tok/sec/GPU
- MFU: 19.7-20%
- ETA from internal counter: 158.8 min for full 2743 iters

## Implications for the v3 study

| Cell | Spec wall-clock | Spec cost | Measured/projected at 19.7% MFU |
|---|---|---|---|
| A0 | 1.75 hr | $25 | 2.65 hr / $81 |
| C1 | 0.8 hr | $13 | ~1.2 hr / $37 (G=2 likely worse: maybe ~10% MFU → ~2.4 hr / $72) |
| C_mid | 1.7 hr | $28 | ~2.6 hr / $78 (worse for G=2: ~5 hr / $150) |
| C2 | 3.4 hr | $54 | ~5.1 hr / $155 (worse for G=2: ~10 hr / $305) |
| V3 | 6.0 hr | $96 | ~9 hr / $275 (G=2: ~18 hr / $549) |
| **Total** | **~$216** | **at spec MFU** | **~$626 best case / ~$1,159 with G=2 fallback** |

The v3 spec budget assumed `_grouped_mm` was available and `torch.compile` worked. Neither held on this host.

## Two paths forward

### Path A — Re-provision with a host that has working `_grouped_mm` + `torch.compile`

Requires finding an NGC image with `_grouped_mm` (different build than `7c8ec84dab.nv25.03`) OR a vanilla pytorch install (e.g., `pytorch/pytorch:2.7.0-cuda12.8-cudnn8-devel`) with pinned compatible triton.

Estimated: 30-60 min to find + provision + smoke. If successful, brings spec costs back to ~$216.

### Path B — Accept reduced MFU and re-budget

Run the v3 study with `TORCHDYNAMO_DISABLE=1 + activation-ckpt + for-loop MoE`, accept ~3× cost overrun. Total ~$650-1,150.

### Recommendation

**Path A.** The cost overrun for "do nothing" is $400-900. The investment for Path A is bounded at ~1-2 hr of provisioning effort. Path A also gives the spec's intended performance characteristics (47% MFU was the calibration target, not 20%).

Specifically: try `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel` instead of NGC. Vanilla PyPI torch 2.7.0 ships `_grouped_mm` (verified by the docstring comment in `core/moe.py:9`). Triton 3.x with vanilla torch is also matched (no `triton_key` issue).

## Files touched during this session

- `scripts/sweep_runner.py` — added `--` separator, dropped `--warmup-frac`, defaulted `WANDB_MODE=offline` (committed in 1a67c8a, pushed to origin/main)
- `dev/auto_findings/2026-05-20-A0-attempt/A0_aborted.log` — salvaged training log for forensic reference (102 step lines, 1 TRAJ_POINT at step 0)
- `dev/auto_findings/2026-05-20-A0-attempt/findings.md` — this document

## Open questions for next session

1. Which NGC build hash (or vanilla pytorch image) reliably ships `_grouped_mm`? The H100_RUNBOOK's "25.03-py3 always has it" claim is now falsified.
2. Should `requirements.txt` carry a `triton<3.7` pin? Test on local Mac (no GPU) vs NGC (GPU + pytorch-triton) — they want different things.
3. Is the v3 spec's MFU assumption (35-47%) ever achievable on Vast H100/H200 in practice, or is it Lambda-cluster-only? The auto/2026-05-17 → 2026-05-20 sessions hit similar issues on H200.
4. For A0 specifically — can the calibration cell run at smaller compute (e.g., C=5e18 instead of 1.5e19) without losing its purpose? It only needs to match nanochat T6 directionally.
