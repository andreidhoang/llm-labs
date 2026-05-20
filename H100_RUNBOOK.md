# H100 Training Runbook — Vast.ai H100

Operator guide for H100 MoE/multimodal training and benchmarking on Vast.ai.
The most important rule: use an NGC PyTorch image with `torch._grouped_mm`.
Without it, `core/moe.py` falls back to a correct but much slower per-expert loop.

## Prerequisites (one-time, on local Mac)

```bash
# 1. vastai CLI
brew install vastai/tap/vastai
vastai set api-key <your-vast-key>

# 2. SSH key on Vast (verify exists)
vastai show ssh-keys              # if empty: vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"

# 3. GitHub PAT (scope: repo) — https://github.com/settings/tokens
export GH_TOKEN=<redacted-github-token>

# 4. Wandb key — https://wandb.ai/authorize
export WANDB_API_KEY=xxx
```

## Required PyTorch Image

Use this image by default:

```bash
export IMAGE=nvcr.io/nvidia/pytorch:25.03-py3
```

Why `25.03-py3`:
- Our 2×H100 smoke on `nvcr.io/nvidia/pytorch:25.01-py3` showed `torch=2.6.0a0` and `torch._grouped_mm=False`.
- NVIDIA's 25.03 PyTorch container release notes list CUDA `12.8.1` and PyTorch `2.7.0a0`.
- The same release notes list driver release `570+` for CUDA `12.8.1`, which matches common Vast H100 hosts.
- NVIDIA's 25.04 image also uses PyTorch `2.7.0a0`, but moves to CUDA `12.9`; many cheap Vast H100 offers advertise `cuda_max_good=12.8`, so `25.03` is the safer default.

> ⚠ **2026-05-20 update — not all 25.03 builds ship `_grouped_mm`.**
> An H200×8 instance pulled NGC `pytorch:25.03-py3` build hash `7c8ec84dab.nv25.03`
> which has `torch._grouped_mm = False` even at the C++ ATen op level
> (`[x for x in dir(torch.ops.aten) if 'group' in x.lower()]` returns only
> `native_group_norm`). The image tag is the same; only the build hash differs.
> Always run the smoke probe below BEFORE committing to a training run, and have
> a fallback plan (see `dev/auto_findings/2026-05-20-A0-attempt/findings.md`).
>
> Vanilla `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel` from Docker Hub is a tested
> alternative that ships `_grouped_mm` from the public PyTorch release.

When selecting offers, require:

```bash
vastai search offers \
  'num_gpus=2 gpu_name in [H100_SXM,H100_PCIE,H100_NVL] reliability>0.95 verified=true cuda_max_good>=12.8' \
  -o 'dph_total' --raw
```

For 8 GPUs:

```bash
vastai search offers \
  'num_gpus=8 gpu_name in [H100_SXM,H100_PCIE,H100_NVL] reliability>0.95 verified=true cuda_max_good>=12.8' \
  -o 'reliability-,dph_total' --raw
```

Immediately after SSH, verify the kernel:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("private _grouped_mm:", hasattr(torch, "_grouped_mm"))
print("public grouped_mm:", hasattr(torch.nn.functional, "grouped_mm"))
assert hasattr(torch, "_grouped_mm"), "wrong image: MoE will use slow fallback"
PY
```

`torch.nn.functional.grouped_mm` is the newer public API documented by PyTorch,
but this repo currently calls the private `torch._grouped_mm` path. Treat public
`grouped_mm` without private `_grouped_mm` as a code-port task, not as a green
light for performance training.

## Provisioning gotchas (2026-05-20)

A full sweep of issues discovered launching A0 on Vast 8×H200 / NGC 25.03 is in
`dev/auto_findings/2026-05-20-A0-attempt/findings.md`. The five most important:

### 1. `pip install -r requirements.txt` overwrites NGC's `pytorch-triton`

NGC ships `pytorch-triton 3.2.0+gitb2684bf3b.nvinternal` which provides the
`triton_key` symbol that NGC's `torch 2.7.0a0` imports. `pip install triton`
(or `pip install -r requirements.txt`) pulls vanilla triton 3.7.0 from PyPI,
which removed `triton_key`. Result: `torch.compile` fails with
`ImportError: cannot import name 'triton_key' from 'triton.compiler.compiler'`.

Fix at install time:

```bash
# Skip the triton line when installing in an NGC image:
grep -v -E "^triton" requirements.txt | grep -v "^#" | grep -v "^$" > /tmp/reqs.txt
pip install -r /tmp/reqs.txt
```

Fix at runtime (workaround if triton already overwritten):

```bash
export TORCHDYNAMO_DISABLE=1   # disables torch.compile; runs eager mode
```

### 2. `WANDB_API_KEY` must be set OR `WANDB_MODE=offline`

`base_train.py` invokes `wandb.init(...)` whenever `--run` is not the literal
string `"dummy"`. Fresh hosts have no key; sweep_runner now defaults
`WANDB_MODE=offline` in the launch env. For direct `torchrun` invocations:

```bash
export WANDB_MODE=offline
```

### 3. `--device-batch-size=32` OOMs at d24/seq=4096 in eager mode

H200 has 140 GB/GPU. With `torch.compile` disabled AND no `--activation-ckpt`,
the trunk activations + intermediate tensors at depth 24 exceed 140 GB.
For d24/seq=4096 in eager mode, use:

```bash
torchrun ... --device-batch-size=16 --activation-ckpt
```

`--activation-ckpt` recomputes activations during backward (~33% compute overhead)
but cuts memory by ~5×. Trade is required when `torch.compile` is unavailable.

### 4. `torchrun --run=X` needs `--` separator before the script

`torchrun`'s argparse prefix-matches `--run-path`, so `--run=A0` to the training
script is consumed by torchrun → "ambiguous option" error. Insert `--`:

```bash
torchrun --nproc-per-node=8 -- scripts/base_train.py --run=A0 ...
```

`sweep_runner.py` already does this (since 2026-05-20).

### 5. Tokenizer + data must be pre-staged before launch

`base_train.py` reads `/root/.cache/nanochat/tokenizer/tokenizer.pkl` and
`/root/.cache/nanochat/base_data_climbmix/shard_*.parquet`. Fresh hosts have
neither. Bootstrap:

```bash
# Tokenizer: scp from local Mac (only 40 KB)
scp -P $PORT -r ~/.cache/nanochat/tokenizer root@$HOST:/root/.cache/nanochat/

# Data: download from HF (~370 MB/shard; ~50 shards for A0 dense @ d24)
ssh -p $PORT root@$HOST 'cd /workspace/llm-labs && \
  PYTHONPATH=. python -m core.dataset -n 60 -w 8'
```

`core.dataset` will fetch from `karpathy/climbmix-400b-shuffle` on HuggingFace
and always also download the validation shard (shard_06542).

### Measured throughput on H200 with these workarounds

| Configuration | MFU | Wall-clock multiplier vs spec |
|---|---|---|
| Spec target (FA3 + torch.compile + grouped_mm) | 47% | 1× |
| **Observed: eager + activation-ckpt + for-loop MoE** | **~20%** | **~2.4×** |
| Expected: torch.compile working, grouped_mm working, no activation-ckpt | ~40% | ~1.2× |

Cost implication: at observed 20% MFU, the v3 study budget projects to
**~$650 instead of ~$216**. Either restore a known-good image OR re-budget;
see `dev/auto_findings/2026-05-20-A0-attempt/findings.md` §"Two paths forward".

## Run the bench (one command)

```bash
cd /path/to/llm
bash scripts/run_h100_bench.sh
```

That's it. The orchestrator does:
1. Search cheapest H100 offer on Vast (verified, reliability >0.95, `cuda_max_good>=12.8`)
2. Rent with `nvcr.io/nvidia/pytorch:25.03-py3` image (NGC, PyTorch 2.7, CUDA 12.8)
3. Wait for `actual_status=running` (~2-5 min image pull)
4. SCP `provision_h100.sh` → run it (idempotent, ~30s: verify FA3 and `_grouped_mm`, install missing pip deps, wandb login)
5. Pull repo `wallclock-port` branch via GitHub API tarball
6. Run `bench_run_all.sh` 5-phase battery (~15-25 min compute):
   - `phase_0_baseline` — no wallclock optimizations
   - `phase_1_chunked_ce` — + chunked cross-entropy
   - `phase_2_compile_full` — + `torch.compile(fullgraph=True, max-autotune)`
   - `phase_3_act_ckpt` — + activation checkpointing
   - `phase_4_fp8` — + FP8 (Float8Linear) — full production stack
7. Generate comparison table + markdown report
8. Auto-push to wandb (live charts during run + final summary)
9. SCP artifacts to local `runs/bench/h100_2gpu_<timestamp>/`
10. Auto-destroy instance (`AUTO_DESTROY=0` to keep alive)

**Cost target: ~$5 for 2×H100 full battery (~70 min wall time).**

## Customizing

```bash
# Recommended default image for MoE performance
IMAGE=nvcr.io/nvidia/pytorch:25.03-py3 bash scripts/run_h100_bench.sh

# 8×H100 production scale
GPUS=8 DEPTH=20 DBS=16 SEQ=2048 bash scripts/run_h100_bench.sh

# Larger model
DEPTH=20 DBS=16 ACCUM=4 bash scripts/run_h100_bench.sh

# Keep instance alive after success (for inspection)
AUTO_DESTROY=0 bash scripts/run_h100_bench.sh

# Different branch
BRANCH=my-branch bash scripts/run_h100_bench.sh

# Skip wandb
WANDB_API_KEY="" bash scripts/run_h100_bench.sh

# Correctness-only smoke on an older image; do not use for perf numbers
ALLOW_GROUPED_MM_FALLBACK=1 IMAGE=nvcr.io/nvidia/pytorch:25.01-py3 bash scripts/run_h100_bench.sh
```

## Training Gate

Before spending on an 8×H100 run:
- Run `scripts/provision_h100.sh` and confirm `has _grouped_mm=True`.
- Run a 2×H100 distributed smoke for 3-10 optimizer steps.
- Run 8×H100 for 50-200 steps with `--compile-fullgraph` off first.
- Only then enable compile/FP8/activation checkpointing one at a time.

Do not accept performance numbers from a run that prints `has _grouped_mm=False`.
Those numbers measure Python/per-expert fallback overhead, not the intended MoE kernel path.

## Watching it live

Two parallel options while the bench runs:

**Wandb (primary)**
```bash
open https://wandb.ai/<your-username>/llm-wallclock-port
```
Per-step `step_ms`, `fwdbwd_ms`, `optim_ms`, `peak_hbm_gb` stream live for each phase.

**Stdout tail**
```bash
# In a second terminal:
ssh -p <port> root@<host> 'tail -f /workspace/ai_labs_2026/runs/bench/run.log'
```

## After the run

Local artifacts at `runs/bench/h100_2gpu_<timestamp>/`:
- `_meta.json` — git/torch/CUDA/GPU/driver snapshot
- `phase_{0..4}_*.json` — per-phase summary metrics
- `phase_{0..4}_*.jsonl` — per-step streaming records
- `report.md` — markdown comparison table
- `run.log` — full stdout

Compare two runs:
```bash
python -m scripts.bench_compare \
  runs/bench/h100_2gpu_<run1>/phase_4_fp8.json \
  runs/bench/h100_2gpu_<run2>/phase_4_fp8.json
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `vastai search` returns 0 offers | H100 marketplace dry | Wait an hour; or check `vastai search offers 'gpu_name=H100_NVL num_gpus=2'` |
| Image pull >15 min | First-pull on this Vast machine | Just wait; subsequent rents on same machine cache it |
| `--require-fa3` fails | Wrong image (FA3 missing) | Verify `IMAGE` includes FA3 (NGC ≥24.10 does); else build from source |
| `has _grouped_mm=False` | Image is too old or PyTorch build lacks grouped GEMM | Use `IMAGE=nvcr.io/nvidia/pytorch:25.03-py3` and require `cuda_max_good>=12.8`; for smoke only set `ALLOW_GROUPED_MM_FALLBACK=1` |
| Wandb auth fails | Bad / expired key | `vastai destroy`, get fresh key from wandb.ai/authorize, re-run |
| GitHub auth (404) | PAT wrong scope or expired | Need `repo` scope on classic PAT, or `Contents:read` on fine-grained |
| Phase 4 (FP8) crashes | Float8Linear vs torch 2.6 issue | Re-run with `--fp8` removed: `IMAGE=... bash run_h100_bench.sh` (and remove phase_4 from bench_run_all.sh temporarily) |
| Compile-fullgraph hits graph break | Data-dependent branch leak | Run with `TORCH_LOGS=graph_breaks` to identify; file fix; meantime remove `--compile-fullgraph` from bench_run_all phases |
| Out-of-disk on Vast | Default 80GB filled by NGC + autotune cache + datasets | Set `DISK=120` |
| Process dies on SSH disconnect | tmux not used by orchestrator (intentional — it's one shot) | Use `bash scripts/run_h100_bench.sh > local.log 2>&1 &` to background locally |

## What this does NOT do

- Multi-node training (single-host only — Vast doesn't expose multi-host clusters)
- Long pretraining runs (use `scripts/base_train.py` directly with `nohup`)
- Auto-PR creation (manual: `gh pr create -B main -H wallclock-port`)
- Save autotune cache to S3 across runs (use `scripts/bench_sync.sh` separately)

## Cost expectations

| Config | Time | Cost (Vast on-demand) |
|---|---|---|
| 2×H100 SXM, 5 phases, depth=12 | ~70 min | ~$5 |
| 2×H100 SXM, 5 phases, depth=20 | ~120 min | ~$8 |
| 8×H100 SXM, 5 phases, depth=20 | ~60 min | ~$25 |
| Image pull (first time on a machine) | +5-8 min | included above |

Cost halves roughly when the Vast machine has cached the NGC image (subsequent
rents on the same `machine_id` skip the pull). Use `vastai search offers ... -o 'reliability-,dph_total'`
to prefer a stable machine you've used before.

## Secrets hygiene

After every run with shared secrets:
- Rotate GH PAT: https://github.com/settings/tokens → revoke + new
- Rotate wandb key: https://wandb.ai/authorize → reset

The orchestrator passes secrets via env (visible briefly in process command-line
on the destroyed instance). Treat as compromised after the run.
