# H100 Bench Runbook — wallclock-port

One-page operator's guide for running the 5-phase wallclock-port battery on
2×H100 (or 8×H100) via Vast.ai.

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

## Run the bench (one command)

```bash
cd /path/to/llm
bash scripts/run_h100_bench.sh
```

That's it. The orchestrator does:
1. Search cheapest 2×H100 SXM offer on Vast (verified, reliability >0.95)
2. Rent with `nvcr.io/nvidia/pytorch:25.01-py3` image (NGC, includes FA3 + gcc + torch 2.6 + `_grouped_mm`)
3. Wait for `actual_status=running` (~2-5 min image pull)
4. SCP `provision_h100.sh` → run it (idempotent, ~30s: verify FA3, install missing pip deps, wandb login)
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
```

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
