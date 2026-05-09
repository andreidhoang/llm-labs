# Wallclock benchmark outputs

This directory holds the raw artifacts from `scripts/bench_wallclock.py`
runs. One subset of files per run battery; multiple batteries diff cleanly.

## File layout

```
runs/bench/
├── _meta.json                      # repro snapshot: git SHA, torch/CUDA/NCCL,
│                                     GPU model, driver, hostname, NCCL env.
│                                     Updated on every phase (latest_run + per-phase log).
│
├── phase_0_baseline.json           # final summary for one phase
├── phase_0_baseline.jsonl          # streaming per-step records (crash-resilient)
├── phase_1_chunked_ce.json
├── phase_1_chunked_ce.jsonl
├── phase_2_compile_full.json
├── phase_2_compile_full.jsonl
├── phase_3_act_ckpt.json
├── phase_3_act_ckpt.jsonl
└── report.md                       # generated Markdown summary for humans
```

## What's in each file

### `_meta.json`

Single source of truth for "what code + what hardware + what env" produced
the numbers. Two runs with identical `_meta.json` should be near-identical
(modulo NCCL/cuBLAS nondeterminism). If a result looks weird, this is the
first place to check.

### `<phase>.json`

Self-contained per-phase summary. Schema:

```json
{
  "phase": "phase_2_compile_full",
  "n_gpu": 2,
  "gpu_name": "NVIDIA H100 80GB HBM3",
  "device_batch_size": 16,
  "max_seq_len": 2048,
  "grad_accum_steps": 2,
  "compile_mode": "max-autotune",
  "compile_fullgraph": true,
  "chunked_ce": true,
  "activation_ckpt": false,
  "fp8": true,
  "metrics": {
    "tokens_per_sec_per_gpu": 47830.5,
    "mfu_pct": 51.2,
    "peak_hbm_gb": 38.4,
    "step_time_ms": 1370.2,
    "fwd_bwd_ms": 1180.5,
    "optim_step_ms": 189.3,
    "optim_pct_of_step": 13.8,
    "compile_first_step_overhead_s": 234.6,
    "step_p50_ms": 1370.2,
    "step_p90_ms": 1395.8,
    "step_min_ms": 1351.0,
    "step_max_ms": 1421.7
  },
  "meta": { ... },                  // copy of _meta.json["latest_run"]
  "per_step_step_ms": [...],
  "per_step_fwdbwd_ms": [...],
  "per_step_optim_ms": [...]
}
```

### `<phase>.jsonl`

One JSON object per measured step, line-buffered + fsynced. If the VM
gets killed mid-run, this still has the steps that completed. Schema:

```jsonl
{"step": 0, "step_ms": 1392.1, "fwdbwd_ms": 1201.3, "optim_ms": 190.8, "peak_hbm_gb": 38.39}
{"step": 1, "step_ms": 1370.5, "fwdbwd_ms": 1180.7, "optim_ms": 189.8, "peak_hbm_gb": 38.41}
...
```

Use case: plot HBM growth, detect memory leak, compute non-default percentiles,
spot stragglers (steps with `step_ms` >> p50 are usually GC or NCCL hiccups).

## Comparing across runs

```bash
# Print comparison table on stdout
python -m scripts.bench_compare runs/bench/phase_0_baseline.json \
                                runs/bench/phase_1_chunked_ce.json \
                                runs/bench/phase_2_compile_full.json \
                                runs/bench/phase_3_act_ckpt.json
```

## Generating a Markdown report

```bash
# Default: read runs/bench/phase_*.json and write runs/bench/report.md
python -m scripts.bench_report

# Or pin the phase order explicitly
python -m scripts.bench_report --out runs/bench/report.md \
                               runs/bench/phase_0_baseline.json \
                               runs/bench/phase_1_chunked_ce.json \
                               runs/bench/phase_2_compile_full.json \
                               runs/bench/phase_3_act_ckpt.json
```

The report includes run metadata, phase configs, baseline-relative metric
deltas, per-step stability from JSONL, and a short "best observed" section.
If `matplotlib` is installed, it also writes `runs/bench/plots/*.png` and
embeds those plots in the Markdown; otherwise the table-only report still
works.

## Pushing to durable storage

VMs die. Push everything (including the `~/.cache/torch/inductor` autotune
cache) to S3/GCS:

```bash
BUCKET=s3://my-bucket RUN_ID=2026-05-04_first bash scripts/bench_sync.sh
# or background:
WATCH=1 BUCKET=... RUN_ID=... bash scripts/bench_sync.sh &
```

On a fresh VM, restore both:

```bash
BUCKET=s3://my-bucket RUN_ID=2026-05-04_first bash scripts/bench_restore.sh
```

The autotune cache restore is the big win: skips 3-5 min of `max-autotune`
on the first phase if the GPU SM matches the cache's source.

## Live observability via wandb

Add `--wandb-project <name>` to any `bench_wallclock.py` invocation. Every
step's record streams to wandb in real time (charts: `step_ms`, `fwdbwd_ms`,
`optim_ms`, `peak_hbm_gb`), plus the final summary lands in `summary.*` and
the JSON + JSONL get uploaded as wandb artifacts.
