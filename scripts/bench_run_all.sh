#!/usr/bin/env bash
# Run the full wallclock benchmark battery on 1 or 2× H100.
#
# Usage:
#   bash scripts/bench_run_all.sh                # single-GPU
#   NPROC=2 bash scripts/bench_run_all.sh        # 2×H100 via torchrun
#   DEPTH=20 NPROC=2 bash scripts/bench_run_all.sh
#
# Each phase writes runs/bench/<phase>.json + .jsonl. Final step runs
# bench_compare + bench_report and (optionally) syncs to S3/GCS.
#
# 5-phase battery (each adds one optimization to the previous):
#   phase_0_baseline       — torch.compile(dynamic=False), no fullgraph, no chunked CE, no act ckpt, no FP8
#   phase_1_chunked_ce     — + chunked cross-entropy
#   phase_2_compile_full   — + torch.compile(fullgraph=True, max-autotune)
#   phase_3_act_ckpt       — + activation checkpointing
#   phase_4_fp8            — + FP8 (Float8Linear) — production stack
#
# Each phase also asserts FA3 is active via --require-fa3 (fail loud if SDPA
# fallback is silently selected, which would tank MFU).

set -euo pipefail

DEPTH="${DEPTH:-12}"
DBS="${DBS:-8}"
ACCUM="${ACCUM:-2}"
SEQ="${SEQ:-2048}"
NPROC="${NPROC:-1}"
WARMUP="${WARMUP:-10}"
MEASURE="${MEASURE:-30}"
WINDOW="${WINDOW:-SSSL}"
REQUIRE_FA3="${REQUIRE_FA3:-1}"   # 1=fail if FA3 not active, 0=allow SDPA fallback (debug only)

REQUIRE_FA3_FLAG=""
[[ "$REQUIRE_FA3" == "1" ]] && REQUIRE_FA3_FLAG="--require-fa3"

# Optional wandb integration. Set WANDB_PROJECT to enable live monitoring + history.
WANDB_FLAGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_FLAGS+=(--wandb-project "$WANDB_PROJECT")
  [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_FLAGS+=(--wandb-entity "$WANDB_ENTITY")
fi

if [[ "$NPROC" -gt 1 ]]; then
  LAUNCH=("torchrun" "--standalone" "--nproc_per_node=$NPROC" "-m" "scripts.bench_wallclock")
else
  LAUNCH=("python" "-m" "scripts.bench_wallclock")
fi

COMMON=(--depth "$DEPTH" --device-batch-size "$DBS" --grad-accum-steps "$ACCUM"
        --max-seq-len "$SEQ" --warmup-steps "$WARMUP" --measure-steps "$MEASURE"
        --window-pattern "$WINDOW" $REQUIRE_FA3_FLAG "${WANDB_FLAGS[@]}")

echo "▶ Phase 0: baseline (no wallclock optimizations)"
"${LAUNCH[@]}" --phase phase_0_baseline --compile-mode default "${COMMON[@]}"

echo "▶ Phase 1: + chunked CE"
"${LAUNCH[@]}" --phase phase_1_chunked_ce --compile-mode default --chunked-ce "${COMMON[@]}"

echo "▶ Phase 2: + torch.compile(fullgraph=True, max-autotune)"
"${LAUNCH[@]}" --phase phase_2_compile_full --compile-mode max-autotune --compile-fullgraph --chunked-ce "${COMMON[@]}"

echo "▶ Phase 3: + activation checkpointing"
"${LAUNCH[@]}" --phase phase_3_act_ckpt --compile-mode max-autotune --compile-fullgraph --chunked-ce --activation-ckpt "${COMMON[@]}"

echo "▶ Phase 4: + FP8 (Float8Linear) — full production stack"
"${LAUNCH[@]}" --phase phase_4_fp8 --compile-mode max-autotune --compile-fullgraph --chunked-ce --activation-ckpt --fp8 "${COMMON[@]}"

echo
echo "▶ Comparison table:"
python -m scripts.bench_compare runs/bench/phase_0_baseline.json \
                                runs/bench/phase_1_chunked_ce.json \
                                runs/bench/phase_2_compile_full.json \
                                runs/bench/phase_3_act_ckpt.json \
                                runs/bench/phase_4_fp8.json

echo
echo "▶ Markdown report:"
python -m scripts.bench_report --out runs/bench/report.md \
                               runs/bench/phase_0_baseline.json \
                               runs/bench/phase_1_chunked_ce.json \
                               runs/bench/phase_2_compile_full.json \
                               runs/bench/phase_3_act_ckpt.json \
                               runs/bench/phase_4_fp8.json

echo
echo "▶ Done. Outputs:"
echo "    runs/bench/_meta.json                       (repro snapshot)"
echo "    runs/bench/phase_*.json                     (per-phase summary)"
echo "    runs/bench/phase_*.jsonl                    (per-step streaming)"
echo "    runs/bench/report.md                        (Markdown summary report)"
if [[ -n "${BUCKET:-}" && -n "${RUN_ID:-}" ]]; then
  echo
  echo "▶ Pushing to $BUCKET/bench/$RUN_ID ..."
  bash scripts/bench_sync.sh
else
  echo
  echo "▶ To push to durable storage, run:"
  echo "    BUCKET=s3://your-bucket RUN_ID=$(date -u +%Y-%m-%d_%H%M%S)_${NPROC}gpu bash scripts/bench_sync.sh"
fi
