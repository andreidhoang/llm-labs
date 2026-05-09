#!/usr/bin/env bash
# Sync benchmark results + autotune cache to S3 (or GCS).
#
# Usage:
#   # one-shot
#   BUCKET=s3://my-bucket RUN_ID=2026-05-04_first bash scripts/bench_sync.sh
#
#   # background watcher (poll every WATCH_INTERVAL seconds)
#   BUCKET=s3://my-bucket RUN_ID=2026-05-04_first WATCH=1 bash scripts/bench_sync.sh
#
#   # GCS instead of S3
#   BUCKET=gs://my-bucket RUN_ID=... bash scripts/bench_sync.sh
#
# What gets synced:
#   runs/bench/                 — JSON, JSONL, _meta.json, report.md, plots
#   ~/.cache/torch/inductor/    — torch.compile autotune cache (huge time-saver
#                                 when re-running on the same GPU arch)
#
# Why upload autotune cache:
#   First run on H100 spends 3-5 min in max-autotune. If you save the cache
#   to S3, the next 8×H100 spin downloads it and skips autotune entirely.
#   Cache is per-SM (sm_90 → sm_90 only), so don't reuse across GPU types.
#
# Env vars:
#   BUCKET           required, e.g. s3://my-bucket or gs://my-bucket
#   RUN_ID           required, unique tag for this experiment battery
#   WATCH            optional, set to 1 for background polling mode
#   WATCH_INTERVAL   optional, seconds between syncs (default 60)
#   AUTOTUNE_CACHE   optional, override default ~/.cache/torch/inductor
#   LOCAL_BENCH_DIR  optional, override default runs/bench

set -euo pipefail

: "${BUCKET:?BUCKET env var required (e.g. s3://my-bucket or gs://my-bucket)}"
: "${RUN_ID:?RUN_ID env var required (e.g. 2026-05-04_first_2xh100)}"

WATCH="${WATCH:-0}"
WATCH_INTERVAL="${WATCH_INTERVAL:-60}"
AUTOTUNE_CACHE="${AUTOTUNE_CACHE:-$HOME/.cache/torch/inductor}"
LOCAL_BENCH_DIR="${LOCAL_BENCH_DIR:-runs/bench}"

REMOTE_BENCH="${BUCKET%/}/bench/${RUN_ID}"
REMOTE_CACHE="${BUCKET%/}/autotune-cache/${RUN_ID}"

# Detect S3 vs GCS based on URI scheme
case "$BUCKET" in
  s3://*)
    if ! command -v aws >/dev/null 2>&1; then
      echo "ERROR: BUCKET starts with s3:// but 'aws' CLI not found" >&2
      exit 1
    fi
    SYNC=(aws s3 sync --no-progress)
    ;;
  gs://*)
    if ! command -v gsutil >/dev/null 2>&1; then
      echo "ERROR: BUCKET starts with gs:// but 'gsutil' not found" >&2
      exit 1
    fi
    SYNC=(gsutil -q -m rsync -r)
    ;;
  *)
    echo "ERROR: BUCKET must start with s3:// or gs://" >&2
    exit 1
    ;;
esac

do_sync() {
  if [[ -d "$LOCAL_BENCH_DIR" ]]; then
    "${SYNC[@]}" "$LOCAL_BENCH_DIR" "$REMOTE_BENCH"
    echo "[$(date -u +%H:%M:%S)] synced $LOCAL_BENCH_DIR → $REMOTE_BENCH"
  fi
  if [[ -d "$AUTOTUNE_CACHE" ]]; then
    "${SYNC[@]}" "$AUTOTUNE_CACHE" "$REMOTE_CACHE"
    echo "[$(date -u +%H:%M:%S)] synced $AUTOTUNE_CACHE → $REMOTE_CACHE"
  fi
}

if [[ "$WATCH" == "1" ]]; then
  echo "Watch mode: syncing every ${WATCH_INTERVAL}s. Ctrl-C / SIGTERM to stop."
  trap 'echo; echo "Final sync before exit..."; do_sync; exit 0' INT TERM
  while true; do
    do_sync || echo "WARNING: sync failed, will retry next cycle" >&2
    sleep "$WATCH_INTERVAL"
  done
else
  do_sync
fi
