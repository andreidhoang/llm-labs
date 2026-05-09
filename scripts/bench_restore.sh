#!/usr/bin/env bash
# Restore previously-synced autotune cache + bench outputs from S3/GCS.
# Run this on a fresh VM before kicking off bench_run_all.sh to skip
# torch.compile autotune (3-5 min savings on first phase).
#
# Usage:
#   BUCKET=s3://my-bucket RUN_ID=2026-05-04_first bash scripts/bench_restore.sh
#
# Env vars (same as bench_sync.sh):
#   BUCKET, RUN_ID, AUTOTUNE_CACHE, LOCAL_BENCH_DIR

set -euo pipefail

: "${BUCKET:?BUCKET env var required}"
: "${RUN_ID:?RUN_ID env var required}"

AUTOTUNE_CACHE="${AUTOTUNE_CACHE:-$HOME/.cache/torch/inductor}"
LOCAL_BENCH_DIR="${LOCAL_BENCH_DIR:-runs/bench}"

REMOTE_BENCH="${BUCKET%/}/bench/${RUN_ID}"
REMOTE_CACHE="${BUCKET%/}/autotune-cache/${RUN_ID}"

case "$BUCKET" in
  s3://*) SYNC=(aws s3 sync --no-progress);;
  gs://*) SYNC=(gsutil -q -m rsync -r);;
  *) echo "BUCKET must be s3:// or gs://" >&2; exit 1;;
esac

mkdir -p "$AUTOTUNE_CACHE" "$LOCAL_BENCH_DIR"

echo "Pulling autotune cache from $REMOTE_CACHE → $AUTOTUNE_CACHE"
"${SYNC[@]}" "$REMOTE_CACHE" "$AUTOTUNE_CACHE" || echo "(no cache yet — first run)"

echo "Pulling bench outputs from $REMOTE_BENCH → $LOCAL_BENCH_DIR"
"${SYNC[@]}" "$REMOTE_BENCH" "$LOCAL_BENCH_DIR" || echo "(no bench outputs yet)"

# Important: torch.compile autotune cache is pinned to GPU SM. Confirm we're
# restoring onto the right hardware family.
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
  echo
  echo "Current GPU: $GPU_NAME"
  echo "Verify this matches the GPU family that produced the autotune cache,"
  echo "or kernels will be re-tuned (no harm, just no time saved)."
fi
