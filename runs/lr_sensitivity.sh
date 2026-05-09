#!/bin/bash
# Phase 1 LR sensitivity check
# Protocol: dev/sweep_design.md §3
#
# Tests whether nanochat's empirical LR defaults are at least near-optimal
# at our middle sweep scale (d6). Three LR multipliers (0.5x, 1.0x, 2.0x)
# applied to the AdamW group LRs at one fixed MoE config.
#
# Pass criterion: loss(1.0x) <= min(loss(0.5x), loss(2.0x)) * 1.05
# Fail action: see dev/sweep_design.md §3d (Option F1: expand to 7 multipliers)
#
# IMPORTANT design notes:
# - MoE config is GPTConfig DEFAULTS (num_experts=8, top_k=2, num_shared=1).
#   This corresponds to G≈2 in our framework, NOT the G=4 the spec mentioned.
#   Acceptable for LR sensitivity testing — the LR transferability question
#   doesn't depend strongly on the specific G value.
#   For the actual G sweep (Phase 2+), need to add MoE CLI flags to
#   base_train.py — TODO before running the main sweep.
# - Matrix LR (Muon) stays UNSCALED across all 3 cells. Muon's Polar Express
#   orthogonalization is approximately scale-invariant for matmul params; we
#   only sweep the AdamW group LRs.
# - dmodel_scale (the 1/sqrt(d) factor) is applied INSIDE setup_optimizer,
#   so we pass UNSCALED base LRs and let the model handle the per-width transfer.
#
# Run from llm/ directory:
#     bash runs/lr_sensitivity.sh
#
# Or with custom NPROC_PER_NODE if not 8:
#     NPROC_PER_NODE=4 bash runs/lr_sensitivity.sh

set -e

# =============================================================================
# Config
# =============================================================================

# Fixed model config for the LR check (middle of our sweep grid)
DEPTH=6                      # d6: model_dim=384, n_layer=6
TARGET_FLOPS=3e19            # ~2 hr per cell on 8xH100 BF16

# Base LRs from base_train.py defaults (these are nanochat's pre-dmodel-scale values)
EMBEDDING_LR_BASE=0.3
UNEMBEDDING_LR_BASE=0.008
MATRIX_LR_BASE=0.02          # Muon — held constant; not scaled
SCALAR_LR_BASE=0.5

# LR multipliers to test (per Phase 1 protocol)
LR_MULTIPLIERS=(0.5 1.0 2.0)

# Output
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
RESULTS_DIR="$NANOCHAT_BASE_DIR/lr_sensitivity_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Activate venv if it exists (nanochat uv convention)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
export OMP_NUM_THREADS=1

# CSV header (only write if file doesn't exist or is empty)
if [ ! -s "$RESULTS_FILE" ]; then
    echo "lr_multiplier,embedding_lr,unembedding_lr,matrix_lr,scalar_lr,depth,target_flops,val_bpb,train_time_sec,timestamp" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# =============================================================================
# Main loop
# =============================================================================

log "Phase 1 LR sensitivity check"
log "  Depth         : d$DEPTH (model_dim=$((DEPTH * 64)))"
log "  Target FLOPs  : $TARGET_FLOPS"
log "  LR multipliers: ${LR_MULTIPLIERS[*]}"
log "  GPUs          : $NPROC_PER_NODE"
log "  Results       : $RESULTS_FILE"
log ""

for mult in "${LR_MULTIPLIERS[@]}"; do

    # Skip if this multiplier already has a row in results
    if grep -q "^${mult}," "$RESULTS_FILE" 2>/dev/null; then
        log "Skipping LR multiplier ${mult}x (already in results)"
        continue
    fi

    # Compute scaled LRs (multiplier applied to AdamW groups; Muon untouched)
    EMB_LR=$(python -c "print($EMBEDDING_LR_BASE * $mult)")
    UNEMB_LR=$(python -c "print($UNEMBEDDING_LR_BASE * $mult)")
    MAT_LR=$MATRIX_LR_BASE
    SCAL_LR=$(python -c "print($SCALAR_LR_BASE * $mult)")

    TAG="lr_check_x${mult}"
    LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

    log "============================================================"
    log "Training cell: LR multiplier = ${mult}x"
    log "  embedding-lr   = $EMB_LR  (base=$EMBEDDING_LR_BASE × $mult)"
    log "  unembedding-lr = $UNEMB_LR  (base=$UNEMBEDDING_LR_BASE × $mult)"
    log "  matrix-lr      = $MAT_LR  (Muon, unscaled)"
    log "  scalar-lr      = $SCAL_LR  (base=$SCALAR_LR_BASE × $mult)"
    log "============================================================"

    START_TIME=$(date +%s)

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        --depth=$DEPTH \
        --target-flops=$TARGET_FLOPS \
        --embedding-lr=$EMB_LR \
        --unembedding-lr=$UNEMB_LR \
        --matrix-lr=$MAT_LR \
        --scalar-lr=$SCAL_LR \
        --run="lr_sensitivity_${TAG}" \
        --model-tag="${TAG}" \
        --core-metric-every=999999 \
        --sample-every=-1 \
        --save-every=-1 \
        2>&1 | tee "$LOG_FILE"

    END_TIME=$(date +%s)
    TRAIN_TIME=$((END_TIME - START_TIME))

    # Extract final val_bpb from log
    VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')
    if [ -z "$VAL_BPB" ]; then
        log "WARNING: could not extract val_bpb from $LOG_FILE — using 0.0 placeholder"
        VAL_BPB="0.0"
    fi

    log "Cell complete: val_bpb=$VAL_BPB, train_time=${TRAIN_TIME}s ($((TRAIN_TIME/60))min)"

    # Append to CSV
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "${mult},${EMB_LR},${UNEMB_LR},${MAT_LR},${SCAL_LR},${DEPTH},${TARGET_FLOPS},${VAL_BPB},${TRAIN_TIME},${TIMESTAMP}" >> "$RESULTS_FILE"
done

# =============================================================================
# Results + pass/fail check
# =============================================================================

log ""
log "================================================================"
log "Phase 1 LR sensitivity check: all cells complete"
log "================================================================"
log ""
log "Results CSV:"
column -t -s',' "$RESULTS_FILE"
log ""

# Pass criterion: loss(1.0x) <= min(loss(0.5x), loss(2.0x)) * 1.05
# Per dev/sweep_design.md §3b
log "Evaluating pass criterion (per dev/sweep_design.md §3b):"
log "  PASS if loss(1.0x) <= min(loss(0.5x), loss(2.0x)) * 1.05"
log ""

python <<EOF
import csv
import sys

results_file = "$RESULTS_FILE"
with open(results_file) as f:
    rows = list(csv.DictReader(f))

losses = {}
for r in rows:
    try:
        m = float(r['lr_multiplier'])
        bpb = float(r['val_bpb'])
        if bpb > 0:
            losses[m] = bpb
    except (ValueError, KeyError):
        continue

required = [0.5, 1.0, 2.0]
missing = [m for m in required if m not in losses]
if missing:
    print(f"INCOMPLETE: missing results for LR multipliers: {missing}")
    sys.exit(1)

l_low, l_mid, l_hi = losses[0.5], losses[1.0], losses[2.0]
best_extreme = min(l_low, l_hi)
threshold = best_extreme * 1.05

print(f"  loss(0.5x) = {l_low:.4f}")
print(f"  loss(1.0x) = {l_mid:.4f}")
print(f"  loss(2.0x) = {l_hi:.4f}")
print(f"  min(extremes) = {best_extreme:.4f}")
print(f"  threshold (×1.05) = {threshold:.4f}")
print()

if l_mid <= threshold:
    print(f"PASS: loss(1.0x)={l_mid:.4f} <= threshold={threshold:.4f}")
    print()
    print("→ Use Karpathy's defaults (×dmodel_scale) for the main sweep.")
    print("→ Document the 3-cell evidence in dev/LOG.md.")
else:
    print(f"FAIL: loss(1.0x)={l_mid:.4f} > threshold={threshold:.4f}")
    print()
    print("→ The 1/sqrt(d) scaling rule is mistuned at d$DEPTH.")
    print("→ Recommended fail action: Option F1 (expand sweep to 7 LR multipliers).")
    print("   Add multipliers {0.25, 0.75, 1.5, 3.0, 4.0} to LR_MULTIPLIERS array")
    print("   and re-run this script. Total cost ~\$210.")
    print("→ See dev/sweep_design.md §3d for full fail-action protocol.")
    sys.exit(2)
EOF

PASS_FAIL_EXIT=$?
log ""
if [ $PASS_FAIL_EXIT -eq 0 ]; then
    log "Phase 1 STATUS: PASS — proceed to Phase 2 main sweep"
elif [ $PASS_FAIL_EXIT -eq 2 ]; then
    log "Phase 1 STATUS: FAIL — apply fail-action F1 before main sweep"
else
    log "Phase 1 STATUS: INCOMPLETE — see errors above"
fi

exit $PASS_FAIL_EXIT
