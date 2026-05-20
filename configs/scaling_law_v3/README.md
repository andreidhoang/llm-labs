# Scaling-Law v3 (minimal) — cell configs

Spec: [`dev/scaling_law_v3_minimal.md`](../../dev/scaling_law_v3_minimal.md). Supersedes `configs/scaling_law/` under Option A.

## The 5 cells

| File | Cell | Phase | Config | C (FLOPs) | Depth | Hidden | Wall-clock | Cost |
|---|---|---|---|---|---|---|---|---|
| `A0.json` | A0 | calibration | dense, text-only | 1.5×10¹⁹ | 24 | 1536 | ~1.6 hr | ~$25 |
| `C1.json` | C1 | fit | MoE G=2 + multimodal r=0.3 | 5×10¹⁸ | 20 | 1280 | ~0.8 hr | ~$13 |
| `C_mid.json` | C_mid | fit (required) | MoE G=2 + multimodal r=0.3 | 1.5×10¹⁹ | 24 | 1536 | ~1.7 hr | ~$28 |
| `C2.json` | C2 | fit | MoE G=2 + multimodal r=0.3 | 3×10¹⁹ | 26 | 1664 | ~3.4 hr | ~$54 |
| `V3_template.json` | V3 | verification | MoE G=2 + multimodal r=0.3 | 6×10¹⁹ | **TBD from fit** | TBD | ~6.0 hr | ~$96 |

**Total: ~$216, ~13.5 GPU-hours, ~4 days elapsed.**

> **Why C_mid is required, not optional:** Hoffmann α is non-identifiable from 2 unique N values (verified empirically; see `dev/scaling_law_v3_minimal.md` §5 cut #2 revision). C_mid is the cheapest cell ($28) that buys parameter identifiability.

## Execution order

```bash
# Pre-flight (CPU, free)
python scripts/preflight.py
for c in A0 C1 C_mid C2; do
  python scripts/sweep_runner.py dry-run --config configs/scaling_law_v3/${c}.json
done

# Phase 0 — calibration (instrument check vs nanochat)
python scripts/sweep_runner.py submit --config configs/scaling_law_v3/A0.json
# Inspect runs/manifest.json: A0 final_val_loss_joint within 5% of nanochat T6?
#   PASS → proceed
#   FAIL → halt, diagnose; do NOT run any multimodal cell while pipeline is biased

# Phase 1 — fit cells (3 unique N values; ALL THREE required for α identifiability)
python scripts/sweep_runner.py submit --config configs/scaling_law_v3/C1.json
python scripts/sweep_runner.py submit --config configs/scaling_law_v3/C_mid.json
python scripts/sweep_runner.py submit --config configs/scaling_law_v3/C2.json

# Fit + preregister
python scripts/sweep_runner.py fit-hoffmann
# Inspect: residuals < 5% log-space, slope sensible
#   FAIL with curvature → add a midpoint cell ($28), refit
#   PASS → fill V3_template.json with predicted (depth, hidden_size, val_bpb)
#          → copy to V3.json
#          → COMMIT to git + dev/LOG.md BEFORE next step

# Phase 2 — verification (preregistered)
python scripts/sweep_runner.py submit --config configs/scaling_law_v3/V3.json
python scripts/plot_predicted_vs_actual.py
```

## Gates

| Gate | Trigger | PASS | FAIL |
|---|---|---|---|
| **G_anchor** | A0 final val_bpb vs nanochat T6 | Within 5% relative → 🟢 inheritance locked | Outside 5% → halt; do not run multimodal cells |
| **C (fit)** | Hoffmann residuals + slope sane | Generate V3 config, preregister, proceed | Add midpoint cell or escalate |
| **D (verification)** | V3 actual vs predicted val_bpb | Within ±0.1 nats → fit validated | Honest writeup of failure |

## Why these specific configs

- **Depths in {20, 24, 26}** — inside nanochat's validated LR-transfer band d∈[20, 28] (per `core/model.py:468-475`).
- **A0 mirrors nanochat T6 setup** (d24, dense, text-only, ClimbMix) — that's what makes it a calibration cell, not a fit point.
- **C1/C_mid/C2 span ~6× compute** (5e18 → 1.5e19 → 3e19) at 3 distinct depths (d20/d24/d26). The 3 unique N values are required for Hoffmann α identifiability.
- **V3 extrapolates only ~2×** from C2's upper anchor (3e19 → 6e19), inside the regime spanned by the fit cells.
- **G=2 (top_k=2, num_shared=1, num_experts=8)** — nanochat MoE config T3 verbatim. This study does NOT sweep G.
- **r=0.3 fixed** — Option A scope decision; r-sweep deferred to Option B.

## What's NOT in this directory

- `B1.json` / `B2.json` — dense + multimodal cells. Cut (v3 §5 cut #9); modality decomposition out of scope.
- `F*_s/m/l.json` — old v2 9-cell IsoFLOP grid. Lives in `configs/scaling_law/` for historical reference; not used by this design.
