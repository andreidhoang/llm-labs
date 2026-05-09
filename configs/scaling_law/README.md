# Scaling-Law Cell Configs

Frontier-style v2 configs for `dev/scaling_law_self_assignment.md`.

The key upgrade from the old 5-cell draft is that every fitting compute scale is
a real IsoFLOPs curve with three widths. We no longer treat a single cell as an
`N_opt(C)` point.

| File | Phase | C (FLOPs) | depth | model_dim | Cost |
|---|---|---|---|---|---|
| `F1_s.json` | 1 (fit) | 5×10¹⁸ | 18 | 1152 | ~$9 |
| `F1_m.json` | 1 (fit) | 5×10¹⁸ | 20 | 1280 | ~$9 |
| `F1_l.json` | 1 (fit) | 5×10¹⁸ | 22 | 1408 | ~$9 |
| `F2_s.json` | 1 (fit) | 1.5×10¹⁹ | 22 | 1408 | ~$25 |
| `F2_m.json` | 1 (fit) | 1.5×10¹⁹ | 24 | 1536 | ~$25 |
| `F2_l.json` | 1 (fit) | 1.5×10¹⁹ | 26 | 1664 | ~$25 |
| `F3_s.json` | 1 (fit) | 3×10¹⁹ | 22 | 1408 | ~$35 |
| `F3_m.json` | 1 (fit) | 3×10¹⁹ | 24 | 1536 | ~$35 |
| `F3_l.json` | 1 (fit) | 3×10¹⁹ | 26 | 1664 | ~$35 |
| `G3_big_run.template.json` | 2 (verify) | 6×10¹⁹ | TBD | TBD | ~$69 |

**Total envelope:** ~13 hr fitting + ~4.3 hr held-out verification = **~17.3 hr** on 8×H100.
At $16/hr this is **~$277**, before reruns or endpoint-expansion cells.

## Execution order

```bash
# Phase 1 — fitting (run F1_s FIRST as reactive sanity gate)
python scripts/sweep_runner.py submit --config configs/scaling_law/F1_s.json
# Check sanity bounds before continuing (per spec §4.1):
#   no NaN, loss < 2.5, monotone decreasing, routing entropy > 1.45, MFU > 25%
python scripts/sweep_runner.py submit --config configs/scaling_law/F1_m.json
python scripts/sweep_runner.py submit --config configs/scaling_law/F1_l.json
python scripts/sweep_runner.py submit --config configs/scaling_law/F2_s.json
python scripts/sweep_runner.py submit --config configs/scaling_law/F2_m.json
python scripts/sweep_runner.py submit --config configs/scaling_law/F2_l.json
python scripts/sweep_runner.py submit --config configs/scaling_law/F3_s.json
python scripts/sweep_runner.py submit --config configs/scaling_law/F3_m.json
python scripts/sweep_runner.py submit --config configs/scaling_law/F3_l.json

# P1.5 — fit + preregister
python scripts/sweep_runner.py fit
# Manually generate G3_big_run.json from template using fit output
# Commit prediction to dev/LOG.md BEFORE next step

# Phase 2 — verification
python scripts/sweep_runner.py submit --config configs/scaling_law/G3_big_run.json
```

## Cell sizing rationale

Each cell uses `compute_budget_target_flops` as the actual training horizon.
`target_param_data_ratio` is kept close to the implied `D/N` so nanochat's
batch-size and learning-rate transfer heuristics see a consistent scale.

The 9 fit cells span 3 compute scales × 3 widths:
- **C₁ = 5e18:** d18, d20, d22
- **C₂ = 1.5e19:** d22, d24, d26
- **C₃ = 3e19:** d22, d24, d26

Fit acceptance requires the best depth at each compute scale to be interior.
If any scale chooses the smallest or largest tested depth, the curve is not
bracketed; add one expansion cell on that side before fitting.

`G3_big_run.json` is generated post-fit by reading the predicted `(depth,
model_dim)` from the IsoFLOPs power law extrapolated to C=6×10¹⁹. The G3 loss
is held out and must be preregistered before launch.
