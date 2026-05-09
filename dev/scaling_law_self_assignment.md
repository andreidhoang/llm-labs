# Scaling-Law Self-Assignment — CS336-faithful adaptation for Multimodal MoE on 8×H100

> *Self-administered version of CS336 Assignment 3. Faithful to CS336's 2-phase structure (fitting + big-run verification). Adapted from Stanford's text-only B200 API setup to our multimodal MoE 8×H100 vast.ai setup. **No anchor cell — HP transfer detected reactively via Phase 1 sanity bounds.***

**Status:** frontier-style v2 spec. Independent of `sweep_design.md` (v3 speedrun) — this study and v3 are non-overlapping; either or both can be executed.

**Owner:** operator-researcher (see `MEMORY.md` user role).

**Lineage:**
- `sweep_design_v1.md` — original 7-phase plan, $1068 (superseded)
- `sweep_design_g_sweep_deferred.md` — G-sweep at fixed r=0.3, $322 (deferred)
- `sweep_design.md` — v3 multimodal speedrun recipe (8 cells, $262, current main focus)
- `scaling_law_self_assignment.md` (this file) — CS336-faithful 2-phase fit + verification, upgraded from a 5-cell sparse fit to a 9-cell true IsoFLOPs fit (**independent track**)

---

## 0. Why this exists

CS336 Assignment 3 tests one of the most decision-relevant skills in modern AI research: **given a fitting budget, predict the optimal config for a "big run" at much larger compute, then verify your prediction.** Frontier labs do this for every multi-million-dollar training. Stanford grades on prediction-vs-actual loss agreement.

We adapt it to our setup with **maximum faithfulness to CS336's structure** — 2 phases (fit + verify), but make one senior-researcher correction versus the old draft: every fitting compute scale gets a real 3-width IsoFLOPs curve. We no longer treat a single run as `N_opt(C)`.

| | CS336 (Stanford) | Our adaptation |
|---|---|---|
| Hardware | B200 GPU, single | 8×H100 node, vast.ai |
| Compute access | Remote training API (`hyperturing.stanford.edu`) | Local `torchrun` + `sweep_runner.py` we build |
| Architecture | Text-only Transformer | **Multimodal MoE** (frozen SigLIP2 + MoE trunk, G=2) |
| Data | DCLM, seq_len=512 | ClimbMix + LAION-Recap, seq_len=4096, r=0.3 |
| Phase 1 fitting budget | 12 B200-hours | **~13 hr 8×H100, 9 fit cells** |
| Phase 2 big run | 48 B200-hours | **~4.3 hr 8×H100 held-out extrapolation** |
| Output | Predicted (config, loss) → automatically graded | Predicted (config, loss) → we run G3 verification ourselves |

**What we KEEP from CS336:**
- 2-phase structure: fitting → big run verification
- IsoFLOPs methodology (multiple N at multiple C scales, find min, fit power law)
- Predicted-vs-actual loss as the primary quality gate
- Preregistration discipline (commit prediction to git BEFORE running G3)

**What we DELIBERATELY DON'T add:**
- ❌ Preemptive HP-transfer anchor cell — detect reactively via F1.s sanity bounds (cheaper, same information value if HP failure is low-probability)
- ❌ Speedrun ablations — that's `sweep_design.md` v3's territory
- ❌ G-sweep — deferred study
- ❌ μP transfer test — we inherit Karpathy's d12-d26 envelope; reactively detect if violated

**Senior-researcher framing:** smallest design that a serious lab would still recognize as an IsoFLOPs scaling-law study: 3 compute scales × 3 widths, a full loss-surface fit, endpoint-expansion gates, and a preregistered held-out verification run.

---

## 1. Compute equivalence — the B200 → 8×H100 translation

### 1.1 The arithmetic

```
B200 BF16 dense peak:    2.25 PFLOP/s
H100 BF16 dense peak:    989 TFLOP/s   →   ratio 2.27× per GPU

Per-node throughput @ 50% MFU:
  1 × B200:              1.125 PFLOP/s effective
  8 × H100:              8 × 0.495 PFLOP/s = 3.96 PFLOP/s effective
                                              ──────
                         8×H100 ≈ 3.52 B200 equivalents per node
```

### 1.2 Budget translation

| Allocation | Wall-clock 8×H100 | vast.ai cost @ $16/hr |
|---|---|---|
| Phase 1 fitting (9 cells, 3 compute scales × 3 widths) | ~13 hr | **~$208** |
| Phase 2 held-out verification (1 predicted-optimal config) | ~4.3 hr | **~$69** |
| Total | ~17.3 hr | **~$277** |

Leaves budget room for one endpoint-expansion cell if a bracket fails.

### 1.3 MFU realism — wall-clock estimates have 30-95% upside risk

The §1.2 budget table assumes ~47-50% MFU (consistent with dense BF16 training).
**Per `nanochat/dev/LOG.md` 2026-02-19, MoE drops MFU to ~35% at d18** due to
`torch._grouped_mm` dispatch overhead. At our larger depths it may recover
somewhat, but plan for the worst case.

**At 35% MFU, real wall-clock is 30-95% higher per cell:**

| Cell | Spec estimate (47-50% MFU) | At 35% MFU (MoE realistic) | Δ |
|---|---|---|---|
| F1.* | 0.6 hr | ~0.8 hr | +33% |
| F2.* | 1.6 hr | ~2.1 hr | +30% |
| F3.* | 2.2 hr | ~3.0 hr | +35% |
| G3 | 4.3 hr | ~6.0 hr | +40% |
| **Total** | **17.3 hr (~$277)** | **~23 hr (~$370)** | **+33%** |

**Action:** F1.s is the FIRST cell run (reactive sanity gate). Its measured
wall-clock + MFU directly calibrates the rest of the budget. If F1.s shows
35% MFU, plan for ~$370 not $277. If it shows 45%+, the original $277 holds.

→ **Budget the upper bound (~$400) up-front; refund the rest if MFU is good.**
Do NOT discover at cell 7 that you've blown budget.

### 1.4 The honest constraint — depth cap

Stanford's 48 B200-hour big run = ~2×10²⁰ FLOPs ≈ Chinchilla-opt depth d32. **Outside our HP envelope (d12-d26).**

**Decision:** fit inside C ∈ {5e18, 1.5e19, 3e19} and verify at C ≈ 6e19. This is a real held-out extrapolation while staying close enough that HP transfer risk is manageable. If the predicted optimum lands beyond d26, run one explicit d28 expansion/sanity cell before G3 rather than pretending d26 is interior.

Our "big run" is therefore smaller than Stanford's in absolute compute, but the methodology is identical.

---

## 2. The two phases (CS336-faithful)

```
Phase 1 — Fitting        Phase 2 — Big run
($208)                   ($69)
──────────────           ────────
9 cells across           1 held-out predicted-optimal
3 compute scales         config from Phase 1 fit
                         
                         Compares actual vs predicted
                         loss (CS336 grading mechanism)
```

**That's it.** Two sequential phases, one preregistration step in between, two gates. Pure CS336.

**Reactive HP-transfer detection in Phase 1:** the very first cell (F1.s, ~$9) doubles as a sanity check. If its loss curve is wildly off expected behavior (e.g., NaN, loss > 2.5, or loss curve flat from step 100 onward), halt before spending the rest of Phase 1.

---

## 3. Cell grid (10 cells, frontier-style v2)

All cells: 8×H100 BF16 vast.ai, multimodal r=0.3, Karpathy nanochat HPs verbatim, MoE G=2 (`num_experts=8, top_k=2, num_shared=1`), depths inside d12-d26 validated envelope.

### 3.1 The 10 cells

| Cell | Phase | C (FLOPs) | depth | model_dim | Role | Wall-clock | Cost |
|---|---|---|---|---|---|---|---|
| **F1.s** | 1 (fit) | 5×10¹⁸ | d18 | 1152 | small width | ~0.6 hr | ~$9 |
| **F1.m** | 1 (fit) | 5×10¹⁸ | d20 | 1280 | middle width | ~0.6 hr | ~$9 |
| **F1.l** | 1 (fit) | 5×10¹⁸ | d22 | 1408 | large width | ~0.6 hr | ~$9 |
| **F2.s** | 1 (fit) | 1.5×10¹⁹ | d22 | 1408 | small width | ~1.6 hr | ~$25 |
| **F2.m** | 1 (fit) | 1.5×10¹⁹ | d24 | 1536 | middle width | ~1.6 hr | ~$25 |
| **F2.l** | 1 (fit) | 1.5×10¹⁹ | d26 | 1664 | large width | ~1.6 hr | ~$25 |
| **F3.s** | 1 (fit) | 3×10¹⁹ | d22 | 1408 | small width | ~2.2 hr | ~$35 |
| **F3.m** | 1 (fit) | 3×10¹⁹ | d24 | 1536 | middle width | ~2.2 hr | ~$35 |
| **F3.l** | 1 (fit) | 3×10¹⁹ | d26 | 1664 | large width | ~2.2 hr | ~$35 |
| **G3** | 2 (verify) | 6×10¹⁹ | TBD from fit | TBD | held-out extrapolation | ~4.3 hr | ~$69 |

### 3.2 Why these specific (C, depth) choices

**Phase 1 design:** 3 compute scales × 3 widths each. Each compute scale produces a real U-shaped IsoFLOPs curve. The minimum at that scale becomes an empirical `N_opt(C)` point.

**Compute scale spacing:** 5×10¹⁸, 1.5×10¹⁹, 3×10¹⁹ are roughly 3× apart in log space. Wider spacing = better slope estimate; tighter = more cells per dollar but worse extrapolation. Three log-spaced scales is the **standard sweet spot for power-law fitting**.

**Width selection per scale:**
- C=5×10¹⁸: depths {18, 20, 22} bracket Chinchilla-opt ≈ d20
- C=1.5×10¹⁹: depths {22, 24, 26} bracket Chinchilla-opt ≈ d24
- C=3×10¹⁹: depths {22, 24, 26} test whether the optimum is still inside the validated envelope

All fitting cells remain inside Karpathy's d12-d26 validated envelope. If F3.l wins at d26, the design has not bracketed C3; add a d28 expansion/sanity cell before fitting G3.

**9-cell fit advantage over 5-cell draft:**

| Property | 5-cell draft | 9-cell v2 |
|---|---|---|
| C1/C2 minima | bracketed but coarse | bracketed with middle point |
| C3 minimum | assumed from one cell | empirically measured |
| Endpoint detection | weak | explicit at every C |
| Loss-surface fit | underdetermined | enough points for Hoffmann-form fit |
| Scientific claim | preliminary | credible small-scale scaling law |

→ **The v2 design estimates the curve instead of hoping one guessed point is optimal.**

### 3.3 G3 verification cell — the falsification mechanism

**G3 is not another fitting point.** It is the falsification mechanism for the entire study.

CS336's grading: "predicted vs actual loss agreement." Without G3, fit quality is unfalsifiable.

G3 config is **derived post-fit** from Phase 1 results at held-out C=6×10¹⁹. Configuration and predicted loss are committed to `dev/LOG.md` BEFORE running.

### 3.4 Compute accounting rule

For MoE, fit and report **active scaling parameters and measured training FLOPs**, not total parameters. `C = 6ND` is only an intuition; the manifest must store `flops_per_token_estimated`, `actual_tokens`, `n_active_params`, and `n_total_params` from the model.

---

## 4. Phase ordering with gates

```
PHASE 1 — Fitting (9 cells, ~$208, ~13 hr)
  │
  │  Run F1.s FIRST (cheapest sanity gate)
  │       │
  │       └─── REACTIVE GATE 0: F1.s sanity bounds met?
  │            (no NaN, loss < 2.5, loss decreasing over time)
  │            PASS → continue Phase 1
  │            FAIL → halt, diagnose before running the grid
  │
  │  Run remaining 8 fit cells:
  │    F1.m, F1.l, F2.s, F2.m, F2.l, F3.s, F3.m, F3.l
  │
  └─── GATE B: all 9 cells complete; min-N at every C is interior
       PASS → proceed to fit
       FAIL: endpoint min → add 1-2 expansion cells on that side
       ▼
P1.5 FIT + PREREGISTER ($0, ~1 hr CPU)
  │
  └─── GATE C: power-law and Hoffmann fits agree; residuals small
       PASS → commit prediction to dev/LOG.md, generate G3 config, proceed
       MARGINAL (a outside but residuals small): document caveat, proceed
       FAIL (large residuals OR slope outside): scaling law form misspecified;
            consider parametric form L(N,D) = E + A/N^α + B/D^β with explicit fit
       ▼
PHASE 2 HELD-OUT VERIFICATION (1 cell, ~$69, ~4.3 hr)
  │
  └─── GATE D: actual val loss within ±0.1 of preregistered prediction
       PASS → fit validated; write up
       FAIL → honest report of fit failure with diagnosis
```

### 4.1 The reactive sanity bounds for F1.s (replaces preemptive anchor)

When F1.s completes, before proceeding to the remaining 8 fit cells, check:

| Sanity check | Reasonable bound | Failure interpretation |
|---|---|---|
| Final val_loss/joint NaN or Inf? | No | Numerical instability; check grad clip, fp16 overflow, init |
| Final val_loss/joint > 2.5? | < 2.5 | Training catastrophically failed; likely HP/code issue |
| Loss decreasing over last 25% of training? | Monotone or near-monotone | Optimizer not converging; LR mistuned |
| Routing entropy at end > 0.7·log(8) = 1.45? | Yes | Routing collapse; bias mechanism failed |
| MFU > 25%? | Yes | Throughput broken; infrastructure issue |

**If any fail:** halt and diagnose. Do not spend the full grid on broken HP transfer or infrastructure.

**Quantitative bounds estimation:**

For F1.s at d18 multimodal G=2 r=0.3 C=5e18:
- **Expected**: val loss ≈ 1.0-1.2 (extrapolated from nanochat dense d18 ~0.85 + multimodal penalty)
- **Sanity envelope**: [0.7, 1.8]
- **Halt trigger**: loss > 1.8 at end of training, OR loss > 2.0 at any eval

These bounds are loose but catch catastrophic failures. Tighter bounds would risk false positives.

### 4.2 The preregistration discipline (load-bearing)

Between P1.5 fit and Phase 2 big run, **commit prediction to `dev/LOG.md`** in this format:

```markdown
## YYYY-MM-DD: Scaling-law fit prediction (PREREGISTERED — committed before G3)

Source: P1 IsoFLOPs grid completed at git SHA <abc1234>
Cells used: F1.s, F1.m, F1.l, F2.s, F2.m, F2.l, F3.s, F3.m, F3.l

3-point fit:
  C₁ = 5e18, N_opt(C₁) = ___ (winner from F1.s/F1.m/F1.l)
  C₂ = 1.5e19, N_opt(C₂) = ___ (winner from F2.s/F2.m/F2.l)
  C₃ = 3e19, N_opt(C₃) = ___ (winner from F3.s/F3.m/F3.l)

Fitted power law (log-log linear regression, 3 points):
  N_opt(C) = k_N × C^a, where a = ___ ± ___, k_N = ___
  D_opt(C) = k_D × C^b, where b = 1 - a
  R² = ___ (must be > 0.95 for fit acceptance)
  Residuals: ___ (must be < 5% of value for power-law assumption to hold)

Sanity check: a + b = 1.000 ✓ (must hold by construction)

For G3 verification cell at C = 6×10¹⁹:
  Predicted N_opt = ___ params (depth d___, model_dim ___)
  Predicted D_opt = ___ tokens
  Predicted final val_loss/joint = ___ (from Hoffmann loss-surface fit)

This prediction is committed BEFORE running G3 verification cell.
G5 PASS criterion: actual val_loss within ±0.1 of predicted.

Git SHA at preregistration: <hash>
```

**Without preregistration:** free to "adjust" prediction after seeing G3 results → not falsifiable. **With preregistration:** prediction is binding; quality of fit is empirically testable.

This is the SINGLE most important methodology discipline in CS336 and in this adaptation.

---

## 5. Methodology

### 5.1 IsoFLOPs fitting (P1.5 details)

**Step 1:** Load Phase 1 cell results from `runs/manifest.json`.

**Step 2:** For each compute scale, find min-loss N and reject unbracketed endpoint minima:
```python
import json
import numpy as np
from collections import defaultdict

with open("runs/manifest.json") as f:
    manifest = json.load(f)

by_C = defaultdict(list)
for run in manifest['runs']:
    if run['phase'] == '1_fitting':
        by_C[run['compute_budget_target_flops']].append(run)

optimal_points = []
for C in sorted(by_C):
    cells = by_C[C]
    if len(cells) < 3:
        raise ValueError(f"C={C:.0e} has only {len(cells)} cells; need >=3 widths")

    cells = sorted(cells, key=lambda r: r['n_active_params'])
    best_idx, best = min(enumerate(cells), key=lambda x: x[1]['final_val_loss_joint'])
    if best_idx == 0 or best_idx == len(cells) - 1:
        raise ValueError(
            f"C={C:.0e} minimum is at endpoint depth "
            f"{best['architecture_config']['num_hidden_layers']}; add expansion cell"
        )

    optimal_points.append({
        'C': C,
        'N_opt': best['n_active_params'],
        'D_opt': C / (6 * best['n_active_params']),
        'min_loss': best['final_val_loss_joint'],
        'winning_depth': best['architecture_config']['num_hidden_layers']
    })

for p in optimal_points:
    print(f"C={p['C']:.0e}: depth={p['winning_depth']}, N={p['N_opt']:.2e}, loss={p['min_loss']:.3f}")
```

Mock output:
```
C=5e+18:    depth=20, N=3.70e+08, loss=1.025  ← winner of {F1.s, F1.m, F1.l}
C=1.5e+19:  depth=24, N=6.40e+08, loss=0.875  ← winner of {F2.s, F2.m, F2.l}
C=3e+19:    depth=26, N=8.30e+08, loss=0.798  ← winner of {F3.s, F3.m, F3.l}
```

Note these (C, N_opt) pairs are mathematically consistent with slope a≈0.45 (Chinchilla-ish; not Hoffmann's 0.5 but well inside [0.4, 0.6]). If your real Phase 1 produces winners that imply a≈0 (e.g., same N_opt at adjacent C scales — symptom of a flat U-curve), the slope is meaningless and you need either more cells per scale OR a different parametric form.

**Step 3:** Fit power law in log-log space with proper CIs (3 points → can compute residuals + bootstrap CI):
```python
log_C = np.log([p['C'] for p in optimal_points])
log_N = np.log([p['N_opt'] for p in optimal_points])

# Linear fit
a, log_k_N = np.polyfit(log_C, log_N, deg=1)
k_N = np.exp(log_k_N)

# Residuals (power-law assumption check)
fitted_log_N = a * log_C + log_k_N
residuals = log_N - fitted_log_N
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((log_N - log_N.mean()) ** 2)
R_squared = 1 - ss_res / ss_tot

# Bootstrap CI for slope (3 points → resample with replacement)
slopes = []
for _ in range(1000):
    idx = np.random.choice(3, 3, replace=True)
    if len(set(idx)) >= 2:  # need ≥2 unique points
        a_boot, _ = np.polyfit(log_C[idx], log_N[idx], deg=1)
        slopes.append(a_boot)
slope_ci_low = np.percentile(slopes, 5)
slope_ci_high = np.percentile(slopes, 95)

print(f"N_opt(C) = {k_N:.4e} × C^{a:.4f}")
print(f"Slope 90% CI: [{slope_ci_low:.4f}, {slope_ci_high:.4f}]")
print(f"R²: {R_squared:.4f}")
print(f"Max residual (log-space): {np.max(np.abs(residuals)):.4f}")
print(f"Sanity: a + b = {a + (1-a):.4f} ✓ (algebraic identity)")
```

Mock output (for the §5.1 winners shown above):
```
N_opt(C) = 1.43e+00 × C^0.451
Informal slope range from bootstrap: [0.40, 0.50]
R²: 0.993
Max residual (log-space): 0.054  (0.27% relative → power-law assumption holds)
Sanity: a + b = 1.000 ✓ (algebraic identity from C = 6ND)
```

**Gate C decision:** R² = 0.993 (>0.95 ✓), slope 0.451 in [0.4, 0.6] ✓, residuals < 5% ✓. **GATE C PASS.**

**Honest CI caveat:** with only 3 (C, N_opt) points, the bootstrap "CI" is informal. Resampling 3 points with replacement covers very few unique slopes; the range above is directional ("uncertain in this band") not a publishable confidence interval. Real CIs require multi-seed runs or within-curve repeated measurements (out of budget for v1).

### 5.2 Loss prediction

With 9 fitting cells, we can fit Hoffmann's full form as the primary loss predictor:

```python
# Approach: fit L(N, D) = E + A/N^α + B/D^β over all 9 P1 cells
from scipy.optimize import minimize

def hoffmann_loss(params, N_arr, D_arr, L_arr):
    E, log_A, log_B, alpha, beta = params
    A = np.exp(log_A)
    B = np.exp(log_B)
    L_pred = E + A / (N_arr ** alpha) + B / (D_arr ** beta)
    return np.sum((np.log(L_arr) - np.log(L_pred)) ** 2)  # log-space MSE

# Use all 9 P1 cells as training data
N_arr = np.array([r['n_active_params'] for r in p1_runs])
D_arr = np.array([r['actual_tokens'] for r in p1_runs])
L_arr = np.array([r['final_val_loss_joint'] for r in p1_runs])

# Multi-init L-BFGS to find global min (50 random starts)
best_loss = float('inf')
best_params = None
for seed in range(50):
    np.random.seed(seed)
    init = [
        np.random.uniform(0.5, 1.5),  # E
        np.random.uniform(0, 5),      # log_A
        np.random.uniform(0, 5),      # log_B
        np.random.uniform(0.2, 0.6),  # alpha
        np.random.uniform(0.2, 0.6),  # beta
    ]
    result = minimize(hoffmann_loss, init, args=(N_arr, D_arr, L_arr),
                      method='L-BFGS-B',
                      bounds=[(0.5, 2.0), (-10, 10), (-10, 10), (0.1, 1.0), (0.1, 1.0)])
    if result.fun < best_loss:
        best_loss = result.fun
        best_params = result.x

E, log_A, log_B, alpha, beta = best_params
A, B = np.exp(log_A), np.exp(log_B)

# Predict G3 loss at predicted-optimal config
target_C = 6e19
N_pred = k_N * (target_C ** a)  # from §5.1
D_pred = target_C / (6 * N_pred)
L_pred_G3 = E + A / (N_pred ** alpha) + B / (D_pred ** beta)

print(f"Hoffmann fit: L(N, D) = {E:.3f} + {A:.2e}/N^{alpha:.3f} + {B:.2e}/D^{beta:.3f}")
print(f"G3 prediction: N={N_pred:.2e}, D={D_pred:.2e}, predicted_loss={L_pred_G3:.4f}")
```

Mock output:
```
Hoffmann fit: L(N, D) = 1.452 + 8.32e+02/N^0.341 + 1.27e+03/D^0.287

G3 prediction (C = 6×10¹⁹):
  N_opt unconstrained = 1.19e+09 (depth ≈ d30, OUTSIDE d12-d26 envelope ⚠️)
  → Apply envelope cap: depth=d26, N=8.30e+08
  D_opt at capped N = 6e19 / (6 × 8.3e8) = 1.20e+10 ≈ 12B tokens
  Predicted final loss with capped config: 0.78
```

→ Predicted G3 loss: **0.78** (Hoffmann form, capped to envelope).

**G3 pre-flight depth check (per Obs B):** if N_pred maps to depth >d26 (as in the mock above), the design has implicitly extrapolated. **Two responses:**
- (a) Run a single F3.xl expansion cell at d28 ($35-50) to extend the bracket BEFORE G3 — confirms or refutes whether the IsoFLOP minimum is really beyond d26
- (b) Accept the d26 cap and document the bias in the writeup ("G3 trained at the upper envelope edge; predicted-vs-actual loss may underweight the contribution of larger N")

Default: (a) if budget allows; (b) otherwise. **Both are honest; neither is silently extrapolating.**

### 5.3 G3 verification readout

After G3 completes, compute and write to `dev/LOG.md`:

| Quantity | Predicted | Actual | Verdict |
|---|---|---|---|
| Final val_loss/joint | 0.823 | _ | ±0.05 = "fit good"; ±0.1 = "fit acceptable"; >0.15 = "fit failed" |
| N_opt (capped to envelope) | 897M (capped to ~830M @ d26) | (config used) | (depth ≤ d26 enforced) |
| D_opt | 3.71B | (actual tokens trained) | within ±5% expected |

---

## 6. Implementation: minimal `scripts/sweep_runner.py`

Same as before — ~150 lines Python, 1 day work. CLI surface:

```bash
python scripts/sweep_runner.py submit --config configs/scaling_law/F1_s.json
python scripts/sweep_runner.py status
python scripts/sweep_runner.py budget
python scripts/sweep_runner.py fit  # post-Phase-1 fitting + preregister prediction
```

### 6.1 Real workflow trace

```bash
$ python scripts/sweep_runner.py budget
{
  "used_hours": 0.0,
  "remaining_hours": 20.0,
  "total_hours": 20.0
}

# Phase 1, F1.s FIRST (reactive sanity check)
$ python scripts/sweep_runner.py submit --config configs/scaling_law/F1_s.json
SUBMIT: F1.s (estimated 0.58 hr)
DONE: F1.s in 0.49 hr, final_loss=1.105

# If sanity bounds pass, run the remaining 8 fit cells.
$ python scripts/sweep_runner.py submit --config configs/scaling_law/F1_m.json
$ python scripts/sweep_runner.py submit --config configs/scaling_law/F1_l.json
$ python scripts/sweep_runner.py submit --config configs/scaling_law/F2_s.json
$ python scripts/sweep_runner.py submit --config configs/scaling_law/F2_m.json
$ python scripts/sweep_runner.py submit --config configs/scaling_law/F2_l.json
$ python scripts/sweep_runner.py submit --config configs/scaling_law/F3_s.json
$ python scripts/sweep_runner.py submit --config configs/scaling_law/F3_m.json
$ python scripts/sweep_runner.py submit --config configs/scaling_law/F3_l.json

# Fit + preregister after all 9 cells complete and all minima are interior.
$ python scripts/sweep_runner.py fit
N_opt(C) = 4.21e-02 × C^0.530
Slope 90% CI: [0.485, 0.578]
R² = 0.987
Hoffmann form: L(N, D) = 1.452 + 832/N^0.341 + 1273/D^0.287

For C = 6e19:
  Predicted N_opt = ___
  Predicted D_opt = ___
  Predicted final loss = 0.823

PREREGISTERING prediction to dev/LOG.md... DONE
Generated configs/scaling_law/G3_big_run.json from prediction. Review before submitting.

# Phase 2 verification
$ python scripts/sweep_runner.py submit --config configs/scaling_law/G3_big_run.json
```

---

## 7. HP-space restrictions

| Hyperparameter | Sweep? | Value |
|---|---|---|
| `depth` (n_layer) | ✅ Yes (Phase 1 axis) | {18, 20, 22, 24, 26} subset across cells |
| `hidden_size` | Auto-derived | depth × 64 (aspect ratio) |
| `num_attention_heads` | Auto-derived | hidden_size / 128 |
| `head_dim` | Fixed | 128 (H100-native) |
| `total_train_tokens` | Auto-derived | from `--target-flops` and depth |
| `train_batch_size` | Auto-derived | Bergsma `B ∝ D^0.383` (nanochat default) |
| `peak_lr` (matrix Muon) | Fixed | 0.02 |
| `peak_lr` (AdamW emb) | Fixed | 0.3 |
| `warmup_frac` | Fixed | 0.05 |
| `weight_decay` | Fixed | 0.28 |
| `num_experts` (G axis) | Fixed | G=2 |
| `mix_ratio_r` | Fixed | 0.3 |
| `max_seq_len` | Fixed | 4096 |
| `dtype` | Fixed | bfloat16 |
| `model_seed` | Fixed | 0 |

---

## 8. Logging schema

W&B project: **`nanochat`** (matches `scripts/base_train.py:110` default — single project for all studies). Each cell uses `--run <cell_id>` so e.g. `F1_s`, `F1_m`, ..., `G3` appear as distinct runs filterable by name. If a per-study project is wanted later, add a `--wandb-project` CLI flag to `base_train.py` — not blocking for v1.

### 8.1 Headline-tier (feeds the fit + verification)

**Per-eval:** `val_loss/joint`, `val_loss/text`, `val_loss/vision`, bootstrap CIs

**Per-step:** `loss/total`, `loss/text`, `loss/vision`, `step_time_ms`, `mfu`, `r_actual`

**NOTE on per-modality loss semantics (post Track B'' refinement):**
`loss/vision` is NOT "loss at vision token positions" (those are ignore_index = -1
since vision is context, not target — per `dev/multimodal_spec.md` §2.5.5).
It IS "text-token CE loss when ANY vision token appears in the recent attention
window" (default window = 32 tokens). Implemented in `core/dataloader.py
:synthetic_multimodal_loader` (and `core/multimodal_data.py:real_multimodal_loader`)
by tagging text-after-vision positions with `modality_mask=1`.

**`r_actual` semantics:** logged as `n_vision_total / (n_text_total + n_vision_total)`
where both counts are over MODALITY_MASK-tagged positions (not raw image_pad
positions). So r_actual ≈ fraction of valid loss positions that have vision
context. **For r=0.3 target on raw image_pad fraction, expect r_actual ~0.6-0.9**
because the vision-context window inflates the count past the raw vision-token rate.

### 8.2 Diagnostic-tier
Same as v3 §7.2: routing entropy, dead expert counts, grouped_mm overhead, etc.

### 8.3 Per-cell metadata
`cell_id`, `phase`, `git_sha`, `seed`, full `GPTConfig`, `n_active_params`, `flops_per_token_estimated`.

---

## 9. Decision gates (kill switches)

| Gate | Trigger | Pass action | Fail action |
|---|---|---|---|
| **G0 (CPU)** | Local `pytest tests/test_multimodal_joint_forward.py` | Proceed to GPU | Halt; debug `core/multimodal.py` stubs |
| **REACTIVE 0** | F1.s sanity bounds (§4.1) | Continue Phase 1 | Halt; diagnose HP/code issue |
| **B (Phase 1 IsoFLOPs)** | All 9 cells complete; min-N at every compute scale is interior | Proceed to fit | Endpoint min → add 1-2 cells at extended depth |
| **C (Fit sanity)** | R² > 0.95; slope a ∈ [0.4, 0.6]; max residual < 5% | Commit prediction; proceed to Phase 2 | Outside → consider Hoffmann parametric form; if still fails, report misspecification |
| **D (Verification)** | G3 actual loss vs predicted, ±0.1 tolerance | Fit validated; write up | Honest report of failure; diagnose |

---

## 10. Timeline

```
Week 1 (CPU work):
  Day 1-3:  Implement core/multimodal.py stubs (per multimodal_spec.md §2.5)
  Day 4:    Implement scripts/sweep_runner.py (~150 lines Python)
  Day 5:    Generate 10 config JSONs in configs/scaling_law/
  Day 6:    Local CPU joint forward smoke at d=4 → Gate G0

Week 2 (vast.ai 8×H100):
  Day 1 morning:    F1.s (~$9, 0.6 hr) → REACTIVE GATE 0
                    Other 8 fit cells (~12.5 hr serial; parallelizable)
                    → Gate B
  Day 1 evening:    Fit + preregister prediction to dev/LOG.md → Gate C
  Day 2:            G3 verification cell (~4.3 hr) → Gate D
  
  Total GPU: ~17.3 hours, ~$277 before endpoint expansion

Week 3 (CPU analysis):
  Day 1-2:  Plots (3 figures: IsoFLOPs curves, scaling law fit, predicted-vs-actual)
  Day 3-4:  Writeup in dev/LOG.md
  Day 5:    Commit reproducible recipe JSON pinned to git SHA

  ARTIFACT SHIPPED: scaling law + verification + reproducible recipe
```

**Total wall-clock:** ~3 weeks elapsed, ~17.3 GPU-hours before endpoint expansion.
**Total budget:** ~$277 before endpoint expansion.

---

## 11. Risk model

Inherits multimodal-pipeline risks from `sweep_design.md` v3 §5. Adds these scaling-law-specific:

| # | Risk | Mitigation | If materializes |
|---|---|---|---|
| SL1 | **HP transfer to our setup fails** (vast.ai vs Lambda, our patches) | F1.s reactive sanity bounds (§4.1) | Halt after F1.s. Run targeted diagnostic before continuing. |
| SL2 | **3-point fit slope outside [0.4, 0.6]** | Gate C: check R² + slope CI | If R² > 0.95 and slope marginally outside: document, proceed. If R² < 0.95 or slope very different: switch to Hoffmann parametric form (full L(N,D) fit), document. |
| SL3 | **Lowest-loss N at multi-width scale is at endpoint** | Gate B: bracketing check | Add 1 cell at extended depth (e.g., d28 if F2.l won at d26) — accept slight envelope extrapolation with footnote. Cost ~$30. |
| SL4 | **Predicted G3 loss off by >0.1** | Gate D | Honest report. Diagnose: endpoint expansion missing? Hoffmann form misspecified? Genuine non-power-law at our scale? |
| SL5 | **G3 wall-clock > 1.5× predicted** | Cancel trigger at 1.5× budget | Kill cell; either accept partial result or re-rent and retry. |
| SL6 | **Multimodal val loss noisier than text** | Bootstrap CIs on val loss; ≥16 evals per cell | Increase n_evals from 16 to 32 (no extra GPU cost — just more frequent eval). |

**Key risk-management principle:** **reactive detection** at F1.s covers HP-transfer failure before the expensive grid. If it fails, stop and debug; do not continue because the remaining cells would no longer measure a clean scaling law.

---

## 12. Deliverables

### 12.1 `dev/LOG.md` entries (frozen after committed)

- **Preregistered prediction** entry (committed before G3) — written at end of Week 2
- **Final analysis** entry with predicted vs actual table, fit constants, Gate D verdict

### 12.2 Plots (in `dev/plots/`)

1. `isoflops_curves.png`: 3 IsoFLOP curves (loss vs N at C₁, C₂, C₃), showing U-shape and minima
2. `scaling_law_N.png`: log-log scatter (3 points: C, N_opt) + fitted line + extrapolation to G3 + bootstrap CI band
3. `predicted_vs_actual.png`: scatter all 10 cells' predicted vs actual losses (calibration check)

### 12.3 Reproducible recipe

`configs/scaling_law/recipe_final.json`: predicted-optimal config from fit, pinned to git SHA. Reproducible via `python scripts/sweep_runner.py submit --config configs/scaling_law/recipe_final.json`.

### 12.4 Writeup section in `dev/LOG.md` (~500 words)

CS336-style structure:
- **Methodology:** what we did + why (this doc serves as reference)
- **Results:** N_opt(C), D_opt(C), Hoffmann form L(N, D), predicted+actual G3 loss, fit quality (R² + Gate D verdict + slope CI)
- **Honest limitations:** 3 compute-scale fit, single G value, single seed, no μP transfer test, depth capped at d26 unless endpoint expansion is triggered
- **Future work:** add 4th fitting scale, sweep G ∈ {1, 4, 8} (deferred), sweep r values

---

## 13. Relationship to v3 speedrun (clean separation)

This study and `sweep_design.md` v3 are **non-overlapping** — different goals, different cells, different deliverables.

| | v3 speedrun (`sweep_design.md`) | This study (scaling law) |
|---|---|---|
| Goal | Speedrun recipe at d24 (fastest wall-clock to capability) | Scaling law fit + verification (decision-relevant for any C) |
| Cells | 8 (smoke + anchor + baseline + 5 ablations) | 10 (9 fit + 1 verify) |
| Compute scales | 1 (C=2e19 anchored at d24) | 3 fit scales (C=5e18, 1.5e19, 3e19) + 1 held-out verification (C=6e19) |
| G values | 1 (G=2) + 1 ablation (G=1 dense) | 1 (G=2 throughout) |
| Output | Per-knob speedup table + composed recipe | Power law + Hoffmann form L(N,D) + predicted (config, loss) + verification |
| HP transfer detection | v3's anchor cell | Reactive via F1.s sanity bounds |
| Methodology rigor | Karpathy speedrun discipline | CS336 preregistration discipline |
| Cost | $262 | ~$277 before endpoint expansion |

**Sequencing options:**

1. **Either alone** — both are self-contained studies
2. **Speedrun first** ($262) → scaling law next (~$277) = ~$539 total before endpoint expansion
3. **Scaling law first** (~$277) → speedrun next ($230 — drop v3 anchor if F1.s verified HP transfer) = ~$507 total
4. **Both simultaneously** — possible but requires careful budget coordination

**Senior-researcher recommendation:** if budget = $300 and only one study possible, pick by goal:
- Production team asking "fastest recipe?" → v3 speedrun
- Production team asking "what config for compute X?" → this scaling law study
- Research lab wanting publishable result → this study (preregistration + verification = paper-quality)

---

## 14. Cross-references

- CS336 Assignment 3 (Stanford) — original assignment this adapts (faithful)
- `dev/sweep_design.md` (v3 speedrun) — independent parallel study
- `dev/sweep_design_g_sweep_deferred.md` — even more comprehensive G-sweep (deferred)
- `dev/sweep_design_v1.md` — original 7-phase ambition (superseded)
- `dev/multimodal_spec.md` §2.5 — training methodology first principles (load-bearing for HP inheritance assumption)
- `dev/LOG.md` — chronology; preregistration entry from P1.5 lives here
- `core/moe.py` — Karpathy MoE port (G=2 default we use)
- `core/multimodal.py` — multimodal pipeline (15 stubs at write-time; must be done before G0)
- `nanochat/dev/LOG.md` 2026-02-19 / 2026-02-21 — upstream verification; F1.s sanity bounds are derived by extrapolation from Karpathy's d18-d24 published numbers

---

## 15. What this study deliberately is NOT

- ❌ Preemptive HP-transfer anchor — reactive detection via F1.s instead (saves $32, same info value)
- ❌ Speedrun ablations — that's `sweep_design.md` v3's territory
- ❌ G-sweep (G ∈ {1, 4, 8}) — deferred
- ❌ μP HP-transfer pre-test — we inherit Karpathy's HPs; reactively detect via F1.s
- ❌ Modality-asymmetry measurement — FAIR Mar-2026 published this
- ❌ Novel architecture — Qwen3.5-VL + nanochat MoE both verbatim
- ❌ Multi-seed reproducibility — single seed; future work
- ❌ Downstream benchmark eval (MMMU/MMBench) — val loss as proxy
- ❌ FP8, batch tuning, ViT-unfreeze ablations — those are speedrun knobs (v3)

Senior researcher's edge: smallest CS336-faithful design that still has true IsoFLOPs curves, a held-out verification run, and endpoint-expansion gates.

---

## TL;DR

**What this is:** CS336 Assignment 3, faithfully adapted — 2 phases (fit + verify), upgraded to true 3-width IsoFLOPs curves at every fitting compute scale.

**What we ship:**
1. Multimodal MoE scaling law (9-cell IsoFLOPs fit + Hoffmann L(N,D) form) at r=0.3, G=2 with bootstrap CI
2. Predicted-vs-actual loss verification (CS336 grading mechanism)
3. Reproducible JSON config for the predicted-optimal recipe

**Budget:** ~$277 / ~17.3 GPU-hours / 3 weeks elapsed before endpoint expansion.

**10 cells:** 9 fit cells across 3 compute scales (C=5e18, 1.5e19, 3e19) + 1 held-out verification at C=6e19.

**Discipline:** preregister prediction to `dev/LOG.md` BEFORE running G3. Predicted-vs-actual within ±0.1 = pass; otherwise honest report.

**Next action:** confirm execution path, then start Track B (multimodal stubs ~1.5 weeks + sweep_runner.py ~1 day) before spending any GPU money.
