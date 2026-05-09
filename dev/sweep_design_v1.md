# Sweep Design — operational spec for the Plan.md MoE scaling sweep

The keystone document. Converts `docs/Plan.md` §4 prose into a concrete,
phase-by-phase, executable spec with budgets, go/no-go gates, kill
conditions, and logging schemas.

**This is the pre-registration document.** Methodology choices are fixed
here BEFORE seeing data, per Plan §8 #1. Subsequent infrastructure work
(multimodal pipeline, engine integration, fit toolkit, GPU sourcing) is
sized against the requirements specified here — not the reverse.

---

## 0. Goal recap (Plan.md §9, condensed)

> ⚠️ **SUPERSEDED by §0a "CURRENT PLAN v2" below (2026-05-02 scope reduction).**
> The original goal (vary r, sweep G × r × C) was scientifically novel but
> over-scoped for our $1K budget AND beyond what frontier labs publish.
> Scope reduced to fixed-r multimodal granularity scaling.

Original goal: construct **modality-conditional scaling laws** for early-fusion
native multimodal Mixture-of-Experts:

> How do MoE granularity G, active fraction φ, and total/active ratio
> depend on the vision/language data mix r?

Validate via two head-to-head verification runs at scale (~8×10²⁰ FLOPs)
comparing the predicted-optimal config against a DeepSeek-V3-style baseline.

---

## 0a. CURRENT PLAN (v2, 2026-05-02)

After scope reduction from "r as swept axis" to "r fixed at production mix":

### Revised goal

> What is the compute-optimal MoE granularity G\* for early-fusion
> multimodal MoE, at production-realistic vision/text mix r = 0.3?

This is the **multimodal extension of Krajewski 2024 / Towards Greater
Leverage 2025** — both did text-only granularity scaling. Nobody has
published multimodal MoE granularity scaling. Our contribution.

### Revised cells (9 total)

```
G ∈ {1, 4, 8}              ← granularity sweep (Llama-4 / intermediate / DeepSeek-V3 style)
C ∈ {1e19, 3e19, 1e20}      ← three compute scales for IsoFLOP fit
r = 0.3 (fixed)             ← Qwen3.5-style production mix; not swept
N per cell: Chinchilla-implied (1 N per cell, no IsoFLOP curve fit)

Total: 3 × 3 × 1 = 9 cells
```

### Revised budget

```
Phase 0 smoke (1 cell, ~$5)            on 1×H100
Phase 1 LR check (3 cells, ~$90)       on 8×H100; text-only at d6 G=4 (LR transferability
                                       is data-modality-independent; text-only fine)
Main sweep (9 cells, ~$250)            on 8×H100, multimodal at r=0.3

Total: ~$345 (was $595 in v1)
```

### What we drop and why

| Original axis | Status | Reason |
|---|---|---|
| r ∈ {0, 0.5} sweep | DROPPED | Frontier labs don't ship text-only models in 2026; r=0 cells redundant. r=1 (vision-only) academic, not production-relevant. Pick ONE production mix instead. |
| Phase D verification (V1 vs V2) | DROPPED | $1500 verification can't justify on $1K budget; preliminary scaling-law result publishable on its own |
| φ axis sweep | DROPPED | G is the load-bearing question; φ is a separate paper |

### What we keep claiming

✅ "Compute-optimal G\* for multimodal MoE at production mix is X"
✅ "G\* shape across compute scales follows / disagrees with Krajewski monotonic"
✅ "We replicate the granularity debate (DeepSeek-V3 vs Llama-4 vs Qwen3-VL choice) in multimodal context"
✅ Architecture follows Qwen3.5-VL early fusion (video-capable; image used in this study)

### What we cannot claim

❌ "G\* shifts with modality mix" (we don't sweep r)
❌ "Modality asymmetry exists at our scale" (not measured here; published elsewhere)
❌ "V1 outperforms V2 by predicted Δ" (no verification runs at scale)

### Cross-references

- LOG entry 2026-05-02 "Scope reduction: drop r-as-axis, fix at production mix"
- multimodal_spec.md §7 — updated to note r is fixed not swept

---

**Falsifiable thesis** (Plan §2): vision-heavy mixes prefer higher G
and higher φ. Predict the dependence functionally; verify with V1 vs V2.

---

## 1. Sweep architecture — 7 phases, sequenced by risk

Each phase has a **go/no-go gate**. A failed gate either:
- (a) Triggers re-budgeting (e.g., CompleteP fails → 2× compute on Phase A)
- (b) Defers the project (e.g., Phase 0 infra unstable → fix infra first)
- (c) Cuts scope (e.g., budget exhausted → drop later phases)

Phases ordered so highest-risk methodology questions (CompleteP transfer,
infra stability) gate the expensive phases.

```
Phase 0  →  Phase 1  →  Phase 2  →  Phase A  →  Phase B  →  Phase C  →  Phase D
infra       CompleteP   CompleteP   text-only   modality   multimodal  verification
smoke       LR sweep    transfer    granularity sweep      IsoFLOPs    runs
                                    × φ sweep
```

| Phase | Purpose | Cells | GPU-hrs | Rough $ | Go/no-go gate |
|---|---|---|---|---|---|
| 0 | Infra smoke + per-cell timing calibration | 3 | ~30 | ~$15 | Training stable, MFU > 30%, per-cell time within 20% estimate. **Else: defer project** |
| 1 | CompleteP base-shape LR sweep at d_model=384 | 8 | ~25 | ~$13 | Best LR converges, others sit ≥2× worse. **Else: HP transfer is broken — investigate before Phase 2** |
| 2 | CompleteP transfer check at d_model={768, 1536} | 12 | ~35 | ~$18 | Transfer holds within 1.3×. **Else: per-scale tuning → 2× compute on Phase A → re-budget** |
| A | Text-only granularity × active fraction (post-cut: 24 cells) | 24 | ~140 | ~$70 | All cells finish, no NaN, fit recovers known-from-Phase-1 LR optima. **Else: investigate** |
| B | Modality mix r sweep at best (G, φ) | 20 | ~220 | ~$110 | Per-modality loss decomposition is non-degenerate (vision loss ≠ text loss). **Else: tokenizer floor too high, Risk 3 materialized** |
| C | IsoFLOPs × multimodal at r=0.5 | 36 | ~610 | ~$305 | Phase C fit converges with Huber + 50-init; CV held-out scale within 5%. **Else: parametric form misspecified — refit with cubic correction or report negative result** |
| D | Verification — V1 (predicted-optimal) vs V2 (DeepSeek baseline) | 2 | ~880 | ~$440 | V1 predicts measured loss within fitted CI. **Else: scaling law refuted — still publishable** |
| **Total** | | **105** | **~1940** | **~$971** | (excludes 10% buffer = $1068 total commit) |

**Note on $/GPU-hour:** Lambda Labs 8×H100 instance ≈ $25/hr; per-GPU
basis ≈ $3.13/hr (this table). vast.ai may run cheaper if availability
permits. Numbers above use the Lambda figure as conservative.

---

## 2. Phase 0 — Infra smoke + timing calibration

**Goal:** confirm the training stack works end-to-end on real GPU before
committing the full budget.

**Cells (3):**

| Cell | Config | Compute | Expected wall-clock |
|---|---|---|---|
| 0.1 | D4 dense (n_layer=4, d_model=256) | 5×10¹⁸ FLOPs | ~10 min on 8×H100 |
| 0.2 | D4 MoE (G=2, φ=0.25, n_shared=1) | 5×10¹⁸ FLOPs | ~12 min on 8×H100 |
| 0.3 | D6 MoE (G=4, φ=0.10, n_shared=1) | 1×10¹⁹ FLOPs | ~25 min on 8×H100 |

**Pass criteria:**
- All 3 train to completion without NaN/Inf
- For MoE cells: routing entropy at end > `0.7 · log(num_experts)`
- For MoE cells: 0 dead experts in last 10% of training
- MFU > 30% per cell (dense baseline; MoE may be lower per the nanochat
  finding — log it, no hard fail)
- Per-cell wall-clock within 20% of estimate (catches throughput surprises)

**Fail action:** **DEFER PROJECT**. Per Plan §7 Risk 6: "If the training
stack isn't stable after Phase 0, defer the project a week; don't push
on a broken infra."

**Logging:** full schema (§9) for every step.

---

## 3. Phase 1 — LR sensitivity check (Muon + 1/√d transfer validation)

**Goal:** confirm that nanochat's empirical 1/√d AdamW LR scaling
(`dmodel_scale = (model_dim / 768)**-0.5`) is at least near-optimal at
our smaller scales (d4-d8). Karpathy validated this scaling at d12-d26;
we extrapolate ~1.7× below his validated range. This phase is the cheap
insurance check.

**This SUPERSEDES the original CompleteP-based Phase 1+2 design.** We
deliberately do NOT implement CompleteP per Plan §3:
- nanochat's MoE infrastructure uses MuonAdamW + 1/√d scaling
- Implementing CompleteP from scratch would cost ~1 week
- For our budget-constrained preliminary study, the deviation from Plan
  §3 is acceptable and explicitly documented (see Phase 14: Methodology
  audit checklist + dev/LOG.md 2026-05-02 entry on the optimizer
  decision)
- Risk: recovered absolute α/β exponents may not match published AdamW-
  CompleteP-trained scaling laws. Relative effect (G\*(r) shape) should
  be preserved.

### 3a. Phase 1 protocol — 3-cell LR sensitivity check

**Cells (3):**

Single config — pick the cell at the middle of our sweep grid so the
result is most representative:

```
config:    d6 (model_dim=384, n_layer=6) MoE
           G=4 (16 routed experts, top_k=4, n_shared=1)
           r=0.0 (text-only — vision adds no LR concerns since SigLIP2 frozen)
compute:   C = 3×10¹⁹ FLOPs per cell (~2 hr each on 8×H100 BF16)
LR sweep:  3 multipliers on the AdamW group base LRs
           (matrix_lr stays at 0.02 — Muon is scale-invariant; we don't sweep it)
```

| Cell | AdamW LR multiplier | What it tests |
|---|---|---|
| 1.1 | **0.5×** the dmodel_scale-derived LR | Below-optimal LR — should give worse loss |
| 1.2 | **1.0×** the dmodel_scale-derived LR | Karpathy's transferred default — should be ~optimal |
| 1.3 | **2.0×** the dmodel_scale-derived LR | Above-optimal LR — should give worse loss (or NaN) |

**Concrete LRs at d6** (`dmodel_scale = (384/768)**-0.5 = √2 ≈ 1.41`):

| Group | Base × dmodel_scale | 0.5× | 1.0× | 2.0× |
|---|---|---|---|---|
| wte | 0.2 × 1.41 | 0.141 | 0.283 | 0.566 |
| lm_head | 0.004 × 1.41 | 0.0028 | 0.0056 | 0.0113 |
| value_embeds | 0.1 × 1.41 | 0.071 | 0.141 | 0.283 |
| (matrix params via Muon) | 0.02 (unscaled) | 0.02 | 0.02 | 0.02 |

### 3b. Pass criteria

The 1.0× cell (Karpathy's default) should be **at least as good as the
0.5× and 2.0× cells, OR within 5% of whichever is best.**

Specifically:
- **PASS:** `loss(1.0×) ≤ min(loss(0.5×), loss(2.0×)) × 1.05`
- **FAIL:** the 1.0× cell is >5% worse than the better of the two extremes

Plus sanity:
- All 3 cells must complete without NaN/Inf
- Routing entropy at end > 0.7·log(num_experts) for all 3 cells

### 3c. Pass action

If PASS: use Karpathy's defaults (`unembedding_lr=0.004, embedding_lr=0.2,
matrix_lr=0.02, scalar_lr=0.5` × dmodel_scale for AdamW groups) for all
sweep cells in Phase 2+. Document the 3-cell evidence in dev/LOG.md.

### 3d. Fail action

If FAIL: the 1/√d scaling rule is mistuned at our scale. Two options:

**Option F1 (cheap, ~$60):** expand the LR sweep at d6 to {0.25×, 0.5×,
1.0×, 1.5×, 2.0×, 3.0×, 4.0×} = 7 cells × 2 hr = 14 hr × $15 = $210
total. Pick best LR; freeze for the sweep.

**Option F2 (more expensive, ~$200):** also re-validate at d4 and d8
with 3 LRs each = 6 more cells. Establishes per-scale optima, used
instead of dmodel_scale.

**Default fail action: F1.** Spend $200 total on Phase 1 instead of
$90, accept the operational delay.

### 3e. Cost + time

| | Cost | Wall-clock |
|---|---|---|
| Phase 1 nominal (3 cells) | ~$90 (3 × 2 hr × $15/hr × 8 GPUs / 8) | ~6 hr GPU sequential, or ~2 hr if parallel on 8 GPUs |
| Phase 1 if F1 fail-action triggered | ~$210 | ~14 hr |

### 3f. Why this matters for the sweep

The sweep is `(C, G, r)` × Chinchilla-N. If LRs are mistuned at any cell:
- That cell's loss is artificially high
- Cell looks worse than it is
- Recovered G\* is contaminated by HP noise, not architecture

Phase 1 spends $90 to gain confidence that this contamination is bounded.
At our budget ($1000 total sweep), $90 is 9% of budget for a load-bearing
sanity check. Worth it.

**Without Phase 1**: we trust Karpathy's HPs at scales he didn't directly
validate. Probably fine, but no evidence.

**With Phase 1**: we have empirical evidence the LRs are within 5% of
optimal at our middle scale. Defendable in the writeup.

---

## 4. Phase 2 — (DEFERRED — was CompleteP transfer check)

**Status:** This phase was originally designed to validate CompleteP HP
transfer per Plan §3. We deliberately do NOT implement CompleteP for
this preliminary study — see Phase 1 above for the rationale and the
3-cell LR sensitivity check that replaces it.

**If results from the main sweep look anomalous** (e.g., wildly different
loss curves at different scales suggest mistuning), revisit by adding a
proper per-scale LR validation (Option F2 in Phase 1 fail action) at
small additional cost.

---

## 5. Phase A — Text-only granularity × active fraction (post-cut)

**Goal:** calibrate the dense MoE scaling law against existing work
(Krajewski 2024, "Towards Greater Leverage" 2025 — Plan §1 Finding 3).
Sanity check that we recover known results before adding modality.

**The cut.** Plan §4 says "cut Phase A from 48 → 24 runs." Concrete
choice:

- Original: 3 compute × 4 G × 4 φ = 48
- Cut: drop the φ extremes (φ=1.0 is dense baseline, separate runs;
  φ=0.05 is ultra-sparse, less informative for granularity question)
- Keep φ ∈ {0.10, 0.25}: 3 × 4 × 2 = **24 cells**

**Cells:**

| C (FLOPs) | N_active (≈ Chinchilla-optimal) | n_layer (named) | D (tokens) | Wall-clock per cell |
|---|---|---|---|---|
| 2×10¹⁹ | ~13M | D4 | ~260M | ~30 min on 8×H100 |
| 6×10¹⁹ | ~22M | D6 | ~450M | ~80 min |
| 2×10²⁰ | ~41M | D8 | ~820M | ~270 min |

For each (C, N) pair: sweep `G ∈ {1, 2, 4, 8} × φ ∈ {0.10, 0.25}` = 8 cells.
Total 3 × 8 = **24 cells**, ~140 GPU-hours.

**Add to `core/configs.py`:** named configs `D4_MoE`, `D6_MoE`, `D8_MoE`
parameterized by `(G, φ, n_shared)`. (Currently `core/configs.py` only has
D12-D26 dense.)

**Pass criteria:**
- All 24 cells complete; no NaN.
- Per Plan §1 Finding 3: at fixed C, loss should vary monotonically with G
  in some range (Krajewski's monotonic claim) OR show a clear interior
  optimum (the "Towards Greater Leverage" Jul 2025 contradiction). Either
  result is publishable; flat-with-G is suspicious and triggers
  investigation.
- Fit `L(N, D, G)` per Phase A data with `core/scaling_fit/`. Recovered
  α, β must be within published ranges (Hoffmann α≈0.34, β≈0.28 for
  dense; modern MoE 6-28 tokens/active-param per Plan §1 Finding 3 —
  expect β ≈ 0.20-0.35 range).

**Fail action:** if α/β way out of range, methodology bug. Investigate
before Phase B.

**Cancel trigger per cell:** loss at step 100 > 1.5× expected → kill
cell, investigate; re-run after fix.

**Output:** Phase A scaling law fit; best (G, φ) at each compute scale;
publishable text-only result on its own (could be a smaller paper before
the full multimodal contribution).

---

## 6. Phase B — Modality mix sweep

**Goal:** sweep r ∈ {0.25, 0.50, 0.75} at the best (G, φ) from Phase A,
plus G and φ sweeps at r=0.5 to calibrate Phase C.

**Prerequisites:**
- Multimodal pipeline complete (Cosmos-Tokenizer, interleaved data, early-fusion)
- Tokenizer floor measured (Plan §7 Risk 3 mitigation — see §11 below)

**Cells (20):**

| Subgroup | Configs | Cells |
|---|---|---|
| B.1 mix sweep | (G_best, φ_best) × r ∈ {0.25, 0.5, 0.75} × 3 compute | 9 |
| B.2 G sweep at r=0.5 | G ∈ {1, 2, 4, 8} × φ_best × 1 compute (middle) | 4 |
| B.3 φ sweep at r=0.5 | G_best × φ ∈ {0.05, 0.10, 0.25, 0.50} × 1 compute | 4 |
| B.4 dense baseline at r=0.5 | dense × 3 compute | 3 |
| **Total** | | **20** |

**Per-cell config:** average ~2×10¹⁹ FLOPs each. Wall-clock ~10 hr/cell
on 8×H100 BF16 for D6-D8 scale models.

**Pass criteria:**
- For r > 0 cells: per-modality loss decomposition is non-degenerate
  (vision loss > text loss is fine; vision loss == text loss within
  noise is suspicious — tokenizer floor saturated)
- Routing entropy stays above 0.7·log(E) for all cells
- For mixed-modality cells: at least 1 expert per layer shows per-modality
  usage ratio > 1.5:1 (specialization signal exists; basics/moe/ found 2.0
  is a marginal threshold at toy scale, expect cleaner signal at production)

**Fail action:**
- Per-modality loss saturated → Risk 3 materialized → measure tokenizer
  floor explicitly (§11), report in LOG, possibly re-run with
  higher-resolution Cosmos-Tokenizer (1024 tokens/image instead of 256).
- No specialization signal → modality asymmetry isn't strong enough at
  this scale → still publishable as "negative result on specialization"
  per Plan §6.

**Output:** modality-aware (G, φ) preferences; informs Phase C grid;
per-modality scaling-law calibration.

---

## 7. Phase C — IsoFLOPs × multimodal

**Goal:** the main multimodal IsoFLOPs grid for the modality-conditional
fit. Plan §4 says 36 cells; that stands.

**Cells (36):**

`r = 0.5` fixed (best signal-to-noise from Phase B). Sweep:

| Axis | Values | Cardinality |
|---|---|---|
| Compute C | {2×10¹⁹, 6×10¹⁹, 2×10²⁰} | 3 |
| Granularity G | {2, 4, 8} | 3 (drop G=1 dense — covered separately) |
| Active fraction φ | {0.05, 0.10, 0.25} | 3 |
| Compute scales × G × φ | | **27** |

Plus 9 dense baselines at the same (C, N) for Phase D's "compute-matched
N_active" comparison: **36 cells total.**

**Compute:** ~3×10¹⁹ FLOPs/cell average → ~610 GPU-hours.

**Use FP8** (per Plan's compute reduction; 1.7× throughput). FP8 gates:
- TransformerEngine integration must be in place
- Per nanochat dev/moe_fp8.md guidance: routed experts (3D nn.Parameter)
  stay bf16; shared expert (Linear) gets FP8'd. Document this as
  partial conversion in LOG.

**Pass criteria:**
- All 36 cells complete
- Approach 3 fit `L(N, D, G)` for Phase C (modality-mix held at r=0.5)
  converges with Huber + 50-init
- Held-out CV across compute scales: predicted vs measured loss at the
  held-out scale within 5% (Plan §5.2)

**Fail action:**
- If CV says >5% disagreement → parametric form misspecified
- Add cubic correction term `+ C/N^(3α)` per `core/scaling_fit/spec.md` V5
  protocol; refit
- If still >5% → form is wrong; report explicitly (negative result on
  parametric form choice; valuable methodology contribution)

**Output:** Phase C scaling law fit at r=0.5; predicted-optimal (G, φ, N)
config for Phase D V1.

---

## 8. Phase D — Verification (the headline result)

**Goal:** head-to-head test of the modality-conditional law against a
frontier baseline.

**Cells (2):**

| Run | Config | Compute |
|---|---|---|
| **V1** | Predicted-optimal from Phase C fit, at r=0.5 | 8×10²⁰ FLOPs |
| **V2** | DeepSeek-V3-style baseline: G=8, φ=0.06, same N_active and N_total budget as V1, at r=0.5 | 8×10²⁰ FLOPs |

**Compute:** each ~880 GPU-hours wall-clock (~3 days each on 8×H100 with
FP8). **This is half the total budget.**

**Pass criteria** (per Plan §6):
- **V1 wins by predicted Δ ± uncertainty** → modality-conditional law
  validated
- **V1 loses** → fit was wrong, here's where, here's what we'd change
  (still publishable per Plan §6)
- **V1 wins by less than predicted Δ** → law's direction right, magnitude
  off (also publishable)

**All three outcomes are publishable.** Plan's central scientific honesty.

**Output:** the headline result for Plan.md's paper.

---

## 9. Per-cell logging schema

Every cell logs to W&B project `multimodal-moe-scaling`. Schema:

### Per-step (every step, lightweight)

| Field | Type | Notes |
|---|---|---|
| `step` | int | |
| `loss/total` | float | training loss |
| `loss/text` | float | per-modality decomposition (Plan §5.3); for r>0 cells |
| `loss/vision` | float | per-modality decomposition; for r>0 cells |
| `grad_norm/total` | float | post-clip total grad norm |
| `lr/muon`, `lr/adamw_emb`, ... | float | per-group LR (sanity for CompleteP transfer) |
| `tokens_per_sec` | float | throughput |
| `mfu` | float | computed via `core/common.py:get_peak_flops()` (per nanochat); add if missing |
| `step_time_ms` | float | wall-clock per step |

### Per-50-steps (medium frequency)

| Field | Type | Notes |
|---|---|---|
| `routing/entropy_per_layer` | list[float] | one per MoE block |
| `routing/max_share_per_layer` | list[float] | dominant expert's share per block |
| `routing/n_dead_per_layer` | list[int] | experts with 0 tokens in window |
| `routing/router_bias_norm` | float | how far the bias has drifted |
| `grad_norm/router`, `grad_norm/experts`, `grad_norm/attn` | float | per-group |

### Per-eval (every N steps; N = 5% of total cell steps, ≥10 evals/cell)

| Field | Type | Notes |
|---|---|---|
| `val_loss/text` | float | held-out text validation loss |
| `val_loss/vision` | float | held-out vision validation loss |
| `val_loss/joint` | float | weighted by mix ratio r |
| `routing/specialization_text` | list[float] | per-expert per-modality usage ratio (text bias); per Shukor 2025 + Plan §5.4 |
| `routing/specialization_vision` | list[float] | per-expert per-modality usage ratio (vision bias) |
| `routing/specialization_score` | float | aggregated `1 - H(p_modality)` per Shukor §5.4 |

### Per-cell metadata (write once at start, save with checkpoint)

| Field | Type | Notes |
|---|---|---|
| `cfg.dump` | dict | full GPTConfig dataclass dump |
| `sweep.phase`, `sweep.cell_id`, `sweep.parent_phase` | str | trace-ability |
| `seed` | int | |
| `git_sha` | str | reproducibility |
| `gpu_specs` | dict | which GPU model, driver, CUDA version |
| `dtype_compute` | str | "bfloat16" or "fp8_e4m3" |
| `n_active_params`, `n_total_params` | int | from `model.num_scaling_params()` |
| `flops_per_token_estimated` | int | from `model.estimate_flops()` |
| `cell_budget_target_flops` | int | the C value this cell aims to hit |

---

## 10. Per-cell config dataclass

Add to `core/sweep/`:

```python
@dataclass
class SweepCell:
    # Identity
    phase: str                  # "0", "1", "2", "A", "B", "C", "D"
    cell_id: str                # e.g. "A.D6.G4.phi010" (human-readable)
    parent_phase: str | None    # which phase's output feeds in (e.g. C → D)
    
    # Model config (extends GPTConfig)
    model_config: GPTConfig     # n_layer, n_embd, moe fields, etc.
    
    # Training config
    n_steps: int                # determined from compute_budget_flops + flops_per_step
    batch_size: int             # tokens per step
    sequence_len: int           # 4096 per Plan §3
    lr_base: float              # CompleteP-transferred from Phase 1
    optimizer_kind: str         # "MuonAdamW"
    dtype: str                  # "bfloat16" or "fp8_e4m3"
    
    # Data config
    text_data: str              # "DCLM-baseline" path
    vision_data: str | None     # "LAION-Recap-12M" path; None for r=0
    interleaved_data: str | None # "OBELICS-style" path; for r ∈ (0, 1)
    mix_ratio_r: float          # 0 (text-only) to 1 (vision-only)
    tokens_per_image: int       # 256 (Cosmos-Tokenizer default)
    
    # Budget + cancel triggers
    compute_budget_flops: int
    wallclock_budget_seconds: int
    cancel_if_loss_above: float   # at step 100, kill if loss > this
    cancel_if_mfu_below: float    # sustained over 1000 steps
    
    # Reproducibility
    seed: int
    
    # Logging
    wandb_project: str = "multimodal-moe-scaling"
    eval_every: int = -1        # auto = max(100, n_steps // 20)
```

The sweep driver (`scripts/run_sweep.py`, future) iterates `list[SweepCell]`,
runs each, handles failures.

---

## 11. Tokenizer floor measurement protocol (Risk 3)

Plan §7 Risk 3: VQ tokenizer reconstruction error puts a floor on vision
loss; if floor is high, modality asymmetry signal saturates.

**Protocol** (one-time, ~$30):
- Train a small (D4) model on **vision-only** data (r=1) until convergence
  on the validation set
- Inspect plateau loss: `loss_floor_vision`
- Compare to expected text loss at the same N: `loss_text_at_same_N`
- Report ratio: `loss_floor_vision / loss_text_at_same_N`
- **PASS:** ratio in [1.0, 2.0] (vision is somewhat harder than text but
  not pinned by tokenizer)
- **FAIL:** ratio > 3.0 → tokenizer floor too high; consider 1024-token-
  per-image Cosmos variant; re-budget if switching

This runs as part of Phase 0 if Cosmos-Tokenizer is integrated by then,
else as Phase B prerequisite.

---

## 12. GPU sourcing decision

**Decision point:** after Phase 0 passes (commit ~$15 to validate infra
on 1×H100 first).

**Primary:** Lambda Labs 8×H100 instance, $25/hr × 168hr = **$4,200**
for 1 wall-clock week.

**Backup:** vast.ai, ~$15-20/hr equivalent if available; OR Modal at
$30/hr but easier orchestration.

**Pre-Phase-0 work** (cheap GPU rental, ~$15-30 total):
- Validate `core/moe/` verifier passes on real GPU (~$10)
- Run P3 grouped_mm validation when implemented
- Run tokenizer floor measurement (~$30)

**Total budget commitment:** $4,500 (10% buffer over $4,200) once
Phase 0 passes.

**Cancel:** if Phase 0 fails infra checks, defer the Lambda commit by
1 week per Plan §7 Risk 6.

---

## 13. Cancel triggers / kill conditions

Apply to every cell in every phase:

| Trigger | Action | Why |
|---|---|---|
| NaN/Inf in loss at any step | Kill cell, log, restart from last checkpoint with grad-clip 1.5× tighter | Training instability; per Plan §7 Risk 1 |
| Loss > 1.5× expected at step 100 | Kill cell, investigate config | Either bug or genuinely bad config — both warrant pause |
| Sustained MFU < 25% for 1000 steps | Kill cell, infrastructure issue | Per nanochat finding, MoE expects ~35-40% MFU; <25% is broken |
| Cell exceeds 2× wall-clock budget | Kill cell | Time is money; 2× over means estimate was wrong, re-plan |
| Routing entropy < 0.5·log(E) for 100 consecutive steps | Kill cell | Routing collapsed; bias mechanism failed; investigate before continuing |
| Any expert receives 0 tokens for >100 consecutive steps | Kill cell | Dead expert; bias mechanism failed |
| W&B logging fails for >50 consecutive steps | Pause cell, fix logging, resume | Don't run blind |

**No silent failures.** Every kill produces a `dev/LOG.md` entry.

---

## 14. Methodology audit checklist

Per Plan §8 (frontier-grade vs paper-mill). For the eventual paper:

- [ ] **Pre-registered hypothesis** committed to version control before
      Phase B starts (this doc IS that pre-registration)
- [ ] **CompleteP / μP-MoE empirical validation** before Phase A
      (Phase 1+2)
- [ ] **Modality-conditional Approach-3 fit** with Huber loss, multi-init
      (50), bootstrap CIs (block at config level), held-out CV across
      compute scales (per `core/scaling_fit/spec.md`)
- [ ] **Two head-to-head verification runs** at scale (Phase D)
- [ ] **Expert specialization measurements** across phases B, C, D
      (Plan §5.4)
- [ ] **Honest reporting:** every run logged to `dev/LOG.md`; every config
      dumped; every disagreement with literature investigated
- [ ] **Limitations section:** vocab fixed, context fixed, no audio,
      no NSA, no inference-aware optimization, no RLHF (Plan §8 #7)

---

## 15. What this doc specifies for other infrastructure

This sweep design is the spec other components must satisfy:

### `core/configs.py`
- Add named configs: `D4_MoE(G, φ, n_shared)`, `D6_MoE(...)`, `D8_MoE(...)`
- Existing D12-D26 dense remain (used in Phase D potentially)
- Each named config sets `n_layer`, `n_embd`, `n_head`, `n_kv_head`,
  `sequence_len=4096`, `window_pattern="L"` (no sliding window per Plan §3)

### `core/engine.py` (P4)
- Per-step logging schema (§9 above)
- Modality-decomposed loss (per `loss_text` / `loss_vision` masks based
  on token modality tag)
- Cancel-trigger evaluation hooks
- Gradient-accumulation-aware bias update (`update_moe_load_balance()`
  fires ONCE per optimizer step, not per micro-batch)
- Multi-GPU `dist.all_reduce` of `_token_counts` BEFORE bias update
  (per `core/moe/spec.md` §9 scope guard, deferred to P4)

### Multimodal pipeline (Plan §10 items 1-3)
- Cosmos-Tokenizer integration with **256 tokens per image** (Plan §3)
- Unified vocab: 32K text + 16K visual = 48K (Plan §3)
- Interleaved data loader supporting `mix_ratio_r ∈ [0, 1]`
- Per-token modality tag (for §9 loss decomposition)
- Data sources: DCLM-baseline (text), LAION-Recap-12M (vision),
  OBELICS (interleaved) — per Plan §4
- **Decision needed:** which Cosmos-Tokenizer variant (Cosmos-Tokenizer
  is a family; pick the discrete-image variant compatible with 256
  tokens/image at our chosen resolution)

### `core/scaling_fit/` (deferred per spec.md)
- Re-scope to text-only `L(N, D, G)` for Phase A consumption first
- Add modality decomposition `L(N, D, G, r)` for Phase B+C consumption
- Verifier C5 (CV detects misspecification) as Phase C pass criterion
  (§7 above)

### Checkpoint format
- Save 3D MoE weights, `router_bias` buffer (persistent), training step,
  optimizer state, RNG state, sweep cell metadata
- Resume must reproduce loss curve bit-exactly for at least 100 steps
  (sanity on serialization correctness)

### W&B project setup
- One project: `multimodal-moe-scaling`
- One run per cell, named by `cell_id` (§10)
- Tags: `phase`, `dtype`, `r`, `G`, `phi`
- Group runs by phase for easy comparison

---

## 16. Total compute budget — concrete numbers

| Phase | Cells | GPU-hrs | $/cell ($25/8GPU/hr basis) | Phase total $ |
|---|---|---|---|---|
| 0 | 3 | 30 | ~$5 | ~$15 |
| 1 | 8 | 25 | ~$1.5 | ~$13 |
| 2 | 12 | 35 | ~$1.5 | ~$18 |
| A | 24 | 140 | ~$3 | ~$70 |
| B | 20 | 220 | ~$5.5 | ~$110 |
| C | 36 | 610 | ~$8.5 | ~$305 |
| D | 2 | 880 | ~$220 | ~$440 |
| **Subtotal** | **105** | **1940** | | **~$971** |
| Buffer (10%) | | 194 | | ~$97 |
| **Total commit** | | **~2134** | | **~$1068** |

**Compared to Plan §4 estimate** (1344 GPU-hours after FP8 + Phase A
cuts): we're **40% over.** Three options to close the gap:
1. Skip Phase 2 if Phase 1 looks really clean (-35 GPU-hrs)
2. Skip Phase B.4 dense baselines if Phase A dense baselines suffice (-30)
3. Reduce Phase D verification runs to 6×10²⁰ FLOPs each (-300)
4. Run Phases A+B+C exclusively in FP8 (1.7× throughput) (-560)

**Recommended:** option 4 (FP8 throughout) brings total to ~$1100, well
within $4,200 1-week budget. FP8 partial conversion (per nanochat
dev/moe_fp8.md): shared expert FP8'd, routed experts stay bf16. Document
in LOG.

---

## 17. Sequencing — one-page execution checklist

```
WHEN GPU ACCESS LANDS:

[ ] 1×H100 rental (~$15 total)
    [ ] Run scripts/verify_core_moe.py — must 5/5 PASS on real GPU
    [ ] If P3 grouped_mm is implemented: run its verifier (numerical
        equivalence with loop)
    [ ] Run tokenizer floor measurement (Phase 0 prerequisite)

[ ] DECISION POINT: commit to 8×H100 1-week rental? ($4,500)
    [ ] If Phase 0 prereqs all pass → YES, book Lambda
    [ ] If anything looks fragile → defer 1 week, fix infra

[ ] On 8×H100 (Day 1):
    [ ] Phase 0 (~30 GPU-hrs, ~4 wall-clock hrs)
    [ ] GO/NO-GO gate

[ ] Day 1-2:
    [ ] Phase 1 (~25 GPU-hrs)
    [ ] Phase 2 (~35 GPU-hrs)
    [ ] CompleteP transfer GO/NO-GO

[ ] Day 2-3:
    [ ] Phase A (~140 GPU-hrs)
    [ ] Phase A scaling law fit; sanity vs Krajewski/literature
    [ ] If Phase A passes: PUBLISHABLE TEXT-ONLY RESULT (could be a 
        smaller paper while Phase B+C+D run)

[ ] Day 3-5:
    [ ] Phase B (~220 GPU-hrs)
    [ ] Tokenizer floor confirmation; modality decomposition non-degenerate

[ ] Day 5-6:
    [ ] Phase C (~610 GPU-hrs); FP8 throughout
    [ ] Phase C fit; CV detect misspecification?
    [ ] Predict V1 config

[ ] Day 6-8:
    [ ] Phase D (V1 + V2, ~880 hrs each, parallel — both fit on 8 GPU
        wall-clock if run sequentially over 6 days, or split if budget
        allows running on 4+4 split)

[ ] Day 8-10:
    [ ] Analysis: compare V1 vs V2; write up findings; LOG.md entry
    [ ] If V1 wins → headline result; if V1 loses → still publishable
        per Plan §6
```

---

## 18. What this doc does NOT specify (deliberately)

- **Specific multimodal pipeline implementation** — separate spec, sized
  against §15 requirements above
- **Specific engine.py modifications** — derived from §9 logging schema
  and §15 engine requirements
- **Specific fit toolkit code structure** — re-scope `core/scaling_fit/`
  per §15 (text-only first, modality later)
- **Specific Cosmos-Tokenizer variant** — flagged as decision needed in §15
- **Specific prompt formats / dataset preprocessing details** — to be
  spec'd as part of multimodal pipeline build
- **Specific paper outline / claims** — emerges from data, not pre-spec'd

---

## 19. Cross-references

- `docs/Plan.md` — the project goal this design serves
- `core/moe/spec.md` — MoE layer current state (P1+P2 shipped)
- `core/scaling_fit/spec.md` — fit toolkit spec (deferred per pushback;
  re-scope per §15)
- `dev/LOG.md` — chronology of decisions and findings
- `dev/GPU_QUEUE.md` — P3 grouped_mm + other GPU-blocked work
- `nanochat/dev/LOG.md` — reference for MoE wall-clock / FP8 / format
  conventions

---

## 20. What "current state" means for this doc

This is **v1 of the sweep design**, written before any sweep cells have
run. As phases execute, this doc gets updated with measured numbers
(replacing estimates), failed gates trigger LOG entries that may modify
subsequent phases, and sections become "what happened" rather than
"what we plan."

**If the sweep doesn't end up matching this doc, write the deviation in
`dev/LOG.md` with reasoning.** Don't quietly drift.
