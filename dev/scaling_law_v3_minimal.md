# Scaling-Law v3 (minimal) — first-principles redesign

> Delta-spec on top of `dev/scaling_law_self_assignment.md`. Reviewed end-to-end from a frontier-lab senior-researcher lens. Cuts everything not load-bearing for a single falsifiable claim. Treats `nanochat/dev/LOG.md` as **measured prior**, not literature.
>
> **Reading order:** §1 first (you must sign off on the hypothesis fork before any of this is valid). §5 is the most important section — that is where the redesign earns its keep.

**Owner:** operator-researcher.
**Status:** v3 minimal design. Supersedes the 10-cell plan in `dev/scaling_law_self_assignment.md` *only if* §1A is selected. If §1B is selected, this doc is incomplete and we expand it.

---

## §1. The hypothesis fork — pick one, in writing

There are two questions a senior reviewer would not let you conflate:

| Option | Question this study answers | What you commit to | Cell envelope |
|---|---|---|---|
| **A** | "Does a clean multimodal-MoE scaling law exist at our setup, and does it predict held-out loss within ±0.1 nats at C = 6×10¹⁹?" | G = 2 fixed throughout. Modality mix r = 0.3 fixed. Output is a single Hoffmann fit on the actual target architecture + verification. G\*(r) and modality decomposition explicitly deferred. | **5 cells, ~$216** |
| **B** | "How does G\* shift with modality mix r — the contested DeepSeek vs Llama-4 question?" | G axis (at least 3 values) × r axis (at least 2 values) × ≥2 compute scales. Output is G\*(r) curve with CIs. | 18+ cells, ~$700+ |

**This doc specifies Option A.** Option B is the headline science from `docs/Plan.md` and `docs/SL_VI.md` §1, and it remains the project's eventual contribution — but Option A is the **prerequisite**. Without a verified scaling-law fit on our pipeline first, every G\*(r) datapoint is a measurement of an uncalibrated instrument.

**If you want B instead, stop reading and we re-spec.** The cell budget, the gates, and the falsification target are all different. Do not silently treat A as B's substitute.

The rest of this document assumes A is selected.

---

## §2. What nanochat already proved — the audit (not "assumed transferable")

Frontier-lab discipline: separate **measured** from **assumed**. The current spec (`scaling_law_self_assignment.md` §15) lists nanochat as one of many references. That under-sells the actual leverage. nanochat is **measurement data we already paid for**.

Re-cast `docs/SPEC.md` §3.2's T-table into "what we can inherit vs what we cannot":

### 🟢 Verified-transferable (re-using is sound, no cell needed)

| Tag | What nanochat measured | Why it transfers without re-test |
|---|---|---|
| **T2** | `dmodel_lr_scale = (d_model/768)^-0.5` anchored at d=24 | Pure architectural HP scaling rule. Optimizer (Muon + AdamW) is unchanged from nanochat. We use the same code path. |
| **T3** | nanochat MoE: 8 routed + 1 shared, top-2 sigmoid, aux-loss-free balancing, `_grouped_mm` dispatch, Muon-compatible 3D tensors, active-FLOP accounting | We **port the same code** (`core/moe.py`). Not "transfer" — same artifact. |
| **T4** | MoE per-step quality improves at d18; MFU regresses 46 → 35% (`_grouped_mm` overhead) | Confirms G = 2 MoE is operational at our depth range. Calibrates the **wall-clock budget**: plan for 35% MFU, not 47%. |
| **T6** | ClimbMix-400B > FineWeb-EDU; d24 reaches GPT-2 in ~2h on 8×H100 (~7B tokens) | Anchors expected loss values at d24, C ≈ 2×10¹⁹. **This is the single most load-bearing inheritance** — it tells us what loss to expect at our anchor cell before we burn it. |
| **T7** | `B_opt ∝ D^0.383` (Bergsma) tuned at d12 with B=2¹⁹ | Same batch-size formula in `base_train.py`. No retune. |

### 🟡 Validated-band-limited (transfers inside a regime, fails outside)

| Tag | What we know | Boundary |
|---|---|---|
| **T2-extended** | LR-transfer formula is anchored at d=24. Empirically validated in d ∈ [20, 28]. Below d ≈ 16 it overshoots (auto sessions 1–2 found MATRIX_LR ≈ 0.04 optimal at d=8, the formula predicts ~0.035 too high). | **Stay inside d ∈ [20, 28] for the fit.** Older spec used d18; cut. |
| **T4-mfu** | MFU regression for MoE measured at d18. At larger d it may recover, but not measured. | Budget for 35% MFU at d=20–22, possibly 40–45% at d=26. Reconcile at first cell. |

### 🔴 Cited but unverified-at-our-setup (do not anchor against without test)

| Tag | Claim | Why we cannot assume it holds |
|---|---|---|
| **T10** | CompleteP (Cerebras 2025) HP transfer for width+depth, dense | Muon compatibility unverified. Outside our scope. |
| **T11** | μP-for-MoE 2025 routing-LR scaling | Not primary-verified. Outside scope. |
| **T12** | **FAIR/NYU Mar 2026: α_vision < α_text and β_vision > β_text** | **Load-bearing in `docs/Plan.md` §1 Finding 2. We have not independently measured this.** Must be tested, not assumed. See §3. |

### What we can NOT inherit from nanochat

- **Anything multimodal.** nanochat is text-only. The moment we add vision tokens at r = 0.3, every loss number nanochat published becomes a different quantity.
- **Anything at d < 20 or d > 28.** Out of validated LR band.
- **FP8.** nanochat's verdict was "FP8 + MoE incompatible" (T5). We stay BF16.

### Net result of the audit

**Phase 1's first cell does not need to fit anything.** It needs to **verify the inheritance is real** — that running our pipeline at d=24, dense, text-only, C ≈ 1.5–2×10¹⁹ reproduces nanochat's published number within 5%. If yes, every 🟢 row above is locked in. If no, the whole scaling-law study halts until divergence is diagnosed.

This single insight collapses the old 9-cell IsoFLOPs grid into something much smaller.

---

## §3. What this study adds beyond nanochat — the delta

nanochat has dense text-only at d∈[20, 28]. **This study measures the scaling law of the actual target architecture (MoE G=2 + multimodal r=0.3) and verifies it on a held-out compute scale.** We do not attempt to decompose the delta into vision-vs-architecture components — that decomposition requires dense+multimodal cells (B1, B2) which are cut from this design.

The fit:

```
L_MoE_MM(N_active, D) = E + A / N^α + B / D^β            (Hoffmann form)
```

fit on data from C1, C2 (two compute scales of the target architecture), then used to predict the optimal (N\*, D\*) at C = 6×10¹⁹. V3 trains at that prediction; predicted vs actual loss is the falsification step.

**What we can claim from this design:**
- Our MoE+multimodal pipeline has a scaling law of fitted form X with predicted exponents (α, β).
- That fit predicts held-out loss at C = 6×10¹⁹ within ±Y nats.

**What we cannot claim from this design (deferred):**
- Whether modality asymmetry (FAIR Mar 2026, T12) is present — needs dense+MM cells we cut.
- How G or r shift the law — needs the Option B sweep.
- Whether MoE per-cell quality from nanochat T4 translates to changed exponents or just a constant offset.

**Why this is still worth running:** a verified scaling law on the actual target architecture is the prerequisite for every downstream architectural study (G\*(r), r-sweep, μP transfer). Without it, every follow-up datapoint is measurement on an uncalibrated instrument. This study delivers the calibrated instrument.

---

## §4. Cell list — 5 cells, ~$216

Every cell answers exactly one question. Cells with no question get cut. Cells whose question is answered by a 🟢 nanochat row get cut. Cells that decompose deltas we don't claim to publish get cut.

| # | Cell ID | Config | C (FLOPs) | Depth | What it isolates | Wall-clock (35% MFU) | Cost |
|---|---|---|---|---|---|---|---|
| 1 | **A0** | dense, text-only, r=0 | 1.5×10¹⁹ | d24 | **Instrument calibration vs nanochat T6.** Reproduce expected val_bpb within 5%. **Not a fit point** — this is the measurement-instrument check that locks 🟢 inheritance before any multimodal cell runs. If fail, halt project; subsequent cells would share the bias. | ~1.6 hr | ~$25 |
| 2 | **C1** | MoE G=2, multimodal r=0.3 | 5×10¹⁸ | d20 | Lower fit point on the actual target architecture. | ~0.8 hr | ~$13 |
| 3 | **C_mid** | MoE G=2, multimodal r=0.3 | 1.5×10¹⁹ | d24 | **Middle fit point. REQUIRED for parameter identifiability** — without a third unique N, Hoffmann α is mathematically non-identifiable (the E ↔ A·N^-α trade-off is unconstrained). Verified empirically; see §5 cut #2 revision. | ~1.7 hr | ~$28 |
| 4 | **C2** | MoE G=2, multimodal r=0.3 | 3×10¹⁹ | d26 | Upper fit point on target architecture. Together with C1, C_mid → identifies full Hoffmann (E, A, α, B, β) with bootstrap CIs from trajectory data. | ~3.4 hr | ~$54 |
| 5 | **V3** | MoE G=2, multimodal r=0.3 | 6×10¹⁹ | **predicted from fit (~d28 or d26-capped)** | **Held-out preregistered verification.** ±0.1 nats Gate D. This is the falsification mechanism — without it, the rest is curve-fitting. | ~6.0 hr | ~$96 |
|   |   |   |   |   | **Total** | **~13.5 hr** | **~$216** |

**Why C_mid is non-negotiable:** verified empirically (this revision, May 2026). With only 2 unique N values, two qualitatively different fits — one with E≈1.6, α≈0.34 (truth-like) and one with E≈0.5, α≈0.12 (bound-clamped) — both achieve perfect training residual on synthetic clean data. They diverge on V3's predicted N\* (config) by ~17%, although they agree on V3's predicted L\* (loss) within ~0.003 nats. Predicting loss is robust; predicting config is not. Since V3 trains *at* the predicted config, getting that config wrong invalidates the falsification. C_mid fixes this for $28.

**Why these specific cells?** First-principles derivation in §5. Why A0 is non-negotiable insurance, not optional: see §5 cut #4 (revised). Why C_mid is non-negotiable identifiability: see §5 cut #2 (revised).

---

## §5. The cut list — what we removed and why ⚠️

This section is the senior-researcher review. Every cut is justified or it doesn't happen.

### Cut #1: All d18 cells

**Old spec:** F1.s, F1.m, F1.l at depths {18, 20, 22}.

**Cut reason:** Per `core/model.py:471-475` (commit ca747d6, May 2026), the LR-transfer formula `(d/768)^-0.5` is anchored at d=24 and **empirically overshoots below d ≈ 16**. Auto sessions 1–2 had to retune LR by hand at d=8. The formula is validated in d ∈ [20, 28] only. d18 is *just* outside that band on the low side.

**Consequence of keeping d18:** the fitted exponent `α` would be partly measuring HP-transfer drift, not the loss surface. You cannot tell whether F1.s losing to F1.m is because d20 is genuinely better at that compute scale, or because d18's LR is mis-set by ~10%. This is a measurement-instrument confound and a senior reviewer would flag it instantly.

**Replacement:** depth axis is now {20, 22, 24, 26}. Bracket Chinchilla-optimal depth (~d22–d26 at our compute scales) cleanly inside the validated band.

### Cut #2: Middle widths at each compute scale — REVISED after identifiability analysis

**Old spec:** 3 widths per compute scale = 9 fit cells, "real IsoFLOP curve at every C."

**Earlier v3 draft (INCORRECT, now corrected):** "trajectory fitting at 2 unique N values is sufficient because each run's eval points identify both α and β." That claim was **mathematically wrong**, verified May 2026.

**The identifiability finding:** in `L(N, D) = E + A·N^-α + B·D^-β`, the parameter α is constrained *only* by how L varies across N. With 2 unique N values (`N₁`, `N₂`), the data yields two equations of the form `E + A·N_i^-α = c_i` — three unknowns (E, A, α), two equations → one degree of freedom. The fit converges to obj=0 (perfect data residual) at qualitatively different (E, A, α) settings; β identifies cleanly from per-trajectory D-variation but α does not.

**Empirical verification (this revision):** synthetic data from Hoffmann with truth (E=1.6, A=400, α=0.34, B=800, β=0.28), 32 points across 2 N values, 50-restart L-BFGS. Result: fit lands at (E=0.5, A=15.4, α=0.115, B=799, β=0.280) with obj=0 — same data fit as truth, different α by 0.225. With a third N (3 cells, 48 points), the fit recovers all 5 parameters to within 0.05.

**Cut #2 is therefore REVERSED:** the middle compute-scale cell (C_mid) is required, not optional. **5 cells, not 4.**

**Trajectory logging stays load-bearing:** it gives β identifiability per cell (~16 D-values per cell at no extra GPU cost) and bootstrap CIs. The ½ day engineering investment is still net positive. It just doesn't *replace* the need for 3 unique N values.

**Replacement:** 3 multimodal-MoE cells (C1, C_mid, C2) at three compute scales spanning 6× in C, with trajectory logging for full Hoffmann fit.

### Cut #3: The third compute scale (F3.*)

**Old spec:** 3 compute scales {5e18, 1.5e19, 3e19} × 3 widths.

**Cut reason:** the third scale was load-bearing only if you're fitting both exponents from IsoFLOP minima alone (no trajectory data). With nanochat dense as a measured anchor for `α_text, β_text` at one scale and our own dense-multimodal trajectory points giving `(δ_α, δ_β)`, the third scale is redundant for identifying the perturbation. Held-out verification at C = 6×10¹⁹ replaces the third fit scale.

**Trade-off accepted:** narrower extrapolation range. We're fitting on C ∈ [5e18, 3e19] and predicting at C = 6×10¹⁹ — a 2× extrapolation from the upper edge, not a 4× extrapolation from the middle. Slightly higher prediction error; documented as v3 limitation.

### Cut #4: Reactive sanity gate as a multi-axis cell

**Old spec:** F1.s doubles as "reactive sanity gate" for HP transfer — at d18, multimodal, MoE, smallest scale all at once.

**Cut reason:** A sanity gate should test **one thing at a time**. F1.s confounds four axes; if it fails, you cannot localize which one broke.

**Replacement:** **A0 is now framed as instrument calibration, not a sanity gate.** It is dense + text-only at the d=24 configuration nanochat has measured. Its purpose is not "does training work" (CPU tests cover code correctness) — it is "does our pipeline produce the *quantitative* numbers a calibrated instrument should produce at the configuration nanochat ran."

**Why this matters and why A0 is non-negotiable:** CPU tests verify code correctness (shapes, scatter, forward/backward execute, gradient checks pass). They do **not** verify the bugs that produce systematic loss bias:

| Bug class | CPU tests catch it? | A0 catches it? |
|---|---|---|
| Tokenizer off-by-one (BOS handling, EOS shift) | No | Yes — loss offset from nanochat |
| Wrong loss reduction axis | No | Yes |
| LR schedule misapplied (warmup, decay shape) | No | Yes — final loss shifts 5–10% |
| Data loader sampling bug, eval set leak | No | Yes |
| Optimizer hyperparam drift from nanochat | No | Yes |
| MFU calibration off (budget at 47% real is 30%) | No | Yes — wall-clock comparison |
| Numerical instability converging to biased min | No | Yes |

These are the bugs that make pipelines "work but produce wrong numbers." They are invisible until compared against a known external benchmark. A0 *is* that comparison.

**Cost/value math:** A0 costs $25. Probability of a calibration issue in a newly-multimodal pipeline = ~15–25% (subtle bugs are common in new pipelines). If the issue exists and A0 is skipped, all four downstream cells share the bias → $163 spent on a biased fit + V3 fails Gate D for unclear reasons → debug cost ≥ one extra cell. **Expected value of A0 is positive at $25.**

**A0 is not part of the fit** (the fit uses C1, C2 only). A0 is part of the measurement instrument's calibration — analogous to calibrating a thermometer before publishing temperatures. It is removed only if a future revision shows a $0 alternative that catches the same bug classes.

### Cut #5: Preemptive endpoint-expansion plan

**Old spec:** "If F3.l wins at d26, add d28 expansion cell" — formal gate plus pre-budgeted $30.

**Cut reason:** with 2 cells per loss curve (cut #2), there is no "endpoint" in the IsoFLOP-min sense. The Hoffmann fit interpolates between points; it does not require an interior minimum. The expansion logic is replaced by Gate C's residual check (§6).

**What we keep:** if the *fit* extrapolates predicted G3 depth above d26, we either run a single d28 cell to extend the bracket OR cap at d26 and document the bias. Same decision as old §5.2, but it's a fit-driven decision after Phase 1, not a preemptive budget line.

### Cut #6: Multi-init L-BFGS with 50 random starts in the fit script

**Old spec:** §5.2 prescribes 50-restart L-BFGS for the Hoffmann fit.

**Cut reason:** 50 restarts is non-convex-fit cargo culting at this data size. With 4 cells × ~16 eval points = ~64 (N, D, L) points (if trajectory logging lands) or 4 endpoint points (if not), 5 parameters are still well-identified by a handful of restarts. 50 was lifted from Besiroglu's Chinchilla replication where they had ~400 datapoints across the literature.

**Replacement:** 10 restarts with diverse initial conditions. If best/worst differ by > 5% on training residual, escalate to 50. Otherwise 10 is enough.

### Cut #7: Bootstrap CI on slope from 3 IsoFLOP-min points

**Old spec:** §5.1 bootstrap CI from 3 (C, N_opt) points (also flagged as "informal" in the original — kept anyway).

**Cut reason:** 3 points bootstrap-resampled gives at most 7 distinct subsets, most of which yield identical slopes. The "CI" is a fiction. Old spec already documented this honestly; we now act on the documentation by removing the fake CI.

**Replacement:** if Approach 3 (trajectory logging) lands, block-bootstrap on the trajectory points within each (N, D) curve — that gives a real CI. If not, report slope with no CI and the verification cell as the only error bar. Honest framing: "this is a point estimate; the falsification mechanism is the held-out verification, not the fit's confidence interval."

### Cut #8: W&B project sprawl & per-modality loss diagnostic targets

**Old spec:** §8 prescribes routing entropy ≥ 1.45, dead-expert counts, grouped_mm overhead, per-modality loss windows of 32, `r_actual` semantics, etc.

**Cut reason:** these are diagnostic and useful but not all load-bearing for the scaling-law claim. For *this* study, only val_loss/joint and val_loss/text actually appear in any decision gate. Routing entropy matters only if it collapses (catastrophic, easy to spot from final loss > 1.8). Dead-expert count similarly.

**Replacement:** §8.1 (headline-tier metrics) stays. §8.2 (diagnostic tier) becomes "logged but not gated." Reduces operational complexity without losing the safety net — collapse still shows up as a failed loss number.

### Cut #9: Dense + multimodal cells (B1, B2) — and the modality-decomposition claim

**Earlier v3 draft:** B1 and B2 (dense + multimodal at C₁ and C₃) isolated `δ_vision` by holding architecture dense while turning vision on. Together with C1, C2 they enabled the full decomposition `L_MoE_MM = L_dense_text + δ_vision + δ_MoE + δ_interaction`.

**Cut reason:** the decomposition is only load-bearing if we publish a claim about the *source* of asymmetry (vision-driven vs architecture-driven) — i.e., a FAIR Mar 2026 (T12) replication study. The project's stated novelty (per `docs/Plan.md`) is G\*(r), not FAIR replication. For the prerequisite question (Option A: "does a clean MoE+MM scaling law exist and predict held-out?"), the decomposition is not load-bearing. Our deployment is MoE+MM; the fit on MoE+MM alone is what predicts V3.

**What we lose:**
- Cannot test FAIR's α_vision < α_text claim from this study.
- Cannot publish "MoE absorbs the asymmetry" or its refutation.
- If V3 misses, cannot localize whether the miss is vision-related or MoE-related from this dataset alone (must add a dense+MM cell at debug time, ~$30 only if needed).

**Why acceptable:** $59 saved (B1+B2). FAIR replication is a separate study; pretending we're doing it on the side dilutes scope. Diagnostic localization can be deferred until needed and paid for then.

### Things we did NOT cut, and why

- **Preregistration before V3** (old §4.2) — non-negotiable; this is the falsification mechanism, not optional rigor.
- **Gate D ±0.1 nats tolerance** — same tolerance as CS336 / Hoffmann replications; principled.
- **BF16 over FP8** — T5 says FP8 + MoE incompatible in our codebase. Fixed.
- **`n_active_params` for compute accounting** — required by `C = 6ND` correctness under MoE.
- **r = 0.3 fixed** — Option A scope decision (§1); deferring r-sweep to Option B.
- **A0 dense+text-only calibration cell** — see cut #4 revision; this is instrument calibration, not redundant with multimodal cells.

---

## §6. Gates and preregistration — point to old spec, two changes

Everything in `dev/scaling_law_self_assignment.md` §4 + §9 still applies, with these two amendments:

**Amendment 1: Gate B (was "interior minimum at every C") becomes "fit residual < 5% in log-space on each curve."** Trajectory or 2-point endpoint fitting does not produce "endpoints." Replace the endpoint-min check with a residual check on the parametric fit.

**Amendment 2: A0 has a new gate (G_anchor)** that runs *before* Gate B:

| Gate | Trigger | Pass | Fail |
|---|---|---|---|
| **G_anchor** | A0 final val_bpb vs nanochat T6 expected value at d24, ~7B tokens, ClimbMix | Within 5% relative → 🟢 inheritance locked, proceed to Phase 1 | Outside 5% → halt. Divergence diagnosis required before any multimodal cell. |

The preregistration entry in `dev/LOG.md` (old §4.2) is unchanged — same format, written between Phase 1 and V3.

---

## §7. Honest tradeoffs of the cuts

A senior reviewer would ask: "what did you sacrifice?" Answer in writing, in advance.

| What we sacrificed | Why we judge it acceptable |
|---|---|
| **3-point fit on the MoE+MM curve (C1, C_mid, C2)** | Three unique N values — the minimum that makes Hoffmann α identifiable (verified empirically, §5 cut #2 revision). With trajectory logging across the 3 cells → ~48 (N, D, L) points for fitting (E, A, α, B, β) with block-bootstrap CIs. Robust to single-cell noise: a wildly off C_mid shows up as a fit-residual spike at Gate C. |
| Narrower extrapolation range (2× from upper edge, not 4× from middle) | Smaller absolute extrapolation distance → lower expected prediction error at V3. Net positive on prediction quality; net negative on "scaling law range" published claim. Honest framing wins. |
| **No modality decomposition (dropped B1, B2)** | Cannot publish a claim about "vision vs architecture drives the delta from dense text." Our deployment is MoE+MM; the joint delta is what we ship. Decomposition is a separate study, deferred. |
| No third compute scale at the fit (C-mid is contingent) | Reduces robustness to non-power-law curvature. Caught by Gate C residual check on the trajectory; if residuals > 5%, we add C-mid ($28). |
| Single seed | Same as old spec. Documented as v4 future work. The single-seed risk is partly absorbed by V3 — a wildly noisy C1 or C2 shows up as a V3 miss, which we'd then investigate. |
| No G axis | Explicit Option A scope decision. G\*(r) is the next study, after this one's pipeline is verified. |
| Diagnostic metrics no longer gate decisions | Catastrophic failures (routing collapse, dead experts) still show up in final loss > 1.8. Sub-catastrophic dynamics issues become a v4 study, not this one's. |
| Cannot test FAIR Mar 2026 modality asymmetry (T12) from this dataset | T12 is 🔴 in §2 — load-bearing in the broader project narrative but not what this study targets. Tested in a future B1-B2-revival study or in the Option B G\*(r) sweep. |

---

## §8. What this study deliberately is NOT (carry-over with corrections)

Inherits the "is NOT" list from `dev/scaling_law_self_assignment.md` §15, plus:

- ❌ **Not** a refit of nanochat dense text-only scaling. We **inherit T2, T3, T4, T6, T7** as measured.
- ❌ **Not** a G\*(r) measurement. Explicitly deferred to a v4 spec gated on this study's success.
- ❌ **Not** a CS336-grade study. Old spec marketed itself as "CS336-faithful" with 9 cells; v3 is smaller. We are honest that this is a smaller study with a sharper hypothesis, not a Stanford assignment clone.
- ❌ **Not** a publication-ready scaling law if Gate D fails. Failed Gate D means an honest writeup of failure, same as old spec.

---

## §9. Execution sequence (terse)

```
Day -1 (CPU, free):
  - python scripts/preflight.py
  - python scripts/sweep_runner.py dry-run --config configs/scaling_law_v3/A0.json
  - Decide: implement trajectory logging in base_train.py + sweep_runner.py?
            (yes → +½ day eng, +real CIs; no → 2-point endpoint fit, no CIs)

Day 0 (vast.ai provisioned):
  - A0 → G_anchor check (~$25, ~1.6 hr)
    PASS (within 5% of nanochat T6) → 🟢 inheritance locked, proceed
    FAIL → halt, diagnose; do NOT run any multimodal cell while pipeline is biased

Day 1:
  - C1 (~$13, ~0.8 hr)
  - C_mid (~$28, ~1.7 hr)
  - C2 (~$54, ~3.4 hr)

Day 1-2 evening:
  - python scripts/sweep_runner.py fit-hoffmann
  - Gate C: residuals < 5% log-space, alpha in [0.2, 0.6], beta in [0.2, 0.6],
            warning "only 2 unique N" must NOT appear
    PASS → fill V3_template.json with predicted (depth, hidden_size, val_bpb)
           → copy to V3.json, PREREGISTER in dev/LOG.md, commit to git
    FAIL → diagnose; do not run V3 with a broken fit

Day 2-3:
  - V3 (~$96, ~6 hr)
  - Gate D readout

Day 3-4:
  - python scripts/plot_predicted_vs_actual.py
  - dev/LOG.md writeup
  - Destroy vast.ai instance
```

**Total wall-clock (at 35% MFU assumption):** ~4 days elapsed, ~13.5 GPU-hours, ~$216.
**Compared to specs in lineage:**
- v2 (`scaling_law_self_assignment.md`): 10 cells, $277–370 (35% MFU)
- v3 6-cell intermediate: $247
- v3 4-cell draft (REVERTED): $188 — α non-identifiable, V3 config prediction off ~17%
- **v3 5-cell (this spec, current): $216.** ~25–40% off v2. The cells that survived the cut are the ones first-principles-required for identifiability (3 unique N's, 1 calibration, 1 verification). Cutting any further breaks the math; spending more is the v2 territory.

> ⚠ **2026-05-20 measured MFU update — partially resolved.**
>
> First attempt (NGC `pytorch:25.03-py3` build `7c8ec84dab.nv25.03`): measured
> **19.7% MFU** in eager mode with `--activation-ckpt`. NGC build lacked
> `torch._grouped_mm`. Budget projected to ~$650.
>
> Resolution (later same day): switched to verified image
> `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` (Docker Hub PyPI release).
> This image ships `torch._grouped_mm` + matching triton with working
> `torch.compile`. Fixed a latent bug in `core/moe.py` (histc → bincount,
> commit `8fa6236`) exposed by inductor's strict type checking.
>
> A0 still pending re-launch on this verified stack. Expected MFU based on
> spec assumptions: 35-47% → budget returns to ~$216. Will verify at next
> launch.
>
> See `dev/auto_findings/2026-05-20-A0-attempt/findings.md` for full incident
> + resolution log.

---

## §10. Cross-references

- `dev/scaling_law_self_assignment.md` — the v2 spec this delta-supersedes (under Option A only)
- `docs/SPEC.md` §3.2 — T-table (nanochat transfer audit, this doc's §2 inherits and re-categorizes)
- `docs/Plan.md` §1 — original G\*(r) thesis (this doc's §1B option)
- `docs/SL_VI.md` — Vietnamese long-form intro to the project
- `core/model.py:468-475` — the LR-transfer regime-limit comment (this doc's cut #1 anchor)
- `nanochat/dev/LOG.md` 2026-02-19, 2026-02-21 — the measured prior we inherit (§2)
- `dev/auto_findings/lessons.md` — d<16 LR retune empirical evidence (cut #1)
- `dev/sweep_design.md` — independent v3 speedrun, non-overlapping with this study

---

## TL;DR (one paragraph)

This v3 minimal spec cuts the 10-cell IsoFLOP design to **5 cells (~$216)** by (1) treating nanochat dense text-only as **measured prior**, not as something to refit; (2) restricting the depth axis to nanochat's validated LR-transfer band d ∈ [20, 28]; (3) collapsing per-scale width sweeps to 3 compute scales × 1 width each, enabled by Approach-3 trajectory logging (½ day engineering) — **NOT 2 cells**, which is mathematically insufficient: Hoffmann α is non-identifiable below 3 unique N values, verified empirically (§5 cut #2 revision, May 2026); (4) keeping A0 as **instrument calibration** against nanochat (not a fit point — see §5 cut #4 revision; cheap insurance against systematic pipeline bias); (5) dropping the dense+multimodal cells B1, B2 because modality decomposition is not in this study's scope (§5 cut #9). The hypothesis tested is narrower than earlier drafts: "the MoE+multimodal scaling law on our pipeline predicts held-out loss at C=6×10¹⁹ within ±0.1 nats." FAIR Mar 2026 replication and G\*(r) measurement are explicitly deferred. Falsification is the preregistered V3 cell with ±0.1 nats Gate D. The 9 cuts and their justifications are in §5 — that is the section that earns this redesign its keep.
