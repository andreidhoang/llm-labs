# Sweep Design v3 — Multimodal Speedrun (Karpathy-style extension)

**Status:** pre-registration document for the main GPU sweep. Replaces v2 G-sweep design (preserved as `sweep_design_g_sweep_deferred.md` for later execution if budget materializes). Methodology choices are fixed here BEFORE seeing data.

**Owner:** operator-researcher driving the project (see `MEMORY.md` user role).

**Lineage:**
- `sweep_design_v1.md` — original 7-phase ambitious plan (G × φ × r full surface, $1068)
- `sweep_design_g_sweep_deferred.md` — v2: G-sweep at fixed r=0.3 ($322, 11 cells). **Deferred, not abandoned** — execute if (a) v3 finishes ahead of time, OR (b) v3 reveals G-sensitivity worth measuring directly.
- `sweep_design.md` (this file) — v3: multimodal speedrun at fixed Karpathy-G config ($261, 8 cells). **Current focus.**

---

## 0. Thesis vision

> *Karpathy made nanochat shippable by treating wall-clock-to-capability as the single optimization target. We extend his discipline into the multimodal regime that nobody has speedrun-optimized.*

### 0.1 The single goal

> **What is the fastest 8×H100 recipe to reach a fixed multimodal capability target at production mix r = 0.3, and which optimization knobs contribute most to the speedup?**

This is the **multimodal analog of nanochat's GPT-2 speedrun.** Karpathy's text-only result: GPT-2 capability in 2h01min on 8×H100, ClimbMix 7B tokens, d24 dense. The multimodal regime that ALL frontier 2026 models train in (Qwen3.5-VL, Gemini, GPT-4o, DeepSeek-V3) has **no equivalent published speedrun number, no equivalent published optimization breakdown.**

### 0.2 Why this contribution beats a G-sweep

The G-sweep (deferred design) answers an interesting science question: "is text-derived G\* the same in multimodal regime?" Production teams care, but the answer mostly tells them **how to set one architecture knob**.

The speedrun answers an actionable engineering question: "for a fixed budget, what recipe ships fastest?" Production teams care because **every percentage point of wall-clock improvement compounds across $10M-$100M training runs.** A 30% wall-clock reduction in multimodal MoE training is worth >$1M per major training run for any frontier lab. **A $261 measurement that informs that decision is asymmetrically valuable.**

Concrete contributions this study delivers that the G-sweep doesn't:
- **Capability-matched wall-clock breakdown** — analog of nanochat's "FP8 = +5% capability-matched speedup" but for multimodal
- **Reusable recipe** — pinned config + git SHA other teams can `torchrun`
- **Per-knob attribution** — quantifies which optimizations transfer from text to multimodal regime
- **Direct comparison anchor** — same node, same code, vs Karpathy's published 2-hour text-only number

### 0.3 What "capability target" means here

We define multimodal capability target self-referentially (no external benchmark dependency):

```
1. Run anchor cell (text-only d24 r=0.0)        → reproduces nanochat baseline (val loss ~0.715)
2. Run baseline cell (d24 G=2 r=0.3 multimodal) → measures L_baseline_mm
3. Define capability target T = L_baseline_mm + 0.005 (slightly above plateau)
4. For each ablation cell: wall-clock-to-T = first eval where val_loss/joint < T
5. Report all ablations as % wall-clock reduction vs baseline
```

This **doesn't require MMMU/MMBench eval infrastructure**. It uses the project's own val-loss curve as the reference frame. Karpathy uses the same self-anchoring trick for nanochat ablations (LOG 2026-02-19 FP8: "+5% capability-matched speedup" is defined this way).

### 0.4 Falsifiable hypotheses (preregistered)

> **H₀ (FP8-on-shared):** FP8 conversion of the shared expert (per nanochat `dev/moe_fp8.md`) gives ≥3% wall-clock-to-T speedup at multimodal r=0.3 — replicating nanochat's text-only +5% finding within 2× tolerance.
>
> **H₁ (ViT freeze schedule):** unfreezing the ViT after 5% of training (vs always frozen) gives **negligible or negative** wall-clock impact at our scale, because additional gradient flow through SigLIP2 (543M params) costs more compute than it adds capability per token. The frozen-ViT design choice (`dev/multimodal_spec.md` decision #1) is empirically validated.
>
> **H₂ (sequence length):** seq_len = 8192 (video-friendly) gives ≥10% wall-clock penalty vs 4096 (image-only) at fixed r=0.3, due to attention quadratic cost dominating at d24+ scale. **Implication:** image-only deployment should use 4096; video deployment must accept the cost.
>
> **H₃ (batch scaling — Bergsma "Power Lines"):** doubling batch from `B_opt(D)` to `2 × B_opt(D)` gives <2% wall-clock change at d24, validating the `B ∝ D^0.383` scaling rule transfers to multimodal.
>
> **H₄ (MoE wall-clock penalty):** dense (G=1) reaches T faster than MoE (G=2) at multimodal r=0.3 — replicating nanochat's text-only 2026-02-19 finding for multimodal. **The actionable answer to "is MoE worth it for multimodal at our scale?"**

H₄ is the most-load-bearing finding. H₀-H₃ are individual optimization measurements; H₄ is the architecture-vs-engineering call.

### 0.5 What this study deliberately is NOT

- ❌ A G-sweep (deferred to `sweep_design_g_sweep_deferred.md`)
- ❌ A per-modality scaling-law fit (out of budget; see deferred doc)
- ❌ A modality-asymmetry measurement (FAIR Mar-2026 published this)
- ❌ A novel architecture contribution (Qwen3.5-VL early-fusion verbatim)
- ❌ A novel kernel optimization effort (we use upstream `torch._grouped_mm` + `torchao` FP8 as-is)
- ❌ A downstream benchmark eval (no MMMU/MMBench/CORE — val loss as proxy)

Senior researcher's edge: doing a small, sharp, **shippable** measurement that maps to dollar cost in a way the G-sweep doesn't.

---

## 1. Driving principles

These four rules govern every methodology choice. Carried forward from v2 (deferred doc) verbatim because they apply identically here.

1. **Don't re-verify what upstream verified.** nanochat verified MoE correctness, Muon+AdamW, the 1/√d AdamW LR scaling rule at d12–d26. We inherit; we don't re-prove.
2. **The only genuinely novel contribution is multimodal integration + multimodal speedrun discipline.** Every choice that doesn't serve those is overhead.
3. **Stay inside Karpathy's validated HP envelope** (d ∈ [12, 26], `--target-param-data-ratio 12`, default LRs). If a design forces extrapolation, change the design.
4. **Wall-clock is a first-class metric.** Every cell reports per-step val loss AND wall-clock-to-target-T. The whole project IS the wall-clock principle made operational for multimodal.

---

## 2. Headline measurement

**Single output:** speedup-decomposition table comparing each ablation against the multimodal baseline cell.

| Cell | Config delta vs baseline | Wall-clock to T | % vs baseline | Goal-2 verdict |
|---|---|---|---|---|
| **Baseline (B0)** | d24 G=2 r=0.3, BF16, ViT frozen, seq=4096, B=B_opt(d24) | T_baseline (sec) | 100% (reference) | — |
| Anchor (A0) | d24 G=2 r=0.0 (text-only) | T_anchor | derived | HP-transfer falsifier; not in speedup table |
| **FP8-on-shared (E1)** | + FP8 on shared expert (BF16 routed experts) | T_E1 | (T_E1 / T_baseline - 1) × 100 | H₀ pass/fail |
| **ViT unfrozen @5% (E2)** | + unfreeze SigLIP2 after 5% of tokens | T_E2 | (idem) | H₁ pass/fail |
| **seq_len 8192 (E3)** | + max_seq_len=8192 | T_E3 | (idem) | H₂ pass/fail |
| **Batch ×2 (E4)** | + total_batch_size = 2 × B_opt | T_E4 | (idem) | H₃ pass/fail |
| **Dense G=1 (E5)** | + num_experts=1, top_k=1 | T_E5 | (idem) | H₄ pass/fail |

**Decision rule:** an ablation "ships" if wall-clock-to-T improves by ≥3% with no per-step val loss penalty at the target T (i.e., reaches T at lower wall-clock without sacrificing what it reaches T at). Bootstrap CI computed by resampling per-step throughput over the last 50% of each cell's training (steady-state regime).

**Final recipe construction:** sum the speedups of all cells that "ship" → predicted compounded wall-clock for full recipe. The recipe is the **headline deliverable.**

---

## 3. Cell grid (8 cells, ~$261)

### 3.1 Per-cell config

All cells: 8×H100 BF16, vast.ai $16/hr. Inherits Karpathy's HP defaults (matrix_lr=0.02, embedding_lr=0.3, unembedding_lr=0.008, scalar_lr=0.5, target-param-data-ratio=12).

| Cell | Purpose | C (FLOPs) | Depth | r | Config delta | Wall-clock | Cost |
|---|---|---|---|---|---|---|---|
| **P0** | Multimodal integration smoke | 1×10¹⁸ | d12 | 0.3 | G=2 BF16 baseline | ~25 min | $5 |
| **A0** | Nanochat-anchor (HP-transfer falsifier) | 2×10¹⁹ | d24 | **0.0** | G=2 BF16 (text-only, no `--multimodal`) | ~2.0 hr | $32 |
| **B0** | Multimodal baseline (defines T) | 2×10¹⁹ | d24 | 0.3 | G=2 BF16, ViT frozen, seq=4096 | ~2.0 hr | $32 |
| **E1** | FP8-on-shared ablation | 2×10¹⁹ | d24 | 0.3 | + `--fp8` (shared expert only; routed stay BF16) | ~1.9 hr | $30 |
| **E2** | ViT unfreeze ablation | 2×10¹⁹ | d24 | 0.3 | + ViT unfrozen after 5% of tokens | ~2.1 hr | $34 |
| **E3** | Seq-length ablation | 2×10¹⁹ | d24 | 0.3 | + max_seq_len=8192 | ~3.0 hr | $48 |
| **E4** | Batch-scaling ablation | 2×10¹⁹ | d24 | 0.3 | + total_batch_size = 2 × B_opt(d24) | ~2.0 hr | $32 |
| **E5** | Dense baseline (H₄ test) | 2×10¹⁹ | d24 | 0.3 | + num_experts=1, top_k=1 (drop MoE) | ~1.8 hr | $29 |
| **Total** | | | | | | **~16.4 hr** | **~$262** |

**Single compute scale C = 2×10¹⁹** (= nanochat's main-run scale = d24 = ~$32 baseline cell). Reasoning:
- Speedup measurements are **ratios**; a single scale is sufficient to measure them
- Per nanochat 2026-02-19 FP8 entry: optimizations show different magnitudes at different scales, but the SIGN and ranking are stable from d24 onwards
- One scale = 8 cells fits $300 budget; multiple scales would 3× the cost for marginal additional info

**Single G value: G = 2** (Karpathy's nanochat default: `num_experts=8, top_k=2, num_shared=1`). This is the only G value verified by upstream at this scale. If the speedrun finding generates interest, the deferred G-sweep can be run later to test whether the recipe transfers across G values.

### 3.2 What's varied across the ablation cells

Each ablation cell **changes exactly one knob** vs the multimodal baseline B0. This is the standard speedrun-decomposition discipline (one-at-a-time ablation, not Latin square). It costs more cells than a fractional factorial design but produces directly attributable speedups.

| Knob | Baseline value | Ablation value | Why test |
|---|---|---|---|
| FP8 (E1) | off (BF16) | on for shared expert | Test nanochat 2026-02-XX +5% text-only finding transfers to multimodal |
| ViT freeze schedule (E2) | always frozen | unfrozen after 5% | Test SPEC.md §4.1.4 joint-training assumption (currently superseded by frozen design but worth measuring) |
| Seq length (E3) | 4096 | 8192 | Test cost of video-ready context vs image-only |
| Batch size (E4) | B_opt(D) per Bergsma | 2 × B_opt(D) | Test if Bergsma's `B ∝ D^0.383` is tight at multimodal |
| MoE on/off (E5) | G=2 (8 experts top-2) | G=1 (dense) | Test if nanochat MoE wall-clock-loss finding holds at multimodal |

---

## 4. Phase ordering

| Phase | Purpose | Cells | Cost | Gate to next phase |
|---|---|---|---|---|
| **Phase 0** | Multimodal integration smoke | 1 (P0) | $5 | Trains stably 200 steps, no NaN, per-modality losses both decreasing |
| **Phase 0.A** | Nanochat-anchor (HP-transfer falsifier) | 1 (A0) | $32 | Val loss within 3% of nanochat published d24 ClimbMix curve at steps 100/1k/10k/end |
| **Phase Baseline** | Establish multimodal capability target T | 1 (B0) | $32 | All Phase 0/0.A criteria + B0 trains to FLOPs budget; T = L_baseline_mm + 0.005 fixed and committed to git BEFORE running E1-E5 (preregistration) |
| **Phase Ablations** | Five one-knob ablations | 5 (E1-E5) | ~$173 | Each cell completes; speedup vs B0 reported with bootstrap CI |
| **Phase Recipe** | Compound winning ablations into final recipe (post-hoc, no new cells) | 0 | — | Composed recipe is reproducible and predicted speedup matches measured speedup within CI |
| **Total** | | **8 cells** | **~$262** | |

### 4.1 Phase 0 protocol (multimodal integration smoke) — unchanged from v2

```bash
torchrun --nproc-per-node=8 scripts/base_train.py \
  --depth 12 --num-experts 8 --top-k 2 --num-shared-experts 1 \
  --target-flops 1e18 --max-seq-len 4096 \
  --multimodal --mix-ratio 0.3 \
  --run smoke_mm_d12_g2_r03
```

Pass criteria: no NaN/200 steps; both per-modality losses decreasing; routing entropy > 0.7·log(8); zero dead experts; vision tokens correctly scattered; determinism (same seed → step-100 loss to 1e-5).

### 4.2 Phase 0.A protocol (nanochat-anchor — HP-transfer falsifier)

```bash
# EXACT nanochat main run config — no MoE, no multimodal
torchrun --nproc-per-node=8 scripts/base_train.py \
  --depth 24 --num-experts 1 --top-k 1 --num-shared-experts 1 \
  --target-param-data-ratio 12 --max-seq-len 2048 \
  --run anchor_d24_text_baseline
```

Compare val loss curve to nanochat published d24 ClimbMix baseline at steps 100/1k/10k/end with 3% tolerance per checkpoint. Wall-clock to step 14K within 110% of nanochat's 2h01min → throughput parity confirmed.

Fail actions (graduated): 3-5% off → tolerate, document; 5-10% off → halt, $50 mini-LR sweep at d24; >10% off → infrastructure bug, diagnose.

### 4.3 Phase Baseline protocol (establish T)

```bash
torchrun --nproc-per-node=8 scripts/base_train.py \
  --depth 24 --num-experts 8 --top-k 2 --num-shared-experts 1 \
  --target-param-data-ratio 12 --max-seq-len 4096 \
  --multimodal --mix-ratio 0.3 \
  --run baseline_mm_d24_g2_r03
```

Train to FLOPs budget. At end, record L_baseline_mm = `val_loss/joint` averaged over last 3 evals. **Commit T = L_baseline_mm + 0.005 to git** in `dev/LOG.md` before running any ablation cell. This is the preregistration step.

### 4.4 Phase Ablations protocol

Run E1-E5 in any order (no inter-cell dependencies). Each cell uses the same `--target-param-data-ratio 12` budget as B0 (so all reach a comparable FLOPs amount). For each cell:

1. Train to FLOPs budget exhaustion
2. Post-hoc compute `wall_clock_to_T(cell) = first eval where val_loss/joint < T`
3. Record speedup: `(T_baseline - T_cell) / T_baseline × 100%`
4. Bootstrap CI by resampling per-step throughput over steady-state region

### 4.5 Phase Recipe (post-hoc, no GPU cost)

For each ablation that "ships" (≥3% wall-clock improvement, no val loss penalty at T), include in the composed recipe. Predicted recipe speedup = sum of individual speedups (assumes additivity; documented assumption that may require validation in a future phase if recipe is high-stakes).

Optional: if budget allows after main 8 cells, add **1 verification cell** running the composed recipe end-to-end ($32). Predicted vs measured speedup is the recipe-additivity check.

---

## 5. Risk model

| # | Risk | Mitigation | If materializes |
|---|---|---|---|
| 1 | **Multimodal integration bug** in `core/multimodal.py` (15 stubs at write-time) | Phase 0 smoke + local CPU joint forward smoke at d=4 + determinism contract | Halt, debug; do not proceed to Phase 0.A |
| 2 | **HP transfer failure to our setup** (vast.ai vs Lambda; our shard order vs Karpathy's) | Phase 0.A anchor cell with 3% tolerance; doubles as wall-clock baseline | 3-5% → tolerate; 5-10% → $50 mini-LR sweep at d24; >10% → infra bug |
| 3 | **r-drift** (actual token-level vision fraction ≠ 0.3) | Token-level mix accounting in dataloader; assert `\|mean(r_actual) - 0.3\| < 0.02` rolling 100 batches | Adjust dataloader sampler; re-run affected cells |
| 4 | **MFU << 35%** at d24 on real hardware | nanochat measured 35-46% at d18-d24; we expect similar | If MFU < 25% sustained: each cell costs 1.7× budget; either accept or skip lowest-priority ablation (E3 seq_len) |
| 5 | **FP8-on-shared instability** (E1 cell) | torchao convert_to_float8_training is verified by nanochat at d24; wrap in try/except, fall back to BF16 if NaN | Mark E1 as "not applicable to multimodal at our scale"; report negative result |
| 6 | **ViT unfreeze causes loss explosion** (E2 cell) | Gradual unfreeze (linear warmup from 0% to 100% over 1K steps after the 5% trigger); cancel if grad_norm > 5× baseline | Mark E2 as "joint training requires careful schedule"; report; consider as future work |
| 7 | **Recipe additivity fails** (composed speedup ≠ sum of individual) | Optional verification cell; documented assumption | Report measured composed vs predicted; explain via 1-2 candidate causes (cache effects, optimizer interaction, etc.) |

---

## 6. Cancel triggers (per cell)

Apply to every cell in Phase 0/0.A/Baseline/Ablations:

| Trigger | Action |
|---|---|
| NaN/Inf in loss | Kill cell; tighten grad-clip 1.5×; restart from checkpoint |
| Loss at step 100 > 1.5× expected (vs baseline cell B0 curve) | Kill cell; investigate config |
| Sustained MFU < 25% for 1000 steps | Kill cell; infrastructure issue |
| Wall-clock exceeds 1.5× budget | Kill cell; re-plan |
| Routing entropy < 0.5·log(num_experts) for 200 consecutive steps | Kill cell (B0/E1-E4 only; E5 dense exempt) |
| Dead expert (any expert receives 0 tokens for >200 consecutive steps) | Kill cell (B0/E1-E4 only; E5 exempt) |
| Per-batch `\|r_actual - 0.3\|` > 0.05 sustained | Pause cell; fix dataloader |
| Grad norm > 5× baseline (E2 only — ViT unfreeze instability) | Kill cell; report E2 as failed |

Every kill produces a `dev/LOG.md` entry. **No silent failures.**

---

## 7. Logging schema (bi-tier)

W&B project: `multimodal-speedrun-d24-r03`. One run per cell, named `{cell_id}` from §3.1.

### 7.1 Headline-tier metrics

**Per-step:**
- `loss/total`, `loss/text`, `loss/vision`
- `step_time_ms`, `tokens_per_sec`, `mfu`
- `wall_clock_elapsed_sec` (cumulative)

**Per-eval (every 5% of cell steps, ≥10 evals/cell):**
- `val_loss/text`, `val_loss/vision`, `val_loss/joint`
- `val_loss/joint_bootstrap_ci_low`, `val_loss/joint_bootstrap_ci_high`

**Post-hoc derived (computed after cell complete):**
- `wall_clock_to_T` (target T from Phase Baseline)
- `speedup_vs_baseline` (E1-E5 only)

### 7.2 Diagnostic-tier metrics

**Per-step:**
- `grad_norm/total`
- `lr/muon`, `lr/adamw_emb`, `lr/adamw_unemb`, `lr/adamw_scalar`
- `r_actual` (per-batch token-level vision fraction)

**Per-50-steps:**
- `routing/entropy_per_layer` (list[float])
- `routing/n_dead_per_layer` (list[int])
- `throughput/grouped_mm_overhead_ratio` — fraction of step time in `_grouped_mm` dispatch (the nanochat-identified bottleneck; explains H₄ outcome)
- `throughput/fp8_compute_ratio` (E1 only) — fraction of FLOPs in FP8 vs BF16
- `throughput/vit_forward_ratio` (E2 only when unfrozen) — fraction of step time in ViT forward+backward

### 7.3 Per-cell metadata
- Full `GPTConfig` dump
- `cell_id`, `git_sha`, `seed`, `gpu_specs` (vast.ai instance ID, CUDA version)
- `n_active_params`, `n_total_params`, `flops_per_token_estimated`
- `wall_clock_budget_sec`
- `target_T` (for ablation cells, written from Phase Baseline output)

---

## 8. What was deferred (not dropped) and why

The G-sweep design (`sweep_design_g_sweep_deferred.md`) is preserved and is **scheduled for execution in any of these scenarios:**

1. **v3 finishes ahead of time and budget remains:** run G ∈ {1, 4, 8} ablations at the discovered fastest-recipe config. Adds 2 cells (G=4 and G=8 since G=1 is E5 already and G=2 is B0). Cost: ~$64.
2. **v3 reveals G-sensitivity worth investigating:** if E5 (dense G=1) wall-clock differs dramatically from B0 (G=2), the G axis becomes load-bearing for production decisions and the full sweep is justified.
3. **External funding or 2nd budget cycle:** if a sponsor / lab grants a follow-up $300+ budget specifically for G\* science, deferred design is execute-ready.

**The G-sweep is NOT abandoned. It is sequenced after the speedrun.** Speedrun is engineering-first; G-sweep is science-first; both have value but speedrun ships earlier and provides the recipe foundation for any subsequent G measurements.

---

## 9. Reviewer Q&A

> **"Why a single compute scale? You can't fit any scaling law from one C."**

We're not fitting a scaling law. We're measuring per-knob wall-clock speedups, which are RATIOS and meaningful at any single scale. Multi-scale would multiply cells by 3 for marginal information; the scale we picked (d24 = nanochat main-run scale) is the one most directly comparable to Karpathy's published numbers and thus the highest-information single point.

> **"Why fix G=2? Isn't G\* the interesting question?"**

G=2 is Karpathy's verified config — the only G with upstream wall-clock measurements we can compare against. Sweeping G IS interesting (see deferred doc) but mixing it with the speedrun ablations would conflate "what's the fastest recipe" with "what's the optimal architecture." Two papers, sequenced.

> **"Why not include MMMU or MMBench eval?"**

Capability target T is defined self-referentially via val loss (§0.3), matching Karpathy's "capability matched" convention from nanochat LOG. Adding external benchmarks (MMMU/MMBench) would require eval pipeline implementation (~1 week of CPU work, no GPU benefit) and would introduce a new noise source. Future work; not in this scope.

> **"What if the ablations don't compose additively?"**

Documented in §4.5 Phase Recipe and Risk #7. Optional verification cell ($32) tests this directly. If they don't, we report individual speedups + measured composite + the gap, attributing to candidate causes. The non-additivity itself becomes a finding ("FP8 + batch×2 interact via cache effects" or similar).

> **"Why not sweep r as well? Multimodal mix matters."**

r=0.3 is the production-modal mix (per dev/LOG 2026-05-02). r-axis sweep is a separate paper (Plan.md original ambition, deferred). Speedrun is anchored at one production-realistic point to maximize wall-clock measurement precision.

> **"This isn't novel science, this is engineering."**

Correct, and that's the point. nanochat is engineering; nanochat is also one of the most cited / used / valuable artifacts of 2026 because it MAKES SOMETHING SHIPPABLE. Multimodal MoE training has no nanochat-equivalent. We're trying to be it.

> **"What's the deliverable for someone reading the writeup?"**

A reproducible 8×H100 multimodal MoE training recipe + speedup decomposition + a number ("multimodal capability X reached in Y hours, full breakdown attached"). Other teams can clone the recipe, swap in their own dataset, and ship.

---

## 10. Trade-offs flagged honestly

### 10.1 Single-scale measurement
We accept that speedups may differ at d12 (smaller models) or d30+ (larger). nanochat's FP8 entry explicitly notes this: "d12 was still slower with FP8; d26+ shows gains." Our recipe is anchored at d24 ≈ GPT-2 scale and **may or may not transfer to other scales.** Honest footnote in the writeup; deferred verification.

### 10.2 Single G value
G=2 is one architecture choice. Speedups measured here may be specific to the (G=2, num_experts=8, top_k=2) operating point. Bigger MoE configs (G=8) might amortize FP8 overhead differently. The deferred G-sweep can answer this if ever executed.

### 10.3 Recipe additivity assumption
We sum individual speedups to predict composed recipe wall-clock. Optional verification cell tests this. If we skip the verification cell to save $32, the predicted composed speedup carries an unmeasured uncertainty.

### 10.4 ViT freeze schedule (E2) is only one of many possible schedules
"Unfreeze after 5%" is one point in a continuous space. We're not searching the schedule space; we're testing "always frozen vs gradual unfreeze." Stronger claim ("optimal schedule is X") would need its own sweep.

---

## 11. Wall-clock methodology (carried forward from v2 §11, sharpened)

Wall-clock IS the headline here, not a side metric. This section is the most important methodology section in the doc.

### 11.1 What "wall-clock" means precisely

Same definitions as v2:
- `wall_clock_elapsed_sec`: clock time since first optimizer step on the production node, BF16 (or FP8 for E1), no profiler attached
- `wall_clock_to_T(cell)`: seconds at which `val_loss/joint` first crosses below T, computed from logged eval points + linear interpolation
- `tokens_per_sec`: sum across 8 ranks
- `mfu`: nanochat's `core/common.py:get_peak_flops()` (BF16 989 TFLOP/s peak per H100)
- `grouped_mm_overhead_ratio`: PyTorch profiler over a 100-step window once per training cell, mid-training

**All cells run on the same vast.ai 8×H100 instance** if at all possible. Cross-instance variance can be 5-10% from CPU/host noise — would contaminate our small-margin measurements. Single-instance discipline is the most important infra rule.

### 11.2 The T derivation (Phase Baseline output)

```
T = L_baseline_mm + 0.005

where L_baseline_mm = mean(val_loss/joint over last 3 evals of B0)
```

The +0.005 margin is chosen so that:
- All ablation cells plausibly reach T (if margin were 0, only ablations with lowest final loss reach it)
- Not at noise floor (typical eval variance is ~0.001-0.002 for d24)
- Aligns with nanochat's "5% capability matched" convention (5% in their context ≈ 0.005 absolute val loss at d24 GPT-2 plateau)

T is committed to `dev/LOG.md` as preregistration before any ablation runs. Changing T post-hoc after seeing ablation results would be p-hacking.

### 11.3 Speedup attribution

For each ablation cell E_i with wall-clock-to-T = T_i:

```
speedup_i = (T_baseline - T_i) / T_baseline × 100%
```

Bootstrap CI: resample steady-state tokens/sec from last 50% of training (1000 resamples), recompute T_i for each resample, take 5th/95th percentile of T_i distribution.

**Decomposition by cost component** (per nanochat 2026-02-19 diagnostic):
- `frac_compute = compute_time / total_step_time`
- `frac_grouped_mm = _grouped_mm_kernel_time / total_step_time`  
- `frac_dataloader = dataloader_idle_time / total_step_time`
- `frac_comm = ddp_all_reduce_time / total_step_time`

Report all four fractions per cell. The fractions explain WHERE each ablation's speedup came from (e.g., E1 should reduce `frac_compute` via FP8; E4 should reduce `frac_dataloader` if larger batch better hides I/O).

### 11.4 The composed recipe

After all 5 ablations, identify which "ship" (≥3% speedup, no per-step penalty at T):

```
shipped_ablations = {E_i : speedup_i ≥ 3% AND val_loss_at_T(E_i) ≤ T + ε}
predicted_composed_speedup = sum(speedup_i for E_i in shipped_ablations)
predicted_composed_wall_clock = T_baseline × (1 - predicted_composed_speedup / 100)
```

This is the **headline deliverable.** A single sentence: *"Multimodal capability T (val loss ≤ X) reached in Y hours on 8×H100 with recipe R, vs Y_baseline hours for the unoptimized baseline. Speedup attribution: FP8=α%, batch=β%, dense=γ%, ..."*

### 11.5 What wall-clock is NOT

Same as v2 §11.5: not end-to-end pipeline (data prep, eval rendering, checkpoint I/O excluded), not inference, not downstream task scores, not cost-optimal across cloud providers. **Time-to-target-loss on a fixed 8×H100 node, training-loop only.**

---

## 12. Cross-references

- `docs/Plan.md` — original (now-superseded) project ambition
- `docs/SPEC.md` — original spec; supersedes by `dev/multimodal_spec.md` and this doc
- `dev/multimodal_spec.md` — frozen-SigLIP design; this doc inherits its decisions
- `dev/LOG.md` — running chronology; entries 2026-05-02 (scope reduction) and current entry (v3 pivot to speedrun) are load-bearing
- `dev/sweep_design_v1.md` — preserved for chronology (original 7-phase plan)
- `dev/sweep_design_g_sweep_deferred.md` — **deferred, not abandoned** — execute if v3 finishes early or G-sensitivity turns out load-bearing
- `core/moe.py` — Karpathy's MoE port (G=2 default config we use here)
- `nanochat/dev/LOG.md` 2026-02-19 — upstream MoE finding (per-step win, wall-clock loss; H₄ replicates this for multimodal)
- `nanochat/dev/LOG.md` 2026-02-21 — ClimbMix dataset switch, 27% wall-clock improvement (the speedrun discipline incarnate)
- `nanochat/dev/LOG.md` 2026-02-XX — FP8 +5% capability-matched speedup (H₀ replicates for multimodal)
- `nanochat/dev/moe_fp8.md` — FP8-on-shared-expert recipe (E1 cell uses this)
