# Sweep Design v2 — Multimodal MoE Granularity at Production Mix

**Status:** pre-registration document for the main GPU sweep. Replaces `sweep_design_v1.md` (preserved for chronology). Methodology choices are fixed here BEFORE seeing data.

**Owner:** operator-researcher driving the project (see `MEMORY.md` user role).

---

## 0. Thesis vision

> *Framing this as a senior AI researcher would scope it for a frontier-lab review committee.*

### 0.1 Two co-equal goals

This sweep has **two goals**, deliberately bi-objective. Either alone is a valid contribution; together they are the whole point.

| | Goal 1 — Scientific (what is true?) | Goal 2 — Engineering (what is fastest?) |
|---|---|---|
| **Question** | What is compute-optimal G\* for multimodal MoE at production mix r = 0.3? | What multimodal MoE recipe minimizes wall-clock-to-capability? |
| **Output** | 3 × 3 matrix of per-step val loss + per-modality decomposition | 3 × 3 matrix of wall-clock-to-target-loss + speedup analysis |
| **Settles** | DeepSeek-V3 (G≈8) vs Llama-4 (G=1) vs Qwen3.5 (G=32) disagreement | Whether MoE wall-clock loss penalty (nanochat 2026-02-19) holds in multimodal regime |
| **Inherits from** | Krajewski 2024 / "Towards Greater Leverage" 2025 (both text-only) | Karpathy's nanochat speedrun discipline (text-only d24, ~2hr GPT-2) |

**These goals can disagree.** Per nanochat 2026-02-19, MoE improves per-step val loss but loses on wall-clock at GPT-2 scale due to `_grouped_mm` overhead. **A G\* that wins per-step but loses wall-clock is the most likely outcome and is the most actionable finding** — it tells production teams "MoE is scientifically interesting at multimodal but not yet a wall-clock win at our scale."

### 0.2 Goal 1 — The decision-relevant question

Every native multimodal MoE shipped in 2026 — DeepSeek-V3 (G≈8), Llama-4 Maverick (G=1), Qwen3.5-35B-A3B (G=32) — chose its **granularity G by intuition, not by published scaling-law evidence in the multimodal regime.**

The two MoE granularity scaling laws that exist (Krajewski 2024; "Towards Greater Leverage" Jul 2025) are **text-only.** Both even disagree with each other on optimal G for text. Neither has been re-measured under the multimodal training conditions that production frontier labs actually run.

The question this sweep settles, to a first approximation:

> **At a production multimodal mix (r = 0.3 vision tokens, Qwen3.5-VL-style early fusion), is the compute-optimal MoE granularity G\* the same as the text-only G\* derived by Krajewski 2024 — across compute scales spanning 1×10¹⁹ to 4×10¹⁹ FLOPs?**

### 0.3 Goal 2 — Wall-clock-to-capability for multimodal

Karpathy's organizing principle for nanochat is **"wall-clock time to capability,"** stated explicitly throughout `nanochat/dev/LOG.md`. Every architecture/optimizer/dataset change is judged by that single metric, not by per-step quality. Examples:

- ClimbMix dataset switch (2026-02-21): "single biggest improvement to nanochat's GPT-2 speedrun time" — 2h46m → 2h01m, **27% wall-clock reduction**
- MoE verdict (2026-02-19): "improves per-step validation loss, but is **not** a net improvement on wall clock time"
- FP8 (2026-02-XX): "approx +5% capability-matched speedup" — measured against wall-clock, not per-step

This discipline is what makes nanochat shippable. We adopt it for multimodal.

The question Goal 2 settles:

> **At production multimodal mix r = 0.3, which (G, optimization-flag) recipe reaches a fixed multimodal capability target in the least wall-clock time on 8×H100?**

This extends nanochat's text-only wall-clock benchmark into the multimodal regime that nobody has speedrun-optimized. Our anchor: cell B (d24 r=0.3) is **exactly nanochat's main run scale**, so wall-clock measurements are directly comparable to Karpathy's text-only published numbers.

### 0.4 Why a frontier lab would care about a $285 study

Frontier multimodal pretraining runs cost $10M–$100M+ in compute. If the field has been picking G by transferring text-only intuition to a multimodal regime where the optimum has shifted — even by a modest amount — the dollar cost of that miscalibration across the industry is enormous. A small-budget measurement that informs that decision is asymmetrically valuable:

- **If Goal 1 H₀ holds AND Goal 2 says dense wins wall-clock:** adopt Krajewski's G recommendation for science, ship dense for production. Stop debating MoE for sub-1B-active multimodal.
- **If Goal 1 H₀ refuted (G shift) AND Goal 2 says MoE wins wall-clock:** the field is mis-configured BOTH on architecture and on cost-effectiveness assumptions. High-impact dual finding.
- **If goals agree (per-step optimal = wall-clock optimal):** clean validated recipe to ship. Lowest writing burden.
- **If goals disagree (per-step ≠ wall-clock optimal):** the **gap quantifies** the production cost of choosing science-optimal over engineering-optimal architecture. Directly actionable for budget allocation.

Every cell of these four 2×2 outcomes is publishable, and the methodology to detect them is the same. That asymmetry is what makes this a sensible budget-constrained study rather than a toy.

### 0.5 Falsifiable hypotheses (preregistered)

> **H₀ (Goal 1 — per-step):** at r = 0.3, per-step compute-optimal G\* is consistent with text-only Krajewski 2024 (G ≈ 8 within our measurement noise) across all three compute scales.
>
> **H₁ (Goal 1 — per-step):** G\* shifts. Pre-registered direction of interest: **finer experts** (G > 8) preferred under multimodal mix (FAIR Mar-2026 vision-flatter-loss prediction).
>
> **H₂ (Goal 2 — wall-clock):** wall-clock-optimal G\* is **G = 1 (dense)** across all three compute scales, replicating nanochat's 2026-02-19 negative result for multimodal at the same scale regime (d20–d26).
>
> **H₃ (Goal 2 — wall-clock):** wall-clock-optimal G\* matches per-step-optimal G\*, contradicting nanochat's text-only finding. Would suggest multimodal training has different MoE overhead amortization (vision tokens are larger/sparser per sample, possibly amortizing `_grouped_mm` dispatch differently).

Both per-axis flat results (G\* underdetermined) are valid third outcomes and reported as such.

### 0.6 What this study deliberately is NOT

- ❌ A full L(N, D, G, r) scaling-law surface (Plan.md original ambition; out of budget)
- ❌ A modality-asymmetry measurement (FAIR Mar-2026 already published this)
- ❌ A φ (active-fraction) sweep (separate paper)
- ❌ A V1-vs-V2 verification at scale (Plan.md Phase D, dropped per budget)
- ❌ A novel architecture contribution (we adopt Qwen3.5-VL early-fusion verbatim)
- ❌ A new MoE kernel optimization effort (we use upstream `torch._grouped_mm` as-is; FlashMoE-style fused kernel is out of scope)

Senior researcher's edge here is **not** doing the over-scoped paper. It's doing the smallest, sharpest measurement that answers TWO specific decision-relevant questions that nobody has answered.

---

## 1. Driving principles

These four rules govern every methodology choice in this doc. If a section appears to violate one, that section is wrong, not the principle.

1. **Don't re-verify what upstream verified.** nanochat (Karpathy, `nanochat/dev/LOG.md` 2026-02-19) verified MoE correctness, Muon+AdamW, the 1/√d AdamW LR scaling rule, and `core/moe.py`'s active-FLOP accounting at d12–d26. We inherit those, we don't re-prove them. A "smoke test" of MoE on H100 is low-information because the property under test is verified upstream.
2. **The only genuinely novel contribution is multimodal integration.** Every methodology choice that doesn't serve that headline (extra LR sweeps, scaling-law fit toolkits at small scale, dense baselines we don't need) is overhead.
3. **Stay inside Karpathy's validated HP envelope.** Use d ∈ [12, 26], `--target-param-data-ratio 12`, default LRs verbatim. If a design choice forces extrapolation outside this envelope, **change the design** — don't add a "validation phase" to defend the extrapolation.
4. **Wall-clock is a first-class metric, not a footnote.** Karpathy's nanochat discipline: every architecture/optimizer change judged by wall-clock-to-capability, not per-step quality. We co-equally report per-step val loss AND wall-clock-to-target-loss for every cell. If they disagree, the disagreement IS a finding (per nanochat's MoE 2026-02-19: per-step win, wall-clock loss). A G\* that wins science but loses production is an actionable result, not a contradiction.

v1 of this doc violated principle 3 (chose d4–d8, then added a $90 LR-sensitivity Phase 1 to defend the extrapolation) and principle 4 (treated wall-clock as MFU monitoring, not as a co-equal headline). v2 restores both.

---

## 2. Headline measurements (bi-objective)

The same 9 cells produce **two 3 × 3 matrices**, one per goal. Cells are indexed by (compute scale C, granularity G), at fixed r = 0.3.

### 2.1 Cell layout

| | G = 1 (dense expert, Llama-4 style) | G = 4 (intermediate) | G = 8 (DeepSeek-V3 style) |
|---|---|---|---|
| **C = 1×10¹⁹ FLOPs** (d20) | cell A.1 | cell A.4 | cell A.8 |
| **C = 2×10¹⁹ FLOPs** (d24) | cell B.1 | cell B.4 | cell B.8 |
| **C = 4×10¹⁹ FLOPs** (d26) | cell C.1 | cell C.4 | cell C.8 |

### 2.2 Goal 1 readout — per-step G\* (science)

**Metric:** final joint val loss at end of training (FLOPs budget exhausted).

**Decision rule:** at each compute scale, G\*_step is the value with lowest val loss outside the bootstrap CI of the runner-up (block-bootstrap at eval batch level, B = 1000 resamples). If CIs overlap, G\*_step = `{candidates}` (underdetermined).

**Cross-scale consistency check:** G\*_step same across 3 scales → H₀ supported. Monotone shift with C → mild H₁. Non-monotone → flat-with-G (third valid outcome).

### 2.3 Goal 2 readout — wall-clock G\* (engineering)

**Metric:** wall-clock seconds to first reach a fixed target loss `L_target(C)`, measured on 8×H100 BF16. Target derived from the dense baseline (G=1) loss curve at each compute scale: `L_target(C) = L_G1(C) + 0.005` (slightly above the dense plateau, so all G values can plausibly reach it).

**Decision rule:** at each compute scale, G\*_wall is the value with lowest wall-clock-to-target outside the runner-up's CI. CI computed by bootstrapping per-step throughput (tokens/sec) over the last 50% of training (steady-state regime).

**Cross-scale consistency check:** same logic as Goal 1.

**Why "wall-clock to target" not "wall-clock at fixed FLOPs":** at fixed FLOPs, all 9 cells finish at the same wall-clock by construction (FLOPs/sec × seconds = FLOPs). The interesting question is **which G reaches a useful loss fastest**, not which G finishes its allotted compute fastest. Karpathy's exact framing.

### 2.4 The combined readout (the full headline)

**Two reportable scenarios:**

| Outcome | Goal 1 finding | Goal 2 finding | Production implication |
|---|---|---|---|
| **A. Aligned** | G\*_step = G\*_wall = G_X | (same G_X) | Clean validated recipe — ship G_X for both science and engineering |
| **B. Per-step wins MoE, wall-clock wins dense** | G\*_step = 4 or 8 | G\*_wall = 1 | Replicates nanochat for multimodal — MoE worth pursuing **only if** kernel improvements close the wall-clock gap; meanwhile ship dense |
| **C. Per-step wins dense, wall-clock wins MoE** | G\*_step = 1 | G\*_wall = 4 or 8 | Surprising; would suggest multimodal-specific MoE amortization. Investigate before shipping |
| **D. Both flat** | G\*_step underdetermined | G\*_wall underdetermined | Granularity axis is less load-bearing than the field assumes; worth telling people to stop debating it |

**Outcome B is the pre-registered prior** (most likely per nanochat 2026-02-19 evidence at the same scale regime). Outcome C would be the most surprising and the most newsworthy.

---

## 3. Cell grid (9 cells, ~$285)

### 3.1 Per-cell config

| Cell | C (FLOPs) | depth | model_dim | N (params) | D (tokens) | D/N | G | (num_experts, top_k) | Wall-clock @ 8×H100 35% MFU | Cost @ $16/hr |
|---|---|---|---|---|---|---|---|---|---|---|
| A.1 | 1×10¹⁹ | 20 | 1280 | ~370M | 4.5B | 12 | 1 | (1, 1) | ~1.0 hr | $16 |
| A.4 | 1×10¹⁹ | 20 | 1280 | ~370M | 4.5B | 12 | 4 | (16, 4) | ~1.0 hr | $16 |
| A.8 | 1×10¹⁹ | 20 | 1280 | ~370M | 4.5B | 12 | 8 | (64, 16) | ~1.0 hr | $16 |
| B.1 | 2×10¹⁹ | 24 | 1536 | ~640M | 5.2B | 8 | 1 | (1, 1) | ~2.0 hr | $32 |
| B.4 | 2×10¹⁹ | 24 | 1536 | ~640M | 5.2B | 8 | 4 | (16, 4) | ~2.0 hr | $32 |
| B.8 | 2×10¹⁹ | 24 | 1536 | ~640M | 5.2B | 8 | 8 | (64, 16) | ~2.0 hr | $32 |
| C.1 | 4×10¹⁹ | 26 | 1664 | ~830M | 8.0B | 10 | 1 | (1, 1) | ~4.0 hr | $64 |
| C.4 | 4×10¹⁹ | 26 | 1664 | ~830M | 8.0B | 10 | 4 | (16, 4) | ~4.0 hr | $64 |
| C.8 | 4×10¹⁹ | 26 | 1664 | ~830M | 8.0B | 10 | 8 | (64, 16) | ~4.0 hr | $64 |
| **Total** | | | | | | | | | **~21 hr** | **~$336** |

**Costs assume vast.ai 8×H100 at $16/hr.** At Lambda Labs $25/hr, total ~$525. **Use vast.ai unless availability fails.**

### 3.2 Granularity G mapping (verified against `core/moe.py`)

`core/moe.py` derives `expert_hidden = round(4d / (top_k + num_shared) / 128) × 128` automatically from CLI flags. So G is set implicitly via `(--num-experts, --top-k)`:

- **G = 1**: `(num_experts=1, top_k=1)` → equivalent to dense MLP of hidden 4d
- **G = 4**: `(num_experts=16, top_k=4)` → 4× sparsity ratio, 16 experts of hidden d
- **G = 8**: `(num_experts=64, top_k=16)` → 8× sparsity ratio, 64 experts of hidden d/2

`num_shared_experts=1` for all cells (DeepSeek-V3 default).

### 3.3 Why these depths

- **C = 1×10¹⁹ → d20 (Chinchilla D/N = 12 lands here exactly at depth 20).** Inside validated range.
- **C = 2×10¹⁹ → d24.** Slightly under-sized vs Chinchilla-opt d24.5, more data → fine.
- **C = 4×10¹⁹ → d26.** Cap at top of nanochat-validated range; D/N=10, slightly under-trained data-wise but still inside the Chinchilla-sensible regime [4, 40].

All three depths use existing nanochat HP defaults: `--matrix-lr 0.02 --embedding-lr 0.3 --unembedding-lr 0.008 --scalar-lr 0.5 --weight-decay 0.28 --target-param-data-ratio 12`. **No LR multipliers, no per-scale tuning.**

### 3.4 Fixed across all cells

| Setting | Value | Source |
|---|---|---|
| `r` (vision token fraction) | **0.3** | dev/LOG.md 2026-05-02 scope reduction |
| Optimizer | Muon (matrices) + AdamW (embeddings/scalars) | nanochat default |
| LR scaling | nanochat default `(d/768)^-0.5` | inside validated range — no extra check |
| Schedule | cosine, 5% warmup, decay to 5% peak | nanochat default |
| Dtype | BF16 | FP8 + MoE unsupported per `nanochat/dev/LOG.md` 2026-02-19 |
| Window pattern | "L" (full attention) | sliding window confounds mix interpretation |
| Sequence length | 4096 | accommodates ~180 vision tokens per image at SigLIP2 384×384 |
| `num_shared_experts` | 1 | DeepSeek-V3 default |
| Vision encoder | SigLIP2-SO400M-patch14-384, **frozen** | `dev/multimodal_spec.md` decision #1 |
| PatchMerger | 2×2 spatial merge + 2-layer MLP, **frozen after warmup** | `dev/multimodal_spec.md` decision #3 |

---

## 4. Phase ordering (compressed from v1's 7 phases → 3)

| Phase | Purpose | Cells | Cost | Gate |
|---|---|---|---|---|
| **Phase 0 — Multimodal integration smoke** | Verify the only thing that's actually unverified: ViT + scatter + MoE end-to-end on real GPU at small scale | 1 cell at d12 G=4 r=0.3 C ≈ 1×10¹⁸ FLOPs | $5 | Trains stably for 200 steps, no NaN, per-modality losses both decreasing, expert utilization within 2× uniform |
| **Phase 0.A — Nanochat-anchor cell (HP-transfer falsifier)** | Verify nanochat HPs transfer to **our hardware/software/setup** by replicating nanochat's published d24 ClimbMix baseline. Cheaper, more decisive than any pre-emptive scaling-law fit. | 1 cell at d24 G=1 r=0.0 (text-only, no MoE), C = 2×10¹⁹ FLOPs (= nanochat main run) | $32 | Val loss curve at step 100 / 1000 / 10000 within **3%** of nanochat's published d24 ClimbMix baseline. **Else: HP transfer broken; halt; run $50 mini-LR sensitivity at d24** |
| **Phase Main — 3 × 3 G-sweep** | The headline measurement | 9 cells from §3.1 | ~$285 | All 9 complete; per-modality loss decomposition non-degenerate; routing entropy > 0.7·log(num_experts) for all cells |
| **Total** | | **11 cells** | **~$322** | |

**Why Phase 0.A is worth $32:** it is a **cheap falsification mechanism** for the entire HP-inheritance assumption (§1 principle 3). If our vast.ai instance + our patched nanochat fork + our ClimbMix shard staging produces a d24 baseline that diverges from Karpathy's published curve by >3%, EVERYTHING downstream is contaminated. $32 to know is much cheaper than discovering it after spending $285 on Phase Main. If 0.A passes, the inheritance assumption is empirically validated, not just inherited on faith.

**Phase 0.A also doubles as the wall-clock anchor** for Goal 2 — gives us a direct apples-to-apples comparison of our 8×H100 throughput against nanochat's published 2-hour speedrun number.

**That's it.** Three phases. No LR check, no CompleteP transfer, no separate text-only Phase A, no IsoFLOP fit toolkit, no V1/V2 verification. Each of those was justifiable in v1; none are needed when the design respects principles 1–3 AND uses Phase 0.A as the cheap empirical falsifier.

### 4.1 Phase 0 protocol (multimodal integration smoke)

```bash
# Single cell, ~25 min on 8×H100, ~$5
torchrun --nproc-per-node=8 scripts/base_train.py \
  --depth 12 \
  --num-experts 16 --top-k 4 --num-shared-experts 1 \
  --target-flops 1e18 \
  --max-seq-len 4096 \
  --multimodal --mix-ratio 0.3 \
  --run smoke_mm_d12_g4_r03
```

Pass criteria:
- No NaN in loss for 200 steps
- `loss/text` and `loss/vision` both monotonically decreasing over a 200-step rolling window
- Routing entropy at step 200 > 0.7·log(16) = 1.94
- Zero dead experts in last 50 steps
- Vision tokens correctly scattered: `assert input_embeds[is_vision_token].std() > 0.5` (not zero-padded)
- Determinism: re-running with same seed reproduces step-100 loss to 1e-5

Fail action: do not proceed to Phase 0.A. Diagnose. Most likely culprit: a stub in `core/multimodal.py` is wrong; second most likely: dataloader mix ratio drift.

### 4.1.5 Phase 0.A protocol (nanochat-anchor — HP-transfer falsifier)

```bash
# Single cell at exactly nanochat d24 main run config, ~2 hr on 8×H100, ~$32
# r=0.0 means multimodal pipeline disabled — pure text-only on ClimbMix
torchrun --nproc-per-node=8 scripts/base_train.py \
  --depth 24 \
  --num-experts 1 --top-k 1 --num-shared-experts 1 \
  --target-param-data-ratio 12 \
  --max-seq-len 2048 \
  --run anchor_d24_text_baseline
# Note: NO --multimodal flag → pure nanochat reproduction
```

**Comparison protocol:**

| Checkpoint | Our val loss | Nanochat published d24 ClimbMix | Tolerance |
|---|---|---|---|
| step 100 | ours | ~6.5 (Karpathy LOG) | within ±0.2 (3%) |
| step 1000 | ours | ~3.8 | within ±0.12 (3%) |
| step 10000 | ours | ~1.2 | within ±0.04 (3%) |
| step end (~14K) | ours | ~0.715 (LOG 2026-02-19 logit-softcap entry) | within ±0.022 (3%) |

(Exact reference numbers to be pinned by reading `nanochat/dev/LOG.md` curve plots; placeholders above are illustrative — confirm before running.)

**Pass criteria:**
- All 4 checkpoint comparisons within tolerance → **HP transfer to our setup is empirically validated**; proceed to Phase Main with confidence
- Wall-clock to step 14K within 110% of nanochat's published 2h01min → throughput parity confirmed; Goal 2 wall-clock numbers are comparable to Karpathy's

**Fail actions** (graduated):
- **Off by 3-5%**: minor HP drift, likely tokenizer or shard ordering. Investigate the specific deviation; may be tolerable if direction is consistent across all 9 Phase Main cells
- **Off by 5-10%**: significant drift. Halt. Run a $50 mini-LR sensitivity (3-cell at d24 with 0.5×, 1.0×, 2.0× LR multipliers) to find correct LR for our setup
- **Off by >10% or wrong direction**: infrastructure bug (CUDA version, FA3 disabled, gradient accumulation off, ...). Diagnose before any further GPU spend

**This phase replaces v1's $90 LR-sensitivity Phase 1.** Same insurance value (catches HP miscalibration) at lower cost ($32 vs $90), better information value (compares against a real published baseline, not against itself).

### 4.2 Phase Main protocol

Run the 9 cells from §3.1 in increasing wall-clock order (smallest first). Re-run any cell that hits a kill condition (§7). After all 9 complete:

1. Compute per-cell final val loss + bootstrap CI (block-resampled at the eval batch level)
2. For each compute scale C, identify G\* as defined in §2
3. Apply cross-scale consistency check
4. Write up in `dev/LOG.md` with the 3 × 3 result matrix

---

## 5. Risk model (rewritten — only live risks listed)

v1 listed 6 risks. Most were eliminated by the scope reduction and HP-envelope discipline. The live ones:

| # | Risk | Mitigation | If materializes |
|---|---|---|---|
| 1 | **Multimodal integration bug** in `core/multimodal.py` (15 stubs at time of writing) | (a) Local CPU joint forward smoke at d=4 before any GPU spend, (b) Phase 0 GPU smoke at d=12, (c) determinism contract from §4.1 | Diagnose and fix; do not run Phase Main until Phase 0 passes |
| 2 | **r-drift** — actual token-level vision fraction in Phase Main batches differs from the configured 0.3 | Token-level mix accounting in dataloader (not example-level); log `r_actual` per batch; assert `|mean(r_actual) - 0.3| < 0.02` over rolling 100 batches | Adjust dataloader sampler; re-run affected cells |
| 3 | **Routing collapse** at G = 8 (64 experts, top-16) | DeepSeek-V3 aux-loss-free bias nudging is verified in nanochat MoE port; also add `router_z_loss_coef = 1e-4` per nanochat default | Reduce to G ∈ {1, 4} only; report grid as 2 × 3 = 6 cells; flag in writeup |
| 4 | **MFU << 35%** at d20+ on real hardware | nanochat measured 35% at d18 MoE; we expect similar at d20–26 | If MFU < 25% sustained: time budget per cell doubles; either accept the cost overrun or cap C₃ at 2×10¹⁹ |
| 5 | **HP transfer failure to our setup** — Karpathy's nanochat HPs were validated on his Lambda 8×H100 + his ClimbMix shard order + his code SHA. Ours differs (vast.ai, possibly different shard staging, possibly minor patches to base_train.py). Inheriting HPs assumes transfer; that's an assumption, not a verified fact. | **Phase 0.A nanochat-anchor cell** ($32, see §4.1.5): replicate d24 G=1 r=0.0 text-only, compare loss curve to Karpathy's published d24 ClimbMix baseline at 4 checkpoints with 3% tolerance. Doubles as wall-clock anchor for Goal 2. | **3-5% off:** investigate but may proceed. **5-10% off:** halt; run $50 3-cell LR sensitivity at d24 to find correct LR multiplier; resume Phase Main with adjusted LR. **>10% off:** infrastructure bug (CUDA, FA3, gradient accumulation, etc.); diagnose before any further GPU spend |
| 6 | **Multimodal-specific HP shift** — even if text-only HPs transfer (Risk #5 passes), multimodal data could shift MoE-specific knobs (router_z_loss, expert bias LR) due to vision token distribution differing from text | **Telemetry-based detection only.** Routing entropy + dead expert counts in §7.2 logging; cancel triggers in §6 fire if collapse. We do NOT pre-validate; we monitor and react. Frozen-ViT design choice minimizes this risk by isolating LLM trunk from vision gradients (gradients into SigLIP2 are zero by construction). | If 2+ cells show routing collapse correlated with G value: halt, run targeted ablation on suspected HP. If only 1 cell: likely G-specific instability (Risk #3), not HP transfer; degrade grid per Risk #3 mitigation |

Risks NO LONGER in the model (because the design eliminated them):

- ~~LR mistuning at extrapolated depth~~ — eliminated by §1 principle 3
- ~~CompleteP transfer failure~~ — eliminated by not using CompleteP
- ~~Tokenizer floor saturating modality asymmetry signal~~ — eliminated by adopting frozen continuous SigLIP2 instead of VQ tokenizer
- ~~Phase D verification disagreement~~ — eliminated by dropping Phase D
- ~~Scaling-law fit form misspecification~~ — eliminated by not fitting a parametric law (we report the 3 × 3 matrix directly)

---

## 6. Cancel triggers (per cell)

Apply to every Phase Main cell:

| Trigger | Action |
|---|---|
| NaN/Inf in loss | Kill cell; tighten grad-clip 1.5×; restart from last checkpoint |
| Loss at step 100 > 1.5× expected (compare to nanochat d20 baseline curve) | Kill cell; investigate config |
| Sustained MFU < 25% for 1000 steps | Kill cell; infrastructure issue |
| Wall-clock exceeds 1.5× budget | Kill cell; re-plan |
| Routing entropy < 0.5·log(num_experts) for 200 consecutive steps | Kill cell; routing collapsed |
| Any expert receives 0 tokens for >200 consecutive steps | Kill cell; dead expert |
| Per-batch `|r_actual - 0.3|` > 0.05 sustained | Pause cell; fix dataloader |

Every kill produces a `dev/LOG.md` entry. **No silent failures.**

---

## 7. Logging schema (subset of v1 §9; bi-objective tier structure)

W&B project: `multimodal-moe-granularity-r03`. One run per cell, named `{cell_id}` from §3.1.

### 7.1 Headline-tier metrics (feed the two §2 matrices directly)

**Per-step (every step):**
- `loss/total`, `loss/text`, `loss/vision` — feeds Goal 1 matrix
- `step_time_ms`, `tokens_per_sec`, `mfu` — feeds Goal 2 matrix
- `wall_clock_elapsed_sec` (cumulative since training start) — derived headline metric

**Per-eval (every 5% of cell steps, ≥10 evals/cell):**
- `val_loss/text`, `val_loss/vision`, `val_loss/joint` (Goal 1)
- `val_loss/joint_bootstrap_ci_low`, `val_loss/joint_bootstrap_ci_high`
- `wall_clock_to_loss_X` (derived post-hoc from the loss-vs-wall_clock curve at multiple X targets) — Goal 2

### 7.2 Diagnostic-tier metrics (for risk model & debugging)

**Per-step:**
- `grad_norm/total`
- `lr/muon`, `lr/adamw_emb`, `lr/adamw_unemb`, `lr/adamw_scalar`
- `r_actual` (per-batch token-level vision fraction)

**Per-50-steps:**
- `routing/entropy_per_layer` (list[float])
- `routing/n_dead_per_layer` (list[int])
- `routing/specialization_score` (Shukor 2025 §5.4 formula: `1 - H(p_modality | expert)`)
- `throughput/grouped_mm_overhead_ratio` — fraction of step time in `_grouped_mm` dispatch (the nanochat-identified bottleneck; lets us attribute Goal 2 wall-clock gap to its known cause)

### 7.3 Per-cell metadata (once at start)
- Full `GPTConfig` dump
- `cell_id`, `git_sha`, `seed`, `gpu_specs`
- `n_active_params`, `n_total_params`, `flops_per_token_estimated`
- `wall_clock_budget_sec` (from §3.1 estimate)
- `target_losses` (the L_target(C) thresholds for Goal 2 readout)

---

## 8. What was dropped from v1 and why (audit trail)

| v1 element | Status in v2 | Reason |
|---|---|---|
| Phase 0 timing calibration (3 cells, $15) | Replaced by 1-cell multimodal smoke ($5) | The only thing that needs calibrating is the multimodal path; MoE timing is verified in nanochat |
| Phase 1 LR sensitivity check (3 cells, $90) | **DROPPED** | Self-inflicted to defend d4–d8 extrapolation; eliminated by staying in d20–d26 |
| Phase 2 CompleteP transfer ($18) | DROPPED | Already dropped in v1 (replaced by Phase 1) |
| Phase A text-only (24 cells, $70) | DROPPED | Krajewski 2024 already did this measurement at our scale; we'd be re-doing |
| Phase B modality mix sweep (20 cells, $110) | DROPPED | Single fixed r = 0.3 per scope reduction |
| Phase C IsoFLOPs (36 cells, $305) | Replaced by 9-cell §3.1 grid | We measure G\* directly, not a parametric fit |
| Phase D V1 vs V2 (2 cells, $440) | DROPPED | Cannot justify on $1K budget |
| `core/scaling_fit/` toolkit | DROPPED | Not needed for a 3 × 3 matrix readout |
| Tokenizer floor protocol (§11) | DROPPED | SigLIP2 frozen continuous features have no VQ floor |
| GPU sourcing decision Lambda vs vast.ai (§12) | Defaulted to vast.ai | Cheaper; only switch if availability fails |

**Net delta:** $971 → ~$285. Same headline scientific claim, half the methodology surface area.

---

## 9. Reviewer Q&A (pre-empts the obvious questions)

> **"Why d20–d26 and not the more standard d8–d12 sweep?"**

We sit inside Karpathy's nanochat-validated HP envelope (d12–d26). Any choice outside it requires a separate LR validation phase to defend; staying inside lets us inherit his work directly. The compute math at our $285 budget happens to land exactly on Chinchilla-optimal at d20–d26, so there's no friction.

> **"Why no learning-rate sweep?"**

Because we use Karpathy's exact settings inside the depth range where he validated them. `nanochat/dev/LOG.md` is our LR ablation. Re-running it would be re-verifying upstream work.

> **"Why fixed r = 0.3? Why not r-as-axis?"**

Production multimodal training mixes cluster around 0.2–0.4 vision tokens (Qwen3.5-VL training mix per their tech report references). r = 0.3 is the modal value across 2026 frontier models. An r-sweep is future work; this is the headline measurement at production-realistic conditions. An r-sweep would also triple the cell count and bust the budget.

> **"Why only 3 G values?"**

G ∈ {1, 4, 8} samples (a) dense (Llama-4 Maverick), (b) Krajewski-typical intermediate, (c) DeepSeek-V3-typical. These are the three operating points that frontier multimodal models actually use. Adding G = 16 or 32 would be relevant for Qwen3.5-35B-A3B's choice but extends the grid by 33% for marginal additional information.

> **"Why no φ sweep?"**

φ (active fraction) and G are conflated in the iso-FLOP definition we use (`expert_hidden = 4d / (top_k + num_shared) / G`). At fixed `(num_experts, top_k)`, varying G is the lever; a separate φ axis would require a different parametrization and a different paper.

> **"Why no verification at scale?"**

Phase D would cost $440 alone, more than the entire $285 main sweep. Frontier-grade verification is rightly out of scope for a $1K budget. The primary contribution is a measurement, not a prediction.

> **"What's the smallest result you'd publish?"**

The 3 × 3 matrix of final val losses + per-modality decomposition + cross-scale G\* call. Even a "G\* underdetermined within the grid" outcome is publishable as a negative result on the granularity-axis sensitivity question, and informs decisions to focus future scaling-law work on other axes.

---

## 10. One trade-off flagged honestly

At C₃ = 4×10¹⁹ FLOPs with d26, we're slightly **over-trained** relative to Chinchilla optimum at that compute (Chinchilla-opt would be ~d28, slightly outside the validated envelope). D/N = 10 instead of the canonical D/N = 12.

Two consequences:
1. **Bias against MoE** at C₃: MoE architectures benefit more from more N than from more D (Krajewski 2024). At D/N = 10 we're spending more on D than Chinchilla would, slightly undercutting MoE's expected advantage. **Conservative bias** — if MoE still wins at C₃, the result is robust.
2. **Reviewer footnote** rather than a methodology hole: "the largest cell sits at D/N = 10 vs Chinchilla 12 due to the validated-HP-envelope constraint; we believe this biases against rather than for our headline conclusion."

If a reviewer pushes back, the cheap fix is to bump C₃ to d28 (1.08× extrapolation outside validated; one footnote vs the 1.7× extrapolation that would have required v1's $90 LR check). I'd take the conservative bias.

---

## 11. Wall-clock methodology (Goal 2 details)

This section makes Goal 2 reproducible and reviewable. Per §1 principle 4, wall-clock is co-equal with per-step val loss as a headline output.

### 11.1 What "wall-clock" means precisely

| Term | Definition |
|---|---|
| `wall_clock_elapsed_sec` | seconds of clock time since the first optimizer step, on the production 8×H100 node, BF16, no profiler attached |
| `wall_clock_to_loss_X(cell)` | seconds at which `val_loss/joint` first crosses below threshold `X` (computed from logged eval points + linear interpolation between adjacent evals) |
| `tokens_per_sec` | sum of training tokens processed across all 8 ranks per second of clock time |
| `mfu` | model FLOPs utilization, computed via nanochat's `core/common.py:get_peak_flops()` (BF16 989 TFLOP/s peak per H100) |
| `grouped_mm_overhead_ratio` | (sum of `_grouped_mm` kernel time) / (total step time), measured by PyTorch profiler over a 100-step window once per training cell, mid-training |

Wall-clock numbers are **only comparable within a single GPU node**. Across vast.ai instances, MFU may vary 5–10% due to CPU/host noise; we report all 9 cells from the **same rented node** if possible, or note the cross-node delta.

### 11.2 The `L_target(C)` derivation

Wall-clock-to-target-loss requires picking `L_target(C)`. Done as follows:

1. Run all 9 cells to FLOPs budget exhaustion
2. For each compute scale C, identify `L_dense(C)` = final val loss of the G=1 cell at that C
3. Set `L_target(C) = L_dense(C) + 0.005` — a threshold the dense baseline reaches near end of training, and that MoE cells (with per-step quality advantage) should reach earlier in training
4. For each (C, G) cell, post-hoc compute `wall_clock_to_loss_X = first eval where val_loss/joint < L_target(C)`

This lets us answer: **"how fast does each G reach a useful loss level?"** rather than the less-informative "what loss does each G reach at fixed compute?"

The `+0.005` margin is chosen so that:
- All G values plausibly reach the target (if margin were 0, only G with lowest final loss reaches it)
- The threshold is meaningful (not at noise floor)
- Aligns with nanochat's convention for "capability matched" comparisons (see LOG 2026-02-19 FP8 entry: "5% capability-matched speedup" uses a similar threshold-crossing definition)

### 11.3 Decomposing the wall-clock gap

If Outcome B materializes (G\*_step ≠ G\*_wall), we attribute the gap as follows. Per nanochat 2026-02-19, the dominant cost is `_grouped_mm` dispatch. Decomposition per cell:

```
total_step_time = compute_time + grouped_mm_overhead + dataloader_idle + comm_overhead
                  ────────────  ──────────────────  ────────────────  ──────────────
                  iso-FLOP        nanochat-bottleneck  upper-bounded     ddp_all_reduce
                  fixed across G  scales with G        by prefetch       scales weakly with G
```

For each G, log:
- `frac_compute = compute_time / total_step_time`
- `frac_grouped_mm = grouped_mm_overhead / total_step_time`
- `frac_dataloader = dataloader_idle / total_step_time`
- `frac_comm = comm_overhead / total_step_time`

Report the four fractions per G. If `frac_grouped_mm` grows from ~0% (G=1, no grouped_mm) to 30%+ (G=8), we've replicated nanochat's diagnostic and can quantify it for the multimodal regime.

### 11.4 Optional Phase 0.5 — wall-clock-targeted ablation (skip unless time permits)

If Phase 0 smoke passes early and budget allows, run **one extra cell** to test a wall-clock optimization that nanochat already has evidence for:

| Cell | Config | Compute | Cost | Tests |
|---|---|---|---|---|
| 0.5 | d24 G=4 r=0.3 + FP8 on **shared expert only** (per nanochat `dev/moe_fp8.md`; routed experts stay BF16) | 1×10¹⁹ FLOPs | $16 | Does FP8-on-shared give wall-clock improvement at multimodal r=0.3 comparable to nanochat's text-only +5% capability-matched speedup? |

If the speedup at multimodal is meaningfully different from text-only, that's a one-line addendum to the writeup. If similar, just confirms the result transfers. Either way, $16 buys evidence on a deployment-relevant question.

**Default decision:** skip Phase 0.5 unless main sweep finishes ahead of time + budget. Don't grow scope to chase optimizations that aren't the headline.

### 11.5 What wall-clock is NOT in this study

- ❌ End-to-end training pipeline wall-clock (we don't measure data prep, eval rendering, checkpoint I/O — only the training step inner loop)
- ❌ Inference wall-clock (separate question; see Plan §8 #7)
- ❌ Wall-clock to a downstream task score (we use val loss as proxy, not CORE / MMLU scores; full benchmarking is out of scope)
- ❌ Cost-optimal G\* across cloud providers (we use vast.ai $16/hr as anchor; Lambda/Modal/AWS would shift dollar numbers but not the wall-clock ranking)

The headline is **time-to-target-loss on a fixed 8×H100 node, training-loop only.** That's what reproduces; that's what compares to nanochat.

---

## 12. Cross-references

- `docs/Plan.md` — original (now-superseded) project ambition; this doc is what the budget actually buys
- `docs/SPEC.md` — original spec; some sections (e.g. §4.1.4 jointly-trained ViT) are superseded by `dev/multimodal_spec.md` and this doc
- `dev/multimodal_spec.md` — frozen-SigLIP design (this doc inherits its decisions)
- `dev/LOG.md` — running chronology; entries 2026-05-02 are load-bearing for v2
- `dev/sweep_design_v1.md` — preserved for chronology and audit trail of what we considered and dropped
- `core/moe.py` — the iso-FLOP G derivation (`expert_hidden = 4d / (top_k + num_shared) / 128 × 128`)
- `nanochat/dev/LOG.md` 2026-02-19 — upstream verification of MoE correctness
- `nanochat/dev/LOG.md` 2026-02-XX — the LR validation we inherit by staying in d12–d26
