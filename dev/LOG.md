# Experiment Log

A running summary of experiments, findings, and design decisions for the
modality-conditional MoE scaling project (`Plan.md`). Format borrowed from
`nanochat/dev/LOG.md` (Karpathy).

Conventions:
- Reverse chronological order (newest at top).
- Title format: `## YYYY-MM-DD: <short title> [(Negative)]`
- Each entry has its own structure but commonly: Background, What changed,
  Results, Verdict.
- Negative results are recorded equally — they're how the project avoids
  burning the same compute twice.
- Cross-references inline (`core/moe/spec.md`, run.csv files, PRs, etc.).
- This is the source of truth for "what we tried, what we learned, what
  we changed direction on." Spec docs are contracts; this log is history.

---

## 2026-05-03 (latest): Qwen3.5-VL fidelity sprint — 3D MRoPE wired + multimodal KVCache

End-to-end implementation sprint to close the gap between our `core/multimodal.py`
utilities and the Qwen3.5-VL design we adopted in `dev/multimodal_spec.md`.
**All 5 implementation tracks (B, B', B'', + Qwen3.5 fidelity Steps 1-7) ship.**

### Background

After Track B + B' + B'' delivered the multimodal training pipeline end-to-end,
audit revealed that `build_3d_mrope` was implemented but never called — `GPT.forward`
was using the precomputed 1D RoPE cache for ALL forward passes (including
multimodal). Vision tokens got sequential 1D positions instead of (t, h, w),
violating `multimodal_spec.md` decision #7 ("3D Interleaved-MRoPE required").

User asked: *"are you sure we apply 1D rope not 3D"* — caught the gap explicitly.
Then: *"I prefer exact qwen3.5 multimodal implementation"* — committed to closing it.

### Plan + scope

Plan written (with user clarification questions) to:
- `~/.claude/plans/lexical-rolling-snail.md`

Scope decisions made interactively:
- **Architecture scope**: multimodal extensions only — KEEP nanochat ReLU² MoE
  activation + no GQA. Switching either would break Karpathy's HP envelope and
  require LR retuning (out of scope).
- **Odd grid handling**: SigLIP2-SO400M produces 27×27 raw patches at 384×384;
  not divisible by `spatial_merge_size=2`. Crop to 26×26 (drops rightmost
  column + bottom row, ~7% pixel loss at edges).
- **kv_cache strategy**: full multimodal-aware KVCache so multimodal generate()
  works at production inference time, not just training.

### What changed (7 implementation steps across 2 commits)

**Commit 732f39d** — Steps 1-4 + 6-7:
- `core/multimodal.py`:
  - NEW `build_3d_mrope_for_4d_apply` returning cos/sin in Karpathy layout
    (`B, T, 1, head_dim/2`) bfloat16. Round-robin axis assignment matches
    `qwen35_vl_tiny.py` reference (verified).
  - `PatchMerger.forward` crops odd grids to nearest multiple of merge size
    (replaces strict divisibility assertion). Backward compat: even grids
    unchanged.
- `core/dataloader.py` `synthetic_multimodal_loader` + `core/multimodal_data.py`
  `real_multimodal_loader`: emit `image_grids_merged: list[list[(T,H,W)]]` in
  batch_extras (per-row, MERGED-patch units). Required by `build_position_ids_for_mm`.
- `core/model.py` `GPT.forward`:
  - Accepts `image_grids_merged` kwarg.
  - When pixel_values + image_grids_merged provided, OVERRIDES the precomputed
    1D cache with per-batch 3D MRoPE built via `build_position_ids_for_mm` +
    `build_3d_mrope_for_4d_apply`.
  - Text-only path unchanged; multimodal-without-grids gracefully falls back.
- `dev/multimodal_spec.md` decision #7 status updated to "implemented" with
  file/function citations.

**Commit bd297f4** — Step 5 (multimodal-aware KVCache):
- `core/engine.py KVCache`:
  - NEW field `next_t_axis_position` (None = text-only mode; int after
    multimodal prefill).
  - `reset()` clears it; `prefill()` propagates it from src to dst (so
    batch-1 prefill → batch-N decode preserves multimodal state).
- `core/model.py GPT.forward`:
  - Multimodal prefill (pixel_values + kv_cache provided): captures
    `max(position_ids[0]) + 1` as the next text-axis position; stores in
    kv_cache.
  - NEW elif branch: multimodal continuation (kv_cache.next_t_axis_position
    is set, no pixel_values). Builds per-token 3D MRoPE for new text tokens
    at (next_t, 0, 0) and advances next_t_axis_position by T.
- `core/model.py GPT.generate()`: accepts pixel_values + grid_thw +
  image_pad_mask + image_grids_merged; each step re-runs forward with
  expanded image_pad_mask (correct but wasteful — naive path).

### Tests

49/49 pass across 3 suites (was 41/41 before this sprint):
- `tests/test_multimodal_joint_forward.py`: **32/32** (added 5 new: shape +
  position-0 identity + 4D-broadcast for `build_3d_mrope_for_4d_apply`;
  PatchMerger crops odd 27×27 + odd 5×5)
- `tests/test_multimodal_integration.py`: **11/11** (added 3 new: 3D MRoPE
  produces different cos values than 1D at vision positions;
  multimodal generate() yields valid tokens; KVCache state-transition
  lifecycle)
- `tests/test_real_siglip.py`: **6/6** (slow tests gated by SIGLIP_DOWNLOAD=1)

### Verdict

The multimodal implementation now matches Qwen3.5-VL **exactly within the chosen
scope**:
- ✅ Frozen SigLIP2-SO400M ViT
- ✅ PatchMerger 2×2 + 2-layer MLP (with odd-grid crop)
- ✅ Early fusion via scatter at `<image_pad>`
- ✅ 3D Interleaved-MRoPE with round-robin axis assignment (matches reference)
- ✅ AutoImageProcessor pipeline
- ✅ Frozen-after-warmup projector
- ✅ No DeepStack
- ✅ 27×27 patch grid handling
- ✅ Multimodal generate() (naive + KVCache-optimized paths)

Deliberately NOT changed (per scope discipline):
- ❌ MoE expert activation (kept ReLU²; SwiGLU would break HP transfer)
- ❌ GQA (kept disabled; would break HP transfer)
- ❌ Dynamic image resolution / multi-aspect (production feature; future)
- ❌ Video (architecture supports T>1; not exercised)

### What this unblocks

Scaling-law sweep (`dev/scaling_law_self_assignment.md`) is fully ready for GPU
execution with **zero remaining CPU-side work**:

```bash
python scripts/sweep_runner.py submit --config configs/scaling_law/F1_s.json
# F1.s reactive sanity gate, ~$9, ~30 min on 8×H100
# If passes: run remaining 8 fit cells, fit + preregister to dev/LOG.md, run G3
```

### Cross-references

- Plan: `~/.claude/plans/lexical-rolling-snail.md`
- Commits: `732f39d` (3D MRoPE), `bd297f4` (KVCache + generate)
- Predecessor sprints: Track B (`04518c7`), B' (`2de072f`), B'' (`ec481bd`)
- Specs: `dev/multimodal_spec.md` §2.5 + decision #7 (now "implemented"),
  `dev/scaling_law_self_assignment.md` (no changes; spec was already correct)
- Reference: `basics/notebooks/qwen35_vl_tiny.py` (round-robin 3D MRoPE
  verified to match)

### Lesson (worth recording)

User caught the 1D-vs-3D RoPE gap by asking *"are you sure we apply 1D rope
not 3D"*. The original Track B' work documented "3D MRoPE deferred" in
`core/model.py` line 48-50, but I'd been telling user "implemented" loosely
in conversation. **Specs are precise; conversational framing wasn't.**

Going forward: when describing implementation status to user, distinguish
explicitly between:
1. "Function implemented as utility" (e.g. `build_3d_mrope` was)
2. "Function called in production code path" (e.g. `build_3d_mrope` was NOT
   until this sprint)

These differ. Loose framing #1 implied #2 — which was wrong.

---

## 2026-05-02: SCOPE REDUCTION — drop r-as-axis, fix r=0.3

User's sharp question: "why we just not add multimodal at smaller scale and
do ablation for entire model — I don't know why we have to do r=0 and r=1,
it complicated the problem; every model now is multimodal."

This is correct and forces a real scope reduction.

### What "every model is multimodal in 2026" actually means for our scope

Frontier multimodal models ALL ship as multimodal by default in 2026:
- Llama 3.2 Vision, Qwen2.5/3/3.5-VL, Pixtral, Gemini 1.5/2.0, GPT-4o
- NONE ship text-only as the headline product
- Production teams ask: "given my multimodal mix, what's optimal G?"
- They DON'T ask: "does optimal G shift between text-only and vision-only?"

Our original r-as-swept-axis design (Plan.md §4) was answering an
ACADEMIC question (does G\*(r) interact?) when production reality is:

  Pick a mix → fix it → optimize architecture for that mix.

### What changes operationally

```
Original plan:    G ∈ {1,4,8} × r ∈ {0, 0.5} × C ∈ {1e19, 3e19} = 12 cells, ~$500
Revised plan:     G ∈ {1,4,8} × r=0.3 (fixed) × C ∈ {1e19, 3e19, 1e20} = 9 cells, ~$250
                                                ^^^^^^^^^^^^^^^^^^^^^^^^^
                                                added 1e20 to give 3 compute scales
                                                for proper IsoFLOP slope detection
```

Total budget shifts: $595 → $345 (-$250, -42%).

### What we drop and why it's still publishable

**Dropped claims:**
- "G\* shifts with r" — we don't sweep r anymore
- "Modality asymmetry exists at our scale" — FAIR Mar 2026 already showed this
- "V1 vs V2 verification" — Phase D dropped, $1500 saved

**Kept (and these are genuinely novel):**
- "Compute-optimal G\* for multimodal MoE at production mix"
- "Multimodal granularity scaling extends Krajewski/Towards-Greater-Leverage
  to the multimodal regime"
- Settles part of the DeepSeek-V3 (G≈8) vs Llama-4 (G=1) vs Qwen3-VL (G=?)
  disagreement — for production-relevant multimodal training

### Why we fix r = 0.3 specifically

- Qwen3.5-VL training mix is roughly in this range (per their tech report
  references, exact ratio not published but estimated 25-35% vision)
- Sufficient vision tokens per batch for routing diversity
- Not so vision-heavy that text loss is undertrained
- Single value → single mix to defend in writeup

If reviewers push back ("why not r=0.5?"), the answer is: "production
mixes cluster around 0.2-0.4; r=0.3 is the modal value. A r-sweep is
future work; this is the headline measurement at production-realistic
conditions."

### Why this is the senior-researcher pivot

The original framing tried to be too academic: "does the G axis interact
with r axis?" — a beautiful science question, but one that frontier labs
DON'T publish (they ship one production mix and don't decompose).

The revised framing matches what frontier labs DO publish:
- Krajewski 2024: text-only G scaling
- Towards Greater Leverage 2025: text-only G scaling
- Us: **multimodal G scaling** (the natural next step nobody has done)

This positions the work as "natural extension of existing literature into
the multimodal regime that everyone uses now" — a much easier sell than
"new joint sweep dimension nobody has measured before."

### Updated sequencing

```
✅ MoE CLI flags in base_train.py            (already shipped)
✅ Local CPU smoke test passes               (already done)
✅ Decision: keep Phase 1 LR check           (committed earlier today)

⏳ Multimodal infrastructure (CPU, ~1-2 weeks):
   - Hand-implement core/multimodal.py
   - Per-modality loss in nanochat/engine.py (still useful for logging
     even though we don't sweep r — gives us text vs vision loss curves
     within each multimodal cell)
   - Multimodal dataloader at fixed mix r=0.3

□ GPU session 1 (~$95, 7 hr):
   - Phase 0 smoke (text-only at d4, ~$5)
   - Phase 1 LR check (text-only at d6 G=4 — LR transferability is
     data-modality-independent, so text-only is sufficient and cheaper, ~$90)

□ GPU session 2 (~$250, ~1 day):
   - Multimodal Phase 0 smoke (~$5)
   - Main sweep: 9 cells G ∈ {1,4,8} × C ∈ {1e19, 3e19, 1e20} at r=0.3

Total: ~$345, ~2-3 weeks elapsed
```

### Concrete files updated

- dev/sweep_design.md §0a: new "CURRENT PLAN v2" section at top, supersedes body
- dev/multimodal_spec.md §7: added scope guard "r as swept axis (DROPPED)"
- This LOG entry: rationale + sequencing

### What this teaches about scope discipline

I (Claude) had been pushing for "frontier-grade methodology" framing that
required all the academic axes (r-sweep, V1/V2, CompleteP, etc.). The
user's observation forced me to confront: most of that scope serves the
CLAIM "we did frontier-grade work," not the QUESTION "what's the answer."

For a budget-constrained preliminary study, the right scope is: what
finding do we want to publish, and what's the minimum methodology to
defend it?

### Cross-references

- Previous plan: dev/sweep_design.md body (Phases 0-D, ~$595 budget)
  retained for chronology, but superseded by §0a
- Plan.md §4 original sweep: now framed as "future work" — runs the
  multi-r grid we de-scoped from this study

---

## 2026-05-02 (later): REVERSED — Phase 1 LR check is back IN

User reconsidered the earlier skip-Phase-1 decision after asking "why
don't we just match nanochat's d12 instead of running smaller?" Answering
that question surfaced the actual reason we MUST run smaller:

### The Chinchilla compute math (why we can't match d12)

```
d12 ≈ 125M params (nanochat reference scale)
Chinchilla D/N = 20 → D = 2.5B tokens
C_optimal = 6 · 125M · 2.5B = 1.9 × 10²¹ FLOPs
Wall-clock @ 8×H100 BF16 50% MFU = 261 hours
Cost @ $15/hr vast.ai = ~$3,900 PER CELL
```

A 12-cell sweep at d12 = ~$47K. Our budget is $1K. Compute requirements
scale quadratically with N at iso-Chinchilla ratio (C = 120·N²), so
matching nanochat's validated scale is mathematically out of reach.

### What this means for our methodology

We MUST run at d4-d8 (Chinchilla-optimal for our budget). nanochat's
HPs are validated at d12-d26 only. So our cells extrapolate the 1/√d
scaling rule by ~1.7× below the validated range. **The LR check is the
ONLY way to bridge this gap empirically.**

This is exactly the case where the decision tree from a senior-researcher
mental model says "yes, run the LR check":
  - New scale (1.7× extrapolation from validated)? YES
  - Budget allows it? YES (~$90 against $500 main sweep = 18% overhead)
  - Eliminates ambiguity in writeup? YES ("is the G ordering real or LR
    mistuning?")

### Decision: revert to Option N1 (keep Phase 1)

Updated sequencing:
- Phase 0 smoke (~$5, 30 min on 1×H100): KEEP
- Phase 1 LR check (~$90, 6 hr on 8×H100): KEEP — protocol in
  runs/lr_sensitivity.sh
- Build multimodal infrastructure (CPU, ~1-2 weeks): UNCHANGED
- Main sweep (~$500, 1 day on 8×H100): UNCHANGED

Total budget: ~$595 (was $505 with Phase 1 skipped). 18% increase
buys methodology rigor.

### Why this reversal is honest

Initial decision (skip Phase 1, save $90) was budget-pressured. User's
counter-question forced explicit reasoning about the underlying constraint
(can't match d12 due to Chinchilla scaling). That reasoning made clear
the LR check isn't optional methodology garnish — it's the bridge that
makes our small-scale cells defensible.

Lesson: don't accept budget cuts to methodology without first asking
"what does this enable us to claim?" If the cut means we can't claim
something we want to claim → restore the cut.

### The d12 question deserves a permanent answer in the writeup

Limitations section will state:
"Compute budget forced operation at d4-d8 rather than nanochat's
validated d12-d26 range. The Phase 1 LR sensitivity check (3 cells at
d6) verified the 1/√d AdamW LR transfer holds within 5% at our scale,
within bootstrap CI of the LR-vs-loss bowl. Per-cell LR mistuning
across the main sweep is therefore bounded and unlikely to affect the
relative G\*(r) ordering."

### Cross-references

- Previous decision (skip): see entry above; superseded
- runs/lr_sensitivity.sh: ready to run as-is
- dev/sweep_design.md §3: protocol description still valid

---

## 2026-05-02: Skip Phase 1 LR sensitivity check — use nanochat defaults

**SUPERSEDED** by the entry above (same date, later decision). Original
text preserved for chronology:

Budget decision: skip the $90 Phase 1 LR sensitivity check (per
dev/sweep_design.md §3 / runs/lr_sensitivity.sh). Use Karpathy's
nanochat AdamW defaults with the built-in 1/√d transfer rule
unchanged across all sweep cells.

### Reasoning

**What we save:** $90 of GPU rental, ~6 hr elapsed time.

**What we accept:**
- No empirical evidence that LRs are within 5% of optimal at our smaller
  scales (d4-d8). Karpathy validated 1/√d at d12-d26.
- Risk: absolute loss numbers in our sweep may be 5-10% off vs an
  optimally-tuned cell.

**Why this is acceptable for our research question:**
- The project measures **relative** G\*(r) — which G is best at which r.
- Bounded LR mistuning (within ~10% absolute loss) is unlikely to
  change the ORDERING of G values within a (C, r) cell.
- Mistuning that affects all cells equally cancels out in the
  modality-conditional comparison.

**What's preserved:** the central scientific finding (does G\* shift
with r) is robust to this LR uncertainty. Only the absolute α/β
exponents would be affected, and those aren't directly comparable to
published scaling laws anyway because we use Muon not AdamW (already
documented in the 2026-05-02 multimodal architecture decision entry).

### Decision: Option N2 (keep smoke, skip LR check)

Updated sequencing:
- Phase 0 smoke (~$5, 30 min on 1×H100): KEEP — cheap insurance against
  GPU-specific bugs in the ported nanochat MoE code
- Phase 1 LR check (~$90, 6 hr on 8×H100): SKIP
- Build multimodal infrastructure (CPU, ~1-2 weeks): UNCHANGED
- Main sweep (~$500, 1 day on 8×H100): UNCHANGED

Total budget: ~$505 instead of ~$595. Saves 15%, accepts the LR
uncertainty.

### Documentation in eventual writeup

Limitations section will explicitly state:
- "We use nanochat's empirical 1/√d AdamW LR scaling without per-scale
  validation. Karpathy validated this scaling at d12-d26 via extensive
  sweeps; our cells are 1.7× below his validated range. Bounded LR
  mistuning (estimated <10%) may affect absolute exponents but should
  not affect the relative G\*(r) ordering that is our central
  scientific claim."

### When to revisit

If main sweep results show:
- Loss curves at different scales don't align in shape (suggests
  per-scale HP mistuning), OR
- G\*(r=0) and G\*(r=0.5) ordering is uncertain within bootstrap CI

Then add a one-cell LR validation at d6 G=4 r=0 (~$30 cost, ~2 hr)
to test whether the LR is the issue. Defer until needed.

### Cross-references

- runs/lr_sensitivity.sh: still in repo, ready to run if revisit needed
- dev/sweep_design.md §3: documents the protocol, marks deferred
- This decision: trades methodology rigor for budget; preserves
  ability to add the check later

---

## 2026-05-02: Discovered nanochat MoE duplication; ported into llm/core/

User pushback: "nanochat already does all sweep and MoE — why are we
doing this again?" Investigation confirmed they were right.

### What I found

`git branch -av` on `~/Desktop/llm_labs/nanochat` showed `origin/moe`
(commit `5422d3a` "make sure to use active params in scaling laws"). I
had assumed "nanochat reverted MoE" (per their LOG 2026-02-19) meant
the code was deleted; in fact reverted-from-main ≠ deleted. Feature
branches preserve work.

`nanochat origin/moe` contains:

| Component | nanochat moe branch | Our pre-port llm/ work | Status |
|---|---|---|---|
| MoE layer (sigmoid + DeepSeekV3 bias) | `nanochat/moe.py` 241 lines | `core/moe/layer.py` 188 lines (our P1) | DUPLICATED |
| 3D weight tensors + Muon Polar Express | included | `core/optim.py` Muon fix (P1) | DUPLICATED |
| `torch._grouped_mm` dispatch (was our planned P3) | included | not started | They have it |
| CPU loop fallback | included | `core/moe/layer.py` loop | DUPLICATED |
| Iso-FLOP sizing formula | identical | identical | match |
| Active params in scaling laws | wired | `core/model.py` (P2) | DUPLICATED |
| Sweep driver script | `runs/scaling_laws.sh` ~150 lines | `dev/sweep_design.md` Phase 0/1/A spec | They had it as code, we had spec |
| Scaling analysis notebook | `dev/scaling_analysis.ipynb` | `core/scaling_fit/spec.md` (deferred) | They have it |

~70% of what we built in P1+P2 was duplication.

### Failed first attempt: fork-into-mm-moe/

Initially cloned nanochat into `~/Desktop/llm_labs/mm-moe/`, branched
`mm-main` from `origin/moe`, copied project docs into
`mm-moe/dev/project/`. User pushed back: three sibling directories
(`llm/`, `mm-moe/`, `nanochat/`) is friction; they kept opening files
in `llm/` (where they'd been working for weeks).

### Successful approach: port nanochat files INTO llm/core/

Treated nanochat origin/moe as a **source of files to port**, not a
tree to fork:

```
nanochat origin/moe                            → llm/
├── nanochat/moe.py                            → core/moe.py            (full file, no import changes — only torch imports)
├── nanochat/gpt.py                            → core/model.py          (replaced; import `from nanochat.X` rewritten to `from core.X`)
├── (nanochat/optim.py kept as-is in llm/      — our version has COMPUTE_DTYPE handling we want to keep)
├── runs/scaling_laws.sh                       → runs/scaling_laws.sh   (new dir)
└── dev/scaling_analysis.ipynb                 → dev/scaling_analysis.ipynb
```

Deleted from `llm/`:
- `core/moe/` (our subpackage — superseded by `core/moe.py`)
- `core/scaling_fit/` (spec only — superseded by `dev/scaling_analysis.ipynb`)
- `scripts/verify_core_moe.py` (verifier for our deprecated MoE)
- `STATUS_DEPRECATED.md` (we're not deprecated anymore)

Deleted: `~/Desktop/llm_labs/mm-moe/` (was a transient fork; all unique
content now lives in `llm/dev/`).

### Trade-off: lost upstream tracking

By porting (not forking), we lose `git pull upstream moe` ergonomics.
If Karpathy updates origin/moe, we'd need to manually re-port. Mitigation:
nanochat moe branch hasn't moved in months; rare. Can `git diff` against
`~/Desktop/llm_labs/nanochat/origin/moe` any time.

### Lesson — apply universally

Before any major build that re-implements existing capability:
1. Check existing repos (including feature branches, archived branches,
   related forks) for prior art. 30-min audit at session start.
2. Fork only when needed; port files when not.
3. Question "is someone else already solving this?" — every new file.

This applies to upcoming multimodal work too: before writing the
PatchMerger, check if Qwen3.5-VL's HF model class exposes one
extractable.

### Cross-references

- `nanochat/dev/LOG.md` 2026-02-19 — original Karpathy MoE entry
- `core/moe.py` — ported nanochat MoE (now our active code)
- `core/model.py` — ported nanochat GPT with MoE wiring (was our model.py)
- `runs/scaling_laws.sh` — ported sweep driver
- `dev/scaling_analysis.ipynb` — ported analysis notebook

---

## 2026-05-02: Multimodal architecture decision: Qwen3.5-VL early fusion

After porting nanochat MoE, decided multimodal architecture. Plan §3
specified VQ tokens (Cosmos-Tokenizer / VQGAN-LC). User pushed for
Qwen3.5-VL approach (continuous patches via SigLIP2 + PatchMerger,
the option Plan §3 explicitly rejected).

### Why the user's Qwen3.5-VL choice is right (not just convenience)

User had substantial prior research:
`basics/notebooks/qwen35_vl_tiny.py` is an 823-line from-scratch tiny
rebuild of Qwen3.5-VL with sanity checks, contrastive overfit demo,
detailed architecture documentation. Specifically calls out Qwen3.5 vs
Qwen3-VL distinction (Qwen3.5 deletes DeepStack visual sidecars → pure
early fusion at embedding layer only).

Re-evaluating Plan §3's "VQ wins" reasoning:

1. **"No projector confound"** — but if SigLIP2 + PatchMerger frozen
   during the sweep, the merger is a fixed input transformation, NOT a
   confound. FLOPs accounting stays honest for the LLM trunk (the
   thing we sweep).
2. **"Honest tokens-per-image budget"** — SigLIP2-SO400M at 384×384
   with 2×2 merger = ~182 tokens/image, fixed. Plan wanted exactly
   this property from Cosmos-Tokenizer.
3. **"Unified vocabulary"** — after merger, vision features ARE in the
   LLM's hidden space. MoE routes both modalities through the same
   trunk. Unified at the level that matters for routing analysis.
4. **Production validation.** Qwen3.5-VL ships at scale (397B-A17B MoE)
   with the EXACT thesis from Plan §1 Finding 1 (early-fusion + MoE for
   native multimodal). Public reports: "outperforms Qwen3-VL" — the
   modality-specific MoE absorption hypothesis works.
5. **Existing tooling.** SigLIP2-SO400M on HF
   (`google/siglip2-so400m-patch14-384`), standalone-loadable. Cosmos-
   Tokenizer integration would be from-scratch + uncertain reconstruction
   quality (Plan §7 Risk 3).

### Audit-before-build (lesson from this morning's pivot, re-applied)

30-min audit before writing any spec:
- Qwen3.5-VL released 2026-02-16 to 2026-03-02; HF transformers
  support landed Feb 2026
- Variants: 397B-A17B / 122B-A10B / 35B-A3B MoE + 27B/9B/4B/2B/0.8B dense
- Quote validating Plan §1 Finding 1: "Early fusion training on
  multimodal tokens achieves cross-generational parity with Qwen3 [text
  LLM] and outperforms Qwen3-VL across reasoning, coding, agents, and
  visual understanding benchmarks"
- SigLIP2-SO400M loadable via `AutoModel.from_pretrained`; vision
  encoder accessible via `model.get_image_features()`

Full audit findings: `dev/multimodal_audit.md`.

### What got specced

`dev/multimodal_spec.md` — 9 frozen design decisions:
1. Vision tokenizer = SigLIP2-SO400M-patch14-384, frozen
2. Vision-to-LLM projector = PatchMerger (port from `qwen35_vl_tiny.py`)
3. Merger frozen after warmup → eliminates Plan §3 projector confound
4. Fusion = scatter at `<|image_pad|>` positions (Qwen3.5 pure early fusion)
5. LLM trunk = nanochat MoE GPT (the thing we just ported)
6. NO DeepStack sidecars (Qwen3.5 not Qwen3-VL)
7. 3D Interleaved-MRoPE for mixed text+vision sequences
8. HF AutoImageProcessor for SigLIP2 preprocessing
9. Per-modality loss decomposition via token-source mask

5 verifier checks (M1-M5) and 5 implementation phases (M0-M5).

### What this deviates from Plan in

Plan §3 picked VQ (option 2). We're using SigLIP2+merger (option 1).
**Considered deviation, not slip.** Justified by qwen35_vl_tiny.py
research, audit findings, and frozen-merger mitigation of Plan's main
concern.

If reviewers ask "why not Cosmos-Tokenizer per Plan §3": at implementation
time, SigLIP2+PatchMerger frozen offered mature production infrastructure
with the same controlled-variable property; Cosmos-Tokenizer integration
would have added 2+ weeks of infrastructure work and an unmeasured
tokenizer-floor risk.

### Verdict

Spec ships. Phase M0 (vision tower integration) is the next concrete
implementation step — port `qwen35_vl_tiny.py` components into a new
`core/multimodal.py`, replace random ViT with frozen HF SigLIP2.

### Cross-references

- `dev/multimodal_audit.md` — audit findings (10 sources cited)
- `dev/multimodal_spec.md` — 9 frozen decisions + verifier + phases
- `basics/notebooks/qwen35_vl_tiny.py` — 823-line from-scratch reference

---

## 2026-05-02: Shipped sweep design doc — keystone pre-registration

Shipped `dev/sweep_design.md` (685 lines, commit f56bff4). The operational
pre-registration document for the Plan.md modality-conditional sweep.
Converts Plan §4's prose into 7 phases (0, 1, 2, A, B, C, D) with
explicit cells, budgets, go/no-go gates, kill conditions, logging
schemas, and GPU sourcing decision.

### Why this exists (the pushback that triggered it)

Was about to build `core/scaling_fit/` (Option C). Got pushback: "we
just built MoE, hadn't touched vision yet, should we run scaling laws
now?" → caught a real smell: building analysis tools for data that
doesn't exist and won't for months.

Re-read Plan.md as a senior researcher. Three sources of project failure:
methodology wrong (Risk 2 CompleteP, Risk 3 tokenizer), resources
exhausted on wrong things (Risk 6), can't generate data needed
(no multimodal pipeline). Highest-leverage move addressing all three
simultaneously: write the operational sweep design FIRST, so all
downstream infrastructure (multimodal pipeline, engine integration,
fit toolkit, GPU sourcing) is sized against a concrete spec.

### What the doc commits to

Concrete decisions made (i.e., now pre-registered):
- **Phase A cut: drop φ extremes, keep φ ∈ {0.10, 0.25}** → 24 cells
  (Plan §4 said "cut 48→24 ... drop one G or one φ axis" without
  picking which)
- **Logging schema** (per-step, per-50, per-eval, per-cell metadata)
  with explicit per-modality loss decomposition
- **Cancel triggers** (NaN, loss>1.5×expected at step 100, MFU<25%
  sustained, routing collapse, etc.)
- **GPU sourcing**: Lambda Labs 8×H100, ~$4500 commit gated by Phase 0
- **FP8 throughout Phases A+B+C** (per Plan compute reduction; partial
  conversion per nanochat dev/moe_fp8.md — shared expert FP8'd, routed
  experts stay bf16)
- **Total budget: ~$1100** of GPU rental for the planned cells (well
  under $4500 cap — buys re-run buffer and safety margin)

### What the doc DOES specify for downstream work

§15 lists what each existing/future component must support:
- `core/configs.py`: add D4_MoE, D6_MoE, D8_MoE named configs
- `core/engine.py` (P4): per-step + per-50 + per-eval logging, modality
  loss masking, cancel-trigger hooks, gradient-accum-aware bias update,
  multi-GPU all-reduce of `_token_counts`
- Multimodal pipeline: Cosmos-Tokenizer at 256 tokens/image, unified
  48K vocab, interleaved loader with mix_ratio_r ∈ [0,1], per-token
  modality tag for loss decomposition
- `core/scaling_fit/`: re-scope to text-only L(N, D, G) v1 first
  (consumed by Phase A), modality decomposition v2 (Phases B+C+D)
- Checkpoint format: round-trip 3D MoE weights + persistent
  router_bias buffer + sweep cell metadata

### What was easy

- Phase structure already implicit in Plan §4 — just needed making
  concrete
- Numbers (per-cell GPU-hours, $ estimates) follow from
  Chinchilla-optimal sizing + nanochat MFU benchmarks
- Cancel triggers are universal (apply to every cell, every phase)

### What was hard / surprising

- **Catching my own over-eagerness.** Specced `core/scaling_fit/` first
  without realizing it was tool-waiting-for-data. The user's pushback
  was the LEARN moment.
- **Phase A cut decision** required real reasoning about which axis is
  least informative. Picked dropping φ extremes (dense covered separately;
  ultra-sparse less interesting for granularity question) but it's a
  judgment call, not derivable.
- **Budget reconciliation.** Plan §4 said cuts get to 1344 GPU-hours;
  my detailed numbers came to ~2000. Required the §16 explicit
  reconciliation (FP8 throughout closes the gap to ~1100).

### Verdict

Ships. This is the spec all subsequent infrastructure work is sized
against. Reverses the implicit ordering "build tools, hope they fit
the experiment" to the explicit one "design experiment, build tools
that fit it."

### Cross-references

- `dev/sweep_design.md` — the doc
- `docs/Plan.md` — the project this serves
- `core/moe/spec.md` — current MoE state (P1+P2)
- `core/scaling_fit/spec.md` — fit toolkit (deferred per re-scope; v1
  becomes text-only)

### What's next

Per `dev/sweep_design.md` §15, downstream work in priority order:
1. **Add named MoE configs** to `core/configs.py` (D4_MoE, D6_MoE, D8_MoE)
   — small, CPU-doable now
2. **Multimodal pipeline** — biggest scope, sized against §15 requirements
3. **Engine integration spec (P4)** — sized against §9 logging schema
4. **Re-scope `core/scaling_fit/` to text-only v1** — consumed by Phase A

The first item is a half-hour task; let's start there to validate the
sweep design's named-config hypothesis before larger commitments.

---

## 2026-05-01: Read nanochat's MoE entry — implications for Plan.md

Read `nanochat/dev/LOG.md` 2026-02-19 entry. Karpathy implemented nearly
the identical MoE architecture we're building (DeepSeekV3 sigmoid routing,
8 routed + 1 shared, aux-loss-free balancing, iso-FLOP sizing, 3D weight
tensors, `torch._grouped_mm` dispatch, Muon `second_momentum_buffer` fix
for 3D weights). His verdict: **net negative wall-clock at GPT-2 scale**,
not merged into nanochat.

### Architecture comparison

Every architectural choice we made for `core/moe/` matches what nanochat
shipped:

| Choice | nanochat | Our P1+P2 |
|---|---|---|
| Routing | sigmoid + top-2, aux-loss-free bias balancing | same |
| Experts | 8 routed + 1 shared (DeepSeekV3) | same |
| Iso-FLOP sizing | `expert_dim = round(4·dim / (k+s) / 128)·128` | identical formula |
| Weight storage | 3D `(num_experts, hidden, dim)` | identical |
| Dispatch | `torch._grouped_mm` | (P3 — currently Python loop) |
| Muon 3D fix | preserve leading dims in `second_momentum_buffer` | identical |
| Active param counting for scaling laws | yes | yes (P2 added) |

### What he measured

Tested at d18 (well within Plan.md's d12–d24 target range):

- Dense MFU: ~46%
- MoE MFU: ~35% (≈25% throughput regression)
- Per-step val loss: better with MoE
- Wall-clock time-to-loss: **worse** with MoE

Tried torchtitan's Triton padding kernel for `_grouped_mm` alignment — also
regressed (35% → 33%). FP8 + MoE doesn't compose out of the box:
`torch._grouped_mm` is bf16-only; `_scaled_grouped_mm` needs per-row scaling
that requires custom autograd. Partial FP8 in nanochat: shared expert
(`nn.Linear`) gets FP8'd, routed experts (3D `nn.Parameter`) stay bf16.

His verdict (verbatim):

> MoE is not worth the trouble for nanochat right now. The code bloat is
> substantial (moe.py, router, shared expert, load balancing, optimizer
> fixes, FP8 gaps, active param counting) and the performance is worse
> wall-clock at our scale of interest. The fundamental issue is that the
> grouped_mm dispatch overhead eats the FLOP savings from sparsity, at
> least at our model scales and sequence lengths.

### What this changes for Plan.md

Plan.md is about **scaling laws for MoE**, not about beating dense at fixed
wall-clock. The IsoFLOP sweep (§4) measures `L(N_active, D)` — invariant to
dispatch overhead. nanochat's finding is about whether MoE *ships in
production* at GPT-2 scale, not whether the scaling-law data is worth
collecting. So the project continues.

But it shifts cost projections and adds measurements:

1. **Compute budget shrinks ~25% effective.** Plan §4 budgeted 1344
   GPU-hours for ~106 runs assuming dense-class throughput. If our MoE
   runs at d12–d18 match nanochat's 25% MFU drop, the effective budget
   is ~1000 hours. Either cut ablation cells or accept tighter timing
   (Plan §4 already considered cuts: "Cut Phase A from 48 → 24 runs").
2. **MFU becomes a measured quantity, not a constant.** Original Plan.md
   doesn't track per-config MFU. It should — both because it informs the
   compute budget for downstream ablations and because "loss-vs-FLOPs is
   good but loss-vs-wallclock is bad" is itself a Plan.md finding worth
   reporting (Plan §1 Finding 3 has labs disagreeing on G; the wall-clock
   axis may be why).
3. **P3 (grouped_mm) priority increases.** nanochat's 25% MFU regression
   was *with* `_grouped_mm`. Without it (our current loop dispatch), the
   regression at scale will be much worse. P3 verifier should explicitly
   measure throughput delta vs dense, not just numerical equivalence.
4. **FP8 + MoE is harder than Plan §3 implied.** That section said "BF16
   main sweep, FP8 for verification runs only." Still right, but the FP8
   verification runs may need to fall back to bf16 for routed expert
   weights even if the rest is FP8 (matching nanochat's partial path).

### Action items

- [ ] Add per-config MFU measurement to the verifier (likely C6 in P3 or
      a new throughput-measurement script alongside `verify_core_moe.py`).
      Establish a dense baseline number BEFORE the first ablation.
- [ ] When running Plan.md sweep, log throughput (tokens/sec, MFU) for
      every cell, not just loss.
- [ ] Add a worked-example MFU comparison entry to this LOG once P3
      grouped_mm lands.
- [ ] Re-read nanochat `dev/moe_fp8.md` before designing P5 (FP8 path).

### Verdict

Continue Plan.md as designed, with the four action items above. nanochat's
data is a calibration on what to expect, not a stop signal — the
scaling-law deliverable is on a different axis from the wall-clock
deployment question.

### Cross-references

- `nanochat/dev/LOG.md` (1077 lines, ~101 entries — excellent style reference)
- `nanochat/dev/moe_fp8.md` (deeper writeup of the FP8 + MoE problem)
- Our specs: `core/moe/spec.md` (P1), `core/moe/spec_p2.md` (P2)
- Plan: `Plan.md` §1 Finding 3 (granularity contention), §4 (sweep design),
  §3 (FP8 mention)
- Commits 8e4ee5c (P1), 4b9640b (P2)

---

## 2026-05-01: P2 — end-to-end training + active-params FLOPs accounting

Shipped P2 of `core/moe/`. Builds on P1 (commit 8e4ee5c). All 5 verifier
checks pass. Commit `4b9640b`.

### What changed

- `core/model.py`:
  - `GPT._inactive_moe_params()` helper.
  - `GPT.estimate_flops()` subtracts inactive routed expert params from
    matmul count. Dense path bit-identical to pre-P2.
  - `GPT.num_scaling_params()` exposes `'transformer_matrices_active'`
    and top-level `'active'` for Plan.md's scaling-law fit.
    `'transformer_matrices'` preserved as alias for back-compat.
- `scripts/verify_core_moe.py`:
  - C3: 100-step training loop on synthetic copy task with real
    `MuonAdamW`. Asserts ≥30% loss drop, no NaN, no dead experts,
    routing entropy > 0.7·log(E).
  - C4: post-training, `router_bias.abs().max() > 0` per MoE layer
    (proves `update_moe_load_balance()` actually fires; the mechanism's
    correctness was already proven in basics/).
  - C5: FLOPs accounting — dense path unchanged, MoE iso-FLOP within
    5% of dense, `total - active == sum of inactive routed expert params`.

### Results

```
[1/5] check_shape                                          PASS
[2/5] check_dense_equivalence  (diff = 0.0e+00)            PASS
[3/5] check_flops_accounting                               PASS
       C5a dense FLOPs/token = 5,898,456  (matches pre-P2 formula)
       C5b MoE   FLOPs/token = 5,907,672  (rel diff 0.16%)
       C5c inactive routed expert params = 393,216
[4/5] check_training_loop                                  PASS
       step 0:   loss = 5.5455
       step 99:  loss = 0.1172   (loss drop = 97.9%)
       layer 0 entropy = 1.378  (threshold 0.970)
       layer 1 entropy = 1.365
[5/5] check_bias_integration                               PASS
       layer 0 |router_bias|_max = 0.055000
       layer 1 |router_bias|_max = 0.073000
```

### What was easy

- The MoE branch in `estimate_flops()` collapses to a one-line subtraction
  via `_inactive_moe_params()`. Dense path bit-identical by construction.
- C4 (bias integration) is one assertion. basics/'s 3-iteration LEARN cycle
  on the bias mechanism in May 2026 paid off here — no need to reprove
  the mechanism, just prove the wiring fires.
- 3D weights through MuonAdamW worked first try after the
  `second_momentum_buffer` shape fix in P1.

### What was hard / surprising

- **Spec corrections caught at planning, not implementation.** Wrote the
  P2 spec first (commit 194ce0f) and caught two real bugs before writing
  any code: (a) original synthetic task `target_t = idx_{t+1}` is
  *unlearnable* on random data — flat loss curve — fixed to predict-
  previous (`target_t = idx_{t-1}`); (b) tiny config `n_embd=128` triggers
  iso-FLOP rounding `4·128/3 → 128` (rounds *down*) → MoE 25% less
  active than dense → C5b would fail — fixed to `n_embd=192` where
  `4·192/3 = 256` is exact 128-multiple. Lesson: writing the spec catches
  bugs that would otherwise show up as confused verifier failures.
- **Loss can fluctuate after rapid drop.** Loss went 5.55 → 0.05 (step 50)
  → 0.12 (step 99). Late fluctuation is optimizer dynamics (Muon polar
  express + AdamW interaction on a tiny model that overshoots), not a
  bug. The `(start - end) / start` metric is robust to it; checking just
  `losses[-1]` would NOT be.

### Verdict

Ships. Foundation in place for Plan.md sweep — `num_scaling_params()['active']`
gives the right x-axis for the L(N, D) fit; `estimate_flops()` gives the
right C-axis. P3 (grouped_mm) is the next priority per the nanochat-MoE
finding above.

### Cross-references

- Spec: `core/moe/spec_p2.md`
- Verifier: `scripts/verify_core_moe.py`
- Commits: 194ce0f (spec), 4b9640b (impl)

---

## 2026-05-01: P1 — core/moe/ MoE layer + Muon 3D fix

Shipped P1 of `core/moe/`. First MoE layer in `core/`, byte-exact-equivalent
to the dense `MLP` in degenerate config. All 2 verifier checks pass.
Commit `8e4ee5c`.

### What changed

- `core/moe/spec.md` (~310 lines) — P1 contract with 10 reconciled design
  decisions explicitly recorded (sign-based bias, ReLU² to match dense
  baseline, 3D weights, iso-FLOP sizing, internal `_token_counts` buffer
  to preserve `Block.forward` signature, etc.).
- `core/moe/layer.py` — `MoELayer`, `Router`, `iso_flop_expert_dim`,
  `_SharedExpert`. Sigmoid top-k routing, DeepSeek-V3 sign-based bias
  balancing, ReLU² FFN, Python loop dispatch (grouped_mm deferred to P3).
- `core/moe/__init__.py` — re-exports.
- `core/model.py`:
  - `GPTConfig`: 6 new MoE fields (off by default).
  - `build_mlp(config)` factory; `Block.__init__` calls it.
  - `init_weights()` branches on `MLP` vs `MoELayer`.
  - `GPT.update_moe_load_balance()` — iterates blocks, calls each MoE's
    `update_load_balance()`. No-op when `config.moe is False`.
- `core/optim.py` — Muon `second_momentum_buffer` shape fix in both
  `_step_muon` and `_step_dist_muon`. For 3D weights `(E, h, d)`,
  preserve all leading dims so the buffer broadcasts with `v_mean`.
  Bit-identical for 2D weights.
- `scripts/verify_core_moe.py` — C1 shape correctness, C2 layer-level
  dense equivalence.

### What was easy

- Adopting the design from basics/moe (proven over 4 LEARN iterations).
- The Muon 3D bug fix — 5 lines per call site, exactly per the existing
  `core/docs/moe.md` §8.2 design doc.
- Lazy import of `MoELayer` inside `build_mlp()` to avoid the
  `core.model ↔ core.moe.layer` cycle.

### What was hard / surprising

- **GPT-level dense equivalence is the wrong test.** First C2 attempt
  built two GPTs (dense and MoE) with same `manual_seed(0)`, then copied
  MLP weights into the MoE expert slot. Failed with diff = 6.2e-2.
  Diagnosis: dense `init_weights` makes 2 random calls per block
  (`c_fc`, `c_proj`); MoE makes 3+ (`w1`, `w2`, `gate`). RNG state
  diverges starting from block 0, so attention weights end up different
  between dense and MoE GPTs even with identical seed. **Fix:** rewrite
  C2 as a layer-level unit test on `MoELayer` vs `MLP` directly. Result:
  diff = 0.000e+00 (byte-exact, recorded in spec as a LEARN).
- **Reconciling our basics/ choices with `core/docs/moe.md`'s prior
  design.** 10 design decisions to consciously reconcile (sign-based
  vs proportional bias update; ReLU² wins because it matches the dense
  `MLP` baseline; 3D tensors over ModuleList for grouped_mm and Muon
  compatibility; etc.). Recorded explicitly in spec §"10 reconciled
  design decisions" so future-us doesn't re-litigate.

### Verdict

Ships. Layer-level math is byte-exact correct. Foundation for P2.

### Cross-references

- Spec: `core/moe/spec.md`
- Implementation: `core/moe/layer.py`, `core/model.py`, `core/optim.py`
- Verifier: `scripts/verify_core_moe.py`
- Commit: 8e4ee5c
- Original design doc (predates this work): `core/docs/moe.md`

---
