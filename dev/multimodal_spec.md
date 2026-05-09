# Multimodal extension spec

The single living spec for the multimodal extension to nanochat MoE.
Describes WHAT we build to add vision capability to mm-moe. For audit
findings that grounded these decisions, see `multimodal_audit.md`.

---

## 1. Goal

Add Qwen3.5-style vision capability to nanochat-mm-moe so the modality-
conditional scaling sweep (Plan §4 Phases B/C/D) can run. Specifically:
take an image, pass through frozen SigLIP2 vision tower + (frozen or
warmup-trained) PatchMerger, scatter the resulting features into the
LLM's embedding-layer input at `<|image_pad|>` positions, and let
nanochat's MoE trunk process the unified text+vision sequence.

Architecture is **pure early fusion** (Qwen3.5 style, not Qwen3-VL's
DeepStack). Per Plan §1 Finding 1 — early-fusion + MoE is the new default.

---

## 2. The 9 frozen design decisions

| # | Choice | Decision | Rationale |
|---|---|---|---|
| 1 | Vision tokenizer | **SigLIP2-SO400M-patch14-384, frozen** | Production-scale, mature HF support, matches Qwen3.5-VL choice |
| 2 | Vision-to-LLM projector | **PatchMerger: 2x2 spatial compress + 2-layer MLP → llm_hidden** | Qwen3.5 standard; ports directly from `qwen35_vl_tiny.py` |
| 3 | Merger training | **Frozen after a brief warmup phase** | Eliminates Plan §3's "projector confound" objection; lets the LLM trunk be the only thing scaling |
| 4 | Fusion mechanism | **Scatter at `<|image_pad|>` positions in input embeddings** | Qwen3.5 pure early fusion; one place where vision enters the trunk |
| 5 | LLM trunk | **nanochat MoE GPT (unchanged)** | Reuse the work we already verified; sweep target |
| 6 | DeepStack sidecars | **NO** (Qwen3.5 not Qwen3-VL) | Per Plan §1 Finding 1 + qwen35_vl_tiny.py hypothesis |
| 7 | Position encoding for vision | **3D Interleaved-MRoPE** (already in qwen35_vl_tiny.py) — implemented in `core/multimodal.py:build_3d_mrope_for_4d_apply` and wired into `core/model.py:GPT.forward` via the `image_grids_merged` kwarg | Required for vision tokens to interact with text via attention |
| 8 | Image preprocessing | **HF `AutoImageProcessor`** for SigLIP2 (handles resize, normalize) | Don't reinvent; use SigLIP2's official preprocessing |
| 9 | Per-modality loss decomposition | **Track token-source mask in dataloader; mask CE loss per modality at logging time** | Plan §5.3 requirement; small wrapper around existing nanochat loss |

---

## 2.5 Training methodology — first principles

The 9 decisions above are WHAT we do. This section is WHY, derived from first principles. Implementers should refer here when tempted to deviate.

### 2.5.1 Why ViT is frozen forever (Decision #1)

Three arguments, all from first principles:

1. **Compute economy:** SigLIP2-SO400M = 543M params. Trainable would 2.5× per-step wall-clock (forward+backward + optimizer state). At our $300 budget, that's effectively cutting compute by 2.5×. Not affordable.
2. **Quality preservation:** SigLIP2 was pretrained on 4B+ image-text pairs at 1024-H100-week scale. We have 0.001% of that compute. Re-training degrades it (small-batch noise, insufficient data diversity).
3. **Confound elimination:** Frozen ViT → vision representation space invariant across cells. Any G\*-vs-G' difference attributable to MoE trunk, not "ViT alignment shifted." Scaling-law fits become defendable.

**Implementation:** `requires_grad=False` on every SigLIP2 param + `.eval()` mode + `torch.no_grad()` wrapper around forward. Activation memory = 0; backward = no-op for ViT.

### 2.5.2 Why projector is trainable then frozen at 5% (Decision #3)

PatchMerger MUST be initially trainable — it learns to map 1152-dim ViT space to 1536-dim LLM embedding space, and at step 0 the LLM embedding space is random Gaussian. There's no precomputed alignment.

Frozen-after-warmup eliminates **Plan §3 "projector confound":** if projector keeps adapting throughout training, "MoE capability gain" is conflated with "projector adapted to MoE." Two moving targets = unattributable. Freezing after alignment establishes one moving target (the LLM trunk) → clean attribution.

**Why specifically 5%:** matches nanochat warmup convention; empirically most alignment learning concentrates in the warmup window; projector gradient norms typically drop 10-100× after this.

**Implementation:** at `step == int(0.05 * total_steps)`, set `requires_grad=False` on all merger params. Or equivalently: linear-decay merger LR from peak to 0 over first 5%.

### 2.5.3 Why LLM trunk trains from random init, NOT from pretrained-text checkpoint

Two populations to sample:
- **(A) From scratch:** clean scaling-law / pretraining regime. Direct comparison to Krajewski 2024 / nanochat text-only.
- **(B) Pretrained-text first:** fine-tuning regime. G\* measured here doesn't transfer to pretraining decisions (which is what frontier labs make).

For a SCIENCE measurement, sample the population that generalizes. Pretraining measurements generalize to fine-tuning; fine-tuning measurements don't generalize to pretraining. → **(A).**

(For a SHIPPING model: production teams prefer (B) — faster convergence to capability. We're measuring, not shipping.)

### 2.5.4 Why joint training from step 0, NO staged alignment

Qwen2.5-VL (3-stage), Qwen3-VL (4-stage), Qwen3.5 (1-stage joint) all exist. Qwen team's own paper finds Qwen3.5 wins across reasoning/coding/agent/visual benchmarks.

**First-principles reason:** LLM optimization landscape is shaped by data seen during training. Staged alignment lets the LLM hidden space evolve WITHOUT vision-friendly directions, then asks it to "rewire" later. Joint from step 0 evolves the hidden space WITH multimodal context from t=0 — no rewiring cost.

**Why this is safe (despite seeming chaotic):** frozen ViT bounds the gradient noise. ViT outputs are stable feature vectors from step 0; only the projector + LLM are noisy. The "noise" is a stable distribution over un-aligned-but-consistent vision embeddings, which the LLM learns to interpret like "exotic text tokens."

→ **Frozen ViT is what enables joint-from-step-0 discipline.** Without frozen ViT, joint training might be too noisy and staging might be required.

### 2.5.5 Why loss is computed on text positions ONLY (vision tokens are context)

Three arguments:

1. **Vocabulary mismatch:** LLM unembed projects to 32K text vocab. Vision "tokens" are continuous ViT features, not discrete vocab entries. Putting loss on vision positions would require a separate vision unembedding head (added params, scope creep) or VQ tokenizer (we rejected — Plan §7 Risk 3).
2. **Matches downstream task structure:** all real multimodal tasks output text (captioning, VQA, visual reasoning). Pretraining objective should match downstream usage.
3. **Information theory:** vision tokens enter the loss INDIRECTLY via attention — text tokens at position N attend to vision tokens at positions 2..N-1. Loss surface still rewards "good vision representations" through downstream text prediction quality, just not via direct vision-position prediction.

**`per_modality_loss_decomposition` clarification:** does NOT mean "loss on vision tokens vs loss on text tokens" (vision tokens have no loss). Means **"text-token loss when vision context is in the recent window vs text-only context."** Diagnostic metric for "is the model benefiting from vision," not optimization target.

**Implementation:** mask CE loss with `(input_ids != IMAGE_PAD_TOKEN_ID)`. Vision positions effectively contribute `loss * 0`.

### 2.5.6 Why interleaved batches, NOT blocked

Blocked: every gradient update sees only one regime (text OR multimodal). Optimizer state (Muon momentum, AdamW m/v) accumulates regime-specific direction. When regime flips, optimizer "forgets" → wasted capacity.

Interleaved: every update sees mixed regime → optimizer learns the stable joint direction. More efficient + smoother loss curves + easier cancel-trigger setting.

**r=0.3 means token-level mix (not example-level)** averaged over rolling 100 batches with `|mean(r_actual) - 0.3| < 0.02` (assertion in dataloader).

### 2.5.7 Why we inherit nanochat optimizer/LR EXACTLY (no multimodal-specific retuning)

Frozen ViT + frozen-after-warmup projector → trunk optimizer ONLY sees text-token gradients (vision positions have no loss; ViT/projector gradients are bounded or zero).

→ **From the trunk optimizer's perspective, this looks identical to text-only training with weird input embeddings.**

→ Karpathy's HPs at d24 should transfer with high fidelity. **Phase 0.A anchor cell** in `sweep_design.md` empirically falsifies this assumption ($32 cost; replicates nanochat d24 baseline within 3% tolerance).

This decision (inherit verbatim) is **only safe** because of the upstream design choices (frozen ViT, frozen projector, text-only loss). Each frozen knob is what licenses HP inheritance. Unfreezing any of them would force HP retuning.

### 2.5.8 Summary — design closure

The 7 decisions above form a **closed system** where each enables the next:

```
Frozen ViT  ────────►  Bounds gradient noise
                       │
                       ▼
Frozen projector ────► Eliminates moving-target confound
after warmup           │
                       ▼
Text-only loss ──────► No vision-position gradient flow
                       │
                       ▼
Joint training ──────► Optimizer sees stable text-token-grad
from step 0            distribution from t=0
                       │
                       ▼
Inherit nanochat HPs   ◄──── Empirically falsified by Phase 0.A
                       │
                       ▼
$300 budget            ◄──── Affordable BECAUSE of all above
measurement
```

Removing any single decision breaks the chain. E.g., "let's also train the ViT" would force HP retuning, blow compute budget, and add a confound — three failures cascading from one change.

→ **Senior researcher discipline:** when tempted to add one knob, trace what cascades. If the chain breaks, don't add the knob. If the chain holds, document why.

---

## 3. API surface

### `mm-moe/nanochat/multimodal.py` (new)

```python
class VisionTower(nn.Module):
    """Frozen SigLIP2-SO400M + PatchMerger.

    Forward: (pixel_values, grid_thw) -> (total_merged_tokens, llm_hidden_size)
    SigLIP2 weights frozen. PatchMerger weights frozen after warmup.
    """
    def __init__(self, llm_hidden_size: int, freeze_merger: bool = True): ...
    def forward(self, pixel_values, grid_thw) -> Tensor: ...

class PatchMerger(nn.Module):
    """2x2 spatial compress + 2-layer MLP. Ports from qwen35_vl_tiny.py.

    grouped_dim = vision_embed_dim * spatial_merge_size² (= 1152 * 4 = 4608)
    Layer 1: Linear(grouped_dim, grouped_dim) + GELU
    Layer 2: Linear(grouped_dim, llm_hidden_size)
    """

def scatter_vision_features(
    inputs_embeds: Tensor,           # (B, S, D)
    vision_features: Tensor,         # (total_merged, D)
    image_pad_mask: Tensor,          # (B, S) bool
) -> Tensor:
    """Replace embeddings at image_pad positions with vision features.
    Returns (B, S, D) with vision features in place. Pre-fusion sanity:
    image_pad_mask.sum() must equal vision_features.shape[0]."""

def build_position_ids_for_mm(
    input_ids: Tensor,
    image_pad_token_id: int,
    image_grids_merged: list[list[tuple[int, int, int]]],
) -> Tensor:
    """3D position ids (3, B, seq) for MRoPE.
    Already implemented in qwen35_vl_tiny.py — port directly."""
```

### `mm-moe/nanochat/gpt.py` modifications

Extend `GPT.forward` signature (additive, backwards-compatible):

```python
def forward(
    self,
    idx,                       # (B, S) text tokens (with <|image_pad|> placeholders)
    targets=None,
    pixel_values=None,         # (N_img, 3, T, H, W) preprocessed images
    grid_thw=None,             # (N_img, 3) per-image (T, H, W) in patch units
    image_grids_merged=None,   # for MRoPE position construction
    kv_cache=None,
):
    # 1. Text embedding (existing)
    # 2. NEW: if pixel_values is not None:
    #      vision_features = self.vision_tower(pixel_values, grid_thw)
    #      inputs_embeds = scatter_vision_features(inputs_embeds, vision_features, image_pad_mask)
    # 3. Position IDs: 1D existing OR 3D MRoPE if multimodal
    # 4. Trunk forward (existing — no changes; MoE blocks process unified seq)
    # 5. lm_head + loss (existing; per-modality decomposition done in caller)
```

### `mm-moe/nanochat/dataset.py` extensions

```python
class MultimodalDataset:
    """Yields batches with optional vision data:
        idx:                (B, S) — text tokens with <|image_pad|> placeholders
        targets:            (B, S) — next-token targets, with -1 (mask) for prefix
        pixel_values:       (N_img, 3, T, H, W) or None
        grid_thw:           (N_img, 3) or None
        image_grids_merged: list[list[tuple]] or None
        modality_mask:      (B, S) — 0 = text token, 1 = vision token (for loss decomposition)
    """
```

Data sources to support (Plan §4):
- Pure text: existing nanochat ClimbMix or DCLM-baseline
- Pure vision: LAION-Recap-12M (image+caption pairs, formatted as `<image><caption>`)
- Interleaved: OBELICS (text-image-text-image webdocs)

Mix ratio `r` controls the fraction of tokens from vision-bearing batches
vs pure-text batches.

### Per-modality loss decomposition

In the training loop:
```python
logits, _ = model(idx, ..., pixel_values=pixel_values, ...)
loss_full = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1, reduction='none')
loss_full = loss_full.view(B, S)

# Decompose
text_mask = (modality_mask == 0) & (targets != -1)
vision_mask = (modality_mask == 1) & (targets != -1)

loss_text = loss_full[text_mask].mean() if text_mask.any() else torch.tensor(0.0)
loss_vision = loss_full[vision_mask].mean() if vision_mask.any() else torch.tensor(0.0)
loss = loss_full[targets != -1].mean()  # the actual training loss

# Log: loss, loss_text, loss_vision separately
```

### Per-modality expert utilization (Plan §5.4)

Extend nanochat's existing routing logging:
```python
# In each MoE block:
# - For each token, know its modality (passed through forward as a buffer)
# - Count tokens-per-expert separately by modality
# - Log per-layer specialization score: 1 - H(p_modality_per_expert)
```

Implementation approach: thread modality_mask through `Block.forward` →
`MoE.forward`. The MoE forward already knows which token went to which
expert; just split the count by modality.

---

## 4. Invariants

1. **SigLIP2 weights NEVER trained.** Always loaded with `requires_grad=False`.
2. **PatchMerger frozen after warmup phase.** Warmup ≈ first 5% of training,
   then `requires_grad=False`. Configurable via `--vision-merger-train-frac`.
3. **Vision features always pass through `scatter_vision_features`.** No
   alternative fusion path (no DeepStack, no cross-attention).
4. **`Block.forward` signature unchanged for text-only batches.**
   Multimodal threading is additive — text-only training works exactly
   as nanochat does today.
5. **Per-token modality tag is a buffer**, not a parameter. Doesn't enter
   the optimizer; only used for logging + loss decomposition.
6. **Image processing identical to SigLIP2's official preprocessing.**
   Use `AutoImageProcessor.from_pretrained("google/siglip2-so400m-patch14-384")`
   without modification.
7. **For text-only runs (`r=0`), vision tower is never instantiated.**
   No memory cost when not needed.

---

## 5. Verifier contract (`scripts/verify_multimodal.py`)

Five checks. Mirror the `verify_core_moe.py` structure.

### M1 — Vision tower load + forward
- Load SigLIP2-SO400M from HF.
- Pass dummy 384×384 image (random tensor); verify output shape matches expected.
- Confirm SigLIP2 params have `requires_grad=False`.

### M2 — Scatter correctness
- Build `inputs_embeds (B=1, S=10, D=768)` of zeros.
- Build `vision_features (4, 768)` of ones.
- Build `image_pad_mask (1, 10)` with True at positions [2, 3, 7, 8].
- Run `scatter_vision_features`.
- Assert: positions [2, 3, 7, 8] are now ones; others remain zero.

### M3 — Multimodal forward shape
- Build tiny GPT with `moe=True` + vision tower.
- Forward with 1 image + 16 text tokens (4 of which are `<|image_pad|>`).
- Verify logits shape `(B, S, vocab)`.
- Verify per-modality counts: `text_tokens = 12, vision_tokens = 4`.

### M4 — Vision actually flows (contrastive overfit)
- Per `qwen35_vl_tiny.py`'s sanity check #4 — but with our integrated model.
- Build TWO examples: identical text, different images, different targets.
- Train for 200 steps.
- Assert loss → near-zero (model learned to use vision).
- Sanity: shuffle vision features → loss rises (vision really used).

### M5 — Per-modality loss decomposition fires
- Run a mixed batch (some pure text, some vision-bearing).
- Verify `loss_text` and `loss_vision` are non-zero, finite, and roughly
  in expected ranges (`loss_text` similar to text-only; `loss_vision`
  may be larger).

---

## 6. Sequencing — phases

### Phase M0 — Vision tower integration (CPU-friendly)
- Port `VisionTower`, `PatchMerger`, `scatter_vision_features`,
  `build_position_ids_for_mm` from `qwen35_vl_tiny.py` into
  `mm-moe/nanochat/multimodal.py`.
- Replace random `VisionPatchEmbed` + `VisionBlock` stack with
  HF SigLIP2 inference.
- Verify M1, M2 pass.

### Phase M1 — GPT.forward extension
- Extend `nanochat/gpt.py:GPT.forward` to accept `pixel_values`,
  `grid_thw`, `image_grids_merged`.
- Add scatter step before main trunk.
- Switch position embedding from 1D RoPE to 3D MRoPE when multimodal.
- Verify M3 passes.

### Phase M2 — Per-modality loss + logging
- Extend training step to compute `loss_text` and `loss_vision`.
- Extend MoE routing logging with per-modality utilization.
- Verify M5 passes.

### Phase M3 — Multimodal data pipeline
- Add LAION-Recap-12M loader (vision-only batches).
- Add OBELICS loader (interleaved batches).
- Add mix ratio `r` configurable via CLI.

### Phase M4 — Verifier M4 (contrastive overfit on real model)
- Two-example contrastive batch through the integrated model.
- 200 steps, assert loss drops; vision-shuffle ablation.
- This is the gate that proves end-to-end works.

### Phase M5 — Tokenizer floor measurement (Plan §7 Risk 3)
- Train a small (D4) model on vision-only data for ~100 steps.
- Measure `loss_floor_vision` (the asymptote of vision loss).
- Compare to text loss at same scale. Pass criterion: ratio ∈ [1.0, 2.0].
- This is a GPU phase (~$30 on 1×H100), gated by M0-M4 passing on CPU.

---

## 7. Scope guards (NOT in v1)

- DeepStack sidecars (Qwen3-VL style multi-layer fusion). Per decision #6.
- Continuous patch tokens without merger (would require LLM hidden = vision hidden).
- **r as a swept axis (DROPPED 2026-05-02).** Original Plan.md had r ∈ {0,
  0.25, 0.5, 0.75} as a sweep dimension to measure G\*(r) interaction. We
  reduced scope after the user's correct observation: "every model in 2026
  is multimodal — why sweep text-only or vision-only baselines?" Frontier
  labs (Llama 3.2 Vision, Qwen2.5/3/3.5-VL, Gemini, GPT-4o) ship multimodal
  by default. Production-relevant question: G\* at ONE realistic mix.
  We fix r = 0.3 (Qwen3.5-VL-style) and sweep G × C only. See
  sweep_design.md §0a for revised plan and dev/LOG.md for rationale.
- **Video support (architecturally READY, deliberately UNUSED).** Per Plan §4
  ("audio/video: absolutely not. Three modalities with one researcher and 8
  GPUs is delusional.") The Qwen3.5-VL architecture we follow handles video
  natively (Conv3d patch embed, 3D MRoPE with t-axis, grid_thw with T
  dimension). Our `core/multimodal.py` template inherits this — no rewrite
  needed to add video later. But for THIS sweep we don't use it because:
  one 10s 1fps video ≈ 5,000 vision tokens vs ~182 for one image (30×
  more compute per cell), and a 3rd modality axis (r_video) would
  multiply our (G × r) sweep grid by 4-8× more cells. Future-work
  paper: extend r ∈ [0, 1] (text vs vision) to (r_image, r_video) and
  measure G\* across the three-modality plane. Doable in ~$2K + 2-3
  weeks CPU work; not in this study.
- Trainable vision tower (frozen per decision #1).
- Captioning-specific objectives or masking schemes; standard next-token prediction only.
- Image generation (multimodal output). Plan is for understanding only at this stage.
- Multiple images per sequence in v1 — start with one image per sequence; multi-image as v2 if needed.

---

## 8. Cross-references

- `multimodal_audit.md` — audit findings that grounded the 9 design decisions
- `qwen35_vl_tiny.py` (in deprecated `llm/basics/notebooks/`) — 823-line
  working from-scratch reference; ports cleanly to v1
- `Plan.md` §3 (architecture choices), §4 (sweep design), §5.3
  (modality decomposition), §5.4 (specialization), §7 Risk 3 (tokenizer floor)
- `sweep_design.md` Phases B/C/D — what consumes this multimodal capability
- `dev/LOG.md` 2026-05-02 PIVOT entry — why we're forking nanochat
- `core/moe/spec.md` (in deprecated `llm/`) — original MoE design rationale
  (now superseded by nanochat's MoE)

---

## 9. Open questions deferred to implementation

- **HF Qwen3_5_VL class structure**: separate from Qwen3_VL or shared? Resolve
  on first import. If clean separation exists, may be able to reuse their
  `VisionTower` directly (less work). Otherwise, port from qwen35_vl_tiny.py.
- **PatchMerger weight init**: random vs Qwen3.5-VL pretrained weights?
  If we can extract Qwen3.5-VL's PatchMerger weights, warmup phase
  becomes optional → speedup.
- **MRoPE vs 1D RoPE for text-only**: nanochat uses 1D RoPE. Switching to
  3D MRoPE may affect text-only performance. Investigate: does MRoPE
  reduce to 1D RoPE when vision tokens are absent?
- **Multi-image sequences**: Plan §3 sequence length 4096 fits "~3 images
  at 1024 vision tokens each + some text" — need to confirm our
  ~182-tokens-per-image (SigLIP2 + 2x2 merger) lets us fit more images.
- **Image preprocessing thread safety**: HF AutoImageProcessor in
  multi-worker dataloader may have issues; check.

---

## 10. What "current state" means

This spec is v1. Phase M0 implementation hasn't started. As phases ship,
this doc updates to reflect implementation reality. Phase 6 LEARN
findings (per `dev/LOG.md` workflow) live there, not here.
