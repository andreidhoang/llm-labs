# Multimodal audit (2026-05-02)

Findings from the audit-before-build pass on Qwen3.5-VL and the vision
tower components. Done before writing `multimodal_spec.md` so the spec
is grounded in actual availability, not assumptions.

---

## 1. Qwen3.5-VL availability

**Released 2026-02-16 to 2026-03-02** (about 2.5 months ago at audit time).

**Family:** vision-language models with both dense AND MoE variants —
perfectly aligned with our project's scaling-law focus on G/φ/r.

| Variant | Type | Notes |
|---|---|---|
| Qwen3.5-VL-397B-A17B | MoE (largest) | release Feb 16 |
| Qwen3.5-VL-122B-A10B | MoE | release Feb 24 |
| Qwen3.5-VL-35B-A3B | MoE | release Feb 24, this is the "small MoE" that our small-scale ablations would map to |
| Qwen3.5-VL-27B | dense | release Feb 24 |
| Qwen3.5-VL-9B / 4B / 2B / 0.8B | dense | release Mar 2 |

**HF Transformers support** landed Feb 2026. Documentation at HF
`model_doc/qwen3_5` (text-only) and `qwen3_vl` (the VL family — Qwen3-VL
and likely Qwen3.5-VL share machinery, distinguished by config; needs
verification at code level).

**Quote that confirms the early-fusion hypothesis:**
> "Early fusion training on multimodal tokens achieves cross-generational
> parity with Qwen3 [text LLM] and outperforms Qwen3-VL models across
> reasoning, coding, agents, and visual understanding benchmarks."

This is the production validation of `basics/notebooks/qwen35_vl_tiny.py`'s
core hypothesis (delete DeepStack sidecars, rely on MoE for modality
specialization). Qwen team's own results say it works.

---

## 2. SigLIP2-SO400M (the vision tower)

**Model id:** `google/siglip2-so400m-patch14-384`

**Architecture:**
- ~400M parameters
- 384×384 image input
- 14×14 patches → 27×27 = 729 patches per image
- After 2×2 PatchMerger: ~182 vision tokens per image
- (Compare to Plan §3 Cosmos-Tokenizer default: 256 tokens/image — same order)

**Loading:**
```python
from transformers import AutoModel, AutoProcessor
ckpt = "google/siglip2-so400m-patch14-384"
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)
```

**Inference path:**
```python
inputs = processor(images=[image], return_tensors="pt").to(model.device)
with torch.no_grad():
    image_embeddings = model.get_image_features(**inputs)
```

**Vision encoder is standalone-accessible** — exactly what we need. Freeze
it, treat as fixed feature extractor, no training during the scaling-law
sweep.

Trained on WebLI dataset using ~2048 TPU-v5e chips. Production-grade,
well-tested.

---

## 3. DeepStack vs Pure Early-Fusion (architectural distinction)

Per the audit search:

**Qwen3-VL DeepStack approach:**
> "DeepStack ... fuses features from multiple ViT layers rather than using
> only the final layer output. ... hidden states are extracted at specific
> intermediate transformer layers. Each extracted hidden state is passed
> through a dedicated PatchMerger (with its own learned weights),
> producing a feature tensor of shape (num_merged_tokens, hidden_size).
> These features are then injected into the first N text decoder layers
> during forward pass."

**Qwen3.5 (purer early-fusion) approach** — per `basics/notebooks/qwen35_vl_tiny.py`'s
documented analysis:
> Qwen3.5 deletes the deepstack pieces:
> ```python
> class Qwen3_5VisionModel(Qwen3VLVisionModel):
>     def __init__(self, config, *inputs, **kwargs):
>         super().__init__(config, *inputs, **kwargs)
>         del self.deepstack_visual_indexes
>         del self.deepstack_merger_list
> ```
> Vision features injected ONLY at the embedding layer.
> Hypothesis: with 256-expert MoE + hybrid GatedDeltaNet, modality-specific
> processing emerges in the experts without needing explicit cross-layer
> visual sidecars.

**For our project: pure early-fusion (Qwen3.5 style)** is the right
choice — matches Plan §1 Finding 1 and gives the cleanest scaling-law
substrate (no extra scaling axis from DeepStack sidecars).

---

## 4. Integration paths considered

**Option A: Use HF `Qwen3_5_VL` directly, replace its LLM with our nanochat MoE**
- Pro: get production vision pipeline for free
- Con: Qwen3.5-VL has a particular LLM (hybrid GatedDeltaNet); replacing it = significant rework on the HF model class side
- Status: DEFERRED unless audit shows clean separation

**Option B: Use Qwen3.5-VL's vision tower (SigLIP2 + PatchMerger) frozen, plug into nanochat**
- Pro: minimal scope; leverage HF SigLIP2 + custom PatchMerger
- Pro: matches our existing `qwen35_vl_tiny.py` code (which already implements PatchMerger)
- Con: we own the integration glue
- Status: **CHOSEN** — see `multimodal_spec.md`

**Option C: Don't use Qwen3.5-VL machinery; build from `qwen35_vl_tiny.py` + frozen SigLIP2**
- Pro: full control, fewer dependencies
- Con: we'd have to write the PatchMerger weights initialization or train it from scratch
- Status: Backup if Option B has compat issues

---

## 5. NVIDIA Megatron-Bridge supports Qwen3.5-VL

Found `https://docs.nvidia.com/nemo/megatron-bridge/latest/models/vlm/qwen35-vl.html`
in the search. NVIDIA's Megatron framework has explicit Qwen3.5-VL
support. Useful as a reference for production-scale training conventions
(though we're not switching frameworks — we're using nanochat).

---

## 6. Open questions deferred to implementation

- Does HF transformers expose `Qwen3_5_VLForConditionalGeneration` as a
  separate class, or is it a config flag on `Qwen3VLForConditionalGeneration`?
  → Resolve when we first import; affects how we extract vision tower
- What's the exact output shape of `model.get_image_features()`?
  Documentation says "embeddings" — need to verify it's `(N, hidden)` of
  `(N, 729, hidden)` (per-patch features) so we can run our own merger
- Does SigLIP2 require image preprocessing (mean/std normalization,
  resize) that we need to match? → `AutoImageProcessor` handles this
- What's the recommended freeze pattern for production fine-tuning?
  → check Qwen3.5-VL training scripts on QwenLM GitHub

---

## 7. Sources

- [Qwen3-VL HF docs](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [Qwen3.5 HF docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen/Qwen3.5-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [SigLIP2-SO400M model card](https://huggingface.co/google/siglip2-so400m-patch14-384)
- [Qwen3-VL DeepStack writeup](https://thesalt.substack.com/p/qwen3-vl-deepstack-fusion-interleaved)
- [Qwen3-VL DeepWiki architecture](https://deepwiki.com/QwenLM/Qwen3-VL/4.2-model-architecture)
- [HF Transformers v4.57.0 release notes](https://github.com/huggingface/transformers/releases/tag/v4.57.0)
- [QwenLM/Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [NVIDIA Megatron-Bridge Qwen3.5-VL docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/vlm/qwen35-vl.html)
- [`basics/notebooks/qwen35_vl_tiny.py`](../../../llm/basics/notebooks/qwen35_vl_tiny.py) — pre-pivot from-scratch reference (823 lines, working)
