"""
Multimodal extensions for early-fusion vision-language modeling.

Architecture follows the Qwen3.5-VL pure early-fusion design (no DeepStack
sidecars — see basics/notebooks/qwen35_vl_tiny.py docstring for the
distinguishing detail). Vision path:

    image (H, W, 3)
    → AutoImageProcessor (resize, normalize)
    → frozen SigLIP2-SO400M ViT (729 patches @ 384x384, patch=14)
    → PatchMerger (2x2 spatial compress + 2-layer MLP → llm_hidden)
    → ~182 vision tokens per image
    → scatter at <|image_pad|> positions in input embeddings
    → standard LLM trunk (core/model.py GPT with MoE)

Position handling: 3D Interleaved-MRoPE for the LLM trunk so vision tokens
have meaningful (t, h, w) positions while text tokens have linear (t, 0, 0)
positions.

Spec:           dev/multimodal_spec.md
Reference:      basics/notebooks/qwen35_vl_tiny.py (823-line from-scratch tiny rebuild;
                this file's docstrings and signatures derive from it directly)
Audit:          dev/multimodal_audit.md
SigLIP2:        google/siglip2-so400m-patch14-384 (HF AutoModel-loadable;
                vision encoder accessible via model.get_image_features())

All stubs implemented. References to qwen35_vl_tiny.py for source-of-truth
on RoPE math, PatchMerger merge arithmetic, and scatter mechanics.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from core._layers import Linear  # auto-cast Linear (BF16 input × FP32 weight)


# =============================================================================
# Config
# =============================================================================

@dataclass
class MultimodalConfig:
    """Configuration for the multimodal extension. Distinct from GPTConfig
    because the vision side has its own knobs (image resolution, merger
    settings) and the LLM-side multimodal additions (special tokens) are
    additive to GPTConfig.

    Production defaults match Qwen3.5-VL conventions where applicable.
    """
    # Vision tower (SigLIP2-SO400M-patch14-384 from HF)
    siglip_model_id: str = "google/siglip2-so400m-patch14-384"
    image_size: int = 384                   # SigLIP2-SO400M native resolution
    patch_size: int = 14                    # SigLIP2 native patch size
    vision_embed_dim: int = 1152            # SigLIP2-SO400M hidden dim

    # PatchMerger (vision-to-LLM projector)
    spatial_merge_size: int = 2             # 2x2 spatial compression
    freeze_merger: bool = True              # frozen after warmup; eliminates Plan §3 projector confound
    merger_warmup_steps: int = 0            # 0 = freeze immediately (no warmup); >0 = train then freeze

    # VIDEO: add when enabling video path:
    #   temporal_patch_size: int = 2        # frames-per-temporal-patch (Qwen3.5 default)
    #   max_video_frames:    int = 16       # frame-sampling cap per video
    #   video_pad_token_id:  int = -1       # OPTIONAL: separate from image_pad for per-modality loss split

    # Special token IDs (added to the LLM tokenizer; reserve unused IDs in vocab)
    vision_start_token_id: int = -1         # MUST be set to a real vocab id by the caller
    vision_end_token_id: int = -1
    image_pad_token_id: int = -1            # the placeholder text token where vision features get scattered

    # MRoPE
    rope_theta: float = 10000.0


# =============================================================================
# RoPE utilities for multimodal sequences
# =============================================================================
#
# Two flavors of RoPE are used:
#   - 2D RoPE for the ViT (handled internally by SigLIP2 — we don't reimplement)
#   - 3D Interleaved-MRoPE for the LLM trunk on multimodal sequences
#
# The 3D MRoPE differs from 1D RoPE (already in core/model.py) only in
# how position ids are constructed: instead of (B, seq) of integers, it's
# (3, B, seq) where the 3 is for (t, h, w) coordinates. For pure-text
# tokens the (h, w) coords are 0; for vision tokens they encode the patch's
# spatial location in the image.
#
# The actual rotation math is identical to 1D RoPE — only the "position"
# fed in differs.

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim into two halves and swap with a sign flip: (-x_back, x_front).

    This is the standard rotary formulation: cos * x + sin * rotate_half(x).
    Equivalent to multiplying by e^{i theta} when (x_front, x_back) is treated
    as (real, imag) of complex pairs.

    Reference: qwen35_vl_tiny.py:139-144

    Args:
        x: tensor of any shape, with last dim = head_dim (must be even)

    Returns:
        tensor of same shape with the rotation applied to the last dim
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to x.

    Generic helper: works for any tensor shape as long as cos/sin broadcast
    against x. Used by both vision (when needed) and 3D MRoPE in the LLM trunk.

    Args:
        x:   (..., seq, head_dim)
        cos: broadcasts against x's last 2 dims (typically (seq, head_dim) or (B, seq, head_dim))
        sin: same shape as cos

    Returns:
        rotated x of same shape
    """
    return x * cos + _rotate_half(x) * sin


def build_3d_mrope(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin for Qwen3-VL / Qwen3.5 3D Interleaved-MRoPE.

    Qwen2-VL used *chunked* MRoPE: head_dim was partitioned into three contiguous
    segments for (t, h, w). The frequency spectrum was thus biased — t got only
    low-freq slots, w got only high-freq slots. This hurt long-video
    understanding.

    Qwen3-VL / Qwen3.5 switched to *interleaved* MRoPE: t, h, w components are
    interleaved across all frequency bands. Each axis sees both low and high
    freqs.

    For the tiny rebuild (qwen35_vl_tiny.py:221-250) we implement interleaving
    with a simple round-robin: dim_i is assigned to axis (i % 3). Production
    uses mrope_section = [11, 11, 10] style partitioning that sums to head_dim//2.

    Args:
        position_ids: (3, B, seq) — three axes [t, h, w], each already filled in
                        (see build_position_ids_for_mm below)
        head_dim:     int, must be even
        theta:        RoPE base frequency

    Returns:
        cos, sin: each of shape (B, seq, head_dim)
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    # (1) Standard RoPE inverse frequencies, shape (head_dim // 2,)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                                            device=position_ids.device,
                                            dtype=torch.float32) / head_dim))
    # (2) Round-robin axis assignment (t, h, w, t, h, w, ...) per frequency
    num_freqs = inv_freq.shape[0] # [D/2]
    axis_for_freq = torch.arange(num_freqs, device=inv_freq.device) % 3 # [0, 1, 2, 0, 1, 2, ...]
    # (3) Gather position per frequency: (num_freqs, B, seq)
    pos_per_freq = position_ids[axis_for_freq] # [D/2, B, seq]
    # (4) Permute to (B, seq, num_freqs) and multiply by inv_freq (broadcast)
    pos_per_freq = pos_per_freq.permute(1, 2, 0).to(inv_freq.dtype) # [B, seq, D/2]
    angles = pos_per_freq * inv_freq # [B, seq, D/2]
    # (5) Duplicate to full head_dim per the rotate_half convention
    angles_full = torch.cat([angles, angles], dim=-1) # [B, seq, D]
    # (6) Return cos, sin
    return angles_full.cos(), angles_full.sin() # [B, seq, D]


def build_3d_mrope_for_4d_apply(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3D Interleaved-MRoPE producing cos/sin in Karpathy's apply_rotary_emb layout.

    Differs from `build_3d_mrope` in two ways:
    1. Returns shape (B, T, 1, head_dim//2) — the 4D layout that broadcasts
        against (B, T, n_head, head_dim) query/key tensors used by core/model.py
        apply_rotary_emb (which splits the last dim into two halves).
    2. Does NOT duplicate to full head_dim; the half-pair convention used by
        apply_rotary_emb (line 68-74 of core/model.py) only needs head_dim//2
        frequencies — the two halves rotate together with the same cos/sin.

    Round-robin axis assignment matches both `build_3d_mrope` (above) and the
    qwen35_vl_tiny.py reference (line 228-230). Production Qwen3.5 uses
    `mrope_section=[24, 20, 20]` for head_dim=128 — close to round-robin's
    (22, 21, 21) split for 64 freqs. Round-robin is the documented simplification.

    Args:
        position_ids: (3, B, seq) — three axes [t, h, w]
        head_dim:     int, must be even
        theta:        RoPE base frequency (10000 default)

    Returns:
        cos, sin: each of shape (B, seq, 1, head_dim//2), bfloat16
                (matches Karpathy's _precompute_rotary_embeddings layout)
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                                            device=position_ids.device,
                                            dtype=torch.float32) / head_dim))
    num_freqs = inv_freq.shape[0]  # = head_dim // 2
    axis_for_freq = torch.arange(num_freqs, device=inv_freq.device) % 3
    pos_per_freq = position_ids[axis_for_freq]                # (D/2, B, T)
    pos_per_freq = pos_per_freq.permute(1, 2, 0).to(inv_freq.dtype)  # (B, T, D/2)
    angles = pos_per_freq * inv_freq                          # (B, T, D/2)
    cos = angles.cos().to(torch.bfloat16).unsqueeze(2)        # (B, T, 1, D/2)
    sin = angles.sin().to(torch.bfloat16).unsqueeze(2)
    return cos, sin


# =============================================================================
# PatchMerger — vision-to-LLM projector
# =============================================================================
#
# Compresses 2x2 spatial patch groups into one token and projects to LLM
# hidden dim. The key arithmetic: grouped_dim = vision_embed_dim * merge².
# In real Qwen3.5:
#     context_dim   = 1152 (SigLIP2-SO400M)
#     merge_size    = 2
#     grouped_dim   = 1152 * 4 = 4608
#     llm_hidden    = 2048 (35B-A3B) or 4096 (235B)
#     → 2-layer MLP: 4608 → 4608 → llm_hidden
#
# This is THE projector that Plan §3 worried about ("separate optimization
# landscape for the projector"). We mitigate by freezing it after a brief
# warmup (or freezing immediately if pretrained weights are extracted from
# Qwen3.5-VL's released checkpoint — see open question in multimodal_spec.md).

class PatchMerger(nn.Module):
    """2x2 spatial patch group → one token → project to LLM hidden dim.

    Reference: qwen35_vl_tiny.py:381-431

    Forward signature:
        x:        (total_patches, vision_embed_dim) — flattened patches across
                    all images in the batch (concatenated, image-boundaries
                    tracked via grid_thw)
        grid_thw: (num_images, 3) — (T_i, H_i, W_i) in patch units per image

    Returns:
        (total_merged_tokens, llm_hidden_size)
        where total_merged_tokens = sum_i (T_i * H_i * W_i / merge²)

    VIDEO: this module is already video-ready. The (T, H, W, D) reshape inside
    forward iterates over T naturally — image (T=1) is the degenerate case of
    video (T>1). No code change needed when video is enabled.
    """

    def __init__(
        self,
        vision_embed_dim: int,
        llm_hidden_size: int,
        spatial_merge_size: int = 2,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.vision_embed_dim = vision_embed_dim
        self.llm_hidden_size = llm_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.rms_norm_eps = rms_norm_eps
        self.grouped_dim = vision_embed_dim * (spatial_merge_size ** 2)
        self.fc1 = Linear(self.grouped_dim, self.grouped_dim)
        self.fc2 = Linear(self.grouped_dim, llm_hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
            x:        (total_patches, vision_embed_dim)
            grid_thw: (num_images, 3) — patch-unit dimensions per image

            Steps (per qwen35_vl_tiny.py:401-431):
            1. Norm the input: x = self.norm(x)
            2. For each image (T, H, W) in grid_thw:
                a. Slice this image's chunk: chunk = x[offset : offset + T*H*W]
                b. Reshape to (T, H, W, D)
                c. Spatial 2x2 merge: assert H % merge == 0 and W % merge == 0
                d. Reshape to (T, H//merge, merge, W//merge, merge, D)
                e. Permute and reshape to (T*H//merge*W//merge, merge*merge*D)
                f. Append to outs
            3. Concat all chunks along dim 0 → (total_merged, grouped_dim)
            4. Apply MLP: fc2(GELU(fc1(x))) → (total_merged, llm_hidden_size)
        """
        x = F.rms_norm(x, (self.vision_embed_dim,), eps=self.rms_norm_eps)
        merge = self.spatial_merge_size # 2
        outs = []
        offset = 0
        for (T, H, W) in grid_thw.tolist(): # (1, 27, 27)
            n = T * H * W # number of patches in the image
            chunk = x[offset : offset + n].reshape(T, H, W, self.vision_embed_dim) # (1, 27, 27, 1152)
            # Crop to nearest multiple of merge if H or W is odd (e.g. SigLIP2 27x27).
            # Drops the rightmost column / bottom row of patches. Documented behavior.
            H_eff = (H // merge) * merge # 13 * 2 = 26
            W_eff = (W // merge) * merge # 26 * 2 = 52
            if H_eff != H or W_eff != W: # False
                chunk = chunk[:, :H_eff, :W_eff, :].contiguous() # (1, 26, 26, 1152)
            # (T, H', merge, W', merge, D) → permute to bring merge dims adjacent → flatten
            chunk = chunk.reshape(T, H_eff // merge, merge, W_eff // merge, merge, self.vision_embed_dim) # (1, 13, 2, 13, 2, 1152)
            chunk = chunk.permute(0, 1, 3, 2, 4, 5).contiguous() # (1, 13, 13, 2, 2, 1152)
            chunk = chunk.reshape(T * (H_eff // merge) * (W_eff // merge), self.grouped_dim) # (169, 4608)
            outs.append(chunk) # [169, 4608]
            offset += n  # Advance over the FULL n input patches (including any dropped by crop)
        x = torch.cat(outs, dim=0) # (169, 4608)
        return self.fc2(F.gelu(self.fc1(x))) # (169, 1536)


# =============================================================================
# VisionTower — frozen SigLIP2 + PatchMerger
# =============================================================================

class VisionTower(nn.Module):
    """Frozen SigLIP2-SO400M vision encoder + PatchMerger projection.

    SigLIP2 weights are loaded from HF (`google/siglip2-so400m-patch14-384`
    by default) with requires_grad=False. PatchMerger is freeze-able via
    config; default is to freeze immediately.

    Forward:
        pixel_values: (N_images, 3, H, W) — preprocessed by HF AutoImageProcessor
        grid_thw:     (N_images, 3) — patch-unit dimensions per image
                                    (will be (1, H_p, W_p) for static images
                                    where H_p = H // patch_size etc.)

    Returns:
        vision_features: (total_merged_tokens, llm_hidden_size)
                        — the per-token projected features ready to be scattered
                        into the LLM input embedding sequence

    VIDEO: when enabling video, accept pixel_values shape (N_videos, T, 3, H, W).
    Recommended approach: per-frame extraction through frozen SigLIP2 (Option A
    in dev review). Reshape to (N_videos*T, 3, H, W) before SigLIP2 forward,
    then let PatchMerger handle T via grid_thw. Avoids retraining a Conv3d
    patch embed and keeps the vision encoder a fixed input transformation
    (which is what the scaling-law study requires).
    """

    def __init__(
        self,
        llm_hidden_size: int,
        siglip_model_id: str = "google/siglip2-so400m-patch14-384",
        spatial_merge_size: int = 2,
        freeze_merger: bool = True,
        vision_encoder: nn.Module | None = None,
        vision_embed_dim: int | None = None,
    ):
        """If `vision_encoder` is provided, use it directly (mock-friendly for
        unit tests); otherwise lazy-import HF transformers and load
        `siglip_model_id`. `vision_embed_dim` is required when passing a
        `vision_encoder`; otherwise read from the loaded model's config.
        """
        super().__init__()
        self.llm_hidden_size = llm_hidden_size
        self.spatial_merge_size = spatial_merge_size

        if vision_encoder is not None:
            assert vision_embed_dim is not None, "vision_embed_dim required when passing vision_encoder"
            self.siglip = vision_encoder
            self.vision_embed_dim = vision_embed_dim
        else:
            from transformers import AutoModel
            self.siglip = AutoModel.from_pretrained(siglip_model_id)
            cfg = self.siglip.config
            inner = getattr(cfg, "vision_config", cfg)
            self.vision_embed_dim = inner.hidden_size

        self.freeze_siglip()
        self.merger = PatchMerger(self.vision_embed_dim, llm_hidden_size, spatial_merge_size)
        if freeze_merger:
            self.freeze_merger_now()

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Steps:
        1. Run SigLIP2 vision encoder — get per-patch features.
            API call needs verification: probably
                outputs = self.siglip.vision_model(pixel_values=pixel_values)
                patch_features = outputs.last_hidden_state  # (N_imgs, num_patches, vision_embed_dim)
            Alternative if vision_model is exposed differently:
                patch_features = self.siglip.get_image_features(pixel_values=pixel_values)
            — but that one returns POOLED features (1 vector per image), not
            per-patch. We want per-patch. Verify on first import.
        2. Flatten across images: (N_imgs, num_patches, D) → (total_patches, D)
        3. Apply PatchMerger(flat, grid_thw) → (total_merged, llm_hidden)
        4. Return

        VIDEO: branch on pixel_values.ndim. If 5 (N_videos, T, 3, H, W), reshape
        to (N_videos*T, 3, H, W), run SigLIP2 as if it were N_videos*T images,
        then proceed normally. PatchMerger reads T from grid_thw. If 4, current
        image path. ~5 lines added at top of forward.
        """
        if pixel_values.ndim == 5:
            N_v, T, C, H, W = pixel_values.shape
            pixel_values = pixel_values.reshape(N_v * T, C, H, W)

        if hasattr(self.siglip, "vision_model"):
            outputs = self.siglip.vision_model(pixel_values=pixel_values)
            patch_features = outputs.last_hidden_state
        else:
            patch_features = self.siglip(pixel_values)

        if patch_features.dim() == 3:
            N, P, D = patch_features.shape
            patch_features = patch_features.reshape(N * P, D)

        return self.merger(patch_features, grid_thw)

    @torch.no_grad()
    def freeze_siglip(self) -> None:
        """Idempotent. Sets requires_grad=False on all SigLIP2 params and puts
        the module in eval mode. Called from __init__ but exposed for explicit
        re-freezing (e.g., after a load_state_dict that may have unfrozen things).
        """
        for p in self.siglip.parameters():
            p.requires_grad = False
        self.siglip.eval()

    def freeze_merger_now(self) -> None:
        """Called by the training loop after merger_warmup_steps to freeze the
        PatchMerger. After this call, only the LLM trunk params are trainable
        (which is the goal for the scaling-law sweep — the projector becomes
        a fixed input transformation).
        """
        for p in self.merger.parameters():
            p.requires_grad = False


# =============================================================================
# Fusion utilities
# =============================================================================
#
# The Qwen3.5 fusion mechanism: text tokens contain placeholder <|image_pad|>
# tokens at positions where images "live"; vision features replace those
# embeddings entirely. After scatter, the unified sequence flows through the
# standard LLM trunk — the trunk doesn't know which positions are vision vs
# text (modality info is tracked separately for loss decomposition + logging).

def scatter_vision_features(
    inputs_embeds: torch.Tensor,
    vision_features: torch.Tensor,
    image_pad_mask: torch.Tensor,
) -> torch.Tensor:
    """Replace embeddings at image_pad positions with vision features.

    Reference: qwen35_vl_tiny.py:655-664 (the EARLY FUSION step)

    Args:
        inputs_embeds:   (B, S, D) — text embeddings (with zeros or random at pad positions)
        vision_features: (total_merged, D) — concatenated across all images in the batch
        image_pad_mask:  (B, S) bool — True at positions that should be replaced

    Returns:
        (B, S, D) — same shape as inputs_embeds, with vision features at pad positions

    Pre-check: the count of True in image_pad_mask MUST equal vision_features.shape[0]
                or the dataloader is mis-aligned. Assert and fail loudly if not.

    Note on cloning: inputs_embeds may be a view; clone before mutating to
    avoid silent in-place issues with autograd.
    """
    n_pad = int(image_pad_mask.sum().item())
    assert n_pad == vision_features.shape[0], (
        f"image_pad_mask has {n_pad} True positions but vision_features has "
        f"{vision_features.shape[0]} tokens — dataloader misaligned"
    )
    out = inputs_embeds.clone()
    out[image_pad_mask] = vision_features.to(inputs_embeds.dtype)
    return out


def build_position_ids_for_mm(
    input_ids: torch.Tensor,
    image_pad_token_id: int,
    image_grids_merged: list[list[tuple[int, int, int]]],
) -> torch.Tensor:
    """Construct (3, B, seq) position ids for 3D Interleaved-MRoPE.

    For each batch item we walk the input_ids left-to-right. Text tokens
    advance only the t-axis; vision tokens (runs of image_pad_token_id) are
    annotated with their (t, h, w) grid coordinates from image_grids_merged.

    Reference: qwen35_vl_tiny.py — see build_position_ids_for_mm function.
    Production Qwen3-VL has a more elaborate construction supporting video,
    interleaved multi-image, etc. For our v1 (single image per sequence,
    no video) the logic is straightforward.

    VIDEO: image_grids_merged is already typed list[list[(T, H, W)]] — T>1
    is the video case. The walk needs an extra outer loop over t in range(T)
    inside each image/video block; t-axis advances by t, then by T total when
    the run finishes. ~5 lines added vs the image-only walk.

    Args:
        input_ids:           (B, S) — text tokens with image_pad_token_id at vision positions
        image_pad_token_id:  the placeholder token id
        image_grids_merged:  list of length B; each entry is a list of (T, H, W) tuples
                            in MERGED-patch units (after PatchMerger compression),
                            one tuple per image in that batch item

    Returns:
        position_ids: (3, B, S) — t-axis at index 0, h-axis at 1, w-axis at 2
                    For text tokens: (t, 0, 0)
                    For vision tokens: (t, h, w) per the image grid
    """
    B, S = input_ids.shape
    pos = torch.zeros(3, B, S, dtype=torch.long, device=input_ids.device)
    for b in range(B):
        next_t = 0
        img_idx = 0
        s = 0
        ids = input_ids[b]
        while s < S:
            if int(ids[s].item()) == image_pad_token_id:
                # Start of a vision run; consume the next image's (T, H, W) grid
                T, H, W = image_grids_merged[b][img_idx]
                run_len = T * H * W
                # Walk the run: row-major over h, w, with t advancing per H*W block
                idx = 0
                for t in range(T):
                    for h in range(H):
                        for w in range(W):
                            if s + idx >= S:
                                break
                            pos[0, b, s + idx] = next_t + t
                            pos[1, b, s + idx] = h
                            pos[2, b, s + idx] = w
                            idx += 1
                # Advance next_t past the image (T merged time-steps consumed)
                next_t += T
                s += run_len
                img_idx += 1
            else:
                # Text token: (next_t, 0, 0)
                pos[0, b, s] = next_t
                next_t += 1
                s += 1
    return pos


# =============================================================================
# Per-modality logging utilities
# =============================================================================

def per_modality_loss_decomposition(
    logits: torch.Tensor,
    targets: torch.Tensor,
    modality_mask: torch.Tensor,
    ignore_index: int = -1,
) -> dict[str, torch.Tensor]:
    """Compute total loss + per-modality decomposition.

    Plan §5.3 requirement. The total loss equals the standard cross-entropy
    used for backprop; the per-modality losses are for LOGGING ONLY (do not
    backprop through them separately).

    Args:
        logits:        (B, S, vocab) — model output
        targets:       (B, S) — next-token targets (with ignore_index for masked positions)
        modality_mask: (B, S) — 0 = text token, 1 = vision token
        ignore_index:  positions with this target value are excluded (e.g., padding)

    VIDEO: when video is enabled and you want separate loss_image vs loss_video
    logging, extend modality_mask to {0=text, 1=image, 2=video} and add
    loss_image / loss_video / n_image / n_video to the returned dict. Backprop
    still uses the unified loss; the split is logging-only.

    Returns:
        {
            'loss':        scalar — total CE loss for backprop (reduction='mean')
            'loss_text':   scalar — mean CE on text positions only (or 0 if none)
            'loss_vision': scalar — mean CE on vision positions only (or 0 if none)
            'n_text':      int — count of text positions used in loss_text
            'n_vision':    int — count of vision positions used in loss_vision
        }
    """
    B, S, V = logits.shape
    loss_full = F.cross_entropy(
        logits.reshape(B * S, V),
        targets.reshape(B * S),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(B, S)
    valid = targets != ignore_index
    text_mask = (modality_mask == 0) & valid
    vision_mask = (modality_mask == 1) & valid

    zero = torch.zeros((), device=logits.device, dtype=loss_full.dtype)
    loss = loss_full[valid].mean() if valid.any() else zero
    loss_text = loss_full[text_mask].mean() if text_mask.any() else zero
    loss_vision = loss_full[vision_mask].mean() if vision_mask.any() else zero

    return {
        "loss": loss,
        "loss_text": loss_text,
        "loss_vision": loss_vision,
        "n_text": text_mask.sum(),
        "n_vision": vision_mask.sum(),
    }


# =============================================================================
# Verifier hooks (for scripts/verify_multimodal.py)
# =============================================================================
#
# These are not part of the runtime path — they're sanity-check helpers
# that the verifier script uses. Implementing them is part of Phase M0.

def _check_siglip_frozen(vision_tower: VisionTower) -> bool:
    """Returns True iff every SigLIP2 param has requires_grad=False.
    Used by verifier check M1."""
    return all(not p.requires_grad for p in vision_tower.siglip.parameters())


def _check_scatter_idempotent_on_text_only(
    inputs_embeds: torch.Tensor,
    vision_features: torch.Tensor,
    image_pad_mask: torch.Tensor,
) -> bool:
    """For a batch with no image_pad tokens (modality_mask all-zeros),
    scatter should return inputs_embeds unchanged. Used by verifier check M2.
    """
    empty = vision_features[:0]
    all_false = torch.zeros_like(image_pad_mask, dtype=torch.bool)
    out = scatter_vision_features(inputs_embeds, empty, all_false)
    return torch.equal(out, inputs_embeds)
