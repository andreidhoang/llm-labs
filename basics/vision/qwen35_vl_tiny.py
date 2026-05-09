"""
Qwen3.5 Vision-Language: Tiny-Scale From-Scratch Rebuild
=========================================================

Faithful reproduction of the multimodal fusion pipeline of Qwen3.5 at a scale
small enough to overfit on a laptop CPU in <60 seconds.

The model follows Qwen3.5's canonical three-module design — with the critical
distinguishing choice that separates Qwen3.5 from Qwen3-VL:

    * Qwen3-VL:  vision features injected at ViT layers 8, 16, 24 into the
                 LLM via DeepStack residual sidecars. Early fusion + late
                 reinforcement.

    * Qwen3.5:   vision features injected ONLY at the embedding layer.
                 Pure early fusion. This is visible in the HuggingFace source:

                     class Qwen3_5VisionModel(Qwen3VLVisionModel):
                         def __init__(self, config, *inputs, **kwargs):
                             super().__init__(config, *inputs, **kwargs)
                             del self.deepstack_visual_indexes   # <-- evidence
                             del self.deepstack_merger_list      # <-- evidence

The Qwen3.5 hypothesis: with 256-expert MoE + hybrid GatedDeltaNet,
modality-specific processing emerges in the experts without needing explicit
cross-layer visual sidecars. The merger alone is the interface.

Pipeline (data trace):
    raw image (H, W, 3)
      → preprocess + frame-duplicate
      → (1, 3, 2, H, W)                       [T=2 treats image as 2-frame video]
      → Conv3d patch_embed                     [kernel = (2, 14, 14)]
      → (num_patches, vision_embed_dim)
      → ViT blocks with 2D-RoPE, cu_seqlens var-len attention
      → (num_patches, vision_embed_dim)
      → PatchMerger: 2x2 spatial compress + 2-layer MLP
      → (num_patches / 4, llm_hidden_size)
      → scatter into LLM input embeddings at <|image_pad|> positions
      → LLM decoder with 3D Interleaved-MRoPE
      → lm_head → logits

Tiny config (vs. production Qwen3.5-35B-A3B):
    image_size       = 56   (real: 448–4096+ dynamic)
    patch_size       = 14   (real: 14)
    vision_embed_dim = 64   (real: 1152, SigLIP2-SO400M)
    llm_hidden_size  = 128  (real: 2048 for 35B-A3B, 4096 for 235B)
    num_vit_layers   = 2    (real: 27)
    num_llm_layers   = 2    (real: 40 hybrid)

Correctness checks at the bottom:
    1. Shape contract at every interface
    2. Loss at init ≈ ln(vocab_size) (uniform-prediction sanity)
    3. Overfit one batch → loss → 0
    4. Ablation: shuffle vision tokens → loss rises (vision is actually used)
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    # Image / video processing
    image_size: int = 56
    patch_size: int = 14                    # spatial patch (each patch → 1 token before merger)
    temporal_patch_size: int = 2            # image is tiled as 2 identical frames → 1 temporal patch
    in_channels: int = 3
    spatial_merge_size: int = 2             # 2x2 spatial compression in the merger

    # Vision Transformer (SigLIP-2 style, tiny)
    vision_embed_dim: int = 64
    vision_mlp_ratio: float = 4.0
    num_vision_layers: int = 2
    num_vision_heads: int = 4

    # Language model (tiny dense proxy for the real hybrid GatedDeltaNet+MoE)
    vocab_size: int = 256
    llm_hidden_size: int = 128              # must equal vision_embed_dim * spatial_merge_size^2
    llm_intermediate_size: int = 256
    num_llm_layers: int = 2
    num_llm_heads: int = 4
    num_kv_heads: int = 2                    # GQA: 4 q heads / 2 kv heads
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    # Special tokens
    vision_start_token_id: int = 250
    vision_end_token_id: int = 251
    image_pad_token_id: int = 252
    bos_token_id: int = 253
    eos_token_id: int = 254

    @property
    def vision_head_dim(self) -> int:
        return self.vision_embed_dim // self.num_vision_heads

    @property
    def llm_head_dim(self) -> int:
        return self.llm_hidden_size // self.num_llm_heads

    def __post_init__(self):
        # Merger = concat spatial_merge_size² patches (→ grouped_dim) then MLP
        # down to llm_hidden_size. grouped_dim and llm_hidden_size can differ;
        # fc2 is an arbitrary linear projection between them.
        # 2D RoPE requires head_dim % 4 == 0 (half for h, half for w, each in pairs).
        assert self.vision_head_dim % 4 == 0, self.vision_head_dim
        # MRoPE requires head_dim % 2 == 0.
        assert self.llm_head_dim % 2 == 0, self.llm_head_dim


# =============================================================================
# Building blocks
# =============================================================================

class RMSNorm(nn.Module):
    """Root-mean-square layer norm. Standard frontier LLM choice (no bias, no mean-subtraction)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # rsqrt(mean(x^2) + eps) is more numerically stable than 1/sqrt(mean + eps)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim into two halves and swap with a sign flip: (-x_back, x_front).
    This is the standard rotary formulation: cos * x + sin * rotate_half(x)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


# =============================================================================
# 2D RoPE for the Vision Transformer
# =============================================================================
#
# Derivation:
#   - Each patch has a grid position (h, w).
#   - Split head_dim into two halves:
#       * first half (dim 0 .. d/2)   rotated by h-coordinate
#       * second half (dim d/2 .. d)  rotated by w-coordinate
#   - Within each half, we use the standard RoPE frequency series:
#       θ_i = theta^(-2i / (d/2)), i ∈ [0, d/4)
#   - The rotation at position p is multiplication by e^{i p θ} in each
#     (2i, 2i+1) complex pair.
#
# Tensor shape: (seq_len, head_dim) for both cos and sin.
# At attention time: cos/sin are broadcast against q/k of shape (seq, n_heads, head_dim).

def build_2d_rope(grid_h: int, grid_w: int, head_dim: int, theta: float = 10000.0,
                  device=None, dtype=torch.float32):
    """Return cos, sin of shape (grid_h * grid_w, head_dim).

    Layout:
        dim[0 : head_dim//4]           — cos of h * freq_i
        dim[head_dim//4 : head_dim//2] — cos of h * freq_i (repeated, pair-structured)
        ...
    Using the rotate_half convention, we build:
        θ_h = position_h * inv_freq_h  (size head_dim//2 total)
        θ_w = position_w * inv_freq_w  (size head_dim//2 total)
        full = [θ_h || θ_w] → (seq, head_dim//1) but doubled because
        rotate_half pairs (i) with (i + head_dim/2).
    """
    assert head_dim % 4 == 0, "2D RoPE needs head_dim divisible by 4"
    half = head_dim // 2  # budget for h; other half for w

    # inv_freq has half/2 values because we pair dims.
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, 2, device=device, dtype=dtype) / half))

    hs = torch.arange(grid_h, device=device, dtype=dtype)   # (grid_h,)
    ws = torch.arange(grid_w, device=device, dtype=dtype)   # (grid_w,)

    # (grid_h, half/2) and (grid_w, half/2) — each row is position * inv_freq
    h_freqs = torch.outer(hs, inv_freq)
    w_freqs = torch.outer(ws, inv_freq)

    # Tile across the 2D grid: for every (h, w), concatenate [h_freqs[h] | w_freqs[w]]
    # This gives (grid_h, grid_w, half/2 + half/2) = (grid_h, grid_w, half)
    grid = torch.zeros(grid_h, grid_w, half, device=device, dtype=dtype)
    grid[:, :, : half // 2] = h_freqs[:, None, :]
    grid[:, :, half // 2 :] = w_freqs[None, :, :]

    # Flatten to sequence order (row-major over (h, w)): (grid_h * grid_w, half)
    freqs = grid.reshape(-1, half)

    # The rotate_half convention pairs dim i with dim (i + half). So we need
    # cos/sin of shape (seq, 2*half) = (seq, head_dim) where cos[:, :half] == cos[:, half:].
    freqs_full = torch.cat([freqs, freqs], dim=-1)  # (seq, head_dim)
    return freqs_full.cos(), freqs_full.sin()


# =============================================================================
# 3D Interleaved-MRoPE for the LLM
# =============================================================================
#
# Qwen2-VL used *chunked* MRoPE: head_dim was partitioned into three contiguous
# segments for (t, h, w). The frequency spectrum was thus biased — t got only
# low-freq slots, w got only high-freq slots. This hurt long-video understanding.
#
# Qwen3-VL / Qwen3.5 switched to *interleaved* MRoPE: t, h, w components are
# interleaved across all frequency bands. Each axis sees both low and high freqs.
#
# For the tiny rebuild I implement interleaving with a simple round-robin:
# dim_i is assigned to axis (i % 3). This is the essence; production uses
# mrope_section = [11, 11, 10] style partitioning that sums to head_dim//2.

def build_3d_mrope(position_ids: torch.Tensor, head_dim: int, theta: float = 10000.0):
    """
    position_ids: (3, B, seq) — three axes [t, h, w], each already filled in
                  (see build_position_ids_for_mm below).
    Returns cos, sin of shape (B, seq, head_dim).
    """
    assert head_dim % 2 == 0
    # Standard RoPE frequencies (length head_dim // 2)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                                             device=position_ids.device,
                                             dtype=torch.float32) / head_dim))
    # Interleaving: assign each of the head_dim//2 frequencies to an axis in
    # round-robin (t, h, w, t, h, w, ...). This is the "interleaved" spirit.
    num_freqs = inv_freq.shape[0]
    axis_for_freq = torch.arange(num_freqs, device=inv_freq.device) % 3  # (num_freqs,)

    # Gather the right position for each frequency: position_ids[axis_for_freq[i]]
    # position_ids: (3, B, seq); we want (B, seq, num_freqs)
    # For each freq i, we look up position_ids[axis_for_freq[i], :, :] and scale by inv_freq[i]
    B, S = position_ids.shape[1], position_ids.shape[2]
    # (num_freqs, B, S): pick the right axis per freq
    pos_per_freq = position_ids[axis_for_freq]  # (num_freqs, B, S)
    # (B, S, num_freqs)
    pos_per_freq = pos_per_freq.permute(1, 2, 0).to(inv_freq.dtype)
    # Multiply by frequency
    angles = pos_per_freq * inv_freq  # (B, S, num_freqs) broadcast

    # Extend to full head_dim via rotate_half convention: duplicate
    angles_full = torch.cat([angles, angles], dim=-1)  # (B, S, head_dim)
    return angles_full.cos(), angles_full.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., seq, head_dim). cos/sin broadcast against x."""
    return x * cos + _rotate_half(x) * sin


# =============================================================================
# Variable-length attention via cu_seqlens (the ViT's key mechanism)
# =============================================================================
#
# Why: different images have different resolutions → different patch counts.
# Padding to max and masking wastes compute. Instead, concatenate all image
# token sequences along dim 0 and use cumulative sequence lengths:
#
#   cu_seqlens = [0, len_0, len_0 + len_1, ..., total]
#
# Production uses flash_attn_varlen_func. For a transparent rebuild I emulate
# it with an explicit block-diagonal mask. Identical numerics.

def varlen_block_diag_mask(cu_seqlens: torch.Tensor, total_len: int,
                           device=None) -> torch.Tensor:
    """Build (total_len, total_len) mask: True where attention is BLOCKED.
    Tokens only attend within their own segment (one image's patches)."""
    mask = torch.ones(total_len, total_len, device=device, dtype=torch.bool)
    for i in range(len(cu_seqlens) - 1):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        mask[s:e, s:e] = False
    return mask  # True = block


# =============================================================================
# Vision Transformer
# =============================================================================

class VisionPatchEmbed(nn.Module):
    """Conv3d-based patch embedding.

    The Conv3d is the key to unified image/video processing. For an image, the
    preprocessing pipeline duplicates the single frame to make it a 2-frame
    'video'; the temporal_patch_size=2 kernel then collapses that back into 1
    temporal position. For an actual video of T frames, it produces T/2
    temporal patches. Either way, the downstream is a flat token sequence.

    Shape: (N_total_images, 3, 2, H, W) → (N_patches_total, vision_embed_dim)
    where N_patches_total = sum over images of (H_i // p) * (W_i // p).
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv3d(
            cfg.in_channels, cfg.vision_embed_dim,
            kernel_size=(cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size),
            stride=(cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size),
            bias=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (N, 3, T, H, W).  For images, T=2 (duplicated frame).
        x = self.proj(pixel_values)                  # (N, D, T/t_p, H/p, W/p)
        # Collapse spatial + temporal grid into a token sequence per image, then
        # concatenate across the batch (no padding — cu_seqlens handles it).
        x = x.flatten(2).transpose(1, 2)             # (N, seq_i, D)
        x = x.reshape(-1, self.cfg.vision_embed_dim) # (sum seq_i, D)
        return x


class VisionAttention(nn.Module):
    """Bidirectional attention across all patches WITHIN a single image.
    Uses 2D RoPE on q, k. Segmented by cu_seqlens so patches of image A never
    attend to patches of image B."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_vision_heads
        self.head_dim = cfg.vision_head_dim
        self.qkv = nn.Linear(cfg.vision_embed_dim, 3 * cfg.vision_embed_dim, bias=True)
        self.o = nn.Linear(cfg.vision_embed_dim, cfg.vision_embed_dim, bias=True)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        # x: (total_patches, embed_dim); rope: (total_patches, head_dim)
        S, D = x.shape
        qkv = self.qkv(x).reshape(S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(1)                       # each: (S, H, d_h)

        # Apply 2D RoPE on q and k. rope_cos/sin are per-position (S, d_h).
        # Broadcast over heads.
        q = apply_rope(q, rope_cos[:, None, :], rope_sin[:, None, :])
        k = apply_rope(k, rope_cos[:, None, :], rope_sin[:, None, :])

        # Attention: (S, H, d_h) × (S, H, d_h) → (H, S, S)
        # We iterate heads by moving head dim forward.
        q = q.transpose(0, 1)  # (H, S, d_h)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(attn_mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)    # (H, S, d_h)
        out = out.transpose(0, 1).reshape(S, D)
        return self.o(out)


class VisionMLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(cfg.vision_embed_dim * cfg.vision_mlp_ratio)
        self.fc1 = nn.Linear(cfg.vision_embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, cfg.vision_embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class VisionBlock(nn.Module):
    """Pre-norm transformer block. RMSNorm matches Qwen3.5; original SigLIP2 uses LayerNorm."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.vision_embed_dim, cfg.rms_norm_eps)
        self.attn = VisionAttention(cfg)
        self.norm2 = RMSNorm(cfg.vision_embed_dim, cfg.rms_norm_eps)
        self.mlp = VisionMLP(cfg)

    def forward(self, x, rope_cos, rope_sin, attn_mask):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin, attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerger(nn.Module):
    """Compress 2x2 spatial patch groups into one token and project to LLM hidden dim.

    The key arithmetic: grouped_dim = vision_embed_dim * merge^2. In real Qwen3.5:
        context_dim   = 1152 (SigLIP2-SO400M)
        merge_size    = 2
        grouped_dim   = 1152 * 4 = 4608
        llm_hidden    = 2048 (35B) or 4096 (235B)
        → 2-layer MLP: 4608 → 4096 → llm_hidden

    In our tiny rebuild: 64 * 4 = 256 → 128 = llm_hidden.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        grouped_dim = cfg.vision_embed_dim * (cfg.spatial_merge_size ** 2)
        self.norm = RMSNorm(cfg.vision_embed_dim, cfg.rms_norm_eps)
        self.fc1 = nn.Linear(grouped_dim, grouped_dim)
        self.fc2 = nn.Linear(grouped_dim, cfg.llm_hidden_size)

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        x: (total_patches, vision_embed_dim)
        grid_thw: (num_images, 3) — (T_i, H_i, W_i) in patch units for each image.
        Returns: (total_merged_tokens, llm_hidden_size)
            where total_merged_tokens = sum_i (T_i * H_i * W_i / merge^2)
        """
        x = self.norm(x)
        merge = self.cfg.spatial_merge_size
        outs = []
        offset = 0
        for (T, H, W) in grid_thw.tolist():
            n = T * H * W
            # (n, D) → (T, H, W, D)
            chunk = x[offset:offset + n].reshape(T, H, W, self.cfg.vision_embed_dim)
            # Spatial 2x2 merge. H and W must be divisible by merge.
            assert H % merge == 0 and W % merge == 0, (
                f"H={H} W={W} must be divisible by merge_size={merge}"
            )
            chunk = chunk.reshape(T, H // merge, merge, W // merge, merge,
                                  self.cfg.vision_embed_dim)
            # Concatenate the 2x2 patch group along the feature dim:
            # (T, H', W', merge*merge*D)
            chunk = chunk.permute(0, 1, 3, 2, 4, 5).contiguous()
            chunk = chunk.reshape(T * (H // merge) * (W // merge),
                                  merge * merge * self.cfg.vision_embed_dim)
            outs.append(chunk)
            offset += n
        x = torch.cat(outs, dim=0)           # (total_merged, merge^2 * D)
        x = self.fc2(F.gelu(self.fc1(x)))    # (total_merged, llm_hidden)
        return x


class VisionTower(nn.Module):
    """Full vision path: patch embed → N ViT blocks → patch merger."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = VisionPatchEmbed(cfg)
        self.blocks = nn.ModuleList([VisionBlock(cfg) for _ in range(cfg.num_vision_layers)])
        self.final_norm = RMSNorm(cfg.vision_embed_dim, cfg.rms_norm_eps)
        self.merger = PatchMerger(cfg)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (N, 3, T=2, H, W) — already preprocessed.
        grid_thw:     (N, 3) — (T_i, H_i, W_i) in *patch* units:
                      T_i = T_pixel // temporal_patch_size
                      H_i = H_pixel // patch_size
                      W_i = W_pixel // patch_size
        """
        x = self.patch_embed(pixel_values)
        # Build per-image RoPE. Each image has a fresh (0..H, 0..W) grid;
        # different images are separated by cu_seqlens so their positions don't interact.
        rope_cos_parts, rope_sin_parts = [], []
        seq_lens = []
        for (T, H, W) in grid_thw.tolist():
            assert T == 1, "image path: T=1 (single temporal patch)"
            c, s = build_2d_rope(H, W, self.cfg.vision_head_dim, self.cfg.rope_theta,
                                 device=x.device, dtype=x.dtype)
            rope_cos_parts.append(c)
            rope_sin_parts.append(s)
            seq_lens.append(T * H * W)
        rope_cos = torch.cat(rope_cos_parts, dim=0)  # (total_patches, head_dim)
        rope_sin = torch.cat(rope_sin_parts, dim=0)

        cu_seqlens = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
                                  device=x.device)
        total_len = cu_seqlens[-1].item()
        attn_mask = varlen_block_diag_mask(cu_seqlens, total_len, device=x.device)

        for block in self.blocks:
            x = block(x, rope_cos, rope_sin, attn_mask)
        x = self.final_norm(x)
        return self.merger(x, grid_thw)


# =============================================================================
# Language model side (tiny GQA transformer, stand-in for hybrid GDN+MoE)
# =============================================================================
#
# We use a dense GQA transformer here because the point of this rebuild is to
# demonstrate the *fusion* and positional-encoding mechanics. You already have
# Qwen35MoE / GatedDeltaNet implementations in your own codebase; swap them in
# wherever you see `LLMBlock` below.

class LLMAttention(nn.Module):
    """Causal self-attention with GQA and 3D MRoPE on q, k."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_llm_heads
        self.n_kv = cfg.num_kv_heads
        self.head_dim = cfg.llm_head_dim

        self.q = nn.Linear(cfg.llm_hidden_size, cfg.num_llm_heads * self.head_dim, bias=False)
        self.k = nn.Linear(cfg.llm_hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(cfg.llm_hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(cfg.num_llm_heads * self.head_dim, cfg.llm_hidden_size, bias=False)
        # Qwen3-style per-head RMSNorm on q and k. Cheap stability win.
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, x, position_ids):
        # x: (B, S, D). position_ids: (3, B, S) — MRoPE axes.
        B, S, _ = x.shape
        q = self.q(x).reshape(B, S, self.n_heads, self.head_dim)
        k = self.k(x).reshape(B, S, self.n_kv, self.head_dim)
        v = self.v(x).reshape(B, S, self.n_kv, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = build_3d_mrope(position_ids, self.head_dim, self.cfg.rope_theta)
        # cos/sin: (B, S, head_dim) → broadcast over heads
        q = apply_rope(q, cos[:, :, None, :], sin[:, :, None, :])
        k = apply_rope(k, cos[:, :, None, :], sin[:, :, None, :])

        # GQA: repeat k, v so the head count matches q
        repeat = self.n_heads // self.n_kv
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

        # Move to (B, H, S, d_h)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Causal mask
        causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, -1)
        return self.o(out)


class LLMMLP(nn.Module):
    """SwiGLU MLP."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.llm_hidden_size, cfg.llm_intermediate_size, bias=False)
        self.up = nn.Linear(cfg.llm_hidden_size, cfg.llm_intermediate_size, bias=False)
        self.down = nn.Linear(cfg.llm_intermediate_size, cfg.llm_hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LLMBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.llm_hidden_size, cfg.rms_norm_eps)
        self.attn = LLMAttention(cfg)
        self.norm2 = RMSNorm(cfg.llm_hidden_size, cfg.rms_norm_eps)
        self.mlp = LLMMLP(cfg)

    def forward(self, x, position_ids):
        x = x + self.attn(self.norm1(x), position_ids)
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# Position ID construction — the meeting point of text and vision
# =============================================================================
#
# For MRoPE to work, every token needs a (t, h, w) position.
#
# Text tokens:       (t, t, t)         — all three axes get the plain text index
# Image pad tokens:  (0, h, w)         — t=0 (single frame), (h, w) over the merged grid
# Tokens AFTER the image must skip past the image's max position so they
# follow "in time". This is what Qwen2-VL's rope_deltas accounts for.

def build_position_ids_for_mm(input_ids: torch.Tensor,
                              image_pad_token_id: int,
                              image_grids_merged: list[tuple[int, int, int]]
                              ) -> torch.Tensor:
    """
    input_ids: (B, S) — already has <|image_pad|> tokens in place.
    image_grids_merged: list per batch item, one entry per image:
        (T_merged, H_merged, W_merged) after the merger's spatial compression.
        For our tiny case T=1, H=W=2.

    Returns: (3, B, S) — (t_ids, h_ids, w_ids).
    """
    B, S = input_ids.shape
    out = torch.zeros(3, B, S, dtype=torch.long, device=input_ids.device)

    for b in range(B):
        ids = input_ids[b].tolist()
        t = h = w = 0
        pos = 0
        image_iter = iter(image_grids_merged[b] if b < len(image_grids_merged) else [])
        img_grid = None
        img_pos = 0
        img_consumed = 0
        i = 0
        while i < S:
            tok = ids[i]
            if tok == image_pad_token_id:
                if img_grid is None:
                    img_grid = next(image_iter)
                    img_consumed = 0
                T_m, H_m, W_m = img_grid
                img_len = T_m * H_m * W_m
                # Fill image tokens: (t=0..T_m-1 cycling, h, w on grid)
                base_pos = pos
                for k in range(img_len):
                    ti = k // (H_m * W_m)
                    hi = (k // W_m) % H_m
                    wi = k % W_m
                    out[0, b, i + k] = base_pos + ti
                    out[1, b, i + k] = base_pos + hi
                    out[2, b, i + k] = base_pos + wi
                # After the image, text resumes at max(t,h,w) + 1
                pos = base_pos + max(T_m, H_m, W_m)
                i += img_len
                img_grid = None
            else:
                out[0, b, i] = pos
                out[1, b, i] = pos
                out[2, b, i] = pos
                pos += 1
                i += 1
    return out


# =============================================================================
# The full Qwen3.5-VL-tiny model with early fusion
# =============================================================================

class Qwen35VLTiny(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.vision = VisionTower(cfg)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.llm_hidden_size)
        self.blocks = nn.ModuleList([LLMBlock(cfg) for _ in range(cfg.num_llm_layers)])
        self.final_norm = RMSNorm(cfg.llm_hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.llm_hidden_size, cfg.vocab_size, bias=False)

    def forward(self,
                input_ids: torch.Tensor,           # (B, S)
                pixel_values: torch.Tensor | None, # (N_img, 3, T, H, W)
                grid_thw: torch.Tensor | None,     # (N_img, 3), patch units
                image_grids_merged: list[list[tuple[int, int, int]]]
                ) -> torch.Tensor:
        B, S = input_ids.shape

        # 1. Text embedding lookup
        inputs_embeds = self.embed_tokens(input_ids)     # (B, S, D)

        # 2. Vision path → merged vision features
        if pixel_values is not None and pixel_values.numel() > 0:
            vis_feats = self.vision(pixel_values, grid_thw)   # (total_merged, D)

            # 3. EARLY FUSION — the whole point of this file.
            #    Replace embeddings at <|image_pad|> positions with vision features.
            pad_mask = (input_ids == self.cfg.image_pad_token_id)   # (B, S) bool
            n_pad = pad_mask.sum().item()
            assert n_pad == vis_feats.shape[0], (
                f"image_pad count {n_pad} != vision features {vis_feats.shape[0]}"
            )
            # Scatter: flatten (B, S) → S*B, select pad positions, assign.
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[pad_mask] = vis_feats.to(inputs_embeds.dtype)

        # 4. Build MRoPE position IDs
        position_ids = build_position_ids_for_mm(
            input_ids, self.cfg.image_pad_token_id, image_grids_merged
        )

        # 5. LLM forward
        x = inputs_embeds
        for block in self.blocks:
            x = block(x, position_ids)
        x = self.final_norm(x)
        return self.lm_head(x)


# =============================================================================
# Sanity checks + overfit demo
# =============================================================================

def make_contrastive_batch(cfg: Config, seed: int = 0):
    """Build TWO examples with identical text prefixes but different images and
    different targets. Overfitting this pair requires the model to look at the
    image — text alone gives no signal to distinguish example 0 from example 1.

    This is the clean way to validate that vision features actually flow into
    the LLM decision. A single-example test can be solved from text alone.
    """
    torch.manual_seed(seed)
    H_p = cfg.image_size // cfg.patch_size       # 4
    W_p = cfg.image_size // cfg.patch_size       # 4
    T_p = 1
    merge = cfg.spatial_merge_size               # 2
    n_merged_per_img = (T_p * H_p * W_p) // (merge ** 2)   # 4

    # Two visually distinct images. Making them *very* different (strong positive
    # vs. strong negative) gives a clean contrastive signal at tiny scale.
    img_a = torch.ones (1, 3, cfg.temporal_patch_size, cfg.image_size, cfg.image_size) * 1.5
    img_b = torch.ones (1, 3, cfg.temporal_patch_size, cfg.image_size, cfg.image_size) * -1.5
    pixel_values = torch.cat([img_a, img_b], dim=0)          # (2, 3, 2, 56, 56)
    grid_thw = torch.tensor([[T_p, H_p, W_p], [T_p, H_p, W_p]], dtype=torch.long)
    image_grids_merged = [[(T_p, H_p // merge, W_p // merge)],
                          [(T_p, H_p // merge, W_p // merge)]]

    # Identical text context for both examples; DIFFERENT target tokens.
    # The only way to pick the right target is by reading the image.
    prefix = [10, 20, 30]
    target_a = [100]
    target_b = [200]

    def build_ids(target):
        return ([cfg.bos_token_id] + prefix
                + [cfg.vision_start_token_id] + [cfg.image_pad_token_id] * n_merged_per_img
                + [cfg.vision_end_token_id] + target + [cfg.eos_token_id])

    ids_a = build_ids(target_a)
    ids_b = build_ids(target_b)
    input_ids = torch.tensor([ids_a, ids_b], dtype=torch.long)   # (2, S)

    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    # Mask predictions of <|image_pad|> (we shouldn't train the model to output pads)
    pad_mask = (input_ids == cfg.image_pad_token_id)
    shift_pad = torch.zeros_like(pad_mask)
    shift_pad[:, :-1] = pad_mask[:, 1:]
    labels[shift_pad] = -100

    return input_ids, labels, pixel_values, grid_thw, image_grids_merged


def run_sanity_checks():
    cfg = Config()
    torch.manual_seed(42)
    model = Qwen35VLTiny(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[shape] total parameters: {n_params:,}")

    input_ids, labels, pixel_values, grid_thw, image_grids_merged = make_contrastive_batch(cfg)
    print(f"[shape] input_ids                = {tuple(input_ids.shape)}")
    print(f"[shape] pixel_values             = {tuple(pixel_values.shape)}")
    print(f"[shape] grid_thw (patch units)   = {grid_thw.tolist()}")

    # --- Intermediate shape trace ---
    with torch.no_grad():
        vis_feats = model.vision(pixel_values, grid_thw)
        print(f"[shape] vision features          = {tuple(vis_feats.shape)}   "
              f"(expected: n_merged * N_img = 4 * 2 = 8 tokens, dim=llm_hidden=128)")
        logits = model(input_ids, pixel_values, grid_thw, image_grids_merged)
        print(f"[shape] logits                   = {tuple(logits.shape)}")

    # --- Loss at init sanity ---
    with torch.no_grad():
        logits = model(input_ids, pixel_values, grid_thw, image_grids_merged)
        loss_init = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), labels.reshape(-1), ignore_index=-100
        )
    expected_init = math.log(cfg.vocab_size)
    print(f"[init] loss           = {loss_init.item():.3f}   "
          f"expected ≈ ln(vocab) = {expected_init:.3f}")

    # --- Overfit the contrastive pair ---
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print("\n[overfit] stepping 200 iterations on the contrastive pair:")
    for step in range(201):
        logits = model(input_ids, pixel_values, grid_thw, image_grids_merged)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), labels.reshape(-1), ignore_index=-100
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            print(f"    step {step:3d}   loss = {loss.item():.4f}")
    loss_final = loss.item()

    # --- Ablation 1: swap images between the two examples ---
    # If the model is using vision, predicting the same targets for SWAPPED
    # images should give a worse loss (targets no longer match the images).
    swapped_pixels = pixel_values.flip(0).contiguous()
    model.eval()
    with torch.no_grad():
        logits_real = model(input_ids, pixel_values, grid_thw, image_grids_merged)
        loss_real = F.cross_entropy(
            logits_real.reshape(-1, cfg.vocab_size), labels.reshape(-1), ignore_index=-100
        ).item()

        logits_swap = model(input_ids, swapped_pixels, grid_thw, image_grids_merged)
        loss_swap = F.cross_entropy(
            logits_swap.reshape(-1, cfg.vocab_size), labels.reshape(-1), ignore_index=-100
        ).item()

    print(f"\n[ablation] loss (correct image→target pairing) = {loss_real:.4f}")
    print(f"[ablation] loss (images swapped)               = {loss_swap:.4f}")
    print(f"[ablation] Δ (swap − correct) = {loss_swap - loss_real:.4f}   "
          f"must be >> 0 → image content determines the target")

    # --- Ablation 2: argmax of final-position logits ---
    with torch.no_grad():
        # The target is at position S-2 (predicted from position S-3 in a shifted-label setup).
        # Find the position just before the target: it's right after <|vision_end|>.
        vend = (input_ids == cfg.vision_end_token_id).nonzero()[:, 1]   # col indices per row
        target_pred_positions = vend  # logits at vend predict the next token = target
        for b in range(2):
            pos = target_pred_positions[b].item()
            pred_real = logits_real[b, pos].argmax().item()
            pred_swap = logits_swap[b, pos].argmax().item()
            true_target = 100 if b == 0 else 200
            print(f"    example {b}: true={true_target}  "
                  f"pred(real)={pred_real}  pred(swap)={pred_swap}")

    print("\n[summary]")
    print(f"    init_loss  ≈ ln(vocab):        {'OK' if abs(loss_init.item() - expected_init) < 1.0 else 'FAIL'}")
    print(f"    overfit converges:             {'OK' if loss_final < 0.05 else 'FAIL'}")
    print(f"    vision-ablation Δ > 0.5:       {'OK' if (loss_swap - loss_real) > 0.5 else 'FAIL'}")


if __name__ == "__main__":
    run_sanity_checks()
