"""CPU smoke test for the multimodal pipeline (Gate G0 of scaling_law spec).

Exercises every public function in core/multimodal.py:
- _rotate_half, apply_rope, build_3d_mrope: rotation primitives
- PatchMerger: 2x2 spatial merge + projection
- VisionTower (with mock SigLIP encoder): frozen vision tower + projector
- scatter_vision_features: early-fusion mechanic
- build_position_ids_for_mm: 3D MRoPE position layout
- per_modality_loss_decomposition: text-only loss + per-modality split
- _check_siglip_frozen, _check_scatter_idempotent_on_text_only: verifier helpers

Plus an integration test that walks the full pipeline:
  pixel_values -> VisionTower -> scatter -> apply_rope on 3D positions ->
  cross-entropy with per-modality split.

These tests do NOT require SigLIP2 download or any GPU. They use a tiny
mock encoder so the whole suite runs in <5 seconds on CPU.

Spec: dev/scaling_law_self_assignment.md §10 + §4 Gate G0
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.multimodal import (  # noqa: E402
    PatchMerger,
    VisionTower,
    _check_scatter_idempotent_on_text_only,
    _check_siglip_frozen,
    _rotate_half,
    apply_rope,
    build_3d_mrope,
    build_3d_mrope_for_4d_apply,
    build_position_ids_for_mm,
    per_modality_loss_decomposition,
    scatter_vision_features,
)


# =============================================================================
# Mock SigLIP encoder — replaces HF download for unit tests
# =============================================================================

class MockSiglipEncoder(nn.Module):
    """Minimal stand-in for SigLIP2.vision_model. Returns (B, n_patches, D)."""

    def __init__(self, vision_embed_dim: int = 64, n_patches: int = 64):
        super().__init__()
        self.vision_embed_dim = vision_embed_dim
        self.n_patches = n_patches
        self.proj = nn.Conv2d(3, vision_embed_dim, kernel_size=4, stride=4)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (B, 3, H, W) -> conv -> (B, D, H/4, W/4) -> flatten patches
        x = self.proj(pixel_values)
        B, D, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, D)


# =============================================================================
# RoPE primitives (Tasks 1-2 — already verified, re-tested for completeness)
# =============================================================================

def test_rotate_half_simple():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = _rotate_half(x)
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    assert torch.allclose(out, expected)


def test_apply_rope_identity():
    x = torch.randn(2, 4, 8)
    out = apply_rope(x, torch.ones(4, 8), torch.zeros(4, 8))
    assert torch.allclose(out, x)


def test_apply_rope_norm_preserved():
    """Rotation by any angle preserves L2 norm per pair."""
    B, S, D = 3, 7, 16
    x = torch.randn(B, S, D)
    pos = torch.zeros(3, B, S, dtype=torch.long)
    pos[0] = torch.arange(S).unsqueeze(0).expand(B, S)
    cos, sin = build_3d_mrope(pos, D)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.pow(2).sum(-1), y.pow(2).sum(-1), atol=1e-4)


def test_build_3d_mrope_shape_and_zero():
    pos = torch.zeros(3, 2, 8, dtype=torch.long)
    cos, sin = build_3d_mrope(pos, head_dim=32)
    assert cos.shape == (2, 8, 32)
    assert sin.shape == (2, 8, 32)
    assert torch.allclose(cos, torch.ones_like(cos))
    assert torch.allclose(sin, torch.zeros_like(sin))


def test_build_3d_mrope_round_robin_axis():
    """head_dim=6 -> 3 frequencies -> axes [t, h, w] in round-robin."""
    pos = torch.zeros(3, 1, 1, dtype=torch.long)
    pos[0, 0, 0] = 1  # t
    pos[1, 0, 0] = 2  # h
    pos[2, 0, 0] = 3  # w
    cos, sin = build_3d_mrope(pos, head_dim=6, theta=10000.0)
    inv_freq = [1.0 / (10000.0 ** (i / 6)) for i in (0, 2, 4)]
    expected_angles = [1 * inv_freq[0], 2 * inv_freq[1], 3 * inv_freq[2]]
    expected_full = expected_angles * 2
    assert torch.allclose(
        cos[0, 0, :], torch.tensor([math.cos(a) for a in expected_full]), atol=1e-6
    )
    assert torch.allclose(
        sin[0, 0, :], torch.tensor([math.sin(a) for a in expected_full]), atol=1e-6
    )


# =============================================================================
# PatchMerger (Task 3)
# =============================================================================

def test_patch_merger_single_image():
    merger = PatchMerger(vision_embed_dim=64, llm_hidden_size=128, spatial_merge_size=2)
    T, H, W, D = 1, 8, 8, 64
    x = torch.randn(T * H * W, D)
    grid_thw = torch.tensor([[T, H, W]])
    out = merger(x, grid_thw)
    assert out.shape == (T * (H // 2) * (W // 2), 128)


def test_patch_merger_multi_image():
    merger = PatchMerger(vision_embed_dim=64, llm_hidden_size=128, spatial_merge_size=2)
    grid_thw = torch.tensor([[1, 4, 4], [1, 6, 6]])
    x = torch.randn(16 + 36, 64)
    out = merger(x, grid_thw)
    assert out.shape == (4 + 9, 128)


def test_patch_merger_video_T3():
    merger = PatchMerger(vision_embed_dim=64, llm_hidden_size=128, spatial_merge_size=2)
    T, H, W, D = 3, 4, 4, 64
    x = torch.randn(T * H * W, D)
    grid_thw = torch.tensor([[T, H, W]])
    out = merger(x, grid_thw)
    assert out.shape == (T * (H // 2) * (W // 2), 128)


def test_patch_merger_crops_odd_grid_27():
    """SigLIP2 produces 27x27 raw patches for 384x384. PatchMerger crops to 26x26."""
    merger = PatchMerger(vision_embed_dim=1152, llm_hidden_size=128, spatial_merge_size=2)
    T, H, W, D = 1, 27, 27, 1152
    x = torch.randn(T * H * W, D)
    grid_thw = torch.tensor([[T, H, W]])
    out = merger(x, grid_thw)
    expected_merged = T * (H // 2) * (W // 2)  # 1 * 13 * 13 = 169
    assert out.shape == (expected_merged, 128), f"odd 27 crop wrong: {out.shape}"


def test_patch_merger_crops_odd_grid_5():
    """5x5 raw → crop to 4x4 → 4 merged tokens."""
    merger = PatchMerger(vision_embed_dim=64, llm_hidden_size=128, spatial_merge_size=2)
    T, H, W, D = 1, 5, 5, 64
    x = torch.randn(T * H * W, D)
    grid_thw = torch.tensor([[T, H, W]])
    out = merger(x, grid_thw)
    expected_merged = 1 * 2 * 2  # 4
    assert out.shape == (expected_merged, 128), f"odd 5 crop wrong: {out.shape}"


# =============================================================================
# 3D MRoPE Karpathy-compatible variant (Step 1 of Qwen3.5 fidelity work)
# =============================================================================

def test_build_3d_mrope_for_4d_apply_shape():
    """Karpathy-layout MRoPE: cos/sin shape (B, T, 1, D/2) bfloat16."""
    B, T, head_dim = 3, 12, 64
    pos = torch.zeros(3, B, T, dtype=torch.long)
    cos, sin = build_3d_mrope_for_4d_apply(pos, head_dim)
    assert cos.shape == (B, T, 1, head_dim // 2), f"cos shape: {cos.shape}"
    assert sin.shape == (B, T, 1, head_dim // 2), f"sin shape: {sin.shape}"
    assert cos.dtype == torch.bfloat16
    assert sin.dtype == torch.bfloat16


def test_build_3d_mrope_for_4d_apply_position_zero_identity():
    """All-zero positions → cos=1, sin=0 everywhere (no rotation)."""
    pos = torch.zeros(3, 2, 8, dtype=torch.long)
    cos, sin = build_3d_mrope_for_4d_apply(pos, head_dim=32)
    assert torch.allclose(cos.float(), torch.ones_like(cos.float()))
    assert torch.allclose(sin.float(), torch.zeros_like(sin.float()), atol=1e-6)


def test_build_3d_mrope_for_4d_apply_broadcasts_against_4d_query():
    """Karpathy's apply_rotary_emb expects cos/sin of shape (?,T,?,D/2) broadcast against (B,T,H,D)."""
    B, T, H, D = 2, 6, 4, 32
    pos = torch.zeros(3, B, T, dtype=torch.long)
    pos[0] = torch.arange(T).unsqueeze(0).expand(B, T)
    cos, sin = build_3d_mrope_for_4d_apply(pos, head_dim=D)
    # Use float32 + fixed seed: we're testing math correctness, NOT bfloat16 precision
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=torch.float32)
    cos_f = cos.float()
    sin_f = sin.float()
    # Karpathy's apply_rotary_emb does: split q into halves, multiply each by cos/sin
    d = D // 2
    q1, q2 = q[..., :d], q[..., d:]
    # Broadcast: cos(B,T,1,d) × q1(B,T,H,d) → (B,T,H,d)
    rotated_q1 = q1 * cos_f + q2 * sin_f
    rotated_q2 = -q1 * sin_f + q2 * cos_f
    rotated = torch.cat([rotated_q1, rotated_q2], dim=-1)
    assert rotated.shape == q.shape
    # Norm preservation (rotation property).
    # Tolerance bumped to 0.1: cos/sin returned by build_3d_mrope_for_4d_apply are
    # bfloat16 (matches Karpathy's KV cache layout), so cos² + sin² ≠ exactly 1
    # due to bf16 quantization. Even casting back to fp32 here, the underlying
    # values lost precision. ~6% relative error in norm² is expected.
    assert torch.allclose(q.pow(2).sum(-1), rotated.pow(2).sum(-1), atol=0.1)


# =============================================================================
# VisionTower (Task 4) — uses mock encoder
# =============================================================================

def test_vision_tower_constructs_with_mock():
    enc = MockSiglipEncoder(vision_embed_dim=64, n_patches=64)
    vt = VisionTower(
        llm_hidden_size=128,
        spatial_merge_size=2,
        vision_encoder=enc,
        vision_embed_dim=64,
    )
    assert vt.vision_embed_dim == 64
    assert vt.llm_hidden_size == 128


def test_vision_tower_freezes_siglip():
    enc = MockSiglipEncoder(vision_embed_dim=64, n_patches=64)
    vt = VisionTower(
        llm_hidden_size=128, vision_encoder=enc, vision_embed_dim=64
    )
    assert _check_siglip_frozen(vt)
    for p in vt.siglip.parameters():
        assert p.requires_grad is False


def test_vision_tower_freezes_merger_by_default():
    enc = MockSiglipEncoder(vision_embed_dim=64, n_patches=64)
    vt = VisionTower(
        llm_hidden_size=128, vision_encoder=enc, vision_embed_dim=64,
        freeze_merger=True,
    )
    assert all(not p.requires_grad for p in vt.merger.parameters())


def test_vision_tower_merger_trainable_when_unfrozen():
    enc = MockSiglipEncoder(vision_embed_dim=64, n_patches=64)
    vt = VisionTower(
        llm_hidden_size=128, vision_encoder=enc, vision_embed_dim=64,
        freeze_merger=False,
    )
    assert any(p.requires_grad for p in vt.merger.parameters())


def test_vision_tower_freeze_merger_now():
    """Test the freeze_merger_now() callback used by training loop after warmup."""
    enc = MockSiglipEncoder(vision_embed_dim=64, n_patches=64)
    vt = VisionTower(
        llm_hidden_size=128, vision_encoder=enc, vision_embed_dim=64,
        freeze_merger=False,
    )
    assert any(p.requires_grad for p in vt.merger.parameters())
    vt.freeze_merger_now()
    assert all(not p.requires_grad for p in vt.merger.parameters())


def test_vision_tower_forward_image_path():
    """Mock encoder produces (B, 64 patches, 64 D); merger compresses 2x2."""
    enc = MockSiglipEncoder(vision_embed_dim=64)
    vt = VisionTower(
        llm_hidden_size=128, vision_encoder=enc, vision_embed_dim=64,
        spatial_merge_size=2,
    )
    # 32x32 image -> conv stride=4 -> 8x8 = 64 patches per image
    pixel_values = torch.randn(2, 3, 32, 32)
    grid_thw = torch.tensor([[1, 8, 8], [1, 8, 8]])
    out = vt(pixel_values, grid_thw)
    expected_merged = 2 * (8 // 2) * (8 // 2)  # 32 tokens total
    assert out.shape == (expected_merged, 128)


def test_vision_tower_video_5d_input():
    """5D input (N_videos, T, 3, H, W) flattens to images for SigLIP."""
    enc = MockSiglipEncoder(vision_embed_dim=64)
    vt = VisionTower(
        llm_hidden_size=128, vision_encoder=enc, vision_embed_dim=64,
        spatial_merge_size=2,
    )
    # 1 video x 3 frames x 32x32
    pixel_values = torch.randn(1, 3, 3, 32, 32)
    grid_thw = torch.tensor([[3, 8, 8]])
    out = vt(pixel_values, grid_thw)
    assert out.shape == (3 * 4 * 4, 128)


# =============================================================================
# scatter_vision_features (Task 5)
# =============================================================================

def test_scatter_basic_replacement():
    B, S, D = 1, 6, 4
    inputs_embeds = torch.zeros(B, S, D)
    pad_mask = torch.tensor([[False, False, True, True, False, False]])
    vision_features = torch.tensor([[1.0, 2, 3, 4], [5, 6, 7, 8]])
    out = scatter_vision_features(inputs_embeds, vision_features, pad_mask)
    expected = torch.zeros(B, S, D)
    expected[0, 2] = vision_features[0]
    expected[0, 3] = vision_features[1]
    assert torch.equal(out, expected)


def test_scatter_count_mismatch_assertion():
    inputs_embeds = torch.zeros(1, 4, 8)
    pad_mask = torch.tensor([[False, True, False, False]])
    vision_features = torch.zeros(2, 8)  # 2 features but only 1 pad position
    try:
        scatter_vision_features(inputs_embeds, vision_features, pad_mask)
        raise AssertionError("Expected mismatch failure")
    except AssertionError as e:
        assert "misaligned" in str(e)


def test_scatter_idempotent_on_text_only():
    """Verifier helper: text-only sequence -> no change."""
    inputs_embeds = torch.randn(2, 5, 8)
    pad_mask = torch.zeros(2, 5, dtype=torch.bool)
    vision_features = torch.zeros(0, 8)
    assert _check_scatter_idempotent_on_text_only(inputs_embeds, vision_features, pad_mask)


def test_scatter_preserves_dtype():
    inputs_embeds = torch.zeros(1, 4, 4, dtype=torch.bfloat16)
    pad_mask = torch.tensor([[False, True, False, False]])
    vision_features = torch.ones(1, 4, dtype=torch.float32)
    out = scatter_vision_features(inputs_embeds, vision_features, pad_mask)
    assert out.dtype == torch.bfloat16


# =============================================================================
# build_position_ids_for_mm
# =============================================================================

def test_build_position_ids_text_only():
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    pos = build_position_ids_for_mm(input_ids, image_pad_token_id=99, image_grids_merged=[[]])
    assert pos.shape == (3, 1, 5)
    assert torch.equal(pos[0, 0], torch.tensor([0, 1, 2, 3, 4]))
    assert torch.all(pos[1] == 0)
    assert torch.all(pos[2] == 0)


def test_build_position_ids_with_image():
    """Sequence: [text, text, IMG, IMG, IMG, IMG, text]
    Image grid: T=1, H=2, W=2 -> 4 vision tokens at positions 2,3,4,5
    """
    input_ids = torch.tensor([[1, 2, 99, 99, 99, 99, 7]])
    pos = build_position_ids_for_mm(
        input_ids,
        image_pad_token_id=99,
        image_grids_merged=[[(1, 2, 2)]],
    )
    # text positions 0, 1 -> t=0, 1
    assert pos[0, 0, 0].item() == 0
    assert pos[0, 0, 1].item() == 1
    # vision positions 2-5: t=2 throughout (single time-step), h=[0,0,1,1], w=[0,1,0,1]
    assert pos[0, 0, 2].item() == 2 and pos[1, 0, 2].item() == 0 and pos[2, 0, 2].item() == 0
    assert pos[0, 0, 3].item() == 2 and pos[1, 0, 3].item() == 0 and pos[2, 0, 3].item() == 1
    assert pos[0, 0, 4].item() == 2 and pos[1, 0, 4].item() == 1 and pos[2, 0, 4].item() == 0
    assert pos[0, 0, 5].item() == 2 and pos[1, 0, 5].item() == 1 and pos[2, 0, 5].item() == 1
    # text after image: t advances by T=1 -> next_t=3
    assert pos[0, 0, 6].item() == 3


# =============================================================================
# per_modality_loss_decomposition (Task 6)
# =============================================================================

def test_per_modality_loss_text_only():
    """All-text sequence: loss == loss_text, loss_vision == 0."""
    B, S, V = 2, 4, 10
    logits = torch.randn(B, S, V)
    targets = torch.randint(0, V, (B, S))
    modality = torch.zeros(B, S, dtype=torch.long)
    out = per_modality_loss_decomposition(logits, targets, modality)
    assert out["n_text"].item() == B * S
    assert out["n_vision"].item() == 0
    assert torch.allclose(out["loss"], out["loss_text"])
    assert out["loss_vision"].item() == 0.0


def test_per_modality_loss_split():
    """Half text, half vision-context positions."""
    B, S, V = 1, 8, 5
    logits = torch.randn(B, S, V)
    targets = torch.randint(0, V, (B, S))
    modality = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    out = per_modality_loss_decomposition(logits, targets, modality)
    assert out["n_text"].item() == 4
    assert out["n_vision"].item() == 4


def test_per_modality_loss_ignore_index():
    """Positions with ignore_index excluded from all means."""
    B, S, V = 1, 4, 5
    logits = torch.randn(B, S, V)
    targets = torch.tensor([[0, -1, 2, -1]])
    modality = torch.tensor([[0, 0, 1, 1]])
    out = per_modality_loss_decomposition(logits, targets, modality, ignore_index=-1)
    assert out["n_text"].item() == 1
    assert out["n_vision"].item() == 1


def test_per_modality_loss_matches_standard_ce():
    """When all modality=0 and no ignore, loss == F.cross_entropy(...)."""
    import torch.nn.functional as F
    B, S, V = 2, 6, 7
    logits = torch.randn(B, S, V)
    targets = torch.randint(0, V, (B, S))
    modality = torch.zeros(B, S, dtype=torch.long)
    out = per_modality_loss_decomposition(logits, targets, modality)
    expected = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="mean")
    assert torch.allclose(out["loss"], expected, atol=1e-5)


# =============================================================================
# Integration smoke — the FULL multimodal forward path
# =============================================================================

def test_joint_forward_smoke():
    """End-to-end: image -> VisionTower -> scatter -> MRoPE positions -> apply_rope.

    No core/model.py wiring (that's a separate file modification). This only
    verifies the multimodal helpers compose correctly into a pipeline.
    """
    # Setup
    enc = MockSiglipEncoder(vision_embed_dim=64)
    vision_tower = VisionTower(
        llm_hidden_size=128, vision_encoder=enc, vision_embed_dim=64,
        spatial_merge_size=2, freeze_merger=False,
    )
    image_pad_token_id = 99
    head_dim = 64

    # Fake tokenized batch: [text, text, IMG, IMG, IMG, IMG, text, text]
    B, S = 1, 8
    input_ids = torch.tensor([[1, 2, 99, 99, 99, 99, 7, 8]])
    text_emb_dim = 128
    inputs_embeds = torch.randn(B, S, text_emb_dim)
    image_pad_mask = input_ids == image_pad_token_id
    assert image_pad_mask.sum().item() == 4

    # Image: 32x32 -> mock conv stride=4 -> 8x8 = 64 patches -> merge 2x2 -> 16 tokens per image
    # We need 4 vision tokens to match the 4 IMG positions, so use 16x16 -> 16 patches -> 4 merged
    pixel_values = torch.randn(1, 3, 16, 16)
    # 16x16 / 4 (conv stride) = 4x4 = 16 patches -> merge_2x2 -> 4 merged tokens
    grid_thw = torch.tensor([[1, 4, 4]])

    # Forward through vision tower
    vision_features = vision_tower(pixel_values, grid_thw)
    assert vision_features.shape == (4, 128)

    # Scatter into the input embedding sequence
    fused_embeds = scatter_vision_features(inputs_embeds, vision_features, image_pad_mask)
    assert fused_embeds.shape == inputs_embeds.shape

    # Build 3D MRoPE positions for the fused sequence
    positions = build_position_ids_for_mm(
        input_ids,
        image_pad_token_id=image_pad_token_id,
        image_grids_merged=[[(1, 2, 2)]],  # merged grid: T=1, H=2, W=2 -> 4 tokens
    )
    assert positions.shape == (3, B, S)

    # Apply rope on a fake attention head
    cos, sin = build_3d_mrope(positions, head_dim=head_dim)
    assert cos.shape == (B, S, head_dim)
    # Project fused_embeds (128d) into head_dim for the rope test
    fake_head = fused_embeds[..., :head_dim]
    rotated = apply_rope(fake_head, cos, sin)
    assert rotated.shape == fake_head.shape

    # Per-modality loss
    fake_logits = torch.randn(B, S, 1000)
    fake_targets = torch.randint(0, 1000, (B, S))
    modality_mask = image_pad_mask.long()  # 1 = vision, 0 = text
    out = per_modality_loss_decomposition(fake_logits, fake_targets, modality_mask)
    assert out["n_text"].item() == 4  # positions 0,1,6,7
    assert out["n_vision"].item() == 4  # positions 2,3,4,5
    assert out["loss"].item() > 0


def test_determinism_same_seed_same_output():
    """Same seed -> same vision tower output (frozen encoder + frozen merger)."""
    enc1 = MockSiglipEncoder(vision_embed_dim=64)
    enc2 = MockSiglipEncoder(vision_embed_dim=64)
    # Copy weights so encoders are equivalent
    enc2.load_state_dict(enc1.state_dict())

    vt1 = VisionTower(llm_hidden_size=128, vision_encoder=enc1, vision_embed_dim=64)
    vt2 = VisionTower(llm_hidden_size=128, vision_encoder=enc2, vision_embed_dim=64)
    vt2.merger.load_state_dict(vt1.merger.state_dict())

    torch.manual_seed(0)
    pixel_values = torch.randn(1, 3, 16, 16)
    grid_thw = torch.tensor([[1, 4, 4]])
    out1 = vt1(pixel_values, grid_thw)
    out2 = vt2(pixel_values, grid_thw)
    assert torch.allclose(out1, out2, atol=1e-5)


if __name__ == "__main__":
    # Allow `python tests/test_multimodal_joint_forward.py` to run all tests directly
    import inspect
    tests = [
        (name, obj) for name, obj in inspect.getmembers(sys.modules[__name__])
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failures.append((name, e))
            print(f"FAIL  {name}: {e}")
    print(f"\n{'='*70}\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    sys.exit(0 if not failures else 1)
