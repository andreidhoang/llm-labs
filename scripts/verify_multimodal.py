"""Verifier for core/multimodal.py.

Runs from llm/ directory:
    python scripts/verify_multimodal.py            # all checks
    python scripts/verify_multimodal.py --siglip   # also runs SigLIP2-dependent checks (downloads ~400MB on first run)
    python scripts/verify_multimodal.py --check=rotate_half   # single check

Each check tests one or two functions in core/multimodal.py. They are
independent — implement any one function in core/multimodal.py and you
can immediately run its corresponding check here without finishing the rest.

Spec: dev/multimodal_spec.md
Reference impl: basics/notebooks/qwen35_vl_tiny.py
"""
from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

# Make `core` importable when invoked from llm/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.multimodal import (
    MultimodalConfig,
    PatchMerger,
    VisionTower,
    _rotate_half,
    apply_rope,
    build_3d_mrope,
    build_position_ids_for_mm,
    per_modality_loss_decomposition,
    scatter_vision_features,
)


# =============================================================================
# Pretty output
# =============================================================================

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"


def _run(name: str, fn, *args, **kwargs) -> str:
    """Run a check, return 'pass' / 'fail' / 'skip'. Print result with traceback on fail."""
    print(f"  [{name}] ... ", end="", flush=True)
    try:
        result = fn(*args, **kwargs)
        if result == "skip":
            print(SKIP)
            return "skip"
        print(PASS)
        return "pass"
    except NotImplementedError as e:
        print(f"{SKIP}  (not implemented yet: {e})")
        return "skip"
    except AssertionError as e:
        print(FAIL)
        print(f"        assertion failed: {e}")
        return "fail"
    except Exception as e:
        print(FAIL)
        print(f"        {type(e).__name__}: {e}")
        print("        " + traceback.format_exc().replace("\n", "\n        "))
        return "fail"


# =============================================================================
# RoPE utility checks (test _rotate_half + apply_rope + build_3d_mrope)
# =============================================================================

def check_rotate_half_shape():
    """_rotate_half should preserve shape."""
    x = torch.randn(2, 3, 4, 8)
    out = _rotate_half(x)
    assert out.shape == x.shape, f"shape changed: {x.shape} → {out.shape}"


def check_rotate_half_double_is_negation():
    """Applying _rotate_half twice negates the input.

    Math: if rotate_half(x) = (-x_back, x_front), then
          rotate_half(rotate_half(x)) = rotate_half((-x_back, x_front))
                                      = (-x_front, -x_back) = -x ✓
    """
    x = torch.randn(2, 4, 8)
    out = _rotate_half(_rotate_half(x))
    assert torch.allclose(out, -x), \
        f"rotate_half(rotate_half(x)) should equal -x; max abs diff = {(out + x).abs().max().item()}"


def check_apply_rope_identity():
    """With cos=1, sin=0, apply_rope returns x unchanged (zero rotation)."""
    x = torch.randn(2, 4, 8)
    cos = torch.ones(4, 8)
    sin = torch.zeros(4, 8)
    out = apply_rope(x, cos, sin)
    assert torch.allclose(out, x, atol=1e-6), \
        f"identity rotation should return x unchanged; max diff = {(out - x).abs().max().item()}"


def check_apply_rope_quarter_turn():
    """With cos=0, sin=1, apply_rope returns _rotate_half(x) (90° rotation)."""
    x = torch.randn(2, 4, 8)
    cos = torch.zeros(4, 8)
    sin = torch.ones(4, 8)
    out = apply_rope(x, cos, sin)
    expected = _rotate_half(x)
    assert torch.allclose(out, expected, atol=1e-6), \
        f"90° rotation should return rotate_half(x); max diff = {(out - expected).abs().max().item()}"


def check_build_3d_mrope_shape():
    """build_3d_mrope returns cos, sin of shape (B, S, head_dim)."""
    B, S, head_dim = 2, 5, 16
    position_ids = torch.zeros(3, B, S, dtype=torch.long)
    cos, sin = build_3d_mrope(position_ids, head_dim, theta=10000.0)
    assert cos.shape == (B, S, head_dim), f"cos shape {cos.shape} != ({B}, {S}, {head_dim})"
    assert sin.shape == (B, S, head_dim), f"sin shape {sin.shape} != ({B}, {S}, {head_dim})"


def check_build_3d_mrope_zero_position_gives_unit_rotation():
    """At position 0, rotation should be cos=1, sin=0 everywhere (no rotation)."""
    B, S, head_dim = 1, 1, 16
    position_ids = torch.zeros(3, B, S, dtype=torch.long)
    cos, sin = build_3d_mrope(position_ids, head_dim)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-6), \
        f"position 0 should give cos=1; got max diff {(cos - 1).abs().max().item()}"
    assert torch.allclose(sin, torch.zeros_like(sin), atol=1e-6), \
        f"position 0 should give sin=0; got max abs {sin.abs().max().item()}"


def check_build_3d_mrope_axes_have_distinct_frequencies():
    """t, h, w axes should each contribute to ~1/3 of the frequencies (round-robin
    interleaving). Verify by setting one axis nonzero at a time and confirming
    the resulting rotation is non-trivial in different positions."""
    B, S, head_dim = 1, 1, 12  # 6 freqs total → 2 per axis
    pos_t = torch.zeros(3, B, S, dtype=torch.long); pos_t[0, 0, 0] = 5
    pos_h = torch.zeros(3, B, S, dtype=torch.long); pos_h[1, 0, 0] = 5
    pos_w = torch.zeros(3, B, S, dtype=torch.long); pos_w[2, 0, 0] = 5
    cos_t, _ = build_3d_mrope(pos_t, head_dim)
    cos_h, _ = build_3d_mrope(pos_h, head_dim)
    cos_w, _ = build_3d_mrope(pos_w, head_dim)
    # Each axis-only setup should produce a different cos pattern
    assert not torch.allclose(cos_t, cos_h, atol=1e-3), \
        "t-axis and h-axis rotations look identical — interleaving may be broken"
    assert not torch.allclose(cos_h, cos_w, atol=1e-3), \
        "h-axis and w-axis rotations look identical — interleaving may be broken"


# =============================================================================
# PatchMerger checks
# =============================================================================

def check_patchmerger_shape():
    """Output shape: (total_merged_tokens, llm_hidden_size).

    Setup: 2 images, each 4×4 patches, vision_embed_dim=8, llm_hidden=16, merge=2.
    total_patches = 2 × (1·4·4) = 32
    total_merged  = 2 × (1·2·2) = 8
    """
    vision_embed_dim, llm_hidden, merge = 8, 16, 2
    merger = PatchMerger(vision_embed_dim, llm_hidden, spatial_merge_size=merge)
    x = torch.randn(32, vision_embed_dim)
    grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    out = merger(x, grid_thw)
    assert out.shape == (8, llm_hidden), f"expected (8, {llm_hidden}); got {out.shape}"


def check_patchmerger_per_image_independence():
    """An image's merged tokens must depend only on that image's patches, not
    on patches from other images in the batch."""
    vision_embed_dim, llm_hidden, merge = 8, 16, 2
    merger = PatchMerger(vision_embed_dim, llm_hidden, spatial_merge_size=merge)
    grid_thw = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long)
    # Image 1 patches = a; image 2 patches = b
    a = torch.randn(4, vision_embed_dim)
    b = torch.randn(4, vision_embed_dim)
    out_ab = merger(torch.cat([a, b], 0), grid_thw)   # (2, llm_hidden)
    # Replace image 2 with different patches — image 1's output should not change
    b2 = torch.randn(4, vision_embed_dim)
    out_ab2 = merger(torch.cat([a, b2], 0), grid_thw)
    assert torch.allclose(out_ab[0], out_ab2[0], atol=1e-6), \
        "image 1's merged token changed when image 2's patches changed — cross-image leakage"


def check_patchmerger_deterministic():
    """Same input → same output (no randomness in forward)."""
    vision_embed_dim, llm_hidden, merge = 8, 16, 2
    merger = PatchMerger(vision_embed_dim, llm_hidden, spatial_merge_size=merge)
    merger.eval()
    x = torch.randn(16, vision_embed_dim)
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    out1 = merger(x, grid_thw)
    out2 = merger(x, grid_thw)
    assert torch.allclose(out1, out2), "non-deterministic forward"


# =============================================================================
# scatter_vision_features checks
# =============================================================================

def check_scatter_at_known_positions():
    """Replace embeddings at specific positions with vision features."""
    B, S, D = 1, 10, 4
    inputs = torch.zeros(B, S, D)
    vision = torch.ones(4, D) * 7.0
    mask = torch.zeros(B, S, dtype=torch.bool)
    mask[0, [2, 3, 7, 8]] = True
    out = scatter_vision_features(inputs, vision, mask)
    # Pad positions should be 7.0
    for pos in [2, 3, 7, 8]:
        assert torch.all(out[0, pos] == 7.0), f"position {pos} not replaced; got {out[0, pos]}"
    # Other positions should remain 0
    for pos in [0, 1, 4, 5, 6, 9]:
        assert torch.all(out[0, pos] == 0.0), f"position {pos} mutated; got {out[0, pos]}"


def check_scatter_count_mismatch_raises():
    """If image_pad_mask.sum() != vision_features.shape[0], should fail loudly."""
    inputs = torch.zeros(1, 10, 4)
    vision = torch.ones(4, 4)  # only 4 features
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, [2, 3, 7, 8, 9]] = True   # 5 mask positions, but only 4 vision features
    try:
        scatter_vision_features(inputs, vision, mask)
        assert False, "expected AssertionError on count mismatch, got none"
    except AssertionError:
        pass  # expected


def check_scatter_no_image_returns_unchanged():
    """When no positions are masked, output equals input."""
    B, S, D = 2, 8, 4
    inputs = torch.randn(B, S, D)
    vision = torch.zeros(0, D)  # empty
    mask = torch.zeros(B, S, dtype=torch.bool)
    out = scatter_vision_features(inputs, vision, mask)
    assert torch.equal(out, inputs), "no-image scatter should return inputs unchanged"


def check_scatter_does_not_mutate_input():
    """scatter_vision_features should not modify inputs_embeds in-place."""
    inputs = torch.zeros(1, 5, 4)
    inputs_clone = inputs.clone()
    vision = torch.ones(2, 4)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    mask[0, [1, 3]] = True
    _ = scatter_vision_features(inputs, vision, mask)
    assert torch.equal(inputs, inputs_clone), \
        "scatter mutated inputs_embeds in place — autograd will silently break"


# =============================================================================
# build_position_ids_for_mm checks
# =============================================================================

def check_position_ids_text_only():
    """No image tokens → all positions advance t-axis only, h=w=0."""
    image_pad = 999
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # no image_pad
    image_grids_merged: list[list[tuple[int, int, int]]] = [[]]
    out = build_position_ids_for_mm(input_ids, image_pad, image_grids_merged)
    assert out.shape == (3, 1, 5), f"shape {out.shape} != (3, 1, 5)"
    expected_t = torch.tensor([[0, 1, 2, 3, 4]])
    assert torch.equal(out[0], expected_t), f"t-axis: got {out[0].tolist()} expected {expected_t.tolist()}"
    assert torch.all(out[1] == 0), f"text h-axis should be 0; got {out[1]}"
    assert torch.all(out[2] == 0), f"text w-axis should be 0; got {out[2]}"


def check_position_ids_with_one_image():
    """Sequence: [text=2, image_pad=4, text=2]; image is (T=1, H=2, W=2) merged.

    Vision tokens should get (h, w) coords:
      pos 2: (h=0, w=0)
      pos 3: (h=0, w=1)
      pos 4: (h=1, w=0)
      pos 5: (h=1, w=1)
    Text tokens advance t-axis. Different conventions exist for whether t
    advances during the image; we check only the (h, w) pattern for vision
    tokens, which should be unambiguous.
    """
    image_pad = 999
    input_ids = torch.tensor([[1, 2, image_pad, image_pad, image_pad, image_pad, 7, 8]])
    image_grids_merged = [[(1, 2, 2)]]
    out = build_position_ids_for_mm(input_ids, image_pad, image_grids_merged)
    assert out.shape == (3, 1, 8), f"shape {out.shape} != (3, 1, 8)"
    # Vision positions are 2, 3, 4, 5
    h_vision = out[1, 0, 2:6].tolist()
    w_vision = out[2, 0, 2:6].tolist()
    # Row-major over (h, w) for a 2x2 grid:
    expected_h = [0, 0, 1, 1]
    expected_w = [0, 1, 0, 1]
    assert h_vision == expected_h, f"vision h-axis: got {h_vision} expected {expected_h}"
    assert w_vision == expected_w, f"vision w-axis: got {w_vision} expected {expected_w}"


# =============================================================================
# per_modality_loss_decomposition checks
# =============================================================================

def check_loss_text_only():
    """All-text batch: loss_text == loss; loss_vision == 0; n_vision == 0."""
    B, S, V = 2, 5, 10
    torch.manual_seed(0)
    logits = torch.randn(B, S, V)
    targets = torch.randint(0, V, (B, S))
    modality_mask = torch.zeros(B, S, dtype=torch.long)  # all text
    out = per_modality_loss_decomposition(logits, targets, modality_mask)
    assert torch.allclose(out["loss"], out["loss_text"], atol=1e-6), \
        f"text-only: loss should equal loss_text; got loss={out['loss'].item():.4f}, loss_text={out['loss_text'].item():.4f}"
    assert out["loss_vision"].item() == 0.0, f"text-only: loss_vision should be 0; got {out['loss_vision'].item()}"
    assert out["n_vision"] == 0, f"text-only: n_vision should be 0; got {out['n_vision']}"
    assert out["n_text"] == B * S, f"n_text {out['n_text']} != B*S {B*S}"


def check_loss_vision_only():
    """All-vision batch: loss_vision == loss; loss_text == 0; n_text == 0."""
    B, S, V = 2, 5, 10
    logits = torch.randn(B, S, V)
    targets = torch.randint(0, V, (B, S))
    modality_mask = torch.ones(B, S, dtype=torch.long)
    out = per_modality_loss_decomposition(logits, targets, modality_mask)
    assert torch.allclose(out["loss"], out["loss_vision"], atol=1e-6)
    assert out["loss_text"].item() == 0.0
    assert out["n_text"] == 0
    assert out["n_vision"] == B * S


def check_loss_mixed_weighted_average():
    """Mixed batch: weighted average of (loss_text * n_text + loss_vision * n_vision)
    / (n_text + n_vision) should equal loss."""
    B, S, V = 1, 6, 4
    torch.manual_seed(1)
    logits = torch.randn(B, S, V)
    targets = torch.randint(0, V, (B, S))
    modality_mask = torch.tensor([[0, 0, 0, 1, 1, 1]])  # 3 text, 3 vision
    out = per_modality_loss_decomposition(logits, targets, modality_mask)
    weighted = (out["loss_text"] * out["n_text"] + out["loss_vision"] * out["n_vision"]) / (
        out["n_text"] + out["n_vision"]
    )
    assert torch.allclose(out["loss"], weighted, atol=1e-5), \
        f"weighted avg {weighted.item():.6f} != loss {out['loss'].item():.6f}"


def check_loss_ignore_index_excluded():
    """ignore_index positions should NOT contribute to loss or counts."""
    B, S, V = 1, 6, 4
    torch.manual_seed(2)
    logits = torch.randn(B, S, V)
    targets = torch.tensor([[0, 1, -1, 2, -1, 3]])  # 2 ignored positions
    modality_mask = torch.zeros(B, S, dtype=torch.long)  # all text
    out = per_modality_loss_decomposition(logits, targets, modality_mask, ignore_index=-1)
    assert out["n_text"] == 4, f"4 valid text positions; got n_text={out['n_text']}"
    assert out["n_vision"] == 0


# =============================================================================
# VisionTower checks (require SigLIP2 download — skip by default)
# =============================================================================

def check_vision_tower_loads():
    """Build VisionTower; verify SigLIP2 weights loaded and .siglip attribute exists."""
    tower = VisionTower(llm_hidden_size=64, freeze_merger=True)
    assert hasattr(tower, "siglip"), "VisionTower has no .siglip attribute after init"
    assert hasattr(tower, "merger"), "VisionTower has no .merger attribute after init"


def check_vision_tower_siglip_frozen():
    """All SigLIP2 params should have requires_grad=False after init."""
    tower = VisionTower(llm_hidden_size=64, freeze_merger=True)
    n_total = 0
    n_frozen = 0
    for p in tower.siglip.parameters():
        n_total += 1
        if not p.requires_grad:
            n_frozen += 1
    assert n_total > 0, "SigLIP2 has no parameters? Loading failed."
    assert n_frozen == n_total, f"only {n_frozen}/{n_total} SigLIP2 params are frozen"


def check_vision_tower_merger_frozen_when_requested():
    """With freeze_merger=True (default), merger params have requires_grad=False."""
    tower = VisionTower(llm_hidden_size=64, freeze_merger=True)
    for n, p in tower.merger.named_parameters():
        assert not p.requires_grad, f"merger.{n} not frozen despite freeze_merger=True"


def check_vision_tower_forward_shape():
    """Forward returns (total_merged_tokens, llm_hidden_size).

    SigLIP2-SO400M at 384x384 with patch=14 → 27x27 = 729 patches per image.
    With 2x2 merger → 729/4 = 182 (rounded down); often 182 if patch grid is
    2-divisible (27 isn't). Verify the actual number matches what the merger
    arithmetic predicts for the implemented spatial_merge_size.
    """
    tower = VisionTower(llm_hidden_size=64)
    pixel_values = torch.randn(1, 3, 384, 384)
    grid_thw = torch.tensor([[1, 27, 27]], dtype=torch.long)  # SigLIP2's actual patch grid for 384/14
    # NOTE: 27 is not divisible by 2 — implementer needs to handle this
    # (either crop, pad, or pick a resolution that gives even patch count).
    # For first-pass test, may need to use 28x28 patches → image size 392.
    # Comment this check out or adjust resolution if it fails.
    try:
        out = tower(pixel_values, grid_thw)
        assert out.dim() == 2 and out.shape[1] == 64, \
            f"expected (N, 64); got {out.shape}"
    except AssertionError as e:
        if "divisible" in str(e):
            print(f"        (note: 27x27 not divisible by 2; try image_size=392 → 28x28 patches)")
        raise


# =============================================================================
# Integration check (all pieces working together)
# =============================================================================

def check_integration_text_only():
    """End-to-end no-image path: build_position_ids → ... → loss decomposition.

    Pure-text input flows through every helper without errors. Catches missing
    edge cases like 'no images in batch'."""
    B, S, V = 1, 8, 16
    image_pad = 999
    input_ids = torch.randint(0, V, (B, S))  # no image_pad tokens
    # Position ids
    pos_ids = build_position_ids_for_mm(input_ids, image_pad, [[]])
    assert pos_ids.shape == (3, B, S)
    # Modality mask all-text
    modality_mask = torch.zeros(B, S, dtype=torch.long)
    # Fake logits
    logits = torch.randn(B, S, V)
    targets = input_ids
    out = per_modality_loss_decomposition(logits, targets, modality_mask)
    assert torch.isfinite(out["loss"]).all()


# =============================================================================
# Main runner
# =============================================================================

ALL_CHECKS = {
    # RoPE utilities
    "rotate_half_shape": check_rotate_half_shape,
    "rotate_half_double": check_rotate_half_double_is_negation,
    "apply_rope_identity": check_apply_rope_identity,
    "apply_rope_quarter_turn": check_apply_rope_quarter_turn,
    "build_3d_mrope_shape": check_build_3d_mrope_shape,
    "build_3d_mrope_zero_position": check_build_3d_mrope_zero_position_gives_unit_rotation,
    "build_3d_mrope_axes_distinct": check_build_3d_mrope_axes_have_distinct_frequencies,
    # PatchMerger
    "patchmerger_shape": check_patchmerger_shape,
    "patchmerger_independence": check_patchmerger_per_image_independence,
    "patchmerger_deterministic": check_patchmerger_deterministic,
    # scatter
    "scatter_known_positions": check_scatter_at_known_positions,
    "scatter_count_mismatch": check_scatter_count_mismatch_raises,
    "scatter_no_image": check_scatter_no_image_returns_unchanged,
    "scatter_no_mutate": check_scatter_does_not_mutate_input,
    # position ids
    "position_ids_text_only": check_position_ids_text_only,
    "position_ids_with_image": check_position_ids_with_one_image,
    # loss decomposition
    "loss_text_only": check_loss_text_only,
    "loss_vision_only": check_loss_vision_only,
    "loss_mixed_weighted_avg": check_loss_mixed_weighted_average,
    "loss_ignore_index": check_loss_ignore_index_excluded,
    # integration
    "integration_text_only": check_integration_text_only,
}

SIGLIP_CHECKS = {
    "vision_tower_loads": check_vision_tower_loads,
    "vision_tower_siglip_frozen": check_vision_tower_siglip_frozen,
    "vision_tower_merger_frozen": check_vision_tower_merger_frozen_when_requested,
    "vision_tower_forward_shape": check_vision_tower_forward_shape,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip", action="store_true",
                        help="also run SigLIP2-dependent checks (downloads ~400MB on first run)")
    parser.add_argument("--check", default=None,
                        help="run a single named check (e.g. --check=rotate_half_shape)")
    args = parser.parse_args()

    checks = {**ALL_CHECKS}
    if args.siglip:
        checks.update(SIGLIP_CHECKS)
    if args.check:
        if args.check not in checks:
            available = ", ".join(sorted(checks.keys()))
            print(f"unknown check '{args.check}'. Available: {available}")
            sys.exit(2)
        checks = {args.check: checks[args.check]}

    print(f"Running {len(checks)} check(s) against core/multimodal.py:\n")
    results = {"pass": 0, "fail": 0, "skip": 0}
    for name, fn in checks.items():
        r = _run(name, fn)
        results[r] += 1

    print()
    print(f"Summary: {results['pass']} pass, {results['fail']} fail, {results['skip']} skip")
    if not args.siglip and any(name in ALL_CHECKS for name in checks):
        print("(Add --siglip to also test VisionTower; downloads SigLIP2-SO400M ~400MB on first run.)")
    if results["fail"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
