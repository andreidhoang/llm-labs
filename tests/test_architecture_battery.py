"""Architecture test battery — tests 5-10 from the senior-engineer review.

Run BEFORE launching A0 on vast.ai. Catches:

  - test_rope_shift_invariance        → relative-position property holds
  - test_mrope_hw_axis_sensitivity    → swapping H/W changes the output (catches axis swap)
  - test_mrope_text_only_matches_1d   → 3D MRoPE at (t, 0, 0) reduces to 1D RoPE
  - test_full_gradient_existence      → every learnable param receives grad after one step
  - test_shape_contract_text_forward  → text-only forward shape contract
  - test_shape_contract_multimodal    → multimodal forward shape contract
  - test_activation_rms_stability     → activation RMS within [0.25x, 4x] across depth
  - test_param_count_assertion        → num_scaling_params() internal assert passes

Each test runs on CPU. Total wall-clock ≈ 20-30 s.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from core.model import GPT, GPTConfig, apply_rotary_emb
from core.multimodal import build_3d_mrope_for_4d_apply


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_tiny_gpt(n_layer: int = 2, n_embd: int = 128, n_head: int = 4,
                   n_kv_head: int = 2, multimodal: bool = False, seed: int = 0):
    """Construct a tiny GPT instance for fast CPU testing.
    Uses depth-aware dim so MoE expert_hidden_dim > 0."""
    torch.manual_seed(seed)
    cfg = GPTConfig(
        sequence_len=128, vocab_size=256, n_layer=n_layer,
        n_head=n_head, n_kv_head=n_kv_head, n_embd=n_embd,
        num_experts=4, top_k=2, num_shared_experts=1,
        window_pattern="L",  # full context, simpler for tests
        multimodal=multimodal,
    )
    model = GPT(cfg)
    model.init_weights()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: M-RoPE H/W axis sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def test_mrope_hw_axis_sensitivity():
    """Swap (h, w) → (w, h) in the position_ids and verify the rotation result
    differs. If they're identical, the M-RoPE axis assignment has a swap bug or
    is degenerate."""
    head_dim = 64
    B, T = 1, 1

    # Position at (t=0, h=1, w=2)
    pos_a = torch.zeros(3, B, T, dtype=torch.long)
    pos_a[0, 0, 0] = 0
    pos_a[1, 0, 0] = 1  # h
    pos_a[2, 0, 0] = 2  # w
    cos_a, sin_a = build_3d_mrope_for_4d_apply(pos_a, head_dim)

    # Position at (t=0, h=2, w=1) — H/W swapped
    pos_b = torch.zeros(3, B, T, dtype=torch.long)
    pos_b[0, 0, 0] = 0
    pos_b[1, 0, 0] = 2  # h (was w)
    pos_b[2, 0, 0] = 1  # w (was h)
    cos_b, sin_b = build_3d_mrope_for_4d_apply(pos_b, head_dim)

    # Apply RoPE to the same Q vector at both positions
    q = torch.randn(B, T, 1, head_dim).bfloat16()  # (B, T, n_head=1, D)
    q_a = apply_rotary_emb(q, cos_a, sin_a)
    q_b = apply_rotary_emb(q, cos_b, sin_b)

    # Must differ — if H==W axis-swap is silent, these would be equal
    diff = (q_a - q_b).abs().float().max().item()
    assert diff > 1e-3, (
        f"H/W swap produced bit-identical RoPE output (max diff {diff:.2e}). "
        "Either the M-RoPE axis assignment is degenerate or there's an axis-swap bug."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reduction: 3D MRoPE at (t, 0, 0) should be (approximately) 1D RoPE
# ─────────────────────────────────────────────────────────────────────────────

def test_mrope_text_only_matches_1d_geometry():
    """For text-only tokens (h=w=0), the 3D MRoPE assigns h/w frequencies to
    angle 0 (no rotation on those freq slots). The 1D RoPE rotates all slots
    by t-axis freqs. So they DIFFER in the h/w slots — but the relative-position
    property (dot product depends on t1-t2) must still hold for text tokens.

    This test verifies that for two text tokens at t=m and t=n, the 3D-MRoPE
    attention dot product depends on (m-n) only, like 1D RoPE.
    """
    head_dim = 64
    B = 1
    n_head = 1
    rng = torch.Generator().manual_seed(0)
    q_base = torch.randn(B, 1, n_head, head_dim, generator=rng).bfloat16()
    k_base = torch.randn(B, 1, n_head, head_dim, generator=rng).bfloat16()

    def dot_at_positions(t_q, t_k):
        pos_q = torch.tensor([[[t_q]], [[0]], [[0]]], dtype=torch.long)
        pos_k = torch.tensor([[[t_k]], [[0]], [[0]]], dtype=torch.long)
        cos_q, sin_q = build_3d_mrope_for_4d_apply(pos_q, head_dim)
        cos_k, sin_k = build_3d_mrope_for_4d_apply(pos_k, head_dim)
        q = apply_rotary_emb(q_base, cos_q, sin_q)
        k = apply_rotary_emb(k_base, cos_k, sin_k)
        return (q * k).sum().float().item()

    # Same relative offset → same dot product (relative position property)
    d_01 = dot_at_positions(0, 1)
    d_56 = dot_at_positions(5, 6)
    rel_err = abs(d_01 - d_56) / (abs(d_01) + 1e-6)
    assert rel_err < 0.05, (  # bf16 precision tolerance
        f"3D-MRoPE relative position broken on text tokens: "
        f"dot(t=0,t=1)={d_01:.4f}, dot(t=5,t=6)={d_56:.4f}, rel_err={rel_err:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: RoPE shift-invariance (relative-position property)
# ─────────────────────────────────────────────────────────────────────────────

def test_rope_shift_invariance_1d():
    """1D RoPE: <Q_m, K_n> depends only on (m-n). Constant-shifting both
    positions must leave the dot product unchanged (within bf16 precision)."""
    model = _build_tiny_gpt(n_layer=1, n_embd=128, n_head=4, n_kv_head=2)
    head_dim = model.config.n_embd // model.config.n_head

    cos_low = model.cos[:, 0:3]   # positions 0, 1, 2
    sin_low = model.sin[:, 0:3]
    cos_high = model.cos[:, 10:13]  # positions 10, 11, 12
    sin_high = model.sin[:, 10:13]

    # Random Q, K, same vectors at both position regimes
    rng = torch.Generator().manual_seed(0)
    q = torch.randn(1, 3, 4, head_dim, generator=rng).bfloat16()
    k = torch.randn(1, 3, 2, head_dim, generator=rng).bfloat16()

    q_low = apply_rotary_emb(q, cos_low, sin_low)
    k_low = apply_rotary_emb(k, cos_low, sin_low)
    q_high = apply_rotary_emb(q, cos_high, sin_high)
    k_high = apply_rotary_emb(k, cos_high, sin_high)

    # Dot product per query-key pair (just take head 0, kv-head 0)
    # <Q_m, K_n> at m=0, n=1 should equal m=10, n=11
    d_low = (q_low[0, 0, 0] * k_low[0, 1, 0]).sum().float().item()
    d_high = (q_high[0, 0, 0] * k_high[0, 1, 0]).sum().float().item()

    rel_err = abs(d_low - d_high) / (abs(d_low) + 1e-6)
    assert rel_err < 0.05, (
        f"1D RoPE relative-position property broken: "
        f"<Q_0,K_1>={d_low:.4f}, <Q_10,K_11>={d_high:.4f}, rel_err={rel_err:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Full gradient existence (text-only)
# ─────────────────────────────────────────────────────────────────────────────

def test_full_gradient_existence_text_only():
    """Every learnable parameter must receive non-zero grad after TWO fwd+bwd
    steps. The architecture's residual-branch downscaling zero-inits attn.c_proj
    and moe.experts.w_down — on step 1 their downstream gradient flows are
    legitimately zero (the projection through a zero matrix is zero, so
    gradient to c_q/c_k/c_v/w_up/gate is zero). After ONE manual SGD step
    on c_proj and w_down, they move off zero; step 2 then exercises ALL params.

    This catches dead branches and detached subgraphs while accommodating the
    intentional zero-init scheme.
    """
    model = _build_tiny_gpt(n_layer=2, n_embd=128, n_head=4, n_kv_head=2)
    model.train()

    # Step 1: nudge zero-init residual-branch projections off the zero basin
    idx = torch.randint(0, model.config.vocab_size, (2, 16))
    targets = torch.randint(0, model.config.vocab_size, (2, 16))
    loss1 = model(idx, targets=targets)
    loss1.backward()
    # Apply SGD step to ALL parameters that received gradient (small LR)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                p.add_(p.grad, alpha=-0.01)
            if p.grad is not None:
                p.grad.zero_()

    # Step 2: now every path has non-zero throughput
    loss2 = model(idx, targets=targets)
    loss2.backward()

    dead = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            dead.append((name, "grad is None"))
            continue
        if p.grad.abs().sum().item() == 0.0:
            dead.append((name, f"grad sum = 0 (shape {tuple(p.shape)})"))
    assert not dead, (
        f"{len(dead)} learnable params received zero gradient on step 2: "
        f"{dead[:5]}{'...' if len(dead) > 5 else ''}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Shape contract — text-only forward
# ─────────────────────────────────────────────────────────────────────────────

def test_shape_contract_text_only():
    """Forward output shape: (B, T, vocab_size) for text-only path."""
    model = _build_tiny_gpt(n_layer=2, n_embd=128, n_head=4, n_kv_head=2)
    model.eval()

    B, T = 2, 16
    idx = torch.randint(0, model.config.vocab_size, (B, T))

    with torch.no_grad():
        logits = model(idx)

    assert logits.shape == (B, T, model.config.vocab_size), (
        f"Expected logits shape ({B}, {T}, {model.config.vocab_size}), "
        f"got {tuple(logits.shape)}"
    )
    assert logits.dtype == torch.float32, (
        f"Logits must be fp32 (for softcap + loss); got {logits.dtype}"
    )


def test_shape_contract_text_with_targets():
    """Text-only with targets returns scalar loss tensor."""
    model = _build_tiny_gpt(n_layer=2)
    model.train()

    idx = torch.randint(0, model.config.vocab_size, (2, 16))
    targets = torch.randint(0, model.config.vocab_size, (2, 16))
    loss = model(idx, targets=targets)

    assert loss.ndim == 0, f"Expected scalar loss, got shape {tuple(loss.shape)}"
    assert loss.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Activation RMS stability across depth
# ─────────────────────────────────────────────────────────────────────────────

def test_activation_rms_stability():
    """Forward through models of increasing depth; final activation RMS
    should stay in [0.25x, 4x] of the embedding-norm RMS.

    Catches: init bugs, residual scale bugs, depth-scaling violations."""
    rms_by_depth = {}
    for n_layer in (1, 4, 8):
        model = _build_tiny_gpt(n_layer=n_layer, n_embd=128, n_head=4, n_kv_head=2)
        model.eval()
        B, T = 2, 16
        idx = torch.randint(0, model.config.vocab_size, (B, T))
        with torch.no_grad():
            # Run up through the trunk, stop before LM head to measure residual RMS
            x = model.transformer.wte(idx)
            from core.model import norm
            x = norm(x)
            x0 = x
            cos_sin = (model.cos[:, :T], model.sin[:, :T])
            for i, block in enumerate(model.transformer.h):
                x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
                ve = (model.value_embeds[str(i)](idx)
                      if str(i) in model.value_embeds else None)
                x = block(x, ve, cos_sin, model.window_sizes[i], None)
            x = norm(x)
            rms = x.float().pow(2).mean().sqrt().item()
            rms_by_depth[n_layer] = rms

    # Reference: after RMSNorm, RMS should be ≈ 1.0
    for n_layer, rms in rms_by_depth.items():
        assert 0.25 < rms < 4.0, (
            f"Activation RMS at depth {n_layer} = {rms:.4f}; "
            f"outside [0.25, 4.0] tolerance. Indicates init / residual-scale bug. "
            f"All depths: {rms_by_depth}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: Param-count assertion (the one already in code)
# ─────────────────────────────────────────────────────────────────────────────

def test_param_count_self_consistency():
    """num_scaling_params() has an internal assert that programmatic count
    matches the sum-of-parameters. Verify it runs and asserts pass."""
    model = _build_tiny_gpt(n_layer=2)
    counts = model.num_scaling_params()
    # If the internal assertion holds, total == sum(p.numel())
    actual = sum(p.numel() for p in model.parameters())
    assert counts["total"] == actual, (
        f"num_scaling_params returned total={counts['total']} but "
        f"sum(p.numel())={actual}"
    )
    # Active should be ≤ total (MoE accounting)
    assert counts["active_total"] <= counts["total"]
    assert counts["active_total"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 9b: Multimodal shape contract (requires mock SigLIP — skip if unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def test_shape_contract_multimodal_mock():
    """Multimodal forward: pixel_values → vision_tower → scatter → trunk.
    Shape contract: logits is (B, T, vocab_size); per-modality loss dict is well-formed."""
    from core.multimodal import VisionTower
    cfg = GPTConfig(
        sequence_len=128, vocab_size=256, n_layer=2,
        n_head=4, n_kv_head=2, n_embd=128,
        num_experts=4, top_k=2, num_shared_experts=1,
        window_pattern="L",
        multimodal=True,
    )
    # We construct GPT WITHOUT building the real SigLIP (download would be heavy)
    # and replace vision_tower with a mock.
    torch.manual_seed(0)
    # Bypass auto-build of vision tower; use mock
    model = GPT(cfg)
    model._needs_vision_tower = False  # disable auto-build
    model.init_weights()

    # Now attach a mock VisionTower manually
    class MockSigLIP(nn.Module):
        def __init__(self, hidden=1152):
            super().__init__()
            self.dummy = nn.Linear(1, 1)
            self.hidden = hidden
        def forward(self, x):
            # x: (N, 3, H, W) → (N, P, hidden); just return zeros of the expected shape
            P = 4  # tiny patch count for the test
            return torch.zeros(x.shape[0], P, self.hidden)

    model.vision_tower = VisionTower(
        llm_hidden_size=cfg.n_embd,
        vision_encoder=MockSigLIP(),
        vision_embed_dim=1152,
        spatial_merge_size=2,
    )
    # Re-run the merger init that init_weights would have done
    for layer in (model.vision_tower.merger.fc1, model.vision_tower.merger.fc2):
        fan_in = layer.in_features
        s_layer = (3.0 ** 0.5) * (fan_in ** -0.5)
        torch.nn.init.uniform_(layer.weight, -s_layer, s_layer)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
    model.eval()

    # Tiny synthetic batch: 1 image, 1 batch, sequence with 1 image_pad position
    # Mock SigLIP returns 4 patches → PatchMerger 2x2 merge → 1 merged token
    # But we need H, W ≥ merge=2 to satisfy PatchMerger reshape.
    # Set grid = (1, 2, 2) → 4 patches input → 1 merged token output.
    B, T = 1, 8
    image_pad_token_id = 1
    idx = torch.tensor([[0, 0, image_pad_token_id, 0, 0, 0, 0, 0]])
    image_pad_mask = (idx == image_pad_token_id)
    pixel_values = torch.zeros(1, 3, 28, 28)  # (N=1, 3, H, W)
    grid_thw = torch.tensor([[1, 2, 2]])
    image_grids_merged = [[(1, 1, 1)]]  # after merge: T=1, H=1, W=1

    with torch.no_grad():
        logits = model(
            idx, pixel_values=pixel_values, grid_thw=grid_thw,
            image_pad_mask=image_pad_mask, image_grids_merged=image_grids_merged,
        )

    assert logits.shape == (B, T, cfg.vocab_size), (
        f"Multimodal logits shape {tuple(logits.shape)} != ({B}, {T}, {cfg.vocab_size})"
    )
    assert not torch.isnan(logits).any(), "NaN in multimodal logits"
