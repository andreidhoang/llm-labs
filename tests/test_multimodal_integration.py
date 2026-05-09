"""Integration test for Track B' wiring.

Exercises the FULL multimodal training path end-to-end on CPU:
- synthetic_multimodal_loader produces batch_extras
- GPT.forward (with multimodal=True) accepts batch_extras and returns dict-shaped loss
- Per-modality loss split is non-degenerate
- Backward pass works (gradients flow)
- Text-only path still works on the same multimodal-enabled model

Uses a mock SigLIP encoder + tiny model + synthetic data (no HF download, no GPU).

Spec: dev/scaling_law_self_assignment.md Gate G0
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.dataloader import synthetic_multimodal_loader  # noqa: E402
from core.model import GPT, GPTConfig  # noqa: E402
from core.multimodal import VisionTower  # noqa: E402


class MockSiglip(nn.Module):
    """Tiny stand-in for SigLIP2.vision_model — returns (B, n_patches, D)."""
    def __init__(self, vision_embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(3, vision_embed_dim, kernel_size=4, stride=4)

    def forward(self, pixel_values):
        x = self.proj(pixel_values)
        B, D, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, D)


def _build_multimodal_model(vocab_size=256, n_embd=64):
    cfg = GPTConfig(
        sequence_len=128, vocab_size=vocab_size, n_layer=2,
        n_head=4, n_kv_head=4, n_embd=n_embd,
        num_experts=4, top_k=2, num_shared_experts=1,
        window_pattern="L",
        multimodal=True, vision_embed_dim=64, vision_spatial_merge_size=2,
        image_pad_token_id=vocab_size - 1,
    )
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device="cpu")
    m._needs_vision_tower = False  # skip HF download
    m.init_weights()
    m.vision_tower = VisionTower(
        llm_hidden_size=n_embd, vision_encoder=MockSiglip(vision_embed_dim=64),
        vision_embed_dim=64, spatial_merge_size=2, freeze_merger=True,
    )
    return m, cfg


def _fake_text_loader(B=2, S=64, vocab=256, n_batches=3):
    """Yields (inputs, targets, state_dict) tuples mimicking the real loader."""
    torch.manual_seed(0)
    for i in range(n_batches):
        inputs = torch.randint(0, vocab - 10, (B, S), dtype=torch.long)
        targets = torch.randint(0, vocab - 10, (B, S), dtype=torch.long)
        state_dict = {"pq_idx": 0, "rg_idx": 0, "epoch": 1}
        yield inputs, targets, state_dict


def test_synthetic_loader_produces_extras():
    """synthetic_multimodal_loader wraps text loader and adds batch_extras."""
    base = _fake_text_loader(B=2, S=64, vocab=256, n_batches=2)
    loader = synthetic_multimodal_loader(
        base, mix_ratio=0.3, image_grid_thw_raw=(1, 4, 4),
        image_pad_token_id=255, image_size_pixels=16, seed=0, device="cpu",
    )
    inputs, targets, extras, state = next(loader)
    assert inputs.shape == (2, 64)
    assert targets.shape == (2, 64)
    assert "pixel_values" in extras
    assert "grid_thw" in extras
    assert "image_pad_mask" in extras
    assert "modality_mask" in extras
    # Verify pad mask matches image_pad_token_id positions in inputs
    assert torch.equal(extras["image_pad_mask"], inputs == 255)
    # modality_mask now categorizes by vision-context (per multimodal_spec.md §2.5.5):
    # tokens AFTER vision (within window) get tagged 1 too. So:
    # modality_mask >= image_pad_mask (entry-wise)
    assert (extras["modality_mask"] >= extras["image_pad_mask"].long()).all()
    # And strictly greater in sum (text-after-vision tokens are tagged too)
    assert extras["modality_mask"].sum() > extras["image_pad_mask"].sum()


def test_synthetic_loader_mix_ratio_in_range():
    """r_actual (raw vision-token fraction from image_pad_mask) should be near target.

    Note: this measures raw image_pad rate, NOT modality_mask rate. modality_mask
    additionally tags vision-context text tokens which inflates beyond the pixel-token
    fraction. The r=0.3 contract is on the raw pixel-token count.
    """
    base = _fake_text_loader(B=4, S=128, vocab=256, n_batches=5)
    loader = synthetic_multimodal_loader(
        base, mix_ratio=0.3, image_grid_thw_raw=(1, 4, 4),
        image_pad_token_id=255, image_size_pixels=16, seed=0, device="cpu",
    )
    rs = []
    for inputs, targets, extras, state in loader:
        n_vision_pad = extras["image_pad_mask"].sum().item()
        n_total = inputs.numel()
        rs.append(n_vision_pad / n_total)
    avg_r = sum(rs) / len(rs)
    # Synthetic placement is approximate (rounds to image runs); accept ±0.10 of target
    assert 0.20 <= avg_r <= 0.40, f"r_actual {avg_r} too far from 0.3 target"


def test_synthetic_loader_pixel_values_shape():
    base = _fake_text_loader(B=2, S=64, vocab=256, n_batches=1)
    loader = synthetic_multimodal_loader(
        base, mix_ratio=0.3, image_grid_thw_raw=(1, 4, 4),
        image_pad_token_id=255, image_size_pixels=16, seed=0, device="cpu",
    )
    inputs, targets, extras, state = next(loader)
    n_imgs = extras["pixel_values"].shape[0]
    assert extras["pixel_values"].shape == (n_imgs, 3, 16, 16)
    assert extras["grid_thw"].shape == (n_imgs, 3)
    # All grids should be (1, 4, 4) per the function arg
    assert torch.all(extras["grid_thw"] == torch.tensor([1, 4, 4]))


def test_synthetic_loader_determinism():
    """Same seed → same image_pad layout + same synthetic pixels."""
    def make():
        torch.manual_seed(0)  # ensure base loader is also deterministic
        base = _fake_text_loader(B=2, S=64, vocab=256, n_batches=1)
        return synthetic_multimodal_loader(
            base, mix_ratio=0.3, image_grid_thw_raw=(1, 4, 4),
            image_pad_token_id=255, image_size_pixels=16, seed=42, device="cpu",
        )
    inputs1, targets1, extras1, _ = next(make())
    inputs2, targets2, extras2, _ = next(make())
    assert torch.equal(inputs1, inputs2)
    assert torch.equal(extras1["image_pad_mask"], extras2["image_pad_mask"])
    assert torch.allclose(extras1["pixel_values"], extras2["pixel_values"])


def test_full_multimodal_training_step():
    """Loader -> model.forward -> per-modality loss dict -> backward.

    This is the smallest end-to-end smoke that proves the full Track B' wiring
    works. If this passes, the GPU runs should also work (modulo SigLIP2
    download + actual MFU concerns).
    """
    m, cfg = _build_multimodal_model(vocab_size=256, n_embd=64)
    base = _fake_text_loader(B=2, S=64, vocab=256, n_batches=1)
    loader = synthetic_multimodal_loader(
        base, mix_ratio=0.3, image_grid_thw_raw=(1, 4, 4),
        image_pad_token_id=cfg.vocab_size - 1, image_size_pixels=16, seed=0, device="cpu",
    )
    inputs, targets, extras, state = next(loader)

    # Confirm the loader emitted image_grids_merged (Step 3 of 3D MRoPE wiring)
    assert "image_grids_merged" in extras, "loader must emit image_grids_merged"

    # Forward through the multimodal-enabled model — should use 3D MRoPE because
    # image_grids_merged is in extras
    out = m(inputs, targets, **extras)
    assert isinstance(out, dict), f"expected dict (modality_mask was passed), got {type(out)}"
    assert "loss" in out and "loss_text" in out and "loss_vision" in out
    # n_text > 0: text tokens with text-only context contribute to loss/text
    assert out["n_text"].item() > 0
    # n_vision > 0 (post Track B'' refinement): text tokens with vision context
    # in their recent attention window contribute to loss/vision. This is the
    # correct multimodal_spec.md §2.5.5 interpretation — vision is CONTEXT, and
    # loss/vision tracks "text-token loss when vision context is present".
    assert out["n_vision"].item() > 0
    assert out["loss"].item() > 0
    # Sanity: total loss is a weighted blend of the two
    n_t, n_v = out["n_text"].item(), out["n_vision"].item()
    blended = (out["loss_text"].item() * n_t + out["loss_vision"].item() * n_v) / (n_t + n_v)
    assert abs(blended - out["loss"].item()) < 0.01

    # Backward should work (gradients flow into trunk; vision tower is frozen)
    out["loss"].backward()
    n_grads = sum(1 for p in m.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0)
    assert n_grads > 0, "no gradients flowed"
    # Confirm vision tower stayed frozen
    n_vision_grads = sum(1 for p in m.vision_tower.siglip.parameters() if p.grad is not None)
    assert n_vision_grads == 0, "vision encoder gradients should be None (frozen)"


def test_multimodal_generate_naive():
    """GPT.generate() with multimodal kwargs runs end-to-end and yields text tokens."""
    m, cfg = _build_multimodal_model(vocab_size=256, n_embd=64)
    # Build a small multimodal prompt
    input_ids = torch.tensor([[1, 2, 99, 99, 99, 99, 5, 6, 7, 8]])
    input_ids[input_ids == 99] = cfg.vocab_size - 1
    image_pad_mask = (input_ids == cfg.vocab_size - 1)
    pixel_values = torch.randn(1, 3, 16, 16)
    grid_thw = torch.tensor([[1, 4, 4]])
    image_grids_merged = [[(1, 2, 2)]]

    tokens = input_ids[0].tolist()
    gen = m.generate(
        tokens, max_tokens=3, temperature=0.0,
        pixel_values=pixel_values, grid_thw=grid_thw,
        image_pad_mask=image_pad_mask, image_grids_merged=image_grids_merged,
    )
    out_tokens = list(gen)
    assert len(out_tokens) == 3, f"expected 3 tokens, got {len(out_tokens)}"
    assert all(0 <= t < cfg.vocab_size for t in out_tokens), f"token range wrong: {out_tokens}"


def test_kvcache_multimodal_state_transitions():
    """KVCache.next_t_axis_position lifecycle (CPU-friendly — no attention call).

    Full forward+kv_cache integration requires Flash Attention 3 (GPU-only) so we
    test the state-management logic directly. The model's setting of
    next_t_axis_position is exercised by simulating what GPT.forward does.
    """
    from core.engine import KVCache
    from core.multimodal import build_position_ids_for_mm

    head_dim = 16
    kv_cache = KVCache(
        batch_size=1, num_heads=4, seq_len=128,
        head_dim=head_dim, num_layers=2,
        device="cpu", dtype=torch.bfloat16,
    )
    assert kv_cache.next_t_axis_position is None  # text-only mode by default

    # Simulate what GPT.forward does on multimodal prefill:
    # 1. Build position_ids for the prefix
    # 2. Set kv_cache.next_t_axis_position = max(position_ids[0]) + 1
    input_ids = torch.tensor([[1, 2, 99, 99, 99, 99, 5, 6]])
    image_pad_token_id = 99
    image_grids_merged = [[(1, 2, 2)]]
    position_ids = build_position_ids_for_mm(input_ids, image_pad_token_id, image_grids_merged)
    next_t = int(position_ids[0].max().item()) + 1
    kv_cache.next_t_axis_position = next_t

    # Layout: text@t=0, text@t=1, images@t=2 (4 tokens), text@t=3, text@t=4 → max=4, next=5
    assert kv_cache.next_t_axis_position == 5

    # Simulate continuation step: forward advances by T=1
    kv_cache.next_t_axis_position += 1
    assert kv_cache.next_t_axis_position == 6

    # Reset clears multimodal state (back to text-only default)
    kv_cache.reset()
    assert kv_cache.next_t_axis_position is None


def test_kvcache_prefill_copies_multimodal_state():
    """KVCache.prefill() copies next_t_axis_position to the new cache."""
    from core.engine import KVCache

    head_dim = 16
    src = KVCache(batch_size=1, num_heads=4, seq_len=64, head_dim=head_dim, num_layers=2,
                  device="cpu", dtype=torch.bfloat16)
    src.next_t_axis_position = 10
    src.cache_seqlens.fill_(8)

    dst = KVCache(batch_size=4, num_heads=4, seq_len=64, head_dim=head_dim, num_layers=2,
                  device="cpu", dtype=torch.bfloat16)
    dst.prefill(src)
    assert dst.next_t_axis_position == 10


def test_text_only_still_works_on_multimodal_model():
    """Same multimodal-enabled model still trains text-only when no extras passed."""
    m, cfg = _build_multimodal_model(vocab_size=256, n_embd=64)
    inputs = torch.randint(0, 250, (2, 32))
    targets = torch.randint(0, 250, (2, 32))
    loss = m(inputs, targets)
    assert loss.dim() == 0  # scalar loss (text-only path)
    loss.backward()
    n_grads = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
    assert n_grads > 0


def test_3d_mrope_produces_different_values_than_1d():
    """Verify 3D MRoPE values differ from 1D RoPE at vision positions.

    1D RoPE assigns sequential positions (0, 1, 2, ...) to all tokens including vision.
    3D MRoPE assigns (t, h, w) to vision tokens, so cos/sin values differ at those positions.
    """
    from core.multimodal import (
        build_3d_mrope_for_4d_apply,
        build_position_ids_for_mm,
    )

    m, cfg = _build_multimodal_model(vocab_size=256, n_embd=64)

    input_ids = torch.tensor([[1, 2, 99, 99, 99, 99, 5, 6, 7, 8, 9, 10]])
    input_ids[input_ids == 99] = cfg.vocab_size - 1
    image_grids_merged = [[(1, 2, 2)]]

    head_dim = cfg.n_embd // cfg.n_head
    position_ids = build_position_ids_for_mm(
        input_ids,
        image_pad_token_id=cfg.image_pad_token_id,
        image_grids_merged=image_grids_merged,
    )
    cos_3d, _ = build_3d_mrope_for_4d_apply(position_ids, head_dim)
    cos_1d = m.cos[:, : cos_3d.shape[1]]  # same shape but values from sequential positions

    # At text positions where t-axis matches sequential idx, values may match.
    # At VISION positions (2-5), 3D uses (t, h, w) which differs from sequential.
    # So cos[0, 2, 0, :] (3D, position (2, 0, 0)) should differ from cos_1d[0, 2, 0, :]
    # because 3D MRoPE rotates differently for those positions.
    diff = (cos_3d.float() - cos_1d.float()).abs().max().item()
    # If 3D MRoPE were identical to 1D, diff would be ~0. Expect substantial diff at vision rows.
    assert diff > 0.01, f"3D MRoPE should differ from 1D at vision positions; diff={diff}"


def test_synthetic_loader_targets_ignore_at_vision_positions():
    """Targets at vision positions should be -1 (ignored in CE loss)."""
    base = _fake_text_loader(B=2, S=64, vocab=256, n_batches=1)
    loader = synthetic_multimodal_loader(
        base, mix_ratio=0.3, image_grid_thw_raw=(1, 4, 4),
        image_pad_token_id=255, image_size_pixels=16, seed=0, device="cpu",
    )
    inputs, targets, extras, state = next(loader)
    pad_mask = extras["image_pad_mask"]
    assert torch.all(targets[pad_mask] == -1), "vision-position targets should be -1 (ignore)"


if __name__ == "__main__":
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
            import traceback
            print(f"FAIL  {name}: {e}")
            traceback.print_exc()
    print(f"\n{'='*70}\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    sys.exit(0 if not failures else 1)
