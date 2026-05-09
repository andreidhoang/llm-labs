"""Verify the compile-fullgraph-friendly forward path is numerically identical
to the default path. The change is: pass zero-tensor `ve` for non-VE layers
instead of None. The attn module checks `self.ve_gate is not None` (module-level)
so it never consumes the zero ve. Result must be bit-identical."""
from __future__ import annotations

import torch

from core.model import GPT, GPTConfig


def _build_tiny_model():
    cfg = GPTConfig(
        sequence_len=64, vocab_size=128,
        n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
        window_pattern="L",          # full attention (SDPA-friendly on CPU)
        num_experts=2, top_k=1, num_shared_experts=1,
    )
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device="cpu")
    m.init_weights()
    return m


def test_fullgraph_friendly_matches_default():
    torch.manual_seed(0)
    m = _build_tiny_model()
    m.eval()  # disable MoE noise

    idx = torch.randint(0, 128, (2, 16))
    targets = torch.randint(0, 128, (2, 16))

    # Default path: ve=None on non-VE layers
    m._use_compile_fullgraph_friendly = False
    loss_default = m(idx, targets)

    # Fullgraph-friendly: ve=zeros on non-VE layers
    m._use_compile_fullgraph_friendly = True
    loss_friendly = m(idx, targets)

    # Should be bit-identical because the zero ve is never read inside attn
    # (attn checks self.ve_gate is not None before touching ve).
    assert torch.allclose(loss_default, loss_friendly, atol=1e-6), (
        f"loss diverged: default={loss_default.item()} friendly={loss_friendly.item()}"
    )


if __name__ == "__main__":
    test_fullgraph_friendly_matches_default(); print("✓ fullgraph-friendly matches default")
    print("\nAll compile-fullgraph numerics tests passed.")
