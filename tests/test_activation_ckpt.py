"""Activation checkpointing must give identical forward loss and matching gradients
(within FP32 reorder noise) compared to no-checkpointing path."""
from __future__ import annotations

import torch

from core.model import GPT, GPTConfig


def _build_tiny_model():
    cfg = GPTConfig(
        sequence_len=64, vocab_size=128,
        n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
        window_pattern="L",
        num_experts=2, top_k=1, num_shared_experts=1,
    )
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device="cpu")
    m.init_weights()
    return m


def test_act_ckpt_forward_identical():
    torch.manual_seed(0)
    m = _build_tiny_model()
    m.train()
    idx = torch.randint(0, 128, (2, 16))
    targets = torch.randint(0, 128, (2, 16))

    m._use_activation_checkpointing = False
    loss_no_ckpt = m(idx, targets)

    m._use_activation_checkpointing = True
    loss_ckpt = m(idx, targets)

    assert torch.allclose(loss_no_ckpt, loss_ckpt, atol=1e-6), (
        f"forward diverged: no_ckpt={loss_no_ckpt.item()} ckpt={loss_ckpt.item()}"
    )


def test_act_ckpt_grads_match():
    """Gradients must match within FP32 noise (~1e-5 since recomputation is deterministic)."""
    torch.manual_seed(0)

    # Build two identical models with same init
    m1 = _build_tiny_model()
    m2 = _build_tiny_model()
    m2.load_state_dict(m1.state_dict())
    m1.train(); m2.train()

    idx = torch.randint(0, 128, (2, 16))
    targets = torch.randint(0, 128, (2, 16))

    m1._use_activation_checkpointing = False
    loss1 = m1(idx, targets); loss1.backward()

    m2._use_activation_checkpointing = True
    loss2 = m2(idx, targets); loss2.backward()

    assert torch.allclose(loss1, loss2, atol=1e-6)
    # Compare grads on a representative parameter
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if p1.grad is None and p2.grad is None:
            continue
        assert n1 == n2
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-4), (
            f"grad mismatch on {n1}: max diff {(p1.grad - p2.grad).abs().max().item()}"
        )


def test_act_ckpt_inference_path_disabled():
    """When in eval mode, checkpointing must NOT trigger (would break, kv_cache path)."""
    m = _build_tiny_model()
    m.eval()
    m._use_activation_checkpointing = True
    idx = torch.randint(0, 128, (1, 8))
    # Should run inference without raising — no targets so returns logits
    logits = m(idx, targets=None)
    assert logits.shape == (1, 8, m.config.vocab_size)


if __name__ == "__main__":
    test_act_ckpt_forward_identical(); print("✓ act ckpt forward identical")
    test_act_ckpt_grads_match();       print("✓ act ckpt grads match")
    test_act_ckpt_inference_path_disabled(); print("✓ act ckpt skipped in inference")
    print("\nAll activation checkpointing tests passed.")
