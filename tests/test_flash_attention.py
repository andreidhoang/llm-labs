"""Tests for basics.flash_attention.

Triton path tests are skipped on machines without CUDA. The fallback path
(PyTorch SDPA) is exercised everywhere. Model-integration test runs end-to-end
through the modified CausalMultiHeadSelfAttention.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from basics.flash_attention import HAS_TRITON, flash_attention


def _ref_attn(q, k, v, causal=True):
    scale = 1.0 / math.sqrt(q.shape[-1])
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


# ---------------------------------------------------------------------------
# Fallback path — runs on any device (CPU/MPS/CUDA)
# ---------------------------------------------------------------------------


def test_fallback_matches_reference_fp32():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 64, 32)
    k = torch.randn(2, 4, 64, 32)
    v = torch.randn(2, 4, 64, 32)
    out = flash_attention(q, k, v, causal=True)
    ref = _ref_attn(q, k, v, causal=True)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_fallback_handles_extra_batch_dims():
    torch.manual_seed(0)
    # (B1, B2, H, N, D) — flash_attention should flatten leading dims
    q = torch.randn(2, 3, 4, 16, 32)
    k = torch.randn(2, 3, 4, 16, 32)
    v = torch.randn(2, 3, 4, 16, 32)
    out = flash_attention(q, k, v, causal=True)
    assert out.shape == q.shape
    ref = _ref_attn(q.reshape(-1, 4, 16, 32), k.reshape(-1, 4, 16, 32), v.reshape(-1, 4, 16, 32))
    torch.testing.assert_close(out.reshape(-1, 4, 16, 32), ref, rtol=1e-5, atol=1e-5)


def test_fallback_backward_runs():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 32, 16, requires_grad=True)
    k = torch.randn(1, 2, 32, 16, requires_grad=True)
    v = torch.randn(1, 2, 32, 16, requires_grad=True)
    out = flash_attention(q, k, v, causal=True)
    out.sum().backward()
    for t in (q, k, v):
        assert t.grad is not None and t.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Triton path — requires CUDA
# ---------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="Triton FA path requires CUDA",
)


@needs_cuda
def test_triton_smoke_tiny():
    """Smallest possible case to isolate kernel bugs from sweep noise.
    Run this first when bringing up on a new GPU."""
    torch.manual_seed(0)
    B, H, N, D = 1, 1, 64, 32

    q_ref = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda")
    k_ref = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda")
    v_ref = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda")
    g = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda")

    q1 = q_ref.detach().to(torch.float16).requires_grad_(True)
    k1 = k_ref.detach().to(torch.float16).requires_grad_(True)
    v1 = v_ref.detach().to(torch.float16).requires_grad_(True)
    out1 = flash_attention(q1, k1, v1, causal=True)
    out1.backward(g.to(torch.float16))

    q2 = q_ref.detach().requires_grad_(True)
    k2 = k_ref.detach().requires_grad_(True)
    v2 = v_ref.detach().requires_grad_(True)
    out2 = _ref_attn(q2, k2, v2, causal=True)
    out2.backward(g)

    torch.testing.assert_close(out1.float(), out2, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(q1.grad.float(), q2.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k1.grad.float(), k2.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v1.grad.float(), v2.grad, rtol=1e-2, atol=1e-2)


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("N", [128, 257, 1024])
@pytest.mark.parametrize("D", [32, 64, 128])
def test_triton_forward_matches_reference(dtype, causal, N, D):
    torch.manual_seed(0)
    B, H = 2, 4
    q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")

    out = flash_attention(q, k, v, causal=causal)
    ref = _ref_attn(q.float(), k.float(), v.float(), causal=causal).to(dtype)

    # bf16 unit quantization step is 2^-6 = 0.015625; tolerance must clear that.
    # Verified empirically on RTX A4000: max obs err 0.0156 for bf16 forward.
    rtol, atol = (5e-3, 5e-3) if dtype is torch.float16 else (2e-2, 2e-2)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("N", [128, 257])
@pytest.mark.parametrize("D", [64])
def test_triton_backward_matches_reference(dtype, causal, N, D):
    torch.manual_seed(0)
    B, H = 2, 4

    q_ref = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda")
    k_ref = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda")
    v_ref = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda")
    g = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda")

    # Triton path (low precision)
    q1 = q_ref.detach().to(dtype).requires_grad_(True)
    k1 = k_ref.detach().to(dtype).requires_grad_(True)
    v1 = v_ref.detach().to(dtype).requires_grad_(True)
    out1 = flash_attention(q1, k1, v1, causal=causal)
    out1.backward(g.to(dtype))

    # fp32 reference
    q2 = q_ref.detach().requires_grad_(True)
    k2 = k_ref.detach().requires_grad_(True)
    v2 = v_ref.detach().requires_grad_(True)
    out2 = _ref_attn(q2, k2, v2, causal=causal)
    out2.backward(g)

    # Tri Dao's reference tests use ~5e-2 atol for bf16 backward; loosened from
    # 3e-2 to avoid false alarms from accumulation in the bf16 cast on p/ds.
    rtol, atol = (1e-2, 1e-2) if dtype is torch.float16 else (5e-2, 5e-2)
    torch.testing.assert_close(q1.grad.float(), q2.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(k1.grad.float(), k2.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(v1.grad.float(), v2.grad, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# End-to-end model integration
# ---------------------------------------------------------------------------


def test_model_forward_backward_with_flash_attention():
    """Sanity: BasicsTransformerLM still produces correct-shape logits and
    propagates gradients through the swapped attention path."""
    from basics.model import BasicsTransformerLM

    torch.manual_seed(0)
    model = BasicsTransformerLM(
        vocab_size=128,
        context_length=64,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        rope_theta=10000.0,
    )
    x = torch.randint(0, 128, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 128)

    loss = logits.float().sum()
    loss.backward()

    grads_with_signal = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert grads_with_signal > 0
