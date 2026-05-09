"""Verify the FP32 master-moments refactor of core/optim.py:

1. exp_avg / exp_avg_sq / momentum_buffer / second_momentum_buffer are FP32
   regardless of param dtype.
2. Optimizer step works correctly when params include both FP32 (matrix)
   and BF16 (embedding-style cast) tensors.
3. Loss decreases over a few steps on a tiny problem (sanity).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from core.optim import MuonAdamW


def _build_param_groups(params_fp32: list, params_bf16: list, params_muon: list):
    """Mirror the structure setup_optimizer would build."""
    groups = []
    if params_fp32:
        groups.append(dict(kind='adamw', params=params_fp32, lr=1e-3,
                           betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
    if params_bf16:
        groups.append(dict(kind='adamw', params=params_bf16, lr=1e-3,
                           betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
    if params_muon:
        groups.append(dict(kind='muon', params=params_muon, lr=1e-3,
                           momentum=0.95, ns_steps=5, beta2=0.95,
                           weight_decay=0.0))
    return groups


def test_fp32_moments_for_bf16_params():
    """exp_avg should be FP32 even when param is BF16."""
    p_bf16 = torch.zeros(64, 32, dtype=torch.bfloat16, requires_grad=True)
    p_bf16.grad = torch.randn_like(p_bf16)
    opt = MuonAdamW(_build_param_groups([], [p_bf16], []))
    for g in opt.param_groups:
        for k in ['initial_lr']:
            g.setdefault(k, g.get('lr', 1e-3))
    opt.step()
    state = opt.state[p_bf16]
    assert state['exp_avg'].dtype == torch.float32, state['exp_avg'].dtype
    assert state['exp_avg_sq'].dtype == torch.float32, state['exp_avg_sq'].dtype
    # Param itself stays BF16
    assert p_bf16.dtype == torch.bfloat16


def test_fp32_moments_for_fp32_params():
    """Same FP32 buffers when param is FP32 — no surprise."""
    p_fp32 = torch.zeros(64, 32, dtype=torch.float32, requires_grad=True)
    p_fp32.grad = torch.randn_like(p_fp32)
    opt = MuonAdamW(_build_param_groups([p_fp32], [], []))
    for g in opt.param_groups:
        g.setdefault('initial_lr', g.get('lr', 1e-3))
    opt.step()
    state = opt.state[p_fp32]
    assert state['exp_avg'].dtype == torch.float32
    assert state['exp_avg_sq'].dtype == torch.float32


def test_muon_fp32_buffers():
    """Muon momentum_buffer + second_momentum_buffer must be FP32."""
    p_muon = torch.zeros(8, 16, dtype=torch.float32, requires_grad=True)
    p_muon.grad = torch.randn_like(p_muon) * 0.01
    opt = MuonAdamW(_build_param_groups([], [], [p_muon]))
    for g in opt.param_groups:
        g.setdefault('initial_lr', g.get('lr', 1e-3))
    opt.step()
    state = opt.state[p_muon]
    assert state['momentum_buffer'].dtype == torch.float32
    assert state['second_momentum_buffer'].dtype == torch.float32


def test_loss_decreases_mixed_dtype():
    """Smoke: a tiny linear regression with mixed-dtype params actually trains."""
    torch.manual_seed(0)
    # Two parameters: an FP32 matrix and a BF16 bias-like tensor.
    W = torch.randn(8, 4, requires_grad=True)               # FP32 matrix
    b = torch.zeros(4, dtype=torch.bfloat16, requires_grad=True)  # BF16 vector
    target = torch.randn(16, 4)
    x = torch.randn(16, 8)

    # Use a higher LR so we see clear loss decrease in a few steps.
    groups = _build_param_groups([b], [], [W])
    for g in groups:
        g['lr'] = 1e-1
    opt = MuonAdamW(groups)
    for g in opt.param_groups:
        g.setdefault('initial_lr', g.get('lr', 1e-1))

    losses = []
    for _ in range(50):
        opt.zero_grad()
        pred = x @ W + b.float()
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # Smoke: loss should clearly drop. Threshold loose to avoid flakiness on different
    # PyTorch versions / random init drift — we're just checking that mixed dtype
    # doesn't break optimizer convergence, not measuring rate.
    assert losses[-1] < losses[0] * 0.7, f"loss did not decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"


if __name__ == "__main__":
    test_fp32_moments_for_bf16_params(); print("✓ FP32 moments for BF16 params")
    test_fp32_moments_for_fp32_params(); print("✓ FP32 moments for FP32 params")
    test_muon_fp32_buffers();             print("✓ Muon FP32 momentum + second buffers")
    test_loss_decreases_mixed_dtype();    print("✓ Mixed-dtype training reduces loss")
    print("\nAll FP32 master-moments tests passed.")
