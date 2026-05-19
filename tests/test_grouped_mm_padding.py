"""Verify _run_experts_grouped_mm with padding matches the for-loop fallback
on numerics. Tests on CPU (where _grouped_mm isn't available — but we can
exercise the padding LOGIC by mocking the kernel call) AND, when CUDA is
available, the actual end-to-end path.

The padding wrapper:
  1. Pads each expert's chunk to multiple of 8 with zero rows
  2. Calls torch._grouped_mm on the padded data
  3. Slices off padding
must produce the same output as a per-expert for-loop on the original data.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def for_loop_reference(w_up, w_down, x, num_tokens_per_expert):
    """Per-expert for-loop reference impl (matches _run_experts_for_loop)."""
    counts = num_tokens_per_expert.tolist()
    chunks = torch.split(x, [int(c) for c in counts], dim=0)
    w_up_cast = w_up.to(dtype=x.dtype)
    w_down_cast = w_down.to(dtype=x.dtype)
    outs = []
    for i, chunk in enumerate(chunks):
        h = chunk @ w_up_cast[i].T
        h = F.relu(h).square()
        h = h @ w_down_cast[i].T
        outs.append(h)
    return torch.cat(outs, dim=0)


def test_padding_logic_correct_indices():
    """Pure-tensor test: verify the index math used inside the padding wrapper
    correctly maps original tokens to their padded positions and back."""
    from core.moe import _GROUPED_MM_ALIGN
    A = _GROUPED_MM_ALIGN  # 8

    torch.manual_seed(0)
    # Simulate routing: 4 experts, varied counts, none multiple of 8
    num_tokens_per_expert = torch.tensor([5, 3, 17, 2], dtype=torch.int64)
    T = num_tokens_per_expert.sum().item()  # 27
    D = 16

    # Build x with each token's value = its global index (so we can verify roundtrip)
    x = torch.arange(T, dtype=torch.float32).unsqueeze(1).expand(T, D).contiguous()

    # Replicate the padding logic
    pad_per_expert = (-num_tokens_per_expert) % A   # [3, 5, 7, 6]
    padded_n = num_tokens_per_expert + pad_per_expert  # [8, 8, 24, 8]
    assert torch.all(padded_n % A == 0)
    T_padded = (T + pad_per_expert.sum()).item()    # 27 + 21 = 48

    orig_cum = torch.cumsum(num_tokens_per_expert, dim=0)
    cum_pad_before = torch.cumsum(pad_per_expert, dim=0) - pad_per_expert
    token_idx = torch.arange(T, dtype=orig_cum.dtype)
    expert_id = torch.searchsorted(orig_cum, token_idx, right=True)
    target_idx = token_idx + cum_pad_before[expert_id]

    # Verify each original token lands at the right padded slot:
    # First expert (5 tokens) → positions 0..4 (rest 5..7 are padding)
    # Second expert (3 tokens) → positions 8..10 (pad 11..15)
    # etc.
    expected = torch.tensor([
        0, 1, 2, 3, 4,                              # expert 0
        8, 9, 10,                                   # expert 1
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,  # expert 2
        40, 41,                                     # expert 3
    ], dtype=target_idx.dtype)
    assert torch.equal(target_idx, expected), f"target_idx mismatch:\n  got {target_idx.tolist()}\n  exp {expected.tolist()}"

    # Build padded buffer and verify roundtrip
    x_padded = torch.zeros(T_padded, D)
    x_padded.index_copy_(0, target_idx, x)
    x_recovered = x_padded.index_select(0, target_idx)
    assert torch.equal(x_recovered, x), "roundtrip via index_copy + index_select failed"


def test_grouped_mm_eager_allocates_on_cpu_via_mock():
    """CPU-runnable regression guard for the _run_experts_grouped_mm eager path.

    Background: the 2026-05-19 T_padded bug. `new_zeros(T_padded, D)` was called
    with a 0-D Tensor T_padded. NGC's torch 2.9.x rejected this in eager mode
    (`TypeError: argument 'size' must be tuple of ints`). torch.compile masked
    it via tracer specialization; modern torch (≥2.11) silently coerces.

    This test mocks `torch._grouped_mm` to a simple loop so the full eager path
    (padding math + new_zeros allocation + index_copy + grouped_mm + index_select)
    executes on CPU without CUDA. It serves two regression guards:
      1. End-to-end shape/dtype invariants (catches any future change that
         breaks the padding wrapper's contract)
      2. Numerical equivalence with the for-loop reference

    It does NOT directly assert int-vs-Tensor (because modern torch coerces).
    The actual fix (refactor A in core/moe.py) is version-independent code.
    """
    if not hasattr(torch, "_grouped_mm"):
        print("(skip — torch._grouped_mm attribute not available)")
        return
    from core.moe import _run_experts_grouped_mm

    # Mock _grouped_mm: a, b are 2D padded blocks; offs gives per-expert end-offsets.
    # Mimic the actual semantics: for each expert i, a[start:end] @ b[i] = out[start:end].
    original = torch._grouped_mm
    def mock_grouped_mm(a, b, offs=None):
        out_chunks = []
        start = 0
        for i, end in enumerate(offs.tolist()):
            out_chunks.append(a[start:end] @ b[i])
            start = end
        return torch.cat(out_chunks, dim=0)

    torch._grouped_mm = mock_grouped_mm
    try:
        torch.manual_seed(0)
        E = 4
        D = 16
        H = 32
        # Use non-multiple-of-8 counts so padding is non-trivial
        num_tokens_per_expert = torch.tensor([5, 3, 11, 2], dtype=torch.int64)
        T = num_tokens_per_expert.sum().item()
        x = torch.randn(T, D, dtype=torch.bfloat16)
        w_up = torch.randn(E, H, D, dtype=torch.float32) * 0.02
        w_down = torch.randn(E, D, H, dtype=torch.float32) * 0.02

        # This MUST NOT raise TypeError on the new_zeros allocation
        out = _run_experts_grouped_mm(w_up, w_down, x, num_tokens_per_expert)
        assert out.shape == (T, D)
        assert out.dtype == x.dtype

        # Numerical sanity: matches the for-loop reference
        out_loop = for_loop_reference(w_up, w_down, x, num_tokens_per_expert)
        diff = (out.float() - out_loop.float()).abs().max().item()
        assert diff < 1e-2, f"mock-grouped_mm vs for-loop diff {diff} too large"
    finally:
        torch._grouped_mm = original


def test_grouped_mm_matches_for_loop_on_cuda():
    """End-to-end equivalence check on CUDA when _grouped_mm is available."""
    if not torch.cuda.is_available():
        print("(skip — no CUDA)")
        return
    if not hasattr(torch, "_grouped_mm"):
        print("(skip — torch._grouped_mm not available on this torch)")
        return
    from core.moe import _run_experts_grouped_mm

    torch.manual_seed(0)
    E = 4
    D = 64
    H = 128
    num_tokens_per_expert = torch.tensor([13, 7, 21, 3], device="cuda", dtype=torch.int64)
    T = num_tokens_per_expert.sum().item()

    x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
    w_up = torch.randn(E, H, D, device="cuda", dtype=torch.float32) * 0.02
    w_down = torch.randn(E, D, H, device="cuda", dtype=torch.float32) * 0.02

    out_grouped = _run_experts_grouped_mm(w_up, w_down, x, num_tokens_per_expert)
    out_loop = for_loop_reference(w_up, w_down, x, num_tokens_per_expert)

    assert out_grouped.shape == out_loop.shape == (T, D)
    diff = (out_grouped.float() - out_loop.float()).abs().max().item()
    assert diff < 1e-2, f"grouped_mm vs for-loop max diff {diff} too large"


if __name__ == "__main__":
    test_padding_logic_correct_indices()
    print("✓ padding index math correct (CPU-only)")
    test_grouped_mm_eager_allocates_on_cpu_via_mock()
    print("✓ eager allocation path works (CPU regression guard for T_padded bug)")
    test_grouped_mm_matches_for_loop_on_cuda()
    print("✓ grouped_mm matches for-loop (or skipped on CPU)")
    print("\nAll grouped_mm padding tests passed.")
