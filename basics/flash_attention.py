"""Flash Attention 2 (forward + backward) in Triton, with PyTorch SDPA fallback.

Forward: tiled, online softmax, saves logsumexp (LSE) for backward.
Backward: two-kernel design (dKdV scans Q tiles, dQ scans KV tiles) — no atomics,
recomputes S and P from Q,K,LSE rather than materializing them in HBM.

Public API:
    flash_attention(q, k, v, causal=True, sm_scale=None) -> out

Inputs are expected as (..., H, N, D); leading batch dims are flattened internally.
On non-CUDA devices or unsupported dtypes/head-dims, dispatches to
torch.nn.functional.scaled_dot_product_attention (which itself selects FA /
mem-efficient / math backend on CUDA).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # Triton not installed (e.g. macOS)
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if HAS_TRITON:

    # Autotune configs: BLOCK_M × BLOCK_N tile sizes + warps + pipeline stages.
    # 6 configs per kernel × 3 hot kernels = up to 18 first-call compilations
    # (~10–20s warmup). Subsequent calls hit the cache and run at peak.
    # Configs chosen for Ampere (SM 8.x) + Ada (SM 8.9) shared-memory budget;
    # Hopper (SM 9.0) would benefit from BLOCK_M=128/BLOCK_N=128 added.
    _AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=3),
    ]

    # N_CTX_Q, N_CTX_K are positional (autotune cache key). HEAD_DIM, IS_CAUSAL,
    # BLOCK_DMODEL are constexpr and already part of JIT specialization, so the
    # kernel cache automatically separates by them — no need to include here.
    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N_CTX_Q", "N_CTX_K"])
    @triton.jit
    def _fwd_kernel(
        Q, K, V, sm_scale,
        L, Out,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_lb, stride_lh,
        N_CTX_Q, N_CTX_K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_b = tl.program_id(1)
        off_h = tl.program_id(2)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_base = Q + off_b * stride_qb + off_h * stride_qh
        k_base = K + off_b * stride_kb + off_h * stride_kh
        v_base = V + off_b * stride_vb + off_h * stride_vh
        o_base = Out + off_b * stride_ob + off_h * stride_oh
        l_base = L + off_b * stride_lb + off_h * stride_lh

        # Load Q tile: [BLOCK_M, BLOCK_DMODEL]
        q_idx = offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
        q_mask = (offs_m[:, None] < N_CTX_Q) & (offs_d[None, :] < HEAD_DIM)
        q = tl.load(q_base + q_idx, mask=q_mask, other=0.0)

        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        if IS_CAUSAL:
            end_n = tl.minimum((start_m + 1) * BLOCK_M, N_CTX_K)
        else:
            end_n = N_CTX_K

        for start_n in range(0, end_n, BLOCK_N):
            kn = start_n + offs_n  # absolute key positions

            # K loaded as [D, N] for direct QK^T via tl.dot(q, k)
            k_idx = offs_d[:, None] * stride_kd + kn[None, :] * stride_kn
            k_mask = (offs_d[:, None] < HEAD_DIM) & (kn[None, :] < N_CTX_K)
            k = tl.load(k_base + k_idx, mask=k_mask, other=0.0)

            qk = tl.dot(q, k) * sm_scale  # fp32 accumulator

            qk = tl.where(kn[None, :] < N_CTX_K, qk, -float("inf"))
            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= kn[None, :], qk, -float("inf"))

            m_ij = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = alpha * l_i + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v_idx = kn[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v_mask = (kn[:, None] < N_CTX_K) & (offs_d[None, :] < HEAD_DIM)
            v = tl.load(v_base + v_idx, mask=v_mask, other=0.0)

            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # Finalize: divide by l, store output and LSE = m + log(l)
        acc = acc / l_i[:, None]
        lse = m_i + tl.log(l_i)

        o_idx = offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
        o_mask = (offs_m[:, None] < N_CTX_Q) & (offs_d[None, :] < HEAD_DIM)
        tl.store(o_base + o_idx, acc.to(Out.dtype.element_ty), mask=o_mask)
        tl.store(l_base + offs_m, lse, mask=offs_m < N_CTX_Q)


    @triton.jit
    def _bwd_preprocess(
        Out, dO, Delta,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_dob, stride_doh, stride_don, stride_dod,
        stride_db, stride_dh,
        N_CTX,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        # Computes Delta_i = sum_j O_ij * dO_ij — needed for the softmax-Jacobian trick
        # so that backward avoids materializing an N x N P matrix.
        start_m = tl.program_id(0)
        off_b = tl.program_id(1)
        off_h = tl.program_id(2)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        o_base = Out + off_b * stride_ob + off_h * stride_oh
        do_base = dO + off_b * stride_dob + off_h * stride_doh
        d_base = Delta + off_b * stride_db + off_h * stride_dh

        m_mask = (offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
        o = tl.load(
            o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
            mask=m_mask, other=0.0,
        ).to(tl.float32)
        do = tl.load(
            do_base + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod,
            mask=m_mask, other=0.0,
        ).to(tl.float32)

        delta = tl.sum(o * do, axis=1)
        tl.store(d_base + offs_m, delta, mask=offs_m < N_CTX)


    # N_CTX_Q, N_CTX_K are positional (autotune cache key). HEAD_DIM, IS_CAUSAL,
    # BLOCK_DMODEL are constexpr and already part of JIT specialization, so the
    # kernel cache automatically separates by them — no need to include here.
    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N_CTX_Q", "N_CTX_K"])
    @triton.jit
    def _bwd_kernel_dkdv(
        Q, K, V, sm_scale,
        dO, dK, dV,
        L, D_,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_dob, stride_doh, stride_don, stride_dod,
        stride_dkb, stride_dkh, stride_dkn, stride_dkd,
        stride_dvb, stride_dvh, stride_dvn, stride_dvd,
        stride_lb, stride_lh,
        stride_db, stride_dh,
        N_CTX_Q, N_CTX_K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        # One program per (batch, head, kv-tile). Iterates over Q tiles.
        start_n = tl.program_id(0)
        off_b = tl.program_id(1)
        off_h = tl.program_id(2)

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_base = Q + off_b * stride_qb + off_h * stride_qh
        k_base = K + off_b * stride_kb + off_h * stride_kh
        v_base = V + off_b * stride_vb + off_h * stride_vh
        do_base = dO + off_b * stride_dob + off_h * stride_doh
        dk_base = dK + off_b * stride_dkb + off_h * stride_dkh
        dv_base = dV + off_b * stride_dvb + off_h * stride_dvh
        l_base = L + off_b * stride_lb + off_h * stride_lh
        d_base = D_ + off_b * stride_db + off_h * stride_dh

        kv_mask = (offs_n[:, None] < N_CTX_K) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=kv_mask, other=0.0,
        )
        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=kv_mask, other=0.0,
        )

        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

        # Under causal: smallest q-tile that can attend to this kv-tile.
        # k_pos_min = start_n*BLOCK_N; we need q_pos >= k_pos_min ⇒ floor to BLOCK_M boundary.
        if IS_CAUSAL:
            q_start = (start_n * BLOCK_N) // BLOCK_M * BLOCK_M
        else:
            q_start = 0

        for q0 in range(q_start, N_CTX_Q, BLOCK_M):
            offs_m = q0 + tl.arange(0, BLOCK_M)
            q_mask = (offs_m[:, None] < N_CTX_Q) & (offs_d[None, :] < HEAD_DIM)

            q = tl.load(
                q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                mask=q_mask, other=0.0,
            )
            do = tl.load(
                do_base + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod,
                mask=q_mask, other=0.0,
            )
            lse = tl.load(l_base + offs_m, mask=offs_m < N_CTX_Q, other=0.0)
            delta = tl.load(d_base + offs_m, mask=offs_m < N_CTX_Q, other=0.0)

            # Recompute S = Q K^T * scale  → [M, N]
            s = tl.dot(q, tl.trans(k)) * sm_scale
            s = tl.where(offs_n[None, :] < N_CTX_K, s, -float("inf"))
            if IS_CAUSAL:
                s = tl.where(offs_m[:, None] >= offs_n[None, :], s, -float("inf"))

            p = tl.exp(s - lse[:, None])
            # Zero out OOB Q rows so they don't contribute
            p = tl.where(offs_m[:, None] < N_CTX_Q, p, 0.0)

            # dV += P^T @ dO  → [N, D]
            dv += tl.dot(tl.trans(p).to(do.dtype), do)

            # dP = dO @ V^T  → [M, N]; dS = P * (dP - delta) * scale
            dp = tl.dot(do, tl.trans(v))
            ds = (p * (dp - delta[:, None])) * sm_scale

            # dK += dS^T @ Q  → [N, D]
            dk += tl.dot(tl.trans(ds).to(q.dtype), q)

        out_mask = (offs_n[:, None] < N_CTX_K) & (offs_d[None, :] < HEAD_DIM)
        tl.store(
            dk_base + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd,
            dk.to(dK.dtype.element_ty), mask=out_mask,
        )
        tl.store(
            dv_base + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd,
            dv.to(dV.dtype.element_ty), mask=out_mask,
        )


    # N_CTX_Q, N_CTX_K are positional (autotune cache key). HEAD_DIM, IS_CAUSAL,
    # BLOCK_DMODEL are constexpr and already part of JIT specialization, so the
    # kernel cache automatically separates by them — no need to include here.
    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N_CTX_Q", "N_CTX_K"])
    @triton.jit
    def _bwd_kernel_dq(
        Q, K, V, sm_scale,
        dO, dQ,
        L, D_,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_dob, stride_doh, stride_don, stride_dod,
        stride_dqb, stride_dqh, stride_dqn, stride_dqd,
        stride_lb, stride_lh,
        stride_db, stride_dh,
        N_CTX_Q, N_CTX_K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        # One program per (batch, head, q-tile). Iterates over KV tiles.
        start_m = tl.program_id(0)
        off_b = tl.program_id(1)
        off_h = tl.program_id(2)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_base = Q + off_b * stride_qb + off_h * stride_qh
        k_base = K + off_b * stride_kb + off_h * stride_kh
        v_base = V + off_b * stride_vb + off_h * stride_vh
        do_base = dO + off_b * stride_dob + off_h * stride_doh
        dq_base = dQ + off_b * stride_dqb + off_h * stride_dqh
        l_base = L + off_b * stride_lb + off_h * stride_lh
        d_base = D_ + off_b * stride_db + off_h * stride_dh

        q_mask = (offs_m[:, None] < N_CTX_Q) & (offs_d[None, :] < HEAD_DIM)
        q = tl.load(
            q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
            mask=q_mask, other=0.0,
        )
        do = tl.load(
            do_base + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod,
            mask=q_mask, other=0.0,
        )
        lse = tl.load(l_base + offs_m, mask=offs_m < N_CTX_Q, other=0.0)
        delta = tl.load(d_base + offs_m, mask=offs_m < N_CTX_Q, other=0.0)

        dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        if IS_CAUSAL:
            end_n = tl.minimum((start_m + 1) * BLOCK_M, N_CTX_K)
        else:
            end_n = N_CTX_K

        for n0 in range(0, end_n, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            kv_mask = (offs_n[:, None] < N_CTX_K) & (offs_d[None, :] < HEAD_DIM)

            k = tl.load(
                k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=kv_mask, other=0.0,
            )
            v = tl.load(
                v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=kv_mask, other=0.0,
            )

            s = tl.dot(q, tl.trans(k)) * sm_scale
            s = tl.where(offs_n[None, :] < N_CTX_K, s, -float("inf"))
            if IS_CAUSAL:
                s = tl.where(offs_m[:, None] >= offs_n[None, :], s, -float("inf"))

            p = tl.exp(s - lse[:, None])
            p = tl.where(offs_m[:, None] < N_CTX_Q, p, 0.0)

            dp = tl.dot(do, tl.trans(v))
            ds = (p * (dp - delta[:, None])) * sm_scale

            dq += tl.dot(ds.to(k.dtype), k)

        tl.store(
            dq_base + offs_m[:, None] * stride_dqn + offs_d[None, :] * stride_dqd,
            dq.to(dQ.dtype.element_ty),
            mask=q_mask,
        )


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------


class _FlashAttention(torch.autograd.Function):
    """Triton FA2 forward+backward, expecting 4-D (B,H,N,D) inputs."""

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        B, H, N_Q, D = q.shape
        N_K = k.shape[2]

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        BLOCK_DMODEL = max(triton.next_power_of_2(D), 16)

        out = torch.empty_like(q)
        # L is freshly allocated and therefore contiguous → stride along N is 1
        # (the kernel assumes this when indexing l_base + offs_m).
        L = torch.empty((B, H, N_Q), device=q.device, dtype=torch.float32)

        # Autotune supplies BLOCK_M, num_warps, num_stages → callable grid.
        grid = lambda meta: (triton.cdiv(N_Q, meta["BLOCK_M"]), B, H)
        _fwd_kernel[grid](
            q, k, v, sm_scale,
            L, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L.stride(0), L.stride(1),
            N_Q, N_K,
            BLOCK_DMODEL=BLOCK_DMODEL, HEAD_DIM=D,
            IS_CAUSAL=causal,
        )

        ctx.save_for_backward(q, k, v, out, L)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.BLOCK_DMODEL = BLOCK_DMODEL
        ctx.head_dim = D
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, L = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        BLOCK_DMODEL = ctx.BLOCK_DMODEL
        D = ctx.head_dim
        B, H, N_Q, _ = q.shape
        N_K = k.shape[2]

        if not do.is_contiguous():
            do = do.contiguous()

        # Preprocess is a tiny elementwise reduction — not worth autotuning.
        BLOCK_M_PRE = 64

        Delta = torch.empty((B, H, N_Q), device=q.device, dtype=torch.float32)
        _bwd_preprocess[(triton.cdiv(N_Q, BLOCK_M_PRE), B, H)](
            out, do, Delta,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            Delta.stride(0), Delta.stride(1),
            N_Q,
            BLOCK_M=BLOCK_M_PRE, BLOCK_DMODEL=BLOCK_DMODEL, HEAD_DIM=D,
            num_warps=4,
        )

        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # Autotuned kernels — block sizes / warps / stages come from the cached
        # winning config for this (N_Q, N_K, HEAD_DIM, IS_CAUSAL).
        grid_kv = lambda meta: (triton.cdiv(N_K, meta["BLOCK_N"]), B, H)
        _bwd_kernel_dkdv[grid_kv](
            q, k, v, sm_scale,
            do, dk, dv,
            L, Delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            L.stride(0), L.stride(1),
            Delta.stride(0), Delta.stride(1),
            N_Q, N_K,
            BLOCK_DMODEL=BLOCK_DMODEL, HEAD_DIM=D,
            IS_CAUSAL=causal,
        )

        grid_q = lambda meta: (triton.cdiv(N_Q, meta["BLOCK_M"]), B, H)
        _bwd_kernel_dq[grid_q](
            q, k, v, sm_scale,
            do, dq,
            L, Delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            L.stride(0), L.stride(1),
            Delta.stride(0), Delta.stride(1),
            N_Q, N_K,
            BLOCK_DMODEL=BLOCK_DMODEL, HEAD_DIM=D,
            IS_CAUSAL=causal,
        )

        return dq, dk, dv, None, None


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def _can_use_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if not HAS_TRITON or not q.is_cuda:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if not (q.dtype == k.dtype == v.dtype):
        return False
    if q.shape != k.shape or q.shape != v.shape:
        return False  # MQA/GQA + cross-attn unsupported in this version
    if q.shape[-1] > 128:
        return False
    return True


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Drop-in causal scaled-dot-product attention.

    Inputs of shape (..., H, N, D). Leading batch dims are flattened internally.
    Returns output with the same shape as q.
    """
    orig_shape = q.shape
    if q.dim() < 4:
        # add singleton leading batch dim
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        added_batch = True
    else:
        added_batch = False

    # Flatten any extra leading dims into one batch dim
    q4 = q.reshape(-1, *q.shape[-3:])
    k4 = k.reshape(-1, *k.shape[-3:])
    v4 = v.reshape(-1, *v.shape[-3:])

    if _can_use_triton(q4, k4, v4):
        if not q4.is_contiguous():
            q4 = q4.contiguous()
        if not k4.is_contiguous():
            k4 = k4.contiguous()
        if not v4.is_contiguous():
            v4 = v4.contiguous()
        out = _FlashAttention.apply(q4, k4, v4, causal, sm_scale)
    else:
        scale = sm_scale if sm_scale is not None else (1.0 / math.sqrt(q4.shape[-1]))
        out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=causal, scale=scale)

    if added_batch:
        out = out.squeeze(0)
        return out.reshape(orig_shape)
    return out.reshape(orig_shape)
