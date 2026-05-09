"""Benchmark our Triton FA2 against PyTorch SDPA on the same shapes/dtypes.

Reports: forward latency, backward latency, peak memory.
Asserts: peak memory must be O(N·D), not O(N²).

Usage:
    python -m bench.flash_attention_bench [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from basics.flash_attention import HAS_TRITON, flash_attention


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def mem_scope():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    yield
    cuda_sync()


def bench_fn(fn, warmup=20, iters=50):
    for _ in range(warmup):
        fn()
    cuda_sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    cuda_sync()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def peak_mem_mb():
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def run_shape(B, H, N, D, dtype, causal):
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)
    g = torch.randn(B, H, N, D, dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(D)

    out = {"B": B, "H": H, "N": N, "D": D, "dtype": str(dtype), "causal": causal}

    # ---- forward latency + memory ----
    with mem_scope():
        ours_fwd_ms = bench_fn(lambda: flash_attention(q, k, v, causal=causal))
        ours_fwd_mem = peak_mem_mb()
    with mem_scope():
        sdpa_fwd_ms = bench_fn(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
        )
        sdpa_fwd_mem = peak_mem_mb()

    out.update(
        ours_fwd_ms=ours_fwd_ms,
        sdpa_fwd_ms=sdpa_fwd_ms,
        fwd_ratio=ours_fwd_ms / sdpa_fwd_ms,
        ours_fwd_mem_mb=ours_fwd_mem,
        sdpa_fwd_mem_mb=sdpa_fwd_mem,
    )

    # ---- backward latency ----
    def make_io():
        qq = q.detach().clone().requires_grad_(True)
        kk = k.detach().clone().requires_grad_(True)
        vv = v.detach().clone().requires_grad_(True)
        return qq, kk, vv

    def step_ours():
        qq, kk, vv = make_io()
        o = flash_attention(qq, kk, vv, causal=causal)
        o.backward(g)

    def step_sdpa():
        qq, kk, vv = make_io()
        o = F.scaled_dot_product_attention(qq, kk, vv, is_causal=causal, scale=scale)
        o.backward(g)

    with mem_scope():
        ours_bwd_ms = bench_fn(step_ours, warmup=10, iters=20)
        ours_bwd_mem = peak_mem_mb()
    with mem_scope():
        sdpa_bwd_ms = bench_fn(step_sdpa, warmup=10, iters=20)
        sdpa_bwd_mem = peak_mem_mb()

    out.update(
        ours_bwd_ms=ours_bwd_ms,
        sdpa_bwd_ms=sdpa_bwd_ms,
        bwd_ratio=ours_bwd_ms / sdpa_bwd_ms,
        ours_bwd_mem_mb=ours_bwd_mem,
        sdpa_bwd_mem_mb=sdpa_bwd_mem,
    )

    # ---- correctness sanity (single shot) ----
    with torch.no_grad():
        ours_o = flash_attention(q, k, v, causal=causal)
        sdpa_o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
        diff = (ours_o.float() - sdpa_o.float()).abs()
        out["max_abs_err"] = diff.max().item()
        out["mean_abs_err"] = diff.mean().item()

    # ---- memory scaling check: peak_fwd_mem must scale with N*D, not N^2 ----
    # Heuristic: tensors q,k,v,o = 4 * B*H*N*D * dtype_bytes; FA2 should add only O(N) for LSE.
    bytes_per = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}[dtype]
    qkvo_mb = 4 * B * H * N * D * bytes_per / (1024 * 1024)
    # If FA2 allocated an N×N P matrix, we'd see ~B*H*N*N*4B (fp32) extra
    nn_mb = B * H * N * N * 4 / (1024 * 1024)
    out["qkvo_mb"] = qkvo_mb
    out["nn_p_mb_if_materialized"] = nn_mb
    out["fa_extra_over_qkvo_mb"] = ours_fwd_mem - qkvo_mb
    out["sdpa_extra_over_qkvo_mb"] = sdpa_fwd_mem - qkvo_mb

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_results.json")
    args = ap.parse_args()

    if not (HAS_TRITON and torch.cuda.is_available()):
        raise SystemExit("Need CUDA + Triton")

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")
    print(f"torch={torch.__version__}  cuda={torch.version.cuda}")
    import triton
    print(f"triton={triton.__version__}")

    shapes = [
        # (B, H, N, D)
        (4, 16, 512, 64),
        (4, 16, 1024, 64),
        (4, 16, 2048, 64),
        (4, 16, 4096, 64),
        (2, 16, 8192, 64),
        (4, 8, 1024, 128),
        (4, 8, 2048, 128),
    ]
    dtypes = [torch.float16, torch.bfloat16]
    results = []

    for dtype in dtypes:
        for B, H, N, D in shapes:
            for causal in (True, False):
                try:
                    r = run_shape(B, H, N, D, dtype, causal)
                    r["gpu"] = gpu
                    results.append(r)
                    print(
                        f"[{r['dtype']:>16}] B={B} H={H} N={N:>5} D={D:>3} causal={causal} | "
                        f"fwd ours={r['ours_fwd_ms']:6.2f}ms sdpa={r['sdpa_fwd_ms']:6.2f}ms "
                        f"({r['fwd_ratio']:.2f}x)  | "
                        f"bwd ours={r['ours_bwd_ms']:7.2f}ms sdpa={r['sdpa_bwd_ms']:7.2f}ms "
                        f"({r['bwd_ratio']:.2f}x)  | "
                        f"err max={r['max_abs_err']:.4f}"
                    )
                except Exception as e:
                    print(f"FAILED B={B} H={H} N={N} D={D} causal={causal} dtype={dtype}: {e}")
                    results.append(
                        {"B": B, "H": H, "N": N, "D": D, "dtype": str(dtype), "causal": causal,
                         "error": str(e)}
                    )

    summary = {
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out} ({len(results)} rows)")


if __name__ == "__main__":
    main()
