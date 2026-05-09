#!/usr/bin/env python3
"""
Problems (pytorch_attention) + (torch_compile): PyTorch Attention Benchmarking

Benchmarks scaled_dot_product_attention — vanilla vs torch.compile — across:
  - d_model in [16, 32, 64, 128]
  - seq_len  in [256, 1024, 4096, 8192, 16384]
  - batch_size fixed at 8, no head dimension

Measures:
  - Forward pass latency (ms)
  - Memory before backward (MiB)
  - Backward pass latency (ms)

Usage:
  uv run python bench/pytorch_attention_benchmark.py
  uv run python bench/pytorch_attention_benchmark.py --num_warmup 5 --num_measure 100
  uv run python bench/pytorch_attention_benchmark.py --accounting_only
  uv run python bench/pytorch_attention_benchmark.py --no_compile   # skip torch.compile
  uv run python bench/pytorch_attention_benchmark.py --output results/pytorch_attention_benchmark.csv
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basics.model import scaled_dot_product_attention

BATCH_SIZE = 8
D_MODEL_SIZES = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]

OOM = "OOM"

AttnFn = Callable[..., torch.Tensor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync() -> None:
    torch.cuda.synchronize()


def _make_qkv(seq_len: int, d: int, device: torch.device, requires_grad: bool = False):
    shape = (BATCH_SIZE, seq_len, d)
    Q = torch.randn(shape, device=device, requires_grad=requires_grad)
    K = torch.randn(shape, device=device, requires_grad=requires_grad)
    V = torch.randn(shape, device=device, requires_grad=requires_grad)
    return Q, K, V


# ---------------------------------------------------------------------------
# Timing functions — accept the attention fn as a parameter
# ---------------------------------------------------------------------------

def time_forward(attn_fn: AttnFn, seq_len: int, d: int, device: torch.device,
                 num_warmup: int, num_measure: int) -> float | str:
    """Return mean forward latency in ms, or OOM."""
    Q, K, V = _make_qkv(seq_len, d, device)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    try:
        for _ in range(num_warmup):
            attn_fn(Q, K, V)
            _sync()

        total_ms = 0.0
        for _ in range(num_measure):
            start.record()
            attn_fn(Q, K, V)
            end.record()
            _sync()
            total_ms += start.elapsed_time(end)

        return total_ms / num_measure
    except torch.cuda.OutOfMemoryError:
        return OOM
    finally:
        del Q, K, V
        torch.cuda.empty_cache()


def time_backward_and_memory(attn_fn: AttnFn, seq_len: int, d: int, device: torch.device,
                              num_warmup: int, num_measure: int) -> tuple[float | str, float | str]:
    """
    Returns (mean_backward_ms, memory_before_bwd_MiB).

    Memory is sampled after one live forward pass (graph still alive).
    Backward is timed with CUDA events; graph is rebuilt each iteration
    to avoid OOM from retain_graph=True.
    """
    def _fwd():
        Q, K, V = _make_qkv(seq_len, d, device, requires_grad=True)
        out = attn_fn(Q, K, V)
        return out, Q, K, V

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    try:
        # Warmup — important for torch.compile to trigger tracing
        for _ in range(num_warmup):
            out, Q, K, V = _fwd()
            _sync()
            out.sum().backward()
            _sync()
            del out, Q, K, V

        # --- memory sample: one live forward, graph intact ---
        torch.cuda.reset_peak_memory_stats(device)
        out, Q, K, V = _fwd()
        _sync()
        mem_mib = torch.cuda.memory_allocated(device) / (1024 ** 2)
        del out, Q, K, V

        # --- timed backward runs ---
        total_ms = 0.0
        for _ in range(num_measure):
            out, Q, K, V = _fwd()
            loss = out.sum()
            _sync()

            start.record()
            loss.backward()
            end.record()
            _sync()

            total_ms += start.elapsed_time(end)
            del out, Q, K, V, loss

        return total_ms / num_measure, mem_mib

    except torch.cuda.OutOfMemoryError:
        return OOM, OOM
    finally:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Single-backend benchmark loop
# ---------------------------------------------------------------------------

def benchmark_one_backend(
    label: str,
    attn_fn: AttnFn,
    device: torch.device,
    num_warmup: int,
    num_measure: int,
) -> list[dict[str, Any]]:
    rows = []
    for d in D_MODEL_SIZES:
        for seq_len in SEQ_LENS:
            tag = f"[{label}] d={d:>3}, seq={seq_len:>6}"
            print(f"  {tag} ...", end=" ", flush=True)
            torch.cuda.empty_cache()

            fwd = time_forward(attn_fn, seq_len, d, device, num_warmup, num_measure)
            bwd, mem = time_backward_and_memory(attn_fn, seq_len, d, device, num_warmup, num_measure)

            fwd_s = OOM if fwd == OOM else f"{fwd:.2f}"
            bwd_s = OOM if bwd == OOM else f"{bwd:.2f}"
            mem_s = OOM if mem == OOM else f"{mem:.1f}"
            print(f"fwd={fwd_s}ms  mem={mem_s}MiB  bwd={bwd_s}ms")

            rows.append({
                "backend": label,
                "d": d,
                "seq_len": seq_len,
                "fwd_ms": fwd_s,
                "mem_MiB": mem_s,
                "bwd_ms": bwd_s,
                # keep raw floats for speedup calculation
                "_fwd": fwd,
                "_bwd": bwd,
            })
    return rows


# ---------------------------------------------------------------------------
# Comparison table builder
# ---------------------------------------------------------------------------

def make_comparison_table(vanilla_rows: list[dict], compiled_rows: list[dict]) -> pd.DataFrame:
    """Side-by-side vanilla vs compiled with speedup columns."""
    v = {(r["d"], r["seq_len"]): r for r in vanilla_rows}
    c = {(r["d"], r["seq_len"]): r for r in compiled_rows}

    out = []
    for key in v:
        vr, cr = v[key], c[key]

        def _speedup(raw_v, raw_c):
            if raw_v == OOM or raw_c == OOM:
                return "—"
            return f"{raw_v / raw_c:.2f}x"

        out.append({
            "d": vr["d"],
            "seq_len": vr["seq_len"],
            "fwd vanilla (ms)": vr["fwd_ms"],
            "fwd compiled (ms)": cr["fwd_ms"],
            "fwd speedup": _speedup(vr["_fwd"], cr["_fwd"]),
            "mem vanilla (MiB)": vr["mem_MiB"],
            "mem compiled (MiB)": cr["mem_MiB"],
            "bwd vanilla (ms)": vr["bwd_ms"],
            "bwd compiled (ms)": cr["bwd_ms"],
            "bwd speedup": _speedup(vr["_bwd"], cr["_bwd"]),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Memory accounting (theoretical)
# ---------------------------------------------------------------------------

def memory_accounting_table() -> pd.DataFrame:
    rows = []
    for d in D_MODEL_SIZES:
        for seq_len in SEQ_LENS:
            B, dtype = BATCH_SIZE, 4  # FP32
            qkv_mib   = 3 * B * seq_len * d * dtype / (1024 ** 2)
            scores_mib = B * seq_len * seq_len * dtype / (1024 ** 2)
            out_mib    = B * seq_len * d * dtype / (1024 ** 2)
            # PyTorch saves P (softmax output) + Q,K,V for backward
            saved_mib  = scores_mib + qkv_mib
            rows.append({
                "d": d,
                "seq_len": seq_len,
                "Q+K+V (MiB)": f"{qkv_mib:.2f}",
                "P=softmax(S) (MiB)": f"{scores_mib:.2f}",
                "total saved for bwd (MiB)": f"{saved_mib:.2f}",
            })
    return pd.DataFrame(rows)


def run_metadata(device: torch.device, num_warmup: int, num_measure: int) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_MiB": f"{props.total_memory / (1024 ** 2):.1f}",
        "cuda_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_size": BATCH_SIZE,
        "num_warmup": num_warmup,
        "num_measure": num_measure,
    }


def save_benchmark_rows(rows: list[dict[str, Any]], output: Path, metadata: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for row in rows:
        record = {k: v for k, v in row.items() if not k.startswith("_")}
        record.update(metadata)
        records.append(record)

    df = pd.DataFrame(records)
    append = output.exists() and output.stat().st_size > 0
    df.to_csv(output, mode="a", header=not append, index=False)
    print(f"\nSaved {len(df)} benchmark rows to {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_warmup",     type=int,  default=5)
    parser.add_argument("--num_measure",    type=int,  default=100)
    parser.add_argument("--accounting_only", action="store_true")
    parser.add_argument("--no_compile",     action="store_true", help="Skip torch.compile variant")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pytorch_attention_benchmark.csv"),
        help="CSV file to append benchmark rows to.",
    )
    parser.add_argument("--no_output", action="store_true", help="Print results without writing CSV output.")
    args = parser.parse_args()

    # --- Memory accounting ---
    print("=" * 70)
    print("THEORETICAL MEMORY ACCOUNTING  (FP32, batch=8)")
    print("=" * 70)
    print(memory_accounting_table().to_string(index=False))
    print()

    if args.accounting_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"warmup={args.num_warmup}  measure={args.num_measure}\n")
    metadata = run_metadata(device, args.num_warmup, args.num_measure)

    # --- Vanilla ---
    print("=" * 70)
    print("VANILLA  (uncompiled)")
    print("=" * 70)
    vanilla_rows = benchmark_one_backend(
        "vanilla", scaled_dot_product_attention, device, args.num_warmup, args.num_measure
    )

    if args.no_compile:
        df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in vanilla_rows])
        print("\n", df.to_string(index=False))
        if not args.no_output:
            save_benchmark_rows(vanilla_rows, args.output, metadata)
        return

    # --- torch.compile ---
    # torch.compile traces on the first call → warmup must be >= 1.
    # Use the same num_warmup (already >= 1 by default).
    compiled_attn = torch.compile(scaled_dot_product_attention)

    print("=" * 70)
    print("COMPILED  (torch.compile)")
    print("=" * 70)
    compiled_rows = benchmark_one_backend(
        "compiled", compiled_attn, device, max(args.num_warmup, 3), args.num_measure
    )
    all_rows = vanilla_rows + compiled_rows

    # --- Comparison table ---
    print("\n" + "=" * 70)
    print("COMPARISON: vanilla vs torch.compile")
    print("=" * 70)
    comparison = make_comparison_table(vanilla_rows, compiled_rows)
    print(comparison.to_string(index=False))

    # Compact pivot: forward speedup by (seq_len × d)
    print("\n--- Forward speedup (compiled / vanilla, >1 = compiled faster) ---")
    try:
        pivot = comparison[["d", "seq_len", "fwd speedup"]].pivot(index="seq_len", columns="d", values="fwd speedup")
        print(pivot.to_string())
    except Exception:
        pass

    print("\n--- Backward speedup ---")
    try:
        pivot = comparison[["d", "seq_len", "bwd speedup"]].pivot(index="seq_len", columns="d", values="bwd speedup")
        print(pivot.to_string())
    except Exception:
        pass

    if not args.no_output:
        save_benchmark_rows(all_rows, args.output, metadata)


if __name__ == "__main__":
    main()
