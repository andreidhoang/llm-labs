#!/usr/bin/env python3
"""
Benchmark CS336 Table 1 Transformer model training throughput.

Usage examples:
  # Part (a/b): benchmark all model sizes with proper warmup
  uv run python bench/benchmark.py --mode full_step --all_sizes \
      --num_warmup 5 --num_measure 10

  # Part (c): effect of warmup (no warmup, 1 warmup, 2 warmup)
  uv run python bench/benchmark.py --model_size xl --mode full_step \
      --num_warmup 0 --num_measure 10

  # Flash (FA2/auto) vs annotated unfused math baseline — side-by-side:
  uv run python bench/benchmark.py --model_size small --mode forward_backward \
      --attn_backend all --num_warmup 5 --num_measure 10

  # Single backend (flash | math):
  uv run python bench/benchmark.py --model_size small --mode forward_backward \
      --attn_backend math

  # nsys profiling with NVTX block/attn/mlp annotations.
  # --nvtx-capture filters the trace to measured (non-warmup) steps only:
  nsys profile --trace=cuda,nvtx --nvtx-capture="measured_step_*" -o xl_fwd \
    uv run python bench/benchmark.py --model_size xl --mode forward \
      --context_length 256 --num_warmup 2 --num_measure 1 --nvtx

  # Mixed precision (BF16) for section 2.1.5c:
  uv run python bench/benchmark.py --model_size xl --mode forward_backward \
      --mixed_precision --num_warmup 5 --num_measure 10

  # Memory snapshot for section 2.1.6 (upload to pytorch.org/memory_viz):
  uv run python bench/benchmark.py --model_size xl --mode forward --memory_profile \
      --context_length 128
  uv run python bench/benchmark.py --model_size xl --mode full_step --memory_profile \
      --context_length 2048
"""

import argparse
import contextlib
import os
import statistics
import timeit
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

import core.flash_attention as _fa_module
from core.model import GPT, GPTConfig

# Three backends exposed to the CLI:
#   fa3   — Flash Attention 3 (Hopper/H100 only, BF16 required; silently skipped on other GPUs)
#   flash — PyTorch SDPA auto-selects best available kernel (FA2 on A100/H100, cuDNN on Blackwell)
#   math  — explicit annotated O(T²) attention; useful as a baseline to see FA2/FA3 speedup
# On H100: fa3 > flash > math gives the full three-tier comparison.
# On A100/Blackwell: fa3 degrades to flash (HAS_FA3=False), so only flash vs math is meaningful.
_ALL_BACKENDS = ["fa3", "flash", "math"]


@contextlib.contextmanager
def _attn_context(backend: str):
    """Return a context manager that enforces the requested attention backend."""
    old_use_fa3 = _fa_module.USE_FA3
    old_use_annotated_math = _fa_module.USE_ANNOTATED_MATH
    try:
        _fa_module.USE_ANNOTATED_MATH = backend == "math"
        if backend == "fa3":
            _fa_module.USE_FA3 = _fa_module.HAS_FA3  # no-op on non-Hopper; SDPA fallback
        else:
            _fa_module.USE_FA3 = False  # flash uses SDPA auto; math uses explicit annotated path
        yield
    finally:
        _fa_module.USE_FA3 = old_use_fa3
        _fa_module.USE_ANNOTATED_MATH = old_use_annotated_math


# Table 1 model sizes adapted to real training vocab (vocab_size=32768, batch=4, context=512).
# MLP hardcodes 4*n_embd, so d_ff = 4*n_embd for all sizes.
# n_kv_head = n_head (no GQA, matching Table 1 pre-GQA architecture).
# window_pattern="L" gives full-context attention at every layer (standard transformer).
# Note: xl has head_dim=2560/32=80; FA3 only supports power-of-2 head dims, so xl
# silently falls back to PyTorch SDPA — timings are valid but not FA3-accelerated.
MODEL_CONFIGS = {
    "small":  GPTConfig(vocab_size=32768, sequence_len=2048, n_layer=12, n_embd=768,  n_head=12, n_kv_head=12, window_pattern="L"),
    "medium": GPTConfig(vocab_size=32768, sequence_len=2048, n_layer=24, n_embd=1024, n_head=16, n_kv_head=16, window_pattern="L"),
    "large":  GPTConfig(vocab_size=32768, sequence_len=2048, n_layer=36, n_embd=1280, n_head=20, n_kv_head=20, window_pattern="L"),
    "xl":     GPTConfig(vocab_size=32768, sequence_len=2048, n_layer=32, n_embd=2560, n_head=32, n_kv_head=32, window_pattern="L"),
    # 10B omitted: Table 1 sets d_ff=12288 != 4*4608, but MLP hardcodes 4*n_embd.
    # sequence_len=2048 (not 512): with window_pattern="L", long_window=sequence_len. If
    # sequence_len=512 and we run at context=2048, FA3 receives window_size=(512,0) — silent
    # sliding-window cap. Using 2048 gives correct full causal attention at all tested lengths.
    # Rotary embeddings at positions 0-511 are identical regardless of sequence_len.
}


def create_model(size_name: str, device: str = "cuda") -> nn.Module:
    if size_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {size_name}")
    cfg = MODEL_CONFIGS[size_name]
    model = GPT(cfg).to(device)
    model.init_weights()
    return model


def maybe_compile_model(model: nn.Module, use_compile: bool) -> nn.Module:
    if not use_compile:
        return model
    # Keep graphs static for repeatable benchmark measurements.
    return torch.compile(model, dynamic=False)


def _add_nvtx_hooks(model: nn.Module) -> list:
    """
    Inject NVTX push/pop hooks on every Block, its attention sub-module, and MLP.

    Granularity note: flash_attn_func fuses QK^T + softmax + AV into a single CUDA
    kernel (on Hopper+) or PyTorch SDPA (fallback). The 'blockXX.attn' range therefore
    spans QKV projection + attention + output projection. With --attn_backend math,
    the explicit baseline adds nested NVTX ranges for QK, softmax, and PV.
    """
    handles = []

    def _push_range(name: str):
        # Forward pre-hooks must return None; returning range_push depth mutates module inputs.
        torch.cuda.nvtx.range_push(name)

    def _pop_range():
        torch.cuda.nvtx.range_pop()

    for i, block in enumerate(model.transformer.h):
        block_name = f"block{i:02d}"
        attn_name  = f"block{i:02d}.attn"
        mlp_name   = f"block{i:02d}.mlp"

        handles.append(block.register_forward_pre_hook(
            lambda m, inp, n=block_name: _push_range(n)))
        handles.append(block.register_forward_hook(
            lambda m, inp, out, n=block_name: _pop_range()))

        handles.append(block.attn.register_forward_pre_hook(
            lambda m, inp, n=attn_name: _push_range(n)))
        handles.append(block.attn.register_forward_hook(
            lambda m, inp, out, n=attn_name: _pop_range()))

        handles.append(block.mlp.register_forward_pre_hook(
            lambda m, inp, n=mlp_name: _push_range(n)))
        handles.append(block.mlp.register_forward_hook(
            lambda m, inp, out, n=mlp_name: _pop_range()))

    return handles


def run_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    mode: str,
    synchronize,
    mixed_precision: bool = False,
    nvtx_label: Optional[str] = None,
    attn_ctx=None,
) -> float:
    """
    Execute one step and return elapsed GPU time in seconds.

    synchronize: callable — torch.cuda.synchronize on CUDA, no-op on CPU/MPS.
    nvtx_label: when set, wraps the entire step in an NVTX range with this name.
    attn_ctx: context manager from _attn_context(backend); None = module default.
    """
    if attn_ctx is None:
        attn_ctx = contextlib.nullcontext()

    device_type = input_ids.device.type
    autocast_ctx = (
        torch.autocast(device_type, dtype=torch.bfloat16) if mixed_precision
        else contextlib.nullcontext()
    )

    synchronize()
    start = timeit.default_timer()

    if nvtx_label is not None:
        torch.cuda.nvtx.range_push(nvtx_label)

    with attn_ctx, autocast_ctx:
        if mode == "forward":
            with torch.no_grad():
                _ = model(input_ids)

        elif mode == "forward_backward":
            loss = model(input_ids, targets=targets)
            loss.backward()

        elif mode == "full_step":
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids, targets=targets)
            loss.backward()
            optimizer.step()

        else:
            raise ValueError(f"Unknown mode: {mode}")

    if nvtx_label is not None:
        torch.cuda.nvtx.range_pop()

    synchronize()
    return timeit.default_timer() - start


def benchmark(
    model: nn.Module,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    mode: str,
    num_warmup: int,
    num_measure: int,
    device: str,
    synchronize,
    mixed_precision: bool = False,
    use_nvtx: bool = False,
    backend: Optional[str] = None,
    torch_prof=None,
) -> Tuple[float, float]:
    """
    Run warmup + measured iterations.

    Warmup steps run without NVTX labels. Measured steps get labels of the form
    'measured_step_0', 'measured_step_1', ... so nsys --nvtx-capture can isolate them.

    backend: one of _ALL_BACKENDS or None (uses current module default).
    """
    model.train()
    model.zero_grad(set_to_none=True)  # clear any stale grads from a previous backend run
    optimizer = AdamW(model.parameters(), lr=1e-3)
    times: List[float] = []
    measured_idx = 0

    for step in range(num_warmup + num_measure):
        input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        targets   = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

        is_measured = step >= num_warmup
        nvtx_label = f"measured_step_{measured_idx}" if (use_nvtx and is_measured) else None
        attn_ctx = _attn_context(backend) if backend is not None else None

        elapsed = run_step(
            model, input_ids, targets, optimizer, mode,
            synchronize,
            mixed_precision=mixed_precision,
            nvtx_label=nvtx_label,
            attn_ctx=attn_ctx,
        )
        if torch_prof is not None:
            torch_prof.step()

        if is_measured:
            times.append(elapsed)
            measured_idx += 1

    avg = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return avg, std


def main():
    parser = argparse.ArgumentParser(description="Benchmark CS336 Table 1 Transformer steps")
    parser.add_argument("--model_size", type=str, default="small", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--mode", type=str, default="forward",
                        choices=["forward", "forward_backward", "full_step"])
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_measure", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda|cpu|mps")
    parser.add_argument("--all_sizes", action="store_true",
                        help="Sweep across all Table 1 model sizes (for part b).")
    parser.add_argument("--attn_backend", type=str, default="auto",
                        choices=_ALL_BACKENDS + ["auto", "all"],
                        help="flash: FA2/SDPA auto (best available kernel on the GPU) | "
                             "math: annotated unfused O(T²) baseline | all: run both | auto: model default.")
    # Profiling flags
    parser.add_argument("--nvtx", action="store_true",
                        help="Wrap each measured step in an NVTX range and inject "
                             "block/attn/mlp hooks. Combine with nsys "
                             "--nvtx-capture='measured_step_*' to exclude warmup from trace.")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Run forward/backward under torch.autocast BF16 (section 2.1.5c).")
    parser.add_argument("--memory_profile", action="store_true",
                        help="Record one step of memory history and dump a snapshot "
                             "(upload to pytorch.org/memory_viz). Section 2.1.6: use xl at "
                             "context_length=128 and 2048 separately.")
    parser.add_argument("--output", type=str, default="",
                        help="Append results to this CSV file (created if absent).")
    parser.add_argument("--compile", action="store_true",
                        help="Compile the model with torch.compile(dynamic=False) before benchmarking.")
    parser.add_argument("--torch_profile", action="store_true",
                        help="Record a PyTorch profiler trace (TensorBoard-compatible JSON). "
                             "Use one model size/backend per run for clean traces.")
    parser.add_argument("--torch_profile_dir", type=str, default="results/torch_profiler",
                        help="Base directory for torch profiler traces.")
    args = parser.parse_args()

    if args.torch_profile:
        if args.all_sizes:
            parser.error("--torch_profile requires a single --model_size (do not use --all_sizes).")
        if args.attn_backend == "all":
            parser.error("--torch_profile requires one backend (use auto|fa3|flash|math, not all).")
        if args.memory_profile:
            parser.error("--torch_profile cannot be combined with --memory_profile.")
        if args.device != "cuda":
            parser.error("--torch_profile currently expects --device=cuda.")

    synchronize = torch.cuda.synchronize if args.device == "cuda" else lambda: None

    sizes_to_run = list(MODEL_CONFIGS.keys()) if args.all_sizes else [args.model_size]
    backends_to_run = _ALL_BACKENDS if args.attn_backend == "all" else (
        [None] if args.attn_backend == "auto" else [args.attn_backend]
    )

    show_backend_col = args.attn_backend in ("all",) or args.attn_backend in _ALL_BACKENDS

    torch_prof = None
    profile_ctx = contextlib.nullcontext()
    if args.torch_profile:
        mp_tag = "bf16" if args.mixed_precision else "fp32"
        backend_tag = "auto" if args.attn_backend == "auto" else args.attn_backend
        trace_dir = os.path.join(
            args.torch_profile_dir,
            f"{args.model_size}_{args.mode}_{backend_tag}_ctx{args.context_length}_{mp_tag}",
        )
        os.makedirs(trace_dir, exist_ok=True)
        profile_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=args.num_warmup, active=args.num_measure, repeat=1),
            record_shapes=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        )
        print(f"torch.profiler enabled → {trace_dir}")

    if not args.memory_profile:
        hdr = (f"{'Size':<8} {'Params(M)':<12} {'Mode':<18} {'MP':<5} "
               f"{'Avg(ms)':<12} {'Std(ms)':<12} {'Exec'}")
        if show_backend_col:
            hdr = f"{'Backend':<14} " + hdr
        print(hdr)
        print("-" * (len(hdr) + 2))

    with profile_ctx as torch_prof:
        for size_name in sizes_to_run:
            model = create_model(size_name, device=args.device)
            model = maybe_compile_model(model, args.compile)
            num_params = sum(p.numel() for p in model.parameters()) / 1e6

            # ---- NVTX hooks ----
            nvtx_handles = []
            if args.nvtx and args.device == "cuda":
                nvtx_handles = _add_nvtx_hooks(model)

            # ---- Memory profile: one step, dump snapshot, skip normal loop ----
            if args.memory_profile and args.device == "cuda":
                model.train()
                optimizer = AdamW(model.parameters(), lr=1e-3)
                vocab_size = MODEL_CONFIGS[size_name].vocab_size
                input_ids = torch.randint(0, vocab_size, (args.batch_size, args.context_length), device=args.device)
                targets   = torch.randint(0, vocab_size, (args.batch_size, args.context_length), device=args.device)
                torch.cuda.memory._record_memory_history(max_entries=1_000_000)
                run_step(model, input_ids, targets, optimizer, args.mode,
                         synchronize, mixed_precision=args.mixed_precision)
                torch.cuda.memory._record_memory_history(enabled=None)  # stop recording
                snapshot_path = f"{size_name}_ctx{args.context_length}_{args.mode}_memory.pickle"
                torch.cuda.memory._dump_snapshot(snapshot_path)
                peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                print(f"{size_name}  ctx={args.context_length}  {args.mode}  "
                      f"memory snapshot → {snapshot_path}  (peak {peak_mb:.0f} MB)")
                for h in nvtx_handles:
                    h.remove()
                del model
                torch.cuda.empty_cache()
                continue

            # ---- Normal benchmark — iterate over requested backends ----
            for backend in backends_to_run:
                avg, std = benchmark(
                    model=model,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    vocab_size=MODEL_CONFIGS[size_name].vocab_size,
                    mode=args.mode,
                    num_warmup=args.num_warmup,
                    num_measure=args.num_measure,
                    device=args.device,
                    synchronize=synchronize,
                    mixed_precision=args.mixed_precision,
                    use_nvtx=(args.nvtx and args.device == "cuda"),
                    backend=backend,
                    torch_prof=torch_prof,
                )

                mp_label = "BF16" if args.mixed_precision else "FP32"
                compile_label = "compiled" if args.compile else "eager"
                row = (f"{size_name:<8} {num_params:<12.1f} {args.mode:<18} "
                       f"{mp_label:<5} {avg * 1000:<12.2f} {std * 1000:<12.2f} {compile_label}")
                if show_backend_col:
                    row = f"{(backend or 'auto'):<14} " + row
                print(row)

                if args.output:
                    import csv, os as _os
                    write_header = not _os.path.exists(args.output)
                    with open(args.output, "a", newline="") as f:
                        w = csv.writer(f)
                        if write_header:
                            w.writerow(["size", "params_m", "mode", "mp", "compile", "backend",
                                        "context_length", "batch_size", "avg_ms", "std_ms"])
                        w.writerow([size_name, f"{num_params:.1f}", args.mode, mp_label, compile_label,
                                    backend or "auto", args.context_length, args.batch_size,
                                    f"{avg * 1000:.2f}", f"{std * 1000:.2f}"])

            for h in nvtx_handles:
                h.remove()

            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
