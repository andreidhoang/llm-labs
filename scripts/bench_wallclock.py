"""Wall-clock benchmark harness for llm/ training stack.

Standalone — does NOT touch base_train.py. Builds a model identical to
base_train.py would (depth/aspect_ratio/window_pattern), feeds synthetic
random tokens, runs N warmup + M measure steps, captures per-step timing
with CUDA events. Output is a JSON file matching the schema we agreed:

    {
      "config": "<phase tag>",
      "n_gpu": <int>,
      "device_batch_size": <int>,
      "grad_accum_steps": <int>,
      "compile": <bool>, "compile_mode": <str>,
      "activation_ckpt": <bool>,
      "fp8": <bool>,
      "metrics": {
        "tokens_per_sec_per_gpu": <float>,
        "mfu_pct": <float>,
        "peak_hbm_gb": <float>,
        "step_time_ms": <float>,
        "fwd_bwd_ms": <float>,
        "optim_step_ms": <float>,
        "optim_pct_of_step": <float>,
        "compile_first_step_overhead_s": <float>,
      },
      "per_step": [...]   # raw per-step measurements for percentiles
    }

The optim_step_ms is our proxy for cross-rank comm cost — DistMuonAdamW
does all its reduce_scatter/all_gather inside .step(), so optim_step_ms
≈ comm_ms when world_size > 1.

Usage:
    # Single GPU baseline
    python scripts/bench_wallclock.py --phase phase_0_baseline --depth 12 \
        --device-batch-size 8 --grad-accum-steps 4 --measure-steps 30

    # 2× H100 with all optimizations on
    torchrun --nproc_per_node=2 -m scripts.bench_wallclock \
        --phase phase_5_full --depth 20 --device-batch-size 16 \
        --grad-accum-steps 2 --compile-mode max-autotune \
        --activation-ckpt --fp8

Run on:  any CUDA GPU (results only meaningful on H100 for production extrapolation)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from core.common import (
    autodetect_device_type,
    compute_init,
    compute_cleanup,
    print0,
    get_peak_flops,
    COMPUTE_DTYPE,
)
from core.model import GPT, GPTConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "bench"


# ─────────────────────────────────────────────────────────────────
# Reproducibility metadata
# ─────────────────────────────────────────────────────────────────

def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def collect_metadata() -> dict:
    """Snapshot for reproducibility. Run ONCE per benchmark — cheap (~50 ms).

    Captures git state + library versions + GPU/driver/host info. The aim is
    that two runs with identical metadata should be byte-identical (modulo
    NCCL nondeterminism). If anything diverges, this is your audit log.
    """
    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cmdline": sys.argv,
        # Git
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": _git(["status", "--porcelain"]) != "",
        "git_diff_bytes": len(_git(["diff", "HEAD"]).encode()),
        # Torch / CUDA stack
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "",
        "cudnn_version": torch.backends.cudnn.version() or 0,
        "nccl_version": ".".join(str(x) for x in torch.cuda.nccl.version())
                        if torch.cuda.is_available() else "",
    }
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["gpu_count"] = torch.cuda.device_count()
        cap = torch.cuda.get_device_capability(0)
        meta["gpu_capability"] = f"sm_{cap[0]}{cap[1]}"
        # nvidia-smi for driver version (one-shot, ~30 ms)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version,memory.total",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
            ).decode().strip().splitlines()
            if out:
                drv, mem = out[0].split(",")
                meta["driver_version"] = drv.strip()
                meta["gpu_total_mem_mib"] = int(mem.strip())
        except Exception:
            pass
    # NCCL env that matters for repro
    nccl_env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
    if nccl_env:
        meta["nccl_env"] = nccl_env
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wall-clock benchmark for llm/")
    # Identification
    p.add_argument("--phase", type=str, required=True,
                   help="phase tag e.g. phase_0_baseline, phase_2_chunked_ce")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    # Model
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--aspect-ratio", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--window-pattern", type=str, default="SSSL")
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--num-shared-experts", type=int, default=1)
    p.add_argument("--vocab-size", type=int, default=32768,
                   help="synthetic vocab — use 32768 to match real tokenizer")
    # Training step shape
    p.add_argument("--device-batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    # Optimization toggles
    p.add_argument("--compile-mode", type=str, default="default",
                   choices=["off", "default", "reduce-overhead", "max-autotune"])
    p.add_argument("--compile-fullgraph", action="store_true",
                   help="torch.compile(fullgraph=True). Requires graph breaks eliminated.")
    p.add_argument("--activation-ckpt", action="store_true",
                   help="Per-block activation checkpointing (requires phase 3 patch)")
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--fp8-recipe", type=str, default="tensorwise")
    p.add_argument("--require-fa3", action="store_true",
                   help="Fail loudly if FlashAttention-3 is not active (catches silent SDPA fallback)")
    # Bench shape
    p.add_argument("--warmup-steps", type=int, default=10,
                   help="Skip these steps from measurement (compile autotune + JIT warmup)")
    p.add_argument("--measure-steps", type=int, default=30)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device-type", type=str, default="")
    # Live monitoring & persistence
    p.add_argument("--wandb-project", type=str, default="",
                   help="wandb project name (empty = disable wandb)")
    p.add_argument("--wandb-run", type=str, default="",
                   help="wandb run name (default: <phase>-<host>-<sha7>)")
    p.add_argument("--wandb-entity", type=str, default="",
                   help="wandb entity / team (default: account default)")
    p.add_argument("--jsonl-flush-every", type=int, default=1,
                   help="Flush per-step JSONL every N steps (1 = every step, safest)")
    return p.parse_args()


def build_model(args: argparse.Namespace) -> GPT:
    """Replicates the model construction logic in base_train.py."""
    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
        num_experts=args.num_experts,
        top_k=args.top_k,
        num_shared_experts=args.num_shared_experts,
    )
    with torch.device("meta"):
        model = GPT(config)
    return model


def maybe_apply_fp8(model: GPT, recipe: str) -> int:
    """Returns count of converted Linear layers (0 if FP8 not applied)."""
    import torch.nn as nn
    from core.fp8 import Float8LinearConfig, convert_to_float8_training

    def fp8_filter(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if min(mod.in_features, mod.out_features) < 128:
            return False
        return True

    cfg = Float8LinearConfig.from_recipe_name(recipe)
    convert_to_float8_training(model, config=cfg, module_filter_fn=fp8_filter)
    return sum(1 for m in model.modules() if "Float8" in type(m).__name__)


def maybe_enable_activation_ckpt(model: GPT) -> bool:
    """Sets the model attribute checked by GPT.forward; the forward path
    must already understand this flag (added in phase 3). Returns True
    if the model exposes the hook, False if phase 3 hasn't landed yet."""
    if hasattr(model, "_use_activation_checkpointing"):
        model._use_activation_checkpointing = True
        return True
    setattr(model, "_use_activation_checkpointing", True)
    return False  # set anyway, but signal that forward may not honor it


def synthetic_batch(batch_size: int, seq_len: int, vocab_size: int,
                    device: torch.device, generator: torch.Generator):
    """One synthetic (x, y) pair. Random ids in [0, vocab_size)."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len),
                      device=device, dtype=torch.long, generator=generator)
    # next-token target: just shift; for benchmark the actual values don't matter
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y


class StepTimer:
    """CUDA event-based per-step timer with section breakdowns.

    Sections measured:
      total : full step (forward+backward across microsteps + optimizer)
      fwdbwd: sum of forward+backward across all microsteps
      optim : optimizer.step() — proxy for cross-rank comm in DistMuonAdamW

    Crash-resilient: each completed step appends one line to <jsonl_path>
    and flushes. If the VM dies mid-run, you still have N-1 measured steps.
    """

    def __init__(self, jsonl_path: Path | None = None, flush_every: int = 1,
                 wandb_run=None):
        self.events = {}
        self.records = []  # list of dicts per step
        self.jsonl_path = jsonl_path
        self.jsonl_file = None
        if jsonl_path is not None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate any prior partial run for this phase
            self.jsonl_file = jsonl_path.open("w", buffering=1)  # line-buffered
        self.flush_every = max(1, flush_every)
        self.wandb_run = wandb_run

    def _evt(self):
        return torch.cuda.Event(enable_timing=True)

    def step_begin(self):
        self.events["t0"] = self._evt(); self.events["t0"].record()

    def fwdbwd_begin(self):
        self.events["fb0"] = self._evt(); self.events["fb0"].record()

    def fwdbwd_end(self):
        self.events["fb1"] = self._evt(); self.events["fb1"].record()

    def optim_begin(self):
        self.events["op0"] = self._evt(); self.events["op0"].record()

    def optim_end(self):
        self.events["op1"] = self._evt(); self.events["op1"].record()

    def step_end(self, peak_hbm_bytes: int | None = None):
        self.events["t1"] = self._evt(); self.events["t1"].record()
        torch.cuda.synchronize()
        step_idx = len(self.records)
        rec = {
            "step":       step_idx,
            "step_ms":    self.events["t0"].elapsed_time(self.events["t1"]),
            "fwdbwd_ms":  self.events["fb0"].elapsed_time(self.events["fb1"]),
            "optim_ms":   self.events["op0"].elapsed_time(self.events["op1"]),
        }
        if peak_hbm_bytes is not None:
            rec["peak_hbm_gb"] = round(peak_hbm_bytes / 1024**3, 4)
        self.records.append(rec)
        # Stream to JSONL — survives a crash mid-run
        if self.jsonl_file is not None:
            self.jsonl_file.write(json.dumps(rec) + "\n")
            if step_idx % self.flush_every == 0:
                self.jsonl_file.flush()
                os.fsync(self.jsonl_file.fileno())
        # Stream to wandb — live charts
        if self.wandb_run is not None:
            self.wandb_run.log(rec, step=step_idx)
        self.events.clear()
        return rec

    def close(self):
        if self.jsonl_file is not None:
            self.jsonl_file.flush()
            self.jsonl_file.close()
            self.jsonl_file = None


def aggregate_metrics(records, total_tokens_per_step, n_gpu, peak_flops_per_gpu,
                      flops_per_token, peak_hbm_bytes, compile_overhead_s):
    """Take per-step records and compute summary metrics."""
    step_ms = [r["step_ms"] for r in records]
    fwd_ms = [r["fwdbwd_ms"] for r in records]
    op_ms = [r["optim_ms"] for r in records]

    # Use median to avoid outliers from GC, network blips
    step_med = statistics.median(step_ms)
    fwd_med = statistics.median(fwd_ms)
    op_med = statistics.median(op_ms)

    tokens_per_sec_per_gpu = (total_tokens_per_step / n_gpu) / (step_med / 1000)
    flops_per_sec_total = flops_per_token * total_tokens_per_step / (step_med / 1000)
    mfu_pct = 100.0 * flops_per_sec_total / (peak_flops_per_gpu * n_gpu)

    return {
        "tokens_per_sec_per_gpu": round(tokens_per_sec_per_gpu, 1),
        "mfu_pct": round(mfu_pct, 2),
        "peak_hbm_gb": round(peak_hbm_bytes / 1024**3, 3),
        "step_time_ms": round(step_med, 3),
        "fwd_bwd_ms": round(fwd_med, 3),
        "optim_step_ms": round(op_med, 3),
        "optim_pct_of_step": round(100.0 * op_med / step_med, 2),
        "compile_first_step_overhead_s": round(compile_overhead_s, 2),
        "step_p50_ms": round(step_med, 3),
        "step_p90_ms": round(sorted(step_ms)[int(0.9 * len(step_ms))], 3),
        "step_min_ms": round(min(step_ms), 3),
        "step_max_ms": round(max(step_ms), 3),
    }


def init_wandb(args, world_size: int, master: bool, meta: dict):
    """Initialize wandb run if --wandb-project is set. Returns wandb run or None."""
    if not master or not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print0("WARNING: --wandb-project set but wandb not installed; skipping")
        return None
    sha7 = (meta.get("git_sha") or "nogit")[:7]
    run_name = args.wandb_run or f"{args.phase}-{meta.get('host', 'host')}-{sha7}"
    config = {**vars(args), **{f"meta.{k}": v for k, v in meta.items()}, "world_size": world_size}
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        config=config,
        tags=[args.phase, f"gpus={world_size}", meta.get("gpu_capability", "")],
        reinit=True,
    )


def main():
    args = parse_args()

    # ----- compute init -----
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    if device_type != "cuda":
        raise RuntimeError(f"This benchmark requires CUDA; got {device_type}")
    ddp, rank, local_rank, world_size, device = compute_init(device_type)
    master = rank == 0

    torch.manual_seed(args.seed + rank)

    # Reproducibility metadata snapshot (master only — others would be redundant)
    meta = collect_metadata() if master else {}
    if master:
        # Persist a top-level _meta.json for the whole run battery
        args.out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = args.out_dir / "_meta.json"
        # Append/merge: keep prior meta from earlier phase, overwrite per-phase entries
        existing = {}
        if meta_path.exists():
            try:
                with meta_path.open() as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.setdefault("first_run", meta)
        existing.setdefault("phases", {})[args.phase] = {"timestamp_utc": meta["timestamp_utc"],
                                                          "cmdline": meta["cmdline"]}
        existing["latest_run"] = meta
        with meta_path.open("w") as f:
            json.dump(existing, f, indent=2)
        print0(f"Repro meta: git_sha={meta['git_sha'][:8]}{'(dirty)' if meta['git_dirty'] else ''} "
               f"torch={meta['torch_version']} cuda={meta['torch_cuda_version']} "
               f"nccl={meta.get('nccl_version', '?')}")

    gpu_name = torch.cuda.get_device_name(0)
    peak_flops = get_peak_flops(gpu_name)
    print0(f"GPU: {gpu_name} | peak BF16 FLOPS: {peak_flops:.2e} | world_size: {world_size}")
    print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE}")

    # Verify attention impl status — important for production benches; SDPA fallback tanks MFU
    from core.flash_attention import USE_FA3, USE_FA2, USE_FA, HAS_FA3, HAS_FA2, _IMPL_NAME
    print0(f"Attention: impl={_IMPL_NAME} (HAS_FA3={HAS_FA3} HAS_FA2={HAS_FA2})")
    if args.require_fa3 and not USE_FA3:
        raise RuntimeError(
            f"--require-fa3 set but FA3 not active (HAS_FA3={HAS_FA3}, USE_FA3={USE_FA3}). "
            f"Verify Hopper GPU + flash-attn install + COMPUTE_DTYPE=bf16."
        )

    # Initialize wandb (master only)
    wandb_run = init_wandb(args, world_size, master, meta)
    if wandb_run is not None:
        print0(f"wandb: logging to {wandb_run.url}")

    # ----- build model -----
    model = build_model(args)
    print0(f"Model config: depth={args.depth} dim={model.config.n_embd} "
           f"n_head={model.config.n_head} vocab={args.vocab_size}")
    model.to_empty(device=device)
    model.init_weights()

    # FP8 (must happen before torch.compile)
    fp8_active = False
    if args.fp8:
        try:
            n_fp8 = maybe_apply_fp8(model, args.fp8_recipe)
            fp8_active = n_fp8 > 0
            print0(f"FP8 enabled: {n_fp8} Linear layers converted")
        except Exception as e:
            print0(f"FP8 setup failed: {e}; continuing without FP8")

    # Activation checkpointing toggle (requires forward path support — phase 3)
    ckpt_supported = False
    if args.activation_ckpt:
        ckpt_supported = maybe_enable_activation_ckpt(model)
        if not ckpt_supported:
            print0("WARNING: --activation-ckpt set but model.forward does not yet honor it. "
                   "Land phase 3 patch first.")

    # Optimizer
    optimizer = model.setup_optimizer(
        unembedding_lr=0.008, embedding_lr=0.3, scalar_lr=0.5,
        matrix_lr=0.02, weight_decay=0.0,
    )

    # Compile (last, after FP8 conversion)
    compile_overhead_s = 0.0
    if args.compile_mode != "off":
        kw = dict(dynamic=False, mode=args.compile_mode)
        if args.compile_fullgraph:
            kw["fullgraph"] = True
        print0(f"torch.compile(...{kw})")
        model = torch.compile(model, **kw)

    # ----- warmup -----
    gen = torch.Generator(device=device).manual_seed(args.seed + rank)
    B = args.device_batch_size
    T = args.max_seq_len
    V = args.vocab_size
    accum = args.grad_accum_steps
    total_tokens_per_step = B * T * accum * world_size
    flops_per_token = model.estimate_flops() if hasattr(model, "estimate_flops") else \
                      getattr(getattr(model, "_orig_mod", model), "estimate_flops")()

    print0(f"Tokens/step (effective batch): {total_tokens_per_step:,}")
    print0(f"FLOPs/token (estimate): {flops_per_token:.3e}")

    jsonl_path = (args.out_dir / f"{args.phase}.jsonl") if master else None
    timer = StepTimer(jsonl_path=jsonl_path, flush_every=args.jsonl_flush_every,
                      wandb_run=wandb_run)
    torch.cuda.reset_peak_memory_stats(device)

    print0(f"Warming up for {args.warmup_steps} steps "
           f"(includes compile autotune; first step may take minutes)...")

    for step in range(args.warmup_steps):
        if step == 0:
            t_compile = time.time()
        for _ in range(accum):
            x, y = synthetic_batch(B, T, V, device, gen)
            loss = model(x, y)
            if isinstance(loss, dict):
                loss = loss["loss"]
            (loss / accum).backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        if step == 0:
            torch.cuda.synchronize()
            compile_overhead_s = time.time() - t_compile
            print0(f"  step 0 done in {compile_overhead_s:.2f}s "
                   f"(compile + autotune + first kernel JIT)")
        else:
            torch.cuda.synchronize()
            print0(f"  warmup step {step}/{args.warmup_steps} done")

    # disable GC like base_train.py does after warmup
    gc.collect()
    gc.freeze()
    gc.disable()

    # Reset peak memory after warmup so we measure steady-state HBM, not autotune scratch.
    torch.cuda.reset_peak_memory_stats(device)

    # ----- measure -----
    print0(f"Measuring {args.measure_steps} steps...")
    for step in range(args.measure_steps):
        timer.step_begin()
        timer.fwdbwd_begin()
        for _ in range(accum):
            x, y = synthetic_batch(B, T, V, device, gen)
            loss = model(x, y)
            if isinstance(loss, dict):
                loss = loss["loss"]
            (loss / accum).backward()
        timer.fwdbwd_end()
        timer.optim_begin()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        timer.optim_end()
        # Snapshot peak HBM each step so JSONL has memory growth curve
        rec = timer.step_end(peak_hbm_bytes=torch.cuda.max_memory_allocated(device))
        if master and step % 5 == 0:
            print0(f"  step {step:03d}: total={rec['step_ms']:.1f}ms "
                   f"fwdbwd={rec['fwdbwd_ms']:.1f}ms optim={rec['optim_ms']:.1f}ms "
                   f"hbm={rec.get('peak_hbm_gb', 0):.2f}GB")

    timer.close()
    peak_hbm = torch.cuda.max_memory_allocated(device)

    # ----- aggregate -----
    metrics = aggregate_metrics(
        timer.records,
        total_tokens_per_step=total_tokens_per_step,
        n_gpu=world_size,
        peak_flops_per_gpu=peak_flops,
        flops_per_token=flops_per_token,
        peak_hbm_bytes=peak_hbm,
        compile_overhead_s=compile_overhead_s,
    )

    result = {
        "phase": args.phase,
        "n_gpu": world_size,
        "gpu_name": gpu_name,
        "device_batch_size": B,
        "max_seq_len": T,
        "grad_accum_steps": accum,
        "total_batch_size_tokens": total_tokens_per_step,
        "compile_mode": args.compile_mode,
        "compile_fullgraph": args.compile_fullgraph,
        "activation_ckpt": args.activation_ckpt and ckpt_supported,
        "fp8": fp8_active,
        "metrics": metrics,
        # Embed reproducibility meta inside each phase JSON so the file is self-contained.
        "meta": meta,
        "per_step_step_ms": [r["step_ms"] for r in timer.records],
        "per_step_fwdbwd_ms": [r["fwdbwd_ms"] for r in timer.records],
        "per_step_optim_ms": [r["optim_ms"] for r in timer.records],
    }

    # ----- output -----
    if master:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / f"{args.phase}.json"
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
        print0("\n" + "=" * 70)
        print0(f"PHASE: {args.phase}")
        print0("=" * 70)
        for k, v in metrics.items():
            print0(f"  {k:35s}: {v}")
        print0(f"  → wrote {out_path}")
        print0(f"  → wrote {jsonl_path} ({len(timer.records)} per-step rows)")
        print0("=" * 70)

        # Push final summary metrics to wandb as a single log + finish the run
        if wandb_run is not None:
            wandb_run.summary.update({f"final/{k}": v for k, v in metrics.items()})
            wandb_run.summary["phase"] = args.phase
            wandb_run.summary["world_size"] = world_size
            try:
                # Upload the JSON + JSONL as artifacts (easy diff across runs)
                import wandb
                art = wandb.Artifact(name=f"bench-{args.phase}", type="benchmark")
                art.add_file(str(out_path))
                if jsonl_path is not None and jsonl_path.exists():
                    art.add_file(str(jsonl_path))
                wandb_run.log_artifact(art)
            except Exception as e:
                print0(f"  wandb artifact upload failed: {e}")
            wandb_run.finish()

    compute_cleanup()


if __name__ == "__main__":
    main()
