# Benchmarking & Profiling Guide

Two benchmark scripts cover two codebases:

- `bench/benchmark.py` — CS336 Table 1 transformer (standard GPT, `window_pattern="L"`)
- `nanochat/scripts/benchmark.py` — nanochat production model (SSSL sliding-window, muP sizing)

Run from their respective roots:

```bash
# CS336 benchmark
cd /path/to/llm
uv run python bench/benchmark.py [flags]

# nanochat benchmark
cd /path/to/nanochat
python -m scripts.benchmark [flags]
```

Results print to stdout. Append `--output results/run.csv` to any run to persist rows to a CSV file (created if absent, appended if it exists).

---

## GPU Tiers

| GPU | FA3 | FA2/SDPA | BF16 Tensor Cores | VRAM | Notes |
|-----|-----|----------|-------------------|------|-------|
| H100 SXM | Yes (sm90) | Yes | Yes | 80 GB | Full three-tier comparison |
| A100 | No | Yes | Yes | 40/80 GB | FA2 vs math only |
| RTX Pro 6000 Blackwell | No (sm120 ≠ sm90) | Yes | Yes | 96 GB | FA2 vs math; needs PyTorch ≥ 2.6 |
| RTX 4090 / A10G | No | Yes | Yes | 24 GB | xl full_step tight (~20 GB) |

> **xl model caveat**: `head_dim = 2560 / 32 = 80` (non-power-of-2). FA3 silently falls back to SDPA even on H100. Use `small`, `medium`, or `large` for clean FA3 measurements.

> **FA3 requires BF16**: `_resolve_use_fa3()` returns False if `COMPUTE_DTYPE != bfloat16`. Always add `--mixed_precision` when benchmarking FA3.

---

## Attention Backends

Three choices for `--attn_backend`:

| Backend | What runs | When to use |
|---------|-----------|-------------|
| `fa3` | Flash Attention 3 (Hopper only; falls back to SDPA on other GPUs) | H100 BF16 path |
| `flash` | PyTorch SDPA auto — FA2 on A100/H100, cuDNN on Blackwell | Non-FA3 best kernel |
| `math` | Explicit unfused O(T²) attention with NVTX ranges around QK, softmax, and PV | Baseline to quantify speedup and inspect attention sub-steps in nsys |
| `all` | All three, side-by-side rows | Full comparison in one run |
| `auto` | Whatever `USE_FA3` resolved to at import | Default, matches training |

---

## 1. Throughput Sweep — all model sizes

**What it does**: Full training step (forward + backward + optimizer) for every model size. Reports ms/step, tokens/sec, and MFU% on CUDA.

**Why it matters**: Hardware baseline. Shows how throughput degrades with model size and whether you're memory-bandwidth or compute bound.

```bash
# CS336 — small / medium / large / xl
uv run python bench/benchmark.py --mode full_step --all_sizes \
    --mixed_precision --num_warmup 5 --num_measure 10 \
    --output results/throughput.csv

# nanochat — d12 / d20 / d24 / d26
python -m scripts.benchmark --mode full_step --all_sizes \
    --mixed_precision --num_warmup 5 --num_measure 10 \
    --output results/nanochat_throughput.csv
```

---

## 2. Forward vs Backward vs Full Step

**What it does**: Isolates where time is spent — inference, gradient computation, or the complete optimizer update.

**Why it matters**: If forward is fast but full_step is 4× slower, backward or the optimizer is the bottleneck. Guides where to focus optimization effort.

```bash
uv run python bench/benchmark.py --model_size large --mode forward \
    --mixed_precision --num_warmup 5 --num_measure 10
uv run python bench/benchmark.py --model_size large --mode forward_backward \
    --mixed_precision --num_warmup 5 --num_measure 10
uv run python bench/benchmark.py --model_size large --mode full_step \
    --mixed_precision --num_warmup 5 --num_measure 10
```

Rule of thumb: backward ≈ 2× forward. More than 3× suggests memory pressure or inefficient gradient accumulation.

---

## 3. Warmup Effect

**What it does**: Compares 0, 1, and 5 warmup steps to show how kernel-launch overhead inflates early measurements.

**Why it matters**: Without warmup, reported timings are artificially high. Shows how many steps are needed before measurements stabilize.

```bash
uv run python bench/benchmark.py --model_size xl --mode full_step \
    --mixed_precision --num_warmup 0 --num_measure 10
uv run python bench/benchmark.py --model_size xl --mode full_step \
    --mixed_precision --num_warmup 1 --num_measure 10
uv run python bench/benchmark.py --model_size xl --mode full_step \
    --mixed_precision --num_warmup 5 --num_measure 10
```

Watch `±ms` — it drops sharply once the GPU is warm.

---

## 4. Attention Backend Comparison

**What it does**: Runs the model with each attention backend and prints side-by-side rows.

**Why it matters**: Quantifies exactly how much FA3/FA2 saves over explicit unfused attention. At T=2048, the math baseline is 5–20× slower due to materializing the full T×T matrix in HBM. In CS336, `math` also emits nested NVTX ranges (`sdpa_full`, `attn_scores_QK`, `attn_softmax`, `attn_out_PV`) so nsys shows the attention sub-steps separately.

### On H100 — three-tier comparison (FA3 > FA2 > math)

```bash
# CS336 — use small/medium/large, NOT xl (head_dim=80 breaks FA3)
uv run python bench/benchmark.py --model_size small --mode forward_backward \
    --attn_backend all --mixed_precision --context_length 2048 \
    --num_warmup 5 --num_measure 10 --output results/h100_attn.csv
```

Expected output:
```
Backend        Size     Params(M)    Mode               MP    Avg(ms)      Std(ms)
──────────────────────────────────────────────────────────────────────────────────
fa3            small    ...          forward_backward   BF16  ...          ...
flash          small    ...          forward_backward   BF16  ...          ...
math           small    ...          forward_backward   BF16  ...          ...
```

### On A100 / Blackwell — two-tier comparison (FA2 > math)

`fa3` row will show the same timing as `flash` (FA3 unavailable, silently falls back).

```bash
uv run python bench/benchmark.py --model_size small --mode forward_backward \
    --attn_backend all --mixed_precision --context_length 2048 \
    --num_warmup 5 --num_measure 10
```

> **nanochat note**: Short-window SSSL layers (`window=512`) fall into SDPA's explicit-mask path for all non-FA3 backends, adding Python-side tensor construction overhead. The comparison is still apples-to-apples since both `flash` and `math` include it.

---

## 5. Context Length Sweep

**What it does**: Same model at T=512, 1024, 2048 to show how attention cost scales.

**Why it matters**: Attention is O(T²) in memory (unfused) vs O(T) in memory (fused). This sweep shows the inflection point where attention dominates MLP time, and confirms FA3/FA2 keep cost manageable.

```bash
for T in 512 1024 2048; do
  uv run python bench/benchmark.py --model_size small --mode forward_backward \
      --attn_backend all --mixed_precision --context_length $T \
      --num_warmup 3 --num_measure 5 --output results/context_sweep.csv
done

# nanochat — also shows SSSL window benefit
for T in 512 1024 2048; do
  python -m scripts.benchmark --depth 20 --mode forward_backward \
      --attn_backend all --mixed_precision --context_length $T \
      --output results/nanochat_context_sweep.csv
done
```

---

## 6. torch.compile

**What it does**: Applies `torch.compile(model, dynamic=False)` before benchmarking.

**Why it matters**: `torch.compile` gives 10–30% throughput gains by fusing elementwise ops and eliminating Python overhead. Tells you whether the compile latency is worth it for your training run length.

```bash
# Baseline
python -m scripts.benchmark --depth 20 --mode full_step --mixed_precision

# Compiled
python -m scripts.benchmark --depth 20 --mode full_step --mixed_precision --compile
```

> Do **not** combine `--compile` with `--nvtx`. Compiled graphs cannot eliminate Python hook overhead, so NVTX annotations distort compiled-kernel timings.

---

## 7. Mixed Precision (BF16)

**What it does**: Wraps the forward/backward pass in `torch.autocast(dtype=bfloat16)`.

**Why it matters**: BF16 halves activation memory bandwidth and enables Tensor Core acceleration. Expect roughly 1.5–2× throughput uplift on A100/H100. FA3 requires BF16 — this is also the condition under which FA3 activates.

```bash
# FP32 baseline
uv run python bench/benchmark.py --model_size large --mode forward_backward \
    --num_warmup 5 --num_measure 10

# BF16
uv run python bench/benchmark.py --model_size large --mode forward_backward \
    --mixed_precision --num_warmup 5 --num_measure 10
```

H100 BF16 peak is ~990 TFLOPS vs ~67 TFLOPS FP32 — expect ~10–14× speedup on matmul-heavy workloads.

---

## 8. NVTX Profiling (nsys) — H100 priority

**What it does**: Injects NVTX range markers on every block, attention sub-module, and MLP. For CS336 `--attn_backend math`, the explicit baseline also marks QK scores, softmax, and PV output separately. Only measured (non-warmup) steps are labelled so nsys capture is clean.

**Why it matters**: On H100, you can see FA3 as a single wide fused kernel per block vs SDPA as multiple smaller dispatches. With `--attn_backend math`, `blockXX.attn` contains nested `sdpa_full`, `attn_scores_QK`, `attn_softmax`, and `attn_out_PV` ranges, making the unfused baseline easy to inspect. Gaps between kernels reveal CPU/Python launch overhead.

```bash
# CS336 — FA3 path (H100, BF16)
nsys profile --trace=cuda,nvtx --nvtx-capture="measured_step_*" -o h100_fa3 \
  uv run python bench/benchmark.py --model_size small --mode forward_backward \
    --mixed_precision --context_length 2048 --num_warmup 3 --num_measure 1 --nvtx

# CS336 — SDPA path for comparison
nsys profile --trace=cuda,nvtx --nvtx-capture="measured_step_*" -o h100_sdpa \
  uv run python bench/benchmark.py --model_size small --mode forward_backward \
    --attn_backend flash --mixed_precision --context_length 2048 \
    --num_warmup 3 --num_measure 1 --nvtx

# CS336 — explicit math baseline with nested QK / softmax / PV NVTX ranges
nsys profile --trace=cuda,nvtx --nvtx-capture="measured_step_*" -o h100_math_annotated \
  uv run python bench/benchmark.py --model_size small --mode forward_backward \
    --attn_backend math --mixed_precision --context_length 2048 \
    --num_warmup 3 --num_measure 1 --nvtx

# nanochat (labels include window type: block00_S = short-window, block03_L = full-context)
nsys profile --trace=cuda,nvtx -o d20_profile \
  python -m scripts.benchmark --depth 20 --mode forward_backward \
    --mixed_precision --context_length 2048 --num_warmup 5 --num_measure 1 --nvtx
```

Open the `.nsys-rep` file in Nsight Systems. Look for:
- `blockXX.attn` much narrower with FA3 than with SDPA → FA3 fused kernel is running
- `sdpa_full` containing `attn_scores_QK`, `attn_softmax`, and `attn_out_PV` in the `math` trace → annotated explicit baseline is running
- Gap between kernel launches → CPU/Python overhead
- `blockXX.attn` span much larger than `blockXX.mlp` → attention bound (long context)
- Uniform block timing → compute bound (good)

---

## 9. torch.profiler Trace (TensorBoard)

**What it does**: Records a Chrome-compatible trace with CPU + CUDA activity, kernel names, and shapes.

**Why it matters**: Lighter-weight than nsys. On H100 the FA3 kernel appears as `flash_fwd_kernel_*` — lets you verify FA3 is actually dispatching rather than silently falling back to SDPA.

```bash
python -m scripts.benchmark --depth 20 --mode forward_backward \
    --mixed_precision --torch_profile

tensorboard --logdir prof_output
```

Navigate to **PyTorch Profiler** → **Trace** view. Sort by CUDA time to find the top kernels.

---

## 10. Memory Snapshot

**What it does**: Records the full memory allocation timeline for one step and dumps a `.pickle` file.

**Why it matters**: Shows peak memory broken down by tensor — activations, weights, gradients, optimizer state. Upload to `pytorch.org/memory_viz` for an interactive timeline. Essential for deciding where to apply gradient checkpointing.

```bash
# CS336 — forward only (activations only) vs full_step (+ gradients + optimizer state)
uv run python bench/benchmark.py --model_size xl --mode forward \
    --mixed_precision --memory_profile --context_length 128
uv run python bench/benchmark.py --model_size xl --mode full_step \
    --mixed_precision --memory_profile --context_length 2048

# nanochat
python -m scripts.benchmark --depth 20 --mode full_step \
    --mixed_precision --memory_profile
```

Upload the `.pickle` to [pytorch.org/memory_viz](https://pytorch.org/memory_viz) and look for:
- Activation spike during forward
- Second spike when gradients accumulate during backward
- Whether optimizer state fits within VRAM

On H100 (80 GB) all xl runs fit with room to spare. On 24 GB GPUs, xl full_step at T=2048 approaches the limit.

---

## H100 Run Order (recommended sequence)

```bash
# 1. Three-tier attention comparison — the main H100 result
uv run python bench/benchmark.py --model_size small --mode forward_backward \
    --attn_backend all --mixed_precision --context_length 2048 \
    --num_warmup 5 --num_measure 10 --output results/h100_attn.csv

# 2. Throughput at real training conditions
uv run python bench/benchmark.py --mode full_step --all_sizes \
    --mixed_precision --num_warmup 5 --num_measure 10 \
    --output results/h100_throughput.csv

# 3. Context length sweep — FA3 vs math scaling
for T in 512 1024 2048; do
  uv run python bench/benchmark.py --model_size small --mode forward_backward \
      --attn_backend all --mixed_precision --context_length $T \
      --output results/h100_context.csv
done

# 4. nsys trace — verify FA3 kernel is actually dispatching
nsys profile --trace=cuda,nvtx --nvtx-capture="measured_step_*" -o h100_fa3 \
  uv run python bench/benchmark.py --model_size small --mode forward_backward \
    --mixed_precision --context_length 2048 --num_warmup 3 --num_measure 1 --nvtx

# 5. torch.profiler — confirm FA3 kernel name in TensorBoard
python -m scripts.benchmark --depth 20 --mode forward_backward \
    --mixed_precision --torch_profile

# 6. BF16 vs FP32 uplift
uv run python bench/benchmark.py --model_size large --mode forward_backward \
    --num_warmup 5 --num_measure 10
uv run python bench/benchmark.py --model_size large --mode forward_backward \
    --mixed_precision --num_warmup 5 --num_measure 10

# 7. Memory snapshot — full training step memory map
uv run python bench/benchmark.py --model_size xl --mode full_step \
    --mixed_precision --memory_profile --context_length 2048
```

---

## Execution Status in This Repo (`/workspace/llm`)

This section tracks what has already been run in this repository, where outputs live, and what remains.

### Completed benchmark outputs

- `results/h100_attn.csv`
  - `small`, `forward_backward`, BF16, `T=2048`, backends `fa3|flash|math`
- `results/h100_throughput.csv`
  - `full_step`, BF16, all sizes `small|medium|large|xl`
- `results/h100_context.csv`
  - `small`, `forward_backward`, BF16, `T=512|1024|2048`, backends `fa3|flash|math`
- `results/h100_context_by_size.csv`
  - Expanded context sweep by size:
    - `large`: `T=512|1024|2048` for `fa3|flash` (+ `math` at `512|1024`)
    - `xl`: `T=1024|2048` for `fa3|flash`
  - Note: `math` at `large, T=2048` OOMed (expected O(T^2) memory pressure for explicit unfused attention).
- `results/compile_compare.csv`
  - Eager vs `torch.compile(dynamic=False)` on `small|large|xl`, `forward_backward`, BF16, `T=2048`, backend `fa3`

### Completed profiling outputs

- Nsight Systems (`nsys`)
  - Main comparison traces:
    - `results/nsys/h100_fa3.nsys-rep`
    - `results/nsys/h100_sdpa.nsys-rep`
    - `results/nsys/h100_math_annotated.nsys-rep`
  - Deeper matrix traces:
    - `results/nsys/core_model/mode_forward.nsys-rep`
    - `results/nsys/core_model/mode_forward_backward.nsys-rep`
    - `results/nsys/core_model/mode_full_step.nsys-rep`
    - `results/nsys/core_model/ctx_512.nsys-rep`
    - `results/nsys/core_model/ctx_1024.nsys-rep`
    - `results/nsys/core_model/ctx_2048.nsys-rep`
    - `results/nsys/core_model/size_small.nsys-rep`
    - `results/nsys/core_model/size_large.nsys-rep`
    - `results/nsys/core_model/size_xl.nsys-rep`
  - Summary exports (generated via `nsys stats`) are present for key traces, e.g.:
    - `results/nsys/h100_fa3_stats.txt`
    - `results/nsys/h100_sdpa_stats.txt`
    - `results/nsys/h100_math_annotated_stats.txt`

- PyTorch profiler (TensorBoard/Chrome trace JSON)
  - Added repo-local support in `bench/benchmark.py` with:
    - `--torch_profile`
    - `--torch_profile_dir`
  - Traces generated under:
    - `results/torch_profiler/`
      - FA3: `small|large|xl`
      - flash: `small|large|xl`
  - Summaries:
    - `results/torch_profiler/summary.csv`
    - `results/torch_profiler/summary_compact.csv`

- Memory snapshots
  - `xl_ctx128_forward_memory.pickle`
  - `xl_ctx2048_full_step_memory.pickle`

### Code updates already applied for benchmarking/profiling

- `bench/benchmark.py`
  - Fixed NVTX hook pre-hook bug (pre-hooks now return `None` and no longer mutate module inputs).
  - Added `--compile` support using `torch.compile(model, dynamic=False)`.
  - Added `--torch_profile` and `--torch_profile_dir` support.
  - CSV output now includes a `compile` column (`eager` vs `compiled`).

---

## What is still missing (future work)

### 1) Nsight Compute (`ncu`) kernel metrics (important, currently blocked)

`ncu` runs failed with `ERR_NVGPUCTRPERM` due to GPU performance counter restrictions on the host (`RmProfilingAdminOnly: 1`).

Why this matters:
- `nsys` gives timeline and launch behavior.
- `ncu` gives kernel-level metrics needed for deep kernel tuning:
  - occupancy
  - memory throughput utilization
  - warp stall reasons
  - tensor core utilization at kernel granularity

Required host action:
- Enable non-admin perf-counter access (or profile as admin with correct driver settings), reload NVIDIA modules or reboot host, then re-run `ncu`.

### 2) Optional but useful follow-ups

- Repeat a subset of key runs 3x and aggregate mean/std for stability dashboards.
- Add compiled (`--compile`) + `nsys` traces for one representative size to compare eager vs compiled launch-gap behavior.
- If targeting low-latency serving scenarios, add batch-size sweep (`batch=1,2,4`) at fixed `T`.

### 3) Scope note

This guide contains both CS336 (`llm`) and nanochat commands.  
In this repository, only `bench/benchmark.py` paths are runnable; nanochat `scripts.benchmark` commands apply only in the nanochat repo.
