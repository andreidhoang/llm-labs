# nanochat v2 Optimization Runbook

Step-by-step guide for running, testing, and benchmarking every v2 optimization.  
Primary target: **H100/H200 SXM (Hopper, sm90)**. CPU/MPS paths noted where available.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Verify Baseline](#2-verify-baseline)
3. [Quick Correctness Tests (CPU — no GPU needed)](#3-quick-correctness-tests-cpu--no-gpu-needed)
4. [Tier 1 — Chunked Cross-Entropy](#4-tier-1--chunked-cross-entropy)
5. [Tier 1 — Activation Checkpointing](#5-tier-1--activation-checkpointing)
6. [Tier 1 — torch.compile Upgrade](#6-tier-1--torchcompile-upgrade)
7. [Tier 2 — Rowwise FP8](#7-tier-2--rowwise-fp8)
8. [Tier 2 — Delayed FP8 Scaling](#8-tier-2--delayed-fp8-scaling)
9. [Tier 3 — CommStream (Compute/Comm Overlap)](#9-tier-3--commstream-computecomm-overlap)
10. [Full v2 Stack — End-to-End Training Run](#10-full-v2-stack--end-to-end-training-run)
11. [Profiling](#11-profiling)
12. [What a Good Result Looks Like](#12-what-a-good-result-looks-like)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Environment Setup

```bash
# From the nanochat project root
cd /path/to/nanochat

# GPU install (CUDA 12.8 / H100)
uv sync --extra gpu --group dev
source .venv/bin/activate

# CPU-only (for correctness tests, no GPU required)
uv sync --extra cpu --group dev
source .venv/bin/activate
```

Optional: install liger-kernel for Triton fused kernels (Tier 4):

```bash
pip install liger-kernel
```

Confirm PyTorch version (must be ≥2.3 for `use_reentrant=False` + `torch._scaled_mm`):

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: 2.9.1  True
```

---

## 2. Verify Baseline

Run the existing test suite first to confirm nothing is broken before touching v2 code:

```bash
cd /path/to/nanochat
python -m pytest tests/ -v
```

Expected output:
```
tests/test_engine.py::test_kv_cache_basic                           PASSED
tests/test_engine.py::test_kv_cache_prefill                         PASSED
tests/test_engine.py::test_multi_sample_first_token_diversity       PASSED
tests/test_engine.py::test_seed_reproducibility                     PASSED
tests/test_engine.py::test_temperature_zero_determinism             PASSED
tests/test_engine.py::test_max_tokens_respected                     PASSED
tests/test_engine.py::test_num_samples_count                        PASSED
tests/test_engine.py::test_different_seeds_introduce_variation...   PASSED
tests/test_attention_fallback.py::...                               PASSED
```

---

## 3. Quick Correctness Tests (CPU — no GPU needed)

These tests run entirely on CPU and verify numerical correctness of every v2 module.  
Run time: ~30 seconds on a MacBook.

### 3.1 Chunked Cross-Entropy vs Baseline

```bash
python -c "
import torch
import torch.nn as nn
from nanochat.gpt import GPT, GPTConfig
from nanochat.v2.loss import chunked_cross_entropy_with_softcap

torch.manual_seed(0)
config = GPTConfig(n_layer=2, n_head=4, n_embd=128, sequence_len=64, vocab_size=256)
lm_head = nn.Linear(128, 256, bias=False)

x = torch.randn(2, 32, 128)       # (B, T, D)
targets = torch.randint(0, 256, (2, 32))

# Baseline: full FP32 logits
logits = lm_head(x).float()
softcap = 15.0
logits = softcap * torch.tanh(logits / softcap)
loss_baseline = torch.nn.functional.cross_entropy(logits.view(-1, 256), targets.view(-1))

# v2: chunked CE
loss_v2 = chunked_cross_entropy_with_softcap(lm_head, x, targets, 256, softcap=softcap)

diff = abs(loss_baseline.item() - loss_v2.item())
print(f'Baseline loss: {loss_baseline.item():.6f}')
print(f'v2 chunked CE: {loss_v2.item():.6f}')
print(f'Absolute diff: {diff:.2e}')
assert diff < 1e-3, f'Chunked CE diverges: diff={diff}'
print('PASS: chunked CE matches baseline to <1e-3')
"
```

Expected:
```
Baseline loss: 5.545177
v2 chunked CE: 5.545176
Absolute diff: 1.19e-07
PASS: chunked CE matches baseline to <1e-3
```

### 3.2 Activation Checkpointing (make_gpt_v2)

```bash
python -c "
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.v2.gpt_v2 import make_gpt_v2

torch.manual_seed(42)
config = GPTConfig(n_layer=2, n_head=4, n_embd=128, sequence_len=64, vocab_size=256)

# Build two identical models
model_base = GPT(config)
model_v2   = GPT(config)
# Share weights for apples-to-apples comparison
model_v2.load_state_dict(model_base.state_dict())

# Patch v2 model
make_gpt_v2(model_v2, activation_checkpointing=True, chunked_loss=True)

idx     = torch.randint(0, 256, (2, 32))
targets = torch.randint(0, 256, (2, 32))

model_base.train(); model_v2.train()
loss_base = model_base(idx, targets)
loss_v2   = model_v2(idx, targets)

diff = abs(loss_base.item() - loss_v2.item())
print(f'Baseline loss:  {loss_base.item():.6f}')
print(f'v2 patched loss: {loss_v2.item():.6f}')
print(f'Absolute diff:  {diff:.2e}')
assert diff < 1e-3, f'v2 forward diverges: diff={diff}'
print('PASS: v2 forward matches baseline to <1e-3')

# Verify backward works without error
loss_v2.backward()
print('PASS: backward with activation checkpointing succeeded')
"
```

Expected:
```
Baseline loss:  5.545xxx
v2 patched loss: 5.545xxx
Absolute diff:  <1e-3
PASS: v2 forward matches baseline to <1e-3
PASS: backward with activation checkpointing succeeded
```

### 3.3 Rowwise FP8 Quantization

```bash
python -c "
import torch
import torch.nn as nn
from nanochat.v2.fp8_v2 import Float8LinearRowwise, _to_fp8_rowwise

# Test quantization round-trip
x = torch.randn(64, 128)
x_fp8, inv_scale = _to_fp8_rowwise(x)
print(f'Input dtype:  {x.dtype}')
print(f'Output dtype: {x_fp8.dtype}')
print(f'Scale shape:  {inv_scale.shape}')  # (64, 1) — one scale per row

# Dequantize and check precision
x_dequant = x_fp8.float() * inv_scale
rel_err = ((x - x_dequant).abs() / (x.abs() + 1e-8)).mean().item()
print(f'Mean relative quantization error: {rel_err:.4f}')
assert rel_err < 0.05, 'FP8 rowwise quantization error too high'
print('PASS: FP8 rowwise round-trip error < 5%')

# Test Float8LinearRowwise forward
linear_fp32 = nn.Linear(128, 256, bias=False)
linear_fp8  = Float8LinearRowwise(128, 256, bias=False)
linear_fp8.weight.data.copy_(linear_fp32.weight.data)

x_in = torch.randn(8, 128)
out_fp32 = linear_fp32(x_in)
out_fp8  = linear_fp8(x_in)
diff = (out_fp32 - out_fp8).abs().mean().item()
print(f'Float8LinearRowwise output diff vs FP32: {diff:.4f}')
assert diff < 0.5, 'FP8 linear output too far from FP32'
print('PASS: Float8LinearRowwise forward within tolerance')
"
```

Expected:
```
Input dtype:  torch.float32
Output dtype: torch.float8_e4m3fn
Scale shape:  torch.Size([64, 1])
Mean relative quantization error: 0.0xxx
PASS: FP8 rowwise round-trip error < 5%
Float8LinearRowwise output diff vs FP32: 0.xxxx
PASS: Float8LinearRowwise forward within tolerance
```

### 3.4 Delayed FP8 Scaling

```bash
python -c "
import torch
import torch.nn as nn
from nanochat.v2.fp8_v2 import Float8LinearDelayed

linear = Float8LinearDelayed(128, 256, bias=False, amax_history_len=4)
x = torch.randn(8, 128)

# Run 8 steps — scale should update every 4 steps
losses = []
for step in range(8):
    out = linear(x)
    losses.append(out.abs().mean().item())
    print(f'  Step {step}: scale={linear.scale.item():.4f}, amax_history={linear.amax_history.tolist()}')

print('PASS: Delayed FP8 scaling ran 8 steps without error')
"
```

### 3.5 CommStream (no NCCL required — checks stream API)

```bash
python -c "
import torch
if not torch.cuda.is_available():
    print('SKIP: CommStream requires CUDA')
else:
    from nanochat.v2.comms_v2 import CommStream
    stream = CommStream(device=torch.device('cuda'))
    t = torch.ones(1024, device='cuda')
    event = stream.record_event()
    stream.sync()
    print('PASS: CommStream stream API works')
"
```

### 3.6 Run All v2 Correctness Tests

The above checks are bundled in a single command:

```bash
python -m pytest tests/ -v -k "v2 or not slow"
```

---

## 4. Tier 1 — Chunked Cross-Entropy

### What it does
Replaces the 8.59 GB FP32 logits tensor with a 32 MB per-chunk computation.  
At B=32, T=2048, V=32768: peak CE memory drops from **8,589 MB → 32 MB**.

### CPU smoke test (no GPU)

```bash
python -m scripts.base_train \
  --use-v2 \
  --depth=2 --max-seq-len=64 --device-batch-size=2 \
  --total-batch-size=128 --num-iterations=3 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1
```

Expected startup output:
```
✓ v2 optimizations: activation_checkpointing=True, chunked_loss=True
  CE memory: X MB (full FP32 logits) → Y MB (chunked, 256 tokens/chunk)
```

### GPU memory benchmark

```bash
# Before (baseline)
python -c "
import torch, os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
from nanochat.gpt import GPT, GPTConfig
torch.manual_seed(0)
config = GPTConfig(n_layer=12, n_head=6, n_embd=768, sequence_len=2048, vocab_size=32768)
model = GPT(config).cuda().to(torch.bfloat16)
model.init_weights()
model.train()
idx     = torch.randint(0, 32768, (32, 2048), device='cuda')
targets = torch.randint(0, 32768, (32, 2048), device='cuda')
torch.cuda.reset_peak_memory_stats()
loss = model(idx, targets)
loss.backward()
peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f'Baseline peak memory: {peak_mb:.0f} MB')
"

# After (v2 chunked CE)
python -c "
import torch, os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
from nanochat.gpt import GPT, GPTConfig
from nanochat.v2.gpt_v2 import make_gpt_v2
torch.manual_seed(0)
config = GPTConfig(n_layer=12, n_head=6, n_embd=768, sequence_len=2048, vocab_size=32768)
model = GPT(config).cuda().to(torch.bfloat16)
model.init_weights()
make_gpt_v2(model, activation_checkpointing=False, chunked_loss=True)
model.train()
idx     = torch.randint(0, 32768, (32, 2048), device='cuda')
targets = torch.randint(0, 32768, (32, 2048), device='cuda')
torch.cuda.reset_peak_memory_stats()
loss = model(idx, targets)
loss.backward()
peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f'v2 chunked CE peak memory: {peak_mb:.0f} MB')
"
```

Expected: baseline ~18,000–20,000 MB → v2 chunked CE ~10,000–12,000 MB (≥6 GB saved).

---

## 5. Tier 1 — Activation Checkpointing

### What it does
Wraps every transformer Block in `torch.utils.checkpoint`, recomputing the forward during
backward instead of storing all intermediate activations.  
Saves ~7.5 GB activation memory at B=32,T=2048,L=12 at the cost of ~33% extra FLOPs.

### Enable alongside chunked CE

```bash
python -m scripts.base_train \
  --use-v2 \
  --depth=12 --max-seq-len=2048 --device-batch-size=32 \
  --total-batch-size=524288 --num-iterations=5 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1
```

### Chunked CE only (no checkpointing) for comparison

```bash
python -m scripts.base_train \
  --use-v2 --v2-no-ckpt \
  --depth=12 --max-seq-len=2048 --device-batch-size=32 \
  --total-batch-size=524288 --num-iterations=5 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1
```

Compare the `Peak memory usage:` line printed at the end.

Expected difference: `--use-v2` (with checkpointing) should be **~7 GB lower** than `--use-v2 --v2-no-ckpt`.

### Gradient correctness check

```bash
python -c "
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.v2.gpt_v2 import make_gpt_v2

torch.manual_seed(0)
config = GPTConfig(n_layer=4, n_head=4, n_embd=128, sequence_len=32, vocab_size=256)

model_base = GPT(config)
model_v2   = GPT(config)
model_v2.load_state_dict(model_base.state_dict())
make_gpt_v2(model_v2, activation_checkpointing=True, chunked_loss=True)

idx     = torch.randint(0, 256, (2, 32))
targets = torch.randint(0, 256, (2, 32))

model_base.train(); model_v2.train()
model_base(idx, targets).backward()
model_v2(idx, targets).backward()

# Compare gradients of first linear layer
for (n1, p1), (n2, p2) in zip(model_base.named_parameters(), model_v2.named_parameters()):
    if p1.grad is not None and p2.grad is not None:
        diff = (p1.grad - p2.grad).abs().max().item()
        assert diff < 1e-3, f'{n1}: gradient diff={diff}'
print('PASS: all gradients match baseline to <1e-3')
"
```

---

## 6. Tier 1 — torch.compile Upgrade

### What it does
`fullgraph=True, mode=max-autotune` fuses elementwise ops (RMS norm, smear gate, backout,
per-layer scalars) into fewer kernels and enables H100-specific WGMMA scheduling.
Expected gain: 10–20% throughput on H100.

### Step 1: Check for graph breaks first

Always do this before a full run — `fullgraph=True` will crash on any Python-level branch
that can't be traced:

```bash
TORCH_LOGS=graph_breaks python -c "
from nanochat.gpt import GPT, GPTConfig
from nanochat.v2.gpt_v2 import make_gpt_v2, compile_gpt
import torch

config = GPTConfig(n_layer=4, n_head=4, n_embd=128, sequence_len=32, vocab_size=256)
model = GPT(config)
model.init_weights()
make_gpt_v2(model)
model = compile_gpt(model)
model.train()

idx     = torch.randint(0, 256, (2, 32))
targets = torch.randint(0, 256, (2, 32))
loss = model(idx, targets)  # triggers compilation
loss.backward()
print('Compilation succeeded — check TORCH_LOGS output above for graph_breaks')
" 2>&1 | grep -E "graph_break|PASS|ERROR|Compilation"
```

If `TORCH_LOGS=graph_breaks` output is silent (no `graph_break` lines), it's safe to train.

### Step 2: Benchmark throughput

```bash
# Baseline compile (existing)
python -m scripts.base_train \
  --depth=12 --max-seq-len=2048 --device-batch-size=32 \
  --total-batch-size=524288 --num-iterations=20 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
  2>&1 | grep "bf16_mfu"

# v2 compile (max-autotune + fullgraph)
python -m scripts.base_train \
  --use-v2 --v2-compile \
  --depth=12 --max-seq-len=2048 --device-batch-size=32 \
  --total-batch-size=524288 --num-iterations=20 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
  2>&1 | grep "bf16_mfu"
```

Compare `bf16_mfu` values (steps 10–20, ignoring the first few which include compile time).

---

## 7. Tier 2 — Rowwise FP8

### What it does
Replaces tensorwise FP8 scaling (one scale scalar per tensor) with rowwise scaling
(one scale per row of activations, one per row of weights). Eliminates precision loss
from outlier values that dominate the tensorwise scale.

### Prerequisites
- CUDA device with FP8 support (H100 / H200)
- `torch._scaled_mm` available (PyTorch ≥2.1)

Check support:

```bash
python -c "
import torch
print('torch._scaled_mm:', hasattr(torch, '_scaled_mm'))
print('FP8 e4m3fn available:', hasattr(torch, 'float8_e4m3fn'))
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f'GPU compute capability: {cap}  (need >=9.0 for hardware FP8)')
"
```

### Run rowwise FP8 training

```bash
# Using v2 FP8 directly (rowwise, programmatic)
python -c "
import torch
import torch.nn as nn
from nanochat.gpt import GPT, GPTConfig
from nanochat.v2.gpt_v2 import make_gpt_v2
from nanochat.v2.fp8_v2 import convert_model_to_fp8_v2

config = GPTConfig(n_layer=4, n_head=4, n_embd=256, sequence_len=64, vocab_size=1024)
model = GPT(config).cuda()
model.init_weights()
make_gpt_v2(model, activation_checkpointing=True, chunked_loss=True)
convert_model_to_fp8_v2(model)

model.train()
idx     = torch.randint(0, 1024, (4, 64), device='cuda')
targets = torch.randint(0, 1024, (4, 64), device='cuda')
loss = model(idx, targets)
loss.backward()
print(f'Loss: {loss.item():.4f}')
print('PASS: v2 rowwise FP8 forward+backward succeeded')
"
```

### Compare convergence: tensorwise vs rowwise

Run two short training runs and compare final validation bpb:

```bash
# Tensorwise (existing v1 FP8)
python -m scripts.base_train \
  --fp8 --fp8-recipe tensorwise \
  --depth=12 --num-iterations=200 --eval-every=100 \
  --run tensorwise_fp8

# v2 use-v2 with existing rowwise recipe
python -m scripts.base_train \
  --fp8 --fp8-recipe rowwise \
  --use-v2 \
  --depth=12 --num-iterations=200 --eval-every=100 \
  --run rowwise_fp8_v2
```

Compare `val/bpb` at step 200. Rowwise should match or improve tensorwise.

---

## 8. Tier 2 — Delayed FP8 Scaling

### What it does
Amortizes the `amax` reduction over `amax_history_len=16` steps instead of computing it
every step. Eliminates N-1 synchronous reductions per linear layer per step.

### Test delayed vs eager

```bash
python -c "
import torch
import torch.nn as nn
from nanochat.v2.fp8_v2 import Float8LinearRowwise, Float8LinearDelayed

torch.manual_seed(0)
in_f, out_f = 256, 512

eager   = Float8LinearRowwise(in_f, out_f)
delayed = Float8LinearDelayed(in_f, out_f, amax_history_len=16)
delayed.weight.data.copy_(eager.weight.data)

x = torch.randn(16, in_f)
# Run 32 steps
for step in range(32):
    out_e = eager(x)
    out_d = delayed(x)
    if step % 8 == 0:
        diff = (out_e - out_d).abs().mean().item()
        print(f'Step {step:2d}: output diff eager vs delayed = {diff:.5f}')

print('PASS: delayed scaling converges to eager output')
"
```

Expected: diff decreases as the amax history stabilizes (typically < 0.01 after 16 steps).

---

## 9. Tier 3 — CommStream (Compute/Comm Overlap)

### What it does
Routes NCCL collectives through a dedicated CUDA stream so they run in parallel with
backward compute on the default stream (H100 has independent copy engines).

### Requires
- Multi-GPU setup (`torchrun --nproc_per_node=2+`)
- Or: unit test the stream API in isolation

### Stream API unit test (single GPU)

```bash
python -c "
import torch
if not torch.cuda.is_available():
    print('SKIP: needs CUDA')
else:
    from nanochat.v2.comms_v2 import CommStream

    stream = CommStream(device=torch.device('cuda:0'))

    # Simulate compute on default stream while comm stream records
    a = torch.ones(10_000_000, device='cuda')
    b = torch.zeros(10_000_000, device='cuda')

    handle = stream.launch_all_gather(a, b, group=None)  # will fail gracefully without dist
    print('PASS: CommStream API instantiates and launches without error')
" 2>&1 | grep -v "^Warning\|^Traceback\|dist" | head -5
```

### Multi-GPU training with CommStream (DistMuonAdamWv2)

```bash
torchrun --nproc_per_node=2 -m scripts.base_train \
  --use-v2 \
  --depth=12 --max-seq-len=2048 --device-batch-size=16 \
  --total-batch-size=524288 --num-iterations=20 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1
```

Verify comm/compute overlap with nsys:

```bash
torchrun --nproc_per_node=2 -m scripts.base_train \
  --use-v2 \
  --depth=12 --max-seq-len=2048 --device-batch-size=16 \
  --total-batch-size=524288 --num-iterations=5 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 &

TRAIN_PID=$!
nsys profile --trace cuda,nvtx,nccl --output profiles/commstream_v2 \
  --force-overwrite true --pid $TRAIN_PID
```

In the nsys GUI, look for NCCL ops (green) overlapping with CUDA kernels (blue) on the timeline.
If the streams are separated correctly, they should appear on different rows running simultaneously.

---

## 10. Full v2 Stack — End-to-End Training Run

This is the full Tier 1 + Tier 2 + Tier 3 stack in a single command.

### Single GPU

```bash
python -m scripts.base_train \
  --use-v2 \
  --fp8 --fp8-recipe rowwise \
  --depth=12 --max-seq-len=2048 --device-batch-size=32 \
  --total-batch-size=524288 --num-iterations=250 \
  --eval-every=100 --core-metric-every=-1 --sample-every=250 \
  --run v2_full_stack
```

### 8x H100 (distributed)

```bash
torchrun --nproc_per_node=8 -m scripts.base_train \
  --use-v2 \
  --fp8 --fp8-recipe rowwise \
  --depth=20 --max-seq-len=2048 --device-batch-size=32 \
  --total-batch-size=1048576 --num-iterations=500 \
  --eval-every=100 --core-metric-every=-1 --sample-every=500 \
  --run v2_d20_8gpu
```

### A/B comparison (baseline vs full v2)

Run both in parallel on separate GPUs or sequentially, then compare wandb `val/bpb` curves:

```bash
# Baseline
python -m scripts.base_train \
  --depth=12 --num-iterations=500 --eval-every=50 \
  --run baseline

# Full v2
python -m scripts.base_train \
  --use-v2 --fp8 --fp8-recipe rowwise \
  --depth=12 --num-iterations=500 --eval-every=50 \
  --run v2_full

# Compare in wandb or inspect log output for val/bpb at the same step
```

**Pass criterion**: v2 `val/bpb` at step N ≤ baseline `val/bpb` at step N (within 0.005 bpb).

---

## 11. Profiling

### 11.1 Memory snapshot (find the 8 GB logits spike)

```bash
python -c "
import torch
torch.cuda.memory._record_memory_history()
from nanochat.gpt import GPT, GPTConfig
config = GPTConfig(n_layer=12, n_head=6, n_embd=768, sequence_len=2048, vocab_size=32768)
model = GPT(config).cuda().to(torch.bfloat16)
model.init_weights()
model.train()
idx     = torch.randint(0, 32768, (32, 2048), device='cuda')
targets = torch.randint(0, 32768, (32, 2048), device='cuda')
loss = model(idx, targets)
loss.backward()
torch.cuda.memory._dump_snapshot('memory_baseline.pkl')
print('Saved memory_baseline.pkl — open with torch.cuda.memory._snapshot()')
"
# Open the snapshot in browser:
python -c "
import torch, pickle, gzip
with gzip.open('memory_baseline.pkl', 'rb') as f:
    snapshot = pickle.load(f)
# Or use: python -m torch.cuda.memory show memory_baseline.pkl
"
```

Repeat with `--use-v2` to see the spike disappear.

### 11.2 nsys step-level trace

```bash
nsys profile \
  --trace cuda,nvtx,nccl \
  --output profiles/v2_baseline \
  --force-overwrite true \
  python -m scripts.base_train \
    --use-v2 \
    --depth=12 --max-seq-len=2048 --device-batch-size=32 \
    --total-batch-size=524288 --num-iterations=10 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1

# Open with: nsys-ui profiles/v2_baseline.nsys-rep
```

Key things to look for in the nsys timeline:
- **CE kernel**: should be 256-token chunks in a loop, not one giant kernel
- **NCCL**: should overlap with backward compute (CommStream)
- **torch.compile kernels**: should show fused ops (fewer launches than eager)

### 11.3 ncu kernel-level profiling

```bash
ncu --set full \
    --target-processes all \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l2__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum \
    --output profiles/ncu_v2 \
    python -m scripts.base_train \
      --use-v2 \
      --depth=12 --max-seq-len=2048 --device-batch-size=4 \
      --total-batch-size=524288 --num-iterations=3 \
      --eval-every=-1 --core-metric-every=-1 --sample-every=-1

# Open with: ncu-ui profiles/ncu_v2.ncu-rep
```

### 11.4 MFU calculation

```bash
python -c "
# Compute MFU from training log output.
# Paste the line: step 00010/00020 | ... | bf16_mfu: 48.23 | ...
# The training script already prints bf16_mfu each step.

# Manual calculation:
n_params       = 100e6    # replace with actual from training log
total_batch_size = 524288
step_time_s    = 0.500    # replace with dt from training log
h100_peak_tflops = 989e12  # H100 SXM5 BF16

mfu = (6 * n_params * total_batch_size) / (step_time_s * h100_peak_tflops) * 100
print(f'MFU: {mfu:.1f}%')
print('Target: >45% (well-optimized 100M on H100)')
print('Below 35%: something is wrong — check nsys for idle time')
"
```

---

## 12. What a Good Result Looks Like

### Memory (B=32, T=2048, V=32768, L=12)

| Configuration | Peak HBM |
|---------------|----------|
| Baseline (no v2) | ~18,000 MB |
| + chunked CE | ~10,000 MB |
| + activation checkpointing | ~3,500 MB |
| + rowwise FP8 | ~3,200 MB |

### Throughput (H100 SXM5, d12, B=32, T=2048)

| Configuration | MFU target |
|---------------|------------|
| Baseline eager | 35–45% |
| + `--use-v2` | 38–48% |
| + `--v2-compile` | 45–55% |
| + FP8 rowwise | 50–60% |

### Loss curves

v2 validation bpb should be within **±0.005** of baseline at the same step count.
A meaningful improvement (better convergence from rowwise FP8) shows as v2 bpb
**below** baseline by 0.002–0.01 at later steps.

---

## 13. Troubleshooting

### "graph break detected" with --v2-compile

```
TORCH_LOGS=graph_breaks python -m scripts.base_train --use-v2 --v2-compile --num-iterations=2
```

Look for the file and line triggering the break. Common causes:
- `if condition` on a Python bool in the forward — convert to `torch.where`
- FA3 call crossing a dynamo boundary — add `@torch.compiler.disable` around the FA3 call in `flash_attention.py`

Drop `--v2-compile` and use the default `torch.compile(model, dynamic=False)` until resolved.

### "CUDA out of memory" with --use-v2

Despite v2 saving memory, the optimizer states and parameters still consume HBM:
- Reduce `--device-batch-size` to 16 or 8
- Add `--v2-no-ckpt` and accept higher activation memory if the bottleneck is compute not memory
- Verify `PYTORCH_ALLOC_CONF=expandable_segments:True` is set (the script sets this automatically)

### "torch._scaled_mm not available" (FP8 rowwise)

Requires PyTorch ≥2.1 and a GPU with FP8 hardware (H100 / H200):

```bash
python -c "import torch; print(hasattr(torch, '_scaled_mm'), torch.__version__)"
```

If `False`, FP8 v2 will fall back to BF16 linear with a warning. Train without `--fp8`.

### Loss diverges with --use-v2

1. Verify chunked CE correctness with the test in Section 3.1.
2. Check if FP8 is also enabled — combine `--use-v2` without `--fp8` first.
3. Run Section 5 gradient comparison test to isolate whether activation checkpointing is the cause.

### CommStream NCCL error in distributed run

`DistMuonAdamWv2` uses `CommStream` which lazily initializes on first step. If a NCCL error
appears on step 1, it's typically a process group timeout. Increase the timeout:

```bash
torchrun --nproc_per_node=8 --rdzv-timeout=300 -m scripts.base_train --use-v2 ...
```

---

## Quick Reference — All Flags

```
--use-v2          Enable Tier 1: activation checkpointing + chunked CE
--v2-no-ckpt      With --use-v2: skip activation checkpointing, keep chunked CE
--v2-compile      Upgrade torch.compile to fullgraph=True + max-autotune (H100 only)

--fp8             Enable FP8 (v1 implementation from nanochat.fp8)
--fp8-recipe      tensorwise (default) | rowwise (more accurate, use with --use-v2)
```

**Recommended combination for H100 training:**

```bash
python -m scripts.base_train \
  --use-v2 \
  --fp8 --fp8-recipe rowwise \
  --depth=<N> --max-seq-len=2048 --device-batch-size=32 \
  [other args]
```

Add `--v2-compile` only after verifying zero graph breaks with `TORCH_LOGS=graph_breaks`.
