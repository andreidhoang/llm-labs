# FA3 setup — working recipe (nanochat-style)

How to get Flash Attention 3 working with llm-labs on a fresh 1×H100/H200 Vast.ai
instance, **verified end-to-end on 2026-05-19**.

The short answer: **don't use NGC images.** Karpathy's nanochat uses vanilla PyPI
torch installed via `uv`, not NGC's custom-built torch. The `varunneal/flash-attention-3`
hub kernel is built for `torch29-cxx11-cu128-x86_64-linux` ABI — matches vanilla
torch 2.9.x cu128 but **NOT** NGC's `2.8.0a0+nv25.06` custom build.

---

## The recipe (~3 min from fresh instance to FA3 verified)

```bash
# 1. Rent vanilla pytorch image on Vast.ai 1×H100/H200
vastai create instance <OFFER_ID> \
    --image pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime \
    --disk 60 --ssh --direct

# 2. SSH in, install OS-level deps that the runtime image lacks
ssh -p <PORT> root@<HOST>
apt-get update
apt-get install -y build-essential nvidia-cuda-toolkit python3-dev

# 3. Install uv (nanochat's package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

# 4. Clone nanochat (or just inherit its venv setup)
git clone https://github.com/karpathy/nanochat.git
cd nanochat
uv sync --extra gpu          # installs torch 2.9.1+cu128 + kernels 0.11.7

# 5. Activate the venv from anywhere
source /workspace/nanochat/.venv/bin/activate

# 6. Use this venv for llm-labs
cd /workspace
git clone https://github.com/andreidhoang/llm-labs.git
cd llm-labs
pip install einx einops jaxtyping pyarrow filelock Jinja2 PyYAML transformers
# (llm-labs deps not already in nanochat's set)
```

## Verify FA3 loads

```python
# From inside /workspace/llm-labs with the nanochat venv active:
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import sys
sys.path.insert(0, "/workspace/llm-labs")
from core.flash_attention import USE_FA3, USE_FA2, HAS_FA3
print(f"HAS_FA3: {HAS_FA3}")    # Expected: True
print(f"USE_FA3: {USE_FA3}")    # Expected: True
print(f"USE_FA2: {USE_FA2}")    # Expected: False
```

## Run an autoresearch experiment with FA3 active

```bash
cd /workspace/llm-labs
# ... apply session 3 winner config to auto/train_auto.py ...
export DISABLE_GROUPED_MM=1   # workaround for core/moe.py eager-mode bug (see below)
python auto/train_auto.py > run.log 2>&1
grep "^val_bpb:" run.log
```

Expected output: `val_bpb: 1.06X` at ~187M total_tokens, 5 min wall-clock, ~15% MFU
on 1×H200. Comparable to FA2 + NGC + grouped_mm result.

---

## Why our previous attempts failed

We tried FA3 in sessions 3, 4, and a diagnostic session, all on **NGC pytorch images**.
All failed with one of:

| Attempt | What failed | Real cause |
|---|---|---|
| Session 3: `kernels.get_kernel('varunneal/flash-attention-3')` on NGC 25.03 | 401 Unauthorized | the kernel wheel's ABI tag is `torch29-cxx11-cu128`; NGC's `2.7.0a0+nv25.03` is `cxx11` ABI but the custom NV build hashes differently → HF Hub returned 401 because no matching wheel was findable (or repo briefly went private during the failure window) |
| Session 4: `flash-attention/hopper/setup.py install` on NGC 25.03 | abandoned after 45 min CPU compile | source build is correct path, just expensive |
| Diagnostic on NGC 25.06: `kernels.get_kernel('kernels-community/flash-attn3')` | metadata parse error | `kernels==0.14.1` (latest on PyPI) can't parse the newer metadata.json format the hub uses |
| **THIS RECIPE** (vanilla pytorch + uv-installed kernels==0.11.7 + varunneal) | **works** | exact nanochat env: torch 2.9.1+cu128 vanilla, kernels 0.11.7 |

The fundamental issue: **NGC images bake a custom torch build whose ABI hash
doesn't match the prebuilt FA3 kernel wheels.** Karpathy avoids this by not
using NGC.

---

## Performance result — FA3 alone doesn't close the gap to Karpathy

**Surprising finding** from the verification run:

| Config | Tokens trained (5 min) | val_bpb | MFU |
|---|---|---|---|
| FA2 + NGC torch + grouped_mm + torch.compile (session 3 best) | 195M | **1.0575** | ~17% |
| FA3 + vanilla torch + for-loop MoE + torch.compile (this run) | 187M | **1.0599** | 15% |
| Karpathy's published d=8 baseline (FA3 reportedly) | 500M | 0.998 | 40% |

FA3 + vanilla torch is **roughly equivalent** to FA2 + NGC torch at d=8 / 50M params.
The Chinchilla-based prediction that "FA3 closes 62% of the gap to Karpathy" was
**overstated**. It assumed FA3 = 2× model-level throughput, but at d=8 with
seq_len=2048, attention is only a moderate fraction of total compute — FA3
speeds up that fraction but doesn't 2× the whole training step.

**The 5.7% gap to Karpathy's 0.998 is NOT primarily an FA3 issue.** Likely causes:
- inter-host throughput variance on Vast.ai (~25-35% observed across sessions)
- Karpathy's nanochat hyperparameter envelope may differ subtly from ours
- His published number may be on a different compute budget than we infer

---

## Known issues / workarounds

### Issue 1: `core/moe.py:_run_experts_grouped_mm` eager-mode bug

```python
# File: core/moe.py, line 122
x_padded = x_bf16.new_zeros(T_padded, D)
# TypeError: new_zeros() argument 'size' must be tuple of ints,
# but found element of type Tensor at pos 0
```

`T_padded = T + total_pad.sum()` is a Tensor when `total_pad` is a Tensor.
`torch.compile` masks this via tracer tensor→int conversion, but eager-mode
runs hit the error.

**Workaround:** `export DISABLE_GROUPED_MM=1` (forces for-loop expert dispatch,
which has `@torch.compiler.disable` decorator and handles this gracefully).

**Real fix (PR opportunity):** change `T_padded` to `int(T_padded.item())` or
`int(T + total_pad.sum().item())` in `core/moe.py:_run_experts_grouped_mm`.

### Issue 2: torch.compile needs `gcc`, `nvidia-cuda-toolkit`, `python3-dev`

The `pytorch/pytorch:*-runtime` images strip these for size. NGC images include
them. If you skip the `apt-get install` step above, Triton fails to build the
`cuda_utils.c` shim that torch.compile needs.

Symptom: `RuntimeError: Failed to find C compiler` or `Python.h: No such file
or directory` during torch.compile.

### Issue 3: First training step is very slow (~120 sec)

torch.compile + Triton kernel cache cold start. Steady-state is fast (~400ms/step).
Plan for this when budgeting wall-clock.

---

## When to use vanilla pytorch vs NGC

| | NGC pytorch (current default) | Vanilla pytorch (this recipe) |
|---|---|---|
| FA3 hub kernel | doesn't work (ABI mismatch) | works |
| FA2 (pip flash-attn) | pre-installed | needs pip install |
| `torch._grouped_mm` | works | works (torch 2.7+) |
| Image size | ~25 GB | ~3 GB |
| First-time setup | ~2 min (pull + pip install missing deps) | ~5 min (pull + apt + uv sync) |
| Image stability | NVIDIA QA'd | community |
| Multi-GPU networking | NCCL pre-tuned | needs tuning |
| For 8×H100 production sweeps | RECOMMENDED | possible but needs work |
| For 1×GPU autoresearch | works | RECOMMENDED if FA3 needed |

**Senior researcher's call:** keep NGC as the Tier 2 (8×H100) production default
for its NCCL/networking advantages. Use vanilla pytorch + this recipe for Tier 1
autoresearch sessions where FA3 throughput is the bottleneck.

---

## Reproducer

The full reproducer for this writeup is in
[`dev/auto_findings/lessons.md`](../dev/auto_findings/lessons.md) under the
`2026-05-19 (FA3 ecosystem — UNBLOCKED via nanochat-exact setup)` entry.

Compute spent verifying this recipe: ~$5 on 1×H200 over ~75 min wall-clock.
