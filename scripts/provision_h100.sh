#!/usr/bin/env bash
# Idempotent provisioning for a fresh NVIDIA NGC PyTorch container on Vast.ai.
#
# Image assumed: nvcr.io/nvidia/pytorch:25.03-py3 or newer
#   (PyTorch 2.7+, CUDA 12.8+, FA3, gcc/g++, NCCL, cuDNN, Triton — all included)
#
# What this adds on top of NGC:
#   - missing pip deps (wandb, einx, etc.)
#   - wandb login from $WANDB_API_KEY
#   - ~/.provisioned flag → re-runs are fast no-ops
#
# Usage:
#   bash provision_h100.sh                # uses env vars
#   WANDB_API_KEY=xxx bash provision_h100.sh
#
# Env vars (all optional except WANDB_API_KEY when wandb is wanted):
#   WANDB_API_KEY    paste from https://wandb.ai/authorize → wandb login (skipped if empty)
#   EXTRA_PIP        space-separated extra pip packages
#   ALLOW_GROUPED_MM_FALLBACK=1  allow correctness-only MoE smoke without torch._grouped_mm

set -euo pipefail

PROVISIONED_FLAG="$HOME/.provisioned"

if [[ -f "$PROVISIONED_FLAG" ]]; then
  echo "✓ Already provisioned ($(cat "$PROVISIONED_FLAG")). Skipping."
  exit 0
fi

echo "═══════════════════════════════════════════════════════════════════"
echo " Provisioning H100 instance for wallclock-port bench"
echo "═══════════════════════════════════════════════════════════════════"

# ── Verify hardware ──
echo
echo "▶ GPU check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "  ✗ nvidia-smi not found — wrong image?"; exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
           --format=csv,noheader,nounits | head
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
echo "  ✓ $GPU_COUNT GPU(s) detected"

# ── Verify CUDA toolchain ──
echo
echo "▶ CUDA / compiler check"
if command -v nvcc >/dev/null 2>&1; then
  echo "  ✓ nvcc: $(nvcc --version | tail -1 | awk '{print $5,$6}')"
else
  echo "  ⚠ nvcc not in PATH (some FA builds may need it; NGC usually has it)"
fi
if command -v gcc >/dev/null 2>&1; then
  echo "  ✓ gcc: $(gcc --version | head -1 | awk '{print $1,$3,$4}')"
else
  echo "  ⚠ gcc missing — installing..."
  apt-get -qq update >/dev/null && apt-get -qq install -y gcc g++ build-essential >/dev/null
  echo "  ✓ gcc installed"
fi

# ── Verify PyTorch + CUDA visibility ──
echo
echo "▶ PyTorch check"
python -c "
import torch
v = torch.__version__
cv = torch.version.cuda
nvers = '.'.join(str(x) for x in torch.cuda.nccl.version()) if torch.cuda.is_available() else 'N/A'
print(f'  torch={v} cuda={cv} nccl={nvers} n_gpu={torch.cuda.device_count()}')
has_private = hasattr(torch, '_grouped_mm')
has_public = hasattr(torch.nn.functional, 'grouped_mm')
print(f'  has _grouped_mm={has_private}')
print(f'  has torch.nn.functional.grouped_mm={has_public}')
assert torch.cuda.is_available(), 'CUDA not available'
"

if [[ "${ALLOW_GROUPED_MM_FALLBACK:-0}" != "1" ]]; then
  python -c "
import torch
assert hasattr(torch, '_grouped_mm'), (
    'torch._grouped_mm missing. This repo will fall back to a slow per-expert loop. '
    'Use nvcr.io/nvidia/pytorch:25.03-py3 or newer, and verify Vast cuda_max_good>=12.8. '
    'For correctness-only smoke tests, rerun provisioning with ALLOW_GROUPED_MM_FALLBACK=1.'
)
"
else
  echo "  ⚠ ALLOW_GROUPED_MM_FALLBACK=1: MoE grouped-MM performance gate disabled"
fi

# ── Verify FlashAttention ──
echo
echo "▶ FlashAttention check"
python -c "
try:
    import flash_attn
    print(f'  ✓ flash_attn={flash_attn.__version__}')
    # FA3 requires SM90 (Hopper)
    cap = __import__('torch').cuda.get_device_capability(0)
    print(f'  GPU capability: sm_{cap[0]}{cap[1]} {\"(Hopper, FA3 OK)\" if cap[0]>=9 else \"(non-Hopper, SDPA fallback only)\"}')
except ImportError:
    print('  ⚠ flash_attn NOT installed — would need source build (slow on runtime image)')
    print('  → If this image lacks FA3, switch to nvcr.io/nvidia/pytorch:25.03-py3')
"

# ── Install missing pip deps ──
echo
echo "▶ Pip deps"
PIP_PKGS="wandb rustbpe einx einops jaxtyping pyarrow tiktoken filelock psutil"
if [[ -n "${EXTRA_PIP:-}" ]]; then
  PIP_PKGS="$PIP_PKGS $EXTRA_PIP"
fi
# shellcheck disable=SC2086
pip install -q --no-input $PIP_PKGS 2>&1 | tail -3 || echo "  (some pkgs may already be present)"
echo "  ✓ pip deps OK"

# ── Wandb auth ──
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo
  echo "▶ Wandb login"
  wandb login --host=https://api.wandb.ai "$WANDB_API_KEY" 2>&1 | tail -2
fi

# ── Performance / NCCL env ──
echo
echo "▶ NCCL env tuning"
{
  echo "export NCCL_DEBUG=WARN"
  echo "export NCCL_ASYNC_ERROR_HANDLING=1"
  echo "export PYTORCH_ALLOC_CONF=expandable_segments:True"
  echo "export TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_cache"
} >> "$HOME/.bashrc"
mkdir -p /workspace/.inductor_cache

# ── Done ──
echo
echo "═══════════════════════════════════════════════════════════════════"
echo " ✓ Provisioning complete"
echo "═══════════════════════════════════════════════════════════════════"
date -u +"%Y-%m-%d %H:%M:%S UTC" > "$PROVISIONED_FLAG"
echo "Flag: $PROVISIONED_FLAG"
