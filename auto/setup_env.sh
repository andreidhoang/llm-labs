#!/usr/bin/env bash
# auto/setup_env.sh — one-shot setup for FA3-enabled autoresearch on Vast.ai
#
# Automates the recipe documented in auto/FA3_SETUP.md.
# Target image: pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
# Target host:  1×H100/H200 (sm_90)
# Wall-clock:   ~3-5 min from fresh instance to FA3 verified
#
# Idempotent: re-running is safe; checks for existing state at each step.
#
# Usage (on the rented instance):
#   bash auto/setup_env.sh
#   # then: source /workspace/nanochat/.venv/bin/activate
#
# Exit 0 = FA3 verified active. Exit 1 = setup failed (see stderr).

set -euo pipefail

# ─── Color helpers ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

step() { echo -e "\n${YELLOW}━━━ $* ━━━${RESET}"; }
ok()   { echo -e "${GREEN}✓${RESET} $*"; }
fail() { echo -e "${RED}✗${RESET} $*" >&2; exit 1; }

NANOCHAT_DIR="${NANOCHAT_DIR:-/workspace/nanochat}"
LLM_LABS_DIR="${LLM_LABS_DIR:-/workspace/llm-labs}"

# ─── Step 1: OS-level deps the runtime image lacks ─────────────────────────
step "1/5  Installing OS deps (build-essential, nvidia-cuda-toolkit, python3-dev)"
if command -v gcc >/dev/null 2>&1 && [ -f /usr/include/python3.*/Python.h ] 2>/dev/null; then
    ok "OS deps already present, skipping apt-get"
else
    apt-get update -qq
    apt-get install -y -qq build-essential nvidia-cuda-toolkit python3-dev
    ok "build-essential + nvidia-cuda-toolkit + python3-dev installed"
fi

# ─── Step 2: uv (nanochat's package manager) ───────────────────────────────
step "2/5  Installing uv"
if command -v uv >/dev/null 2>&1; then
    ok "uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv installed: $(uv --version)"
fi
# Ensure uv on PATH for subsequent invocations
export PATH="$HOME/.local/bin:$PATH"

# ─── Step 3: nanochat venv (gets torch 2.9.1+cu128 + kernels==0.11.7) ──────
step "3/5  Cloning nanochat + uv sync (torch 2.9.1+cu128 + kernels==0.11.7)"
if [ -d "$NANOCHAT_DIR/.venv" ]; then
    ok "nanochat venv already exists at $NANOCHAT_DIR/.venv, skipping clone"
else
    if [ ! -d "$NANOCHAT_DIR" ]; then
        git clone --depth 1 https://github.com/karpathy/nanochat.git "$NANOCHAT_DIR"
    fi
    cd "$NANOCHAT_DIR"
    uv sync --extra gpu
    ok "nanochat venv ready"
fi

# ─── Step 4: llm-labs extra deps (not in nanochat's set) ───────────────────
step "4/5  Installing llm-labs extras into nanochat venv"
# shellcheck disable=SC1091
source "$NANOCHAT_DIR/.venv/bin/activate"
pip install --quiet einx einops jaxtyping pyarrow filelock Jinja2 PyYAML transformers tiktoken rustbpe
ok "llm-labs extras installed"

# ─── Step 5: Verify FA3 actually loads ─────────────────────────────────────
step "5/5  Verifying FA3 path"
cd "$LLM_LABS_DIR" 2>/dev/null || {
    echo "  (warning: $LLM_LABS_DIR not present; skip FA3 verify but env is ready)"
    echo "  Clone llm-labs into $LLM_LABS_DIR, then re-run this script for full verify."
    ok "Env ready; FA3 verify deferred"
    exit 0
}

python - <<'PY'
import os, sys
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, os.getcwd())
from core.flash_attention import USE_FA3, USE_FA2, HAS_FA3, HAS_FA2
print(f"  HAS_FA3: {HAS_FA3}")
print(f"  HAS_FA2: {HAS_FA2}")
print(f"  USE_FA3: {USE_FA3}")
print(f"  USE_FA2: {USE_FA2}")
if not USE_FA3:
    print("  WARNING: FA3 not active. Check GPU SKU (must be sm_90) and kernels package version.")
    sys.exit(2)
PY

if [ $? -eq 0 ]; then
    ok "FA3 verified active"
else
    echo -e "${YELLOW}!${RESET} FA3 not active (see warnings above). Env is otherwise ready."
fi

echo
echo -e "${GREEN}━━━ setup complete ━━━${RESET}"
echo "  Next: source $NANOCHAT_DIR/.venv/bin/activate"
echo "        cd $LLM_LABS_DIR && python auto/prepare_auto.py"
echo "        export DISABLE_GROUPED_MM=1   # only needed if running pre-refactor-A code"
echo "        python auto/train_auto.py > run.log 2>&1"
