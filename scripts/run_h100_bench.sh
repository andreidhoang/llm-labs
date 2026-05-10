#!/usr/bin/env bash
# One-shot orchestrator: rent N×H100 → provision → bench → pull artifacts → destroy
#
# Usage:
#   GH_TOKEN=<redacted-github-token> WANDB_API_KEY=xxx bash scripts/run_h100_bench.sh
#
# Optional env vars:
#   GPUS              number of GPUs to rent (default 2)
#   BRANCH            git branch to bench (default wallclock-port)
#   GH_REPO           owner/repo (default andreidhoang/ai_labs_2026)
#   IMAGE             Docker image to rent with (default nvcr.io/nvidia/pytorch:25.03-py3)
#   DISK              disk GB (default 80)
#   DEPTH             model depth for bench (default 12)
#   DBS               device batch size (default 8)
#   ACCUM             grad accum steps (default 2)
#   SEQ               max seq len (default 2048)
#   WANDB_PROJECT     wandb project name (default llm-wallclock-port)
#   AUTO_DESTROY      1 = destroy without prompt on success (default 1)
#   LABEL             instance label (default llm-bench-<gpus>xh100)
#
# Required env vars:
#   GH_TOKEN          GitHub PAT with repo scope (used to download tarball)
#   WANDB_API_KEY     wandb key for live monitoring (or set "" to skip wandb)
#
# Cost target: ~$5 for full 5-phase battery on 2×H100 (when next-day repeat
# uses cached autotune via TORCHINDUCTOR_CACHE_DIR persisted to S3).

set -euo pipefail

# ── Required env ──
: "${GH_TOKEN:?GH_TOKEN required (PAT with repo scope)}"
: "${WANDB_API_KEY:=}"  # optional, empty disables wandb

# ── Defaults ──
GPUS="${GPUS:-2}"
BRANCH="${BRANCH:-wallclock-port}"
GH_REPO="${GH_REPO:-andreidhoang/ai_labs_2026}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.03-py3}"
DISK="${DISK:-80}"
DEPTH="${DEPTH:-12}"
DBS="${DBS:-8}"
ACCUM="${ACCUM:-2}"
SEQ="${SEQ:-2048}"
WANDB_PROJECT="${WANDB_PROJECT:-llm-wallclock-port}"
AUTO_DESTROY="${AUTO_DESTROY:-1}"
LABEL="${LABEL:-llm-bench-${GPUS}xh100}"

LOCAL_OUT_DIR="${LOCAL_OUT_DIR:-runs/bench/h100_${GPUS}gpu_$(date -u +%Y-%m-%d_%H%M%S)}"

echo "═══════════════════════════════════════════════════════════════════"
echo " run_h100_bench.sh — full battery on ${GPUS}×H100"
echo "═══════════════════════════════════════════════════════════════════"
echo " GPUs:      $GPUS"
echo " Branch:    $BRANCH ($GH_REPO)"
echo " Image:     $IMAGE"
echo " Bench:     depth=$DEPTH dbs=$DBS accum=$ACCUM seq=$SEQ"
echo " Output:    $LOCAL_OUT_DIR/"
echo " Auto-destroy on success: $AUTO_DESTROY"
echo "═══════════════════════════════════════════════════════════════════"

# ── Find cheapest matching offer ──
# Allow OFFER_ID override (skip search) for known-good machines.
echo
if [[ -n "${OFFER_ID:-}" ]]; then
  echo "▶ Using user-provided OFFER_ID=$OFFER_ID"
  OFFER="$OFFER_ID 0.000 0.000 ? ?"  # values unknown, fill placeholders
else
  echo "▶ Searching for ${GPUS}×H100 offers (SXM preferred, PCIE/NVL fallback)..."
  OFFER=$(vastai search offers \
    "num_gpus=$GPUS gpu_name in [H100_SXM,H100_PCIE,H100_NVL] reliability>0.95 verified=true cuda_max_good>=12.8" \
    -o 'dph_total' --raw 2>/dev/null \
    | python3 -c "
import sys, json
data = json.load(sys.stdin)
if not data:
    print('NONE')
else:
    d = sorted(data, key=lambda x: x.get('dph_total', 999))[0]
    print(f\"{d['id']} {d['dph_total']:.3f} {d['reliability']:.3f} {d.get('cuda_max_good','?')} {d.get('geolocation','?')}\")
")
fi
if [[ "$OFFER" == "NONE" ]]; then
  echo "  ✗ No ${GPUS}×H100 offers right now. Try later or relax filters."
  exit 1
fi
read -r OFFER_ID OFFER_HR OFFER_REL OFFER_CUDA OFFER_LOC <<<"$OFFER"
echo "  ✓ Picked offer $OFFER_ID — \$$OFFER_HR/hr | reliab=$OFFER_REL | cuda_max=$OFFER_CUDA | $OFFER_LOC"

# ── Rent ──
# NGC pytorch:25.03+ entrypoint does not honor Vast's SSH-key injection, so
# `vastai attach ssh` succeeds at the control plane but /root/.ssh/authorized_keys
# is never written and SSH rejects every key. Fix: pass --onstart-cmd that runs
# AFTER the entrypoint, writes our local pubkey, and bounces sshd. Pass the key
# via env (-e ...) and dereference inside the onstart so we never inline the key
# into a command line that quoting could mangle.
echo
echo "▶ Renting offer $OFFER_ID..."
PUBKEY="$(cat ~/.ssh/id_ed25519.pub 2>/dev/null || true)"
if [[ -z "$PUBKEY" ]]; then
  echo "  ✗ ~/.ssh/id_ed25519.pub missing — generate one (ssh-keygen -t ed25519) and re-run."
  exit 1
fi
# base64-encode the pubkey: Vast's --env / --onstart-cmd are space-split by the
# CLI, so any literal spaces inside the key (between algo, blob, and comment)
# would corrupt parsing. b64 → one token, no quoting headaches.
PUBKEY_B64=$(printf '%s' "$PUBKEY" | base64 | tr -d '\n ')
ONSTART_CMD="mkdir -p /root/.ssh && echo $PUBKEY_B64 | base64 -d > /root/.ssh/authorized_keys && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && (service ssh restart 2>/dev/null || /etc/init.d/ssh restart 2>/dev/null || true)"
CREATE_OUT=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" --disk "$DISK" --ssh --direct --label "$LABEL" \
  --onstart-cmd "$ONSTART_CMD" 2>&1)
# vastai prints a Python dict repr (single quotes), not JSON. Match either.
INSTANCE_ID=$(echo "$CREATE_OUT" | python3 -c "
import sys, re
s = sys.stdin.read()
m = re.search(r'[\"\\']new_contract[\"\\']\s*:\s*(\d+)', s)
print(m.group(1) if m else '')")
if [[ -z "$INSTANCE_ID" ]]; then
  echo "  ✗ Rent parse failed (instance MAY still be alive — check manually): $CREATE_OUT"
  exit 1
fi
echo "  ✓ Instance $INSTANCE_ID created"

# Cleanup trap — register IMMEDIATELY after instance ID known, so any later
# failure (boot timeout, ssh fail, bench crash) still destroys the instance.
cleanup() {
  rc=$?
  if [[ "$AUTO_DESTROY" == "1" ]]; then
    echo
    echo "▶ Auto-destroy: vastai destroy instance $INSTANCE_ID"
    echo y | vastai destroy instance "$INSTANCE_ID" 2>&1 | tail -2 || true
  else
    echo
    echo "▶ Manual destroy required:  echo y | vastai destroy instance $INSTANCE_ID"
  fi
  exit $rc
}
trap cleanup EXIT

# ── Wait for boot ──
echo
echo "▶ Waiting for instance to boot (image pull may take 5-10 min on first rent)..."
until vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
      | python3 -c "import sys,json; sys.exit(0 if json.load(sys.stdin).get('actual_status')=='running' else 1)"; do
  sleep 20
  echo -n "."
done
echo
echo "  ✓ Instance running"

# ── Get SSH endpoint via `vastai ssh-url` (returns DIRECT IP, not the proxy
# `ssh_host:ssh_port` shown by `show instance` — the proxy isn't ready yet
# even after actual_status=running, the direct IP is) ──
SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)        # ssh://root@1.2.3.4:51167
SSH_HOST=$(echo "$SSH_URL" | sed -E 's|ssh://root@([^:]+):.*|\1|')
SSH_PORT=$(echo "$SSH_URL" | sed -E 's|ssh://root@[^:]+:([0-9]+)|\1|')
SSH_OPTS=(-p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10)
SCP_OPTS=(-P "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10)
echo "  SSH (direct): ssh -p $SSH_PORT root@$SSH_HOST"

# ── Wait for SSH actually accepting connections (state=running ≠ sshd ready) ──
echo
echo "▶ Probing SSH..."
for i in 1 2 3 4 5 6 7 8 9 10; do
  if ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" 'echo SSH_OK' 2>&1 | grep -q SSH_OK; then
    echo "  ✓ SSH ready (probe $i)"
    break
  fi
  if [[ $i -eq 10 ]]; then
    echo "  ✗ SSH still failing after 100s — abort"; exit 1
  fi
  sleep 10
done

# ── SCP provision script ──
echo
echo "▶ Uploading provision_h100.sh"
scp "${SCP_OPTS[@]}" scripts/provision_h100.sh "root@$SSH_HOST:/tmp/" 2>&1 | tail -1

# ── Provision ──
echo
echo "▶ Running provisioning (verifies FA3 + installs missing pip deps + wandb login)..."
ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" \
  "WANDB_API_KEY='$WANDB_API_KEY' bash /tmp/provision_h100.sh" 2>&1 | tail -30

# ── Pull repo via tarball (we proved git clone is unreliable in containers) ──
echo
echo "▶ Pulling repo $GH_REPO@$BRANCH"
ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" "
set -e
cd /workspace
[ -d ai_labs_2026 ] && rm -rf ai_labs_2026
curl -sL -H 'Authorization: token $GH_TOKEN' \
  https://api.github.com/repos/$GH_REPO/tarball/$BRANCH \
  -o /tmp/repo.tar.gz
size=\$(stat -c '%s' /tmp/repo.tar.gz)
echo \"  tarball: \${size} bytes\"
mkdir -p ai_labs_2026
tar xzf /tmp/repo.tar.gz -C ai_labs_2026 --strip-components=1
rm /tmp/repo.tar.gz
echo '  ✓ extracted'
ls ai_labs_2026/scripts/bench_wallclock.py ai_labs_2026/core/_layers.py >/dev/null
echo '  ✓ files OK'
" 2>&1 | tail -5

# ── Run bench battery ──
echo
echo "▶ Running 5-phase bench battery (depth=$DEPTH dbs=$DBS accum=$ACCUM seq=$SEQ)..."
ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" "
cd /workspace/ai_labs_2026
export WANDB_PROJECT='$WANDB_PROJECT'
NPROC=$GPUS DEPTH=$DEPTH DBS=$DBS ACCUM=$ACCUM SEQ=$SEQ \
  bash scripts/bench_run_all.sh 2>&1 | tee runs/bench/run.log
" 2>&1 | tail -60

# ── Pull artifacts ──
# scp brace expansion (`{a,b,c}`) doesn't work in remote paths — they pass
# through literally. Pull each file pattern separately or use rsync.
echo
echo "▶ Pulling artifacts to $LOCAL_OUT_DIR/"
mkdir -p "$LOCAL_OUT_DIR"
for pat in '_meta.json' 'phase_*.json' 'phase_*.jsonl' 'report.md' 'run.log' 'smoke*.json*'; do
  scp "${SCP_OPTS[@]}" "root@$SSH_HOST:/workspace/ai_labs_2026/runs/bench/$pat" \
    "$LOCAL_OUT_DIR/" 2>/dev/null || true
done
echo "  pulled $(ls -1 "$LOCAL_OUT_DIR/" 2>/dev/null | wc -l | tr -d ' ') files"

echo
echo "═══════════════════════════════════════════════════════════════════"
echo " ✓ Bench complete. Artifacts: $LOCAL_OUT_DIR/"
ls -la "$LOCAL_OUT_DIR/" 2>&1 | tail -8
echo "═══════════════════════════════════════════════════════════════════"

# Cleanup trap will destroy now
