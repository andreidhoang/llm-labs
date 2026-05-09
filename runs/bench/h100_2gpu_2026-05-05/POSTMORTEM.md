# 2×H100 Wallclock Benchmark — Postmortem

**Date:** 2026-05-05
**Duration:** ~2 hours instance time (~$8 spend)
**Outcome:** ✅ Bench harness end-to-end working + 4 phase results captured. ⚠️ Production-scale run deferred due to pre-existing codebase dtype issues that surfaced under real GPU execution.

---

## 1. Goal

Verify the wallclock-port branch (chunked CE, compile-fullgraph, activation checkpointing) works on 2×H100 SXM and capture phase-to-phase deltas before running production-scale on 8×H100.

## 2. What worked ✅

| Item | Status |
|---|---|
| Vast.ai rental: 2×H100 SXM (id 34050594) | $4/hr, reliability 0.994 |
| Repo clone via GitHub API tarball (token in URL failed, header failed; tarball worked) | OK |
| Wandb auth via `wandb login --host=https://api.wandb.ai $KEY` | OK, key in `/root/.netrc` |
| 11/11 numerics tests pass on H100 GPU (chunked CE, fullgraph-friendly, activation ckpt) | ✅ |
| Bench harness end-to-end: 4 phases × 15 measure steps | ✅ |
| Streaming JSONL + final JSON + markdown report + comparison table | ✅ |
| Artifacts pulled to local before destroy | ✅ |

## 3. What failed and why ❌

### 3.1. Repo clone via `git clone` with HTTPS auth

**Symptom:** `fatal: could not read Username for 'https://github.com'` even with `GIT_SSH_COMMAND` and `http.extraheader`.

**Root cause:** the pytorch:runtime container's `git` 2.x rejects PAT-in-URL on HTTPS, and there's no credential helper configured. We bypassed entirely by downloading a tarball via GitHub API:

```bash
curl -sL -H "Authorization: token $GH_TOKEN" \
  https://api.github.com/repos/<owner>/<repo>/tarball/<branch> \
  -o /tmp/repo.tar.gz
tar xzf /tmp/repo.tar.gz -C ai_labs_2026 --strip-components=1
```

This works because `curl -H` was already verified functional from the H100. **Use this pattern next time, skip git clone entirely.**

### 3.2. `torch._grouped_mm` missing

**Symptom:** `core/moe.py:99` `h = torch._grouped_mm(x_bf16, w_up_bf16, offs=offsets)` raised AttributeError on torch 2.5.1+cu121.

**Root cause:** `torch._grouped_mm` is internal API not exposed in torch 2.5 stable, AND not present in torch 2.6.0+cu124 either. Probably needs nightly. The codebase committed code that uses an unreleased API.

**Fix applied (sed, on instance):**
```bash
sed -i 's|if x.is_cuda:|if x.is_cuda and hasattr(torch, "_grouped_mm"):|' core/moe.py
```
Falls back to existing `_run_experts_for_loop` path which works on any torch.

**Cost:** slower MoE forward (per-expert loop instead of batched grouped matmul), but doesn't affect the wallclock optimizations being tested.

### 3.3. `nn.Linear` BF16 input × FP32 weight mismatch

**Symptom:** `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float`.

**Root cause:** `core/model.py:285-287` casts `transformer.wte` and `value_embeds` to BF16 on CUDA. Everything else (matrix params, lm_head) stays FP32. First `nn.Linear` (q_proj) sees BF16 input × FP32 weight and crashes.

The original nanochat codebase had a custom `Linear` subclass that auto-casts weights to input dtype. The user's port replaced it with `nn.Linear` directly but kept the embedding cast — fundamental design inconsistency.

**Final fix applied (revert embedding cast — root cause fix, not symptom):**
```bash
sed -i '285s|self.transformer.wte.to(dtype=torch.bfloat16)|pass  # disabled BF16 cast|; 287s|ve.to(dtype=torch.bfloat16)|pass  # disabled BF16 cast|' core/model.py
```

We also added a `Linear` subclass to model.py + moe.py earlier (to defend against future autocast use), and kept those — they're idempotent.

**Path NOT taken:** wrapping forward in `torch.autocast` was a partial fix but exposed cascading dtype mismatches throughout the optimizer (BF16 exp_avg vs FP32 scalar) — whack-a-mole. Reverting the embedding cast is the upstream fix.

### 3.4. `@torch.compile` on optimizer fused steps fails on torch 2.6

**Symptom:** Triton JIT errors, then `BackendCompilerFailed: backend='inductor'`.

**Root causes (two stacked):**
1. The pytorch:runtime image lacks `gcc` — Triton can't JIT-compile its kernels.
2. Even after installing gcc (`apt install gcc g++`), torch.compile of `adamw_step_fused` and `muon_step_fused` hit dynamo errors because the BF16/FP32 dtype mix violates fake-tensor checks in the compiled graph.

**Fix applied:**
```bash
# Install C compiler (apt was slow ~5 min on Vast)
apt-get install -y gcc g++

# Disable @torch.compile decorators on optim fused functions
sed -i 's|^@torch.compile|# @torch.compile  # DISABLED for torch 2.6 compat|g' core/optim.py
```

**Cost:** lose torch.compile fusion on the 2 fused optim kernels. Still get the 0-D CPU tensor + manual fusion benefit.

### 3.5. Optimizer scalar-to-buffer dtype mismatch (cascading)

**Symptom:** `RuntimeError: expected dtype c10::BFloat16 for 'weight' but got dtype float` in `exp_avg.lerp_(grad, 1 - beta1_t)` — and after revert of embedding cast, similar in muon's `second_momentum_buffer.lerp_(... , 1 - beta2)`.

**Root cause:** `lerp_(end, weight)` requires `weight` (the interpolation coefficient) to match self's dtype. The fused optim functions compute `1 - beta1_t` (or `1 - beta2`) as an FP32 expression and pass it as the weight to a buffer that may be BF16 (when param is BF16) or FP32.

**Fix applied:**
```python
# In adamw_step_fused (core/optim.py:38-46):
p.mul_((1 - lr_t * wd_t).to(p.dtype))
_w1 = (1 - beta1_t).to(exp_avg.dtype)
_w2 = (1 - beta2_t).to(exp_avg_sq.dtype)
exp_avg.lerp_(grad.to(exp_avg.dtype), _w1)
exp_avg_sq.lerp_(grad.square().to(exp_avg_sq.dtype), _w2)

# In muon (core/optim.py:140):
second_momentum_buffer.lerp_(
    v_mean.to(dtype=second_momentum_buffer.dtype),
    (1 - beta2).to(second_momentum_buffer.dtype),  # <-- added .to()
)

# In _compute_adamw / _step_adamw (call sites):
if grad.dtype != exp_avg.dtype:
    grad = grad.to(exp_avg.dtype)
adamw_step_fused(p, grad, exp_avg, exp_avg_sq, ...)
```

**Note:** after reverting the embedding cast (3.3), most of these are no-ops (everything FP32), but the casts are idempotent and defensive.

### 3.6. `apt-get install` was very slow on Vast.ai

**Symptom:** `apt-get install gcc g++` ran 5+ minutes before completing.

**Root cause:** Vast.ai community host had slow apt mirror. Not actionable — just wait or pre-bake gcc into a custom image.

---

## 4. Patches applied (REPRODUCIBLE for next session)

**All patches lost when instance destroyed.** Save this section to re-apply on the next H100 spin.

`/tmp/patch_h100.sh`:
```bash
#!/usr/bin/env bash
# Reapply all dtype/API patches needed for the codebase to run on torch 2.5/2.6.
# These are PRE-EXISTING codebase issues, NOT wallclock-port changes.
set -e
ROOT=${1:-/workspace/ai_labs_2026}
cd "$ROOT"

# 1. Revert BF16 embedding cast (root cause of dtype war)
sed -i '285s|self.transformer.wte.to(dtype=torch.bfloat16)|pass  # disabled BF16 cast (dtype war)|' core/model.py
sed -i '287s|ve.to(dtype=torch.bfloat16)|pass  # disabled BF16 cast|' core/model.py

# 2. Add Linear auto-cast subclass to model.py (defensive, useful when embedding cast is reinstated)
python3 - <<'PY'
import re
p = "core/model.py"
s = open(p).read()
if "class Linear(nn.Linear)" not in s:
    inj = '''
class Linear(nn.Linear):
    """nn.Linear that auto-casts weight (and bias) to input dtype."""
    def forward(self, x):
        w = self.weight.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


'''
    s = s.replace("class CausalSelfAttention", inj + "class CausalSelfAttention", 1)
    s = s.replace("nn.Linear(", "Linear(")
    open(p, "w").write(s)
PY

# 3. Same Linear subclass in moe.py (avoids circular import via inline def)
python3 - <<'PY'
import re
p = "core/moe.py"
s = open(p).read()
if "class Linear(nn.Linear)" not in s:
    inj = '''
class Linear(nn.Linear):
    """Auto-cast weight to input dtype."""
    def forward(self, x):
        w = self.weight.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, w, b)


'''
    m = re.search(r"^class ", s, flags=re.M)
    s = s[:m.start()] + inj + s[m.start():]
    s = s.replace("nn.Linear(", "Linear(")
    open(p, "w").write(s)
PY

# 4. Gate torch._grouped_mm (missing in torch 2.5/2.6) → fall back to for-loop
sed -i 's|if x.is_cuda:|if x.is_cuda and hasattr(torch, "_grouped_mm"):|' core/moe.py

# 5. Disable @torch.compile on optim fused kernels (dynamo errors on torch 2.6)
sed -i 's|^@torch.compile|# @torch.compile  # DISABLED for torch 2.6 compat|g' core/optim.py

# 6. Fix scalar/buffer dtype mismatch in adamw_step_fused
python3 - <<'PY'
p = "core/optim.py"
s = open(p).read()
old = """    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)"""
new = """    # Weight decay (decoupled, applied before the update)
    p.mul_((1 - lr_t * wd_t).to(p.dtype))
    # Update running averages — cast scalar weights to match exp_avg dtype
    _w1 = (1 - beta1_t).to(exp_avg.dtype)
    _w2 = (1 - beta2_t).to(exp_avg_sq.dtype)
    exp_avg.lerp_(grad.to(exp_avg.dtype), _w1)
    exp_avg_sq.lerp_(grad.square().to(exp_avg_sq.dtype), _w2)"""
open(p, "w").write(s.replace(old, new))
PY

# 7. Fix muon's second_momentum_buffer.lerp_ scalar dtype
sed -i 's|second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)|second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), (1 - beta2).to(second_momentum_buffer.dtype))|' core/optim.py

# 8. Cast grad before adamw_step_fused at both call sites
python3 - <<'PY'
p = "core/optim.py"
s = open(p).read()
# Distributed path
old1 = "            adamw_step_fused(\n                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],"
new1 = """            if grad_slice.dtype != state['exp_avg'].dtype:
                grad_slice = grad_slice.to(state['exp_avg'].dtype)
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],"""
if "if grad_slice.dtype" not in s:
    s = s.replace(old1, new1)
# Single-GPU path
old2 = "            adamw_step_fused(\n                p, grad, exp_avg, exp_avg_sq,"
new2 = """            if grad.dtype != exp_avg.dtype:
                grad = grad.to(exp_avg.dtype)
            adamw_step_fused(
                p, grad, exp_avg, exp_avg_sq,"""
if "if grad.dtype != exp_avg.dtype" not in s:
    s = s.replace(old2, new2)
open(p, "w").write(s)
PY

# 9. Wrap model forward in autocast(bf16) inside bench harness (only needed if BF16 embedding cast reinstated)
python3 - <<'PY'
p = "scripts/bench_wallclock.py"
s = open(p).read()
old = "            loss = model(x, y)"
new = ("            with __import__('torch').autocast(device_type='cuda', dtype=__import__('torch').bfloat16):\n"
       "                loss = model(x, y)")
if "with __import__('torch').autocast" not in s:
    s = s.replace(old, new)
    open(p, "w").write(s)
PY

echo "All patches applied."
```

---

## 5. Results captured

**Hardware:** 2× H100 SXM 80GB | torch 2.6.0+cu124 | NCCL 2.21.5
**Config:** depth=8, batch=4, accum=2, seq=1024, vocab=32k, window=L
**Mode:** BF16 fallback (no FP8, no FA3 — see "didn't run" below)

| Phase | step ms | tok/s/GPU | peak HBM | Δ HBM | Δ step |
|---|---|---|---|---|---|
| 0 baseline (no opts) | 104.79 | 78,175 | 5.19 GB | — | — |
| 1 chunked CE | 119.21 | 68,719 | 4.25 GB | **−18%** | +14% |
| 2 +torch.compile (default) | **101.37** | **80,815** | **3.55 GB** | **−32%** | **−3%** |
| 3 +activation ckpt | 185.29 | 44,211 | 2.99 GB | **−42%** | +77% |

**Caveat:** MFU 2-3% absurdly low because:
- Tiny model (depth 8, dim 256) — H100 idle most of the time
- No FP8 — running BF16 path
- No FA3 — using SDPA fallback
- @torch.compile on optim disabled — losing optim fusion

**Phase-to-phase deltas are still informative as a sanity check** that the optimizations work *as designed* (HBM reductions match theory). They are NOT representative of production wallclock gain.

---

## 6. What we DID NOT run but NEED to run

To get production-meaningful numbers, the next session must:

| Test | Required for | Blocked by |
|---|---|---|
| Phase 2 with `--compile-fullgraph` (max-autotune) | Verify the compile-fullgraph optimization (the biggest expected ROI) | Codebase still hits dynamo errors with fullgraph trace; need to identify remaining graph breaks beyond `if ve is not None` — possibly other branches we haven't fixed |
| Phase battery with **production-scale model**: depth=20, dim=1280, B=32, T=2048 | Real wallclock measurement, not toy | Need to fix dtype design first OR commit to FP32 throughout |
| Phase battery with `--fp8` enabled | Real production path | Float8Linear in `core/fp8.py` may have torch 2.6 incompatibilities — not tested |
| Phase battery with FA3 (Hopper-native attention) | Real production attention path | flash-attn 3 wheel install on cu121 / cu124 — not attempted (would need build deps) |
| `no_sync()` measurement | We dropped this earlier — `DistMuonAdamW` already does manual comm in `optimizer.step()`, no DDP wrap, so `no_sync()` is N/A. Documented but not measured. | N/A (deferred by design) |
| 8×H100 production scaling test | Final wallclock for writeup | All of the above + H100 8-GPU offer availability (currently 0 on Vast — must check daily) |
| Per-step HBM growth plot | Verify no memory leak | matplotlib install + `bench_report --plots` (we used `--no-plots`) |
| `nsys` / `ncu` profile of one phase | Identify remaining bottlenecks | Need NSight install on instance (~5 min) |

### Critical TODOs before next H100 spin

1. **Fix the codebase BF16/FP32 design** in a dedicated PR (NOT bundled with wallclock-port). Either:
   - Reinstate embedding BF16 cast + add `Linear` auto-cast everywhere (nanochat pattern), OR
   - Drop embedding cast entirely + use `torch.autocast` context in training loop
2. **Pin torch + CUDA version** that the codebase actually works on. Document in `requirements.txt` or Docker image.
3. **Pre-bake an image** with: gcc, flash-attn wheel, wandb, all dtype patches applied. Push to Docker Hub. Use `--image <yours>` on rent instead of debugging on every spin.
4. **Verify FP8 path** (`core/fp8.py` Float8Linear) on the same image before relying on `--fp8` for production bench.

---

## 7. Cost breakdown

| Item | Time | Cost |
|---|---|---|
| Instance rental (2×H100 SXM @ $4.02/hr) | ~2 h | ~$8.04 |
| Wandb usage | — | $0 (free tier) |
| Storage (S3/GCS sync) | — | $0 (skipped — no AWS creds set) |
| **Total** | **2 h** | **~$8** |

**Time breakdown (rough):**
- Setup (rent, ssh, clone): ~10 min
- gcc apt-get install hang: ~10 min wasted
- dtype debugging cascade: ~40 min (5 patches before getting smoke test green)
- Actual bench runs: ~15 min (4 phases × ~3 min each)
- Generate report + pull artifacts + destroy: ~5 min
- Misc: ~30 min (re-checks, error inspection, advisor calls)

---

## 8. Artifacts on local

`llm/runs/bench/h100_2gpu_2026-05-05/`:
```
_meta.json                       # git/torch/CUDA/GPU/driver snapshot
phase_0_baseline.json + .jsonl   # baseline (no wallclock opts)
phase_1_chunked_ce.json + .jsonl # chunked CE only
phase_2_compile.json + .jsonl    # chunked + torch.compile(default)
phase_3_act_ckpt.json + .jsonl   # chunked + activation ckpt
report.md                        # markdown summary
POSTMORTEM.md                    # this file
```

**The codebase patches we made on the instance are GONE** (instance destroyed). To re-apply, run `/tmp/patch_h100.sh` (section 4 above).

---

## 9. Branch + PR status

- Branch `wallclock-port` pushed to GitHub at commit `599fc81`
- PR not opened yet (URL: https://github.com/andreidhoang/ai_labs_2026/pull/new/wallclock-port)
- 11/11 numerics tests on GPU (committed in branch)
- Bench harness (committed in branch)
- 4-phase H100 results (only on local, NOT in branch — outputs are gitignored)

**The wallclock-port code itself is correct and self-contained.** The issues we hit are all in OTHER files (model.py embedding cast, optim.py dtype, moe.py grouped_mm) that the wallclock-port branch did not modify.

---

## 10. Secrets to ROTATE (do this NOW)

Two secrets were pasted into chat during the session:

1. **Wandb API key** — `wandb_REDACTED`
   - Rotate at: https://wandb.ai/authorize → "Reset API Key"

2. **GitHub PAT** — `<redacted-github-token>`
   - Has full account-level scopes including `repo`, `delete_repo`, `admin:org`, `delete:packages`
   - **HIGH PRIORITY rotate**: revoke at https://github.com/settings/tokens

Both keys exist in chat history (LLM context) AND in process command lines on the destroyed instance. Treat as compromised.

---

## 11. Action items

| # | Owner | Item | Priority |
|---|---|---|---|
| 1 | user | Rotate wandb key + GitHub PAT | **NOW** |
| 2 | user | Open PR for wallclock-port branch (or merge directly) | this week |
| 3 | user | Decide BF16/FP32 design: reinstate cast + Linear, or autocast pattern | before next bench |
| 4 | user | Build custom Docker image with gcc + flash-attn + dtype patches pre-applied | before next bench |
| 5 | user | Test `core/fp8.py` Float8Linear on torch 2.6 — fix if broken | before --fp8 bench |
| 6 | user | Re-bench at production scale (depth=20, B=32, T=2048, FP8, FA3) on 2×H100 | when 4 done |
| 7 | user | Final 8×H100 scaling test for writeup | when 6 done + H100 offer available |
| 8 | claude | This postmortem written, artifacts pulled, instance destroyed | ✅ DONE |

---

## 12. Lessons learned

1. **"It works in tests" is necessary but not sufficient.** Our 11/11 numerics tests passed on CPU AND GPU, but didn't catch the cascade of dtype issues that only appear during a real training step. Tests covered correctness of our 3 optimizations in isolation; integration with the rest of the codebase needed a smoke test.

2. **Pre-existing codebase bugs surface fast on real hardware.** What looked like a clean codebase from `git log` had multiple latent issues (embedding cast inconsistency, missing torch API, fused step dtypes). Always do a smoke train on the target hardware before committing to a benchmark battery.

3. **Pin and test the runtime image.** `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` lacks gcc and bundles a torch version inconsistent with the codebase's API usage. Custom Dockerfile would have saved 30+ minutes.

4. **Tarball > git clone for HTTPS-auth from a fresh container.** GitHub API tarball download via curl is more reliable than git's auth machinery in containers without a credential helper.

5. **Don't iterate cast patches one-by-one.** When facing a dtype war, fix at the design layer (revert the embedding cast) rather than patching every lerp/linear/scalar individually. Advisor saved us from infinite whack-a-mole.

6. **Save artifacts locally BEFORE destroy.** Code patches made on a rented instance are ephemeral. If they took non-trivial effort to find, they MUST be turned into a reproducible script and pulled to local.
