# Multimodal Debug Guide — How to verify, monitor, and recover

> *Senior-researcher discipline for verifying a complex multimodal pipeline before, during, and after GPU training. Each section gives you concrete actions, not just principles.*

---

## 0. The mental framework

When something goes wrong in a 9-cell, $300, 17-hour multimodal scaling-law sweep, you have two failure modes:

1. **Loud failures** — NaN, OOM, crashes. Easy to detect, often easy to fix.
2. **Silent failures** — model trains "fine" but loss curve is suspicious, or predictions don't match reality. Hard to detect; expensive to discover late.

**Senior researcher discipline:** **make every silent failure into a loud one** by instrumenting checks that fire immediately when an invariant breaks. The whole point of the test pyramid (verify_multimodal.py + pytest + preflight + reactive sanity gates) is to convert silent → loud as early as possible.

> *"Test the assumptions you're MOST CONFIDENT about — those are the ones whose violation will surprise you most."*

---

## 1. Pre-flight: 5 layers of verification BEFORE GPU spend

Run before every GPU run:

```bash
python scripts/preflight.py             # Layers 1-4 (CPU, ~1 min)
python scripts/preflight.py --gpu       # + Layer 5 (real SigLIP2 download, ~5 min)
```

| Layer | What it checks | Catches |
|---|---|---|
| **L1 Static** | Imports + config JSONs valid + dirs writable | Typos, missing deps, syntax errors |
| **L2 Unit** | 21 verify_multimodal checks + 49 pytest tests | Function-level bugs in multimodal code |
| **L3 Integration** | Full multimodal forward + backward + scatter | Wiring bugs (e.g., 3D MRoPE not actually used) |
| **L4 Determinism** | Same seed → bit-exact data | Dataloader nondeterminism (silent killer) |
| **L5 GPU smoke** | Real SigLIP2 forward on H100 | HF download issues, FA3 missing, shape mismatch |

**Exit code 0** = cleared for GPU. **Non-zero** = blocker found, halt before spending money.

---

## 2. The 8 invariants that MUST hold (and how to check each)

Every multimodal training run must satisfy these. Verify before, monitor during.

### Invariant 1: Vision encoder is frozen

```python
# Before any forward pass, verify
from core.multimodal import _check_siglip_frozen
assert _check_siglip_frozen(model.vision_tower)  # returns True
assert all(not p.requires_grad for p in model.vision_tower.siglip.parameters())
```

**Telemetry during training:**
- `vision_tower.siglip.parameters()` should have `.grad is None` after backward — even after thousands of steps.
- If any has `.grad is not None`, vision encoder is silently training.

**How to detect silent unfreezing:**
```python
n_grads = sum(1 for p in model.vision_tower.siglip.parameters() if p.grad is not None)
assert n_grads == 0, "Vision encoder is unfrozen — gradient leak"
```

### Invariant 2: Per-token compute is iso-FLOP across G values

For MoE with `expert_hidden_dim = round(4*dim/(top_k+num_shared)/128)*128`:
- Per-token forward FLOPs should match dense baseline within ±5% (rounding noise).

**How to verify:**
```python
# Use core/model.py:GPT.estimate_flops()
flops_per_token = model.estimate_flops()
print(f"FLOPs/token: {flops_per_token:.2e}")
# For d24 dense: ~3.5e9
# For d24 G=2 MoE: ~3.5e9 (same, by iso-FLOP design)
```

If grossly different, configs aren't iso-FLOP — fit will be miscalibrated.

### Invariant 3: image_pad_mask count == vision_features count

The scatter operation requires alignment.

```python
# In scatter_vision_features:
n_pad = int(image_pad_mask.sum().item())
assert n_pad == vision_features.shape[0], "dataloader misaligned"
```

This is already ASSERTED in our code, so it's a loud failure. Good.

**Common cause when it fails:** dataloader produces image_pad runs of length X but vision encoder produces Y tokens per image. Likely:
- Mismatched merge size between dataloader and PatchMerger
- Odd grid handling differs (cropped to 26×26 in PatchMerger but dataloader still expects 27×27=729)

### Invariant 4: 3D MRoPE values differ from 1D at vision positions

If 3D MRoPE was wired but accidentally uses 1D values:
```python
# Test (already in test_multimodal_integration.py)
expected_3d_cos, _ = build_3d_mrope_for_4d_apply(position_ids, head_dim)
cached_1d_cos = model.cos[:, :T]
assert (expected_3d_cos.float() - cached_1d_cos.float()).abs().max().item() > 0.01
```

If max diff is 0, you're using 1D RoPE silently.

### Invariant 5: Routing entropy stays healthy

Per-layer routing entropy:
```python
H = -sum(p * log(p))  # over expert assignments
```

**Pass criterion:** `H_per_layer > 0.7 × log(num_experts)` (= 1.45 for 8 experts).

**Telemetry:**
```python
# In core/moe.py forward, can log:
expert_probs = F.softmax(router_logits, dim=-1)
entropy = -(expert_probs * torch.log(expert_probs + 1e-10)).sum(-1).mean()
assert entropy > 0.7 * math.log(num_experts), "routing collapse"
```

**If entropy drops:** router is collapsing onto few experts. Causes:
- Aux-loss-free bias mechanism broken
- LR too high
- Initialization issue

**Recovery:** halt, investigate router_z_loss_coef + bias_lr.

### Invariant 6: Zero dead experts

A "dead expert" = receives 0 tokens for many consecutive steps.

```python
tokens_per_expert = (router_top_k_indices == expert_id).sum()  # over recent batch
assert tokens_per_expert > 0, f"Expert {expert_id} got 0 tokens"
```

**Sustained 0 tokens** = MoE is wasting capacity. DeepSeek-V3 aux-loss-free bias should prevent this.

### Invariant 7: r_actual stays close to target (mix ratio control)

```python
# In dataloader output:
r_actual = image_pad_mask.sum() / image_pad_mask.numel()
# Rolling 100-batch mean should be |r_actual - 0.3| < 0.02
```

**If r_actual drifts:** dataloader sampler is biased. Check stride placement logic.

### Invariant 8: MFU > 25%

Below this, infrastructure is broken (network, kernel selection, OOM-induced retries).

```python
# Already logged by base_train.py
mfu = (flops_per_step / step_time) / peak_flops
```

---

## 3. Common failure modes + diagnoses

Catalog of "what does the symptom look like → what's the cause."

### Failure A: F1.s loss > 2.5 at end of training

**Symptoms:** Loss never drops below 2 even after 1000 steps. NaN or Inf possible.

**Likely causes (in order of probability):**

1. **HP transfer broken** (highest probability ~50%):
   - Karpathy's HPs were tuned at his Lambda 8×H100 + his code. Our setup may differ.
   - **Fix:** run `$50 mini-LR sweep` per spec §11 Risk SL1.
   
2. **Multimodal pipeline bug** (30%):
   - Check `routing/entropy_per_layer` — if unhealthy → MoE bug.
   - Check `image_pad_mask` count vs `vision_features` shape — if mismatched, scatter would have asserted earlier.
   - Check W&B `loss/text` and `loss/vision` separately — if `loss/text` normal but `loss/vision` huge, vision context is hurting.

3. **Initialization issue** (10%):
   - Check `model.transformer.wte.weight.std()` — should be ~1.0 (per Karpathy's init).
   - Check `model.lm_head.weight.std()` — should be ~0.001.

4. **Data pipeline bug** (10%):
   - Verify `r_actual` ≈ 0.3 (W&B logs).
   - Verify ClimbMix shards loaded (data has expected token distribution).

**Diagnostic sequence:**
```bash
# Look at loss trajectory (W&B export)
# 1. Did loss start at expected value (~10 for text, similar for multimodal at init)?
# 2. Did it decrease initially and then plateau? (HP issue)
# 3. Did it spike to NaN at some step? (numerical issue, gradient explosion)
```

### Failure B: routing entropy collapse

**Symptoms:** `routing/entropy_per_layer` drops below 1.0 (was ~2.0 at init).

**Cause:** router converges onto 1-2 experts, ignoring others. DeepSeek-V3 aux-loss-free bias should prevent but can fail at small scale or with bad LR.

**Recovery:**
- Stop training (cancel trigger should fire automatically).
- Check `router_z_loss_coef` — should be 1e-4 (nanochat default).
- Add aux-loss `coef=0.01 * H_imbalance` if needed.

### Failure C: dead expert (one expert gets 0 tokens)

**Symptoms:** routing/n_dead_per_layer > 0 sustained.

**Cause:** expert bias not nudging strongly enough, or LR too low for routing weights.

**Recovery:**
- Check `expert_bias_lr` — should be 1e-3 per nanochat default.
- If only 1 expert dead in 1 layer: tolerable; mark in writeup.
- If multiple: halt, investigate.

### Failure D: predicted G3 loss off by >0.1 (Gate D fail)

**Symptoms:** Phase 1 fit was clean but G3 actual diverges.

**Diagnoses (in order of likelihood):**

1. **2-point fit underdetermined** (40%): with only 3 (C, N\*) data points, slope CI is wide. Wider extrapolation → larger error.
   - **Fix:** add 4th compute scale, refit. Cost ~$60.

2. **Hoffmann form misspecified** (25%): `L = E + A/N^α + B/D^β` may not capture our regime (e.g., logarithmic correction needed).
   - **Fix:** try `L = E + A/N^α + B/D^β + C × log(N×D)`. Or report the failure.

3. **G3 config is at envelope edge** (20%): If predicted N maps to d27 (we capped to d26), the actual cell is at edge → loss off by capping bias.
   - **Fix:** documented bias; report and accept.

4. **HP transfer subtly broken** (10%): F1.s passed but G3 at d26 might exceed validated envelope.
   - **Fix:** run d28 expansion cell, see if loss matches expected.

5. **Statistical noise** (5%): single seed run; loss has natural variance ~±0.02.

### Failure E: Phase 1 cell stalls (no progress for >5 min)

**Symptoms:** Loss not changing, MFU drops to ~0.

**Cause:**
- DDP communication hang (NCCL bug)
- Disk I/O stall (ClimbMix shard loading)
- GPU thermal throttling

**Recovery:**
- Kill the cell, check `nvidia-smi`.
- Check disk I/O: `iostat 5`.
- Restart from manifest (cell is marked "failed"; can re-submit).

### Failure F: OOM partway through training

**Symptoms:** "CUDA out of memory" mid-cell.

**Cause:** Memory allocation grows over training (e.g., gradient accumulation buffers, FA3 activation memory).

**Fix:** reduce `--device-batch-size` from 32 → 16 (or 8). Bergsma `B_opt` formula auto-adjusts micro-batch count.

### Failure G: Loss curve looks "wavy" (oscillating, not monotone)

**Symptoms:** Loss going up and down significantly each ~50 steps.

**Cause:**
- Batch size too small relative to D (gradient noise dominates)
- LR schedule cosine bottoming out too early (try `--final-lr-frac 0.1`)
- MoE router instability (loss oscillates as router reassigns)

**Fix:** increase batch size or lower LR. Often fixes itself by step 1000+.

---

## 4. Telemetry to watch DURING training

When a cell is running, monitor these in W&B (or printed to stdout):

### Headline (must check every 100 steps)

| Metric | Healthy range | Alarm threshold |
|---|---|---|
| `loss/total` | Decreasing monotonically | NaN, >2.5, plateau >1000 steps |
| `loss/text` | Within 0.1 of `loss/total` | Major divergence |
| `loss/vision` | Reasonable, with vision context | n/a (depends on data) |
| `routing/entropy_per_layer` | >1.45 (= 0.7·log(8)) | <1.0 → collapse |
| `routing/n_dead_per_layer` | 0 | >0 sustained |
| `mfu` | 30-40% | <25% sustained |
| `tokens_per_sec` | Within 110% of estimate | Big drop |
| `r_actual` | 0.28-0.32 (rolling 100 batches) | >0.5 or <0.1 drift |

### Diagnostic (check on demand if headline is suspicious)

| Metric | What it tells you |
|---|---|
| `grad_norm/total` | Optimizer health; should stabilize ~1.0 |
| `lr/muon`, `lr/adamw_emb` | Match expected schedule curve |
| `throughput/grouped_mm_overhead_ratio` | If >40%, MoE is bottlenecked (per nanochat finding) |
| `vision_tower.siglip.[any param].grad` | Should be `None` always (frozen check) |

---

## 5. Recovery procedures

### If F1.s fails reactive gate

```bash
# 1. Inspect the log
cat runs/F1_s.log | tail -100

# 2. Identify which sanity bound failed
python scripts/sweep_runner.py status
# Look at final_val_loss_joint, exit_code

# 3. Diagnose:
#    - NaN at any step → reduce LR by 2x; retry
#    - Loss > 2.5 → check HP transfer; consider mini-LR sweep
#    - Routing collapse → add aux-loss; restart cell
#    - MFU < 25% → infrastructure issue; check FA3, NCCL, disk

# 4. Once fixed, resubmit:
python scripts/sweep_runner.py submit --config configs/scaling_law/F1_s.json
```

### If a Phase 1 cell fails partway

```bash
# 1. Don't run remaining cells until you diagnose
# 2. Check manifest: status="failed" for that cell
# 3. Common: cell timed out — increase max_runtime_seconds in JSON config
# 4. Re-submit (manifest accepts re-submit)
python scripts/sweep_runner.py submit --config configs/scaling_law/F2_l.json
```

### If fit produces slope outside [0.4, 0.6]

```bash
# Gate C marginal — see plot
python scripts/sweep_runner.py plot
# Look at scaling_law_N.png for residuals

# If R² > 0.95 and slope in [0.35, 0.65]: tolerable, document caveat
# If R² < 0.95 or slope outside [0.30, 0.70]: re-run with 4th compute scale
```

### If G3 actual is way off predicted

```bash
# Don't change the prediction post-hoc; that's p-hacking
# Honest report:
echo "G3 verification: predicted=X.XXX, actual=Y.YYY, delta=Z.ZZZ"
# Diagnose per Failure D above, document in dev/LOG.md
```

---

## 6. The "is this real or am I crazy?" sanity check protocol

When a result looks "too good" or "too weird," ask:

1. **Did I see the same number twice?** (Different runs, different seeds)
2. **Does the SHAPE of the loss curve match expectations?**
3. **Did I verify GROUND TRUTH?** (e.g., parse the W&B run config, not memory)
4. **Could there be a OFF-BY-ONE bug in the data?** (e.g., shifted targets)
5. **Could there be a SILENT type cast?** (e.g., int instead of float, fp16 instead of bf16)

Senior researcher discipline: **never trust a single run.** Always verify with at least one of: re-run with different seed, run with mock data and trace through, run in eager mode (not torch.compile) to print intermediate values.

---

## 7. Read-the-code-line-by-line debugging

If verify_multimodal.py + tests all pass but training behaves weirdly, the bug is somewhere we don't have a test for. Steps:

1. **Pick 1 batch, 1 forward pass.**
2. **Print intermediate tensor shapes at every step.**
3. **Compare to expected** (per `core/multimodal.py` docstrings).
4. **Save intermediate tensors** to disk; compare across runs.

Example debug script (run on 1 cell that fails):
```python
import torch
from core.dataloader import synthetic_multimodal_loader
from core.model import GPT, GPTConfig

# ... build minimal model + loader ...

inputs, targets, extras, _ = next(loader)
print(f"inputs.shape={inputs.shape}")
print(f"image_pad_mask sum: {extras['image_pad_mask'].sum().item()}")
print(f"image_grids_merged: {extras['image_grids_merged']}")

# Patch model.forward to print
def patched_forward(self, idx, *args, **kwargs):
    print(f"  forward: idx.shape={idx.shape}")
    if "pixel_values" in kwargs and kwargs["pixel_values"] is not None:
        print(f"    pixel_values.shape={kwargs['pixel_values'].shape}")
        print(f"    grid_thw={kwargs['grid_thw']}")
    return original_forward(self, idx, *args, **kwargs)

# ... patch and run ...
```

---

## 8. Cross-references

- `scripts/preflight.py` — single-command pre-flight verification
- `scripts/verify_multimodal.py` — 21 unit checks (function-level)
- `tests/test_multimodal_*.py` — 49 integration tests
- `dev/scaling_law_self_assignment.md` §4.1 — F1.s reactive sanity bounds
- `dev/scaling_law_self_assignment.md` §11 — risk model + recovery
- `dev/multimodal_spec.md` §2.5 — design first principles
- `nanochat/dev/LOG.md` 2026-02-19 — upstream MoE wall-clock reference

---

## TL;DR

**Before GPU spend:** `python scripts/preflight.py --gpu`. If exit 0, you're cleared.

**During training:** monitor W&B headline metrics; watch for invariant violations.

**After failure:** diagnose by symptom (Section 3), recover per Section 5.

**Senior researcher discipline:** make every silent failure into a loud one. The test pyramid (verify → pytest → preflight → reactive gates) exists to convert silent → loud as early as possible. **Trust your tests, don't trust your memory.**
