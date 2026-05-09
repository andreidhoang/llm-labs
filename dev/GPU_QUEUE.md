# GPU Queue

Tasks that are blocked on GPU access. Resume from here when running on a
machine with CUDA (ideally Hopper for FA3 / `_grouped_mm` / FP8). Each entry
is self-contained — should be readable cold, with enough context to start
without re-reading the whole project.

---

## P3 — `torch._grouped_mm` dispatch + throughput measurement

**Status:** designed but not implemented. `core/moe/layer.py` currently uses
the Python-loop dispatch from P1 (correct, slow at scale).

**Why blocked on GPU:** `torch._grouped_mm` is CUDA-only, bf16-only.
Implementation without testing would ship unverified code. nanochat
(`nanochat/dev/LOG.md` 2026-02-19) confirms the API is undocumented and
trial-and-error to get right.

**Goal:** Add a second dispatch backend (`grouped_mm`), prove it produces
the same outputs as the loop path within bf16 tolerance, measure throughput
delta vs both loop and dense.

### Reference algorithm (from nanochat docs/moe.md §5.4 + LOG entry)

```
# Setup: x_flat (M, n_embd), topk_idx (M, top_k), gates (M, top_k)

flat_expert = topk_idx.reshape(-1)                         # (M·top_k,)
flat_token  = arange(M).unsqueeze(1).expand(-1, top_k).reshape(-1)
flat_gates  = gates.reshape(-1)                            # (M·top_k,)

sort_idx = flat_expert.argsort(stable=True)
sorted_token  = flat_token [sort_idx]
sorted_gates  = flat_gates [sort_idx]
expert_counts = bincount(flat_expert, minlength=num_experts)

# Gather inputs in sorted order
tokens_sorted = x_flat[sorted_token]                       # (M·top_k, n_embd)

# Cumulative offsets (int32, REQUIRED dtype)
offs = expert_counts.cumsum(0).to(torch.int32)             # (num_experts,)

# Two grouped matmuls. Note .mT for col-major right operand.
h   = torch._grouped_mm(tokens_sorted, w1.mT, offs)        # (M·top_k, expert_dim)
h   = F.relu(h).square()
out = torch._grouped_mm(h, w2.mT, offs)                    # (M·top_k, n_embd)

# Weighted scatter back
weighted = out * sorted_gates.unsqueeze(-1)
output = zeros(M, n_embd)
output.scatter_add_(0, sorted_token.unsqueeze(-1).expand_as(weighted), weighted)
```

### Storage layout (already correct from P1)

`w1: (num_experts, expert_dim, n_embd)` and `w2: (num_experts, n_embd, expert_dim)`
are stored as (out, in) per expert. Their `.mT` slices have stride-1 on
the in-dim → column-major as `_grouped_mm` requires. **No layout change
needed.**

### Gotchas (from nanochat)

- bf16 only (not fp32). Cast tokens and weights to bf16 before the call.
- Right operand must be column-major; we get this via `.mT`.
- `offs` must be int32, not int64.
- `_grouped_mm` is undocumented PyTorch internal; API may change.
- FP8 is a separate `torch._scaled_grouped_mm` with per-row scaling — not
  in P3 scope (see P5 entry below).
- Padding token counts to alignment (8 for bf16) is supposed to help but
  nanochat measured a ~2pp MFU regression from the gather/scatter overhead;
  not worth implementing initially.

### Files to modify

1. **`core/model.py`**: add `moe_dispatch: str = "loop"` field to `GPTConfig`
   (allowed: `"loop"`, `"grouped"`).
2. **`core/moe/layer.py`**:
   - Refactor `forward` so the dispatch logic is one method:
     `_dispatch_loop(x_flat, topk_idx, weights)` and
     `_dispatch_grouped(x_flat, topk_idx, weights)`.
   - `forward` selects backend via `self.dispatch`.
   - Add a fallback: if `self.dispatch == "grouped"` but CUDA unavailable
     OR `_grouped_mm` doesn't exist, log a warning and use `_dispatch_loop`.
3. **`scripts/verify_core_moe.py`**: extend with C6 (equivalence) and C7
   (throughput). Both **skip with a `print` warning** when CUDA isn't
   available — don't fail.
4. **`dev/bench_moe.py`** (NEW): standalone benchmark, separate from the
   verifier. Times forward+backward at production scale (e.g., d12,
   d18 configs from `core/configs.py`). Outputs a Markdown table of
   tokens/sec, ms/step, MFU per config. Used to update `dev/LOG.md`
   with a worked-example MFU comparison entry.

### New verifier checks (P3 additions)

**C6 — Loop ≡ grouped_mm dispatch (numerical equivalence).** GPU only.
- Build two `MoELayer` instances with identical weights, one per dispatch
  backend.
- Forward both with the same input.
- Outputs must match within bf16 tolerance (`atol=1e-2`, `rtol=1e-2` —
  bf16 has ~3 decimal digits of precision; relax from P1's `1e-5` fp32).

**C7 — Throughput delta vs dense (numbers reported, no hard assertion).**
GPU only.
- For a fixed config (e.g., D12 dense vs D12_MoE), time `n_iter=20`
  forward+backward passes after a 5-iter warmup.
- Report tokens/sec and (if `get_peak_flops()` exists in `core/common.py`)
  MFU for: dense, MoE-loop, MoE-grouped.
- No assertion on the numbers — just record them. The next LOG entry
  documents what we measured and what it implies.

### Resumption checklist (when GPU lands)

1. Verify CUDA + bf16 available: `torch.cuda.is_available() and
   torch.cuda.get_device_capability() >= (8, 0)`.
2. Try the loop-only verifier first to confirm prior work still passes
   on GPU: `python scripts/verify_core_moe.py`.
3. Build the spec: `core/moe/spec_p3.md` (use spec.md and spec_p2.md as
   templates).
4. Implement per the algorithm above.
5. Run C6 first (correctness) — iterate until matches.
6. Run C7 (numbers).
7. Run `dev/bench_moe.py` at d12 and d18.
8. Add a `dev/LOG.md` entry: "P3 grouped_mm — measured X% MFU regression
   vs dense vs nanochat's 25%, here are the numbers."

### Cross-references

- `nanochat/dev/LOG.md` 2026-02-19 (the writeup; ground truth for gotchas)
- `core/docs/moe.md` §5.4 (the dispatch algorithm)
- `core/moe/spec.md` decision #7 (loop in P1, grouped_mm in P3)

---

## C7-baseline — Dense MFU baseline measurement

**Status:** action item from `dev/LOG.md` 2026-05-01 (nanochat MoE finding).

**Why blocked on GPU:** MFU is `(measured_FLOPs/sec) / (peak_FLOPs/sec)`;
both numerator and denominator are GPU-specific.

**Goal:** Establish a dense-baseline MFU for our model configs (D12, D18,
D20, D24) BEFORE the first MoE ablation runs. Without this, we can't
attribute throughput regressions to MoE specifically vs other infra
issues.

**Files:** `dev/bench_moe.py` (created in P3) handles this. Run with
`--moe=False` for each named config, record results in `dev/LOG.md`.

---

## P5 — Checkpoint round-trip with 3D weights + bias buffer

**Status:** designed only, no code.

**Why blocked on GPU:** ideally tested on multi-GPU FSDP setup, since
`expert_bias` is a non-parameter buffer and FSDP doesn't sync buffers by
default. Single-GPU may pass even with broken multi-GPU semantics.

**Goal:** verify `state_dict()` round-trip preserves:
- 3D `w1` and `w2` parameters (Muon optimizer state too).
- `router_bias` buffer (`persistent=True`, should save).
- `_token_counts` buffer (`persistent=False`, should NOT save — control
  state, reset each `update_load_balance()`).

**Files:** `scripts/verify_core_moe.py` C8 (round-trip test) +
modifications to `core/checkpoint_manager.py` if it doesn't transparently
handle 3D parameters.

---

## P5-FP8 — FP8 + MoE for verification runs

**Status:** flagged in `dev/LOG.md` 2026-05-01 nanochat entry; not designed.

**Why blocked on GPU:** FP8 needs Hopper (SM90+) AND
TransformerEngine/torchao. nanochat investigated thoroughly and
documented gaps in `nanochat/dev/moe_fp8.md`.

**Plan.md context:** §3 says "BF16 main sweep, FP8 for verification runs
only" — so this gates only the 2 final V1/V2 runs in §6, not the bulk
of the sweep.

**Approach (per nanochat):**
1. Routed experts (3D `nn.Parameter`) stay bf16 — `_grouped_mm` doesn't
   support FP8, and `_scaled_grouped_mm` requires per-row scaling that
   needs ~200 lines of custom autograd.
2. Shared expert (`nn.Linear`) gets FP8'd via standard
   `Float8Linear` swap.
3. Document the partial conversion in the relevant LOG entry.

### Resumption checklist

1. Re-read `nanochat/dev/moe_fp8.md` cover to cover.
2. Decide: implement partial FP8 (shared only) or write the full
   `_scaled_grouped_mm` autograd path. nanochat chose partial; we
   should default to that unless we have a strong reason.
3. Test on Hopper.

---

## Notes on running on a fresh GPU machine

Things that need to be true for the GPU work to function:

- `torch >= 2.5` (for `torch._grouped_mm`).
- CUDA-capable GPU with bf16 support (SM 80+ for bf16 matmul; SM 90 for
  FA3 + FP8).
- Existing `requirements.txt` — install with `pip install -r requirements.txt`.
- `core/flash_attention.py` already handles FA3 + SDPA fallback transparently.
- Confirm `COMPUTE_DTYPE` in `core/common.py` resolves to `torch.bfloat16`
  on the target GPU.
- Set `NANOCHAT_DTYPE=bfloat16` env var if auto-detection picks fp32 on
  CPU and you want to override.

For a quick sanity check on a fresh GPU:
```bash
python scripts/verify_core_moe.py    # P1+P2 should pass on GPU too
```
