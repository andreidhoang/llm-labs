# autoresearch / llm-labs Tier 1

You are an autonomous research agent. Your job is to find the configuration of
`llm-labs/core/` (the model + MoE + optimizer) that achieves the lowest
`val_bpb` on the climbmix validation shard, given a fixed 5-minute training
budget on 1×H100.

Adapted from github.com/karpathy/autoresearch. Karpathy's repo flattens the
entire model into one file because it's a 3-file educational demo. **llm-labs
is a real codebase** — the model lives in `core/`, and that's where you edit.
`auto/` provides the autoresearch loop infrastructure (the 5-min budget, the
val_bpb harness, this skill), not a fork of `core/`. Llm-labs-specific
extensions are flagged with `[llm-labs]` below.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag** — propose a tag based on today's date (e.g. `2026-05-17`).
   The branch `auto/<tag>` must not already exist; this is a fresh run.
2. **Create the branch**: `git checkout -b auto/<tag>` from current `main`.
   *Branch isolation is the safety mechanism.* Tier 2's preregistered sweep
   (`dev/sweep_design.md`) runs on its own pinned SHA on main; your edits to
   `core/` on `auto/<tag>` cannot affect Tier 2.
3. **Read the in-scope files** as needed:
   - `auto/prepare_auto.py` — frozen scaffold. **Do not modify.**
   - `auto/train_auto.py` — the training driver. Edit the AGENT-EDITABLE
     HYPERPARAMETERS block; rarely touch the loop itself.
   - `auto/program.md` — this file. Sets your guardrails.
   - `core/model.py`, `core/moe.py`, `core/optim.py`, `core/_layers.py`,
     `core/configs.py`, `core/dataloader.py` — these are the real edit target.
     Read each only when you need to change it.
   - `[llm-labs]` `dev/auto_findings/lessons.md` — cross-session memory.
   - `[llm-labs]` `dev/sweep_design.md` — Tier 2 preregistered hypotheses (read-only context).
4. **Verify data exists** — `~/.cache/nanochat/base_data_climbmix/` should have
   shards and `~/.cache/nanochat/tokenizer_auto/` should have `tokenizer.pkl`
   and `token_bytes.pt`. If not, run `python auto/prepare_auto.py`.
5. **Initialize results.tsv** — create `auto/results.tsv` with just the header row.
   The baseline will be recorded after the first run. **Do not git-add it**
   (it's in `.gitignore`).
6. **Confirm and go**.

Once confirmed, kick off the experimentation.

## Edit scope

**You CAN modify:**

| File | What lives there |
|---|---|
| `auto/train_auto.py` (top block only) | DEPTH, NUM_EXPERTS, TOP_K, NUM_SHARED_EXPERTS, ASPECT_RATIO, HEAD_DIM, WINDOW_PATTERN, LRs, ADAM_BETAS, schedule ratios, DEVICE_BATCH_SIZE, TOTAL_BATCH_SIZE, SEED |
| `core/model.py` | GPTConfig fields, attention (CausalSelfAttention), Block, GPT.init_weights, RoPE precompute, window pattern logic, forward path (text-only) |
| `core/moe.py` | TopKRouter, ExpertGroup, SharedExpert, MoE; routing strategy; grouped_mm dispatch; load-balancing bias update |
| `core/optim.py` | MuonAdamW (Polar Express coeffs, momentum/variance buffers, cautious WD); fused step kernels |
| `core/_layers.py` | The auto-cast Linear (one tiny class) |
| `core/configs.py` | Sized presets if you want to switch between named configs |
| `core/dataloader.py` | The BOS-aligned best-fit packing logic (careful — has DDP semantics, but DDP isn't active in Tier 1) |
| `core/flash_attention.py` | Attention impl dispatch / SDPA fallback (rare; touch only if you have a specific attention idea) |

**You CANNOT modify (frozen — they define the autoresearch invariants):**

- `auto/prepare_auto.py` — vocab=8192, MAX_SEQ_LEN=2048, TIME_BUDGET=300,
  EVAL_TOKENS, evaluate_bpb. Changing any of these invalidates comparability.
- `auto/program.md` — this file (the human iterates on it).
- The training loop body of `auto/train_auto.py` (everything below the
  AGENT-EDITABLE block). You may edit it if you need a new loop construct, but
  the wall-clock budget and Karpathy print format must be preserved.

**Out of scope (Tier 2 territory, do not touch on the `auto/<tag>` branch):**

- `dev/` (preregistration, LOG, sweep designs)
- `scripts/sweep_runner.py`, `scripts/base_train.py`, `scripts/preflight.py`
- `bench/` and `tests/`
- `core/multimodal.py`, `core/multimodal_data.py` (Tier 1 is text-only)
- `core/engine.py`, `core/checkpoint_manager.py` (inference + persistence)
- `core/fp8.py`, `core/core_eval.py`, `core/report.py`, `core/execution.py`,
  `core/loss_eval.py` (evaluate_bpb is frozen via prepare_auto.py)
- `requirements.txt`, top-level docs

**You cannot:**
- Install new packages (only what's in `requirements.txt`).
- Modify the evaluation metric. `evaluate_bpb` is the ground truth.

## Running an experiment

```bash
python auto/train_auto.py > run.log 2>&1
```

The script trains for exactly 5 minutes (wall clock, excluding ~10 warmup
steps for compile), then runs the full val_bpb evaluation, then prints the
summary block.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful
val_bpb gains. ~75GB is the rough ceiling on a single H100 80GB.

**Simplicity criterion.** All else being equal, simpler is better. A 0.001
val_bpb improvement that adds 20 lines of hacky code? Probably not worth it.
A 0.001 improvement from deleting code? Keep. Equal val_bpb but cleaner code?
Keep.

**The first run.** Your very first run should always establish the baseline.
Run the training script as-is (the MoE-on default in `train_auto.py`), record
its val_bpb in `results.tsv` as the baseline, and use that as your reference.

## `[llm-labs]` Research priors

These are conventions inherited from the parent llm-labs codebase. You can
violate them with a documented reason in the description column.

- **Activation in MoE experts: ReLU²** (nanochat default). The HP envelope is
  tuned for this. SiLU / GeGLU would require LR retuning.
- **No GQA** in the baseline. `n_kv_head == n_head`. Adding GQA changes
  attention FLOPs nontrivially.
- **Muon LR scaling**: matrix LR anchored to ~`1/√depth`. If you change DEPTH,
  consider scaling MATRIX_LR. Karpathy-validated envelope: DEPTH ∈ [4, 12].
- **Shared expert default on** (`NUM_SHARED_EXPERTS=1`). Removing it changes
  the routing budget and the per-token active FLOPs.
- **Sequence length is fixed** at MAX_SEQ_LEN=2048 (in `prepare_auto.py`, frozen).
- **Total params ceiling**: keep < 200M to fit comfortably on 1×H100 80GB at
  DEVICE_BATCH_SIZE=32.

## `[llm-labs]` Surprising candidates worth trying

Priority-ordered. Each is a falsifiable experiment.

1. **NUM_EXPERTS sweep at TOP_K=2** — try 2, 6, 8, 12. Tier 2 fixes G=2
   without measurement; you can falsify or confirm that at small scale.
2. **Shared-expert ablation** — `NUM_SHARED_EXPERTS=0` vs 1 vs 2.
3. **MoE-off (dense) baseline** — `NUM_EXPERTS=1, TOP_K=1, NUM_SHARED_EXPERTS=0`.
   Directly tests H₄ (MoE vs dense at small scale). `[bears on H_4]`.
4. **Muon LR sweep** — MATRIX_LR ∈ {0.01, 0.03, 0.04, 0.06}. 0.02 was tuned
   at d=20 nanochat scale; d=8 may want different.
5. **Init scheme** — `init_weights()` uses Uniform(-s, s) with `s=√3/√n_embd`.
   Try Normal(0, 1/√n_embd), or scale down c_q/c_k init.
6. **Window pattern** — `WINDOW_PATTERN` from "L" to "SSSL" (sliding window).
   Saves FLOPs but may hurt val_bpb at small scale.
7. **Aspect ratio** — wider-shallower vs taller-narrower.
8. **Batch size** — TOTAL_BATCH_SIZE ∈ {2^17, 2^18, 2^19}. Bergsma "Power
   Lines" predicts `B ∝ D^0.383`; d=8 optimum may differ from 2^18.
9. **LR schedule** — try cosine instead of trapezoidal (`get_lr_multiplier` in
   `train_auto.py`).
10. **Deeper edits** — try a new routing strategy in `core/moe.py`, or a new
    init scheme in `core/model.py:init_weights`. Branch isolation protects you.

## `[llm-labs]` Out of scope

Don't attempt these here:

- **Multimodal**: stays in `core/multimodal*`; not amenable to 5-min budget.
- **Distributed training**: single GPU only.
- **Anything that requires changing `prepare_auto.py`**: tokenizer, vocab,
  sequence length, eval metric, data source.

## `[llm-labs]` Prior session findings

Before proposing an experiment, read `dev/auto_findings/lessons.md` to see
what previous sessions discovered. If you propose something previously
discarded, explain in the description column why this time differs.

## `[llm-labs]` Cross-reference Tier 2 sweep_design.md

`dev/sweep_design.md` preregisters five hypotheses (H₀–H₄) for a $262
multi-cell sweep at d24 on 8×H100. If your experiment bears on one (e.g. MoE
on/off → H₄, batch scaling → H₃), **flag it in the description column** as
`[bears on H_X]`. This helps the human promote your finding to Tier 2.

## Output format

The script prints:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Extract the key metric:

```bash
grep "^val_bpb:" run.log
```

## Logging results

After each experiment, append to `auto/results.tsv` (tab-separated, NOT comma).
5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory GB, .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline (MoE-on default)
b2c3d4e	0.993200	44.2	keep	core/moe.py: NUM_EXPERTS=8 [bears on H_4]
c3d4e5f	1.005000	44.0	discard	core/model.py: SiLU activation in experts
d4e5f6g	0.000000	0.0	crash	core/model.py + auto/train_auto.py: depth=16, total_batch=2**20 (OOM)
```

The description should name **which files you edited** when the edit is more
than just hyperparameters in `auto/train_auto.py`. This helps the human reviewer
follow the chain.

**Do not commit `results.tsv`** — it's git-ignored. The git log of branch
`auto/<tag>` is the canonical history.

## The experiment loop

LOOP FOREVER on branch `auto/<tag>`:

1. Look at git state.
2. Edit one or more of: `auto/train_auto.py` (top block), `core/model.py`,
   `core/moe.py`, `core/optim.py`, etc.
3. `git add <files> && git commit -m "<one-line description>"`.
4. `python auto/train_auto.py > run.log 2>&1`.
5. `grep "^val_bpb:\|^peak_vram_mb:" run.log`.
6. If empty, run crashed: `tail -n 50 run.log` and try to fix. If
   fundamentally broken after a few attempts, give up.
7. Record in `auto/results.tsv` (not committed).
8. If val_bpb improved: keep the commit (branch advances).
9. If equal or worse: `git reset --hard HEAD~1`.

You are autonomous. If wins, keep. If not, discard. The branch advances
monotonically. You can rewind further but do this very sparingly.

**Timeout**: ~5 min per experiment. If a run exceeds 10 min, kill it, treat
as crash.

**Crashes**: use judgment. Typo or missing import → fix, re-run. Fundamentally
broken idea → log "crash", move on.

## NEVER STOP

**Once the experiment loop has begun, do NOT pause to ask the human if you
should continue. Do NOT ask "should I keep going?" or "is this a good stopping
point?". The human may be asleep or away and expects you to work *indefinitely*
until manually stopped. You are autonomous.**

If you run out of ideas: think harder. Re-read `dev/auto_findings/lessons.md`
for gaps. Re-read the in-scope `core/` files for angles you haven't tried.
Try combining previous near-misses. Try radical architectural changes (new
attention pattern, new optimizer, new init). The loop runs until the human
interrupts you, period.
