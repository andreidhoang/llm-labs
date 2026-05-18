# `auto/` — Tier 1 autoresearch loop for llm-labs

Autoresearch infrastructure built **for** `core/`, not a fork of it.

An AI agent iterates on `llm-labs/core/` (model, MoE, optimizer) directly,
running 5-minute training experiments on 1×H100 and minimizing `val_bpb` on
the climbmix validation shard. This is Tier 1 of llm-labs's two-tier research
stack:

- **Tier 1 (here)** — cheap, agentic, single-GPU; explores `core/` hypothesis
  space. Branch `auto/<tag>` isolates from production.
- **Tier 2 (`dev/sweep_design.md`)** — preregistered 8-cell sweep on 8×H100;
  capability-matched ablations at d24. Runs on pinned main SHA.

Tier 1 generates priors. Tier 2 generates posteriors. Bridge:
`dev/auto_findings/<tag>.md` summarizes each session and surfaces Tier 2
promotion candidates.

> **📊 See [`dev/auto_findings/README.md`](../dev/auto_findings/README.md)
> for the results of 3 sessions, 22 experiments, $13.22 spent.** Includes
> cross-session progress plot, knob attribution waterfall, Chinchilla-style
> throughput analysis, and 5 Tier 2 promotion candidates.

## Architecture choice

llm-labs's model lives in `core/`. We do NOT duplicate `core/` into `auto/`.
The agent edits `core/{model,moe,optim,_layers,configs,dataloader}.py`
directly; `auto/train_auto.py` is a thin (~280-line) training driver that
imports `core.model.GPT` and runs the wall-clock loop.

| File | Role | Edit |
|---|---|---|
| `auto/prepare_auto.py` | Frozen scaffold (vocab=8192, MAX_SEQ_LEN=2048, TIME_BUDGET=300, evaluate_bpb) | Human only (frozen) |
| `auto/train_auto.py` | Training driver: imports `core.model.GPT`, runs 5-min wall-clock loop, prints val_bpb | Agent (hyperparameter block at top) |
| `auto/program.md` | Agent skill | Human only |
| `core/*.py` | The model itself | **Agent (this is where research happens)** |

Safety mechanism: **branch isolation** (`auto/<tag>`), not file isolation.
Tier 2 references a pinned commit SHA on `main`; the agent's edits on
`auto/<tag>` are invisible to it.

## Quick start

All commands run from the **repo root** (`llm-labs/`), not from `auto/`.

```bash
# 1. Install deps (one-time)
pip install -r requirements.txt

# 2. Download data + train 8192-vocab tokenizer (one-time, ~2 min on first run)
python auto/prepare_auto.py

# 3. Smoke a single 5-min training run, verify Karpathy-format summary
python auto/train_auto.py > run.log 2>&1
grep "^val_bpb:" run.log
```

If step 3 prints `val_bpb: 0.XXXXXX`, you're ready for agent sessions.

## Launching a Vast.ai 1×H100 instance

Adapted from `H100_RUNBOOK.md` (which targets 8×H100). 1×H100 spot ≈ $0.40/hr;
overnight session of ~100 experiments runs about $4.

```bash
vastai search offers \
  'num_gpus=1 gpu_name in [H100_SXM,H100_PCIE,H100_NVL] reliability>0.95 verified=true cuda_max_good>=12.8' \
  -o dph_total

vastai create instance <ID> \
  --image nvcr.io/nvidia/pytorch:25.03-py3 \
  --disk 100

# SSH in:
git clone https://github.com/<your-fork>/llm-labs.git && cd llm-labs
git checkout auto/<tag>           # if continuing a session; else create from main
pip install -r requirements.txt
python auto/prepare_auto.py
```

## Launching an agent session

```
Prompt to Claude Code / Codex:
  "Read auto/program.md. Use tag 2026-05-17 (or today's date). Start setup."
```

The agent will:

1. Create branch `auto/<tag>` from `main`.
2. Read `auto/program.md` and as much of `core/` as it intends to modify.
3. Initialize `auto/results.tsv` with the header row.
4. Run the baseline experiment.
5. Iterate: edit `core/` (and/or `auto/train_auto.py` hyperparameters)
   → commit → run → grep → keep-or-reset → repeat.

The agent runs autonomously until you stop it. Per `program.md`, the agent
will not pause for permission.

## After a session — write the findings

After ~8–10 hours of autonomous iteration:

1. Read `auto/results.tsv` and skim `git log auto/<tag>`.
2. Write `dev/auto_findings/<tag>.md`: top wins, patterns, Tier 2 promotion
   candidates (hypotheses flagged `[bears on H_X]` in the description column).
3. Append durable lessons to `dev/auto_findings/lessons.md` so the next
   session's agent doesn't re-discover dead ends.

## Files

```
auto/
├── prepare_auto.py    # FROZEN: data download, tokenizer training, dataloader, evaluate_bpb
├── train_auto.py      # Thin driver (~280 lines). Imports core.model.GPT.
├── program.md         # Agent skill — edit scope, research priors, the loop
├── README.md          # This file
├── .gitignore         # results.tsv, run.log untracked
└── (results.tsv)      # Per-session ledger, git-ignored. Git log IS the registry.

core/                  # The actual edit target. Lives outside auto/.
├── model.py           # GPT, attention, RoPE, value embeddings — agent edits here
├── moe.py             # MoE routing + experts — agent edits here
├── optim.py           # MuonAdamW + fused kernels — agent edits here
├── _layers.py         # auto-cast Linear
├── configs.py         # sized config presets
└── ...                # everything else is out of scope (see program.md)

dev/auto_findings/
├── lessons.md         # cross-session memory (durable findings)
└── <tag>.md           # per-session summaries (write after each overnight)
```

## Differences from Karpathy's `autoresearch`

| | Karpathy upstream | llm-labs Tier 1 |
|---|---|---|
| Model location | Flattened into `train.py` | `core/` (existing modular codebase) |
| Agent edit scope | One file (`train.py`) | `core/{model,moe,optim,...}.py` + `auto/train_auto.py` (hyperparameter block) |
| Safety mechanism | Single-file containment | Branch isolation (`auto/<tag>` ≠ Tier 2 pinned SHA) |
| Baseline | Dense GPT | **MoE-on** (NUM_EXPERTS=4, TOP_K=2, NUM_SHARED=1) |
| Vocab | 8192 | 8192 (matched; val_bpb directly comparable) |
| Time budget | 300 s | 300 s (matched) |
| Package manager | `uv` | `pip` (llm-labs convention) |
| Branch convention | `autoresearch/<tag>` | `auto/<tag>` |
| Cross-session memory | none | `dev/auto_findings/lessons.md` |
| Tier 2 cross-reference | none | `[bears on H_X]` tags in results.tsv |

Karpathy's "edit one file" rule was a containment strategy for a single-file
demo. llm-labs has a real modular `core/`. We use it as designed and let
*branch* isolation play the containment role.

## When to update `auto/program.md`

After each session, the human reviews what worked and didn't, then iterates
`program.md` itself. This is the actual research-org-code Karpathy describes:
"the program.md you converge to is what makes the next session faster."
Common updates:

- Sharpen / reorder the "Surprising candidates" list based on session findings.
- Add new "Out of scope" entries if the agent kept wandering into dead alleys.
- Tighten "Research priors" if the agent kept violating one.
- Promote findings from `lessons.md` into the priors when they're durable enough.
