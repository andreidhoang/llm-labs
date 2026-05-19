# llm-labs

A compact LLM research lab inspired by [nanoGPT](https://github.com/karpathy/nanoGPT)
and [nanochat](https://github.com/karpathy/nanochat): small enough to read end
to end, but extended with modern training and architecture experiments — MoE,
multimodal, FlashAttention, and now an **agentic autoresearch loop** that
iterates on `core/` directly.

---

## ⚡ Autoresearch results (4 sessions, $19.72, 27 experiments)

![cross-session progress](dev/auto_findings/progress.png)

An AI agent iterates on `core/{model,moe,optim,...}.py` directly, running
5-minute training experiments on 1×H100/H200 and minimizing `val_bpb`. Four
overnight sessions on Vast.ai produced:

| | val_bpb | improvement | spent |
|---|---|---|---|
| MoE-on baseline | 1.1207 | — | $0 |
| Session 1 best (MoE→dense + LR tune) | 1.0626 | **−5.2%** | $4.51 |
| Session 2 best (HP fine-tune) | 1.0576 | −5.6% | $5.51 |
| Session 3 best (first core/ edits) | 1.0575 | −5.6% | $3.20 |
| Session 4 (FA3 attempt + FP8) | 1.0797 ⓘ | — (different host) | $6.50 |
| **Karpathy's published d=8 baseline** | **0.998** | (FA3, 2× tokens) | — |

ⓘ Session 4 best is on a slower physical host (~25% less throughput than
sessions 1-3). Within-session delta from same-host FA2 baseline: −0.004
val_bpb at DEPTH=10+FP8 (the sweet spot where FP8 compute savings exceed
quant overhead).

**Six Tier 2 promotion candidates surfaced** across the 4 sessions:
strong cross-scale support for `H₄` (dense beats MoE-on at d=8/5min by 4-5%),
two newly-verified load-bearing components in `core/model.py` (QK-norm,
softcap=15), `[bears on H₀]` evidence that FP8 helps at d=10+ (Tier 2-scale
validated), and the infrastructure finding that FA3 from-source build path
is too costly for Tier 1 (needs pre-built image).

**Multimodal MoE engineering also verified** ([smoke writeup](dev/auto_findings/multimodal_smoke.md)):
real SigLIP2-SO400M + MoE + 3D MRoPE + per-modality loss decomposition all
co-execute correctly on H200 (MFU 40%, mm_bpb 3.15 → 1.91 over 25 steps).
Tier 2 sweep_design.md v3 production path is ready when budget commits.

**FA3 ecosystem unblocked** ([recipe](auto/FA3_SETUP.md)): after research into
nanochat's exact env, FA3 verified working end-to-end with vanilla PyPI torch
2.9.1+cu128 + `kernels==0.11.7` + `varunneal/flash-attention-3` hub kernel.
Earlier failures were specific to NGC's custom torch ABI, NOT the FA3
ecosystem. **Surprising:** FA3 alone (187M tokens, val_bpb=1.060) is roughly
equivalent to FA2+NGC+grouped_mm (195M, 1.058) at d=8 — the Chinchilla
prediction that "FA3 closes 62% of gap to Karpathy" was overstated. The
gap to 0.998 is NOT primarily FA3.

→ **[Read the full findings](dev/auto_findings/README.md)** — per-session
writeups, knob attribution, Chinchilla-style throughput analysis, plots.

→ **[Run autoresearch yourself](auto/README.md)** — Vast.ai 1×H100 quickstart,
agent launch protocol, branch isolation design.

---

## Repository structure

```text
llm-labs/
├── core/             # Main model, MoE, multimodal, optimizer, dataloader
│                     # (the autoresearch agent edits here on auto/<tag> branches)
├── auto/             # Tier 1 autoresearch loop (port of Karpathy's autoresearch)
│   ├── prepare_auto.py   # frozen scaffold (data, tokenizer, evaluate_bpb)
│   ├── train_auto.py     # thin training driver — imports core.model.GPT
│   ├── program.md        # agent skill (the loop, research priors)
│   └── README.md         # autoresearch quickstart
├── dev/auto_findings/    # Session results + plots + cross-session memory
├── scripts/          # Training, evaluation, benchmarking entry points (Tier 2)
├── bench/            # Focused benchmark scripts
├── basics/           # Small educational implementations
├── dev/              # Sweep designs, experiment logs, findings
└── tests/            # Regression tests
```

## Highlights

- **GPT-style decoder** with nanochat-like sized configs (`core/configs.py`),
  RoPE, QK-norm, ResFormer value embeddings, sliding-window attention.
- **Mixture-of-Experts** with sigmoid-gated top-k routing, shared experts,
  DeepSeekV3-style auxiliary-loss-free load balancing, and `torch._grouped_mm`
  dispatch for fast expert execution.
- **MuonAdamW optimizer** — Polar Express orthogonalization + NorMuon variance
  reduction, with `DistMuonAdamW` distributed variant.
- **Multimodal early-fusion path** — frozen SigLIP2 vision tower, 2×2 patch
  merging, 3D multimodal RoPE, per-modality loss decomposition.
- **FlashAttention** integration (FA3 / FA2 / SDPA auto-dispatch).
- **[Two-tier research stack](dev/auto_findings/README.md):** cheap agentic
  Tier 1 ($0.30/experiment) feeds preregistered Tier 2 (`dev/sweep_design.md`,
  $32/cell on 8×H100).

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest

# One training experiment (Tier 1, 5 min on 1×H100)
python auto/prepare_auto.py            # one-time: download data + train tokenizer
python auto/train_auto.py > run.log 2>&1
grep "^val_bpb:" run.log

# Or full Tier 2 sweep (8×H100, see dev/sweep_design.md)
torchrun --nproc-per-node=8 scripts/base_train.py --depth=24 --target-flops=2e19
```

For Vast.ai cloud runs see `H100_RUNBOOK.md` (8×H100) or `auto/README.md`
(1×H100 for autoresearch).

## Design Notes

This project follows the spirit of `nanoGPT`: learn by making the full stack
small, direct, and hackable. Extensions explore what happens when that minimal
base grows toward current LLM systems work:

- **MoE:** replace dense MLP blocks with sparse expert computation while
  keeping per-token FLOPs controlled.
- **Multimodal:** convert image patches into LLM-width tokens and insert them
  directly into the text stream for early-fusion vision-language modeling.
- **Performance:** keep benchmark scripts close to the implementation so model
  changes can be checked against real wall-clock behavior.
- **Autoresearch (new):** delegate hyperparameter search and small
  architectural ablations to an LLM agent running on cheap 1×H100 instances.
  Findings flow into `core/` (single source of truth) and into Tier 2
  preregistration as priors.

## Status

Research code. Expect sharp edges, evolving APIs, and hardware-specific paths,
especially around FlashAttention, grouped GEMM, and multimodal verification.

## Acknowledgements

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — foundational
  educational GPT implementation
- [karpathy/nanochat](https://github.com/karpathy/nanochat) — full
  pretraining → SFT → RL pipeline this lab inherits from
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the
  primitive `auto/` is ported from
