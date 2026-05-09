# llm-labs

`llm-labs` is a compact language-modeling lab inspired by Andrej Karpathy's
`nanoGPT` and `nanochat`: small enough to read end to end, but extended with
modern training and architecture experiments.

The goal is to keep the code explicit. Core model pieces live in plain PyTorch,
training scripts are inspectable, and research features are integrated without
hiding the mechanics behind a large framework.

## Highlights

- GPT-style decoder model with named nanochat-like size configurations.
- Mixture-of-Experts layers with top-k routing, shared experts, and grouped GEMM
  dispatch for efficient expert execution.
- Multimodal early-fusion path with a frozen SigLIP2 vision tower, 2x2 patch
  merging, image-token scatter, and 3D multimodal RoPE.
- FlashAttention and benchmark utilities for studying training and inference
  performance.
- Small educational implementations under `basics/` alongside fuller training
  components under `core/`.

## Layout

```text
core/      Main model, MoE, multimodal modules, data, optimizer, engine
basics/    Smaller educational implementations and notebooks
scripts/   Training, evaluation, benchmarking, and verification entry points
bench/     Focused benchmark scripts
tests/     Regression tests for model, MoE, multimodal, and kernels
```

The top-level `docs/` directory is intentionally excluded from the published
repository; the repo is meant to foreground runnable code.

## Quick Start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Run a training or benchmark script after configuring data and hardware:

```bash
python scripts/base_train.py
python scripts/bench_wallclock.py
```

## Design Notes

This project follows the spirit of `nanoGPT`: learn by making the full stack
small, direct, and hackable. The extensions explore what happens when that
minimal base grows toward current LLM systems work:

- **MoE:** replace dense MLP blocks with sparse expert computation while keeping
  per-token FLOPs controlled.
- **Multimodal:** convert image patches into LLM-width tokens and insert them
  directly into the text stream for early-fusion vision-language modeling.
- **Performance:** keep benchmark scripts close to the implementation so model
  changes can be checked against real wall-clock behavior.

## Status

Research code. Expect sharp edges, evolving APIs, and hardware-specific paths,
especially around FlashAttention, grouped GEMM, and multimodal verification.
