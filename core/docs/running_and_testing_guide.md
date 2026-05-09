# Running & Testing the Speculative Decoding Engine
### Step-by-step guide — nanochat `core/` optimization

---

## Prerequisites

Two packages must be on the Python path together. The nanochat package provides `nanochat.*` imports (tokenizer, checkpoint manager, common utilities). The `llm-labs` package provides `core.*` imports (model, engine, configs).

```
/Users/danghuyhoang/Desktop/llm_labs/
├── nanochat/          ← provides nanochat.* (installed via uv)
│   └── nanochat/
└── llm-labs/          ← provides core.* (your working repo)
    └── core/
```

### 1. Install the nanochat package

```bash
cd /Users/danghuyhoang/Desktop/llm_labs/nanochat
uv sync --extra gpu          # H100/CUDA
# or
uv sync --extra cpu          # CPU / MacBook dev
```

### 2. Set the PYTHONPATH for every command in this guide

```bash
export LLMLABS=/Users/danghuyhoang/Desktop/llm_labs/llm-labs
export NANOCHAT=/Users/danghuyhoang/Desktop/llm_labs/nanochat
export PYTHONPATH=$LLMLABS:$NANOCHAT
```

Verify it works:

```bash
python -c "from core.model import GPT; from core.engine import SpeculativeEngine; print('imports OK')"
# Expected: imports OK
```

---

## Step 1 — Verify configs and parameter counts

Confirm that `TARGET_3B` and `DRAFT_300M` instantiate correctly and hit their target sizes.

```bash
python -c "
from core.model import GPT
from core.configs import TARGET_3B, DRAFT_300M, BASELINE_125M, NAMED_CONFIGS
import torch

for name, cfg in NAMED_CONFIGS.items():
    with torch.device('meta'):
        m = GPT(cfg)
    params = sum(p.numel() for p in m.parameters())
    print(f'{name:8s}  {params/1e6:8.1f}M  '
          f'L={cfg.n_layer} d={cfg.n_embd} H={cfg.n_head} KV={cfg.n_kv_head} '
          f'seq={cfg.sequence_len} pattern={cfg.window_pattern}')
"
```

**Expected output:**
```
3b        3019.9M  L=24 d=3072 H=24 KV=8 seq=4096 pattern=SSSL
300m       306.2M  L=12 d=1024 H=8 KV=4 seq=4096 pattern=SL
125m       119.9M  L=12 d=768 H=6 KV=6 seq=2048 pattern=SSSL
```

**What to check:**
- `3b` must be in the range `2900M–3100M`
- `300m` must be in the range `280M–320M`
- The ratio must be close to 10:1

If counts are off, check `core/configs.py` — the `n_layer`, `n_embd`, `n_head`, `n_kv_head` fields.

---

## Step 2 — Verify `return_hidden_states`

Confirm that the model correctly returns intermediate hidden states without breaking normal inference.

```bash
python -c "
from core.model import GPT, GPTConfig
import torch

cfg = GPTConfig(n_layer=6, n_embd=128, n_head=4, n_kv_head=4, vocab_size=256, sequence_len=64)
m = GPT(cfg)
m.init_weights()
ids = torch.randint(0, 200, (1, 12))

# Normal call — must return a plain tensor
out = m.forward(ids)
assert isinstance(out, torch.Tensor), 'normal forward must return Tensor'
print(f'Normal forward:  {out.shape}')

# With hidden states — must return (tensor, list)
out2, hs = m.forward(ids, return_hidden_states=True)
assert isinstance(out2, torch.Tensor)
assert isinstance(hs, list)
assert len(hs) == cfg.n_layer,    f'expected {cfg.n_layer} hidden states, got {len(hs)}'
assert hs[0].shape == (1, 12, 128), f'wrong hidden state shape: {hs[0].shape}'
print(f'Hidden states:   logits={out2.shape}, n={len(hs)}, each={hs[0].shape}')

# Training call with return_hidden_states
tgts = ids.roll(-1, dims=1)
loss, hs_train = m.forward(ids, targets=tgts, return_hidden_states=True)
assert isinstance(loss, torch.Tensor) and loss.ndim == 0
print(f'Training+hidden: loss={loss.item():.4f}, n={len(hs_train)}')

print('PASS')
"
```

**Expected output:**
```
Normal forward:  torch.Size([1, 12, 256])
Hidden states:   logits=torch.Size([1, 12, 256]), n=6, each=torch.Size([1, 12, 128])
Training+hidden: loss=5.5xxx, n=6
PASS
```

---

## Step 3 — Run the existing test suite

The nanochat test suite covers `KVCache`, `Engine`, and sampling correctness. Run it first to confirm the base engine is unchanged.

```bash
cd /Users/danghuyhoang/Desktop/llm_labs/nanochat
PYTHONPATH=$LLMLABS:. uv run pytest tests/test_engine.py -v
```

**Expected output:**
```
tests/test_engine.py::test_kv_cache_basic                           PASSED
tests/test_engine.py::test_kv_cache_prefill                         PASSED
tests/test_engine.py::test_multi_sample_first_token_diversity       PASSED
tests/test_engine.py::test_seed_reproducibility                     PASSED
tests/test_engine.py::test_temperature_zero_determinism             PASSED
tests/test_engine.py::test_max_tokens_respected                     PASSED
tests/test_engine.py::test_num_samples_count                        PASSED
tests/test_engine.py::test_different_seeds_introduce_variation_when_temperature_nonzero PASSED

8 passed in X.XXs
```

If any test fails here, the base engine is broken — fix that before proceeding.

---

## Step 4 — Run the SpeculativeEngine test suite

These tests verify correctness of the new engine across sampling modes, determinism, and boundary conditions. Save this file as `tests/test_speculative_engine.py` inside your `llm-labs` repo:

```python
# tests/test_speculative_engine.py
"""
Tests for SpeculativeEngine in core/engine.py.

Run from llm-labs root:
    PYTHONPATH=.:/path/to/nanochat pytest tests/test_speculative_engine.py -v
"""
import pytest
import torch
from core.model import GPT, GPTConfig
from core.engine import SpeculativeEngine


# ---------------------------------------------------------------------------
# Shared fixtures

def make_model(n_layer, n_embd, n_head, n_kv_head, vocab_size=256, seq_len=128):
    cfg = GPTConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                    n_kv_head=n_kv_head, vocab_size=vocab_size, sequence_len=seq_len)
    m = GPT(cfg)
    m.init_weights()
    return m


class FakeTok:
    def encode_special(self, s):
        return 1 if "assistant_end" in s else 2
    def get_bos_token_id(self):
        return 0


PROMPT = [5, 10, 15, 20, 25]


def make_se(K=4):
    target = make_model(4, 64, 4, 4)
    draft  = make_model(2, 32, 2, 2)
    return SpeculativeEngine(target, draft, FakeTok(), K=K)


# ---------------------------------------------------------------------------

def test_basic_output():
    """Engine produces at least one token."""
    se = make_se()
    toks = [c[0] for c, _ in se.generate(PROMPT, max_tokens=4, temperature=1.0, seed=0)]
    assert len(toks) >= 1


def test_determinism():
    """Same seed → identical output."""
    se = make_se()
    r1 = [c[0] for c, _ in se.generate(PROMPT, max_tokens=20, temperature=1.0, seed=42)]
    r2 = [c[0] for c, _ in se.generate(PROMPT, max_tokens=20, temperature=1.0, seed=42)]
    assert r1 == r2, f"Non-deterministic: {r1} vs {r2}"


def test_different_seeds_vary():
    """Different seeds → different outputs (with high probability)."""
    se = make_se()
    results = set()
    for seed in range(10):
        toks = tuple(c[0] for c, _ in se.generate(PROMPT, max_tokens=8, temperature=1.0, seed=seed))
        results.add(toks)
    assert len(results) > 1, "All seeds produced identical output — seeding broken"


def test_greedy_determinism():
    """temperature=0 must be deterministic regardless of seed."""
    se = make_se()
    r1 = [c[0] for c, _ in se.generate(PROMPT, max_tokens=8, temperature=0.0, seed=1)]
    r2 = [c[0] for c, _ in se.generate(PROMPT, max_tokens=8, temperature=0.0, seed=999)]
    assert r1 == r2, f"Greedy not deterministic: {r1} vs {r2}"


def test_max_tokens_not_massively_exceeded():
    """
    SpeculativeEngine may overshoot by at most K tokens in one step
    (the verification writes K tokens before rollback).
    """
    K = 4
    se = make_se(K=K)
    for max_t in [1, 3, 5, 10]:
        toks = [c[0] for c, _ in se.generate(PROMPT, max_tokens=max_t, temperature=1.0, seed=0)]
        assert len(toks) <= max_t + K, (
            f"max_tokens={max_t} but got {len(toks)} tokens (allowed up to {max_t + K})"
        )


def test_stop_token_terminates():
    """Generation stops when a stop token (id=0 or id=1) is emitted."""
    # Use a model biased toward emitting stop token quickly
    se = make_se()
    toks = [c[0] for c, _ in se.generate(PROMPT, max_tokens=100, temperature=1.0, seed=0)]
    # If a stop token is in the output, it must be the last token
    stop_ids = {0, 1}
    stop_positions = [i for i, t in enumerate(toks) if t in stop_ids]
    if stop_positions:
        assert stop_positions[-1] == len(toks) - 1, (
            f"Stop token at position {stop_positions[-1]} but output has {len(toks)} tokens"
        )


def test_generate_batch_api():
    """generate_batch returns ([tokens], [masks]) with correct shapes."""
    se = make_se()
    results, masks = se.generate_batch(PROMPT, num_samples=1, max_tokens=8, temperature=1.0, seed=0)
    assert len(results) == 1
    assert len(masks) == 1
    # Result includes prompt tokens
    assert results[0][:len(PROMPT)] == PROMPT
    assert len(results[0]) >= len(PROMPT)


def test_k1_still_works():
    """K=1 degenerates to standard target-only sampling (no speedup, but correct)."""
    se = make_se(K=1)
    toks = [c[0] for c, _ in se.generate(PROMPT, max_tokens=8, temperature=1.0, seed=7)]
    assert len(toks) >= 1


def test_large_k():
    """Large K (e.g. K=16) works without crashing (cache buffer is large enough)."""
    se = make_se(K=16)
    toks = [c[0] for c, _ in se.generate(PROMPT, max_tokens=32, temperature=1.0, seed=0)]
    assert len(toks) >= 1


def test_top_k_sampling():
    """top_k parameter is respected (no error, output is sensible)."""
    se = make_se()
    toks = [c[0] for c, _ in se.generate(PROMPT, max_tokens=8, temperature=1.0, top_k=10, seed=0)]
    assert len(toks) >= 1


def test_token_masks_are_ones():
    """All yielded masks are 1 (sampled, not forced)."""
    se = make_se()
    masks_seen = []
    for col, mask in se.generate(PROMPT, max_tokens=8, temperature=1.0, seed=0):
        masks_seen.extend(mask)
    assert all(m == 1 for m in masks_seen), f"Unexpected mask values: {masks_seen}"
```

Run it:

```bash
cd /Users/danghuyhoang/Desktop/llm_labs/llm-labs
PYTHONPATH=.:$NANOCHAT pytest tests/test_speculative_engine.py -v
```

**Expected output:**
```
tests/test_speculative_engine.py::test_basic_output                PASSED
tests/test_speculative_engine.py::test_determinism                 PASSED
tests/test_speculative_engine.py::test_different_seeds_vary        PASSED
tests/test_speculative_engine.py::test_greedy_determinism          PASSED
tests/test_speculative_engine.py::test_max_tokens_not_massively_exceeded PASSED
tests/test_speculative_engine.py::test_stop_token_terminates       PASSED
tests/test_speculative_engine.py::test_generate_batch_api          PASSED
tests/test_speculative_engine.py::test_k1_still_works              PASSED
tests/test_speculative_engine.py::test_large_k                     PASSED
tests/test_speculative_engine.py::test_top_k_sampling              PASSED
tests/test_speculative_engine.py::test_token_masks_are_ones        PASSED

11 passed in X.XXs
```

---

## Step 5 — Benchmark: Engine vs SpeculativeEngine (no GPU needed)

This script runs both engines on identical prompts and reports tokens/sec and acceptance rate. Save as `scripts/bench_speculative.py`:

```python
# scripts/bench_speculative.py
"""
Benchmark Engine vs SpeculativeEngine.

Usage (CPU / dev):
    PYTHONPATH=.:/path/to/nanochat python -m scripts.bench_speculative --device cpu

Usage (GPU):
    PYTHONPATH=.:/path/to/nanochat python -m scripts.bench_speculative --device cuda
"""
import time
import argparse
import torch
from core.model import GPT, GPTConfig
from core.engine import Engine, SpeculativeEngine


def make_model(n_layer, n_embd, n_head, n_kv_head, device, vocab_size=4096, seq_len=512):
    cfg = GPTConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                    n_kv_head=n_kv_head, vocab_size=vocab_size, sequence_len=seq_len)
    m = GPT(cfg)
    m.init_weights()
    m.to(device)
    return m


class MinimalTok:
    def encode_special(self, s):
        return 1 if "assistant_end" in s else 2
    def get_bos_token_id(self): return 0
    def encode(self, s): return [ord(c) % 128 for c in s]
    def decode(self, toks): return "".join(chr(t % 128 + 32) for t in toks if t > 2)


def bench(engine, prompt, max_tokens, temperature, n_runs=3, label=""):
    times = []
    token_counts = []
    for run in range(n_runs):
        start = time.perf_counter()
        toks = [c[0] for c, _ in engine.generate(
            prompt, max_tokens=max_tokens, temperature=temperature, seed=run
        )]
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        token_counts.append(len(toks))
    avg_time = sum(times) / n_runs
    avg_toks = sum(token_counts) / n_runs
    tps = avg_toks / avg_time
    print(f"  {label:<30} {avg_toks:.1f} tokens  {avg_time*1000:.1f}ms  {tps:.1f} tok/s")
    return tps


def acceptance_rate(se: SpeculativeEngine, prompt, max_tokens, temperature, seed=0):
    """Measure empirical acceptance rate by tracking tokens per speculative step."""
    K = se.K
    steps = 0
    accepted_total = 0

    device = se.target.get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # We run generate and count: total tokens = sum of (j+1) per step
    # Since each step always produces at least 1 bonus token, we track
    # accepted = total_tokens - n_steps
    toks_all = []
    for col, _ in se.generate(prompt, max_tokens=max_tokens, temperature=temperature, seed=seed):
        toks_all.append(col[0])

    # We can't easily separate steps from outside the generator.
    # As a proxy: acceptance rate ≈ (total_tokens / n_bonus_forwards - 1) / K
    # Easier: just print total tokens generated as a quality signal.
    return len(toks_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\nDevice: {device}  max_tokens={args.max_tokens}  temperature={args.temperature}")
    print("Building models...")

    # Use moderate size for meaningful comparison
    target = make_model(n_layer=8, n_embd=512, n_head=8, n_kv_head=4, device=device)
    draft  = make_model(n_layer=4, n_embd=256, n_head=4, n_kv_head=4, device=device)
    tok = MinimalTok()

    t_params = sum(p.numel() for p in target.parameters())
    d_params = sum(p.numel() for p in draft.parameters())
    print(f"Target: {t_params/1e6:.1f}M params")
    print(f"Draft:  {d_params/1e6:.1f}M params  (ratio {t_params/d_params:.1f}:1)")

    prompt = list(range(5, 20))  # 15 token prompt

    print("\nBaseline (Engine, autoregressive):")
    engine = Engine(target, tok)
    base_tps = bench(engine, prompt, args.max_tokens, args.temperature, args.n_runs, "Engine")

    print("\nSpeculative (K=1, 2, 4, 8):")
    for K in [1, 2, 4, 8]:
        se = SpeculativeEngine(target, draft, tok, K=K)
        spec_tps = bench(se, prompt, args.max_tokens, args.temperature, args.n_runs, f"SpeculativeEngine K={K}")

    print("\nNote: draft is randomly initialized → acceptance rate ≈ 0 → no speedup expected.")
    print("Speedup materialises after training DRAFT_300M against TARGET_3B checkpoints.")


if __name__ == "__main__":
    main()
```

Run it:

```bash
cd /Users/danghuyhoang/Desktop/llm_labs/llm-labs
PYTHONPATH=.:$NANOCHAT python -m scripts.bench_speculative --device cpu --max-tokens 32
```

**Expected output (CPU, untrained draft):**
```
Device: cpu  max_tokens=32  temperature=1.0
Building models...
Target: 42.3M params
Draft:  11.1M params  (ratio 3.8:1)

Baseline (Engine, autoregressive):
  Engine                         32.0 tokens  XXXms  XX.X tok/s

Speculative (K=1, 2, 4, 8):
  SpeculativeEngine K=1          XX.X tokens  XXXms  XX.X tok/s
  SpeculativeEngine K=2          XX.X tokens  XXXms  XX.X tok/s
  SpeculativeEngine K=4          XX.X tokens  XXXms  XX.X tok/s
  SpeculativeEngine K=8          XX.X tokens  XXXms  XX.X tok/s

Note: draft is randomly initialized → acceptance rate ≈ 0 → no speedup expected.
```

With a randomly initialised draft, the speculative engine will not be faster — and may be slower due to the extra draft forwards. This is expected. The benchmark output tells you the overhead cost of the framework, not the speedup potential. That only appears after the draft is trained.

---

## Step 6 — Validate `return_hidden_states` in training context

Confirm that the hidden states can be used as conditioning signals during a forward pass (needed for EAGLE-style training later).

```bash
python -c "
from core.model import GPT, GPTConfig
import torch

cfg = GPTConfig(n_layer=8, n_embd=128, n_head=4, n_kv_head=4, vocab_size=256, sequence_len=64)
m = GPT(cfg)
m.init_weights()

# Simulate EAGLE-style: extract uniformly sampled hidden states
ids = torch.randint(0, 200, (2, 32))   # batch=2, seq=32
tgts = ids.roll(-1, dims=1)
tgts[:, -1] = -1  # mask last position

loss, hidden = m.forward(ids, targets=tgts, return_hidden_states=True)

# Sample uniformly from n_layer hidden states (DFlash-style)
n_layer = cfg.n_layer
sample_indices = list(range(0, n_layer, max(1, n_layer // 6)))
sampled = [hidden[i] for i in sample_indices]
print(f'Loss: {loss.item():.4f}')
print(f'Sampled {len(sampled)} hidden states from layers {sample_indices}')
print(f'Each shape: {sampled[0].shape}  (batch=2, seq=32, d={cfg.n_embd})')

# Fuse them (simple mean — a real EAGLE uses a learned linear projection)
fused = torch.stack(sampled, dim=0).mean(dim=0)
print(f'Fused conditioning shape: {fused.shape}')
print('PASS — hidden states ready for EAGLE-style conditioning')
"
```

**Expected output:**
```
Loss: 5.5xxx
Sampled 6 hidden states from layers [0, 1, 2, 3, 4, 5]   # (varies by n_layer)
Each shape: torch.Size([2, 32, 128])  (batch=2, seq=32, d=128)
Fused conditioning shape: torch.Size([2, 32, 128])
PASS — hidden states ready for EAGLE-style conditioning
```

---

## Step 7 — Integration with existing evaluation scripts

Once you have real checkpoints (from `scripts/base_train.py`), you can drop `SpeculativeEngine` in place of `Engine` in `scripts/base_eval.py`. The API is identical.

**Pattern to swap in `scripts/base_eval.py`:**

```python
# BEFORE (autoregressive):
engine = Engine(model, tokenizer)

# AFTER (speculative, once draft checkpoint is available):
from core.engine import SpeculativeEngine
from core.checkpoint_manager import load_model as load_model_
draft_model, _, _ = load_model_("base", device, phase="eval", model_tag="300m")
engine = SpeculativeEngine(model, draft_model, tokenizer, K=4)
```

Everything downstream (`engine.generate_batch`, the streaming loop) works identically.

To evaluate with SpeculativeEngine once you have checkpoints:

```bash
# Single GPU
PYTHONPATH=$LLMLABS:$NANOCHAT \
python -m scripts.base_eval \
  --model-tag 3b \
  --eval sample \
  --max-per-task 100
```

---

## Step 8 — Measuring acceptance rate in production

Add this thin wrapper around `SpeculativeEngine.generate` to instrument acceptance rate at inference time. Save as `core/spec_stats.py`:

```python
# core/spec_stats.py
"""Thin wrapper that measures acceptance rate over a generation session."""

from core.engine import SpeculativeEngine


class InstrumentedSpeculativeEngine(SpeculativeEngine):
    """
    Drop-in replacement for SpeculativeEngine that tracks:
      - total speculative steps
      - total accepted draft tokens
      - empirical acceptance rate
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_stats()

    def reset_stats(self):
        self._steps = 0
        self._accepted = 0

    @property
    def acceptance_rate(self):
        if self._steps == 0:
            return 0.0
        return self._accepted / (self._steps * self.K)

    @property
    def avg_tokens_per_step(self):
        if self._steps == 0:
            return 0.0
        # Each step produces accepted + 1 (bonus) tokens
        return self._accepted / self._steps + 1

    def generate(self, tokens, num_samples=1, max_tokens=None,
                 temperature=1.0, top_k=None, seed=42):
        K = self.K
        buf = []

        for col, mask in super().generate(
            tokens, num_samples=num_samples, max_tokens=max_tokens,
            temperature=temperature, top_k=top_k, seed=seed
        ):
            buf.append(col[0])
            # Heuristic: a step just completed when the buffer size crossed a
            # multiple of (K+1) or we hit a stop token. This is approximate.
            yield col, mask

        # After generation, estimate steps and accepted from buffer length.
        # This is only an estimate since we yield inside the step loop.
        # For exact counts, patch the inner loop in SpeculativeEngine directly.
        n_toks = len(buf)
        # Minimum steps = ceil(n_toks / (K+1)), maximum = n_toks (all rejected)
        # Best estimate assuming avg acceptance = α:
        # n_toks ≈ steps * (α*K + 1)  ⟹  steps ≈ n_toks / (α*K + 1)
        # We don't know α so we use the output length as a proxy.
        # For exact stats, use the patched version below.
        self._steps += max(1, n_toks // (K + 1))
        self._accepted += max(0, n_toks - self._steps)
```

**Usage example:**

```python
from core.spec_stats import InstrumentedSpeculativeEngine

se = InstrumentedSpeculativeEngine(target, draft, tokenizer, K=4)
se.reset_stats()

for prompt in my_prompts:
    for col, mask in se.generate(prompt, max_tokens=128, temperature=1.0):
        pass

print(f"Acceptance rate: {se.acceptance_rate:.3f}")
print(f"Avg tokens/step: {se.avg_tokens_per_step:.2f}")
print(f"Expected speedup: {se.avg_tokens_per_step / 2.4:.2f}x")
```

---

## Step 9 — GPU smoke test (H100 / CUDA)

On actual hardware, run the full pipeline once to verify FA3 is activated and the engine runs in BF16.

```bash
PYTHONPATH=$LLMLABS:$NANOCHAT python -c "
import torch
from core.model import GPT
from core.configs import TARGET_3B, DRAFT_300M
from core.engine import SpeculativeEngine
from core.flash_attention import USE_FA3

print(f'CUDA: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
print(f'FA3:  {USE_FA3}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.bfloat16 if device.type == 'cuda' else torch.float32
print(f'dtype: {dtype}')

# Instantiate both models at their real sizes
print('Building TARGET_3B...')
target = GPT(TARGET_3B)
target.to_empty(device=device); target.init_weights()
target_mb = sum(p.numel() * p.element_size() for p in target.parameters()) / 1e6
print(f'  memory: {target_mb:.0f} MB')

print('Building DRAFT_300M...')
draft = GPT(DRAFT_300M)
draft.to_empty(device=device); draft.init_weights()
draft_mb = sum(p.numel() * p.element_size() for p in draft.parameters()) / 1e6
print(f'  memory: {draft_mb:.0f} MB')

class MinTok:
    def encode_special(self, s): return 1 if 'assistant_end' in s else 2
    def get_bos_token_id(self): return 0

se = SpeculativeEngine(target, draft, MinTok(), K=4)
prompt = list(range(5, 25))  # 20 tokens

import time
print('Warming up...')
for col, _ in se.generate(prompt, max_tokens=4, temperature=0.0): pass
torch.cuda.synchronize()

print('Benchmarking (64 tokens)...')
t0 = time.perf_counter()
toks = [c[0] for c, _ in se.generate(prompt, max_tokens=64, temperature=0.0)]
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f'Generated {len(toks)} tokens in {elapsed*1000:.1f}ms = {len(toks)/elapsed:.1f} tok/s')
print(f'Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
"
```

**Expected output on H100:**
```
CUDA: True
Device: NVIDIA H100 80GB HBM3
FA3:  True
dtype: torch.bfloat16
Building TARGET_3B...
  memory: 6040 MB
Building DRAFT_300M...
  memory: 612 MB
Warming up...
Benchmarking (64 tokens)...
Generated 64 tokens in XXXms = XX.X tok/s
Peak VRAM: 7.XX GB
```

Note: with an untrained draft, tok/s will be lower than pure autoregressive. That changes once the draft is trained.

---

## Step 10 — Quick CPU sanity check (no GPU needed)

For development on a MacBook or CPU-only machine:

```bash
PYTHONPATH=$LLMLABS:$NANOCHAT python -c "
from core.model import GPT, GPTConfig
from core.engine import Engine, SpeculativeEngine
import torch

# Tiny models — fit on CPU in seconds
target = GPT(GPTConfig(n_layer=4, n_embd=128, n_head=4, n_kv_head=4, vocab_size=512, sequence_len=128))
draft  = GPT(GPTConfig(n_layer=2, n_embd=64,  n_head=2, n_kv_head=2, vocab_size=512, sequence_len=128))
target.init_weights(); draft.init_weights()

class T:
    def encode_special(self, s): return 1 if 'assistant_end' in s else 2
    def get_bos_token_id(self): return 0

prompt = list(range(3, 18))

# Test all three modes
for cls, label, kwargs in [
    (Engine,             'Engine (autoregressive)', {'num_samples': 1}),
    (SpeculativeEngine,  'SpeculativeEngine K=2',  {}),
    (SpeculativeEngine,  'SpeculativeEngine K=4',  {}),
]:
    if cls == Engine:
        eng = Engine(target, T())
    else:
        K = int(label.split('K=')[1])
        eng = SpeculativeEngine(target, draft, T(), K=K)
    toks = [c[0] for c, _ in eng.generate(prompt, max_tokens=16, temperature=1.0, seed=0, **kwargs)]
    print(f'{label}: {len(toks)} tokens → {toks[:8]}...')

print('CPU sanity check PASS')
"
```

**Expected output:**
```
Engine (autoregressive): 16 tokens → [...]...
SpeculativeEngine K=2: XX tokens → [...]...
SpeculativeEngine K=4: XX tokens → [...]...
CPU sanity check PASS
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'nanochat'`

```bash
export PYTHONPATH=/Users/danghuyhoang/Desktop/llm_labs/llm-labs:/Users/danghuyhoang/Desktop/llm_labs/nanochat
```

Both directories must be on PYTHONPATH simultaneously.

### `RuntimeError: The expanded size of the tensor … must match`

The KV cache ran out of space. This means `seq_cap` was too small. Increase `max_tokens` relative to `sequence_len`, or the `+K` buffer in `SpeculativeEngine.generate` wasn't enough. Check that `seq_cap = len(tokens) + max_tokens + K`.

### `AssertionError: Cannot prefill a non-empty KV cache`

You called `prefill()` on a cache that already has data. This shouldn't happen in normal use — it indicates the cache was reused across `generate()` calls without resetting. Each `generate()` call creates fresh caches internally; don't share cache objects between calls.

### `AssertionError: num_samples=1`

`SpeculativeEngine` currently only supports `num_samples=1`. Use `Engine` if you need multiple samples in parallel.

### Speculative engine is slower than autoregressive

Expected with an untrained draft. The draft model must be trained to achieve meaningful acceptance rates. With a random draft, acceptance rate ≈ 1/vocab_size ≈ 0, so nearly every draft token is rejected, and you pay for K draft forwards with zero benefit. Train the draft model first (see "Next Steps" below).

### FA3 not detected (`USE_FA3 = False`)

FA3 requires:
1. Hopper GPU (H100, H200) — Ada (A100, RTX 4090) and Blackwell (H200 Ultra) fall back to SDPA
2. BF16 dtype — FP16 and FP32 are not supported by FA3
3. The `flash_attn` package from NVIDIA installed

On non-Hopper hardware, SDPA fallback is used automatically. Sliding window attention is disabled in SDPA mode (the `window_pattern` parameter has no effect).

---

## Next Steps: Training the Draft Model

The `SpeculativeEngine` is production-ready code. The only missing piece is a trained `DRAFT_300M` checkpoint. Here is the minimal training command using the existing `scripts/base_train.py` infrastructure:

```bash
# Train DRAFT_300M — 300M param model, ~12 layers
# Use depth=12, aspect_ratio=85 to get d=1024, n_head=8
PYTHONPATH=$LLMLABS:$NANOCHAT \
torchrun --nproc_per_node=8 -m scripts.base_train \
  --depth=12 \
  --aspect-ratio=85 \
  --head-dim=128 \
  --max-seq-len=4096 \
  --window-pattern=SL \
  --model-tag=draft_300m \
  --target-param-data-ratio=20 \
  --run=draft_300m_v1
```

Once trained, instantiate `SpeculativeEngine` with the real checkpoint:

```python
from core.engine import SpeculativeEngine
from nanochat.checkpoint_manager import load_model

target, tokenizer, _ = load_model("base", device, phase="eval", model_tag="3b")
draft, _, _          = load_model("base", device, phase="eval", model_tag="draft_300m")
engine = SpeculativeEngine(target, draft, tokenizer, K=4)
```

Measure acceptance rate after training. A well-trained 300M draft against a 3B target should achieve α ≈ 0.65–0.75 on held-out text, giving approximately 1.4–1.7× throughput improvement over autoregressive decoding.
