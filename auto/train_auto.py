"""
train_auto.py — autoresearch Tier 1 training driver.

Thin wrapper that imports `core.model.GPT` and runs a single 5-minute training
experiment on 1×H100, then prints val_bpb in Karpathy's autoresearch format.

This file (plus core/) is the agent's edit surface. Crucially, `auto/` is NOT a
fork of `core/` — `core/` IS the model. The agent edits `core/` directly to
test architectural hypotheses; this script just owns the autoresearch-loop
concerns (wall-clock budget, val_bpb metric, output format).

Edit scope (the agent CAN modify):
  - This file's AGENT-EDITABLE HYPERPARAMETERS block (the common case)
  - core/model.py            (GPT, attention, blocks, RoPE, value embeddings)
  - core/moe.py              (TopKRouter, ExpertGroup, SharedExpert, MoE)
  - core/optim.py            (MuonAdamW, fused step kernels — careful)
  - core/_layers.py          (auto-cast Linear)
  - core/configs.py          (sized config presets — optional)
  - core/dataloader.py       (training-data path — careful, has DDP semantics)
  - core/flash_attention.py  (rarely; impl-detail plumbing)

Frozen (DO NOT MODIFY):
  - auto/prepare_auto.py     (vocab=8192, seq_len=2048, time_budget=300, evaluate_bpb)
  - auto/program.md
  - Anything outside core/ and auto/ (scripts/, dev/, bench/, tests/)
  - core/multimodal*, core/engine.py, core/checkpoint_manager.py,
    core/fp8.py, core/core_eval.py, core/report.py, core/execution.py
    (out of scope for Tier 1; live on main for Tier 2)

Safety mechanism is BRANCH isolation, not file isolation: this loop runs on
branch `auto/<tag>`; Tier 2's preregistered sweep (`dev/sweep_design.md`) runs
on its own pinned SHA on main. Agent edits to core/ on the auto branch don't
affect Tier 2.
"""
import os
import sys
import math
import time

# Ensure both core/ (sibling of auto/) and prepare_auto (in auto/) are importable
# regardless of CWD or how this script is invoked.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for p in (_REPO_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

# core/ is the source of truth. Importing from it (rather than duplicating it)
# means agent findings flow back into the production codebase naturally.
from core.model import GPT, GPTConfig  # noqa: E402
from core.common import COMPUTE_DTYPE  # noqa: E402

# auto/prepare_auto.py is the FROZEN scaffold (vocab, eval, dataloader).
from prepare_auto import (  # noqa: E402
    load_tokenizer,
    make_dataloader,
    evaluate_bpb as _evaluate_bpb,
    VOCAB_SIZE, MAX_SEQ_LEN, TIME_BUDGET, EVAL_TOKENS,
)


# =============================================================================
# AGENT-EDITABLE HYPERPARAMETERS
# The cheapest knob to turn. For deeper changes, edit core/.
# =============================================================================

# --- Architecture (consumed below to build GPTConfig) ---
DEPTH = 8
ASPECT_RATIO = 64                   # n_embd = DEPTH * ASPECT_RATIO
HEAD_DIM = 128
NUM_EXPERTS = 1                     # MoE: routed experts
TOP_K = 1                           # MoE: experts active per token (routed)
NUM_SHARED_EXPERTS = 0              # MoE: always-active expert
WINDOW_PATTERN = "L"                # "L" = full causal every layer

# --- Optimizer LRs (passed to model.setup_optimizer) ---
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.008
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.95)

# --- LR schedule shape (used by get_lr_multiplier below) ---
WARMUP_RATIO = 0.05                 # fraction of TIME_BUDGET spent warming up
WARMDOWN_RATIO = 0.3                # fraction of TIME_BUDGET spent cooling down to 0

# --- Training shape ---
DEVICE_BATCH_SIZE = 32
TOTAL_BATCH_SIZE = 2 ** 18          # tokens per optimizer step
SEED = 1337
WARMUP_STEPS_TO_IGNORE = 10         # exclude compile + first iters from wall-clock budget

# =============================================================================
# END AGENT-EDITABLE HYPERPARAMETERS
# =============================================================================


def get_lr_multiplier(progress: float) -> float:
    """Trapezoidal schedule. progress in [0, 1] over the wall-clock budget."""
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO
    cooldown_start = 1.0 - WARMDOWN_RATIO
    if progress < cooldown_start:
        return 1.0
    return max(0.0, (1.0 - progress) / WARMDOWN_RATIO)


def peak_flops_for_device() -> float:
    """Return peak BF16 TFLOP/s for the current GPU, for MFU reporting."""
    if not torch.cuda.is_available():
        return 1e12
    cap = torch.cuda.get_device_capability()
    if cap >= (9, 0):
        return 989e12      # H100 BF16
    if cap >= (8, 0):
        return 312e12      # A100 BF16
    return 100e12          # pre-Ampere fallback


def unwrap(model):
    """Get the underlying nn.Module out of a torch.compile wrapper."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def main():
    # ---------- setup ----------
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('high')

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    assert vocab_size == VOCAB_SIZE, f"tokenizer vocab {vocab_size} != prepare_auto VOCAB_SIZE {VOCAB_SIZE}"

    # ---------- model ----------
    # Aspect-ratio-driven sizing: n_embd = DEPTH * ASPECT_RATIO. With Karpathy's
    # defaults (DEPTH=8, ASPECT_RATIO=64) this gives n_embd=512, ~50M total
    # params — matching Karpathy's published autoresearch d=8 baseline so
    # val_bpb numbers are directly comparable.
    n_embd = DEPTH * ASPECT_RATIO
    n_head = max(1, n_embd // HEAD_DIM)
    cfg = GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=DEPTH,
        n_head=n_head,
        n_kv_head=n_head,           # no GQA in the baseline
        n_embd=n_embd,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        num_shared_experts=NUM_SHARED_EXPERTS,
        window_pattern=WINDOW_PATTERN,
        multimodal=False,           # Tier 1 is text-only
    )
    with torch.device('meta'):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    model.train()

    nparams = unwrap(model).num_scaling_params()
    flops_per_tok = unwrap(model).estimate_flops()
    print(f"[train_auto] depth={DEPTH} n_embd={n_embd} n_head={n_head} "
          f"experts={NUM_EXPERTS} top_k={TOP_K} shared={NUM_SHARED_EXPERTS} "
          f"vocab={vocab_size} compute_dtype={COMPUTE_DTYPE}")
    print(f"[train_auto] active_params={nparams['active_trunk_total']/1e6:.1f}M "
          f"total_params={nparams['trunk_total']/1e6:.1f}M flops/tok={flops_per_tok/1e6:.1f}M")

    # torch.compile gives ~1.3–2× on H100. Drop to eager if compile is unhappy.
    try:
        model = torch.compile(model, dynamic=False, fullgraph=False)
        print("[train_auto] torch.compile enabled")
    except Exception as e:
        print(f"[train_auto] torch.compile failed: {e}; continuing eager")

    # ---------- optimizer ----------
    optimizer = unwrap(model).setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
        adam_betas=ADAM_BETAS,
        scalar_lr=SCALAR_LR,
    )
    print(f"[train_auto] optimizer groups: {len(optimizer.param_groups)}")

    # ---------- data ----------
    train_loader = make_dataloader(tokenizer, B=DEVICE_BATCH_SIZE, T=MAX_SEQ_LEN,
                                   split="train", device=device)
    tokens_per_step = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    grad_accum_steps = max(1, TOTAL_BATCH_SIZE // tokens_per_step)
    effective_batch_tokens = grad_accum_steps * tokens_per_step
    if effective_batch_tokens != TOTAL_BATCH_SIZE:
        print(f"[train_auto] WARNING: TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} not divisible by "
              f"DEVICE_BATCH_SIZE×MAX_SEQ_LEN={tokens_per_step}; effective={effective_batch_tokens}")
    print(f"[train_auto] grad_accum={grad_accum_steps} tokens/step={effective_batch_tokens}")

    # ---------- main loop (wall-clock terminated, NOT step terminated) ----------
    peak_mem_mb = 0.0
    total_train_time = 0.0
    smooth_loss = float('nan')
    smooth_alpha = 0.9
    step = 0
    t0_total = time.time()
    t_train_start = None              # set after WARMUP_STEPS_TO_IGNORE

    while True:
        # Termination: wall-clock budget (excluding warmup-step bookkeeping)
        if t_train_start is not None and (time.time() - t_train_start) >= TIME_BUDGET:
            break

        # LR schedule based on elapsed training fraction
        progress = 0.0 if t_train_start is None else min(1.0, (time.time() - t_train_start) / TIME_BUDGET)
        lr_mult = get_lr_multiplier(progress)
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * lr_mult

        # Gradient accumulation step
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        t_step_start = time.time()
        for _ in range(grad_accum_steps):
            x, y = next(train_loader)
            loss = model(x, y) / grad_accum_steps
            loss.backward()
            step_loss += loss.item()

        # MoE auxiliary-loss-free load balancing (DeepSeekV3-style bias nudge)
        unwrap(model).update_moe_balancing(coeff=1e-3)

        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t_step_start

        # NaN/inf guard — autoresearch will record this as a crash.
        if not math.isfinite(step_loss) or step_loss > 100:
            print(f"FATAL: loss diverged at step {step}: loss={step_loss}")
            sys.exit(1)

        smooth_loss = step_loss if math.isnan(smooth_loss) else smooth_alpha * smooth_loss + (1 - smooth_alpha) * step_loss

        # Start counting the wall-clock budget AFTER warmup steps (compile + first iters)
        if step == WARMUP_STEPS_TO_IGNORE:
            t_train_start = time.time()
        if t_train_start is not None:
            total_train_time = time.time() - t_train_start

        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            peak_mem_mb = max(peak_mem_mb, mem_mb)

        # Brief per-step trace (one line per 10 steps + first few)
        if step % 10 == 0 or step < 5:
            tok_per_sec = effective_batch_tokens / dt if dt > 0 else 0.0
            print(f"step {step:5d} | loss {step_loss:.4f} | dt {dt*1000:.0f}ms | "
                  f"tok/s {tok_per_sec:,.0f} | lr_mult {lr_mult:.3f} | elapsed {total_train_time:.1f}s")

        step += 1

    num_steps = step
    total_seconds = time.time() - t0_total

    # ---------- final val_bpb evaluation ----------
    print(f"[train_auto] training done. running val_bpb eval on {EVAL_TOKENS:,} tokens...")
    model.eval()
    val_bpb = _evaluate_bpb(unwrap(model), tokenizer, B=DEVICE_BATCH_SIZE,
                            T=MAX_SEQ_LEN, device=device)

    # ---------- MFU ----------
    total_tokens = num_steps * effective_batch_tokens
    peak_flops = peak_flops_for_device()
    achieved_flops = total_tokens * flops_per_tok / max(total_train_time, 1e-6)
    mfu_pct = 100.0 * achieved_flops / peak_flops

    # ---------- summary (Karpathy format: each key on its own line so grep "^key:" works) ----------
    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_train_time:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_mem_mb:.1f}")
    print(f"mfu_percent:      {mfu_pct:.2f}")
    print(f"total_tokens_M:   {total_tokens/1e6:.1f}")
    print(f"num_steps:        {num_steps}")
    print(f"num_params_M:     {nparams['active_trunk_total']/1e6:.1f}")
    print(f"depth:            {DEPTH}")


if __name__ == "__main__":
    main()
