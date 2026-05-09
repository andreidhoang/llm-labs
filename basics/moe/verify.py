"""Verifier for the MoE layer + LM (basics/moe).

Run from llm/ directory:
    python -m basics.moe.verify

Spec: basics/moe/spec.md
Produces basics/moe/run.csv with the healthy training run log.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from basics.model import SwiGLU
from basics.moe import (
    BasicsMoETransformerLM,
    MoELayer,
    routing_entropy,
)


VOCAB = 32  # input alphabet AND output classes — true LM, no classification head needed
N_LAYERS = 2
D_MODEL = 64
D_FF = 256
N_HEADS = 4
CONTEXT = 64
DEFAULT_MOE_KWARGS = dict(
    granularity=2, base_experts=4, top_k=2, n_shared=1, bias_update_lr=1e-3,
)


def make_model(moe_kwargs: dict | None = None) -> BasicsMoETransformerLM:
    return BasicsMoETransformerLM(
        vocab_size=VOCAB,
        context_length=CONTEXT,
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_heads=N_HEADS,
        d_ff=D_FF,
        moe_layer_indices=(1,),  # layer 0 dense, layer 1 MoE
        moe_kwargs=moe_kwargs or dict(DEFAULT_MOE_KWARGS),
    )


# ---------- toy task: two modalities, two functions ----------

def gen_batch(B=32, T=16, modality="mixed", device="cpu", generator=None):
    """Modality A: even integers, target = (x_t + x_{t-1}) mod VOCAB  (addition).
    Modality B: odd  integers, target = (x_t * x_{t-1}) mod VOCAB  (multiplication).
    Different *function* per modality is what gives experts a real reason to
    specialize. Targets share the input alphabet so the LM `lm_head` is reused
    directly with no special classification head.
    """
    if modality == "A":
        idx = torch.randint(0, VOCAB // 2, (B, T), generator=generator, device=device) * 2
        tag = torch.zeros(B, dtype=torch.long, device=device)
    elif modality == "B":
        idx = torch.randint(0, VOCAB // 2, (B, T), generator=generator, device=device) * 2 + 1
        tag = torch.ones(B, dtype=torch.long, device=device)
    else:
        half = B // 2
        even = torch.randint(0, VOCAB // 2, (half, T), generator=generator, device=device) * 2
        odd = torch.randint(0, VOCAB // 2, (B - half, T), generator=generator, device=device) * 2 + 1
        idx = torch.cat([even, odd], dim=0)
        tag = torch.cat([
            torch.zeros(half, dtype=torch.long, device=device),
            torch.ones(B - half, dtype=torch.long, device=device),
        ])
        perm = torch.randperm(B, generator=generator, device=device)
        idx, tag = idx[perm], tag[perm]

    x_prev = torch.roll(idx, shifts=1, dims=1)
    sum_target = (idx + x_prev) % VOCAB
    mul_target = (idx * x_prev) % VOCAB
    is_A = (tag == 0).unsqueeze(1)
    target = torch.where(is_A, sum_target, mul_target)
    return idx, target, tag


# ---------- training loop ----------

def train(
    model: BasicsMoETransformerLM,
    steps: int = 500,
    batch_size: int = 32,
    T: int = 16,
    lr: float = 3e-4,
    bias_update_lr: float | None = None,
    log_path: Path | None = None,
    modality: str = "mixed",
    device: str = "cpu",
    seed: int = 0,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if bias_update_lr is not None:
        for moe in model.iter_moe_layers():
            moe.bias_update_lr = bias_update_lr

    g = torch.Generator(device=device).manual_seed(seed)
    history: list[dict] = []
    moe = next(model.iter_moe_layers())
    rolling = torch.zeros(moe.n_routed, dtype=torch.long, device=device)

    for step in range(steps):
        idx, target, _ = gen_batch(batch_size, T, modality=modality, device=device, generator=g)
        logits, moe_aux = model(idx)
        loss = F.cross_entropy(
            logits[:, 1:].reshape(-1, VOCAB),
            target[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.step_router_biases(moe_aux)

        rolling += moe_aux[0]["expert_token_counts"]
        if (step + 1) % 50 == 0:
            total = max(1, int(rolling.sum().item()))
            history.append({
                "step": step + 1,
                "loss": float(loss.item()),
                "routing_entropy": routing_entropy(rolling),
                "n_dead": int((rolling == 0).sum().item()),
                "max_share": float(rolling.max().item()) / total,
                "grad_norm": float(grad_norm.item()),
            })
            rolling.zero_()

    if log_path is not None and history:
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

    return history


# ---------- the 5 verifier checks ----------

def check_shape():
    moe = MoELayer(d_model=64, d_ff=256, granularity=2, base_experts=4, top_k=2, n_shared=1)
    x = torch.randn(2, 16, 64)
    y, aux = moe(x)
    assert y.shape == (2, 16, 64), f"output shape {y.shape}"
    expected = {"expert_token_counts", "router_logits", "selected_experts", "routing_weights"}
    assert set(aux.keys()) == expected, f"aux keys {set(aux.keys())}"
    assert aux["expert_token_counts"].shape == (8,)
    assert aux["router_logits"].shape == (32, 8)
    assert aux["selected_experts"].shape == (32, 2)
    assert aux["routing_weights"].shape == (32, 2)
    weights_sum = aux["routing_weights"].sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), \
        f"weights must sum to 1 per token; got [{weights_sum.min():.4f}, {weights_sum.max():.4f}]"


def check_dense_equivalence():
    """G=1, base=1, top_k=1, n_shared=0 → MoELayer ≡ a single SwiGLU FFN."""
    torch.manual_seed(0)
    d_model, d_ff = 64, 256
    moe = MoELayer(d_model, d_ff, granularity=1, base_experts=1, top_k=1, n_shared=0)
    dense = SwiGLU(d_model, d_ff)
    dense.w1.weight.data.copy_(moe.routed_experts[0].w1.weight)
    dense.w2.weight.data.copy_(moe.routed_experts[0].w2.weight)
    dense.w3.weight.data.copy_(moe.routed_experts[0].w3.weight)

    x = torch.randn(2, 16, d_model)
    y_moe, _ = moe(x)
    y_dense = dense(x)
    diff = (y_moe - y_dense).abs().max().item()
    assert diff < 1e-5, f"max diff {diff}"


def check_anti_collapse(history, n_routed=8):
    threshold = 0.7 * math.log(n_routed)
    last = history[-1]["routing_entropy"]
    assert last > threshold, \
        f"entropy {last:.3f} ≤ threshold {threshold:.3f}; routing collapsed"


def _measure_specialization(model: BasicsMoETransformerLM, n_batches: int = 50) -> float:
    moe = next(model.iter_moe_layers())
    counts_A = torch.zeros(moe.n_routed, dtype=torch.long)
    counts_B = torch.zeros(moe.n_routed, dtype=torch.long)
    g = torch.Generator(device="cpu").manual_seed(123)
    with torch.no_grad():
        for _ in range(n_batches):
            idx, _, _ = gen_batch(B=32, T=16, modality="A", generator=g)
            _, aux = model(idx)
            counts_A += aux[0]["expert_token_counts"]
            idx, _, _ = gen_batch(B=32, T=16, modality="B", generator=g)
            _, aux = model(idx)
            counts_B += aux[0]["expert_token_counts"]
    ratios = []
    for e in range(moe.n_routed):
        a, b = int(counts_A[e]), int(counts_B[e])
        ratios.append(max(a, b) / max(1, min(a, b)))
    return max(ratios)


def check_specialization(seeds=(1, 2, 3)):
    """Specialization is seed-sensitive at toy scale — task converges far enough
    that there's little gradient pressure for routed experts to specialize. Use
    median across multiple seeds rather than pinning to a single lucky/unlucky run.
    """
    ratios = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = make_model()
        train(model, steps=500, seed=seed)
        r = _measure_specialization(model)
        ratios.append(r)
        print(f"       seed={seed}  max ratio={r:.2f}")
    median = sorted(ratios)[len(ratios) // 2]
    print(f"       median across {len(seeds)} seeds: {median:.2f}")
    assert median > 2.0, \
        f"median max-ratio {median:.2f} ≤ 2.0 across seeds {seeds}; no robust specialization"


def _train_collapse_prone(bias_update_lr: float, seed: int = 42, steps: int = 500):
    """Single-modality + no shared expert = setup that *can* collapse,
    needed to test what the bias mechanism actually does.
    """
    torch.manual_seed(seed)
    model = make_model(moe_kwargs=dict(
        granularity=2, base_experts=4, top_k=2, n_shared=0,
        bias_update_lr=bias_update_lr,
    ))
    history = train(model, steps=steps, bias_update_lr=bias_update_lr,
                    modality="A", seed=seed)
    return history[-1]["max_share"], history[-1]["routing_entropy"]


def check_bias_update_has_effect(n_routed=8):
    """Deliberate-break: bias mechanism must reduce deviation-from-uniform vs no-bias.
    Metric is `max_share - 1/n_routed` — distance above the controller's setpoint
    (uniform). Raw max_share would smush the signal against the ~0.125 baseline.
    """
    uniform = 1.0 / n_routed
    with_share, with_H = _train_collapse_prone(bias_update_lr=1e-3)
    no_share, no_H = _train_collapse_prone(bias_update_lr=0.0)
    with_dev = max(0.0, with_share - uniform)
    no_dev = max(0.0, no_share - uniform)
    print(f"       max_share        with-bias={with_share:.3f}  no-bias={no_share:.3f}  uniform={uniform:.3f}")
    print(f"       dev-from-uniform with-bias={with_dev:.4f}  no-bias={no_dev:.4f}")
    print(f"       entropy          with-bias={with_H:.3f}  no-bias={no_H:.3f}")
    reduction = no_dev / max(with_dev, 1e-6)
    assert reduction >= 3.0, (
        f"bias mechanism does not reduce deviation enough: "
        f"with-bias dev={with_dev:.4f}, no-bias dev={no_dev:.4f} "
        f"(reduction {reduction:.1f}× < 3.0×)"
    )


def main():
    log_path = Path(__file__).parent / "run.csv"

    print("[1/5] check_shape ...")
    check_shape()
    print("       PASS")

    print("[2/5] check_dense_equivalence ...")
    check_dense_equivalence()
    print("       PASS")

    print("[3+4/5] training healthy run (500 steps) ...")
    torch.manual_seed(0)
    model = make_model()
    history = train(model, steps=500, log_path=log_path, seed=0)
    print(f"       final loss={history[-1]['loss']:.4f}  "
          f"entropy={history[-1]['routing_entropy']:.3f}  "
          f"n_dead={history[-1]['n_dead']}  "
          f"max_share={history[-1]['max_share']:.3f}")

    print("[3/5] check_anti_collapse ...")
    check_anti_collapse(history)
    print("       PASS")

    print("[4/5] check_specialization (3-seed median) ...")
    check_specialization()
    print("       PASS")

    print("[5/5] check_bias_update_has_effect (deliberate break) ...")
    check_bias_update_has_effect()
    print("       PASS")

    print(f"\nAll 5 checks passed. Run log: {log_path}")


if __name__ == "__main__":
    main()
