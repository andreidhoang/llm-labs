"""Mixture-of-Experts layer for the basics transformer.

Built on top of dense primitives from `basics.model` (Linear, RMSNorm, SwiGLU,
RotaryEmbedding, CausalMultiHeadSelfAttention).

Convention: DeepSeek-V3 §2.1 — sigmoid top-k routing, raw scores renormalized
to sum=1 across selected experts, aux-loss-free balancing via a non-parameter
bias buffer updated as a control loop after each optimizer step.

Spec: basics/moe/spec.md
"""
from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from basics.model import (
    CausalMultiHeadSelfAttention,
    Linear,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
)


class Router(nn.Module):
    """Sigmoid top-k router with aux-loss-free bias balancing.

    Forward computes:
        scores  = sigmoid(W·x)                              # raw affinities
        select  = topk(scores + expert_bias, k=top_k)       # bias for SELECTION ONLY
        weights = scores[select] / sum(scores[select])      # renorm → sum=1 per token

    `expert_bias` is a non-parameter buffer; it is the control signal used by
    `update_router_biases`, not a learned parameter (no autograd path).
    """

    def __init__(self, d_model: int, n_routed: int):
        super().__init__()
        self.n_routed = n_routed
        self.weight = Linear(d_model, n_routed)
        self.register_buffer("expert_bias", torch.zeros(n_routed), persistent=True)

    def forward(self, x_flat: Float[Tensor, " n d_model"], top_k: int):
        logits = self.weight(x_flat)
        scores = torch.sigmoid(logits)
        biased = scores + self.expert_bias
        _, topk_idx = biased.topk(top_k, dim=-1)
        topk_raw = scores.gather(-1, topk_idx)
        weights = topk_raw / (topk_raw.sum(dim=-1, keepdim=True) + 1e-9)
        return logits, topk_idx, weights

    def extra_repr(self):
        return f"n_routed={self.n_routed}"


class MoELayer(nn.Module):
    """Fine-grained MoE layer with shared expert (DeepSeek-MoE / V3 style).

    Granularity G splits the baseline FFN into G smaller experts per "base":
        n_routed_experts = base_experts * G
        d_expert         = d_ff / G
    Active params per token = (top_k + n_shared) * d_expert.

    Returns (y, aux) where aux contains routing diagnostics for logging
    and for the post-step bias update.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        granularity: int = 2,
        base_experts: int = 4,
        top_k: int = 2,
        n_shared: int = 1,
        bias_update_lr: float = 1e-3,
    ):
        super().__init__()
        if d_ff % granularity != 0:
            raise ValueError(f"d_ff ({d_ff}) must be divisible by granularity ({granularity})")
        self.d_model = d_model
        self.d_ff = d_ff
        self.granularity = granularity
        self.base_experts = base_experts
        self.n_routed = base_experts * granularity
        self.top_k = top_k
        self.n_shared = n_shared
        self.d_expert = d_ff // granularity
        self.bias_update_lr = bias_update_lr

        self.router = Router(d_model, self.n_routed)
        self.routed_experts = nn.ModuleList(
            [SwiGLU(d_model, self.d_expert) for _ in range(self.n_routed)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLU(d_model, self.d_expert) for _ in range(n_shared)]
        )

    def forward(self, x: Float[Tensor, " batch seq d_model"]):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        logits, topk_idx, weights = self.router(x_flat, self.top_k)

        y_flat = torch.zeros_like(x_flat)
        token_counts = torch.zeros(self.n_routed, dtype=torch.long, device=x.device)

        for e in range(self.n_routed):
            mask = (topk_idx == e).any(dim=-1)
            if not mask.any():
                continue
            sel = mask.nonzero(as_tuple=True)[0]
            tokens_for_e = x_flat.index_select(0, sel)
            out_e = self.routed_experts[e](tokens_for_e)
            pos = (topk_idx.index_select(0, sel) == e).to(weights.dtype)
            w_e = (weights.index_select(0, sel) * pos).sum(dim=-1, keepdim=True)
            y_flat.index_add_(0, sel, out_e * w_e)
            token_counts[e] = sel.numel()

        for shared in self.shared_experts:
            y_flat = y_flat + shared(x_flat)

        y = y_flat.reshape(B, T, D)
        aux = {
            "expert_token_counts": token_counts,
            "router_logits": logits,
            "selected_experts": topk_idx,
            "routing_weights": weights,
        }
        return y, aux

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, d_ff={self.d_ff}, G={self.granularity}, "
            f"n_routed={self.n_routed}, top_k={self.top_k}, n_shared={self.n_shared}"
        )


@torch.no_grad()
def update_router_biases(moe: MoELayer, token_counts: Tensor) -> None:
    """Sign-based bias update (DeepSeek-V3 §2.1.2). Call after each optimizer step."""
    if moe.bias_update_lr == 0.0:
        return
    target = token_counts.float().mean()
    error = target - token_counts.float()
    moe.router.expert_bias.add_(moe.bias_update_lr * torch.sign(error))


def routing_entropy(token_counts: Tensor) -> float:
    """Shannon entropy of empirical expert usage distribution, in nats."""
    total = int(token_counts.sum().item())
    if total == 0:
        return 0.0
    p = token_counts.float() / total
    p = p[p > 0]
    return float(-(p * p.log()).sum().item())


class MoETransformerBlock(nn.Module):
    """Pre-norm transformer block whose FFN sublayer is a `MoELayer`.

    Forward returns (output, moe_aux). The caller threads aux up through the
    model so it can be logged and consumed by `update_router_biases`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryEmbedding | None,
        moe_kwargs: dict | None = None,
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, positional_encoder=positional_encoder
        )
        self.ffn = MoELayer(d_model=d_model, d_ff=d_ff, **(moe_kwargs or {}))
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        ffn_out, aux = self.ffn(self.ln2(x))
        x = x + ffn_out
        return x, aux
