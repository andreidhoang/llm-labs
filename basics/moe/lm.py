"""Mixed dense/MoE transformer language model.

Built on `basics.model` dense primitives and `basics.moe.layer` MoE primitives.
Import direction is one-way: basics.moe.lm → basics.moe.layer → basics.model.

Spec: basics/moe/spec.md
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from basics.model import (
    Embedding,
    Linear,
    RMSNorm,
    RotaryEmbedding,
    TransformerBlock,
)
from basics.moe.layer import (
    MoELayer,
    MoETransformerBlock,
    update_router_biases,
)

logger = logging.getLogger(__name__)


class BasicsMoETransformerLM(nn.Module):
    """Transformer LM with mixed dense/MoE blocks.

    `moe_layer_indices` selects which layers are `MoETransformerBlock`; the rest
    are dense `TransformerBlock`. Default keeps layer 0 and the last layer dense
    and makes the middle layers MoE — Plan §3 "MoE in later layers, dense
    early/late" convention. Empty for `num_layers <= 2`; opt-in explicitly there.

    Forward returns `(logits, moe_aux)` where `moe_aux` is a list of routing
    aux dicts, one per MoE layer in forward order. Call
    `step_router_biases(moe_aux)` after `optimizer.step()` to drive the
    aux-loss-free balancing controller.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None = 10_000.0,
        moe_layer_indices: tuple[int, ...] | None = None,
        moe_kwargs: dict | None = None,
    ):
        self.config = {
            k: v for k, v in locals().items()
            if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()

        if moe_layer_indices is None:
            moe_layer_indices = tuple(range(1, num_layers - 1))
        self.moe_layer_indices = set(moe_layer_indices)
        for i in self.moe_layer_indices:
            if not 0 <= i < num_layers:
                raise ValueError(
                    f"moe_layer_indices contains {i}, out of range [0, {num_layers})"
                )

        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_head = d_model // num_heads
        self.positional_encoder = (
            RotaryEmbedding(context_length, d_head, rope_theta)
            if rope_theta is not None else None
        )

        layers: list[nn.Module] = []
        for i in range(num_layers):
            if i in self.moe_layer_indices:
                layers.append(MoETransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder,
                    moe_kwargs=moe_kwargs,
                ))
            else:
                layers.append(TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder,
                ))
        self.layers = nn.ModuleList(layers)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        logger.info(
            f"BasicsMoETransformerLM: {num_layers} layers "
            f"({len(self.moe_layer_indices)} MoE at {sorted(self.moe_layer_indices)}); "
            f"{self.get_num_params() / 1e6:.2f}M total params"
        )

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, x: Int[Tensor, " batch seq"]
    ) -> tuple[Float[Tensor, " batch seq vocab"], list[dict]]:
        h = self.token_embeddings(x)
        moe_aux: list[dict] = []
        for layer in self.layers:
            if isinstance(layer, MoETransformerBlock):
                h, aux = layer(h)
                moe_aux.append(aux)
            else:
                h = layer(h)
        h = self.ln_final(h)
        return self.lm_head(h), moe_aux

    def iter_moe_layers(self) -> Iterator[MoELayer]:
        """Iterator over the MoELayer modules in forward order."""
        for layer in self.layers:
            if isinstance(layer, MoETransformerBlock):
                yield layer.ffn

    def step_router_biases(self, moe_aux: list[dict]) -> None:
        """Sign-based bias update across every MoE layer.
        Call once per step, after `optimizer.step()`.
        """
        for moe_layer, aux in zip(self.iter_moe_layers(), moe_aux, strict=True):
            update_router_biases(moe_layer, aux["expert_token_counts"])
