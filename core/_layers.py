"""Shared low-level layers used across model.py, moe.py, multimodal.py.

Defining `Linear` here (instead of in model.py or moe.py) keeps the import
graph acyclic: model.py imports from moe.py, so moe.py cannot import from
model.py. _layers.py has no internal dependencies, so it's safe to import
from anywhere.

Mirrors nanochat's pattern (`Linear` in nanochat/gpt.py is the same idea):
the model casts embeddings to BF16 on CUDA, but matrix params stay FP32.
Without an auto-cast Linear, the first projection sees BF16 input × FP32
weight and crashes. This subclass casts the weight (and bias if present) to
the input's dtype at forward time — same compute, no surprise dtypes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    """nn.Linear that casts weight (and bias) to the input's dtype before matmul.

    Drop-in replacement for `nn.Linear`. Used everywhere matrix params (FP32)
    might receive BF16 input from a cast embedding or autocast region.

    The cast is essentially free — it's a memory read + dtype conversion that
    fuses into the matmul kernel under torch.compile. In eager mode it costs a
    tiny temporary tensor (one per call) but the matmul cost dominates.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)
