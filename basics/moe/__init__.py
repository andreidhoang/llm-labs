"""Mixture-of-Experts subpackage. See spec.md."""
from basics.moe.layer import (
    MoELayer,
    MoETransformerBlock,
    Router,
    routing_entropy,
    update_router_biases,
)
from basics.moe.lm import BasicsMoETransformerLM

__all__ = [
    "BasicsMoETransformerLM",
    "MoELayer",
    "MoETransformerBlock",
    "Router",
    "routing_entropy",
    "update_router_biases",
]
