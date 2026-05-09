"""
Named model configurations matching nanochat's parametric sizing:
    model_dim = depth * aspect_ratio (aspect_ratio=64)
    n_head    = model_dim / head_dim  (head_dim=128)
    n_kv_head = n_head                (no GQA — matches nanochat default)
    sequence_len = 2048
    vocab_size   = 32768
    window_pattern = "SSSL"

Reference sizes (param counts verified against nanochat runs):
    D12  ~125M  — reference/anchor model; hyperparameters tuned here, then μP-transferred up
    D20  ~350M  — nanochat default depth
    D24  ~630M  — speedrun target ("beats GPT-2")
"""

from core.model import GPTConfig

def _make_config(depth, aspect_ratio=64, head_dim=128):
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim  # round up to head_dim multiple
    n_head = model_dim // head_dim
    return GPTConfig(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=depth,
        n_embd=model_dim,
        n_head=n_head,
        n_kv_head=n_head,
        window_pattern="SSSL",
    )

D12 = _make_config(12)   # dim=768,  heads=6  — reference model (~125M)
D14 = _make_config(14)   # dim=896,  heads=7
D16 = _make_config(16)   # dim=1024, heads=8
D18 = _make_config(18)   # dim=1152, heads=9
D20 = _make_config(20)   # dim=1280, heads=10 — default depth (~350M)
D22 = _make_config(22)   # dim=1408, heads=11
D24 = _make_config(24)   # dim=1536, heads=12 — speedrun target (~630M)
D26 = _make_config(26)   # dim=1664, heads=13

# Convenience map for checkpoint loading / CLI flags
NAMED_CONFIGS = {
    "d12": D12,
    "d14": D14,
    "d16": D16,
    "d18": D18,
    "d20": D20,
    "d22": D22,
    "d24": D24,
    "d26": D26,
}
