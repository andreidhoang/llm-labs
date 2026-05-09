"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.common import get_dist_info, print0
from core.optim import MuonAdamW, DistMuonAdamW
from core._layers import Linear  # auto-cast Linear: weight cast to input dtype

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from core.flash_attention import flash_attn
from core.moe import MoE

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    num_experts: int = 8  # MoE: number of routed expert MLPs
    top_k: int = 2  # MoE: number of active routed experts per token
    num_shared_experts: int = 1  # MoE: number of shared (always-active) experts
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Multimodal extension (default off → all existing text-only paths unchanged).
    # When enabled, GPT.forward additionally accepts pixel_values / grid_thw /
    # image_pad_mask / modality_mask kwargs and routes through VisionTower +
    # scatter_vision_features. See dev/multimodal_spec.md §2.5 for design rationale.
    # 3D Interleaved-MRoPE per multimodal_spec.md decision #7 is wired via
    # core.multimodal.build_3d_mrope_for_4d_apply when image_grids_merged is
    # passed to GPT.forward (see forward() below). Falls back to 1D RoPE when
    # multimodal=False or image_grids_merged is None (e.g. during text-only
    # inference on a multimodal-enabled model).
    multimodal: bool = False
    vision_embed_dim: int = 1152          # SigLIP2-SO400M default
    vision_spatial_merge_size: int = 2
    siglip_model_id: str = "google/siglip2-so400m-patch14-384"
    image_pad_token_id: int = -1          # set by tokenizer at runtime when multimodal=True
    freeze_vision_merger: bool = True


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head.
        # Check `self.ve_gate is not None` (module-attribute, trace-time constant) instead of
        # `ve is not None` (data-dependent type) so torch.compile(fullgraph=True) doesn't
        # need a runtime guard. ve_gate and ve are correlated by construction: both are
        # populated for layers where has_ve() returns True.
        if self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.moe = MoE(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.moe(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

        # Multimodal vision tower (optional — None unless config.multimodal is True).
        # Lazily constructed at init_weights() time so meta-device init isn't blocked
        # by HF model download.
        self.vision_tower = None
        if config.multimodal:
            # Placeholder attribute; actual module created in init_weights() to avoid
            # heavy work in meta-device __init__.
            self._needs_vision_tower = True
        else:
            self._needs_vision_tower = False

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            moe.router.gate:     uniform, std=1/sqrt(n_embd)
            moe.experts.w_up:           uniform, std=1/sqrt(n_embd)
            moe.experts.w_down:          zeros
            moe.shared_expert.w_up:      uniform, std=1/sqrt(n_embd)
            moe.shared_expert.w_down:    zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            # MoE: router gate and expert up-projections get uniform, down-projections get zero
            torch.nn.init.uniform_(block.moe.router.gate.weight, -s, s)
            torch.nn.init.uniform_(block.moe.experts.w_up, -s, s)
            torch.nn.init.zeros_(block.moe.experts.w_down)
            if block.moe.shared_expert is not None:
                torch.nn.init.uniform_(block.moe.shared_expert.w_up.weight, -s, s)
                torch.nn.init.zeros_(block.moe.shared_expert.w_down.weight)
            # MoE load balancing buffers (zero after to_empty from meta device)
            block.moe.router.expert_bias.zero_()
            block.moe.router.tokens_per_expert_counter.zero_()

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

        # Multimodal: build VisionTower lazily here (not in __init__ to avoid HF
        # download under meta-device init).
        if getattr(self, "_needs_vision_tower", False):
            from core.multimodal import VisionTower
            self.vision_tower = VisionTower(
                llm_hidden_size=self.config.n_embd,
                siglip_model_id=self.config.siglip_model_id,
                spatial_merge_size=self.config.vision_spatial_merge_size,
                freeze_merger=self.config.freeze_vision_merger,
            )
            # Move to same device/dtype as the trunk
            target_device = self.transformer.wte.weight.device
            self.vision_tower.to(device=target_device)
            if target_device.type == "cuda":
                # Keep frozen SigLIP in bf16 (no gradient → no precision concern)
                self.vision_tower.siglip.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                            self.resid_lambdas.numel() + self.x0_lambdas.numel())
        # MoE: only top_k/num_experts fraction of routed expert params active per token
        # Shared expert is always active so its params stay in the active count
        expert_hidden = self.transformer.h[0].moe.expert_hidden_dim
        routed_params_per_layer = self.config.num_experts * 2 * self.config.n_embd * expert_hidden
        inactive_per_layer = routed_params_per_layer * (self.config.num_experts - self.config.top_k) // self.config.num_experts
        nparams_exclude += inactive_per_layer * self.config.n_layer
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.

        For MoE, 'active_*' fields count only the parameters active per token
        (top_k out of num_experts routed experts, plus shared experts).
        Following DeepSeek convention of reporting both total and active params.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        # MoE: only top_k/num_experts fraction of routed expert params active per token
        # Shared expert is always active so its params stay in the active count
        expert_hidden = self.transformer.h[0].moe.expert_hidden_dim
        routed_params_per_layer = self.config.num_experts * 2 * self.config.n_embd * expert_hidden
        inactive_per_layer = routed_params_per_layer * (self.config.num_experts - self.config.top_k) // self.config.num_experts
        moe_inactive = inactive_per_layer * self.config.n_layer
        active_transformer_matrices = transformer_matrices - moe_inactive
        active_total = total - moe_inactive
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'active_transformer_matrices': active_transformer_matrices,
            'scalars': scalars,
            'moe_inactive': moe_inactive,
            'total': total,
            'active_total': active_total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def update_moe_balancing(self, coeff=1e-3):
        """Update expert routing bias for load balancing. Call before optimizer.step()."""
        for block in self.transformer.h:
            block.moe.router.update_expert_bias(coeff)

    def get_moe_stats(self):
        """Collect MoE routing statistics for logging. Call BEFORE update_moe_balancing (which resets counters)."""
        all_counts = []
        all_biases = []
        for block in self.transformer.h:
            router = block.moe.router
            all_counts.append(router.tokens_per_expert_counter)
            all_biases.append(router.expert_bias)
        counts = torch.stack(all_counts).float()    # (n_layer, num_experts)
        biases = torch.stack(all_biases).float()    # (n_layer, num_experts)
        # Load imbalance: coefficient of variation (std/mean) per layer, averaged
        counts_mean = counts.mean(dim=-1).clamp(min=1)
        counts_std = counts.std(dim=-1)
        load_imbalance = (counts_std / counts_mean).mean().item()
        return {
            "moe/load_imbalance": load_imbalance,
            "moe/expert_bias_std": biases.std().item(),
            "moe/expert_bias_max": biases.abs().max().item(),
        }

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean',
                pixel_values=None, grid_thw=None, image_pad_mask=None,
                modality_mask=None, image_grids_merged=None):
        """Forward pass.

        Text-only path (default): provide `idx` and optionally `targets`.

        Multimodal path: additionally provide `pixel_values` (N_imgs, 3, H, W),
        `grid_thw` (N_imgs, 3) describing patch grids, `image_pad_mask` (B, S)
        marking <image_pad> positions, and `image_grids_merged` (per-row list of
        merged-unit (T,H,W) tuples). When `image_grids_merged` is provided, 3D
        Interleaved-MRoPE is used (per multimodal_spec.md decision #7); otherwise
        the existing 1D RoPE precomputed cache is used.

        If `modality_mask` (B, S) is provided AND targets are given, per-modality
        loss decomposition is returned as a dict — otherwise scalar loss.
        """
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx) # embed current token

        # Multimodal: encode images and scatter into input embeddings at <image_pad> positions
        if pixel_values is not None:
            assert self.vision_tower is not None, (
                "Model not built with multimodal=True; cannot accept pixel_values"
            )
            assert grid_thw is not None and image_pad_mask is not None, (
                "Multimodal forward requires grid_thw and image_pad_mask alongside pixel_values"
            )
            from core.multimodal import scatter_vision_features
            vision_features = self.vision_tower(pixel_values, grid_thw)
            x = scatter_vision_features(x, vision_features, image_pad_mask)

            # Override 1D RoPE with 3D Interleaved-MRoPE per multimodal_spec.md decision #7
            # (only when image_grids_merged is provided — keeps a graceful fallback)
            if image_grids_merged is not None:
                from core.multimodal import (
                    build_3d_mrope_for_4d_apply,
                    build_position_ids_for_mm,
                )
                position_ids = build_position_ids_for_mm(
                    idx,
                    image_pad_token_id=self.config.image_pad_token_id,
                    image_grids_merged=image_grids_merged,
                )  # (3, B, T)
                head_dim = self.config.n_embd // self.config.n_head
                cos_3d, sin_3d = build_3d_mrope_for_4d_apply(position_ids, head_dim)
                cos_sin = (cos_3d, sin_3d)

                # If we're prefilling a KV cache, record the next text-axis position
                # so subsequent text-only continuation calls (with kv_cache + no pixel_values)
                # can build single-position 3D MRoPE without re-encoding images.
                if kv_cache is not None:
                    next_t = int(position_ids[0].max().item()) + 1
                    kv_cache.next_t_axis_position = next_t

        # Multimodal continuation: kv_cache has multimodal state but no new pixel_values.
        # Build per-token 3D MRoPE for the new text tokens at the next t-axis position.
        elif kv_cache is not None and kv_cache.next_t_axis_position is not None:
            from core.multimodal import build_3d_mrope_for_4d_apply
            next_t = kv_cache.next_t_axis_position
            t_axis = torch.arange(next_t, next_t + T, device=idx.device, dtype=torch.long)
            position_ids = torch.zeros(3, B, T, dtype=torch.long, device=idx.device)
            position_ids[0, :, :] = t_axis.unsqueeze(0).expand(B, T)  # text tokens: (t, 0, 0)
            head_dim = self.config.n_embd // self.config.n_head
            cos_3d, sin_3d = build_3d_mrope_for_4d_apply(position_ids, head_dim)
            cos_sin = (cos_3d, sin_3d)
            # Advance next_t for the next forward call
            kv_cache.next_t_axis_position = next_t + T

        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual

        # Activation checkpointing: recompute each block's intermediate activations
        # during backward instead of storing them. Saves ~1.5 GB/layer at B=32,T=2048
        # for cost of ~33% extra compute. Only active during training (not inference)
        # and when there's no kv_cache (kv_cache implies inference path).
        use_act_ckpt = (
            self.training
            and kv_cache is None
            and getattr(self, '_use_activation_checkpointing', False)
        )
        if use_act_ckpt:
            from torch.utils.checkpoint import checkpoint

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # ve passed as tensor on VE layers, None elsewhere. Inside attn, the
            # gate-application is guarded by `self.ve_gate is not None` (module
            # attribute, trace-time constant), so the dead branch never reads ve
            # on non-VE layers — safe for torch.compile fullgraph traces.
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            if use_act_ckpt:
                # use_reentrant=False is required for nested autocast + torch.compile compatibility
                x = checkpoint(block, x, ve, cos_sin, self.window_sizes[i], kv_cache,
                            use_reentrant=False) # use_reentrant=False is required for nested autocast + torch.compile compatibility
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]

        # Forward the lm_head (compute logits) — vanilla path
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is None:
            # inference: just return the logits directly
            return logits

        # training: given the targets, compute and return the loss
        if modality_mask is not None:
            # Per-modality decomposition: returns dict with loss + per-modality split
            from core.multimodal import per_modality_loss_decomposition
            return per_modality_loss_decomposition(logits, targets, modality_mask, ignore_index=-1)

        # Text-only / backward-compatible path
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        return loss

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42,
                pixel_values=None, grid_thw=None, image_pad_mask=None,
                image_grids_merged=None):
        """Naive autoregressive streaming inference.

        Text-only (default): ids = list of token ids; standard generate.

        Multimodal: pass pixel_values + grid_thw + image_pad_mask + image_grids_merged.
        Vision is encoded via the model's VisionTower; 3D Interleaved-MRoPE is used.
        Each generation step re-runs the full forward (including ViT) — correct but
        wasteful at scale. For optimized multimodal generation use the engine.py
        Engine class with KVCache (which leverages our multimodal KVCache extension).

        To make it super simple, this naive path assumes:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, T)

        # Multimodal: cache the current image_pad_mask (will be extended each step)
        # so build_position_ids_for_mm walks the right input shape on each forward.
        is_multimodal = pixel_values is not None and image_grids_merged is not None
        if is_multimodal:
            assert image_pad_mask is not None and grid_thw is not None
            cur_image_pad_mask = image_pad_mask.to(device)

        for _ in range(max_tokens):
            if is_multimodal:
                # Re-encode images each step (naive). Pad image_pad_mask with False for
                # any new generated text tokens.
                B, T_now = ids.shape
                if cur_image_pad_mask.shape[1] < T_now:
                    extra = T_now - cur_image_pad_mask.shape[1]
                    cur_image_pad_mask = torch.cat([
                        cur_image_pad_mask,
                        torch.zeros(B, extra, dtype=torch.bool, device=device),
                    ], dim=1)
                logits = self.forward(
                    ids,
                    pixel_values=pixel_values,
                    grid_thw=grid_thw,
                    image_pad_mask=cur_image_pad_mask,
                    image_grids_merged=image_grids_merged,
                )
            else:
                logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
