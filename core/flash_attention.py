"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on non-Hopper GPUs (including Blackwell), MPS, and CPU.

Usage (drop-in replacement for FA3):
    from core.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import contextlib

import torch
import torch.nn.functional as F


# =============================================================================
# Detection: 3-tier dispatch FA3 → FA2 → SDPA
#   FA3:  HF Kernels Hub (varunneal/flash-attention-3) — Hopper-only, fastest
#   FA2:  local pip-installed flash_attn package — Hopper or Ampere, ~1.5× SDPA
#   SDPA: torch.nn.functional.scaled_dot_product_attention — universal fallback
# =============================================================================

def _smoke_test_fa3(fa3_module):
    """FA3 hub kernels can import OK but fail at first call due to ABI mismatch
    (e.g. 'undefined symbol _ZN3c104cuda9SetDeviceEab' on NGC torch builds).
    Test-call with a tiny tensor to verify the kernel actually executes.
    """
    try:
        q = torch.randn(1, 8, 1, 64, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(1, 8, 1, 64, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(1, 8, 1, 64, dtype=torch.bfloat16, device='cuda')
        _ = fa3_module.flash_attn_func(q, k, v, causal=True)
        return True
    except Exception:
        return False


def _load_flash_attention_3():
    """Try to load Flash Attention 3 from HF Kernels Hub (Hopper sm90 only)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None  # FA3 hub kernels are sm90-only
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        fa3 = get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None
    # Smoke-test the kernel actually runs (catches ABI mismatch on NGC builds)
    if not _smoke_test_fa3(fa3):
        return None
    return fa3


def _load_flash_attention_2():
    """Try to load FA2 from the local flash_attn pip package (any sm80+ GPU)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return None  # FA2 requires Ampere+ (sm80+)
        import flash_attn
        # FA2 needs version >= 2.0; older versions had different APIs
        v = tuple(int(x) for x in flash_attn.__version__.split('.')[:2] if x.isdigit())
        if v < (2, 0):
            return None
        # Re-export under the FA3-compatible interface
        from types import SimpleNamespace
        return SimpleNamespace(
            flash_attn_func=flash_attn.flash_attn_func,
            flash_attn_with_kvcache=flash_attn.flash_attn_with_kvcache,
        )
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
_fa2 = None if HAS_FA3 else _load_flash_attention_2()
HAS_FA2 = _fa2 is not None

# Override for testing: set to 'fa3', 'fa2', 'sdpa', or None (auto)
_override_impl = None


def _resolve_impl():
    """Decide once which attention impl to use, based on availability, override, and dtype.

    Returns: ('fa3', _fa3) | ('fa2', _fa2) | ('sdpa', None)
    """
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return 'fa3', _fa3
    if _override_impl == 'fa2':
        assert HAS_FA2, "Cannot override to FA2: not available on this hardware"
        return 'fa2', _fa2
    if _override_impl == 'sdpa':
        return 'sdpa', None
    # Auto: FA3 (BF16 only) > FA2 (BF16/FP16) > SDPA
    from core.common import COMPUTE_DTYPE
    if HAS_FA3 and COMPUTE_DTYPE == torch.bfloat16:
        return 'fa3', _fa3
    if HAS_FA2 and COMPUTE_DTYPE in (torch.bfloat16, torch.float16):
        return 'fa2', _fa2
    return 'sdpa', None


_IMPL_NAME, _IMPL = _resolve_impl()
USE_FA3 = _IMPL_NAME == 'fa3'
USE_FA2 = _IMPL_NAME == 'fa2'
USE_FA = USE_FA3 or USE_FA2  # any flash attn variant active
USE_ANNOTATED_MATH = False


# =============================================================================
# SDPA helpers
# =============================================================================
def _nvtx_range(name):
    if torch.cuda.is_available():
        return torch.cuda.nvtx.range(name)
    return contextlib.nullcontext()


def annotated_scaled_dot_product_attention(Q, K, V, mask=None, pdrop=0.0):
    """
    Explicit math attention annotated with NVTX markers for nsys timelines.
    Q, K, V are expected in (B, H, T, D) layout.
    """
    with _nvtx_range("sdpa_full"):
        d_k = Q.size(-1)

        with _nvtx_range("attn_scores_QK"):
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
            if mask is not None:
                masked_positions = ~mask if mask.dtype == torch.bool else mask == 0
                scores = scores.masked_fill(masked_positions, float("-inf"))

        with _nvtx_range("attn_softmax"):
            attn_weights = F.softmax(scores, dim=-1)
            if pdrop > 0.0:
                attn_weights = F.dropout(attn_weights, p=pdrop)

        with _nvtx_range("attn_out_PV"):
            output = torch.matmul(attn_weights, V)

    return output


def _causal_window_mask(Tq, Tk, window, device):
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return mask


def _annotated_math_attention(q, k, v, window_size, enable_gqa):
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    if Tq == 1 and window >= 0 and window < Tk:
        start = max(0, Tk - (window + 1))
        k = k[:, :, start:, :]
        v = v[:, :, start:, :]
        Tk = k.size(2)

    if enable_gqa:
        groups = q.size(1) // k.size(1)
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    mask = _causal_window_mask(Tq, Tk, window, q.device).view(1, 1, Tq, Tk)
    return annotated_scaled_dot_product_attention(q, k, v, mask=mask)


def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    if USE_ANNOTATED_MATH:
        return _annotated_math_attention(q, k, v, window_size, enable_gqa)

    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA:
        # Both FA3 (HF Hub) and FA2 (local pip) expose the same flash_attn_func signature
        return _IMPL.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA:
        # Both FA3 + FA2 expose flash_attn_with_kvcache with the same signature
        return _IMPL.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
    annotated_scaled_dot_product_attention=annotated_scaled_dot_product_attention,
)
