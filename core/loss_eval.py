"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist


def _accumulate_bpb_stats(total_nats, total_bytes, loss2d, targets, token_bytes, mask=None):
    """Accumulate nats/bytes for valid target tokens, optionally restricted by mask."""
    loss_flat = loss2d.reshape(-1)
    targets_flat = targets.reshape(-1)
    valid = targets_flat >= 0
    if mask is not None:
        valid = valid & mask.reshape(-1).bool()
    targets_safe = torch.where(valid, targets_flat, torch.zeros_like(targets_flat))
    bytes_flat = torch.where(
        valid,
        token_bytes[targets_safe],
        torch.zeros_like(targets_flat, dtype=token_bytes.dtype),
    )
    total_nats += (loss_flat * (bytes_flat > 0)).sum()
    total_bytes += bytes_flat.sum()
    return total_nats, total_bytes


def _finish_bpb(total_nats, total_bytes):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    return total_nats / (math.log(2) * total_bytes)

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        total_nats, total_bytes = _accumulate_bpb_stats(total_nats, total_bytes, loss2d, y, token_bytes)
    return _finish_bpb(total_nats, total_bytes)


@torch.no_grad()
def evaluate_multimodal_bpb(model, batches, steps, token_bytes):
    """
    Evaluate BPB on multimodal batches and split it by modality_mask.

    The model's multimodal forward returns a logging dict when modality_mask is
    passed, so for per-token BPB we keep modality_mask locally and omit it from
    the model kwargs. This preserves fusion (pixel_values/grid/image_pad_mask)
    while returning unreduced token losses.
    """
    device = model.get_device()
    totals = {
        "total": [
            torch.tensor(0.0, dtype=torch.float32, device=device),
            torch.tensor(0, dtype=torch.int64, device=device),
        ],
        "text": [
            torch.tensor(0.0, dtype=torch.float32, device=device),
            torch.tensor(0, dtype=torch.int64, device=device),
        ],
        "vision": [
            torch.tensor(0.0, dtype=torch.float32, device=device),
            torch.tensor(0, dtype=torch.int64, device=device),
        ],
    }
    total_text_tokens = torch.tensor(0, dtype=torch.int64, device=device)
    total_vision_tokens = torch.tensor(0, dtype=torch.int64, device=device)
    batch_iter = iter(batches)
    for _ in range(steps):
        batch = next(batch_iter)
        if len(batch) == 4:
            x, y, batch_extras, _state = batch
        elif len(batch) == 3:
            x, y, _state = batch
            batch_extras = {}
        else:
            x, y = batch
            batch_extras = {}

        modality_mask = batch_extras.get("modality_mask")
        model_extras = {k: v for k, v in batch_extras.items() if k != "modality_mask"}
        loss2d = model(x, y, loss_reduction='none', **model_extras)
        totals["total"][0], totals["total"][1] = _accumulate_bpb_stats(
            totals["total"][0], totals["total"][1], loss2d, y, token_bytes,
        )
        if modality_mask is not None:
            valid = y >= 0
            text_mask = (modality_mask == 0) & valid
            vision_mask = (modality_mask == 1) & valid
            total_text_tokens += text_mask.sum()
            total_vision_tokens += vision_mask.sum()
            totals["text"][0], totals["text"][1] = _accumulate_bpb_stats(
                totals["text"][0], totals["text"][1], loss2d, y, token_bytes, text_mask,
            )
            totals["vision"][0], totals["vision"][1] = _accumulate_bpb_stats(
                totals["vision"][0], totals["vision"][1], loss2d, y, token_bytes, vision_mask,
            )

    if dist.is_initialized():
        dist.all_reduce(total_text_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_vision_tokens, op=dist.ReduceOp.SUM)

    return {
        "bpb": _finish_bpb(totals["total"][0], totals["total"][1]),
        "bpb_text": _finish_bpb(totals["text"][0], totals["text"][1]),
        "bpb_vision": _finish_bpb(totals["vision"][0], totals["vision"][1]),
        "n_text": total_text_tokens.item(),
        "n_vision": total_vision_tokens.item(),
        "r_actual": total_vision_tokens.item() / max(1, total_text_tokens.item() + total_vision_tokens.item()),
    }
