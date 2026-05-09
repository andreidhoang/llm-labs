"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import torch
import pyarrow.parquet as pq

from core.common import get_dist_info
from core.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets


# =============================================================================
# Multimodal wrapper — synthetic image mode
# =============================================================================
#
# Wraps the text-only loader and periodically substitutes runs of <image_pad>
# tokens (with synthetic random pixel_values) so the multimodal training path
# can be exercised end-to-end on CPU without external image datasets.
#
# Real image data (LAION-Recap, OBELICS-style interleaved) integration is a
# Track B'' task. This wrapper is for:
#   - Local CPU smoke tests
#   - Initial GPU validation that the multimodal forward path runs
#   - F1.s / P0.A reactive sanity check
#
# Determinism: same seed -> same image_pad layout + same synthetic pixels.

def synthetic_multimodal_loader(
    base_loader,
    mix_ratio: float = 0.3,
    image_grid_thw_raw: tuple[int, int, int] = (1, 4, 4),
    spatial_merge_size: int = 2,
    image_pad_token_id: int = -1,
    image_size_pixels: int = 16,
    vision_context_window: int = 32,
    seed: int = 0,
    device: str = "cuda",
):
    """Wraps a text-only loader to inject synthetic image_pad runs.

    Args:
        base_loader:           the underlying text-only loader (yields (x, y, state))
        mix_ratio:             target token-level fraction that should be vision tokens
        image_grid_thw_raw:    RAW patch grid per synthetic image (T, H, W) — what the
                                vision encoder produces BEFORE PatchMerger compression.
                                Number of merged tokens per image = T*H*W / merge^2.
        spatial_merge_size:    PatchMerger merge size (must match VisionTower config)
        image_pad_token_id:    placeholder text token id (must be in tokenizer vocab; pass from caller)
        image_size_pixels:     synthetic image side length in pixels (kept tiny for CPU testing)
        seed:                  RNG seed for determinism
        device:                target device for synthetic pixel_values

    Yields:
        (inputs, targets, batch_extras, state_dict)
        where batch_extras = {
            'pixel_values':   (N_imgs, 3, H, W) tensor,
            'grid_thw':       (N_imgs, 3) RAW patch grid passed to PatchMerger,
            'image_pad_mask': (B, S) bool — True at MERGED-token positions,
            'modality_mask':  (B, S) long — 1 at vision positions, 0 at text,
        }
    """
    assert image_pad_token_id >= 0, "Caller must set a real image_pad_token_id from the tokenizer"
    T_grid, H_grid, W_grid = image_grid_thw_raw
    assert H_grid % spatial_merge_size == 0 and W_grid % spatial_merge_size == 0, (
        f"raw grid ({H_grid},{W_grid}) must be divisible by merge={spatial_merge_size}"
    )
    # Number of MERGED tokens per image = positions that need image_pad placeholders
    tokens_per_image = T_grid * (H_grid // spatial_merge_size) * (W_grid // spatial_merge_size)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    for inputs, targets, state_dict in base_loader:
        # inputs / targets: (B, S) long tensors
        B, S = inputs.shape
        # Decide how many image runs to insert per row to hit ~mix_ratio
        target_vision_tokens_per_row = int(round(mix_ratio * S))
        n_images_per_row = max(1, target_vision_tokens_per_row // tokens_per_image)

        # Build new image_pad_mask, choosing run start positions deterministically.
        # Track per-row image count so we can build image_grids_merged correctly.
        image_pad_mask = torch.zeros(B, S, dtype=torch.bool)
        n_imgs_per_row_actual: list[int] = [0] * B
        for b in range(B):
            # Place runs spaced ~uniformly through the row, leaving room for the run length
            for k in range(n_images_per_row):
                stride = S // (n_images_per_row + 1)
                start = stride * (k + 1)
                if start + tokens_per_image > S:
                    break
                image_pad_mask[b, start : start + tokens_per_image] = True
                n_imgs_per_row_actual[b] += 1
        n_imgs_total = sum(n_imgs_per_row_actual)

        if n_imgs_total == 0:
            # Sequence too short for even one image; pass through as text-only
            yield inputs, targets, {}, state_dict
            continue

        # Overwrite text tokens at pad positions with image_pad_token_id
        new_inputs = inputs.clone()
        new_inputs[image_pad_mask.to(inputs.device)] = image_pad_token_id
        # Targets at vision positions are set to ignore (-1) — vision is context, not target
        new_targets = targets.clone()
        # We mask the *target* corresponding to predicting *next* token from a vision pos;
        # simplest valid choice for synthetic mode is to ignore the next-token loss at vision positions.
        new_targets[image_pad_mask.to(targets.device)] = -1

        # Build batch_extras
        # modality_mask: per multimodal_spec.md §2.5.5, categorize TEXT tokens by
        # whether vision context is in the recent attention window. Vision positions
        # themselves are also marked 1 (but they're ignored in CE loss anyway).
        # text-with-vision-context = 1; text-only-context = 0.
        modality_mask = torch.zeros(B, S, dtype=torch.long)
        for b in range(B):
            for s in range(S):
                # Look back vision_context_window positions for any image_pad
                start = max(0, s - vision_context_window)
                if image_pad_mask[b, start : s + 1].any():
                    modality_mask[b, s] = 1
        modality_mask = modality_mask.to(inputs.device)
        # Synthetic pixel_values: random normal, shape (N_imgs, 3, H_pix, W_pix)
        pixel_values = torch.randn(n_imgs_total, 3, image_size_pixels, image_size_pixels, generator=rng)
        pixel_values = pixel_values.to(inputs.device)
        grid_thw = torch.tensor([[T_grid, H_grid, W_grid]] * n_imgs_total,
                                dtype=torch.long, device=inputs.device)

        # image_grids_merged: per-row list of (T, H, W) in MERGED-patch units (post-PatchMerger crop).
        # Required by build_position_ids_for_mm in core/multimodal.py.
        # Note: merged grid uses (H_grid // merge) * merge / merge = H_grid // merge per axis,
        # which equals the post-crop merged dim.
        H_merged = H_grid // spatial_merge_size
        W_merged = W_grid // spatial_merge_size
        image_grids_merged: list[list[tuple[int, int, int]]] = [
            [(T_grid, H_merged, W_merged)] * n for n in n_imgs_per_row_actual
        ]

        batch_extras = {
            "pixel_values": pixel_values,
            "grid_thw": grid_thw,
            "image_pad_mask": image_pad_mask.to(inputs.device),
            "modality_mask": modality_mask,
            "image_grids_merged": image_grids_merged,
        }
        yield new_inputs, new_targets, batch_extras, state_dict
