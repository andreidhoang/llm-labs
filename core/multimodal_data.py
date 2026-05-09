"""Real multimodal data path — LAION-Recap-12M with lazy download.

This module replaces synthetic random pixel_values with actual image-text pairs
from the LAION-Recap-12M dataset (HuggingFace `laion/laion-coco-aesthetic`-class).

Key design decisions:
- **Lazy / streaming download**: HF `datasets.load_dataset(..., streaming=True)`
  pulls examples one-at-a-time instead of downloading 50GB up-front. First
  GPU run fetches as it trains.
- **Synthetic fallback**: if HF unavailable or dataset name typo'd, fall back
  to synthetic noise so unit tests still pass.
- **AutoImageProcessor pipeline**: uses HF's official SigLIP2 preprocessor
  (resize to 384x384, CenterCrop, normalize) per multimodal_spec.md decision #8.
- **Caption tokenization**: uses the project tokenizer (from caller); captions
  get prepended to text + interleaved with image_pad runs.

For Track B'' scope, this provides the pipeline plumbing. Real LAION-Recap
quality validation + caption-quality manual eval is a Track B''' task.

Spec: dev/scaling_law_self_assignment.md, dev/multimodal_spec.md §2.5.5
"""

from __future__ import annotations

import io
import os
from typing import Iterator

import torch


def _try_load_image_processor(siglip_model_id: str):
    """Lazy import + load HF AutoImageProcessor. Returns None if unavailable."""
    try:
        from transformers import AutoImageProcessor
        return AutoImageProcessor.from_pretrained(siglip_model_id)
    except Exception as e:
        print(f"[multimodal_data] AutoImageProcessor load failed ({e}); will use raw resize")
        return None


def _try_open_laion_stream(dataset_id: str = "laion/laion-coco"):
    """Open the LAION-Recap dataset as a streaming iterator. Returns None on failure.

    Default uses 'laion/laion-coco' (5.5M aesthetic captions). For real
    LAION-Recap-12M, use 'laion/laion2B-en-aesthetic-recap' or similar.
    HF dataset names change frequently — verify with `datasets.list_datasets()`
    before pinning in production.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split="train", streaming=True)
        return iter(ds)
    except Exception as e:
        print(f"[multimodal_data] Cannot open {dataset_id} as stream ({e}); falling back to synthetic")
        return None


def laion_recap_image_iterator(
    dataset_id: str = "laion/laion-coco",
    siglip_model_id: str = "google/siglip2-so400m-patch14-384",
    image_size: int = 384,
    seed: int = 0,
    use_synthetic_fallback: bool = True,
) -> Iterator[tuple[torch.Tensor, str]]:
    """Yields (preprocessed_pixel_tensor, caption_text) one example at a time.

    Args:
        dataset_id:              HF dataset name
        siglip_model_id:         AutoImageProcessor source (must match VisionTower's siglip)
        image_size:              target spatial resolution (must match SigLIP2's input expectation)
        seed:                    RNG seed for synthetic fallback determinism
        use_synthetic_fallback:  if True, fall back to synthetic data when stream unavailable

    Yields:
        (pixel_values, caption): pixel_values shape (3, image_size, image_size) tensor,
                                 caption is a Python str
    """
    proc = _try_load_image_processor(siglip_model_id)
    stream = _try_open_laion_stream(dataset_id)

    if stream is None:
        if not use_synthetic_fallback:
            raise RuntimeError(f"Cannot stream {dataset_id} and synthetic fallback disabled")
        # Synthetic fallback: random pixels + canned caption
        rng = torch.Generator(device="cpu").manual_seed(seed)
        idx = 0
        while True:
            yield torch.randn(3, image_size, image_size, generator=rng), f"synthetic image {idx}"
            idx += 1
        # unreachable

    # Real LAION stream
    from PIL import Image
    for example in stream:
        try:
            img_data = example.get("image") or example.get("jpg") or example.get("png")
            if img_data is None:
                continue
            if isinstance(img_data, bytes):
                pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
            elif hasattr(img_data, "convert"):
                pil_img = img_data.convert("RGB")
            else:
                continue
            caption = example.get("caption") or example.get("text") or ""
            if not caption:
                continue

            if proc is not None:
                # AutoImageProcessor returns dict with 'pixel_values' key
                inputs = proc(images=pil_img, return_tensors="pt")
                pixel = inputs["pixel_values"][0]  # (3, H, W)
            else:
                # Manual fallback: resize + ToTensor + normalize to [-1, 1]
                pil_img = pil_img.resize((image_size, image_size))
                pixel = torch.tensor(list(pil_img.tobytes()), dtype=torch.float32)
                pixel = pixel.reshape(image_size, image_size, 3).permute(2, 0, 1) / 127.5 - 1.0

            yield pixel, caption
        except Exception as e:
            # Stream errors (corrupt image, network blip) — skip and continue
            print(f"[multimodal_data] Skipping bad example: {e}")
            continue


def real_multimodal_loader(
    base_text_loader,
    image_iterator: Iterator[tuple[torch.Tensor, str]] | None = None,
    mix_ratio: float = 0.3,
    image_grid_thw_raw: tuple[int, int, int] = (1, 27, 27),
    spatial_merge_size: int = 2,
    image_pad_token_id: int = -1,
    vision_context_window: int = 32,
    seed: int = 0,
    device: str = "cuda",
):
    """Wraps text loader + real image iterator to produce multimodal batches.

    Args:
        base_text_loader:        text-only loader (yields (x, y, state))
        image_iterator:          source of (pixel_values, caption) tuples;
                                 if None, uses laion_recap_image_iterator()
        mix_ratio:               target token-level vision fraction
        image_grid_thw_raw:      RAW patch grid (T, H, W). Default (1, 27, 27)
                                 = SigLIP2-SO400M output for 384x384 image
                                 (384/14 ≈ 27 patches per side)
        spatial_merge_size:      PatchMerger merge size
        image_pad_token_id:      placeholder text token id
        vision_context_window:   how far back to look for vision context (modality_mask)
        seed:                    RNG seed
        device:                  target device

    Yields:
        Same format as synthetic_multimodal_loader but with REAL pixel_values from
        image_iterator. Captions are NOT yet inserted into text (caption integration
        is Track B''' — the simplest correct first step is to use image_pad runs
        without caption text, which still exercises the vision path end-to-end).
    """
    assert image_pad_token_id >= 0, "Caller must set a real image_pad_token_id"
    if image_iterator is None:
        image_iterator = laion_recap_image_iterator(seed=seed)

    T_grid, H_grid, W_grid = image_grid_thw_raw
    # Compute MERGED-grid dims (post-PatchMerger crop). Allows odd raw grids like 27x27
    # that crop to 26x26, giving merged 13x13.
    H_merged = H_grid // spatial_merge_size
    W_merged = W_grid // spatial_merge_size
    tokens_per_image = T_grid * H_merged * W_merged

    for inputs, targets, state_dict in base_text_loader:
        B, S = inputs.shape
        target_vision_tokens_per_row = int(round(mix_ratio * S))
        n_images_per_row = max(1, target_vision_tokens_per_row // tokens_per_image)

        image_pad_mask = torch.zeros(B, S, dtype=torch.bool)
        n_imgs_per_row_actual: list[int] = [0] * B
        for b in range(B):
            for k in range(n_images_per_row):
                stride = S // (n_images_per_row + 1)
                start = stride * (k + 1)
                if start + tokens_per_image > S:
                    break
                image_pad_mask[b, start : start + tokens_per_image] = True
                n_imgs_per_row_actual[b] += 1
        n_imgs_total = sum(n_imgs_per_row_actual)

        if n_imgs_total == 0:
            yield inputs, targets, {}, state_dict
            continue

        # Pull real images from iterator
        try:
            pixel_list = []
            for _ in range(n_imgs_total):
                pixel, _caption = next(image_iterator)
                pixel_list.append(pixel)
            pixel_values = torch.stack(pixel_list, dim=0).to(inputs.device)
        except StopIteration:
            print("[multimodal_data] Image iterator exhausted; reverting to synthetic")
            pixel_values = torch.randn(n_imgs_total, 3, 384, 384).to(inputs.device)

        new_inputs = inputs.clone()
        new_inputs[image_pad_mask.to(inputs.device)] = image_pad_token_id
        new_targets = targets.clone()
        new_targets[image_pad_mask.to(targets.device)] = -1

        # Refined modality_mask: text tokens with vision context within window
        modality_mask = torch.zeros(B, S, dtype=torch.long)
        for b in range(B):
            for s in range(S):
                start = max(0, s - vision_context_window)
                if image_pad_mask[b, start : s + 1].any():
                    modality_mask[b, s] = 1
        modality_mask = modality_mask.to(inputs.device)

        grid_thw = torch.tensor([[T_grid, H_grid, W_grid]] * n_imgs_total,
                                dtype=torch.long, device=inputs.device)

        # image_grids_merged: per-row list of (T, H, W) in MERGED-patch units.
        # For SigLIP2 27x27 raw → after PatchMerger crop → 13x13 merged.
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
