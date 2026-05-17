"""
prepare_auto.py — frozen scaffold for the autoresearch Tier 1 loop.

This file is read-only during agent sessions. It defines the fixed constants
(vocab size, sequence length, time budget, evaluation token count) that make
all 5-minute experiments commensurable, and exposes the small set of utilities
that auto/train_auto.py imports.

Mirrors github.com/karpathy/autoresearch/prepare.py. Differences:
- vocab_size=8192 kept the same as Karpathy for cross-comparable val_bpb.
- Data download / tokenizer training / val loader / evaluate_bpb are thin
  wrappers around llm-labs's existing core/ utilities (RustBPETokenizer,
  tokenizing_distributed_data_loader_bos_bestfit, evaluate_bpb) rather than
  re-implementations. The wrapper keeps the agent's edit surface (train_auto.py)
  clean and decouples it from core/'s evolution.

Run once after a fresh checkout / clone:
    python auto/prepare_auto.py

This is idempotent: it skips downloads already on disk and skips tokenizer
training if tokenizer.pkl + token_bytes.pt are present.
"""
import os
import sys
import time
import pickle
from multiprocessing import Pool

# Ensure `from core.X import Y` works regardless of invocation directory:
# add the repo root (parent of auto/) to sys.path before importing core.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

# Imports from core/ are allowed here because this file is FROZEN.
# train_auto.py (agent-editable) imports only from prepare_auto, never from core.
from core.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
from core.dataset import DATA_DIR, MAX_SHARD, download_single_file, parquets_iter_batched
from core.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from core.loss_eval import evaluate_bpb as _evaluate_bpb

# Re-export for train_auto.py: a single short name to import.
Tokenizer = RustBPETokenizer

# =============================================================================
# FROZEN CONSTANTS — do not modify in autoresearch runs.
# These are the dimensions that must stay constant for measurements to be
# comparable. If you change them, you invalidate prior results.
# =============================================================================
VOCAB_SIZE = 8192                       # Karpathy default; <-> his published val_bpb numbers
MAX_SEQ_LEN = 2048                      # context length used during training AND eval
TIME_BUDGET = 300                       # seconds of wall-clock training per experiment
EVAL_TOKENS = 40 * 524288               # ~21M tokens of val data for evaluate_bpb (matches Karpathy)
MIN_TRAIN_SHARDS = 4                    # enough shards for BPE training + a few epochs at d8
VAL_SHARD_INDEX = MAX_SHARD             # pinned: shard_06542.parquet is always val
TOKENIZER_SUBDIR = "tokenizer_auto"     # kept distinct from llm-labs's main 32768-vocab tokenizer
# =============================================================================

# Resolve the tokenizer directory off the same base used by core.dataset.
# This keeps the autoresearch tokenizer alongside the data cache it was trained
# on, without colliding with llm-labs's existing vocab=32768 tokenizer at
# ~/.cache/nanochat/tokenizer.
_BASE_DIR = os.path.dirname(DATA_DIR)  # ~/.cache/nanochat
TOKENIZER_DIR = os.path.join(_BASE_DIR, TOKENIZER_SUBDIR)


# ---------------------------------------------------------------------------
# Data preparation

def download_data(num_train_shards: int = MIN_TRAIN_SHARDS, num_workers: int = 4) -> None:
    """Download the first N train shards + the pinned val shard. Idempotent."""
    os.makedirs(DATA_DIR, exist_ok=True)
    ids = list(range(num_train_shards)) + [VAL_SHARD_INDEX]
    print(f"[prepare_auto] downloading {len(ids)} shards into {DATA_DIR}")
    with Pool(processes=num_workers) as pool:
        ok = pool.map(download_single_file, ids)
    n_ok = sum(1 for s in ok if s)
    print(f"[prepare_auto] downloaded {n_ok}/{len(ids)} shards")
    if n_ok != len(ids):
        raise RuntimeError("data download failed — check network and retry")


# ---------------------------------------------------------------------------
# Tokenizer training (one-time, ~2 minutes on the climbmix training shards)

def _train_text_iterator(max_chars: int = 1_000_000_000, doc_cap: int = 10_000):
    """Yield documents from the train split, capped per-doc and in total chars.

    Caps mirror llm-labs/scripts/tok_train.py. The total-char cap (1B) is half
    Karpathy-llm-labs's default 2B because vocab=8192 needs less data to learn
    common merges than vocab=32768.
    """
    n = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            if len(doc) > doc_cap:
                doc = doc[:doc_cap]
            n += len(doc)
            yield doc
            if n > max_chars:
                return


def train_tokenizer() -> None:
    """Train an 8192-vocab BPE on climbmix train shards and save with token_bytes.pt."""
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(_train_text_iterator(), VOCAB_SIZE)
    print(f"[prepare_auto] BPE training: {time.time() - t0:.1f}s, vocab_size={tokenizer.get_vocab_size()}")

    # Save the tiktoken encoding to tokenizer.pkl in TOKENIZER_DIR.
    tokenizer.save(TOKENIZER_DIR)

    # Build the token_bytes tensor used by evaluate_bpb. Special tokens get 0
    # so they don't contribute to bytes-per-byte. This mirrors tok_train.py.
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_bytes = []
    for tid in range(vocab_size):
        s = tokenizer.decode([tid])
        token_bytes.append(0 if s in special_set else len(s.encode("utf-8")))
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
    torch.save(token_bytes, os.path.join(TOKENIZER_DIR, "token_bytes.pt"))
    print(f"[prepare_auto] saved tokenizer + token_bytes to {TOKENIZER_DIR}")


# ---------------------------------------------------------------------------
# Runtime utilities used by auto/train_auto.py

def load_tokenizer() -> RustBPETokenizer:
    """Load the autoresearch tokenizer (vocab=8192) trained by train_tokenizer()."""
    return RustBPETokenizer.from_directory(TOKENIZER_DIR)


def load_token_bytes(device: str = "cpu") -> torch.Tensor:
    """Load token_bytes tensor (int32, shape (VOCAB_SIZE,)) for evaluate_bpb."""
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing. Run `python auto/prepare_auto.py` once to build it."
        )
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def make_dataloader(tokenizer, B: int, T: int = MAX_SEQ_LEN, split: str = "train", device: str = "cuda"):
    """Build a train or val dataloader. Single-GPU (no DDP), BOS-aligned best-fit.

    The underlying dataloader is DDP-aware but get_dist_info() returns world=1
    when torch.distributed is not initialized, so it works single-GPU as-is.

    Yields (inputs, targets) of shape (B, T) on `device`.
    """
    assert split in ("train", "val")
    return tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, B, T, split=split, device=device,
    )


@torch.no_grad()
def evaluate_bpb(model, tokenizer, B: int, T: int = MAX_SEQ_LEN, device: str = "cuda") -> float:
    """Compute val_bpb at full EVAL_TOKENS budget. The single number Karpathy compares.

    Bits-per-byte normalization makes the metric independent of vocab size, so
    a vocab=8192 model and a vocab=32768 model can be compared directly.
    """
    token_bytes = load_token_bytes(device=device)
    val_loader = make_dataloader(tokenizer, B, T, split="val", device=device)
    eval_steps = EVAL_TOKENS // (B * T)
    return _evaluate_bpb(model, val_loader, eval_steps, token_bytes)


# ---------------------------------------------------------------------------
# Entry point: idempotent one-time setup.

def _tokenizer_exists() -> bool:
    return (
        os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer.pkl"))
        and os.path.exists(os.path.join(TOKENIZER_DIR, "token_bytes.pt"))
    )


def _enough_shards_present() -> bool:
    needed = list(range(MIN_TRAIN_SHARDS)) + [VAL_SHARD_INDEX]
    return all(
        os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")) for i in needed
    )


if __name__ == "__main__":
    if not _enough_shards_present():
        download_data()
    else:
        print(f"[prepare_auto] shards already present in {DATA_DIR}")

    if not _tokenizer_exists():
        train_tokenizer()
    else:
        print(f"[prepare_auto] tokenizer already present in {TOKENIZER_DIR}")

    print("[prepare_auto] ready")
