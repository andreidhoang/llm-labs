"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from core.common import compute_init, autodetect_device_type
from core.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding = None
        # Multimodal continuation state (set by GPT.forward after multimodal prefill).
        # When non-None, GPT.forward builds 3D MRoPE for new tokens at this t-axis position
        # instead of using the precomputed 1D RoPE cache. None = text-only mode.
        self.next_t_axis_position = None  # type: int | None

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embedding = None
        self.next_t_axis_position = None

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        # Copy smear state: expand batch=1 prev_embedding to num_samples
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()
        # Copy multimodal continuation state (next_t_axis_position is per-cache, not per-batch)
        self.next_t_axis_position = other.next_t_axis_position

# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        # NOTE: setting the dtype here and in this way is an ugly hack.
        # Currently the repo assumes that cuda -> bfloat16 and everything else -> float32.
        # We need to know the dtype here to call __init__ on KVCache and pre-allocate its tensors.
        # As a quick hack, we're making generate() function inherit and know about this repo-wise assumption.
        # I think there has to be a bigger refactor to deal with device/dtype tracking across the codebase.
        # In particular, the KVCache should allocate its tensors lazily
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


class SpeculativeEngine:
    """
    Lossless speculative decoding (Leviathan et al. 2023).

    Draft model proposes K tokens autoregressively; target verifies all K in one forward pass.
    Rejection sampling is mathematically equivalent to sampling from the target directly.

    Economics on H100 with 3B target / 300M draft:
      - Draft overhead:  K × 10% of one target forward ≈ 40% (K=4)
      - Verification:    1 target forward over K tokens (memory-bandwidth-bound ≈ 1× target)
      - Bonus forward:   1 target forward for the bonus token
      - Total cost:      ~2.4 target-equivalents per step
      - Tokens per step: (1 - α^K)/(1 - α) + 1 ≈ 3.9 at α=0.8, K=4
      - Net speedup:     3.9 / 2.4 ≈ 1.6×  (improves with better draft → higher α)

    Rollback: on partial acceptance (j < K), cache_seqlens -= (K - j). FA3 has already
    written K KV pairs to the cache, but future writes overwrite the stale suffix, so
    the rollback is safe and O(1).

    Smear approximation: prev_embedding is restored to its pre-draft value on rollback.
    This is off by j positions in the smear gate but the gate contribution is small in
    practice. Exact restoration would require saving snapshots during the draft phase.
    """

    def __init__(self, target_model, draft_model, tokenizer, K: int = 4):
        self.target = target_model
        self.draft = draft_model
        self.tokenizer = tokenizer
        self.K = K

    # ------------------------------------------------------------------
    def _make_kv(self, model, batch_size, seq_len, device, dtype):
        m = model.config
        return KVCache(
            batch_size=batch_size, seq_len=seq_len,
            num_heads=m.n_kv_head, head_dim=m.n_embd // m.n_head,
            num_layers=m.n_layer, device=device, dtype=dtype,
        )

    def _compute_probs(self, logits_1_vocab, temperature, top_k):
        """
        logits_1_vocab: (1, vocab) raw logits
        Returns: (vocab,) probability tensor under temperature + top_k sampling.
        Temperature=0 returns a one-hot at argmax (for greedy speculative decoding).
        """
        v = logits_1_vocab[0].float()  # (vocab,)
        if top_k is not None and top_k > 0:
            k = min(top_k, v.size(-1))
            threshold = torch.topk(v, k).values[-1]
            v = v.masked_fill(v < threshold, float('-inf'))
        if temperature == 0.0:
            p = torch.zeros_like(v)
            p[v.argmax()] = 1.0
            return p
        return F.softmax(v / temperature, dim=-1)

    def _sample_from_probs(self, probs, rng, temperature):
        """probs: (vocab,) → scalar token id."""
        if temperature == 0.0:
            return int(probs.argmax().item())
        return int(torch.multinomial(probs, 1, generator=rng).item())

    def _adjusted_sample(self, q_probs, p_probs, rng, temperature):
        """
        Sample from max(0, q - p) / Z — the correction distribution on rejection.
        Guarantees the combined accept/reject process matches the target distribution.
        """
        diff = (q_probs - p_probs).clamp(min=0.0)
        z = diff.sum()
        if z < 1e-9:
            return self._sample_from_probs(q_probs, rng, temperature)
        return int(torch.multinomial(diff / z, 1, generator=rng).item())

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        Drop-in replacement for Engine.generate(), yields (token_column, token_masks).
        token_column is length-1 (num_samples=1 only); tokens are yielded one at a time
        for API compatibility even though multiple tokens are computed per speculative step.
        """
        assert num_samples == 1, "SpeculativeEngine: num_samples=1 (multi-sample support TODO)"
        assert isinstance(tokens, list) and isinstance(tokens[0], int)

        device = self.target.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        K = self.K

        get_special = lambda s: self.tokenizer.encode_special(s)
        stop_tokens = {get_special("<|assistant_end|>"), self.tokenizer.get_bos_token_id()}

        # +K: verification writes K tokens before rollback; cache must hold them temporarily
        seq_cap = len(tokens) + (max_tokens or self.target.config.sequence_len) + K
        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # ----- Prefill both models -----
        t_kv = self._make_kv(self.target, 1, seq_cap, device, dtype)
        d_kv = self._make_kv(self.draft, 1, seq_cap, device, dtype)
        # Both models process the full prompt; we keep the last-position logits.
        t_logits = self.target.forward(ids, kv_cache=t_kv)[:, -1, :]      # (1, vocab)
        d_logits = self.draft.forward(ids, kv_cache=d_kv)[:, -1, :]       # (1, vocab)

        num_generated = 0

        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break

            # ---- DRAFT PHASE ----
            # K autoregressive draft forwards, recording full prob distributions.
            draft_tokens: list[int] = []
            draft_probs:  list[torch.Tensor] = []   # each: (vocab,) float32

            # Snapshot smear states for rollback (O(1), just one tensor clone).
            t_smear = t_kv.prev_embedding.clone() if t_kv.prev_embedding is not None else None
            d_smear = d_kv.prev_embedding.clone() if d_kv.prev_embedding is not None else None

            cur_d = d_logits
            for _ in range(K):
                d_probs = self._compute_probs(cur_d, temperature, top_k)  # (vocab,)
                d_k = self._sample_from_probs(d_probs, rng, temperature)
                draft_tokens.append(d_k)
                draft_probs.append(d_probs)
                d_id = torch.tensor([[d_k]], dtype=torch.long, device=device)
                cur_d = self.draft.forward(d_id, kv_cache=d_kv)[:, -1, :]
            # d_kv now at N+K; cur_d is the draft logit at N+K (discarded for now)

            # ---- VERIFY PHASE ----
            # Single target forward over all K draft tokens.
            # t_verify: (1, K, vocab) — position k gives P_target(·|prefix+draft[0..k])
            draft_ids = torch.tensor([draft_tokens], dtype=torch.long, device=device)
            t_verify = self.target.forward(draft_ids, kv_cache=t_kv)   # (1, K, vocab)
            # t_kv now at N+K

            # ---- REJECTION SAMPLING ----
            # Logit used to CHECK draft[k]:
            #   k=0  → t_logits  (P_target at position N, before any draft token)
            #   k≥1  → t_verify[:, k-1, :]
            accepted = 0
            bonus_token = None
            t_probs_prev = self._compute_probs(t_logits, temperature, top_k)  # (vocab,) at N

            for k in range(K):
                check_logit = t_logits if k == 0 else t_verify[:, k - 1, :]
                t_probs_k = self._compute_probs(check_logit, temperature, top_k)
                d_probs_k = draft_probs[k]
                d_k = draft_tokens[k]

                p_t = float(t_probs_k[d_k])
                p_d = float(d_probs_k[d_k])
                accept_prob = min(1.0, p_t / p_d) if p_d > 1e-12 else 0.0

                if torch.rand(1, generator=rng).item() <= accept_prob:
                    accepted += 1
                    t_probs_prev = t_probs_k   # update running "previous" probs for next k
                else:
                    # Rejection: sample correction token, then break
                    bonus_token = self._adjusted_sample(t_probs_k, d_probs_k, rng, temperature)
                    break
            else:
                # All K accepted: bonus from the last verify position
                bonus_token = self._sample_from_probs(
                    self._compute_probs(t_verify[:, K - 1, :], temperature, top_k), rng, temperature
                )

            # ---- ROLLBACK ----
            # Both caches advanced by K; we keep only `accepted` draft tokens.
            rollback = K - accepted
            if rollback > 0:
                t_kv.cache_seqlens -= rollback
                d_kv.cache_seqlens -= rollback
                # Restore smear state (approximation: off by `accepted` positions, acceptable)
                t_kv.prev_embedding = t_smear
                d_kv.prev_embedding = d_smear

            # ---- COMMIT BONUS ----
            # Feed bonus through both caches so next round starts from a clean state.
            # This also produces t_logits / d_logits for the next speculative step.
            bonus_id = torch.tensor([[bonus_token]], dtype=torch.long, device=device)
            t_logits = self.target.forward(bonus_id, kv_cache=t_kv)[:, -1, :]
            d_logits = self.draft.forward(bonus_id, kv_cache=d_kv)[:, -1, :]

            # ---- YIELD ----
            output = draft_tokens[:accepted] + [bonus_token]
            for tok in output:
                num_generated += 1
                yield [tok], [1]
                if tok in stop_tokens:
                    return

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """Non-streaming wrapper, matches Engine.generate_batch signature."""
        assert num_samples == 1
        get_special = lambda s: self.tokenizer.encode_special(s)
        stop_tokens = {get_special("<|assistant_end|>"), self.tokenizer.get_bos_token_id()}
        result = tokens.copy()
        masks  = [0] * len(tokens)
        for token_col, mask_col in self.generate(tokens, num_samples=1, **kwargs):
            tok = token_col[0]
            if tok in stop_tokens:
                break
            result.append(tok)
            masks.append(mask_col[0])
        return [result], [masks]


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
