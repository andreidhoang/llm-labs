# Ba tối ưu hoá wallclock cần port từ nanochat v2 — Tutorial cấp thấp

> Đối tượng: engineer đã quen PyTorch, hiểu BF16/FP32, biết DDP cơ bản.
> Mục tiêu: hiểu **tận đáy** cơ chế của 3 tối ưu, vì sao nó tiết kiệm wallclock,
> tensor di chuyển ra sao trong HBM/SRAM, và cách ghép vào `llm/core/`.

Thứ tự ROI (return-on-investment) wallclock đã sắp xếp giảm dần:

1. **Full-forward `torch.compile(fullgraph=True, dynamic=False, mode="max-autotune")`** — lợi ích lớn nhất, công nhiều nhất.
2. **Chunked cross-entropy** — bỏ tensor logits FP32 8.6 GB, ~30 dòng code, lợi ích cộng dồn với grad-accum.
3. **Per-block activation checkpointing** — cờ tuỳ chọn, chỉ bật khi memory-bound.

---

## Phần 0 — Bối cảnh: cái gì giới hạn wallclock?

Một bước (step) huấn luyện trên 1 GPU H100 có thể chia làm 4 thành phần thời gian:

```
┌─────────────────────────────────────────────────────────────┐
│  step_time = T_compute + T_mem + T_kernel_launch + T_comm   │
└─────────────────────────────────────────────────────────────┘
            │            │            │              │
            │            │            │              └─ NCCL all-reduce / all-gather
            │            │            └─ overhead khi Python gọi mỗi CUDA kernel (~5-10 µs)
            │            └─ HBM↔SRAM bandwidth, peak memory bóp microbatch
            └─ thực sự tính trên tensor core (FP8/BF16 GEMM, FA3)
```

Cả 3 tối ưu này đánh vào 3 thành phần khác nhau:

| Tối ưu | Tấn công thành phần nào |
|---|---|
| `torch.compile(fullgraph)` | `T_kernel_launch` + `T_mem` (fusion ⇒ ít HBM round-trip) |
| Chunked CE | `T_mem` (peak memory ⇒ cho phép microbatch lớn ⇒ đẩy `T_compute` lên gần đỉnh) |
| Activation checkpointing | `T_mem` (đánh đổi `T_compute` để mở rộng microbatch) |

---

## Phần 1 — `torch.compile(fullgraph=True, dynamic=False, mode="max-autotune")`

### 1.1. Mỗi cờ làm gì ở tầng dưới?

| Cờ | Cơ chế thực thi |
|---|---|
| `fullgraph=True` | Bắt buộc TorchDynamo trace **toàn bộ** forward thành **một** FX graph. Nếu gặp Python branch phụ thuộc dữ liệu (data-dependent), nó **raise lỗi** thay vì tạo graph break. |
| `dynamic=False` | Khoá shape của mọi tensor input. Mỗi shape mới sinh **specialised kernel** mới (recompile). Đổi lại: TorchInductor có thể constant-fold shape, sinh kernel tile-size cố định cực nhanh. |
| `mode="max-autotune"` | Với mỗi GEMM/conv/reduction, Inductor benchmark **nhiều cấu hình tile/block/num_warps** trên GPU thật, chọn cấu hình nhanh nhất rồi cache. Lần chạy đầu chậm (~vài phút), lần sau hưởng. |

### 1.2. Vì sao `fullgraph` quan trọng — minh hoạ "graph break"

Một forward Python điển hình của Transformer block:

```python
def forward(self, x, ve, ...):
    if ve is not None:                # ← branch phụ thuộc dữ liệu
        x = x + attn(x, ve)
    else:
        x = x + attn(x, None)
    x = x + mlp(x)
    return x
```

TorchDynamo gặp `if ve is not None` sẽ:

- **Không bật** `fullgraph`: tạo **graph break** — chia forward thành 2 graph, giữa hai graph là code Python "eager". Mỗi graph phải tự materialize input/output ra HBM.
- **Bật** `fullgraph=True`: raise `Unsupported` ⇒ buộc bạn viết lại code không có branch.

**Tác hại của graph break ở cấp HBM:**

```
TRƯỚC (có graph break, 12 layer):

Layer 0  ──[graph A]──► HBM ──[Python "if"]──► HBM ──[graph B]──► HBM
                         ▲                       ▲                ▲
                         │                       │                │
                    write x_attn            read x_attn       write x_mlp
                    write x_mlp                                read for next layer
                    (4 round-trip HBM/layer × 12 = 48 round-trip)

SAU (fullgraph, 12 layer):

Layer 0..11 ──[1 graph khổng lồ]──► HBM
              tất cả intermediate ở SRAM/registers
              (≈ 1 round-trip lớn — Inductor fuse được pre-norm + attn-out + residual)
```

Trên H100, HBM bandwidth ~3 TB/s; mỗi tensor activation `(B=32, T=2048, D=1024)` BF16 = **128 MB**. Nửa graph break = đọc + ghi 256 MB = ~85 µs lãng phí **mỗi layer**. Với 12 layer × 2 chỗ break = ~2 ms/step trắng.

### 1.3. Thủ thuật của nanochat: thay branch bằng tensor "zero-padding"

Nanochat ở `v2/gpt_v2.py:42-102` thay:

```python
ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
# ...
if ve is not None:
    v = v + gate * ve
```

bằng:

```python
# Mỗi layer luôn nhận tensor ve cùng shape; layer không có VE thì ve = zeros
ve_or_zero = ve if ve is not None else torch.zeros_like(v)
v = v + gate * ve_or_zero    # khi ve_or_zero = 0 ⇒ cộng 0 ⇒ numerics y hệt
```

**Ý tưởng cấp thấp:** Compiler không cần biết `ve` "có hay không". Nó cần shape & dtype cố định. Cộng 0 là 1 GEMM "phí" rất nhỏ (~µs) so với 1 graph break (~85 µs).

### 1.4. Trực quan tensor flow trước/sau

**Trước (eager):**

```
Forward 1 step, 1 layer:

x (128 MB BF16)
   │
   ▼  ┌─ kernel launch (~6 µs)
   pre_norm  ──► x_norm (128 MB) ─── HBM write+read
   │
   ▼  ┌─ kernel launch (~6 µs)
   q_proj    ──► q (128 MB) ─── HBM write+read
   ▼  ┌─ kernel launch (~6 µs)
   k_proj    ──► k ─── HBM write+read
   ▼  ┌─ kernel launch (~6 µs)
   v_proj    ──► v ─── HBM write+read
   ▼
   FA3       ──► attn_out
   ▼
   residual  ──► x_post
   ▼
   ... (MLP cũng 4-5 kernel)

Tổng / layer: ~10 kernel launches, ~10 HBM round-trips
Tổng / forward (12 layer): ~120 kernel launches
```

**Sau (`torch.compile(fullgraph, max-autotune)`):**

```
x ──► [1 fused kernel khổng lồ chạy hết block] ──► x_out
       │
       │  pre_norm + q_proj + k_proj + v_proj + rope
       │  fused thành 1 Triton kernel (đọc x từ HBM 1 lần,
       │  giữ q/k/v trong SRAM, không ghi ra HBM)
       │
       │  → FA3 (vẫn kernel riêng vì viết bằng CUTLASS)
       │
       │  → residual + post_norm + mlp_up + relu² + mlp_down + residual
       │  fused thành 1 Triton kernel
       
Tổng / layer: ~3 kernel (pre, FA3, post-MLP)
Tổng / forward: ~36 kernel launches
Tiết kiệm: ~84 launch × 6 µs = ~500 µs/step
HBM: ~6× ít round-trip → tiết kiệm thêm ~1-3 ms/step trên model 1B
```

### 1.5. Patch cụ thể cho `llm/core/model.py`

**Bước 1**: Audit tất cả Python branch trong forward. Chạy:

```bash
TORCH_LOGS="graph_breaks,recompiles" python -c "
from llm.core.model import GPT, GPTConfig
import torch
model = GPT(GPTConfig(...)).cuda()
model = torch.compile(model, fullgraph=True, dynamic=False, mode='max-autotune')
x = torch.randint(0, 32000, (2, 2048), device='cuda')
y = torch.randint(0, 32000, (2, 2048), device='cuda')
model(x, y).backward()
"
```

Mỗi `Unsupported` log → 1 chỗ cần sửa.

**Bước 2**: Thay tất cả `if x is not None:` thành tensor unconditional. Mẫu chung:

```python
# TRƯỚC
if extra is not None:
    h = h + proj(extra)

# SAU
extra_safe = extra if extra is not None else torch.zeros_like(h)
h = h + proj(extra_safe)   # khi extra=None, proj nhận zero, cộng zero
```

Hoặc gating cleaner:

```python
# Mặc định layer không dùng feature này: gate = 0 (tham số fixed)
gate = self.feature_gate  # nn.Parameter shape (1,), init 0 cho layer-off
h = h + gate * proj(extra_safe)
```

**Bước 3**: Bọc model ở `scripts/base_train.py`:

```python
# Sau model.cuda() và DDP wrap
if args.compile:
    model_for_train = torch.compile(
        model_for_train,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune",
    )
```

**Bước 4**: Lần đầu chạy mất 2-5 phút autotune. Cache ở `~/.cache/torch/inductor`. Để tăng tốc CI, set `TORCHINDUCTOR_CACHE_DIR=/persistent/path`.

### 1.6. Bẫy thường gặp

- **Recompile cascade**: shape thay đổi giữa step (ví dụ batch cuối nhỏ hơn) ⇒ recompile. Fix: pad batch cuối hoặc dùng `dynamic=True` (mất một chút perf).
- **Hyperparam scalar**: `lr` thay đổi mỗi step ⇒ nếu pass như Python float thì recompile. Phải pass như **0-D CPU tensor** (`torch.tensor(lr)` rồi `.fill_(new_lr)`), giống nanochat đã làm trong `optim.py:182-194`.
- **FA3 kernel break**: FA3 viết CUTLASS, không phải Triton ⇒ Inductor không fuse vào được. Đây là graph break **chấp nhận được** (chỉ 1 chỗ giữa pre/post). Để fullgraph pass, wrap FA3 bằng `@torch.compiler.disable` sau đó dùng `mode="reduce-overhead"` thay `max-autotune` cho phần boundary.

---

## Phần 2 — Chunked Cross-Entropy

### 2.1. Vấn đề: tensor logits 8.6 GB

Forward cuối của LM:

```
x (B, T, D)  ──lm_head──►  logits (B, T, V_padded)
                              │
                              ▼  cast .float()  (BF16 → FP32)
                          logits_fp32 (B, T, V_padded)
                              │
                              ▼  softcap: 15·tanh(x/15)
                              ▼  cross_entropy(..., targets)
                          loss (scalar)
```

Với `B=32, T=2048, V=32768` (V padding tới 32832 cho align tensor core):

```
Kích thước logits FP32 = 32 × 2048 × 32768 × 4 bytes
                        = 8,589,934,592 bytes
                        ≈ 8.59 GB     ← 1 tensor duy nhất!
```

Trên H100 80 GB, chỉ riêng tensor này ăn **hơn 10% HBM**. Nó **bóp** microbatch:

```
HBM budget ≈ 80 GB
  - Model weights (BF16, 1B params)         : ~2 GB
  - Master weights FP32 + Adam moments      : ~12 GB
  - Activations (B=32, all layers)          : ~25 GB
  - Logits FP32                             : ~8.6 GB   ←
  - Workspace, comm buffer                  : ~5 GB
  - Reserve cho fragmentation               : ~5 GB
  ─────────────────────────────────────────
  Tổng                                       ≈ 57.6 GB / 80 GB

Muốn B=64? Activations ×2 → 50 GB → OOM.
```

Bỏ tensor 8.6 GB này ⇒ thừa chỗ cho B=48 hoặc T=4096.

### 2.2. Vì sao chia chunk hợp lệ — cấp toán

Cross-entropy có tính **separable** trên chiều token:

```
                     N
CE(logits, target) = Σ  -log softmax(logits_i)[target_i]
                    i=1
                  
                   = Σ  CE_per_token(logits_i, target_i)
                  
                   = Σ_chunks  Σ_token_in_chunk  CE_per_token
                      ↑               ↑
                      lặp ngoài      tính 1 lần / chunk
```

**Quan trọng:** softcap cũng elementwise (chỉ phụ thuộc 1 token). Cả softcap lẫn CE đều **không** cần xem toàn bộ logits cùng lúc. Do đó chia chunk **không đổi numerics** (tới mức chính xác FP32).

### 2.3. Trực quan tensor flow

**Trước (vanilla):**

```
x (B, T, D) = (32, 2048, 1024) BF16, 128 MB
   │
   ▼
   lm_head matmul ──────► logits BF16 (B, T, V_pad) = 4.3 GB
                           │
                           ▼ .float()
                          logits FP32                = 8.6 GB  ← đỉnh!
                           │
                           ▼ softcap (elementwise, đọc+ghi 8.6 GB)
                          logits_capped FP32         = 8.6 GB
                           │
                           ▼ F.cross_entropy
                          loss (scalar)

Peak HBM cho riêng loss path: ~13 GB (BF16 + FP32 + buffer)
HBM bandwidth burn: đọc/ghi 8.6 GB × 3 lần ≈ 25.8 GB traffic
Trên 3 TB/s → ~8.6 ms chỉ cho loss. (Đáng kể!)
```

**Sau (chunked, C=256):**

```
x_flat (B·T, D) = (65536, 1024)
   │
   ▼  loop i = 0, 256, 512, ..., 65280
   │
   ├──► x_chunk (256, 1024) BF16, 0.5 MB
   │     │
   │     ▼  lm_head matmul (chunk × full weight)
   │    logits_chunk BF16 (256, 32832) = 16 MB
   │     │
   │     ▼  .float() + softcap + CE (sum reduction)
   │    loss_chunk (scalar)
   │     │
   │     ▼  loss_accum += loss_chunk
   │
   └──► (chunk tiếp theo, x_chunk cũ giải phóng)

Peak HBM cho loss path: ~32 MB (1 chunk FP32)  ← giảm 268×!
HBM traffic: vẫn ~25.8 GB tổng (vẫn đọc cả vocab × cả token)
NHƯNG: 32 MB chunk vừa khít L2 cache H100 (50 MB)
       ⇒ phần lớn read trùng L2, KHÔNG round-trip HBM
       ⇒ thực tế giảm ~3-5× HBM traffic, nhanh hơn ~2-3×
```

**Sơ đồ memory layout — minh hoạ chunk vs full:**

```
HBM (80 GB)                        L2 cache (50 MB)
┌────────────────────────────┐     ┌──────────────┐
│ weights, activations, ...  │     │              │
│                            │     │              │
│ ┌──────────────────────┐   │     │              │
│ │  logits FP32 8.6 GB  │ ◄─┼─────┼─ KHÔNG fit   │  ← vanilla:
│ │  ████████████████    │   │     │   L2 thrash  │     phải đi HBM
│ └──────────────────────┘   │     │              │     mỗi lần đọc
└────────────────────────────┘     └──────────────┘

HBM                                L2 cache
┌────────────────────────────┐     ┌──────────────┐
│ weights, activations       │     │ ┌──────────┐ │
│ (thừa chỗ cho B=48)        │     │ │ chunk    │ │  ← chunked:
│                            │     │ │ 32 MB    │ │     1 lần đẩy
│   (không có logits 8 GB)   │     │ │ ████████ │ │     vào L2 →
│                            │     │ └──────────┘ │     giữ ở đó
└────────────────────────────┘     └──────────────┘     suốt softcap+CE
```

### 2.4. Patch cụ thể cho `llm/scripts/base_train.py`

**File mới `llm/core/chunked_loss.py`** (copy nguyên từ nanochat, ~30 dòng):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def chunked_cross_entropy_with_softcap(
    lm_head: nn.Linear,
    x: torch.Tensor,         # (B, T, D), BF16
    targets: torch.Tensor,   # (B, T), int64
    vocab_size: int,
    softcap: float = 15.0,
    chunk_size: int = 256,
    ignore_index: int = -1,
) -> torch.Tensor:
    B, T, D = x.shape
    x_flat = x.view(B * T, D)
    targets_flat = targets.view(B * T)

    n_valid = (targets_flat != ignore_index).sum()
    loss_accum = x.new_zeros(())
    weight = lm_head.weight  # (V_padded, D)

    for i in range(0, B * T, chunk_size):
        x_chunk = x_flat[i : i + chunk_size]                          # (C, D)
        t_chunk = targets_flat[i : i + chunk_size]                    # (C,)
        logits = F.linear(x_chunk, weight.to(dtype=x_chunk.dtype))    # (C, V_pad)
        logits = logits[..., :vocab_size]                             # (C, V)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        loss_chunk = F.cross_entropy(
            logits, t_chunk, ignore_index=ignore_index, reduction="sum",
        )
        loss_accum = loss_accum + loss_chunk

    return loss_accum / n_valid.clamp(min=1)
```

**Sửa `llm/core/model.py`** trong `GPT.forward`:

```python
# TRƯỚC
logits = self.lm_head(x)
logits = logits[..., :self.config.vocab_size].float()
logits = 15.0 * torch.tanh(logits / 15.0)
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1)

# SAU
from llm.core.chunked_loss import chunked_cross_entropy_with_softcap
loss = chunked_cross_entropy_with_softcap(
    self.lm_head, x, targets,
    vocab_size=self.config.vocab_size,
    softcap=15.0,
    chunk_size=256,
)
```

### 2.5. Tuning `chunk_size`

```
chunk_size  |  peak chunk size (FP32) |  L2 fit?  |  ghi chú
─────────────────────────────────────────────────────────────
   64       |     8 MB                |  ✓        |  quá nhỏ, GEMM kém efficient
  128       |    16 MB                |  ✓        |  
  256       |    32 MB                |  ✓        |  ← sweet spot H100 (L2=50 MB)
  512       |    64 MB                |  ✗        |  bắt đầu thrash
 1024       |   128 MB                |  ✗        |  về gần baseline
```

H100 có L2 = 50 MB, A100 có L2 = 40 MB → dùng chunk_size=256 là an toàn cho cả hai.

### 2.6. Vì sao "cộng dồn" với grad-accum?

`llm/scripts/base_train.py` hiện gọi all-reduce **mỗi microstep** (vì DDP không có `no_sync()`). Nếu chunked CE cho phép microbatch **gấp đôi**, bạn có thể giảm `grad_accum_steps` đi một nửa ⇒ **một nửa số all-reduce**:

```
Trước:                                  Sau (microbatch ×2):
B_micro = 8                              B_micro = 16
grad_accum = 16                          grad_accum = 8
Tổng B effective = 128                   Tổng B effective = 128
all-reduce / step = 16                   all-reduce / step = 8
                                         ⇒ tiết kiệm 8 × T_allreduce / step
```

Trên cluster 8×H100, T_allreduce ≈ 5-15 ms tuỳ size. Tiết kiệm ~60-100 ms/step nhờ ghép 2 thay đổi.

---

## Phần 3 — Per-block Activation Checkpointing

### 3.1. Activation chiếm bao nhiêu memory?

Khi forward, mỗi layer phải **giữ lại** mọi tensor mà backward cần dùng để tính gradient. Với 1 transformer block:

```
Tensor cần lưu cho backward (per layer, B=32, T=2048, D=1024):

  pre_norm input (x)            : 128 MB BF16
  q, k, v sau projection        : 128 MB × 3 = 384 MB
  attn output trước c_proj      : 128 MB
  post_norm input               : 128 MB
  mlp up                        : 4× hidden = 512 MB
  mlp activation (relu²)        : 512 MB
  ──────────────────────────────────
  Tổng / layer                  : ~1.78 GB

× 12 layer                      = 21.4 GB activations
+ embedding & lm_head            ≈ 24 GB
```

Tức là activations của 1 forward pass với B=32 đã ăn ~24 GB / 80 GB HBM.

### 3.2. Ý tưởng checkpoint: vứt rồi tính lại

```
KHÔNG CHECKPOINT (lưu hết):

Forward:    L0 ──► L1 ──► L2 ──► ... ──► L11 ──► loss
             │     │     │              │
             └─act─┴─act─┴─act─...──────┘─act
             tất cả lưu trong HBM (~21 GB)

Backward:   ◄─── tính grad từ L11 về L0, đọc act từ HBM
            (peak HBM: 21 GB activations + grad buffer)

CHECKPOINT (chỉ lưu ranh giới block):

Forward:    L0 ──► L1 ──► L2 ──► ... ──► L11 ──► loss
             ●     ●     ●              ●
             chỉ lưu input của mỗi block (12 × 128 MB = 1.5 GB)
             intermediate (q,k,v,mlp,...) bị FREE NGAY sau forward layer

Backward L11: 
  - đọc input của L11 (128 MB)
  - RECOMPUTE forward L11 → tái tạo intermediate trong SRAM/HBM
  - tính grad → free intermediate
  - tiếp L10...

Peak HBM: 1.5 GB (boundaries) + 1.78 GB (1 layer đang recompute) ≈ 3.3 GB
TIẾT KIỆM: 21 GB → 3.3 GB ≈ 17.7 GB
```

### 3.3. Trực quan tensor lifetime

**Không checkpoint** — activation sống suốt forward + backward:

```
Time ──►
       Forward                         Backward
  L0:  ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●━ free sau L0.bwd
  L1:  ░━●━━━━━━━━━━━━━━━━━━━━━━━━━━●━━━ free sau L1.bwd
  L2:  ░━░━●━━━━━━━━━━━━━━━━━━━━━━●━━━━━
  ...
  L11: ░━░━░━ ... ━●━━━━━━━━━━━━●━━━━━━━━

 ●  = activation alive trong HBM
 ░  = không liên quan ở đó

 Peak (giữa forward xong, backward chưa bắt đầu): tất cả 12 layer alive!
```

**Có checkpoint** — chỉ ranh giới sống dài, intermediate sống ngắn:

```
Time ──►
       Forward                                 Backward
  L0:  █━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●━ free
  L1:  ░━█━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●━━━━
  L2:  ░━░━█━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●━━━━━━━
  ...
  L11: ░━░━░━ ... ━█━━━━━━━━━━━━━━━━━━━━━━━━━━━━●━━━━━━━━━━━

  intermediate L11:                              ▄▄ recompute → bwd → free
  intermediate L10:                                 ▄▄ recompute → bwd → free
  ...
  intermediate L0:                                                  ▄▄ ...

 █  = boundary tensor (chỉ input của block, ~128 MB/block)
 ▄  = intermediate sống tạm 1 lần recompute
```

### 3.4. Chi phí: ~33% extra compute

Mỗi block phải chạy **forward 2 lần** (1 lần ở forward thật, 1 lần lúc backward để tái tạo activation). Forward chiếm ~1/3 tổng compute (backward ~2/3 vì có 2 GEMM cho gradient input + gradient weight).

```
Compute breakdown / step:

KHÔNG checkpoint:    forward (1F) + backward (2F) = 3F  ← unit "F" = compute 1 forward pass
CÓ checkpoint:       forward (1F) + recompute (1F) + backward (2F) = 4F
                     overhead = 1F / 3F = 33%
```

**Khi nào nên dùng?**

- HBM utilization > 80% và OOM khi tăng B → bật.
- HBM thoải mái, throughput cap bởi compute → **không** bật (lãng phí 33%).

### 3.5. Patch cụ thể

**Sửa `llm/core/model.py`** — wrap mỗi block:

```python
from torch.utils.checkpoint import checkpoint

class GPT(nn.Module):
    def __init__(self, config):
        ...
        self._use_activation_checkpointing = False  # mặc định off

    def forward(self, idx, targets=None, kv_cache=None):
        ...
        for i, block in enumerate(self.transformer.h):
            ...
            if self.training and self._use_activation_checkpointing and kv_cache is None:
                # use_reentrant=False: an toàn với nested autocast & torch.compile
                x = checkpoint(
                    block,
                    x, ve, cos_sin, self.window_sizes[i], kv_cache,
                    use_reentrant=False,
                )
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        ...
```

**Sửa `llm/scripts/base_train.py`** — thêm cờ:

```python
parser.add_argument("--activation-ckpt", action="store_true",
                    help="Bật activation checkpointing per block (tiết kiệm ~17 GB, chậm 33%)")

# Sau khi build model:
if args.activation_ckpt:
    model._use_activation_checkpointing = True
```

### 3.6. Bẫy: checkpoint × torch.compile × DDP

- `use_reentrant=True` (default cũ) **không** tương thích `torch.compile` — phải dùng `use_reentrant=False`.
- DDP gradient hooks: checkpoint recompute không trigger DDP all-reduce duplicate (DDP đã skip nhờ `set_static_graph()`-friendly logic của `use_reentrant=False`). Vẫn OK.
- KV cache (`kv_cache is not None`) là inference path → **không** checkpoint (không có backward).

### 3.7. Selective checkpointing (nâng cao)

Một biến thể mạnh hơn: chỉ checkpoint **attention** (chỗ tốn activation nhất do softmax), giữ MLP. Tiết kiệm ~70% memory với chỉ ~15% extra compute. Cần viết wrapper riêng — để dành đợt sau.

---

## Phần 4 — Thứ tự áp dụng & cách đo

### 4.1. Bảng quyết định

```
Bước nào trước? Phụ thuộc bottleneck hiện tại:

Nếu MFU < 35%:
  → torch.compile trước (graph break đang giết bạn)
  
Nếu OOM khi tăng B:
  → chunked CE trước (rẻ nhất, không đổi numerics)
  → nếu vẫn OOM → activation ckpt
  
Nếu wallclock cap bởi all-reduce:
  → chunked CE để mở B → giảm grad_accum
  → thêm model.no_sync() trên các microstep không phải cuối
```

### 4.2. Cách đo từng bước

```bash
# Baseline
python scripts/base_train.py --device-batch-size 8 --grad-accum-steps 16 \
  --max-iters 100 --log-interval 10
# Note: tokens/sec, MFU, peak HBM

# Sau khi thêm chunked CE
python scripts/base_train.py --device-batch-size 16 --grad-accum-steps 8 \
  --max-iters 100 --log-interval 10
# So sánh: tokens/sec phải tăng ~1.5-2×, peak HBM phải giảm ~7 GB

# Sau khi thêm torch.compile
python scripts/base_train.py --compile --device-batch-size 16 \
  --grad-accum-steps 8 --max-iters 100
# Bỏ qua step 1-5 (đang autotune), so từ step 10
# tokens/sec phải tăng thêm 20-40%, MFU lên ~50-60%

# Sau khi bật activation ckpt (chỉ test khi cần)
python scripts/base_train.py --compile --activation-ckpt \
  --device-batch-size 32 --grad-accum-steps 4 --max-iters 100
# tokens/sec có thể giảm 15-25% nhưng B effective gấp đôi
# → chỉ thắng nếu giảm grad_accum_steps đủ để bù
```

### 4.3. Sanity check numerics

Mỗi tối ưu phải **không** đổi loss curve:

```python
# Test: chạy 50 step với và không có optim, so loss
# Tolerance: chunked CE & activation ckpt phải khớp tới ~1e-4
# torch.compile có thể lệch ~1e-3 do fusion order khác
```

Nếu lệch lớn hơn → có bug (có khả năng softcap order, dtype cast, hoặc rng state).

---

## Tóm tắt một dòng mỗi phần

1. **`torch.compile(fullgraph)`**: ép toàn bộ forward thành 1 graph duy nhất → cắt kernel-launch overhead + fuse elementwise → tiết kiệm vài ms/step. Giá: phải xoá mọi Python branch phụ thuộc dữ liệu.

2. **Chunked CE**: thay vì materialize tensor logits FP32 8.6 GB, chia 256 token mỗi lần → peak 32 MB (vừa L2). Numerics y hệt. Mở thêm chỗ trống cho microbatch lớn → giảm số all-reduce.

3. **Activation checkpointing**: vứt activation intermediate, recompute lúc backward → tiết kiệm ~17 GB, đổi 33% compute. Bật chọn lọc khi memory-bound.

Tất cả cộng dồn: đẩy MFU từ ~35% lên ~55-60% trên 1 node 8×H100, đồng thời mở rộng B effective gấp 2-4× ⇒ ít step hơn cho cùng token budget ⇒ wallclock total giảm ~2×.
