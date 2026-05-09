# Hành Trình Vision Multimodal — First Principles Từ Pixel Đến Loss

> *Giải thích sâu cho senior AI research engineer. Mỗi line of code có WHY, mỗi tensor có shape, mỗi design có first-principles reasoning. Feynman technique: explain như đang dạy 12-year-old, build up to PhD level.*

---

## Phần 0 — Mental Model Cốt Lõi

### 0.1 Bài toán fundamental

Bạn có một **frozen LLM** đã được train trên text. Nó hiểu language. Bạn cũng có một **image**. Câu hỏi:

> **Làm sao để LLM "đọc" image như "đọc" text?**

### 0.2 Phép ẩn dụ Feynman: phòng dịch

Tưởng tượng LLM là một **người dịch** chỉ biết tiếng Anh. Bạn đưa cho họ một bức tranh và yêu cầu họ mô tả. Họ KHÔNG nhìn được tranh — họ chỉ đọc được TEXT.

**Giải pháp:** thuê một **người mô tả tranh** (vision encoder) chuyển tranh thành các "từ giả" mà người dịch có thể "đọc" như text thật. Người mô tả phải:
1. Nhìn vào tranh → hiểu nó
2. Chuyển hiểu biết thành các "vector embeddings" cùng dimension với LLM's text embeddings
3. Đặt các vector đó vào đúng vị trí trong sentence của LLM

→ **Vision encoder = "interpreter" giữa pixel space và LLM embedding space.**

### 0.3 The ONE big idea

```
                                                       ┌──────────────────┐
   Image (3, 384, 384)     Vision Encoder              │                  │
   ──────────────────────► (frozen SigLIP2)            │                  │
                                                       │      LLM         │
                           Vector embeddings           │   (treats them   │
                           giống text embeddings   ───►│   as "weird text │
                                                       │    tokens")      │
   Text "What is this?"                                │                  │
   ──► tokenizer ──► embeddings ───────────────────────┤                  │
                                                       └────────┬─────────┘
                                                                │
                                                                ▼
                                                     "It is a cat" (text out)
```

**Toàn bộ multimodal pipeline = mechanism để turn image thành "fake text tokens" mà LLM trunk consume được mà không cần thay đổi.**

---

## Phần 1 — Image Input: Pixel Tensor

### 1.1 Image là gì về mặt mathematical

Một bức ảnh RGB 384×384:

```python
image: torch.Tensor          # shape: (3, 384, 384)
                              # dtype: float (after preprocessing)
                              # values: ~ [-1.0, 1.0] (normalized)
```

3 channels (R, G, B), mỗi channel = 384×384 = 147,456 pixels. Total = **442,368 numbers** representing one image.

### 1.2 Tại sao 3 channels?

**First principles:** human eye có 3 loại cone cells (red, green, blue). Image format bắt chước human vision. Mỗi pixel có 3 numbers cho cường độ của 3 màu cơ bản. Bất kỳ màu nào bạn nhìn thấy = combination của (R, G, B).

→ Tensor shape `(C=3, H=384, W=384)` là conventions từ deep learning frameworks (PyTorch's "channels first").

### 1.3 Code: AutoImageProcessor

```python
# core/multimodal_data.py:_try_load_image_processor
from transformers import AutoImageProcessor
proc = AutoImageProcessor.from_pretrained("google/siglip2-so400m-patch14-384")

# When called on a PIL image:
inputs = proc(images=pil_img, return_tensors="pt")
pixel_values = inputs["pixel_values"][0]  # shape: (3, 384, 384)
```

**What AutoImageProcessor does internally:**
1. **Resize** to 384×384 (SigLIP2's expected input)
2. **CenterCrop** if input not square
3. **Normalize**: `(pixel - 0.5) / 0.5` → range becomes [-1, 1]
4. **Reorder** channels to (C, H, W)

**Tại sao normalize?** Neural networks train tốt nhất khi inputs có **zero mean, unit variance**. Raw pixel range [0, 255] làm gradient huge và training unstable. Normalize → controlled magnitudes → smooth optimization.

**First principles:** mọi neural network expects **standardized inputs**. Different ranges (e.g. [0, 1] vs [-1, 1]) yield different gradient scales — must match what the encoder was TRAINED on. SigLIP2 was trained with [-1, 1] normalization → we use the same.

---

## Phần 2 — SigLIP2 Vision Encoder: Pixels → Patches → Vectors

### 2.1 Bài toán: làm sao convert (3, 384, 384) tensor thành "vectors LLM hiểu"?

**Naive approach:** flatten image into a vector of 442K numbers, project to LLM hidden dim. Vấn đề:
- Linear projection 442K → 1536 = ~680M params (quá lớn)
- Loses spatial structure (LLM không biết pixel này gần pixel kia)
- Không reusable across different image sizes

**Smart approach (ViT):** chia image thành **patches**, treat each patch as a "token", apply self-attention.

### 2.2 Patches — Feynman analogy

Tưởng tượng bức ảnh là một bức tường gạch lớn. Thay vì describe TỪNG VIÊN GẠCH một, bạn:
1. Chia bức tường thành **squares 14×14 pixels** (mỗi square = 1 patch)
2. Describe MỖI square thành 1 vector (một "từ" trong vocabulary của tranh)
3. Có **27×27 = 729 patches** (vì 384/14 = 27.4 → 27)
4. 729 patches = 729 "image tokens" — LLM-like format!

```
Image 384×384:               Patches:
┌───┬───┬───┬─...┐          [patch_0, patch_1, ..., patch_728]
│ . │ . │ . │    │          
├───┼───┼───┼────┤          Each patch_i ∈ ℝ^1152
│ . │ . │ . │    │          (1152 = SigLIP2's hidden_dim)
├───┼───┼───┼────┤          
│ . │ . │ . │    │          
├───┴───┴───┴────┤          
│ ... 27 rows    │          
└────────────────┘          
```

### 2.3 Code: VisionTower forward

```python
# core/multimodal.py:VisionTower.forward
def forward(self, pixel_values, grid_thw):
    # pixel_values shape: (N_imgs, 3, 384, 384)
    
    # Step 1: SigLIP2 vision_model encodes patches
    if hasattr(self.siglip, "vision_model"):
        outputs = self.siglip.vision_model(pixel_values=pixel_values)
        patch_features = outputs.last_hidden_state
        # patch_features shape: (N_imgs, 729, 1152)
    
    # Step 2: Flatten across images for PatchMerger
    if patch_features.dim() == 3:
        N, P, D = patch_features.shape
        patch_features = patch_features.reshape(N * P, D)
        # shape: (N_imgs * 729, 1152)
    
    # Step 3: PatchMerger compresses + projects
    return self.merger(patch_features, grid_thw)
    # output shape: (N_imgs * 169, 1536)  [after 2x2 merge + projection]
```

### 2.4 Tensor shape walkthrough cho 1 image

```
INPUT:  pixel_values  (1, 3, 384, 384)
        
        ↓ SigLIP2 patch_embed (Conv2d kernel=14, stride=14)
        
intermediate: (1, 1152, 27, 27)   [conv output]
        
        ↓ permute + flatten
        
patch_tokens: (1, 729, 1152)      [729 patches, 1152-dim each]
        
        ↓ ViT blocks (transformer layers — self-attention among patches)
        ↓ FROZEN — gradients don't flow back through here
        
patch_features: (1, 729, 1152)    [contextualized patch embeddings]
        
        ↓ flatten across images
        
flat:    (729, 1152)               [ready for PatchMerger]
```

### 2.5 Tại sao SigLIP2 cụ thể (vs CLIP, DINOv2, etc.)?

**SigLIP2** (Sigmoid Loss for Image-Language Pretraining):
- Pretrained on **4B+ image-text pairs**
- Uses **sigmoid loss** (vs CLIP's softmax) — better data efficiency
- **Higher quality** image embeddings tại scale chúng ta cần
- HF official support, AutoImageProcessor included

**SO400M variant:** ~543M params, trained for higher resolution. Frontier multimodal models (Qwen3.5-VL) use this.

**First principles tại sao FROZEN:**
1. **Compute economy**: 543M params = 2.5× wall-clock if trainable
2. **Quality preservation**: SigLIP2 was trained on 1024 H100 weeks; we have 0.001% of that → re-training degrades it
3. **Confound elimination**: vision representation invariant across cells → scaling-law fits clean

→ Chúng ta TREAT SigLIP2 như một **fixed feature extractor** (giống dùng a calculator). Don't try to "improve" it; just consume its outputs.

### 2.6 Tại sao Conv2d + ViT thay vì pure CNN?

**CNN** is great cho vision but produces **fixed-size feature maps**. Locality bias makes it strong for low-level features (edges, textures).

**ViT** treats image như sequence of tokens → enables **global attention** (any patch can attend to any other patch from layer 1). Superior for high-level reasoning.

**Hybrid (what SigLIP2 does):**
- Small Conv2d at the start ("patch_embed") to convert 14×14 pixels → 1 token. Gives locality bias for free.
- Then 27 Transformer blocks for global reasoning.

→ Best of both worlds. Standard frontier choice 2024-2026.

### 2.7 Self-attention trong ViT (Feynman)

Tưởng tượng 729 patches là 729 sinh viên trong một lớp học. Mỗi sinh viên hỏi:
- *"Để mô tả bản thân tốt hơn, tôi nên CHÚ Ý đến những sinh viên nào khác?"*

**Self-attention** computes:
1. Mỗi patch produce 3 vectors: `Query` (câu hỏi), `Key` (chữ ký), `Value` (thông tin)
2. Mỗi patch's `Q` được match với MỌI `K` của 729 patches → similarity scores
3. Softmax → attention weights (sum to 1)
4. Weighted sum của 729 `V` → updated representation

→ Mỗi patch's output = **mixture of all patches' info, weighted by relevance**.

Tại sao powerful? Ở high layer, patch-of-cat-eye có thể attend strongly to patch-of-cat-tail and patch-of-cat-paw → understand "cat as a whole" rather than just local pixels.

---

## Phần 3 — PatchMerger: Compression + Projection

### 3.1 Bài toán: 729 patches là quá nhiều

LLM trunk có sequence_len = 4096. Nếu mỗi image = 729 vision tokens, chỉ fit ~5 images per row. Plus, 729 tokens cho 1 image = vision dominates the loss budget too much.

**Solution:** **compress** spatial info bằng cách merge 2×2 neighboring patches → 1 token.

```
Before merge:  27×27 = 729 patches
After 2×2:     13×13 = 169 tokens (4× reduction)
```

(With our crop from 27→26: 13×13 = 169 tokens. 26 even, divides by 2.)

### 3.2 Code chi tiết với tensor shapes

```python
# core/multimodal.py:PatchMerger.forward
def forward(self, x, grid_thw):
    # x shape: (729, 1152)  — flattened patches from VisionTower
    # grid_thw: [[1, 27, 27]]  — (T=1 timeframe, H=27 height, W=27 width)
    
    # Step 1: RMSNorm on the feature dimension
    x = F.rms_norm(x, (self.vision_embed_dim,), eps=self.rms_norm_eps)
    # shape unchanged: (729, 1152)
    
    merge = self.spatial_merge_size  # = 2
    outs = []
    offset = 0
    
    for (T, H, W) in grid_thw.tolist():
        n = T * H * W  # = 1 * 27 * 27 = 729
        chunk = x[offset:offset+n].reshape(T, H, W, self.vision_embed_dim)
        # chunk shape: (1, 27, 27, 1152)
        
        # CROP odd grid (27 → 26)
        H_eff = (H // merge) * merge   # 13 * 2 = 26
        W_eff = (W // merge) * merge   # 26
        if H_eff != H or W_eff != W:
            chunk = chunk[:, :H_eff, :W_eff, :].contiguous()
            # chunk shape: (1, 26, 26, 1152)
        
        # Step 2: 2x2 spatial merge
        # Reshape to expose merge dimensions
        chunk = chunk.reshape(T, H_eff // merge, merge, W_eff // merge, merge, self.vision_embed_dim)
        # shape: (1, 13, 2, 13, 2, 1152)
        
        # Permute to bring merge dims adjacent to feature dim
        chunk = chunk.permute(0, 1, 3, 2, 4, 5).contiguous()
        # shape: (1, 13, 13, 2, 2, 1152)
        
        # Flatten merge dims into feature dim
        chunk = chunk.reshape(T * (H_eff // merge) * (W_eff // merge), self.grouped_dim)
        # shape: (169, 4608)  where 4608 = 1152 * 4
        
        outs.append(chunk)
        offset += n
    
    x = torch.cat(outs, dim=0)
    # shape: (169, 4608)
    
    # Step 3: 2-layer MLP project to LLM hidden
    return self.fc2(F.gelu(self.fc1(x)))
    # fc1: 4608 → 4608
    # gelu: nonlinearity
    # fc2: 4608 → 1536  (= LLM trunk hidden dim)
    # output shape: (169, 1536)
```

### 3.3 Feynman: tại sao 2×2 merge works

Tưởng tượng bạn có một bức ảnh đẹp với 4 patches gần nhau:
```
[mắt trái]  [trán phải]
[má trái]   [má phải]
```

Mỗi patch represents 1/729 của bức ảnh. Riêng lẻ chúng không có nhiều ý nghĩa. Nhưng MERGED (concat features), 4 patches together describe **upper-left quadrant of face**.

→ **2×2 merge = encoding spatial neighborhoods explicitly.** Trade off: mất resolution, gain compactness + local context.

### 3.4 Tại sao 2-layer MLP (không 1 layer)?

**1 layer** (just `Linear(4608, 1536)`):
- Pure linear transformation
- Can only do **rotations + scaling** in feature space
- Limited expressivity

**2 layer with GELU nonlinearity**:
- `Linear → GELU → Linear`
- Can express **arbitrary nonlinear functions** (universal approximator)
- Critical for bridging two pretrained spaces (SigLIP space ≠ LLM space)

**Math intuition:** SigLIP's 1152-dim space encodes "vision concepts." LLM's 1536-dim space encodes "language concepts." A linear map can't translate between them — needs nonlinearity.

### 3.5 Tại sao GELU (không ReLU)?

- **ReLU**: `max(0, x)` — sharp cutoff at 0
- **GELU**: `x * Φ(x)` where Φ is Gaussian CDF — smooth around 0

**GELU**:
- Smooth everywhere → smooth gradients → easier optimization
- Standard for transformers (BERT, GPT, ViT all use GELU)
- ~2-3% better validation loss vs ReLU at scale

**First principles**: smooth nonlinearities → backpropagation passes more useful gradient information. Sharp nonlinearities (ReLU) lose info at the boundary.

---

## Phần 4 — Scatter: The Early Fusion Mechanic

### 4.1 Bài toán: làm sao "trộn" vision tokens vào text sequence?

Có **3 design patterns** trong literature:

**Pattern A — Concat (Flamingo, IDEFICS):** [ALL_VISION_TOKENS] + [TEXT_TOKENS]. Vision và text are separate "blocks."
- ❌ Loses interleaving (image inside text doesn't work)
- ❌ Position semantics conflated

**Pattern B — Cross-Attention (Flamingo):** Text trunk has dedicated cross-attention layers that attend to vision features separately.
- ❌ Adds parameters
- ❌ Different compute path → can't reuse text-only weights cleanly

**Pattern C — Scatter at Placeholder (Qwen3.5-VL, ours):** Text contains placeholder tokens `<|image_pad|>`; replace embeddings at those positions with vision features. **Vision tokens flow through THE SAME trunk as text.**
- ✅ Simple, no architectural changes to trunk
- ✅ Perfect interleaving (image anywhere in sentence)
- ✅ Inherits text-only HPs

→ **Pattern C wins** because of architectural simplicity. This is "early fusion": vision enters the trunk at layer 0, treated identically to text.

### 4.2 Feynman analogy: filling blanks

Tưởng tượng bạn có một sentence với BLANKS:
```
"What is in this _ _ _ _ image?"
```

Bốn `_` là placeholders. Mỗi vision token = 1 word fills one blank. Sau khi fill:

```
"What is in this [vec0] [vec1] [vec2] [vec3] image?"
```

Nhưng `[vec0]` không phải là 1 từ tiếng Anh — nó là 1 **vector embedding** trong LLM space (1536-dim). LLM treats nó như "weird word" mà nó chưa thấy bao giờ.

→ **Scatter = filling pre-defined blanks in a sentence with vision-derived embeddings.**

### 4.3 Code: scatter_vision_features

```python
# core/multimodal.py:scatter_vision_features
def scatter_vision_features(inputs_embeds, vision_features, image_pad_mask):
    # inputs_embeds shape: (B=1, S=12, D=1536)
    #   (B batch size, S sequence length, D LLM hidden dim)
    # vision_features shape: (4, 1536)  — 4 vision tokens for 1 image
    # image_pad_mask shape: (B=1, S=12) bool
    #   True at positions where vision should go
    
    # Sanity: count must match
    n_pad = int(image_pad_mask.sum().item())  # = 4
    assert n_pad == vision_features.shape[0], "dataloader misaligned"
    
    # Clone to avoid in-place mutation (autograd safety)
    out = inputs_embeds.clone()
    
    # The MAGIC line: PyTorch fancy indexing
    out[image_pad_mask] = vision_features.to(inputs_embeds.dtype)
    # This replaces D-dim vectors at True positions
    # Equivalent to:
    #   for i, position in enumerate(true_positions):
    #       out[batch_idx, position, :] = vision_features[i]
    
    return out  # shape: (B=1, S=12, D=1536)
```

### 4.4 Tensor walkthrough cho concrete example

Giả sử `input_ids = [1, 2, IMG_PAD, IMG_PAD, IMG_PAD, IMG_PAD, 5, 6, 7, 8, 9, 10]` (12 tokens, image at positions 2-5).

```
BEFORE scatter:
  inputs_embeds shape (1, 12, 1536)
  
  position:  0       1       2       3       4       5       6       7       8       9       10      11
  source:    text    text    img_pad img_pad img_pad img_pad text    text    text    text    text    text
  embedding: text_e  text_e  ZEROS   ZEROS   ZEROS   ZEROS   text_e  text_e  text_e  text_e  text_e  text_e
            (or random init at <image_pad> position)

VISION ENCODE:
  pixel_values (1, 3, 384, 384)
  → SigLIP2 → (1, 729, 1152)
  → flatten → (729, 1152)
  → PatchMerger → (4, 1536)  [merged + projected]

AFTER scatter:
  position:  0       1       2       3       4       5       6       7       8       9       10      11
  source:    text    text    VISION  VISION  VISION  VISION  text    text    text    text    text    text
  embedding: text_e  text_e  vis_0   vis_1   vis_2   vis_3   text_e  text_e  text_e  text_e  text_e  text_e
                            (1536d  1536d   1536d   1536d)
                            → flow into MoE trunk EXACTLY like text embeddings
```

### 4.5 Tại sao `out = inputs_embeds.clone()`?

```python
out = inputs_embeds.clone()  # WHY?
out[image_pad_mask] = vision_features
```

**First principles**: `inputs_embeds` came from `self.transformer.wte(idx)` — a embedding lookup. The result might be a **view** of the embedding table, not a fresh tensor. **In-place mutation on a view** would corrupt the embedding table!

`.clone()` creates an independent copy with its own storage. Now `out[image_pad_mask] = ...` only mutates the clone. Embedding table stays clean.

This is a SUBTLE bug if you forget. PyTorch will silently let you do it and your embeddings get corrupted.

### 4.6 Tại sao PYTORCH FANCY INDEXING `out[image_pad_mask]` works?

```python
out[image_pad_mask] = vision_features
```

`image_pad_mask` shape: `(1, 12)` boolean. `out` shape: `(1, 12, 1536)`.

PyTorch interprets this as:
> "Find ALL positions in `out` where `image_pad_mask` is True (extending mask to cover the trailing 1536-dim by broadcasting). Replace those positions with consecutive entries from RHS."

The RHS `vision_features` has shape `(4, 1536)` — exactly 4 rows of 1536-dim vectors, matching the 4 True positions.

→ Single-line replacement of 4 vectors at correct positions. Pure tensor magic.

---

## Phần 5 — 3D Interleaved-MRoPE: Position Encoding

### 5.1 Vấn đề: Self-attention is permutation invariant

Self-attention computes:
```
attention(Q, K, V) = softmax(QK^T / √d) @ V
```

Nếu bạn shuffle positions của Q và K, output cũng shuffle nhưng VALUES không change. Model không biết position 0 khác position 1.

**Hệ quả:** without position info, "the cat sat" and "sat cat the" produce identical hidden states. Useless for language!

### 5.2 Solution chuẩn: Position embeddings

Thêm 1 vector cho mỗi position:
- Position 0 → vec_0
- Position 1 → vec_1
- ...

Add (or concat) vào token embeddings. Model học vec_0 ≠ vec_1 → distinguishes positions.

**Two flavors:**
- **Learned positional embeddings** (BERT, GPT-2): learn vectors as parameters
- **Fixed sinusoidal/rotary** (Transformer original, LLaMA, GPT-3+): formula-based

### 5.3 Tại sao RoPE (Rotary Position Embedding) thắng learned

**RoPE** rotates Q and K vectors by angle proportional to position:
```
Q_at_position_i = rotate(Q, angle = i × θ)
K_at_position_j = rotate(K, angle = j × θ)
```

Then `Q_at_i · K_at_j` only depends on `(i - j)` (the RELATIVE distance) — not absolute positions. Beautiful property!

**Why this matters:**
- Length extrapolation: model trained on length 2048 can sometimes work at 4096 (relative distances exist at any length)
- No extra parameters
- Translation invariance built in

### 5.4 1D RoPE — Feynman analogy

Tưởng tượng mỗi vector là một mũi tên trên đồng hồ. Position 0 → mũi tên ở 12h. Position 1 → quay 30°. Position 2 → 60°. Position 12 → quay 360° = về chỗ cũ.

Khi bạn dot product `Q@K`:
- Q ở vị trí 5 (đã quay 150°), K ở vị trí 5 → cùng góc → max dot product
- Q ở 5, K ở 7 → góc khác → smaller dot product
- Q ở 5, K ở 5 + 12 → cùng góc again (modular) → max dot product

→ **Dot product encodes relative angle = relative position.**

### 5.5 Tại sao 3D MRoPE cho multimodal?

Text tokens là 1D (sequential). Image patches là 2D (spatial). Video patches là 3D (time + space).

**Naive: dùng 1D RoPE cho mọi thứ.** Vision patches get sequential positions [0, 1, 2, ..., 728]. Loses spatial structure (patch in row 0 col 5 indistinguishable from row 5 col 0).

**Smart: 3D RoPE.** Mỗi token có 3 position numbers `(t, h, w)` for (time, height, width):
- Text token tại sequence pos 5: `(5, 0, 0)` — only t-axis active
- Vision patch in row 2 col 3 of image at time 0: `(0, 2, 3)` — h, w encode spatial location
- Video frame 5, patch row 2 col 3: `(5, 2, 3)` — full 3D

→ Attention can natively distinguish spatial neighbors from temporally close patches.

### 5.6 "Interleaved" — Feynman explanation

Original 3D MRoPE (Qwen2-VL) used **chunked**: split head_dim into 3 contiguous segments, one per axis.
- First 1/3 dimensions encode t
- Middle 1/3 encode h
- Last 1/3 encode w

**Problem**: t-axis only has LOW frequency RoPE channels, w-axis only HIGH frequency. Asymmetric — hurts long video.

**Interleaved (Qwen3-VL+, ours)**: assign each frequency channel to an axis in **round-robin**:
- Channel 0 → t
- Channel 1 → h
- Channel 2 → w
- Channel 3 → t
- Channel 4 → h
- ...

Each axis sees BOTH low and high frequencies → balanced representation.

### 5.7 Code: build_3d_mrope_for_4d_apply

```python
# core/multimodal.py:build_3d_mrope_for_4d_apply
def build_3d_mrope_for_4d_apply(position_ids, head_dim, theta=10000.0):
    """
    position_ids shape: (3, B, T)  — 3 axes [t, h, w]
    Returns cos, sin each (B, T, 1, head_dim/2)  bfloat16
    """
    # Step 1: standard RoPE inverse frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                                              device=position_ids.device,
                                              dtype=torch.float32) / head_dim))
    # shape: (head_dim/2,)
    # For head_dim=64: 32 frequencies, ranging from 1.0 down to 1/10000^(62/64) ≈ 1.6e-4
    
    # Step 2: round-robin axis assignment
    num_freqs = inv_freq.shape[0]                                  # 32
    axis_for_freq = torch.arange(num_freqs) % 3                     # [0,1,2,0,1,2,...]
    # axis_for_freq[0] = 0 (t), [1] = 1 (h), [2] = 2 (w), [3] = 0 (t), ...
    
    # Step 3: gather position per frequency
    pos_per_freq = position_ids[axis_for_freq]    # shape: (32, B, T)
    # For each frequency, look up the right axis position
    # If freq i is assigned to axis 0 (t), pos_per_freq[i, b, t] = position_ids[0, b, t]
    
    # Step 4: permute and apply frequencies
    pos_per_freq = pos_per_freq.permute(1, 2, 0).to(inv_freq.dtype)   # (B, T, 32)
    angles = pos_per_freq * inv_freq                                   # (B, T, 32) broadcast
    
    # Step 5: cos and sin in Karpathy's 4D layout
    cos = angles.cos().to(torch.bfloat16).unsqueeze(2)   # (B, T, 1, 32)
    sin = angles.sin().to(torch.bfloat16).unsqueeze(2)
    
    return cos, sin
```

### 5.8 Tensor walkthrough cho example

Giả sử `head_dim=64`, batch B=1, sequence T=12, image at positions 2-5 with grid (1, 2, 2):

```
input_ids:        [1, 2, IMG, IMG, IMG, IMG, 5, 6, 7, 8, 9, 10]
position[0] (t):  [0, 1, 2,   2,   2,   2,   3, 4, 5, 6, 7, 8 ]
position[1] (h):  [0, 0, 0,   0,   1,   1,   0, 0, 0, 0, 0, 0 ]
position[2] (w):  [0, 0, 0,   1,   0,   1,   0, 0, 0, 0, 0, 0 ]

(For text tokens: only t-axis used, h=w=0)
(For 4 vision tokens at t=2: walk row-major over 2x2 grid → (h,w) = (0,0), (0,1), (1,0), (1,1))

inv_freq shape: (32,) for head_dim=64
axis_for_freq:  [0,1,2, 0,1,2, 0,1,2, ..., 0,1] (round-robin, len 32)

pos_per_freq[0, 0, :]: positions on t-axis at all timesteps for batch 0
                     = [0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8]
pos_per_freq[1, 0, :]: positions on h-axis
                     = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
pos_per_freq[2, 0, :]: positions on w-axis
                     = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
pos_per_freq[3, 0, :]: positions on t-axis again (round-robin)
                     = [0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8]
... etc

After permute: (B=1, T=12, 32)
After * inv_freq: angles (1, 12, 32)
cos: (1, 12, 1, 32)  bfloat16
sin: (1, 12, 1, 32)  bfloat16
```

→ Mỗi token có 1 cos vector và 1 sin vector capturing its position along all 3 axes.

---

## Phần 6 — Apply Rotary Embedding

### 6.1 Code: apply_rotary_emb từ core/model.py

```python
# core/model.py:apply_rotary_emb
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # (B, T, H, D) — multihead attention layout
    d = x.shape[3] // 2                       # = head_dim/2 = 32
    x1, x2 = x[..., :d], x[..., d:]           # split last dim into halves
    y1 = x1 * cos + x2 * sin                   # rotation in half-pair convention
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)              # combine back to (B, T, H, D)
```

### 6.2 Why split into halves? (math first principles)

A 2D rotation by angle θ:
```
[x']   [cos θ   -sin θ] [x]
[y'] = [sin θ    cos θ] [y]
```

To do RoPE on a `head_dim`-D vector, treat it as `head_dim/2` PAIRS of 2D vectors. Rotate each pair by `θ_i = position × inv_freq[i]`.

**Layout convention 1 (Karpathy):**
```
x = [x_0, x_1, ..., x_{d-1} | x_d, x_{d+1}, ..., x_{2d-1}]
     ←─── first half (32) ───→  ←──── second half (32) ───→
     
Pair i: (x_i, x_{i+d}) — i-th pair takes one element from each half
```

**Layout convention 2 (HuggingFace):**
```
x = [x_0, x_1 | x_2, x_3 | x_4, x_5 | ...]
Pair i: (x_{2i}, x_{2i+1}) — adjacent elements
```

Both are mathematically equivalent — just different memory layouts. Karpathy uses convention 1 (faster on GPU due to contiguous memory access in halves).

### 6.3 Why `apply_rotary_emb` does what it does

For pair `(x_i, x_{i+d})`, rotation by angle `θ`:
```
y_i      = x_i × cos(θ) - x_{i+d} × sin(θ)
y_{i+d}  = x_i × sin(θ) + x_{i+d} × cos(θ)
```

Karpathy's code:
```python
y1 = x1 * cos + x2 * sin       # first half
y2 = -x1 * sin + x2 * cos       # second half
```

Wait — sign on first half `+x2*sin` looks like it's NOT standard rotation (should be `-x2*sin`). Let me trace:

Karpathy treats `x1 = x[..., :d]` as "x" and `x2 = x[..., d:]` as "y" of each pair. Then:
- Standard rotation: `x' = x cos - y sin`, `y' = x sin + y cos`
- Karpathy: `y1 = x1 cos + x2 sin`, `y2 = -x1 sin + x2 cos`

**Different sign convention!** This is a "rotation by -θ" or equivalently "rotation in the OTHER direction." Mathematically still a valid rotation; both Q and K rotated by same convention → relative position works the same.

### 6.4 The KEY property: dot product relative

Why is RoPE special? Because:
```
rotated_Q_at_i · rotated_K_at_j = original_Q · rotation_matrix(j - i) · original_K
```

The dot product ONLY depends on `(j - i)`, the **relative position**.

→ Attention `softmax(Q · K^T / √d)` thereby encodes relative distance, which is what language modeling actually needs.

### 6.5 Tại sao chỉ apply lên Q và K (không V)?

```python
# core/model.py:CausalSelfAttention.forward
q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
# v is NOT rotated
```

**First principles**: position info should affect WHO ATTENDS TO WHOM (the attention pattern), not WHAT INFO is being attended to (the value being mixed).

- `Q · K^T` determines **attention weights** → needs position
- `attention_weights @ V` mixes information → doesn't need position

If you rotate V, you'd corrupt the actual information being aggregated.

---

## Phần 7 — MoE FFN trong trunk

### 7.1 Bài toán: FFN dense quá nặng

Standard transformer FFN (in nanochat dense):
```python
# Per layer: 8d² FLOPs per token (8 = 2 × 4 from up/down projection × hidden expansion)
hidden = up_proj(x)   # d → 4d
hidden = relu(hidden).square()  # nanochat's ReLU²
out = down_proj(hidden)  # 4d → d
```

For d=1536, 1 FFN layer = ~9.4M params, ~19M FLOPs per token. Heavy.

**MoE trick:** thay 1 dense FFN bằng N **smaller experts** + 1 router. Each token only goes to top-K experts.

### 7.2 Iso-FLOP MoE design

```python
# core/moe.py: nanochat MoE
expert_hidden_dim = round(4 * dim / (top_k + num_shared) / 128) * 128
```

**Math:** 
- Dense: 1 expert with hidden = 4d. Per-token FLOPs = 8d²
- MoE: K active experts each with hidden = 4d/K. Per-token FLOPs = K × 2d × (4d/K) = 8d² (SAME!)

→ **Per-token compute identical**, but **total params** = K × 4d/K × N_experts × d = 4d² × N_experts.

For G=2 (K=2 active, N=8 experts, +1 shared):
- Per-token compute: same as dense
- Total params: ~9× dense (more capacity, same speed per token)

**Feynman:** like a hospital with **specialists** instead of 1 general practitioner. Each patient (token) only sees the relevant specialist (expert). Each visit costs the same, but the hospital has MORE collective knowledge.

### 7.3 Router (top-K gating)

```python
# Conceptual MoE forward
router_logits = router_gate(x)   # (B, T, num_experts)
top_k_indices, top_k_scores = top_k(router_logits, k=2)
# Each token now has 2 expert IDs and 2 weights

# Dispatch tokens to experts
for each expert_id:
    relevant_tokens = gather_tokens_assigned_to(expert_id)
    expert_output = expert[expert_id](relevant_tokens)
    
# Combine outputs by top_k_scores
combined_output = weighted_sum(expert_outputs, top_k_scores)
```

**The hard part:** efficient implementation. PyTorch's `torch._grouped_mm` does dispatch in single kernel per layer. Karpathy implemented this in nanochat MoE branch (we inherit verbatim per our spec).

### 7.4 Why MoE for multimodal specifically?

**Hypothesis (DeepSeek-V3, Qwen3.5):** different experts may **specialize**:
- Some experts → text-heavy patterns (grammar, syntax)
- Some experts → vision-heavy patterns (spatial reasoning, color)

Router learns to send vision tokens to "vision experts" and text tokens to "text experts." More efficient parameter use.

**Reality:** specialization is moderate (Shukor 2025), not perfect. Still, MoE adds capacity per FLOP — that's the main win.

---

## Phần 8 — Loss: Text-Only Target, Vision as Context

### 8.1 Critical design: vision tokens have NO LOSS TARGET

```python
# core/dataloader.py:synthetic_multimodal_loader
new_targets = targets.clone()
new_targets[image_pad_mask.to(targets.device)] = -1  # -1 = ignore_index
```

Ở vision positions: target = -1 → CE loss skips these positions.

### 8.2 Tại sao vision tokens không có loss target?

3 first-principles reasons:

**Reason 1 — Vocab mismatch:**
- LLM unembed projects to text vocab (32K tokens)
- Vision "tokens" are continuous ViT features, NOT discrete vocab entries
- Predicting next vision token would require either:
  - Discretizing vision (back to VQ tokenizer = different design we rejected)
  - Adding vision unembedding head (extra params, scope creep)

**Reason 2 — Matches downstream task:**
- Real multimodal use cases output TEXT (captioning, VQA, visual reasoning)
- Pretraining objective should match downstream usage
- "Predict next vision token" is not a useful skill for downstream

**Reason 3 — Information theory:**
- Vision tokens enter the loss INDIRECTLY via attention
- Text tokens at position N attend to vision tokens at positions 2..N-1
- Loss surface still rewards "good vision representations" through downstream text prediction quality
- Just no DIRECT supervision on vision positions

### 8.3 Loss computation walkthrough

```python
# core/multimodal.py:per_modality_loss_decomposition
def per_modality_loss_decomposition(logits, targets, modality_mask, ignore_index=-1):
    # logits shape: (B, S, vocab_size)
    # targets shape: (B, S) with -1 at vision positions
    # modality_mask: (B, S) — 1 = vision-context, 0 = text-only-context
    
    B, S, V = logits.shape
    
    # Standard cross-entropy per position
    loss_full = F.cross_entropy(
        logits.reshape(B * S, V),
        targets.reshape(B * S),
        ignore_index=ignore_index,    # vision positions get 0 loss contribution
        reduction="none",
    ).reshape(B, S)
    
    # Mask out ignored positions
    valid = targets != ignore_index
    text_mask = (modality_mask == 0) & valid     # text tokens with text-only context
    vision_mask = (modality_mask == 1) & valid    # text tokens with vision context
    
    loss = loss_full[valid].mean()
    loss_text = loss_full[text_mask].mean() if text_mask.any() else zero
    loss_vision = loss_full[vision_mask].mean() if vision_mask.any() else zero
    
    return {
        "loss": loss,
        "loss_text": loss_text,
        "loss_vision": loss_vision,
        "n_text": text_mask.sum(),
        "n_vision": vision_mask.sum(),
    }
```

### 8.4 modality_mask semantic — KEY subtlety (post Track B'')

`modality_mask` is NOT "is this token vision?" — it's **"is vision context in this token's recent attention window?"**

```python
# core/dataloader.py: synthetic_multimodal_loader
modality_mask = torch.zeros(B, S, dtype=torch.long)
for b in range(B):
    for s in range(S):
        start = max(0, s - vision_context_window)  # window = 32
        if image_pad_mask[b, start : s + 1].any():
            modality_mask[b, s] = 1   # mark as vision-context
```

**Why this semantic?**

Vision positions (image_pad_mask=True) get target=-1, contributing 0 to loss. So splitting loss "at vision positions" gives `loss_vision = 0` always (useless).

Instead, modality_mask asks: *"Of the TEXT tokens (which DO have valid loss), which ones had vision context?"* This split is meaningful:
- `loss_text` = mean CE on text tokens with text-only context
- `loss_vision` = mean CE on text tokens that had vision context recently

→ Tells us: is the model getting BETTER at predicting text after seeing image vs text after seeing only text?

### 8.5 Final loss formula

```
total_loss = (loss_text × n_text + loss_vision × n_vision) / (n_text + n_vision)
```

This is what gets backpropagated. Equivalent to standard CE but with the per-modality breakdown for logging.

---

## Phần 9 — Training Discipline: Why Frozen ViT + Frozen Projector

### 9.1 The closure principle (multimodal_spec.md §2.5.8)

**5 design decisions form a closed system** where each enables the next. Removing one breaks the chain:

```
Frozen ViT  ─────►  Bounds gradient noise
                    │
                    ▼
Frozen projector  ─►  Eliminates moving-target confound
after warmup        │
                    ▼
Text-only loss  ──►  No vision-position gradient flow
                    │
                    ▼
Joint training  ──►  Optimizer sees stable text-token-grad
from step 0         distribution from t=0
                    │
                    ▼
Inherit nanochat HPs  ◄── Empirically falsified by Phase 0.A
                    │
                    ▼
$300 budget         ◄── Affordable BECAUSE of all above
measurement
```

### 9.2 Why frozen ViT (3 first-principles arguments)

**Compute economy:**
- SigLIP2 = 543M params
- Trainable: 2.5× per-step wall-clock (forward + backward + optimizer state)
- At $300 budget, that cuts compute by 60% → unaffordable

**Quality preservation:**
- SigLIP2 was trained on **4B image-text pairs at 1024-H100-week scale**
- We have 0.001% of that compute
- Re-training degrades it (small batches, insufficient data diversity)
- We'd make it WORSE

**Confound elimination:**
- Frozen ViT → vision representation invariant across all 9 cells
- Any G\* effect attributable to MoE trunk, not "ViT alignment shifted"
- Critical for clean scaling-law fits

### 9.3 Why projector trainable then frozen (Plan §3 confound)

**Why initially trainable:** at step 0, LLM input embeddings are RANDOM. PatchMerger must learn to map ViT space → LLM space. No prior alignment exists.

**Why freeze after warmup:** if projector keeps adapting throughout training:
- LLM trunk learns to use vision features
- Projector learns to give LLM what it likes
- **Two moving targets** → "MoE capability gain" conflated with "projector adapted"
- Unattributable

Freezing after 5% warmup:
- Projector establishes "good enough" alignment in initial steps
- Then LLM trunk's job is clear (use FIXED features)
- **One moving target** = clean attribution

### 9.4 Code: freeze mechanisms

```python
# core/multimodal.py:VisionTower
class VisionTower(nn.Module):
    def __init__(self, ..., freeze_merger=True):
        # Load SigLIP2
        self.siglip = AutoModel.from_pretrained(siglip_model_id)
        self.freeze_siglip()        # ALWAYS freeze SigLIP
        
        self.merger = PatchMerger(...)
        if freeze_merger:
            self.freeze_merger_now()   # freeze projector immediately

    @torch.no_grad()
    def freeze_siglip(self):
        for p in self.siglip.parameters():
            p.requires_grad = False
        self.siglip.eval()             # disable dropout, BN running stats
    
    def freeze_merger_now(self):
        for p in self.merger.parameters():
            p.requires_grad = False

    def forward(self, pixel_values, grid_thw):
        # ... use frozen SigLIP
        # NOTE: in production, wrap SigLIP forward in torch.no_grad()
        # to skip activation memory (SigLIP has no gradients)
```

### 9.5 Tại sao `requires_grad=False` AND `.eval()`?

**`requires_grad=False`**: prevents gradient computation. Saves backward memory.

**`.eval()`**: sets BatchNorm/Dropout to inference mode. Without this:
- BatchNorm running stats would update with each forward pass → contaminates the model
- Dropout would still randomly zero features → noise during training

→ Both are needed for true frozen-encoder semantics.

---

## Phần 10 — Putting It All Together: Full Forward Pass

### 10.1 Concrete example trace

Setup:
- Sentence: `"What is in this image?"`
- 1 image (PIL, 800×600)
- mix_ratio: image goes between word 4 and word 5

### 10.2 Step-by-step tensor walkthrough

```
═══════════════════════════════════════════════════════════════════
STEP 1: Tokenization + image_pad insertion
═══════════════════════════════════════════════════════════════════

Text tokens:  "What is in this <IMG_RUN> image"
              [101,  102, 103, 104, IMG_PAD × 169, 105]
              (1 image = 169 merged tokens at SigLIP2 26x26 / 2x2)

input_ids shape: (1, 173)
image_pad_mask:  (1, 173) bool, True at positions 4..172
modality_mask:   (1, 173) long, 1 at positions where vision context recent

═══════════════════════════════════════════════════════════════════
STEP 2: Image preprocessing
═══════════════════════════════════════════════════════════════════

pil_img: PIL Image (800, 600)
    │
    ▼ AutoImageProcessor: resize to 384×384, CenterCrop, normalize
pixel_values: (1, 3, 384, 384)  float32  [-1, 1] range

═══════════════════════════════════════════════════════════════════
STEP 3: VisionTower forward (FROZEN)
═══════════════════════════════════════════════════════════════════

pixel_values: (1, 3, 384, 384)
    │
    ▼ SigLIP2.vision_model (27 transformer layers, 543M params, NO GRADIENT)
    │
    ▼ patch_embed: Conv2d(kernel=14, stride=14)
    │   intermediate: (1, 1152, 27, 27)
    │
    ▼ flatten + permute
    │
patches: (1, 729, 1152)
    │
    ▼ 27 ViT blocks (self-attention among 729 patches)
    │
patch_features: (1, 729, 1152)
    │
    ▼ Flatten across batch
    │
flat: (729, 1152)
    │
    ▼ PatchMerger
    │   1. RMSNorm: (729, 1152)
    │   2. Reshape to (1, 27, 27, 1152)
    │   3. Crop to (1, 26, 26, 1152)  [drop 1 col + 1 row]
    │   4. Reshape to (1, 13, 2, 13, 2, 1152)
    │   5. Permute to (1, 13, 13, 2, 2, 1152)
    │   6. Reshape to (169, 4608)
    │   7. fc1: (169, 4608)
    │   8. GELU activation
    │   9. fc2: (169, 1536)
    │
vision_features: (169, 1536)  ← 169 vision "tokens" in LLM space

═══════════════════════════════════════════════════════════════════
STEP 4: Embed text + scatter vision
═══════════════════════════════════════════════════════════════════

input_ids: (1, 173)
    │
    ▼ self.transformer.wte (embedding lookup, padded vocab)
    │
inputs_embeds: (1, 173, 1536)  bfloat16
    │
    ▼ scatter_vision_features(inputs_embeds, vision_features, image_pad_mask)
    │   - Clone inputs_embeds
    │   - At 169 True positions in image_pad_mask, replace with vision_features
    │
fused_embeds: (1, 173, 1536)
    Position 0:    text embedding for "What"
    Position 1:    text embedding for "is"
    Position 2:    text embedding for "in"
    Position 3:    text embedding for "this"
    Position 4:    vision_features[0]
    Position 5:    vision_features[1]
    ...
    Position 172:  vision_features[168]
    Position 173:  text embedding for "image"

═══════════════════════════════════════════════════════════════════
STEP 5: Build 3D MRoPE positions
═══════════════════════════════════════════════════════════════════

input_ids + image_grids_merged ([(1, 13, 13)])
    │
    ▼ build_position_ids_for_mm — walk left-to-right
    │
position_ids: (3, 1, 173)
    Token 0 ("What"):  (t=0, h=0, w=0)
    Token 1 ("is"):    (t=1, h=0, w=0)
    Token 2 ("in"):    (t=2, h=0, w=0)
    Token 3 ("this"):  (t=3, h=0, w=0)
    Token 4 (IMG):     (t=4, h=0,  w=0)   ← image starts at t=4
    Token 5:           (t=4, h=0,  w=1)
    ...
    Token 16:          (t=4, h=0,  w=12)
    Token 17:          (t=4, h=1,  w=0)
    ...
    Token 172:         (t=4, h=12, w=12)  ← last image patch
    Token 173 ("image"): (t=5, h=0, w=0)  ← image consumed T=1, t advances by 1

═══════════════════════════════════════════════════════════════════
STEP 6: Build 3D MRoPE cos/sin (Karpathy layout)
═══════════════════════════════════════════════════════════════════

position_ids: (3, 1, 173)
head_dim: 128 (n_embd=1536, n_head=12, head_dim=128)
    │
    ▼ build_3d_mrope_for_4d_apply
    │   1. inv_freq (64,)  [head_dim/2 frequencies]
    │   2. axis_for_freq = [0,1,2,0,1,2,...] (round-robin)
    │   3. pos_per_freq (64, 1, 173) — gather position per freq
    │   4. Permute to (1, 173, 64)
    │   5. Multiply by inv_freq
    │   6. cos = angle.cos().unsqueeze(2) → (1, 173, 1, 64)  bfloat16
    │   7. sin = same → (1, 173, 1, 64)
    │
cos: (1, 173, 1, 64)  bfloat16
sin: (1, 173, 1, 64)  bfloat16

═══════════════════════════════════════════════════════════════════
STEP 7: Through MoE Transformer trunk
═══════════════════════════════════════════════════════════════════

x = fused_embeds = (1, 173, 1536)
    │
    ▼ norm(x) — initial RMSNorm
    │
For each of n_layer=24 blocks:
    │
    ▼ x = resid_lambda * x + x0_lambda * x0  [skip connection blending]
    │
    ▼ Block.forward(x, ve, cos_sin, window_size, kv_cache):
    │   - CausalSelfAttention:
    │     * Project to Q, K, V via Linear layers
    │     * apply_rotary_emb(Q, cos, sin)  ← position info injected here
    │     * apply_rotary_emb(K, cos, sin)
    │     * QK norm
    │     * Flash Attention 3: y = softmax(QK^T / √d) @ V
    │   - MoE FFN:
    │     * Router → top-2 experts per token
    │     * Per-token: 2 expert outputs + 1 shared expert
    │     * Combined via router scores
    │
After 24 layers: x = (1, 173, 1536)

═══════════════════════════════════════════════════════════════════
STEP 8: Output projection + softcap
═══════════════════════════════════════════════════════════════════

x: (1, 173, 1536)
    │
    ▼ norm(x) — final RMSNorm
    │
    ▼ self.lm_head: Linear(1536, padded_vocab=32064)
    │
logits: (1, 173, 32064)
    │
    ▼ slice to actual vocab: logits[..., :32000]
    ▼ float32 cast (for numerical stability)
    ▼ softcap: 15 * tanh(logits / 15)  [smooth bound]
    │
logits: (1, 173, 32000)  float32

═══════════════════════════════════════════════════════════════════
STEP 9: Per-modality loss
═══════════════════════════════════════════════════════════════════

logits: (1, 173, 32000)
targets: (1, 173) — position 4..172 are -1, others have next-token IDs
modality_mask: (1, 173) — 1 at text-after-vision positions

per_modality_loss_decomposition(logits, targets, modality_mask)
    │
    ▼ CE per position (with ignore_index=-1)
    │
    ▼ Split by modality_mask
    │
return {
    "loss": scalar (backward-able),
    "loss_text": scalar,
    "loss_vision": scalar,
    "n_text": int,
    "n_vision": int,
}

═══════════════════════════════════════════════════════════════════
STEP 10: Backward pass
═══════════════════════════════════════════════════════════════════

loss.backward()
    │
    ▼ Gradients flow backward through:
    │   ✓ lm_head
    │   ✓ MoE experts (only top-K active)
    │   ✓ Attention (Q, K, V projections, c_proj)
    │   ✓ resid_lambdas, x0_lambdas
    │   ✓ wte (embedding table)
    │   ✓ value_embeds
    │   ✓ PatchMerger (if not frozen — usually frozen)
    │   ✗ SigLIP2 (frozen — no gradient)
    │
Optimizer step: Muon (matrices) + AdamW (embeddings, scalars)
```

---

## Phần 11 — Why This Design Wins (Senior Researcher Perspective)

### 11.1 Compound trade-offs

Single design choice ALONE doesn't make this work. The MAGIC is how they compound:

| Decision                 | Enables                                                |
| ------------------------ | ------------------------------------------------------ |
| Patches not pixels       | Computational tractability                             |
| ViT not CNN              | Global reasoning at every layer                        |
| Frozen ViT               | HP envelope inheritance + cheap compute                |
| 2×2 PatchMerger          | Sequence length manageable                             |
| Scatter at `<image_pad>` | Trunk treats vision as text — no architectural changes |
| 3D MRoPE                 | Vision tokens get spatial position info                |
| Text-only loss           | Architectural simplicity + matches downstream          |
| MoE trunk                | More capacity per FLOP                                 |
| Bergsma `B ∝ D^0.383`    | Optimal batch derived from D automatically             |
| `1/√d` LR scaling        | HPs transfer across model widths                       |

→ **No single decision is the secret. The COMBINATION is. Senior researchers think in compounded constraints.**

### 11.2 What Qwen3.5 chose vs alternatives

| Alternative                              | Why Qwen3.5 rejected                                | Why we follow Qwen3.5                        |
| ---------------------------------------- | --------------------------------------------------- | -------------------------------------------- |
| Late fusion (Qwen2.5-VL 3-stage)         | LLM never sees vision early; needs "rewiring" later | Joint from step 0 = faster convergence       |
| Multi-stage alignment (Qwen3-VL 4-stage) | 1.4B-pair stage costs 1 week                        | Single joint training stage                  |
| DeepStack sidecars                       | More complexity; Qwen team dropped in 3.5           | Simpler single insertion point               |
| VQ tokenizer (discrete)                  | Reconstruction floor on vision loss                 | Continuous SigLIP2 features have no floor    |
| Cross-attention (Flamingo)               | Adds parameters; breaks "vision = text" abstraction | Scatter mechanism is cleaner                 |
| Pure ViT trainable                       | Costs 2.5× compute                                  | Frozen ViT preserves quality + saves compute |

→ Qwen3.5-VL = **the most production-disciplined frontier multimodal design 2026.** Their ablations explicitly say it beats their own Qwen3-VL late-fusion variant.

We follow them VERBATIM (within scope). Not innovating on architecture; innovating on MEASUREMENT (the scaling law).

### 11.3 Senior researcher mental model in 1 paragraph

**Multimodal = mechanism to make image LOOK LIKE text to the LLM trunk, with minimum architectural disruption.** Frozen ViT preserves a billion-dollar pretrained artifact for free. PatchMerger shrinks 729 → 169 to fit budget. Scatter at `<image_pad>` keeps trunk identical to text-only. 3D MRoPE encodes spatial position so attention can natively distinguish "patch in row 5" from "text token at position 5." Text-only loss + ignore_index on vision positions matches downstream task structure (output text, vision is context). MoE adds capacity per FLOP. The whole design is **5 frozen-knob constraints + 5 architectural choices** that compound such that a billion-dollar ViT + a $300 measurement produce a publishable multimodal scaling law. **Master that compound, you master frontier multimodal training.**

---

## Phần 12 — Self-Check Questions (Feynman Final Test)

Bạn hiểu đến mức master nếu trả lời được:

1. **Tại sao SigLIP2 dùng patch_size=14 (không 16 hoặc 32)?**
   *Trả lời:* SigLIP2 was empirically tuned for 384/14 = 27 patches per side at ImageNet-class tasks. Smaller patches = more compute, finer detail. 14 is the sweet spot for SigLIP2's training data + compute.

2. **Tại sao chia head_dim into halves (Karpathy convention) thay vì pairs (HF convention)?**
   *Trả lời:* Mathematically equivalent. Karpathy's halves are CONTIGUOUS in memory → faster GPU kernels. HF's pairs are alternating → slightly worse memory access pattern.

3. **Nếu bạn UNFREEZE SigLIP2, what BREAKS in our scaling law?**
   *Trả lời:* (a) Compute budget cuts by ~60%; (b) ViT representation drifts → "G\* effect" conflated with "ViT representation shift" — confound. Frozen ViT is what licenses cell-to-cell comparability.

4. **Tại sao mix_ratio=0.3 (không 0.1 hoặc 0.5)?**
   *Trả lời:* Production-realistic value. Qwen3.5-VL's training mix estimated at ~25-35% vision tokens. r=0.3 is the modal value across frontier 2026 multimodal models. Single value chosen to defend in writeup.

5. **Tại sao backward pass NEVER goes through SigLIP2 even if you forget `torch.no_grad()`?**
   *Trả lời:* Because `requires_grad=False` on all SigLIP2 params → autograd doesn't track operations on them. `torch.no_grad()` would additionally save activation memory (no need to retain forward activations). With requires_grad=False alone, you save backward compute but NOT activation memory.

6. **3D MRoPE with axis_for_freq round-robin: tại sao không [24, 20, 20] partitioning (Qwen3.5 production)?**
   *Trả lời:* Round-robin is the docstring's documented simplification per qwen35_vl_tiny.py reference. Production uses mrope_section partitioning that gives slightly more freqs to t-axis. Functionally similar at our scale; round-robin is cleaner code.

7. **Tại sao loss/vision = 0 in initial Track B' (without modality_mask refinement)?**
   *Trả lời:* Original modality_mask was = image_pad_mask. Vision positions have target=-1 (ignore). So intersect of "vision_mask" (modality_mask=1) AND "valid" (target≠-1) = empty set → n_vision=0 → loss_vision/0 → handled as 0. Track B'' refined modality_mask to tag TEXT tokens with vision context → meaningful loss_vision.

8. **Khi attention masking is causal (vision token at pos 5 cannot attend to text at pos 100), tại sao MoE for multimodal still works?**
   *Trả lời:* Vision tokens come BEFORE text in our design (image_pad runs are placed in mid-sequence text). Vision tokens attend to PRIOR vision + text tokens. Text tokens AFTER vision can attend to vision (and prior text). This is the "vision as context for text predictions" pattern.

9. **Tại sao chúng ta use bfloat16 cho cos/sin (not float32)?**
   *Trả lời:* Memory bandwidth. cos/sin are precomputed and cached, but during attention they multiply against bf16 Q/K. Casting to fp32 would force conversion → slower. bf16 has same exponent range as fp32, only loses some precision in the mantissa — acceptable for trig values in [-1, 1].

10. **Last test: explain to a non-AI friend in 30 seconds why our pipeline works.**
    *Sample answer:* "We have a frozen 'image describer' (SigLIP2) that turns any image into 169 vector descriptions. We squeeze them through a small projector to make them look like fake words. Then we splice these fake words into a text sentence at specific blank positions. Our language model (which only knows text) processes everything together as if it were a weird sentence. The model learns 'when you see fake-word X near real-word Y, predict real-word Z.' Result: it learns to use vision context to predict text, without ever being explicitly trained on what vision means."

---

## Senior researcher discipline trong 1 câu

**Multimodal = engineering to bridge two pretrained spaces (vision encoder + language model) with the minimum architectural disruption that still allows joint training.** Every design decision serves "minimum disruption to inherit pretrained quality." That's the meta-principle.

You now master multimodal vision pipeline at frontier-lab depth. Câu hỏi nào khác về implementation hoặc design cần đi sâu hơn?