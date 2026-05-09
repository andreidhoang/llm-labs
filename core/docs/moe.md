# Mixture of Experts (MoE): From First Principles

> Written as if a senior research engineer at a frontier lab is walking you through the design.
> Tiếng Anh xen kẽ tiếng Việt — không phải 2 khối riêng biệt.

---

## 1. The Core Problem: Why Do We Need MoE?

In a standard Transformer, every token goes through **the same** MLP at every layer.

```
Token "cat"  →  Block 0 MLP  →  Block 1 MLP  →  Block 2 MLP  →  ...
Token "the"  →  Block 0 MLP  →  Block 1 MLP  →  Block 2 MLP  →  ...
Token "3.14" →  Block 0 MLP  →  Block 1 MLP  →  Block 2 MLP  →  ...
```

But different tokens need **different computations**:
- "the" là stop word — chỉ cần một phép biến đổi đơn giản.
- "3.14" là số — cần reasoning về arithmetic.
- "function" trong code — cần hiểu syntax và scope.

**Ý tưởng then chốt của MoE**: Thay vì một MLP khổng lồ duy nhất, ta có nhiều "chuyên gia" (experts) nhỏ hơn. Mỗi token **chọn** (route đến) một vài chuyên gia phù hợp nhất.

> Đây là **conditional computation**: compute chỉ được chi cho những expert cần thiết.

---

## 2. The Dense MLP Baseline (Điểm chuẩn)

Trong `nanochat`, MLP dense là:

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc   = Linear(n_embd, 4 * n_embd)   # up-project
        self.c_proj = Linear(4 * n_embd, n_embd)   # down-project

    def forward(self, x):        # x: (B, T, n_embd)
        h = self.c_fc(x)         # h: (B, T, 4*n_embd)
        h = F.relu(h).square()   # activation
        return self.c_proj(h)    # out: (B, T, n_embd)
```

**Parameters**: `n_embd * 4*n_embd + 4*n_embd * n_embd = 8 * n_embd²`
**FLOPs per token**: `2 * n_embd * 4*n_embd + 2 * 4*n_embd * n_embd = 16 * n_embd²`

Với `n_embd = 768`:
- Params = `8 * 768² = 4,718,592` (~4.7M)
- FLOPs/token = `16 * 768² = 9,437,184` (~9.4M)

---

## 3. The MoE Layer: High-Level Architecture

```
Input x: (B, T, n_embd)
        │
        ▼
   ┌─────────┐
   │  Router │  → scores: (B*T, num_experts)
   └─────────┘
        │
        ▼
   top-k selection
        │
        ▼
┌──────────────────┐
│  Expert 0        │ ─┐
│  Expert 1        │ ─┤ weighted sum
│  ...             │ ─┤ (only top-k experts per token)
│  Expert N-1      │ ─┘
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  Shared Expert   │  → always active, stabilizes training
└──────────────────┘
        │
        ▼
   Output: (B, T, n_embd)
```

---

## 4. Step-by-Step Forward Pass with Real Numbers

Let's trace với concrete example:
- Batch size `B = 2`, sequence length `T = 4`, so `B*T = 8` tokens
- `n_embd = 6` (nhỏ để dễ theo dõi)
- `num_experts = 4`
- `top_k = 2`
- `expert_dim = 8`

### Step 4.1: Flatten tokens

```python
x: (B, T, n_embd) = (2, 4, 6)
x_flat = x.view(-1, 6)   # (8, 6)
```

Tensor:
```
x_flat = [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],   # token 0 (batch 0, pos 0)
    [0.2, 0.1, 0.4, 0.3, 0.6, 0.5],   # token 1
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],   # token 2
    ...
]  # shape: (8, 6)
```

### Step 4.2: Router computes scores

```python
scores = sigmoid(gate(x_flat))   # gate: Linear(6, 4)
# scores: (8, 4)
```

Ví dụ output:
```
scores = [
    [0.82, 0.15, 0.91, 0.33],   # token 0 → expert 2 mạnh nhất, rồi expert 0
    [0.21, 0.88, 0.45, 0.76],   # token 1 → expert 1, expert 3
    [0.91, 0.12, 0.85, 0.22],   # token 2 → expert 0, expert 2
    ...
]
```

> **Tại sao sigmoid chứ không phải softmax?**
> - Softmax: các experts cạnh tranh với nhau — nếu expert 0 mạnh lên, expert 1 phải yếu đi.
> - Sigmoid: mỗi expert độc lập — score chỉ phụ thuộc vào input, không phụ thuộc vào expert khác.
> Karpathy found sigmoid works better in practice. Intuitively, nếu một token cần cả "math" AND "syntax", softmax sẽ ép nó chọn một trong hai, còn sigmoid cho phép chọn cả hai.

### Step 4.3: Top-k selection

```python
topk_vals, topk_idx = torch.topk(scores, k=2, dim=-1)
# topk_vals: (8, 2) — gate weights
# topk_idx:  (8, 2) — expert indices
```

Ví dụ:
```
topk_idx = [
    [2, 0],   # token 0 → experts 2 và 0
    [1, 3],   # token 1 → experts 1 và 3
    [0, 2],   # token 2 → experts 0 và 2
    ...
]

topk_vals = [
    [0.91, 0.82],   # token 0
    [0.88, 0.76],   # token 1
    [0.91, 0.85],   # token 2
    ...
]
```

### Step 4.4: Normalize gates

```python
gates = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
```

```
gates = [
    [0.91/(0.91+0.82), 0.82/(0.91+0.82)] = [0.526, 0.474],
    [0.88/(0.88+0.76), 0.76/(0.88+0.76)] = [0.536, 0.464],
    [0.91/(0.91+0.85), 0.85/(0.91+0.85)] = [0.517, 0.483],
    ...
]
```

> Gates là trọng số weighted sum. Mỗi token chạy 2 experts, rồi kết hợp output theo tỷ lệ gate.

### Step 4.5: Expert weights

```python
w1: (4, 6, 8)   # 4 experts, each: (n_embd=6, expert_dim=8)
w2: (4, 8, 6)   # 4 experts, each: (expert_dim=8, n_embd=6)
```

Expert 0's weights:
```
w1[0] = [[... 48 numbers ...]]   # shape (6, 8)
w2[0] = [[... 48 numbers ...]]   # shape (8, 6)
```

### Step 4.6: Dispatch và compute

For **expert 0**:
```python
mask = (topk_idx == 0)   # (8, 2)
# token 0 chọn expert 0 ở vị trí thứ 2 → True
# token 2 chọn expert 0 ở vị trí đầu → True
# Các token khác → False

token_idx, gate_idx = mask.nonzero(as_tuple=True)
# token_idx = [0, 2, ...]  # which tokens
# gate_idx  = [1, 0, ...]  # which of their top-k slots

expert_input = x_flat[token_idx]     # (N_tokens_for_expert0, 6)
expert_gate  = gates[token_idx, gate_idx].unsqueeze(1)  # (N_tokens, 1)

# Forward through expert 0
h = F.linear(expert_input, w1[0])     # (N_tokens, 8)
h = F.relu(h).square()                # (N_tokens, 8)
out = F.linear(h, w2[0])              # (N_tokens, 6)

# Weight và accumulate
output[token_idx] += expert_gate * out
```

Lặp lại cho expert 1, 2, 3.

### Step 4.7: Shared Expert

```python
shared_out = shared_expert(x_flat)   # (8, 6), chạy trên TẤT CẢ tokens
output = output + shared_out
```

> Shared expert giống như "expert mặc định" — nó học những kiến thức chung (grammar, common sense) mà mọi token đều cần. Điều này ổn định training rất nhiều vì ngay cả khi routing là ngẫu nhiên ban đầu, shared expert vẫn cung cấp gradient signal hữu ích.

---

## 5. Iso-FLOP Sizing (Không tăng compute)

Đây là điểm **quan trọng nhất** khi so sánh MoE với dense.

**Dense MLP FLOPs**:
```
16 * n_embd²  =  16 * 768²  =  ~9.4M per token per layer
```

**MoE active FLOPs** (chỉ top_k + shared experts):
```
active = top_k + num_shared = 2 + 1 = 3

Mỗi expert có 2 matmuls:
  up:   2 * n_embd * expert_dim
  down: 2 * expert_dim * n_embd
  total per expert: 4 * n_embd * expert_dim

Active FLOPs = active * 4 * n_embd * expert_dim
             = 3 * 4 * 768 * expert_dim
             = 12 * 768 * expert_dim
```

Set `active_FLOPs = dense_FLOPs`:
```
12 * 768 * expert_dim = 16 * 768²
expert_dim = (16 * 768) / 12 = 1024
```

Round up to multiple of 128 (tensor core friendly):
```
expert_dim = 1024  (already divisible by 128 ✓)
```

Code:
```python
active = self.top_k + self.num_shared
self.expert_dim = round(4 * n_embd / active / 128) * 128
# = round(4 * 768 / 3 / 128) * 128
# = round(1024 / 128) * 128 = 8 * 128 = 1024
```

**Active parameters** ( chỉ 3 experts hoạt động mỗi token):
```
w1 active: 3 * 768 * 1024 = 2,359,296
w2 active: 3 * 1024 * 768 = 2,359,296
shared:    768 * 1024 + 1024 * 768 = 1,572,864
Total active: ~6.3M params

vs dense: 8 * 768² = 4.7M params
```

Wait — active params cao hơn dense? Đúng vậy! Đó chính là **trade-off của MoE**:
- Bạn dùng **nhiều parameters hơn** (tổng cộng 8 experts * 2 weights = 8 * 768 * 1024 * 2 ≈ 12.6M)
- Nhưng **FLOPs mỗi token** giữ nguyên (~9.4M)
- Model có **capacity lớn hơn** nhưng không chậm hơn trong inference.

---

## 6. The Load Balancing Problem

Nếu không can thiệp, router sẽ học cách gửi **mọi token đến cùng một expert** — expert đó trở thành "copy của MLP dense", còn các expert khác bị bỏ hoang.

```
Before training: evenly distributed
Expert 0: 25%   Expert 1: 25%   Expert 2: 25%   Expert 3: 25%

After 1000 steps (without load balancing):
Expert 0: 3%    Expert 1: 94%   Expert 2: 2%    Expert 3: 1%
         ↑ collapsed!
```

### Cách 1: Auxiliary Loss (cũ, kém hiệu quả)

Thêm một loss term:
```python
load = fraction of tokens sent to each expert   # (num_experts,)
avg_load = 1.0 / num_experts
aux_loss = num_experts * sum(load_i * avg_load)   # encourages uniform
```

**Vấn đề**: Hyperparameter `aux_loss_weight` rất khó tune. Nếu quá nhỏ → không balance. Nếu quá lớn → router bị ép chọn uniform, mất đi ý nghĩa của specialization.

### Cách 2: Bias Nudging (DeepSeekV3, cách ta dùng)

Không thêm loss. Thay vào đó, **điều chỉnh router bias** sau mỗi step:

```python
# Sau mỗi training step:
avg_load = total_tokens / num_experts
for each expert i:
    if load_i > avg_load:    # overloaded
        bias[i] -= nudge     # giảm score → ít token chọn hơn
    else:                    # underloaded
        bias[i] += nudge     # tăng score → nhiều token chọn hơn
```

Tensor example:
```
num_expert_tokens = [1800, 2200, 1900, 2100]   # out of 8000 total
target = 2000

bias_nudge = 0.001 * (2000 - [1800, 2200, 1900, 2100])
           = 0.001 * [200, -200, 100, -100]
           = [0.2, -0.2, 0.1, -0.1]

new_bias = old_bias + [0.2, -0.2, 0.1, -0.1]
```

Expert 1 bị quá tải → bias giảm → score thấp hơn → ít token chọn hơn.
Expert 0 bị thiếu → bias tăng → score cao hơn → nhiều token chọn hơn.

> **Tại sao cách này hay hơn?** Nó là **feedback loop** — không cần tune hyperparameter phức tạp, chỉ cần một hằng số nudge nhỏ. Nó can thiệp trực tiếp vào routing distribution mà không làm mất gradient signal.

---

## 7. 3D Expert Weights & Muon Optimizer

Đây là chi tiết implementation-specific cho codebase của bạn.

### Dense MLP parameters:
```python
c_fc.weight:   (3072, 768)   → 2D matrix
c_proj.weight: (768, 3072)   → 2D matrix
```

Muon stack các params cùng shape lại và orthogonalize chúng cùng lúc.

### MoE expert parameters:

Nếu ta dùng `ModuleList` của `Linear` layers:
```python
experts = nn.ModuleList([MLP(config) for _ in range(8)])
# Mỗi expert có c_fc (3072, 768) và c_proj (768, 3072)
# Muon thấy: 16 matrices shape (3072, 768) và 16 matrices shape (768, 3072)
```

Điều này **vẫn hoạt động**, nhưng có vấn đề: Muon orthogonalize mỗi expert **độc lập**, nhưng không có cách nào "biết" rằng expert 0 và expert 1 là các phần tử của cùng một layer.

### Karpathy's solution: 3D tensor

```python
self.w1 = nn.Parameter(torch.empty(8, 768, 1024))   # (num_experts, n_embd, expert_dim)
self.w2 = nn.Parameter(torch.empty(8, 1024, 768))   # (num_experts, expert_dim, n_embd)
```

**Shape analysis trong Muon**:
```python
# Muon stack: nếu có n layers, mỗi layer có w1 và w2
# w1 shape = (8, 768, 1024)
# Khi stack n layers: (n, 8, 768, 1024)

# Trong muon_step_fused:
X = stacked_grads  # (n, 8, 768, 1024)
X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
# norm computed over last 2 dims: (768, 1024) for EACH of (n, 8) matrices

# X.mT swaps last 2 dims: (n, 8, 1024, 768)
# X.mT @ X  →  (n, 8, 1024, 1024)
# → Each of the 8*n expert matrices is orthogonalized INDEPENDENTLY ✓
```

Đây là **feature, không phải bug**: Muon tự động xử lý 3D tensors đúng cách!

### Second Momentum Buffer Fix

Tuy nhiên, có một lỗi nhỏ trong `optim.py` hiện tại:

```python
# Dòng 252 (current):
state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
```

Với `shape = (8, 768, 1024)`:
- `shape[-2] = 768`, `shape[-1] = 1024`
- `768 < 1024` → `state_shape = (num_params, 1, 1024)`

Nhưng `v_mean` trong fused kernel là:
```python
v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
# g: (num_params, 8, 768, 1024)
# red_dim = -2 (vì 768 < 1024)
# v_mean: (num_params, 8, 768, 1)
```

`second_momentum_buffer` có shape `(num_params, 1, 1024)` không broadcast được với `v_mean` shape `(num_params, 8, 768, 1)`!

**Fix**: Bao gồm tất cả leading dimensions từ param shape:

```python
extra_dims = shape[:-2]
if shape[-2] >= shape[-1]:
    state_shape = (num_params, *extra_dims, shape[-2], 1)
else:
    state_shape = (num_params, *extra_dims, 1, shape[-1])
# Với (8, 768, 1024): state_shape = (num_params, 8, 1, 1024)
# V_mean (num_params, 8, 768, 1) broadcast với buffer (num_params, 8, 1, 1024) ✓
```

---

## 8. Inference vs Training

### Training
- Mỗi token chạy `top_k` experts → sparse compute
- Cần load balancing updates
- Backprop thông qua router gate

### Inference
- Vẫn cùng một forward pass
- Nhưng batch size thường nhỏ hơn (1 cho chat)
- **Problem**: Với batch=1, mỗi expert chỉ nhận ~0-1 tokens → matmuls rất nhỏ, GPU utilization kém
- **Solution ở scale lớn**: Expert parallelism (đặt mỗi expert trên GPU khác nhau)

> Với model size của bạn (~d20, single node), đây không phải vấn đề lớn.

---

## 9. Integration with Your Codebase

### 9.1 Model Config

```python
@dataclass
class GPTConfig:
    # ... existing fields ...
    moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_num_shared: int = 1
    moe_expert_dim: int | None = None
```

### 9.2 Factory Pattern

```python
def build_mlp_module(config, layer_idx):
    if config.moe:
        return MoELayer(config, layer_idx)
    return MLP(config)
```

### 9.3 Block Update

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = build_mlp_module(config, layer_idx)   # ← factory
```

### 9.4 Weight Init

```python
# For w1 (up-project, like c_fc):
torch.nn.init.uniform_(moe.w1, -s * 0.4, s * 0.4)

# For w2 (down-project, like c_proj):
torch.nn.init.zeros_(moe.w2)

# For router gate:
torch.nn.init.uniform_(moe.gate.weight, -0.02, 0.02)
```

### 9.5 Training Loop Integration

```python
# After optimizer.step()
for block in model.transformer.h:
    if isinstance(block.mlp, MoELayer):
        block.mlp.update_load_balance()
```

### 9.6 Checkpoint Isolation

```python
def variant_tag(self):
    tag = f"d{self.config.n_layer}"
    if self.config.moe:
        tag += f"_moe{self.config.moe_num_experts}t{self.config.moe_top_k}"
    return tag
```

---

## 10. Common Pitfalls

| Pitfall | Why It Happens | Fix |
|---------|---------------|-----|
| **All tokens → 1 expert** | No load balancing | Bias nudging, hoặc aux loss |
| **NaNs in router** | Sigmoid overflow/underflow | Clamp scores, use stable softmax/sig |
| **Slow dispatch** | Python loop over experts | `torch._grouped_mm` khi ready |
| **OOM** | Tạo mask (B*T, num_experts) quá lớn | Sparse operations, hoặc chunked compute |
| **Muon crash** | 3D tensor buffer shape mismatch | Fix `second_momentum_buffer` shape |
| **Incompatible checkpoint** | Load dense into MoE or vice versa | Assert config match in loader |

---

## 11. Tensor Shape Cheat Sheet

Given: `B=32, T=2048, n_embd=768, num_experts=8, top_k=2, expert_dim=1024`

| Tensor | Shape | Notes |
|--------|-------|-------|
| `x` (input) | `(32, 2048, 768)` | activations |
| `x_flat` | `(65536, 768)` | flattened for routing |
| `scores` | `(65536, 8)` | router output |
| `topk_idx` | `(65536, 2)` | selected experts per token |
| `topk_vals` | `(65536, 2)` | raw scores |
| `gates` | `(65536, 2)` | normalized weights |
| `w1` | `(8, 768, 1024)` | all expert up-projections |
| `w2` | `(8, 1024, 768)` | all expert down-projections |
| `expert_input` | `(~8192, 768)` | tokens routed to one expert |
| `expert_output` | `(~8192, 768)` | output before gating |
| `shared_out` | `(65536, 768)` | always computed |
| `output` | `(32, 2048, 768)` | final MoE output |

~8192 = 65536 / 8 experts — đây là lý do tại sao dispatch overhead quan trọng. Nếu không có `grouped_mm`, Python loop over 8 experts với tensors ~8K rows là chấp nhận được nhưng không optimal.

---

*End of first-principles walkthrough. Next step: implementation.*


# Mixture of Experts: From Absolute First Principles

---

## 1. The Fundamental Problem — Tại Sao MoE Tồn Tại?

Hãy bắt đầu từ câu hỏi sâu nhất: **what is a neural network actually doing?**

When a token flows through a dense Transformer MLP, every single parameter fires for every single token. "cat", "3.14", "def", "the" — tất cả đều đi qua **cùng một** set of weights:

```
Token "the"      →  [W_up: 768→3072]  →  ReLU²  →  [W_down: 3072→768]
Token "3.14"     →  [W_up: 768→3072]  →  ReLU²  →  [W_down: 3072→768]
Token "def func" →  [W_up: 768→3072]  →  ReLU²  →  [W_down: 3072→768]
```

Câu hỏi đặt ra: **is this the right inductive bias?** Does "the" really need the same computation as "3.14"?

Không. "the" là function word — syntactic glue. "3.14" cần arithmetic reasoning. "def" cần code structure understanding. They need **different computations**, but a dense MLP gives them **identical ones**.

Đây là **inefficiency ở tầng sâu nhất** — you're paying full compute for every token regardless of what that token actually needs.

---

## 2. The MoE Insight — Conditional Computation

**Core idea**: thay vì một MLP khổng lồ chạy trên mọi token, ta có **nhiều MLP nhỏ** (experts) và mỗi token **chọn** chỉ một vài experts phù hợp.

Đây không phải ý tưởng mới — nó xuất phát từ **"Mixture of Experts"** của Jacobs et al. (1991), nhưng mãi đến Shazeer et al. (2017) và sau đó DeepSeekV3 (2024) mới trở thành production-quality.

The key mathematical property: **conditional computation**

```
Dense MLP:   FLOPs/token = 16 * d²     (ALL params fire)
MoE:         FLOPs/token = 16 * d²     (SAME — but only k+s experts out of N)
             Total params = N * (params per expert)
```

Bạn có thể có **10x more parameters** với **same compute per token**. That's the magic.

---

## 3. Router Design — Cái Não của MoE

### 3.1 Why Sigmoid, Not Softmax?

Đây là một trong những decisions quan trọng nhất và thường bị hiểu sai.

**Softmax router:**
```python
scores = softmax(W_gate @ x)   # scores sum to 1
```

Vấn đề: softmax tạo ra **competition** giữa các experts. Nếu expert 0 score tăng, expert 1 phải giảm — bất kể nội dung token. Đây là **zero-sum game**.

Ví dụ cụ thể: token "3.14" cần cả "arithmetic" expert VÀ "decimal notation" expert. Với softmax, model bị ép chọn một trong hai. Với sigmoid:

```python
scores = sigmoid(W_gate @ x)   # scores are INDEPENDENT
```

Mỗi expert score chỉ phụ thuộc vào input, không phụ thuộc vào experts khác. Token "3.14" có thể score cao cả expert 2 (arithmetic) lẫn expert 5 (number format) simultaneously.

**Intuition sâu hơn**: sigmoid là "do I need this expert?" (binary relevance per expert). Softmax là "which expert is most relevant?" (forced ranking). Cái đầu phản ánh thực tế tốt hơn — a token can genuinely need multiple capabilities.

### 3.2 Tensor Trace: Router Forward Pass

```
Input x: (B=2, T=4, n_embd=6)
Flatten: x_flat = (8, 6)

x_flat = [
  [0.8, 0.2, -0.1,  0.5,  0.3, -0.4],   # token 0
  [0.1, 0.9,  0.7, -0.2,  0.6,  0.1],   # token 1
  [-0.5, 0.3, 0.8,  0.4, -0.1,  0.7],   # token 2
  ...                                     # tokens 3-7
]  # shape: (8, 6)

W_gate: (num_experts=4, n_embd=6)   ← Linear layer weight (no bias initially)

raw = x_flat @ W_gate.T             # (8, 4)
raw = [
  [ 0.42,  1.21, -0.33,  0.87],    # token 0
  [ 1.15, -0.44,  0.92,  0.61],    # token 1
  [-0.21,  0.88,  1.43, -0.55],    # token 2
  ...
]

scores = sigmoid(raw)               # (8, 4) — each in (0, 1)
scores = [
  [0.60, 0.77, 0.42, 0.70],        # token 0
  [0.76, 0.39, 0.71, 0.65],        # token 1
  [0.45, 0.71, 0.81, 0.37],        # token 2
  ...
]
```

Chú ý: các scores này là **independent** — không có constraint nào ép chúng sum to 1.

---

## 4. Top-k Selection và Gate Normalization

### 4.1 Tại Sao Cần Normalize?

Sau khi chọn top-k experts, ta cần **weighted combination** của outputs. Nhưng với gì?

```
token 0: top-2 experts are [1, 3] với scores [0.77, 0.70]

Option A (raw scores):  output = 0.77 * expert1(x) + 0.70 * expert3(x)
Option B (normalized):  output = 0.52 * expert1(x) + 0.48 * expert3(x)
```

Option B (normalized) tốt hơn vì nó ensures the weighted sum has the **same scale** as individual expert outputs. Option A would make the output scale vary wildly depending on absolute score magnitudes.

### 4.2 Load Balancing Bias Trick

Đây là DeepSeekV3's key innovation — **auxiliary-loss-free load balancing**.

Vấn đề: ta muốn routing based on *true expert relevance* (sigmoid scores), nhưng ta cũng cần load balancing. Nếu ta directly bias the scores, ta corrupt the gate weights used for combination.

**Giải pháp**: use **two separate quantities**:
```python
scores = sigmoid(gate(x))                        # true relevance — for GATE WEIGHTS
routing = scores + router_bias                   # biased — for ROUTING DECISION only
topk_vals, topk_idx = topk(routing, k=top_k)    # decide WHICH experts using biased
raw_gates = scores.gather(1, topk_idx)           # use UNBIASED scores for weighting
gates = raw_gates / raw_gates.sum(-1, keepdim=True)
```

Brilliant separation: **routing decision** uses biased scores (load balancing), **output weighting** uses unbiased scores (true relevance). The two are decoupled.

### 4.3 Full Tensor Trace

```
scores (8, 4):
[[0.60, 0.77, 0.42, 0.70],   # token 0
 [0.76, 0.39, 0.71, 0.65],   # token 1
 [0.45, 0.71, 0.81, 0.37],   # token 2
 ...]

router_bias (4,): [0.0, 0.0, 0.0, 0.0]   ← starts at zero, nudged during training

routing = scores + router_bias  # same as scores at init

# top_k=2:
topk_vals (8,2):
[[0.77, 0.70],   # token 0 → experts [1, 3]
 [0.76, 0.71],   # token 1 → experts [0, 2]
 [0.81, 0.71],   # token 2 → experts [2, 1]
 ...]

topk_idx (8,2):
[[1, 3],
 [0, 2],
 [2, 1],
 ...]

# Gather UNBIASED scores for gates:
raw_gates = scores.gather(1, topk_idx)
raw_gates (8,2):
[[0.77, 0.70],   # token 0: expert1=0.77, expert3=0.70
 [0.76, 0.71],   # token 1: expert0=0.76, expert2=0.71
 [0.81, 0.71],   # token 2: expert2=0.81, expert1=0.71
 ...]

# Normalize:
gates = raw_gates / raw_gates.sum(-1, keepdim=True)
gates (8,2):
[[0.77/(0.77+0.70), 0.70/(0.77+0.70)],   =  [0.524, 0.476]
 [0.76/(0.76+0.71), 0.71/(0.76+0.71)],   =  [0.517, 0.483]
 [0.81/(0.81+0.71), 0.71/(0.81+0.71)],   =  [0.533, 0.467]
 ...]
```

---

## 5. Expert Dispatch — The Hard Part

### 5.1 The Core Challenge

Bây giờ ta có 8 tokens, mỗi token cần 2 experts. Tổng cộng 16 (token, expert) pairs. Ta cần:
1. **Group** tokens by expert (for efficient batched matmul)
2. **Compute** each expert's output
3. **Scatter** results back to original token positions

Đây là phần phức tạp nhất về mặt implementation.

### 5.2 Sorting Strategy — Tensor Trace Chi Tiết

```
topk_idx (8, 2):        # shape: (M=8, top_k=2)
[[1, 3],                # token 0 → experts 1 and 3
 [0, 2],                # token 1 → experts 0 and 2
 [2, 1],                # token 2 → experts 2 and 1
 [3, 0],                # token 3 → experts 3 and 0
 [1, 2],                # token 4 → experts 1 and 2
 [0, 3],                # token 5 → experts 0 and 3
 [2, 0],                # token 6 → experts 2 and 0
 [1, 3]]                # token 7 → experts 1 and 3
```

**Flatten both token indices and expert assignments:**

```python
flat_expert = topk_idx.reshape(-1)
# [1, 3, 0, 2, 2, 1, 3, 0, 1, 2, 0, 3, 2, 0, 1, 3]
#  ↑tok0    ↑tok1    ↑tok2    ↑tok3    ↑tok4...
# shape: (16,) = (M * top_k,)

flat_token = arange(8).unsqueeze(1).expand(-1, 2).reshape(-1)
# [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
# shape: (16,)
```

**Sort by expert (argsort):**

```python
sort_idx = flat_expert.argsort(stable=True)
# flat_expert = [1,3,0,2,2,1,3,0,1,2,0,3,2,0,1,3]
# sorted order: expert 0 first, then 1, then 2, then 3
# expert 0 appears at positions: 2,7,10,13 → tokens 1,3,5,6
# expert 1 appears at positions: 0,5,8,14  → tokens 0,2,4,7
# expert 2 appears at positions: 3,4,9,12  → tokens 1,2,4,6
# expert 3 appears at positions: 1,6,11,15 → tokens 0,3,5,7

sorted_expert = flat_expert[sort_idx]
# [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]

sorted_token = flat_token[sort_idx]
# [1,3,5,6, 0,2,4,7, 1,2,4,6, 0,3,5,7]
# ↑expert0  ↑expert1 ↑expert2 ↑expert3

expert_counts = [4, 4, 4, 4]   # perfectly balanced in this example
```

**Gather inputs in sorted order:**

```python
tokens_sorted = x_flat[sorted_token]   # (16, 6)
# rows 0-3:  inputs for expert 0 (tokens 1,3,5,6)
# rows 4-7:  inputs for expert 1 (tokens 0,2,4,7)
# rows 8-11: inputs for expert 2 (tokens 1,2,4,6)
# rows 12-15:inputs for expert 3 (tokens 0,3,5,7)
```

Bây giờ ta có một tensor liên tục trong bộ nhớ, grouped by expert. Đây là điều kiện cần để dùng `grouped_mm`.

### 5.3 The Loop Dispatch (Simple, Always Works)

```python
out = zeros_like(tokens_sorted)   # (16, 6)
offset = 0
for e, count in enumerate([4, 4, 4, 4]):
    t = tokens_sorted[offset:offset+count]    # (4, 6)
    h = F.linear(t, w1[e])                   # (4, expert_dim), w1[e] shape: (expert_dim, n_embd)
    h = F.relu(h).square()                   # (4, expert_dim)
    out[offset:offset+count] = F.linear(h, w2[e])  # (4, 6)
    offset += count
```

### 5.4 The GroupedMM Dispatch (Fast, bf16 only)

`torch._grouped_mm(A, B, offs)` là một **fused CUDA kernel** that does all experts' matmuls in a single GPU call. Đây là key performance trick từ DeepSeekV3.

**Weight storage design**: để `_grouped_mm` hoạt động, B matrix cần ở **col-major** format. Ta thiết kế weight storage từ đầu để satisfy này:

```
w1 stored as: (num_experts, expert_dim, n_embd)   # = (out, in) per expert
# w1.mT has shape: (num_experts, n_embd, expert_dim)
# each slice w1.mT[e] has strides (1, n_embd) → col-major ✓

w2 stored as: (num_experts, n_embd, expert_dim)   # = (out, in) per expert  
# w2.mT has shape: (num_experts, expert_dim, n_embd)
# each slice w2.mT[e] has strides (1, n_embd) → col-major ✓
```

```python
offs = expert_counts.cumsum(0).to(torch.int32)
# [4, 8, 12, 16]  ← cumulative end of each expert's tokens

# Up-project: (16, n_embd) @ col-major (n_embd, expert_dim) per expert
h   = torch._grouped_mm(tokens_sorted, w1.mT, offs)   # (16, expert_dim)
h   = F.relu(h).square()
# Down-project: (16, expert_dim) @ col-major (expert_dim, n_embd) per expert  
out = torch._grouped_mm(h,            w2.mT, offs)   # (16, n_embd)
```

Hai kernel calls thay vì 8 Python iterations — đây là lý do tại sao grouped_mm matters at large scale.

### 5.5 Scatter Back — Weighted Combination

```python
# flat_gates: gate values for each sorted (token, expert) pair
flat_gates = gates.reshape(-1)[sort_idx]   # (16,)
# gates.reshape(-1) = [g00, g01, g10, g11, g20, g21, ...]
#   where gij = gate weight for token i, slot j
# [sort_idx] reorders to match sorted token order

flat_gates = flat_gates.unsqueeze(-1)      # (16, 1) for broadcasting

output = zeros(M=8, n_embd=6)
output.scatter_add_(
    dim=0,
    index=sorted_token.unsqueeze(-1).expand_as(out),  # (16, 6)
    src=out * flat_gates,                              # (16, 6)
)
# scatter_add: for each of the 16 rows, add weighted expert output
# to the token's row in output
# token 0 receives: gate[0,expert1] * expert1_out + gate[0,expert3] * expert3_out
# token 1 receives: gate[1,expert0] * expert0_out + gate[1,expert2] * expert2_out
# etc.
```

---

## 6. The Load Balancing Problem

### 6.1 Expert Collapse — Tại Sao Nó Xảy Ra

Không có intervention, router sẽ học **positive feedback loop**:

```
Step 1: Expert 2 ngẫu nhiên nhận được tokens tốt → loss giảm
Step 2: Router gradient: "expert 2 is good → send more tokens there"
Step 3: Expert 2 nhận nhiều tokens hơn → gets more gradient → improves faster
Step 4: Router learns "always pick expert 2"
Step 5: Other experts starve (no gradient) → become useless
```

Kết quả sau 1000 steps không có load balancing:
```
Expert 0: 3%    Expert 1: 94%   Expert 2: 2%    Expert 3: 1%
```

Expert 1 trở thành copy của dense MLP. Đây là **degenerate MoE** — bạn có 8x params nhưng chỉ dùng 1x.

### 6.2 Bias Nudging — Feedback Loop Ngược Chiều

Sau mỗi optimizer step:

```python
@torch.no_grad()
def update_load_balance(self, nudge=0.001):
    avg = self._token_counts.mean()          # target: uniform distribution
    # _token_counts: [1800, 2200, 1900, 2100] tokens per expert (out of 8000 total, avg=2000)
    
    delta = nudge * (avg - self._token_counts) / (avg + 1e-9)
    # delta = 0.001 * (2000 - [1800, 2200, 1900, 2100]) / 2000
    #       = 0.001 * [200, -200, 100, -100] / 2000
    #       = [+0.0001, -0.0001, +0.00005, -0.00005]
    
    self.router_bias += delta
    # Expert 1 overloaded → bias decreases → routing score decreases → fewer tokens
    # Expert 0 underloaded → bias increases → routing score increases → more tokens
```

**Tại sao cách này hay hơn auxiliary loss?**
- Auxiliary loss cần hyperparameter weight — quá nhỏ thì không balance, quá lớn thì destroy specialization
- Bias nudging is a **direct feedback controller** — no gradient interaction, no extra hyperparameter
- Nó can thiệp vào *routing decision* nhưng không ảnh hưởng đến *gate weights* dùng để combine outputs

---

## 7. Iso-FLOP Sizing — Công Thức Toán Học

Đây là **critical design constraint**: MoE phải match dense FLOPs, otherwise comparison is unfair.

**Dense MLP FLOPs per token per layer:**
```
Up:   2 * n_embd * 4*n_embd = 8 * n_embd²
Down: 2 * 4*n_embd * n_embd = 8 * n_embd²
Total: 16 * n_embd²

Với n_embd=768: 16 * 768² = 9,437,184 FLOPs/token/layer
```

**MoE active FLOPs per token per layer:**
```
active_experts = top_k + num_shared = 2 + 1 = 3

Per expert: 2 * n_embd * expert_dim  (up)
          + 2 * expert_dim * n_embd  (down)
          = 4 * n_embd * expert_dim

Active FLOPs = active * 4 * n_embd * expert_dim
```

**Set equal và solve:**
```
active * 4 * n_embd * expert_dim = 16 * n_embd²
expert_dim = (16 * n_embd) / (4 * active)
           = 4 * n_embd / active
           = 4 * 768 / 3 = 1024

Round to multiple of 128 (tensor core alignment):
expert_dim = round(4 * n_embd / active / 128) * 128 = 1024 ✓
```

Trong code:
```python
active = top_k + num_shared
expert_dim = round(4 * n_embd / active / 128) * 128
```

---

## 8. 3D Weight Tensors và Muon Optimizer

### 8.1 Tại Sao 3D?

Dense MLP weights: `c_fc.weight` có shape `(3072, 768)` — 2D matrix. Muon stack các params cùng shape và orthogonalize chúng cùng lúc.

Nếu dùng `ModuleList` của 8 experts, Muon sẽ thấy 8 separate `(3072, 768)` matrices và orthogonalize từng cái independently. Hoạt động đúng về mặt toán học, nhưng không exploit structure.

3D tensors `(8, 1024, 768)` elegant hơn:
- Muon's Polar Express operates on **last two dims**
- Nó tự động xử lý `(num_experts, expert_dim, n_embd)` đúng cách
- Mỗi expert được orthogonalized independently: `(8, 1024, 768)` → Muon treats each of the 8 slices as separate `(1024, 768)` matrices

### 8.2 The `second_momentum_buffer` Bug

Đây là bug thực sự trong code hiện tại khi dùng 3D tensors. Hãy trace through:

```python
# Current code (broken for 3D):
shape = (8, 1024, 768)   # w1: (num_experts, expert_dim, n_embd)
state_shape = (num_params, shape[-2], 1)   # shape[-2]=1024 >= shape[-1]=768
#           = (num_params, 1024, 1)        # ← MISSING the 8 (num_experts) dim!

# What happens in the fused kernel:
g = stacked_grads    # (num_params, 8, 1024, 768)
v_mean = g.square().mean(dim=-1, keepdim=True)  # (num_params, 8, 1024, 1)

# Buffer shape: (num_params, 1024, 1)
# v_mean shape: (num_params, 8, 1024, 1)
# → RuntimeError: cannot broadcast (num_params, 1024, 1) with (num_params, 8, 1024, 1)
```

**Fix:**
```python
extra_dims = shape[:-2]                   # (8,) for 3D, () for 2D — handles both!
if shape[-2] >= shape[-1]:
    state_shape = (num_params, *extra_dims, shape[-2], 1)
    # 2D: (num_params, 1024, 1)     ← same as before ✓
    # 3D: (num_params, 8, 1024, 1)  ← correct for MoE ✓
else:
    state_shape = (num_params, *extra_dims, 1, shape[-1])
```

---

## 9. Code Architecture — Implementation Blueprint

Bây giờ ta đã hiểu mọi thứ, đây là cấu trúc code sạch cần implement:

### 9.1 File: `core/model.py`

```
GPTConfig                    ← add 5 moe_* fields
MLP                          ← unchanged
MoELayer                     ← NEW: full MoE implementation
  __init__                   ← gate, w1, w2, shared_fc/proj, router_bias buffer
  forward(x)                 ← router → topk → dispatch → shared → scatter
  _dispatch(x, topk_idx, g)  ← sort → grouped_mm or loop → scatter_add
  _grouped_mm(tokens, counts) ← torch._grouped_mm path (bf16 only)
  _loop(tokens, counts)       ← Python loop fallback (any dtype)
  update_load_balance()       ← nudge router_bias after optimizer step
build_mlp(config)             ← NEW: factory, returns MLP or MoELayer
Block.__init__                ← self.mlp = build_mlp(config)
GPT.init_weights              ← add MoE branch (isinstance check)
GPT.setup_optimizer           ← unchanged: matrix_params captures w1/w2 automatically
GPT.estimate_flops            ← adjust for active experts only
GPT.num_scaling_params        ← add active_transformer_matrices
GPT.update_moe_load_balance() ← NEW: helper to call after each optimizer step
```

### 9.2 File: `core/optim.py`

```
_step_muon                   ← fix second_momentum_buffer shape (3 lines)
```

### 9.3 Weight Initialization

```python
# w1: (num_experts, expert_dim, n_embd) — up-projection
torch.nn.init.uniform_(moe.w1, -s * 0.4, s * 0.4)   # same scale as dense c_fc

# w2: (num_experts, n_embd, expert_dim) — down-projection
torch.nn.init.zeros_(moe.w2)                           # zero-init like dense c_proj

# gate: small init so router starts near-uniform
torch.nn.init.zeros_(moe.gate.weight)

# shared expert: same as dense MLP
torch.nn.init.uniform_(moe.shared_fc.weight,   -s * 0.4, s * 0.4)
torch.nn.init.zeros_(moe.shared_proj.weight)
```

**Tại sao zero-init w2?** Output projections zero-initialized means at init, each expert outputs zero → residual stream is unchanged. Điều này cho phép model học **từ từ** những gì mỗi expert nên làm thay vì bắt đầu với random noise trong residual.

### 9.4 Optimizer Setup — Không Cần Thay Đổi

`setup_optimizer` hiện tại collect `matrix_params = list(self.transformer.h.parameters())`. Điều này automatically captures:
- `w1` và `w2` (3D `nn.Parameter`) → Muon, grouped by shape
- `gate.weight` → Muon (it's a Linear weight in transformer.h)
- `shared_fc.weight`, `shared_proj.weight` → Muon

Grouped by shape: w1 shape `(8, 1024, 768)` is unique → its own Muon group. Same for w2. **No changes needed**.

---

## 10. The Full Mental Model

Hãy tổng kết mental model của một senior researcher nhìn vào MoE:

```
Token arrives at MoE layer
        │
        ▼
   Sigmoid Router              "Does this token need each expert?"
   (independent scores)         Independent, not competitive
        │
        ▼
   Top-k + bias nudge           Routing: use biased scores
   Gate normalization            Weighting: use raw scores
        │
        ▼
   Sort tokens by expert         Batch tokens for GPU efficiency
        │
        ▼
   GroupedMM (bf16)              One fused kernel for all experts
   or Loop fallback              Pure Python, any dtype
        │
        ▼
   Shared expert                 Always runs — provides stable gradient signal
   (all tokens)                  Especially important early in training
        │
        ▼
   Scatter-add back              Weighted combination per token
        │
        ▼
   After optimizer.step():
   update_load_balance()         Nudge router_bias to prevent expert collapse
```

**Key insight** cho ablation: mỗi component trong pipeline trên là một **ablation axis**:
- `moe_top_k`: 1 vs 2 vs 4
- `moe_num_shared`: 0 vs 1
- `moe_num_experts`: 4 vs 8 vs 16
- Router type: sigmoid vs softmax
- Load balancing: bias nudge vs auxiliary loss vs none

---

Bây giờ bạn đã hiểu toàn bộ hệ thống từ first principles. Sẵn sàng implement không? Nếu có, ta bắt đầu với `MoELayer` class — từng method một, với explanation trước khi viết code.