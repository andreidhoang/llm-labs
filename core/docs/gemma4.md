# Gemma 4 Architecture: Deep Analysis & Ablation Plan

> **Senior AI Lead Engineer Analysis**  
> Date: April 2026  
> Source: Maarten Grootendorst's "A Visual Guide to Gemma 4"

---

## Executive Summary

Gemma 4 represents a **systems-first approach** to efficient transformer architectures. Unlike previous models that focused primarily on scaling laws, Gemma 4 optimizes for **inference efficiency** through careful attention mechanisms, **memory hierarchy exploitation** (flash storage vs VRAM), and **multimodal-native design**.

---

## Part 1: Core Architectural Principles

### 1.1 Multi-Scale Strategy

| Model | Architecture | Effective Params | Total Params | Key Innovation |
|-------|-------------|------------------|--------------|----------------|
| **E2B** | Dense + PLE | 2B | ~5B* | Per-Layer Embeddings + Audio |
| **E4B** | Dense + PLE | 4B | ~8B* | Per-Layer Embeddings + Audio |
| **31B** | Dense | 31B | 31B | Wide, shallow efficient attention |
| **26B A4B** | MoE | 4B | 26B | Sparse activation + Shared Expert |

*Total includes lookup table embeddings stored in flash

**Key Insight**: Gemma 4 introduces the concept of **"Effective Parameters" (E)** vs **"Active Parameters" (A)**:
- **E (Effective)**: Parameters loaded in VRAM for computation
- **A (Active)**: Parameters actually used in forward pass
- **Sparse**: Total parameters stored (includes experts not activated)

### 1.2 The Attention Revolution

#### 1.2.1 Interleaving Pattern Evolution

```
Gemma 3 (27B):     [L-L-L-L-G] x 6 + L-L-L-L (ends on local!)
Gemma 4 E2B:       [L-L-L-L-G] x 7 (always ends on global)
Gemma 4 Others:    [L-L-L-L-L-G] x N (5:1 ratio)
```

**Critical Change**: Forcing global attention at the final layer ensures the model has **complete context visibility** for output generation.

#### 1.2.2 Sliding Window Specifications

| Model | Local Window | Pattern | Total Layers |
|-------|-------------|---------|--------------|
| E2B | 512 tokens | 4:1 | 35 |
| E4B | 512 tokens | 5:1 | 45 |
| 26B A4B | 1024 tokens | 5:1 | 60 |
| 31B | 1024 tokens | 5:1 | 60 |

#### 1.2.3 Grouped Query Attention (GQA) Hierarchy

```
LOCAL ATTENTION:        GLOBAL ATTENTION:
┌─────────────────┐     ┌─────────────────┐
│ Q1 Q2 → KV      │     │ Q1-Q8 → KV      │
│ Q3 Q4 → KV      │     │ (8:1 ratio)     │
│ (2:1 ratio)     │     │                 │
│                 │     │ Key dims 2x     │
│ Standard GQA    │     │ Keys = Values   │
└─────────────────┘     │ p-RoPE applied  │
                        └─────────────────┘
```

**Why this matters**: Global attention layers are memory-bound. By using:
- 8:1 GQA (vs 2:1 in local)
- K=V sharing (eliminates V cache entirely)
- Doubled key dimensions (compensate for fewer heads)

...the KV cache for global layers is reduced by **~8x** compared to standard attention.

### 1.3 p-RoPE: Positional Encoding Innovation

**Problem**: Standard RoPE applies rotation to all dimensions. Low-frequency pairs (small rotations):
- Add minimal positional information
- Interfere with semantic content
- Stack up misalignment over long contexts

**Solution (p-RoPE with p=0.25)**:
```python
# Standard RoPE
rope_dims = all_dimensions  # 100%

# p-RoPE (p=0.25)
rope_dims = first_25%_dimensions  # high frequencies only
other_dims = 0  # no rotation, preserve semantics
```

**Impact**: 
- High-freq pairs: Sufficient positional tracking
- Low-freq pairs: Pure semantic representation
- Better long-context coherence

### 1.4 Multimodal Architecture

#### Vision Processing Pipeline

```
Input Image
    ↓
Adaptive Resize (maintains aspect ratio) + Padding
    ↓
ViT Patchify (16x16 patches)
    ↓
2D RoPE (separate w/h positional encoding)
    ↓
Vision Transformer (150M or 550M)
    ↓
3x3 Spatial Pooling (9 patches → 1 embedding)
    ↓
Linear Projection + RMSNorm
    ↓
LLM Input (soft tokens)
```

**Soft Token Budgets**:
| Budget | ~Resolution | Pool Factor | Use Case |
|--------|-------------|-------------|----------|
| 70 | 256x256 | 9x | Fast preview |
| 140 | 384x384 | 9x | Standard |
| 280 | 512x512 | 9x | Detail |
| 560 | 768x768 | 9x | High-res |
| 1120 | 1024x1024 | 9x | Maximum |

#### Audio Processing (E2B/E4B only)

```
Raw Audio
    ↓
Mel-Spectrogram (time x frequency)
    ↓
Chunk Grouping
    ↓
2D Conv Downsampling
    ↓
Conformer Encoder (Transformer + Convolution)
    ↓
Linear Projection
    ↓
LLM Input
```

### 1.5 Per-Layer Embeddings (E-Series)

**The Memory Hierarchy Hack**:

```
Traditional:
┌──────────────┐
│ Token Emb    │ ← 262K × 1536 dims = 384M params in VRAM
│ (one lookup) │
└──────────────┘
        ↓
   [All layers use same embedding]

Gemma 4 PLE:
┌─────────────────────────────────────────┐
│ Layer 0: Token Emb (initial)            │ ← VRAM (2M params)
├─────────────────────────────────────────┤
│ Layer 1-35: Per-Layer Embeddings        │ ← FLASH (262K × 35 × 256 = 2.3B)
│   256 dims each, gating + projection    │    Loaded on-demand
└─────────────────────────────────────────┘
```

**Mechanism**:
1. Pre-computed embeddings per layer stored in flash
2. At inference: load once, gate + project at each layer
3. "Reminds" the model of token identity throughout depth
4. Frees VRAM for actual computation

### 1.6 Mixture of Experts (26B A4B)

```
Token Embedding
    ↓
Router → Softmax over 128 experts
    ↓
Select top-8 experts + 1 shared expert (always active)
    ↓
Weighted sum of expert outputs
    ↓
Next layer
```

**Shared Expert**: 3x larger, always active → general knowledge  
**Routed Experts**: 8 selected per token → specialized knowledge

---

## Part 2: Design Philosophy Analysis

### 2.1 Memory-Aware Architecture

Gemma 4 recognizes the **memory hierarchy**:
```
FLASH (cheap, slow, abundant)
    ↓  Load once at start
VRAM (expensive, fast, limited) ← Optimize for this!
    ↓  Compute
SRAM (very fast, very limited) ← Minimize KV cache here
```

### 2.2 Inference-First Optimization

| Technique | Training Cost | Inference Benefit |
|-----------|---------------|-------------------|
| Interleaved attention | Minimal | ~5x speedup via reduced FLOPs |
| GQA (8:1 global) | Minimal | ~8x KV cache reduction |
| K=V sharing | None | 2x KV cache reduction |
| p-RoPE | Minimal | Better long-context quality |
| Per-Layer Embeddings | Higher | Run on devices with less VRAM |
| MoE | Higher | Large model quality at small model speed |

### 2.3 Multimodal-First Design

Unlike bolted-on multimodality:
- Vision/audio encoders trained **jointly** with LLM
- Projection layers learn to align embedding spaces
- Variable resolution handling (not fixed square crops)

---

## Part 3: Ablation Plan

### Phase 1: Attention Mechanism Ablations (Weeks 1-2)

#### Ablation 1.1: Global-Last Layer Validation
**Hypothesis**: Forcing global attention at final layer improves long-context performance

| Variant | Configuration | Metrics |
|---------|--------------|---------|
| Baseline | Gemma 4 pattern (global last) | Long-context QA, coherence |
| Ablation | Standard interleaving (may end local) | Same |

**Expected**: Baseline shows better long-range dependency modeling

#### Ablation 1.2: GQA Ratio Optimization
**Hypothesis**: 8:1 ratio for global is optimal for memory-quality tradeoff

| Variant | Local GQA | Global GQA |
|---------|-----------|------------|
| A | 2:1 | 8:1 (baseline) |
| B | 2:1 | 4:1 |
| C | 2:1 | 16:1 |
| D | 1:1 (MHA) | 2:1 |

**Metrics**: Perplexity, KV cache size, inference latency

#### Ablation 1.3: K=V Sharing Impact
**Hypothesis**: K=V sharing reduces memory without quality loss

| Variant | Global Attention KV | Key Dim |
|---------|---------------------|---------|
| A | K=V (baseline) | 2x normal |
| B | Separate K,V | 2x normal |
| C | K=V | 1x normal |
| D | Separate K,V | 1x normal |

**Watch**: Perplexity vs KV cache tradeoff

### Phase 2: Positional Encoding Ablations (Weeks 3-4)

#### Ablation 2.1: p-RoPE Pruning Ratio
**Hypothesis**: p=0.25 is optimal; different ratios affect different sequence lengths

| Variant | p-value | RoPE Dimensions |
|---------|---------|-----------------|
| A | 0.25 (baseline) | 25% |
| B | 0.5 | 50% |
| C | 0.125 | 12.5% |
| D | 1.0 (full) | 100% |
| E | 0.0 (none) | 0% (no RoPE) |

**Test**: Performance at 512, 2048, 8192, 32k token lengths

#### Ablation 2.2: 2D RoPE for Vision
**Hypothesis**: 2D RoPE is critical for variable aspect ratios

| Variant | Position Encoding | Aspect Ratio Handling |
|---------|-------------------|----------------------|
| A | 2D RoPE (baseline) | Variable + Padding |
| B | 1D RoPE | Variable + Padding |
| C | 2D RoPE | Fixed square resize |
| D | 1D RoPE | Fixed square resize |

**Metrics**: Image captioning accuracy across different aspect ratios

### Phase 3: Multimodal Architecture Ablations (Weeks 5-6)

#### Ablation 3.1: Vision Encoder Scaling
**Hypothesis**: 550M encoder is optimal for larger models; 150M sufficient for small

| Variant | Encoder Size | Model | Task |
|---------|--------------|-------|------|
| A | 150M | E2B | Visual QA |
| B | 550M | E2B | Visual QA |
| C | 150M | 31B | Visual QA |
| D | 550M | 31B (baseline) | Visual QA |

**Watch**: Quality vs inference cost tradeoff

#### Ablation 3.2: Soft Token Budget
**Hypothesis**: Token budget creates quality-efficiency tradeoff curve

| Budget | Pixels Processed | Tokens to LLM | Downstream Task Perf |
|--------|-----------------|---------------|---------------------|
| 70 | ~230K | 70 | Baseline |
| 140 | ~520K | 140 | +? |
| 280 | ~1.1M | 280 | +? |
| 560 | ~2.3M | 560 | +? |
| 1120 | ~4.6M | 1120 | Peak? |

**Expected**: Diminishing returns after certain budget

#### Ablation 3.3: Audio Encoder Analysis (E-series)
**Hypothesis**: Conformer architecture is optimal for audio

| Variant | Encoder | Parameters |
|---------|---------|------------|
| A | Conformer (baseline) | ~100M |
| B | Standard Transformer | ~100M |
| C | CNN-only | ~50M |
| D | None (text-only baseline) | N/A |

**Tasks**: ASR, audio QA, music understanding

### Phase 4: Efficiency Mechanism Ablations (Weeks 7-8)

#### Ablation 4.1: Per-Layer Embeddings (E-series)
**Hypothesis**: PLE improves quality for fixed compute budget

| Variant | Embedding Strategy | VRAM Usage | Flash Usage |
|---------|-------------------|------------|-------------|
| A | PLE (baseline) | 2M | 2.3B |
| B | Single large embedding | 384M | Minimal |
| C | No layer-specific (like base) | 2M | Minimal |
| D | PLE without gating | 2M | 2.3B |
| E | PLE without projection | 2M | 2.3B |

**Metrics**: Perplexity, VRAM required, inference speed

#### Ablation 4.2: MoE Expert Configuration
**Hypothesis**: 128 experts with top-8 is optimal for 26B A4B

| Variant | Total Experts | Active | Shared Expert | Size |
|---------|---------------|--------|---------------|------|
| A | 128 | 8 (baseline) | Yes, 3x | 26B total |
| B | 64 | 4 | Yes, 3x | 13B total |
| C | 256 | 16 | Yes, 3x | 52B total |
| D | 128 | 8 | No | 24B total |
| E | 128 | 4 | Yes, 3x | 26B total |
| F | Dense equivalent | - | - | 4B |

**Metrics**: Quality vs active parameters, routing stability

#### Ablation 4.3: Sliding Window Size
**Hypothesis**: Smaller windows (512 vs 1024) are sufficient with proper global attention

| Variant | Local Window | Global Freq | Target Context |
|---------|-------------|-------------|----------------|
| A | 512 (E-series) | 4:1 or 5:1 | 4K-32K |
| B | 1024 (large) | 5:1 | 4K-32K |
| C | 256 | 4:1 | 4K-32K |
| D | 2048 | 5:1 | 4K-32K |

**Expected**: Smaller windows with more frequent global ≈ larger windows with less frequent

### Phase 5: Integrated System Ablations (Weeks 9-10)

#### Ablation 5.1: End-to-End Efficiency Comparison
Compare full Gemma 4 stack vs alternatives:

| System | Attention | Pos Enc | Multimodal | Efficiency |
|--------|-----------|---------|------------|------------|
| Gemma 4 | Interleaved + GQA | p-RoPE | Joint training | Baseline |
| Gemma 3 | Interleaved + GQA | Standard RoPE | Vision only | -20%? |
| Llama 3 | Full GQA | RoPE | Vision | -40%? |
| Mistral | SWA only | RoPE | Text only | -? |

#### Ablation 5.2: Scaling Law Validation
Test if "Effective Parameters" concept holds:

| Model | Total Params | Active/Effective | Quality/Param |
|-------|--------------|------------------|---------------|
| E2B | ~5B | 2B | ? |
| E4B | ~8B | 4B | ? |
| 26B A4B | 26B | 4B | ? |
| 31B | 31B | 31B | ? |

**Expected**: E2B and 26B A4B show similar quality/performance despite different total params

### Phase 6: Robustness & Edge Cases (Weeks 11-12)

#### Ablation 6.1: Long Context Stress Test
Test p-RoPE and interleaved attention at extreme lengths:
- 64K tokens
- 128K tokens
- 1M tokens (if supported)

**Watch**: 
- Attention entropy collapse
- Positional misalignment
- Needle-in-haystack retrieval

#### Ablation 6.2: Expert Load Balancing (MoE)
Monitor expert utilization:
```python
# Track over validation set
expert_usage_histogram = count_tokens_per_expert()
cv = coefficient_of_variation(expert_usage_histogram)
# Lower CV = better load balancing
```

#### Ablation 6.3: Multimodal Edge Cases
- Very wide images (panoramas)
- Very tall images (screenshots)
- Low-resolution inputs
- Multi-image reasoning

---

## Part 4: Implementation Priorities

### P0 (Critical Path)
1. **Global-last layer validation** - Core architectural claim
2. **K=V sharing efficacy** - Major memory optimization
3. **p-RoPE ratio sweep** - Novel positional encoding

### P1 (High Impact)
4. **GQA ratio optimization** - Memory/quality tradeoff
5. **PLE mechanism validation** - Key efficiency innovation
6. **MoE expert configuration** - Sparse architecture

### P2 (Validation)
7. **Vision encoder scaling** - Multimodal validation
8. **2D RoPE necessity** - Vision-specific
9. **Soft token budgets** - User-facing feature

### P3 (Research)
10. **Audio encoder alternatives** - Niche use case
11. **Extreme context lengths** - Future-proofing
12. **Cross-architecture comparison** - Positioning

---

## Part 5: Success Metrics

### Efficiency Metrics
- **Tokens/sec** on target hardware (phone, laptop, server)
- **Peak VRAM** usage during inference
- **KV cache size** per layer
- **Flash storage** requirements

### Quality Metrics
- **Perplexity** on held-out text
- **MMLU** (general knowledge)
- **GSM8K** (math reasoning)
- **HumanEval** (code)
- **VQAv2** (visual QA)
- **LibriSpeech** (audio, E-series)
- **Long-context retrieval** (needle-in-haystack)

### Stability Metrics
- **Expert load CV** (MoE only)
- **Gradient norm** during training
- **Attention entropy** at long contexts

---

## Part 6: Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| K=V sharing hurts quality | Medium | High | Validate early in Phase 1 |
| p-RoPE unstable at very long contexts | Medium | Medium | Test in Phase 6 |
| PLE lookup overhead exceeds savings | Low | Medium | Profile inference |
| MoE routing collapse | Low | High | Monitor expert utilization |
| 2D RoPE unnecessary complexity | Medium | Low | Compare to 1D in Phase 2 |

---

## Part 7: Resource Estimates

| Phase | GPU Hours | Storage | Notes |
|-------|-----------|---------|-------|
| Phase 1 | 2,000 | 500GB | Multiple attention variants |
| Phase 2 | 1,500 | 400GB | Position encoding sweeps |
| Phase 3 | 3,000 | 1TB | Multimodal data |
| Phase 4 | 4,000 | 1.5TB | MoE training |
| Phase 5 | 2,000 | 800GB | Integration tests |
| Phase 6 | 1,500 | 600GB | Edge cases |
| **Total** | **14,000** | **4.8TB** | Full ablation study |

---

## Conclusion

Gemma 4 represents a **mature systems-aware architecture** that optimizes for the complete inference pipeline rather than just training loss. The key innovations—interleaved attention with global-last guarantees, p-RoPE for semantic preservation, memory-hierarchy exploitation via PLE, and sparse MoE activation—demonstrate a shift from "bigger is better" to "smarter is better."

The ablation plan prioritizes validation of the **memory-efficiency claims** (K=V, GQA ratios, PLE) and **novel positional encoding** (p-RoPE), as these are the highest-risk, highest-reward architectural decisions.

**Next Steps**:
1. Begin Phase 1.1 (global-last validation) immediately
2. Set up inference benchmarking harness for KV cache measurements
3. Prepare multimodal evaluation suite for Phase 3
4. Allocate cluster resources for MoE ablations (Phase 4)

---

*Analysis completed. Ready for execution review.*
