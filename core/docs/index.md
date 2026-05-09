# docs/ index

Quick map of every file here — what it is, whether it reflects current code, and when to read it.

---

## Codebase — how to run things

| file | what it covers |
|------|---------------|
| `running_and_testing_guide.md` | Running and smoke-testing the speculative decoding engine in `core/engine.py`. ⚠️ References `return_hidden_states` (removed) and 3B/300M configs (replaced by D12–D26). |
| `v2_optimization_runbook.md` | Step-by-step instructions for nanochat's v2 tier (chunked CE, activation checkpointing, `torch.compile` fullgraph, rowwise FP8, CommStream). v2 is **not yet merged** into `llm/core/`. |

---

## Architecture deep dives

| file | what it covers | status |
|------|---------------|--------|
| `speculative_decoding_deep_dive.md` | Full engineering spec for `SpeculativeEngine`: theory, KV cache design, rejection sampling, smear gate approximation, perf model. ⚠️ Model sizes reference outdated 3B/300M — current codebase uses D12–D26. | current logic, stale sizes |
| `nanochat_perf_engineering.md` | nanochat internals top-to-bottom: Flash Attention switching, FP8, Muon+AdamW, distributed training, KV cache, v2 bottleneck analysis with exact numbers. | current |
| `moe.md` | MoE from first principles: dense→sparse design, tensor traces, iso-FLOP sizing, load balancing via bias nudging, Muon integration. Research direction, not in codebase. | research |

---

## Research directions (not in codebase)

| file | what it covers |
|------|---------------|
| `gemma4.md` | Gemma 4 architectural audit + 12-week ablation plan: multi-scale sliding windows, GQA hierarchy, p-RoPE, per-layer embeddings, MoE interleaving. |
| `multimodal.md` | Engineering spec for a Qwen3-VL-class vision-language model: vision backbone, DeepStack fusion, MRoPE, three-stage training pipeline. |

---

## Background & reference

| file | what it covers |
|------|---------------|
| `nanochat_vi.md` | Vietnamese-language analysis of nanochat's training run history and leaderboard (Run 1–6, 168 hrs → 1.65 hrs). Good for historical context on what each architectural change bought. |
| `GPUs.md` | Pareto-ranked skill stack for Blackwell GPU kernel programming (2026): async tile model, memory hierarchy, profiling discipline, abstraction layers. |
| `nnvidia-engineer.md` | Full-stack model systems career framework: data → tokenization → architecture → training → post-training → inference → multimodal. Uses Nemotron 3 as the study object. |
| `8-week-frontier-gpu-sprint.md` | 8-week capstone plan for shipping an NVFP4 paged-attention decode kernel on Blackwell B200, integrated into vLLM, targeting ≥1.8× throughput over FP16 on Llama-3-70B. |

---

## Reading order by goal

**"I want to run a training job"**
→ `nanochat_perf_engineering.md` (understand what you're running) → `v2_optimization_runbook.md` (once v2 is merged)

**"I want to understand or extend SpeculativeEngine"**
→ `speculative_decoding_deep_dive.md` → `running_and_testing_guide.md`
→ Note: model sizes in both docs are stale — use D12/D20/D24 from `core/configs.py`

**"I want to add MoE or multimodal"**
→ `moe.md` or `multimodal.md` (research specs, no code yet)

**"I want to study the architecture for a research angle"**
→ `gemma4.md` (ablation methodology) → `nanochat_perf_engineering.md` (implementation details)

**"I want to get better at GPU kernels"**
→ `GPUs.md` → `8-week-frontier-gpu-sprint.md`

---

## Known staleness

- `running_and_testing_guide.md` — `return_hidden_states` parameter no longer exists in `model.py`; model names `TARGET_3B` / `DRAFT_300M` / `BASELINE_125M` replaced by `D12`–`D26` in `core/configs.py`
- `speculative_decoding_deep_dive.md` — same config rename applies
- `v2_optimization_runbook.md` — references `nanochat.v2.gpt_v2`; `llm/core/v2/` does not exist yet
