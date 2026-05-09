# 8-Week Frontier GPU Engineer Sprint
## Capstone-First Compression of the 32-Week Plan

**The single deliverable that defines this plan:**
A NVFP4 paged-attention decode kernel for Blackwell B200, integrated as a vLLM attention backend, with measured ≥1.8× throughput improvement over vLLM's current FP16 paged-attention path on Llama-3-70B decode.

**Working assumption:** You can write a tiled tensor-core matmul today hitting ≥50% cuBLAS. If not, take the 32-week plan instead.

**Effort:** 35–45 focused hrs/week × 8 weeks ≈ 300–360 hours.

**Hardware budget:** ~$1,500–$2,500 (B200 spot ~$2.45–$5/hr × ~300 hrs + H100 spot in weeks 2–3).

---

## The capstone in one paragraph (write this on your wall)

You will ship a single Triton+CUDA kernel that runs paged attention on Blackwell B200, with NVFP4 KV-cache and BF16 query/output, beating vLLM's current decode path by ≥1.8× on Llama-3-70B at batch sizes 32–256 with 4K–32K context. The kernel will land as a draft PR to `vllm-project/vllm` as a registered attention backend. The work will be documented as a single technical blog post (≥4,000 words) with locked-clock benchmarks, NCU traces, roofline analysis, and a comparison table against FP16 paged attention, FP8 paged attention (FlashInfer), and FA-4 forward (where applicable). The blog will be published with @-mentions to target lab researchers, and 5–10 specific warm-intro emails will follow within 72 hours.

That paragraph is the spec. Every hour of every week is paying for it.

---

## Working-backward dependency chain

Reading this top-down explains why each week exists:

```
Week 8: Job hunt + interview prep
    ↑ requires
Week 7: Public launch (blog, PR, outreach)
    ↑ requires
Week 6: vLLM integration + end-to-end benchmarks
    ↑ requires
Week 5: NVFP4 quantization + final kernel optimization
    ↑ requires
Week 4: Blackwell port (tcgen05 + TMEM + 2-SM)
    ↑ requires
Week 3: Working FA-3-style attention forward on Hopper
    ↑ requires
Week 2: Triton attention prototype + paged-KV understanding
    ↑ requires
Week 1: Environment + minimal CUDA refresher
```

Read top-down before starting each week — never start a week without understanding what next week's deliverable needs from this week.

---

## Week 1 — Foundation sprint and capstone scoping

**Hours:** ~35
**Hardware:** local 4090/5090 only

### Day 1 (4 hrs)
- Provision: Spheron + Modal accounts, $500 each. Verify B200 spot availability.
- Clone: `cutlass`, `triton`, `flash-attention`, `vllm`, `sglang`, `ThunderKittens`. Build each.
- Read: vLLM's current paged-attention kernel (`csrc/attention/`). Just read — don't modify yet.

### Day 2–3 (10 hrs)
- Skills audit kernel: write `matmul_tc.cu` from scratch, hit ≥80% cuBLAS at FP16 4096³ on H100. If you can't, stop the 8-week plan and revert to 32-week.
- Profile it with NCU. Commit trace + 800-word note: "what's the gap to cuBLAS, and why."

### Day 4–5 (10 hrs)
- Read FA-1 and FA-2 papers, both end-to-end. Take notes.
- Read vLLM PagedAttention paper. Understand the block table data structure cold.
- Read NVFP4 spec section in NVIDIA's Blackwell tuning guide and the NVFP4 microscaling format paper.

### Day 6–7 (10 hrs)
- Write the capstone spec doc — 1 page. Include: hypothesis, falsifiable predictions, scope (what this kernel is and isn't), kill criteria (if benchmark <1.4× by end of week 5, pivot to FP8 W8A16), metric-to-decision mapping.
- Set up GitHub repo `nvfp4-paged-attention` (public, MIT license). Commit empty skeleton + README + spec doc.

### Week 1 exit criteria
- [ ] Capstone spec committed to repo
- [ ] Skills-audit matmul ≥80% cuBLAS, with NCU trace
- [ ] You can describe vLLM's block table layout from memory
- [ ] FA-2 forward algorithm fits on a whiteboard from memory

---

## Week 2 — Triton attention prototype + KV-cache mastery

**Hours:** ~40
**Hardware:** H100 spot (~$2/hr × ~30 hrs ≈ $60)

### Day 1–2 (10 hrs)
- Reproduce Triton's `06-fused-attention.py` tutorial on H100. Benchmark vs `flash-attn` reference. Acceptance: ≥75% of FA-2 reference at seq_len=8192.

### Day 3–4 (12 hrs)
- Modify your Triton attention to be **decode-only** (Q has 1 token, K/V are full cache).
- Add **paged KV-cache** support: take a `block_table` and `block_size=16`, gather K/V pages from non-contiguous HBM. This is the hard part of paged attention; spend the time.
- Validate numerically against reference attention.

### Day 5–6 (12 hrs)
- Read FA-3 and FA-4 blog posts end-to-end, take detailed notes.
- Read vLLM's V1 attention backend API (`vllm/v1/attention/backends/`). Sketch what registering a custom backend will look like in week 6.

### Day 7 (6 hrs)
- Profile your Triton paged-decode kernel with NCU. Identify top-3 stalls. Document.
- Commit progress + benchmark plot to repo.

### Week 2 exit criteria
- [ ] Triton paged-decode attention working on H100, numerically correct
- [ ] At least 65% of FA-2's compute throughput at decode shapes
- [ ] You understand the vLLM V1 attention backend interface well enough to sketch the integration on a whiteboard

---

## Week 3 — Hopper FA-3-style baseline (the stepping stone)

**Hours:** ~40
**Hardware:** H100 spot (~$2/hr × ~40 hrs ≈ $80)

**Why this week exists:** Going straight to Blackwell tcgen05 is too risky. Hopper WGMMA is the closest legible analogue and the FA-3 design is well-documented. Build the H100 version first, then port — same algorithm, different ISA.

### Day 1–3 (16 hrs)
- Implement `fa3_decode_paged.cu` in raw CUDA C++ with WGMMA + TMA + warp specialization on H100.
- Use the producer/consumer pattern from gau-nernst's Hopper writeups.
- Acceptance: matches Triton baseline within 5% (you can iterate further later).

### Day 4–5 (12 hrs)
- Add KV-cache **dequantization fused into the attention loop** — start with FP8 (E4M3) since it's well-supported, NVFP4 will come in week 5.
- Validate: numerical drift vs FP16 reference < 1% on a 1M-token corpus.

### Day 6–7 (12 hrs)
- Profile the Hopper version with NCU. Roofline analysis: are you bandwidth-bound on KV reads or compute-bound on QK^T?
- Lock clocks, take final benchmark numbers, commit traces.

### Week 3 exit criteria
- [ ] Working Hopper paged-decode kernel with FP8 KV cache
- [ ] Numerical correctness validated
- [ ] Roofline analysis written up (will be reused in the final blog post)
- [ ] You're confident enough in the algorithm to port it; the remaining work is ISA and memory hierarchy

---

## Week 4 — Blackwell port (the highest-risk week)

**Hours:** ~45
**Hardware:** B200 spot (~$3/hr × ~45 hrs ≈ $135)

**Why this is the pivotal week:** if you don't have something running on B200 by end of week 4, the rest of the plan cascades. Set a hard kill criterion on Friday of week 4: working but unoptimized on B200, OR pivot to publishing the Hopper version as the capstone (and accept the lower hiring signal).

### Day 1–2 (12 hrs)
- Read Colfax CUTLASS Blackwell tutorials 1 and 2 end-to-end. Reproduce their basic SM100 GEMM example on your B200 instance.
- Read gau-nernst's `tcgen05 for dummies` cover-to-cover.
- Read the FA-4 blog's "TMEM-resident P" section three times.

### Day 3–4 (16 hrs)
- Port your Hopper kernel to Blackwell:
  - Replace `wgmma` with `tcgen05.mma` (single-CTA initially; cta_group::2 in week 5 if time permits).
  - Move accumulator from registers to TMEM via `tcgen05.alloc`.
  - Update mbarrier protocol for the new asynchronous tensor core semantics.
  - Keep TMA loads structurally similar to Hopper.

### Day 5 (8 hrs)
- Get it correct first, fast second. Acceptance: numerical match to Hopper version within FP-noise; performance can be 30% of Hopper's relative throughput at this point.

### Day 6–7 (9 hrs)
- Add 2-SM Pair-UMMA path (`cta_group::2`) for the QK^T matmul. This halves operand-K traffic to SMEM and is the single biggest Blackwell-specific perf lever for attention.
- Re-benchmark. You should be ≥80% of Hopper absolute throughput by end of week.

### Week 4 exit criteria (HARD KILL CRITERION)
- [ ] Kernel runs correctly on B200 with FP8 KV cache
- [ ] Performance is at least 80% of Hopper version's absolute throughput

If kernel does not run correctly on B200 by Sunday night of week 4: pivot the capstone to "Hopper paged-decode kernel for vLLM" and finish weeks 5–8 with that. Lower ceiling but still a strong artifact.

---

## Week 5 — NVFP4 quantization + optimization to baseline-beating

**Hours:** ~45
**Hardware:** B200 spot (~$135 again)

### Day 1–2 (12 hrs)
- Add NVFP4 KV-cache path. NVFP4 = FP4 (E2M1) values + per-block-of-16 FP8 (E4M3) scales + per-tensor FP32 scale. Implement scale calibration against a calibration set (1k tokens of representative text).
- Validate numerical drift vs FP16 reference < 2% on the same 1M-token corpus.

### Day 3–4 (12 hrs)
- Optimization sprint with locked clocks. Iterate on:
  - Tile size (start 128×128, sweep).
  - SMEM staging depth (3 vs 4 vs 5 stages).
  - TMEM allocation pattern.
  - Persistent kernel + cluster launch (if time).
- Acceptance: ≥85% of B200 BF16 SOL on the QK^T matmul.

### Day 5–6 (12 hrs)
- Comparative benchmarks at locked clocks:
  - vs vLLM FP16 paged-attention (your baseline)
  - vs FlashInfer FP8 paged-attention (the harder bar)
  - vs FA-4 forward where shapes allow
- Required win: ≥1.8× over vLLM FP16 baseline at batch sizes 32, 64, 128, 256, contexts 4K, 16K, 32K.

### Day 7 (6 hrs)
- Final profiling pass, commit all NCU traces, lock the kernel. No more optimization after week 5 — week 6 is integration.

### Week 5 exit criteria
- [ ] NVFP4 KV-cache numerically validated
- [ ] ≥1.8× throughput vs vLLM FP16 baseline (locked clocks, multiple batch/context combinations)
- [ ] Full benchmark table ready for blog post
- [ ] You've stopped touching the kernel

---

## Week 6 — vLLM integration + end-to-end benchmarks

**Hours:** ~40
**Hardware:** B200 spot (~$120)

### Day 1–3 (16 hrs)
- Implement vLLM V1 attention backend wrapper around your kernel. Follow the FlashInfer or FlashAttention backend as the template.
- Get Llama-3-70B serving end-to-end through vLLM with your backend.
- This is more annoying than it sounds: you'll fight Python C++ bindings, dtype conversions, the scheduler interface, and probably KV-cache initialization. Budget the time.

### Day 4–5 (12 hrs)
- End-to-end serving benchmarks: `vllm bench serve`. Measure tokens/sec/$ at fixed latency targets (100ms, 250ms, 500ms TTFT).
- Required end-to-end win: ≥1.4× tokens/sec at the same latency target as vLLM's FP16 baseline. (The end-to-end win is always less than the kernel-microbenchmark win because attention isn't the only thing running.)

### Day 6 (6 hrs)
- Open the **draft** PR to `vllm-project/vllm`. Title: "[Backend] NVFP4 paged-attention for Blackwell". Detailed description with benchmark table.
- Don't wait for review. Move on to publishing.

### Day 7 (6 hrs)
- Repo polish: README with benchmark plots, NCU screenshots, install instructions, citation.

### Week 6 exit criteria
- [ ] Working vLLM integration
- [ ] End-to-end benchmark table on Llama-3-70B
- [ ] Draft PR open at `vllm-project/vllm`
- [ ] Repo README is portfolio-quality

---

## Week 7 — Public launch

**Hours:** ~35
**Hardware:** none (writing week)

### Day 1–3 (15 hrs)
- Write the technical blog post. Target ≥4,000 words. Sections:
  1. Problem framing — why NVFP4 paged attention matters in 2026
  2. Algorithm — online softmax + paged KV + low-precision dequant fused
  3. Hopper baseline (week 3 work, briefly)
  4. Blackwell port (the meat — tcgen05, TMEM, 2-SM, mbarriers)
  5. NVFP4 numerical validation
  6. Locked-clock benchmarks (kernel + end-to-end)
  7. Roofline analysis
  8. Open questions / what would make it even faster
- Style: Tri Dao's FA-4 blog as the model. Honest, technical, no marketing fluff.

### Day 4 (6 hrs)
- Address vLLM PR review comments. (You'll get some by now.)

### Day 5 (4 hrs)
- Publish the blog. Cross-post to your domain + dev.to.
- Twitter/X thread with @-mentions of relevant lab researchers (Tri Dao, the FlashInfer team, vLLM maintainers, the Blackwell perf folks at NVIDIA, ThunderKittens authors).

### Day 6–7 (10 hrs)
- Engage with the responses. People will comment, find bugs, suggest improvements. Engaging publicly is itself a hiring signal — it shows you collaborate.

### Week 7 exit criteria
- [ ] Blog published, ≥4,000 words, with all benchmarks and traces inline
- [ ] vLLM PR with active review conversation
- [ ] Twitter/X thread with engagement (replies, RTs from relevant accounts)

---

## Week 8 — Job hunt sprint

**Hours:** ~30 (less than other weeks because applications happen async)

### Day 1 (4 hrs)
- Resume rewrite. **Anthropic's hiring page is explicit:** lead with artifacts, not credentials. The first line under your name should link to the blog post. Then the GitHub repo. Then the merged-or-pending PR. Then employment history. Then education last.
- 1 page only. PDF.

### Day 2 (4 hrs)
- Personal site: front page = "What I shipped." Top of the page is the capstone with a screenshot of the benchmark plot. Everything else below.

### Day 3–5 (15 hrs)
- Targeted outreach. For each target lab:
  - **OpenAI Inference TL** ($460K–$685K base + PPUs) — find 2–3 named ICs from arXiv papers and X posts. Email 3 sentences max: link to blog, ask for 15 minutes.
  - **Anthropic Performance / Inference** — same approach. Look for people who post on X about Claude infra.
  - **xAI CUDA/GPU Kernel** — smaller team, faster process; cold-apply but reference the blog post in the cover line.
  - **Meta GenAI Infra** — referrals matter most here; find Meta engineers who've cited similar work.
- Submit 4–6 applications in parallel for negotiation leverage. Never apply to one at a time.

### Day 6–7 (7 hrs)
- Interview prep, focused:
  - 3× whiteboard FlashAttention forward + backward, untimed → timed at 30 min
  - 3× whiteboard your own kernel (interviewers WILL ask)
  - 5 NCU traces from your repo — practice classifying bottlenecks in <2 min each
  - 1× system design: "design vLLM from scratch on B200 fleet"
  - 30 min of LeetCode mediums to keep coding muscle warm

### Week 8 exit criteria
- [ ] Resume + portfolio site live
- [ ] ≥4 applications submitted with referral or warm intro
- [ ] At least 1 first-round interview scheduled
- [ ] You can defend every line in your blog post under technical questioning

---

## What success and failure look like at week 8

**Success (target outcome):**
- Blog post published with ≥4,000 words, ≥1.8× kernel speedup demonstrated, NCU traces public
- Draft or merged PR at `vllm-project/vllm`
- Repo with ≥50 stars (organic growth from the blog)
- ≥1 inbound recruiter contact from a target lab
- 1–3 first-round interviews scheduled
- The artifact alone justifies the OpenAI Inference / Anthropic Performance phone screen

**Acceptable degraded outcome (if Blackwell port failed in week 4):**
- Hopper-only paged attention with FP8 KV cache, vLLM-integrated
- Same blog post structure, just no Blackwell section
- Still strong; targets shift toward Senior Kernel Engineer (~$500K–$700K TC) rather than Inference TL (~$800K–$1.4M TC)

**Failure (do not apply yet):**
- No working kernel on B200 by week 5
- Kernel works but benchmarks <1.3× vs vLLM baseline
- No vLLM PR opened
→ Extend by 4–6 weeks, finish the work, then apply. Applying with a half-finished capstone is worse than not applying.

---

## The single discipline that makes this plan work

End-of-day 3-sentence journal: *what I tested today, the result, what I'm doing tomorrow.* Six days a week. No exceptions.

This sounds trivial. It's the difference between shipping in 8 weeks and not shipping at all. The 32-week plan can survive sloppy execution; the 8-week plan cannot. The journal forces the daily reckoning that catches drift before it compounds.

If you skip three days of journal entries in a row, you have already failed week N. Stop, audit honestly, and either restart at the current week or downshift to the 32-week plan. There is no middle path.

---

## Reading list, drastically pruned

**Required (read in this order, before week 4):**
1. FlashAttention 2 paper (Dao)
2. PagedAttention / vLLM paper (Kwon)
3. FlashAttention 4 blog (Dao, March 2026) — the central reference for the whole plan
4. gau-nernst, *tcgen05 for dummies* — clearest tutorial available
5. Colfax Research, CUTLASS Blackwell tutorial part 1 + 2 — TMEM and Pair-UMMA
6. NVFP4 / microscaling format spec (NVIDIA Blackwell tuning guide)

**Skim only if blocked:**
- PMPP chapters relevant to your specific bug
- CUDA Programming Guide async copy chapter
- vLLM V1 architecture docs (when starting week 6)

**Explicitly do not read during the 8 weeks:**
Anything about distributed training, RL, MoE training, custom silicon, ROCm port. Out of scope. Comes back in a follow-on phase if interested.

---

## Why this plan exists

The 32-week plan optimizes for a complete, well-rounded GPU engineer. The 8-week plan optimizes for a single, irrefutable hiring signal that lands you in the OpenAI Inference / Anthropic Performance interview pipeline. They are different products for different starting points. Pick the one whose assumptions match your reality. Don't pick neither.
