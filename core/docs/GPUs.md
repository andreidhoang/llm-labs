Good question to frame carefully, because the answer changed materially in 2024–2026 and most "CUDA tutorials" online are still teaching the Volta/Ampere mental model. Here's what actually matters now if you're targeting B200/B300 (and forward-compatible to Rubin H2 2026).

## The 2026 hardware reality you're programming against

Blackwell Ultra (B300) is the working frontier today: 160 SMs, 20,480 CUDA cores, 5th-gen Tensor Cores supporting FP8, FP6, and the new NVFP4 format, with 256 KB of Tensor Memory (TMEM) per SM (40 MB per GPU), 288 GB HBM3e at 8 TB/s, and PCIe Gen6. The key architectural shift versus Hopper is that warp-synchronous MMA (wgmma) is replaced by tcgen05.mma — a single-thread, fully asynchronous instruction that reads operands from SMEM or TMEM and writes accumulators directly to TMEM, bypassing registers entirely. If you understand nothing else about the new hardware, understand that one sentence — it reorders almost everything about how you write a kernel.

## The Pareto-ranked skill stack

**Tier 1 — Master these first; they are 80% of real performance.**

**1. The async, tile-based mental model.** Modern GPUs are dataflow machines, not SIMT machines. The unit of thought is a tile moving through a software pipeline of (TMA load) → (async MMA) → (epilogue), synchronized by mbarriers, with warp-specialized producers and consumers. On Blackwell the tensor cores operate fully asynchronously, so you issue multiple tcgen05.mma without waiting, using multiple mbarrier objects to wait on different MMA stages independently. If you cannot draw the producer/consumer pipeline of your kernel on a whiteboard with mbarrier arrivals labeled, you don't understand it yet. This single concept subsumes warp specialization, double/multi-buffering, K-stage pipelining, and TMA multicast.

**2. Memory hierarchy and roofline thinking.** HBM (8 TB/s) → L2 → SMEM (~256 KB/SM) → TMEM (256 KB/SM, accumulator-only) → registers. Every kernel decision is justified by where bytes live and which roof you're under. Compute the arithmetic intensity *before* writing code: a GEMM at large M,N,K is compute-bound (push toward NVFP4 to widen the roof); an attention decode is bandwidth-bound (KV-cache layout dominates); a Norm or activation is launch-latency-bound (fuse it). The "lock clocks, run roofline, then optimize" discipline (`nvidia-smi --lock-gpu-clocks=tdp,tdp`) is non-negotiable; you cannot reason about performance on a thermally throttling GPU.

**3. Profiling-driven iteration with Nsight.** Nsight Systems for the timeline (are SMs idle? is there a kernel-launch gap? is comm overlapping compute?), then Nsight Compute for the offending kernel (compute throughput, memory throughput, warp stalls, occupancy, source-attributed sampling). The frontier-lab discipline is: never optimize without a profile, never claim a speedup without locked clocks, never accept a kernel that doesn't beat `torch.compile` by ≥2× — below that bar, the maintenance cost of a hand kernel isn't worth it.

**4. Pick the right abstraction layer; don't drop too low.** The 2026 stack:
- **cuBLAS / cuDNN / FlashAttention-4** — your floor. Never reinvent. FA-4 hits 1605 TFLOPs/s on B200 BF16 (71% utilization), 1.3× cuDNN 9.13, 2.7× Triton. If your kernel competes with FA-4 on attention, you'd better have a very specific reason.
- **Triton** — your everyday kernel language for fused ops, custom attention variants, MoE routing kernels. Performance-portable, ~80% of peak with 5% of the lines.
- **CUTLASS / CuTe** — production-grade kernels where you need the last 20%. The CuTe layout algebra is the single most important DSL to learn; once you internalize "shapes and strides" thinking, everything else becomes obvious.
- **ThunderKittens** — fast iteration on novel kernels (linear attention, state-space, exotic fusions). As of TK 2.0 (January 2026) it has full Blackwell support including MXFP8 and NVFP4 and is used in production at Together, Jump Trading, and Cursor.
- **Raw PTX / inline ASM** — only for the last 1%. Practically: tcgen05 PTX intrinsics for kernels CUTLASS doesn't yet expose cleanly.

**5. FlashAttention internals, deeply.** Attention is the central kernel of frontier models, and reading/rewriting FA is the best single exercise for kernel skill. Master online softmax, IO-aware tiling, and the FA-4 ideas: software-emulated exp via polynomial approximation on FMA units to dodge the MUFU.EX2 bottleneck, P stored in TMEM to relieve SMEM traffic, and 2-CTA MMA mode that partitions accumulators across a CTA pair to halve operand-B traffic on the backward pass. These are not attention tricks — they are the new playbook for any compute-heavy kernel on Blackwell.

**6. Low-precision numerics.** NVFP4 combines FP8 (E4M3) micro-block scaling on 16-value blocks with FP32 tensor-level scaling, targeting near-FP8 accuracy at ~1.8× storage savings vs FP8 and ~3.5× vs FP16. You need to understand: where each format works (FP8 for training matmuls, NVFP4/MXFP8 mostly for inference today, BF16 still the safe default for training-loop master copies), how scale calibration is done (per-tensor, per-channel, per-block, dynamic vs static), and how to *measure* numerical drift (cosine similarity vs FP32 reference, NaN/Inf monitors, gradient-norm comparisons across precisions). Going to FP4 in training is still research-grade; going to FP4 in inference is table stakes.

**7. Distributed/collective programming.** A single kernel matters less than overlapping it with comm. Master: NCCL primitives (all-reduce, reduce-scatter, all-gather), compute-comm overlap via streams and CUDA graphs, the topology you're actually running on (NVLink5 1.8 TB/s intra-node, NVSwitch, Quantum-X800/Spectrum-X 800 Gb/s inter-node), and the symmetric-memory primitives (NVSHMEM, distributed shared memory across cluster). For training: FSDP2 vs TP vs EP vs PP and when each pays off. For inference: disaggregated prefill/decode, KV-cache paging, speculative decoding.

**Tier 2 — the next 15% of value.**

Blackwell-specifics you'll reach for once Tier 1 is solid: TMEM allocation/deallocation discipline (tcgen05.alloc/dealloc, the per-warp ¼ access restriction), 2-SM Pair-UMMA via `cta_group::2`, thread-block cluster launches and DSMEM, TMA multicast, cluster launch control. PyTorch internals: torch.compile / Inductor, custom ops with proper meta/abstract registrations, autograd correctness for custom kernels. Forensic debugging: cuda-memcheck/compute-sanitizer, NaN propagation tracing, silent-data-corruption (SDC) detection.

**Tier 3 — last 5%, only if you're shipping kernels into a production library.**

Raw PTX/SASS reading, MLIR/Triton-IR if you're contributing to compilers, GPU microarchitectural quirks (the SFU bottleneck on B200 vs the 2× SFU on B300, scheduler issue rates, TMEM bank layout), and deep CUTLASS internals (epilogue visitor trees, collective builder customization).

## What to *not* spend time on

- **Hand-rolled CUDA C++ for ops that have a library implementation.** It is almost never worth it. cuBLAS, cuDNN, FA-4, and CUTLASS are written by people whose entire job for years was that one kernel.
- **Optimizing the wrong thing.** Roofline first. A 2× speedup on a kernel taking 3% of step time is a 0.6% wall-clock win that you'll spend two weeks on.
- **Old patterns that no longer apply.** Anything taught with `__syncthreads`-heavy designs, or that puts MMA accumulators in registers, or that uses `wmma` instead of `wgmma`/`tcgen05.mma` — that's pre-Hopper and is actively worse on Blackwell.
- **Gaming-Blackwell tutorials.** sm_120 (consumer 5090) and sm_100 (data-center B200/B300) are different ISAs in important ways; tutorials must specify which. Most "Blackwell programming" content on the open web is sm_120 and won't have tcgen05 or TMEM at all.

## The minimum learning loop, concretely

If you have a quarter to invest, the highest-EV path is:

1. Two weeks: read the CUDA Programming Guide async chapter + the CUTLASS/CuTe tutorial series (Colfax Research has the cleanest writeups), and write a Hopper WGMMA matmul that hits 90% of cuBLAS.
2. Two weeks: port that kernel to Blackwell tcgen05.mma + 2-SM Pair-UMMA, hit 95%+ of cuBLAS at FP16, then add NVFP4.
3. Three weeks: rewrite FA-2 in Triton, then study the FA-4 blog and reproduce the forward pass on B200 with TMEM-resident P. This single exercise teaches you 80% of frontier kernel craft.
4. Three weeks: a real distributed exercise — a flash-decoding kernel for paged KV cache with NCCL compute-comm overlap, profiled end-to-end on a multi-node B300 setup.
5. Continuous: read every Tri Dao / Together AI / Hazy Research / NVIDIA tech blog post the day it drops; profile your assumptions monthly.

The meta-point: top-1% kernel engineers don't differ from average ones in the size of their CUDA vocabulary. They differ in (a) ruthless profiler discipline before optimizing, (b) taste in what *not* to write from scratch, and (c) treating numerical correctness as a first-class deliverable equal to perf. Internalize those three and the specific PTX instructions become a lookup-table problem.