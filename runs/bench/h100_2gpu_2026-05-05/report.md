# Wallclock Benchmark Report

## Run Context
| field | value |
| --- | --- |
| generated | 2026-05-04 18:15:02 UTC |
| git | /- (clean) |
| host | 96fc32fa1fbf |
| platform | Linux-6.8.0-106-generic-x86_64-with-glibc2.35 |
| torch/CUDA/NCCL | 2.6.0+cu124/12.4/2.21.5 |
| GPU | 2 x NVIDIA H100 80GB HBM3 (sm_90) |
| driver | 535.288.01 |

## Phase Configs
| phase | GPUs | GPU | DBS | accum | seq | flags |
| --- | --- | --- | --- | --- | --- | --- |
| phase_0_baseline | 2 | NVIDIA H100 80GB HBM3 | 4 | 2 | 1,024 | (none) |
| phase_1_chunked_ce | 2 | NVIDIA H100 80GB HBM3 | 4 | 2 | 1,024 | (none) |
| phase_2_compile | 2 | NVIDIA H100 80GB HBM3 | 4 | 2 | 1,024 | compile=default |
| phase_3_act_ckpt | 2 | NVIDIA H100 80GB HBM3 | 4 | 2 | 1,024 | (none) |

## Summary Metrics
Deltas are relative to the first phase listed.

| metric | phase_0_baseline | phase_1_chunked_ce | phase_2_compile | phase_3_act_ckpt |
| --- | --- | --- | --- | --- |
| step ms | 104.79 | 119.21 (+13.8%, worse) | 101.37 (-3.3%, better) | 185.29 (+76.8%, worse) |
| fwd+bwd ms | 94.29 | 101.82 (+8.0%, worse) | 88.73 (-5.9%, better) | 168.28 (+78.5%, worse) |
| optim ms (approx comm) | 10.49 | 16.70 (+59.2%, worse) | 12.54 (+19.5%, worse) | 16.68 (+59.0%, worse) |
| optim % | 10.01 | 14.01 (+40.0%, worse) | 12.37 (+23.6%, worse) | 9.00 (-10.1%, better) |
| tok/sec/GPU | 78,175.0 | 68,718.5 (-12.1%, worse) | 80,814.8 (+3.4%, better) | 44,210.8 (-43.4%, worse) |
| MFU % | 2.34 | 2.06 (-12.0%, worse) | 2.42 (+3.4%, better) | 1.32 (-43.6%, worse) |
| peak HBM GB | 5.19 | 4.25 (-18.0%, better) | 3.55 (-31.6%, better) | 2.99 (-42.4%, better) |
| compile overhead s | 0.69 | 0.71 (+2.9%, worse) | 25.84 (+3644.9%, worse) | 0.78 (+13.0%, worse) |

## Stability
| phase | steps | step p50 ms | step p90 ms | step stdev ms | optim p50 ms | max HBM GB |
| --- | --- | --- | --- | --- | --- | --- |
| phase_0_baseline | 15 | 104.79 | 106.98 | 3.88 | 10.49 | 5.19 |
| phase_1_chunked_ce | 15 | 119.21 | 122.52 | 4.15 | 16.70 | 4.25 |
| phase_2_compile | 15 | 101.37 | 106.19 | 3.90 | 12.54 | 3.55 |
| phase_3_act_ckpt | 15 | 185.29 | 189.39 | 4.66 | 16.68 | 2.99 |

## Best Observed
- fastest median step: `phase_2_compile` at 101.37
- highest throughput per GPU: `phase_2_compile` at 80,814.8
- highest MFU: `phase_2_compile` at 2.42
- lowest peak HBM: `phase_3_act_ckpt` at 2.99

## Artifacts
| artifact | path |
| --- | --- |
| metadata | _meta.json |
| phase summaries | phase_*.json |
| per-step streams | phase_*.jsonl |
| report | report.md |
