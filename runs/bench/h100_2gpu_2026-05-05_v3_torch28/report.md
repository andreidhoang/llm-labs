# Wallclock Benchmark Report

## Run Context
| field | value |
| --- | --- |
| generated | 2026-05-04 20:18:15 UTC |
| git | /- (clean) |
| host | 6e269f0d6c49 |
| platform | Linux-6.8.0-106-generic-x86_64-with-glibc2.39 |
| torch/CUDA/NCCL | 2.8.0a0+5228986c39.nv25.06/12.9/2.27.3 |
| GPU | 2 x NVIDIA H100 80GB HBM3 (sm_90) |
| driver | 535.288.01 |

## Phase Configs
| phase | GPUs | GPU | DBS | accum | seq | flags |
| --- | --- | --- | --- | --- | --- | --- |
| phase_0_baseline | 2 | NVIDIA H100 80GB HBM3 | 8 | 2 | 2,048 | compile=default |
| phase_1_chunked_ce | 2 | NVIDIA H100 80GB HBM3 | 8 | 2 | 2,048 | compile=default |
| phase_3_act_ckpt | 2 | NVIDIA H100 80GB HBM3 | 8 | 2 | 2,048 | compile=default |
| phase_4_fp8 | 2 | NVIDIA H100 80GB HBM3 | 8 | 2 | 2,048 | compile=default, fp8 |

## Summary Metrics
Deltas are relative to the first phase listed.

| metric | phase_0_baseline | phase_1_chunked_ce | phase_3_act_ckpt | phase_4_fp8 |
| --- | --- | --- | --- | --- |
| step ms | 224.94 | 224.44 (-0.2%, better) | 444.22 (+97.5%, worse) | 571.25 (+154.0%, worse) |
| fwd+bwd ms | 201.80 | 201.76 (-0.0%, better) | 425.16 (+110.7%, worse) | 551.53 (+173.3%, worse) |
| optim ms (approx comm) | 19.01 | 19.04 (+0.2%, worse) | 18.99 (-0.1%, better) | 19.03 (+0.1%, worse) |
| optim % | 8.45 | 8.48 (+0.4%, worse) | 4.27 (-49.5%, better) | 3.33 (-60.6%, better) |
| tok/sec/GPU | 145,677.8 | 145,998.5 (+0.2%, better) | 73,764.6 (-49.4%, worse) | 57,362.1 (-60.6%, worse) |
| MFU % | 11.82 | 11.85 (+0.3%, better) | 5.99 (-49.3%, worse) | 4.66 (-60.6%, worse) |
| peak HBM GB | 20.89 | 15.94 (-23.7%, better) | 9.50 (-54.5%, better) | 9.51 (-54.5%, better) |
| compile overhead s | 25.75 | 30.50 (+18.4%, worse) | 4.94 (-80.8%, better) | 5.03 (-80.5%, better) |

## Stability
| phase | steps | step p50 ms | step p90 ms | step stdev ms | optim p50 ms | max HBM GB |
| --- | --- | --- | --- | --- | --- | --- |
| phase_0_baseline | 20 | 223.42 | 263.23 | 41.40 | 19.00 | 20.89 |
| phase_1_chunked_ce | 20 | 224.40 | 246.98 | 34.43 | 19.03 | 15.94 |
| phase_3_act_ckpt | 20 | 444.11 | 471.87 | 30.78 | 18.98 | 9.50 |
| phase_4_fp8 | 20 | 571.01 | 643.41 | 57.74 | 19.01 | 9.51 |

## Best Observed
- fastest median step: `phase_1_chunked_ce` at 224.44
- highest throughput per GPU: `phase_1_chunked_ce` at 145,998.5
- highest MFU: `phase_1_chunked_ce` at 11.85
- lowest peak HBM: `phase_3_act_ckpt` at 9.50

## Artifacts
| artifact | path |
| --- | --- |
| metadata | _meta.json |
| phase summaries | phase_*.json |
| per-step streams | phase_*.jsonl |
| report | report.md |
