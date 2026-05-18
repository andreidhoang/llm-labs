# Lessons across autoresearch sessions

Durable findings the next session's agent should read **before** proposing
experiments. If you propose something listed here as `DO NOT RETRY` without a
specific reason it would differ this time, you're wasting compute.

Referenced from `auto/program.md` § "Prior session findings."

## Format

```
### YYYY-MM-DD (session auto/YYYY-MM-DD)
- Tried: <one-line description of the change>
- Result: <val_bpb delta vs baseline; or "crash" or "no signal">
- Lesson: <what we now believe>
- Status: KEEP | RETRY-WITH-CONDITIONS | DO-NOT-RETRY
- Optional: bears on H_X (Tier 2 hypothesis) → see dev/sweep_design.md
```

## Entries

### 2026-05-18 (session auto/2026-05-18)

- LESSON: WARMUP_RATIO=0 at d=8 hurts (+0.025). Keep ≥0.05 across small models.
  Status: DO-NOT-RETRY at d=8 with MATRIX_LR≥0.04.
- LESSON: TOTAL_BATCH_SIZE=2^19 overshoots at d=8 (+0.037 regression). 2^18 near-optimal.
  Status: DO-NOT-RETRY at d=8. [bears on H_3]
- LESSON: MATRIX_LR=0.06 marginally regresses from 0.04 at d=8 (+0.002).
  Status: 0.04 is near-optimal for d=8 dense; testing 0.05 may find slight improvement.
- LESSON: WD=0.2, EMBED=0.8, β1=0.8, SSSL window each produce small marginal wins
  (~0.001 each) on top of session 1 winner. HP space approaching saturation in
  this direction; next session should escalate to architectural variants.
- LESSON: Cross-session replication of session 1 iter4 was exact (Δ=0.0006).
  System is reproducible to noise floor.
- LESSON: Cross-session best so far = 1.0576 (iter9 of session 2). Down 5.7% from
  Karpathy's published 0.998 baseline; gap explained by FA2 vs FA3.
