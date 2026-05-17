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

(none yet — first session writes here)
