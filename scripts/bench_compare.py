"""Print a comparison table from bench JSON outputs.

Usage:
    python -m scripts.bench_compare runs/bench/phase_0_baseline.json \
                                    runs/bench/phase_1_chunked_ce.json ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


METRICS = [
    ("step_time_ms",            "step ms",           "↓"),
    ("fwd_bwd_ms",              "fwd+bwd ms",        "↓"),
    ("optim_step_ms",           "optim ms (≈comm)",  "↓"),
    ("optim_pct_of_step",       "optim %",           "↓"),
    ("tokens_per_sec_per_gpu",  "tok/sec/GPU",       "↑"),
    ("mfu_pct",                 "MFU %",             "↑"),
    ("peak_hbm_gb",             "peak HBM GB",       "↓"),
]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    rows = []
    for p in sys.argv[1:]:
        path = Path(p)
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            continue
        with path.open() as f:
            rows.append(json.load(f))

    if not rows:
        sys.exit(2)

    headers = ["metric"] + [r["phase"] for r in rows]
    widths = [max(len(h), 24) for h in headers]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for key, label, direction in METRICS:
        cells = [f"{label} ({direction})"]
        baseline = rows[0]["metrics"].get(key)
        for i, r in enumerate(rows):
            v = r["metrics"].get(key, "—")
            if isinstance(v, (int, float)) and isinstance(baseline, (int, float)) and i > 0 and baseline != 0:
                delta = (v - baseline) / baseline * 100
                cell = f"{v:>10.2f}  ({delta:+.1f}%)"
            elif isinstance(v, (int, float)):
                cell = f"{v:>10.2f}"
            else:
                cell = str(v)
            cells.append(cell)
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    print()
    print("Configs:")
    for r in rows:
        flags = []
        if r.get("compile_mode") not in (None, "off"):
            flags.append(f"compile={r['compile_mode']}{'+fullgraph' if r.get('compile_fullgraph') else ''}")
        if r.get("chunked_ce"): flags.append("chunked_ce")
        if r.get("activation_ckpt"): flags.append("act_ckpt")
        if r.get("fp8"): flags.append("fp8")
        print(f"  {r['phase']:30s}: {', '.join(flags) or '(none)'}")


if __name__ == "__main__":
    main()
