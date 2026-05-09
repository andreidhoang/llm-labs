"""Generate a Markdown report from wall-clock benchmark outputs.

Usage:
    python -m scripts.bench_report
    python -m scripts.bench_report runs/bench/phase_0_baseline.json \
                                  runs/bench/phase_1_chunked_ce.json

By default this reads runs/bench/phase_*.json, enriches the report with
per-step JSONL files when present, and writes runs/bench/report.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH_DIR = REPO_ROOT / "runs" / "bench"

METRICS = [
    ("step_time_ms", "step ms", "lower"),
    ("fwd_bwd_ms", "fwd+bwd ms", "lower"),
    ("optim_step_ms", "optim ms (approx comm)", "lower"),
    ("optim_pct_of_step", "optim %", "lower"),
    ("tokens_per_sec_per_gpu", "tok/sec/GPU", "higher"),
    ("mfu_pct", "MFU %", "higher"),
    ("peak_hbm_gb", "peak HBM GB", "lower"),
    ("compile_first_step_overhead_s", "compile overhead s", "lower"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Markdown report for runs/bench")
    p.add_argument("phase_json", nargs="*", type=Path,
                   help="phase JSON files. Defaults to <bench-dir>/phase_*.json")
    p.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH_DIR,
                   help="directory containing _meta.json and phase JSON/JSONL files")
    p.add_argument("--out", type=Path, default=DEFAULT_BENCH_DIR / "report.md",
                   help="Markdown output path")
    p.add_argument("--title", type=str, default="Wallclock Benchmark Report")
    p.add_argument("--no-plots", action="store_true",
                   help="skip optional PNG plot generation")
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A VM can die mid-write; keep all complete rows before it.
                break
    return rows


def phase_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    parts = stem.split("_")
    for part in parts:
        if part.isdigit():
            return int(part), stem
    return 999, stem


def discover_phase_jsons(args: argparse.Namespace) -> list[Path]:
    paths = args.phase_json or sorted(args.bench_dir.glob("phase_*.json"), key=phase_sort_key)
    return [p for p in paths if p.name != "_meta.json" and p.exists()]


def fmt_value(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        return f"{value:.{digits}f}"
    return str(value)


def pct_delta(value: Any, baseline: Any, direction: str) -> str:
    if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)):
        return ""
    if baseline == 0 or value == baseline:
        return ""
    raw = (value - baseline) / baseline * 100.0
    good = raw > 0 if direction == "higher" else raw < 0
    marker = "better" if good else "worse"
    return f" ({raw:+.1f}%, {marker})"


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(md_escape(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_escape(c) for c in row) + " |")
    return "\n".join(lines)


def first_meta(meta_path: Path, phases: list[dict[str, Any]]) -> dict[str, Any]:
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            return meta.get("latest_run") or meta.get("first_run") or {}
        except Exception:
            pass
    for phase in phases:
        if isinstance(phase.get("meta"), dict) and phase["meta"]:
            return phase["meta"]
    return {}


def config_flags(phase: dict[str, Any]) -> str:
    flags: list[str] = []
    compile_mode = phase.get("compile_mode")
    if compile_mode and compile_mode != "off":
        suffix = "+fullgraph" if phase.get("compile_fullgraph") else ""
        flags.append(f"compile={compile_mode}{suffix}")
    if phase.get("chunked_ce"):
        flags.append("chunked_ce")
    if phase.get("activation_ckpt"):
        flags.append("act_ckpt")
    if phase.get("fp8"):
        flags.append("fp8")
    return ", ".join(flags) or "(none)"


def step_values(phase: dict[str, Any], jsonl_rows: list[dict[str, Any]], key: str) -> list[float]:
    if jsonl_rows:
        values = [row.get(key) for row in jsonl_rows]
        return [float(v) for v in values if isinstance(v, (int, float))]
    summary_key = {
        "step_ms": "per_step_step_ms",
        "fwdbwd_ms": "per_step_fwdbwd_ms",
        "optim_ms": "per_step_optim_ms",
    }.get(key)
    values = phase.get(summary_key or "", [])
    return [float(v) for v in values if isinstance(v, (int, float))]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return ordered[idx]


def stability_row(phase: dict[str, Any], jsonl_rows: list[dict[str, Any]]) -> list[str]:
    steps = step_values(phase, jsonl_rows, "step_ms")
    optim = step_values(phase, jsonl_rows, "optim_ms")
    hbm = [float(row["peak_hbm_gb"]) for row in jsonl_rows
           if isinstance(row.get("peak_hbm_gb"), (int, float))]
    stdev = statistics.stdev(steps) if len(steps) > 1 else None
    return [
        phase["phase"],
        fmt_value(len(steps)),
        fmt_value(percentile(steps, 0.50)),
        fmt_value(percentile(steps, 0.90)),
        fmt_value(stdev),
        fmt_value(percentile(optim, 0.50)),
        fmt_value(max(hbm) if hbm else phase.get("metrics", {}).get("peak_hbm_gb")),
    ]


def metric_table(phases: list[dict[str, Any]]) -> str:
    headers = ["metric"] + [p["phase"] for p in phases]
    baseline = phases[0].get("metrics", {}) if phases else {}
    rows: list[list[str]] = []
    for key, label, direction in METRICS:
        cells = [label]
        for i, phase in enumerate(phases):
            metrics = phase.get("metrics", {})
            value = metrics.get(key)
            cell = fmt_value(value)
            if i > 0:
                cell += pct_delta(value, baseline.get(key), direction)
            cells.append(cell)
        rows.append(cells)
    return md_table(headers, rows)


def config_table(phases: list[dict[str, Any]]) -> str:
    rows = []
    for phase in phases:
        rows.append([
            phase["phase"],
            fmt_value(phase.get("n_gpu")),
            phase.get("gpu_name", "-"),
            fmt_value(phase.get("device_batch_size")),
            fmt_value(phase.get("grad_accum_steps")),
            fmt_value(phase.get("max_seq_len")),
            config_flags(phase),
        ])
    return md_table(
        ["phase", "GPUs", "GPU", "DBS", "accum", "seq", "flags"],
        rows,
    )


def best_line(phases: list[dict[str, Any]], metric: str, label: str, higher: bool) -> str:
    candidates = [
        (p, p.get("metrics", {}).get(metric))
        for p in phases
        if isinstance(p.get("metrics", {}).get(metric), (int, float))
    ]
    if not candidates:
        return f"- {label}: n/a"
    phase, value = max(candidates, key=lambda x: x[1]) if higher else min(candidates, key=lambda x: x[1])
    return f"- {label}: `{phase['phase']}` at {fmt_value(value)}"


def maybe_write_plots(
    phases: list[dict[str, Any]],
    jsonl_by_phase: dict[str, list[dict[str, Any]]],
    out_path: Path,
    no_plots: bool,
) -> list[Path]:
    if no_plots:
        return []
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    plot_dir = out_path.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for key, ylabel, filename in [
        ("step_ms", "step time (ms)", "step_time.png"),
        ("peak_hbm_gb", "peak HBM (GB)", "peak_hbm.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        has_data = False
        for phase in phases:
            rows = jsonl_by_phase.get(phase["phase"], [])
            values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
            if not values:
                continue
            ax.plot(range(len(values)), values, marker="o", linewidth=1.4, label=phase["phase"])
            has_data = True
        if not has_data:
            plt.close(fig)
            continue
        ax.set_xlabel("measured step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = plot_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def render_report(
    phases: list[dict[str, Any]],
    jsonl_by_phase: dict[str, list[dict[str, Any]]],
    meta: dict[str, Any],
    title: str,
    out_path: Path,
    plot_paths: list[Path],
) -> str:
    generated = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    sha = meta.get("git_sha", "")
    sha_short = sha[:8] if sha else "-"
    dirty = "dirty" if meta.get("git_dirty") else "clean"
    context_rows = [
        ["generated", generated],
        ["git", f"{meta.get('git_branch', '-')}/{sha_short} ({dirty})"],
        ["host", meta.get("host", "-")],
        ["platform", meta.get("platform", "-")],
        ["torch/CUDA/NCCL", f"{meta.get('torch_version', '-')}/{meta.get('torch_cuda_version', '-')}/{meta.get('nccl_version', '-')}"],
        ["GPU", f"{meta.get('gpu_count', '-') } x {meta.get('gpu_name', '-')} ({meta.get('gpu_capability', '-')})"],
        ["driver", meta.get("driver_version", "-")],
    ]

    stability_rows = [
        stability_row(phase, jsonl_by_phase.get(phase["phase"], []))
        for phase in phases
    ]

    lines = [
        f"# {title}",
        "",
        "## Run Context",
        md_table(["field", "value"], context_rows),
        "",
        "## Phase Configs",
        config_table(phases),
        "",
        "## Summary Metrics",
        "Deltas are relative to the first phase listed.",
        "",
        metric_table(phases),
        "",
        "## Stability",
        md_table(
            ["phase", "steps", "step p50 ms", "step p90 ms", "step stdev ms", "optim p50 ms", "max HBM GB"],
            stability_rows,
        ),
        "",
        "## Best Observed",
        best_line(phases, "step_time_ms", "fastest median step", higher=False),
        best_line(phases, "tokens_per_sec_per_gpu", "highest throughput per GPU", higher=True),
        best_line(phases, "mfu_pct", "highest MFU", higher=True),
        best_line(phases, "peak_hbm_gb", "lowest peak HBM", higher=False),
    ]

    if plot_paths:
        lines.extend(["", "## Plots"])
        for path in plot_paths:
            rel = path.relative_to(out_path.parent)
            label = path.stem.replace("_", " ")
            lines.append(f"![{label}]({rel.as_posix()})")

    lines.extend([
        "",
        "## Artifacts",
        md_table(
            ["artifact", "path"],
            [
                ["metadata", "_meta.json"],
                ["phase summaries", "phase_*.json"],
                ["per-step streams", "phase_*.jsonl"],
                ["report", out_path.name],
            ],
        ),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    phase_paths = discover_phase_jsons(args)
    if not phase_paths:
        raise SystemExit(f"no phase JSON files found in {args.bench_dir}")

    phases = [load_json(path) for path in phase_paths]
    for phase, path in zip(phases, phase_paths):
        phase.setdefault("phase", path.stem)

    jsonl_by_phase = {
        phase["phase"]: load_jsonl(path.with_suffix(".jsonl"))
        for phase, path in zip(phases, phase_paths)
    }
    meta = first_meta(args.bench_dir / "_meta.json", phases)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_paths = maybe_write_plots(phases, jsonl_by_phase, args.out, args.no_plots)
    report = render_report(phases, jsonl_by_phase, meta, args.title, args.out, plot_paths)
    args.out.write_text(report)
    print(f"wrote {args.out}")
    if plot_paths:
        for path in plot_paths:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
