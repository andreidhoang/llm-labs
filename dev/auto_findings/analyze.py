"""
analyze.py — generate progress.png from an autoresearch session's results.tsv.

Karpathy's autoresearch repo has analysis.ipynb + progress.png. This is the
script equivalent: takes a results.tsv (5 columns: commit, val_bpb, memory_gb,
status, description), outputs a PNG showing:
  - val_bpb of each experiment as a scatter (color-coded by status)
  - the rolling-minimum "best so far" curve
  - annotation of the best experiment

Usage:
    python dev/auto_findings/analyze.py \
        --tsv dev/auto_findings/raw/2026-05-17_results.tsv \
        --out dev/auto_findings/plots/2026-05-17.png

Re-runnable. Pure function: TSV -> PNG. Same input -> same output.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_session(tsv_path: Path) -> pd.DataFrame:
    """Load TSV, drop session-markers / crashes, compute best-so-far curve."""
    df = pd.read_csv(tsv_path, sep="\t")
    # Drop rows where val_bpb == 0 (crashes and session-markers).
    # These aren't real experiments — they break the cumulative-min line.
    df = df[df["val_bpb"] > 0].reset_index(drop=True)
    df["experiment"] = range(1, len(df) + 1)
    df["best_so_far"] = df["val_bpb"].cummin()
    df["improvement"] = df["best_so_far"].shift(1) - df["best_so_far"]
    df["improvement"] = df["improvement"].fillna(0).clip(lower=0)
    return df


def plot_session(df: pd.DataFrame, out_png: Path, title: str = None) -> None:
    """One-PNG summary of an autoresearch session."""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Scatter: each experiment, colored by status.
    # Filled markers ('o', 's') use white edgecolor for contrast against the line;
    # unfilled markers ('x') need linewidths only — matplotlib ignores edgecolor on them.
    color_map = {"keep": "#2ca02c", "discard": "#d62728", "crash": "#7f7f7f"}
    marker_map = {"keep": "o", "discard": "x", "crash": "s"}
    for status, group in df.groupby("status"):
        marker = marker_map.get(status, "o")
        kwargs = dict(
            c=color_map.get(status, "#666"),
            marker=marker,
            s=100, alpha=0.9,
            label=f"{status} (n={len(group)})",
            zorder=3,
        )
        if marker == "x":
            kwargs["linewidths"] = 2.5  # thick stroke so x is visible
        else:
            kwargs["edgecolors"] = "white"
            kwargs["linewidths"] = 0.7
        ax.scatter(group["experiment"], group["val_bpb"], **kwargs)

    # Line: best-so-far (the record curve). This is the actual "progress".
    ax.plot(
        df["experiment"], df["best_so_far"],
        color="#1f77b4", linewidth=2.5, label="best so far",
        zorder=2,
    )

    # Annotate winning experiment.
    best_idx = df["val_bpb"].idxmin()
    best = df.loc[best_idx]
    desc = best["description"]
    if len(desc) > 50:
        desc = desc[:47] + "..."
    ax.annotate(
        f"BEST: val_bpb={best['val_bpb']:.4f}\n"
        f"commit {best['commit']}\n"
        f"{desc}",
        xy=(best["experiment"], best["val_bpb"]),
        xytext=(20, -40), textcoords="offset points",
        fontsize=9, color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#1f77b4", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#1f77b4"),
    )

    # Reference line: Karpathy's published d=8 baseline (val_bpb ~0.998).
    ax.axhline(0.998, color="#aaa", linestyle="--", linewidth=1, alpha=0.7,
               label="Karpathy d=8 published (~0.998)")

    ax.set_xlabel("experiment #", fontsize=11)
    ax.set_ylabel("val_bpb (lower = better)", fontsize=11)
    ax.set_title(title or f"autoresearch progress — {out_png.stem}", fontsize=12, pad=12)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.3, zorder=1)

    # Headline numbers in the corner.
    summary = (
        f"experiments: {len(df)}\n"
        f"baseline:    {df.iloc[0]['val_bpb']:.4f}\n"
        f"best:        {df['val_bpb'].min():.4f}\n"
        f"improvement: {df.iloc[0]['val_bpb'] - df['val_bpb'].min():.4f} "
        f"({100*(df.iloc[0]['val_bpb'] - df['val_bpb'].min())/df.iloc[0]['val_bpb']:.1f}%)"
    )
    ax.text(
        0.02, 0.02, summary,
        transform=ax.transAxes, fontsize=9, family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#999"),
    )

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] saved {out_png}")


def plot_cross_session(tsv_paths: list[Path], out_png: Path) -> None:
    """Master view: best val_bpb across all sessions over time."""
    fig, ax = plt.subplots(figsize=(11, 5))

    all_records = []
    for tsv_path in sorted(tsv_paths):
        tag = tsv_path.stem.replace("_results", "")
        df = load_session(tsv_path)
        ax.plot(df["experiment"], df["best_so_far"], linewidth=2,
                marker="o", markersize=4, label=f"{tag} (best={df['val_bpb'].min():.4f})")
        all_records.append((tag, df["val_bpb"].min(), len(df)))

    ax.axhline(0.998, color="#aaa", linestyle="--", linewidth=1, alpha=0.7,
               label="Karpathy d=8 published")

    ax.set_xlabel("experiment # (per session)", fontsize=11)
    ax.set_ylabel("val_bpb best-so-far", fontsize=11)
    ax.set_title("autoresearch progress across sessions", fontsize=12, pad=12)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] saved {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", type=Path, help="path to one session's results.tsv")
    p.add_argument("--out", type=Path, help="output PNG path")
    p.add_argument("--title", default=None, help="plot title")
    p.add_argument("--all", action="store_true",
                   help="generate cross-session progress.png from dev/auto_findings/raw/*.tsv")
    args = p.parse_args()

    if args.all:
        raw_dir = Path("dev/auto_findings/raw")
        tsv_paths = list(raw_dir.glob("*_results.tsv"))
        if not tsv_paths:
            raise SystemExit(f"no results.tsv files in {raw_dir}")
        out = args.out or Path("dev/auto_findings/progress.png")
        plot_cross_session(tsv_paths, out)
    else:
        if not args.tsv or not args.out:
            raise SystemExit("--tsv and --out required (or use --all)")
        df = load_session(args.tsv)
        plot_session(df, args.out, args.title)
