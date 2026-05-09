"""IsoFLOP curves: loss vs N at each compute scale.

Reads runs/manifest.json + plots one U-curve per compute scale, marking the
empirical N* (minimum-loss point) at each. Saves to dev/plots/isoflops_curves.png.

Usage:
    python scripts/plot_isoflops.py
    python scripts/plot_isoflops.py --manifest runs/manifest.json --output dev/plots/iso.png

Spec: dev/scaling_law_self_assignment.md §12.2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _estimate_n_active(arch_config: dict) -> int:
    """12 × n_layer × hidden_size² formula (per spec §3)."""
    return 12 * arch_config["num_hidden_layers"] * (arch_config["hidden_size"] ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(REPO_ROOT / "runs" / "manifest.json"))
    parser.add_argument("--output", default=str(REPO_ROOT / "dev" / "plots" / "isoflops_curves.png"))
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return 1

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}; nothing to plot.")
        return 1

    manifest = json.loads(manifest_path.read_text())
    fit_runs = [r for r in manifest["runs"]
                if r.get("phase") == "1_fitting" and r.get("status") == "completed"
                and r.get("final_val_loss_joint") is not None]

    if len(fit_runs) < 2:
        print(f"Only {len(fit_runs)} completed fitting runs; need ≥2 to plot IsoFLOPs.")
        return 1

    # Group by compute scale
    by_C = defaultdict(list)
    for r in fit_runs:
        C = r["compute_budget_target_flops"]
        N = r.get("n_active_params") or _estimate_n_active(r["architecture_config"])
        by_C[C].append({
            "N": N,
            "loss": r["final_val_loss_joint"],
            "depth": r["architecture_config"]["num_hidden_layers"],
            "cell_id": r["cell_id"],
        })

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(by_C)))

    for color, C in zip(colors, sorted(by_C)):
        cells = sorted(by_C[C], key=lambda c: c["N"])
        Ns = [c["N"] for c in cells]
        losses = [c["loss"] for c in cells]
        depths = [c["depth"] for c in cells]

        # Plot line
        ax.plot(Ns, losses, "o-", color=color, label=f"C = {C:.0e}", linewidth=2, markersize=8)

        # Mark the minimum (empirical N*)
        min_idx = int(np.argmin(losses))
        ax.scatter([Ns[min_idx]], [losses[min_idx]], s=200, color=color,
                   edgecolors="black", linewidths=2, zorder=5,
                   label=f"  N*(C={C:.0e}) = {Ns[min_idx]:.1e} @ d{depths[min_idx]}")

    ax.set_xscale("log")
    ax.set_xlabel("N (active params)", fontsize=12)
    ax.set_ylabel("val_loss/joint", fontsize=12)
    ax.set_title("IsoFLOP curves: loss vs N at each compute scale", fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, which="both")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # Endpoint-min warnings (per spec §3.2)
    print("\nEndpoint check:")
    for C in sorted(by_C):
        cells = sorted(by_C[C], key=lambda c: c["N"])
        if len(cells) < 3:
            print(f"  C={C:.0e}: only {len(cells)} widths — too few for U-curve")
            continue
        min_idx = int(np.argmin([c["loss"] for c in cells]))
        if min_idx in (0, len(cells) - 1):
            print(f"  ⚠ C={C:.0e} minimum at endpoint d{cells[min_idx]['depth']} — "
                  f"add expansion cell on the {'small' if min_idx == 0 else 'large'}-N side")
        else:
            print(f"  ✓ C={C:.0e} minimum interior at d{cells[min_idx]['depth']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
