"""Scaling law fit plot: log-log scatter of (C, N_opt) + fitted power law +
extrapolation to G3 + bootstrap uncertainty band.

Reads runs/manifest.json. Saves to dev/plots/scaling_law_N.png and scaling_law_D.png.

Usage:
    python scripts/plot_scaling_law.py
    python scripts/plot_scaling_law.py --target-c 6e19 --output-dir dev/plots/

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
    return 12 * arch_config["num_hidden_layers"] * (arch_config["hidden_size"] ** 2)


def _fit_power_law(C_arr, N_arr):
    """Returns slope a, intercept k, R²."""
    log_C = np.log(C_arr)
    log_N = np.log(N_arr)
    a, log_k = np.polyfit(log_C, log_N, deg=1)
    k = np.exp(log_k)
    fitted = a * log_C + log_k
    residuals = log_N - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((log_N - log_N.mean()) ** 2))
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, k, R2, residuals


def _bootstrap_slope(C_arr, N_arr, n_iter=1000, seed=0):
    rng = np.random.default_rng(seed)
    log_C = np.log(C_arr)
    log_N = np.log(N_arr)
    n = len(C_arr)
    slopes = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        if len(set(idx.tolist())) >= 2:
            a_boot, _ = np.polyfit(log_C[idx], log_N[idx], deg=1)
            slopes.append(a_boot)
    return slopes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(REPO_ROOT / "runs" / "manifest.json"))
    parser.add_argument("--target-c", type=float, default=6e19, help="G3 compute scale for extrapolation marker")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "dev" / "plots"))
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
        print(f"No manifest at {manifest_path}")
        return 1

    manifest = json.loads(manifest_path.read_text())
    fit_runs = [r for r in manifest["runs"]
                if r.get("phase") == "1_fitting" and r.get("status") == "completed"
                and r.get("final_val_loss_joint") is not None]

    # Find min-N per compute scale
    by_C = defaultdict(list)
    for r in fit_runs:
        C = r["compute_budget_target_flops"]
        N = r.get("n_active_params") or _estimate_n_active(r["architecture_config"])
        by_C[C].append((N, r["final_val_loss_joint"], r["architecture_config"]["num_hidden_layers"]))

    optimal_points = []
    for C in sorted(by_C):
        cells = sorted(by_C[C], key=lambda x: x[1])  # sort by loss (ascending)
        N_opt, loss_opt, depth_opt = cells[0]
        optimal_points.append((C, N_opt, loss_opt, depth_opt))

    if len(optimal_points) < 2:
        print(f"Only {len(optimal_points)} compute scales; need ≥2 to fit a power law.")
        return 1

    C_arr = np.array([p[0] for p in optimal_points])
    N_arr = np.array([p[1] for p in optimal_points])
    D_arr = C_arr / (6 * N_arr)
    L_arr = np.array([p[2] for p in optimal_points])

    # Fit power law on N
    a_N, k_N, R2_N, residuals_N = _fit_power_law(C_arr, N_arr)
    a_D, k_D, R2_D, residuals_D = _fit_power_law(C_arr, D_arr)

    # Bootstrap CI on slope (informal)
    slopes_N = _bootstrap_slope(C_arr, N_arr)
    if slopes_N:
        slope_ci_low = np.percentile(slopes_N, 5)
        slope_ci_high = np.percentile(slopes_N, 95)
    else:
        slope_ci_low = slope_ci_high = a_N

    print(f"N_opt(C) = {k_N:.4e} × C^{a_N:.4f}")
    print(f"D_opt(C) = {k_D:.4e} × C^{a_D:.4f}")
    print(f"Slope a (N): {a_N:.4f}, informal range from bootstrap [{slope_ci_low:.4f}, {slope_ci_high:.4f}]")
    print(f"R² (N fit): {R2_N:.4f}")
    print(f"Sanity: a + b = {a_N + a_D:.4f} (should be 1.0)")

    # ─── Plot 1: scaling_law_N.png ───
    fig, ax = plt.subplots(figsize=(9, 6))
    # Data points
    ax.scatter(C_arr, N_arr, s=120, color="C0", zorder=5, label="(C, N_opt) data")
    # Annotate each point
    for C, N, L, d in optimal_points:
        ax.annotate(f"d{d}\nL={L:.3f}", xy=(C, N), xytext=(8, 8),
                    textcoords="offset points", fontsize=9)

    # Fitted line, extrapolated to target
    C_smooth = np.logspace(np.log10(C_arr.min() * 0.5), np.log10(args.target_c * 2), 100)
    N_fit = k_N * (C_smooth ** a_N)
    ax.plot(C_smooth, N_fit, "-", color="C0", alpha=0.7,
            label=f"Fit: N = {k_N:.2e} × C^{a_N:.3f} (R²={R2_N:.3f})")

    # Informal CI band from bootstrap
    if slopes_N:
        N_low = k_N * (C_smooth ** slope_ci_low)
        N_high = k_N * (C_smooth ** slope_ci_high)
        ax.fill_between(C_smooth, N_low, N_high, color="C0", alpha=0.15,
                        label=f"Informal slope range [{slope_ci_low:.3f}, {slope_ci_high:.3f}]")

    # G3 extrapolation marker
    N_pred = k_N * (args.target_c ** a_N)
    ax.scatter([args.target_c], [N_pred], s=200, marker="*", color="red", zorder=6,
               label=f"G3 prediction: N = {N_pred:.2e} @ C={args.target_c:.0e}")
    ax.axvline(args.target_c, color="red", linestyle="--", alpha=0.3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("C (FLOPs)", fontsize=12)
    ax.set_ylabel("N_opt (active params)", fontsize=12)
    ax.set_title(f"Scaling law: N_opt vs C  (slope a={a_N:.3f})", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_law_N.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'scaling_law_N.png'}")

    # ─── Plot 2: scaling_law_D.png ───
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(C_arr, D_arr, s=120, color="C1", zorder=5, label="(C, D_opt) data")

    D_smooth = k_D * (C_smooth ** a_D)
    ax.plot(C_smooth, D_smooth, "-", color="C1", alpha=0.7,
            label=f"Fit: D = {k_D:.2e} × C^{a_D:.3f} (R²={R2_D:.3f})")

    D_pred = args.target_c / (6 * N_pred)
    ax.scatter([args.target_c], [D_pred], s=200, marker="*", color="red", zorder=6,
               label=f"G3 prediction: D = {D_pred:.2e} @ C={args.target_c:.0e}")
    ax.axvline(args.target_c, color="red", linestyle="--", alpha=0.3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("C (FLOPs)", fontsize=12)
    ax.set_ylabel("D_opt (training tokens)", fontsize=12)
    ax.set_title(f"Scaling law: D_opt vs C  (slope b={a_D:.3f})", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_dir / "scaling_law_D.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'scaling_law_D.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
