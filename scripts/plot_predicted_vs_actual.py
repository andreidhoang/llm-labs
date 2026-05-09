"""Predicted vs actual loss calibration plot.

For each completed cell, plot (predicted_loss, actual_loss). Diagonal line = perfect
calibration. Highlights the G3 verification cell separately. Reports Gate D verdict.

Predicted losses come from a Hoffmann L(N,D) fit on the Phase 1 cells; actuals
from the manifest. G3's predicted value should be in dev/LOG.md (preregistered).

Usage:
    python scripts/plot_predicted_vs_actual.py
    python scripts/plot_predicted_vs_actual.py --predicted-g3-loss 0.823

Spec: dev/scaling_law_self_assignment.md §12.2 + §5.3 (Gate D)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _estimate_n_active(arch_config: dict) -> int:
    return 12 * arch_config["num_hidden_layers"] * (arch_config["hidden_size"] ** 2)


def _fit_hoffmann(N_arr, D_arr, L_arr, n_init: int = 50, seed: int = 0):
    """Fit L(N, D) = E + A/N^α + B/D^β with multi-init L-BFGS.

    Returns (E, A, B, alpha, beta, fit_quality).
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("scipy required: pip install scipy")
        return None

    def loss_fn(params, N, D, L):
        E, log_A, log_B, alpha, beta = params
        A = np.exp(log_A)
        B = np.exp(log_B)
        L_pred = E + A / (N ** alpha) + B / (D ** beta)
        L_pred = np.clip(L_pred, 1e-6, None)  # avoid log of negative
        return float(np.sum((np.log(L) - np.log(L_pred)) ** 2))

    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_params = None
    for trial in range(n_init):
        init = [
            rng.uniform(0.5, 1.5),
            rng.uniform(0.0, 5.0),
            rng.uniform(0.0, 5.0),
            rng.uniform(0.2, 0.6),
            rng.uniform(0.2, 0.6),
        ]
        try:
            result = minimize(
                loss_fn, init, args=(N_arr, D_arr, L_arr),
                method="L-BFGS-B",
                bounds=[(0.5, 2.0), (-10, 10), (-10, 10), (0.1, 1.0), (0.1, 1.0)],
            )
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
        except Exception:
            continue

    if best_params is None:
        return None

    E, log_A, log_B, alpha, beta = best_params
    A, B = float(np.exp(log_A)), float(np.exp(log_B))
    return float(E), A, B, float(alpha), float(beta), float(best_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(REPO_ROOT / "runs" / "manifest.json"))
    parser.add_argument("--predicted-g3-loss", type=float, default=None,
                        help="If set, use this as G3 predicted loss (overrides Hoffmann fit)")
    parser.add_argument("--output", default=str(REPO_ROOT / "dev" / "plots" / "predicted_vs_actual.png"))
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
    runs = manifest["runs"]
    fit_runs = [r for r in runs
                if r.get("phase") == "1_fitting" and r.get("status") == "completed"
                and r.get("final_val_loss_joint") is not None]
    g3_runs = [r for r in runs
               if r.get("phase") == "2_verification" and r.get("status") == "completed"
               and r.get("final_val_loss_joint") is not None]

    if len(fit_runs) < 2:
        print(f"Need ≥2 fitting cells; have {len(fit_runs)}.")
        return 1

    # Build arrays for Hoffmann fit
    N_arr = np.array([r.get("n_active_params") or _estimate_n_active(r["architecture_config"])
                      for r in fit_runs])
    D_arr = np.array([r["compute_budget_target_flops"] / (6 * n) for r, n in zip(fit_runs, N_arr)])
    L_arr = np.array([r["final_val_loss_joint"] for r in fit_runs])

    fit_result = _fit_hoffmann(N_arr, D_arr, L_arr)
    if fit_result is None:
        print("Hoffmann fit failed.")
        return 1
    E, A, B, alpha, beta, fit_quality = fit_result
    print(f"Hoffmann fit: L(N, D) = {E:.3f} + {A:.2e}/N^{alpha:.3f} + {B:.2e}/D^{beta:.3f}")
    print(f"Fit log-MSE: {fit_quality:.4f}")

    # Predict for each cell (in-sample; expect tight match)
    predicted_in_sample = E + A / (N_arr ** alpha) + B / (D_arr ** beta)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # In-sample scatter (Phase 1 cells)
    ax.scatter(predicted_in_sample, L_arr, s=100, color="C0", alpha=0.7,
               label=f"Phase 1 fit cells (n={len(fit_runs)})", zorder=5)
    for r, pred in zip(fit_runs, predicted_in_sample):
        ax.annotate(r["cell_id"], xy=(pred, r["final_val_loss_joint"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)

    # G3 prediction + actual (the falsifier)
    g3_predicted = None
    g3_actual = None
    if args.predicted_g3_loss is not None:
        g3_predicted = args.predicted_g3_loss
    elif g3_runs:
        # Compute G3 prediction from fit
        g3 = g3_runs[0]
        g3_N = g3.get("n_active_params") or _estimate_n_active(g3["architecture_config"])
        g3_D = g3["compute_budget_target_flops"] / (6 * g3_N)
        g3_predicted = E + A / (g3_N ** alpha) + B / (g3_D ** beta)

    if g3_runs:
        g3_actual = g3_runs[0]["final_val_loss_joint"]

    if g3_predicted is not None and g3_actual is not None:
        ax.scatter([g3_predicted], [g3_actual], s=300, marker="*", color="red",
                   edgecolors="black", linewidths=2, zorder=6,
                   label=f"G3 verification: pred={g3_predicted:.3f}, actual={g3_actual:.3f}")
        delta = abs(g3_predicted - g3_actual)
        # Gate D verdict
        if delta <= 0.05:
            verdict = "GATE D PASS (fit GOOD: ≤0.05)"
        elif delta <= 0.10:
            verdict = "GATE D PASS (fit ACCEPTABLE: ≤0.10)"
        elif delta <= 0.15:
            verdict = "GATE D MARGINAL (delta in [0.10, 0.15])"
        else:
            verdict = f"GATE D FAIL (delta {delta:.3f} > 0.15)"
        print(f"\nG3 verification: predicted={g3_predicted:.4f}, actual={g3_actual:.4f}, delta={delta:.4f}")
        print(verdict)

    # Diagonal (perfect calibration)
    all_vals = list(predicted_in_sample) + list(L_arr)
    if g3_predicted is not None:
        all_vals.append(g3_predicted)
    if g3_actual is not None:
        all_vals.append(g3_actual)
    lo, hi = min(all_vals) * 0.9, max(all_vals) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="Perfect calibration (y=x)")
    # ±0.1 tolerance band
    ax.fill_between([lo, hi], [lo - 0.1, hi - 0.1], [lo + 0.1, hi + 0.1],
                    color="gray", alpha=0.15, label="±0.1 tolerance (Gate D)")

    ax.set_xlabel("Predicted val_loss/joint", fontsize=12)
    ax.set_ylabel("Actual val_loss/joint", fontsize=12)
    ax.set_title("Predicted vs Actual Loss Calibration", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
