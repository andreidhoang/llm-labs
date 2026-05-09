"""Minimal local "training API" for the scaling-law self-assignment.

Replaces Stanford's hyperturing API for our 8xH100 vast.ai setup. Reads cell
configs from configs/scaling_law/*.json, submits via torchrun, tracks
wall-clock budget in runs/manifest.json, parses final val loss from training
output.

CLI:
    python scripts/sweep_runner.py submit --config configs/scaling_law/F1_s.json
    python scripts/sweep_runner.py status
    python scripts/sweep_runner.py budget
    python scripts/sweep_runner.py fit
    python scripts/sweep_runner.py dry-run --config configs/scaling_law/F1_s.json

Spec: dev/scaling_law_self_assignment.md §6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "runs" / "manifest.json"
DEFAULT_TOTAL_BUDGET_HOURS = 20.0  # ~$320 at vast.ai $16/hr 8xH100


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {
            "total_budget_hours": DEFAULT_TOTAL_BUDGET_HOURS,
            "used_hours": 0.0,
            "runs": [],
        }
    return json.loads(MANIFEST.read_text())


def save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2))
    tmp.replace(MANIFEST)


def build_torchrun_cmd(config: dict) -> list[str]:
    """Translate a JSON cell config to a torchrun base_train.py invocation."""
    arch = config["architecture_config"]
    moe = config["moe_config"]
    opt = config["optimizer_config"]
    mm = config.get("multimodal_config", {})

    cmd = [
        "torchrun",
        "--nproc-per-node=8",
        "scripts/base_train.py",
        f"--depth={arch['num_hidden_layers']}",
        f"--head-dim={arch['head_dim']}",
        f"--num-experts={moe['num_experts']}",
        f"--top-k={moe['top_k']}",
        f"--num-shared-experts={moe['num_shared_experts']}",
        f"--max-seq-len={config['max_seq_len']}",
        f"--matrix-lr={opt['matrix_lr']}",
        f"--embedding-lr={opt['embedding_lr']}",
        f"--unembedding-lr={opt['unembedding_lr']}",
        f"--scalar-lr={opt['scalar_lr']}",
        f"--weight-decay={opt['weight_decay']}",
        f"--warmup-frac={opt['warmup_frac']}",
        f"--final-lr-frac={opt['final_lr_frac']}",
        f"--run={config['cell_id']}",
    ]
    # Compute budget: prefer target-flops, else target-param-data-ratio
    if "compute_budget_target_flops" in config:
        cmd.append(f"--target-flops={config['compute_budget_target_flops']}")
    if "target_param_data_ratio" in config:
        cmd.append(f"--target-param-data-ratio={config['target_param_data_ratio']}")
    if mm.get("enabled"):
        cmd.append("--multimodal")
        cmd.append(f"--mix-ratio={mm['mix_ratio_r']}")
    return cmd


# Match nanochat's actual output formats:
#   "Step 14000 | Validation bpb: 0.715432"
#   "Minimum validation bpb: 0.715000"
#   "final val_loss/joint: 0.823" (multimodal — if base_train.py is updated to print this)
_VAL_BPB_RE = re.compile(r"validation\s+bpb\s*[:=]\s*([0-9.]+)", re.IGNORECASE)
_MIN_VAL_BPB_RE = re.compile(r"minimum\s+validation\s+bpb\s*[:=]\s*([0-9.]+)", re.IGNORECASE)
_VAL_LOSS_JOINT_RE = re.compile(r"val[_/-]?loss[/_-]joint\s*[:=]\s*([0-9.]+)", re.IGNORECASE)

# Param/FLOPs parsers — base_train.py prints lines like:
#   "active_total              : 245,123,456"
#   "total                     : 280,000,000"
#   "Estimated FLOPs per token: 1.234e+09"
_ACTIVE_PARAMS_RE = re.compile(r"^active_total\s*[:=]\s*([\d,]+)", re.IGNORECASE | re.MULTILINE)
_TOTAL_PARAMS_RE = re.compile(r"^total\s*[:=]\s*([\d,]+)", re.IGNORECASE | re.MULTILINE)
_FLOPS_PER_TOKEN_RE = re.compile(r"FLOPs\s+per\s+token\s*[:=]\s*([\d.e+\-]+)", re.IGNORECASE)
_NUM_ITERS_RE = re.compile(r"Calculated\s+number\s+of\s+iterations.*?[:=]\s*([\d,]+)", re.IGNORECASE)
_TOTAL_TOKENS_RE = re.compile(r"Total\s+training\s+FLOPs\s+estimate\s*[:=]\s*([\d.e+\-]+)", re.IGNORECASE)


def parse_final_val_loss(stdout: str) -> float | None:
    """Extract final validation loss from training stdout.

    Tries (in order):
      1. val_loss/joint (multimodal — only if base_train.py prints it)
      2. Minimum validation bpb (nanochat preferred metric)
      3. Last "Validation bpb" line (fallback)
    Returns None if none found.
    """
    # Prefer joint loss if printed (multimodal)
    matches = _VAL_LOSS_JOINT_RE.findall(stdout)
    if matches:
        return float(matches[-1])
    # Then minimum val bpb (nanochat reports this at end of training)
    matches = _MIN_VAL_BPB_RE.findall(stdout)
    if matches:
        return float(matches[-1])
    # Last fallback: most recent val bpb line
    matches = _VAL_BPB_RE.findall(stdout)
    if matches:
        return float(matches[-1])
    return None


def _parse_int_with_commas(s: str) -> int:
    return int(s.replace(",", ""))


def parse_n_active_params(stdout: str) -> int | None:
    """Extract active_total parameter count from base_train.py's 'Parameter counts' block."""
    matches = _ACTIVE_PARAMS_RE.findall(stdout)
    return _parse_int_with_commas(matches[-1]) if matches else None


def parse_n_total_params(stdout: str) -> int | None:
    """Extract total parameter count."""
    matches = _TOTAL_PARAMS_RE.findall(stdout)
    return _parse_int_with_commas(matches[-1]) if matches else None


def parse_flops_per_token(stdout: str) -> float | None:
    """Extract estimated FLOPs per token (from print 'Estimated FLOPs per token: X')."""
    matches = _FLOPS_PER_TOKEN_RE.findall(stdout)
    return float(matches[-1]) if matches else None


def parse_num_iterations(stdout: str) -> int | None:
    """Extract num_iterations (from print 'Calculated number of iterations from ... : X')."""
    matches = _NUM_ITERS_RE.findall(stdout)
    return _parse_int_with_commas(matches[-1]) if matches else None


def submit(config_path: Path, dry_run: bool = False) -> dict | None:
    """Submit one cell. Refuses if it would exceed budget."""
    config = json.loads(config_path.read_text())
    estimated_hours = config["max_runtime_seconds"] / 3600

    m = load_manifest()
    remaining = m["total_budget_hours"] - m["used_hours"]
    if estimated_hours > remaining:
        print(
            f"REFUSED: {config['cell_id']} estimated {estimated_hours:.2f}hr exceeds "
            f"remaining {remaining:.2f}hr (used {m['used_hours']:.2f}/{m['total_budget_hours']:.2f})"
        )
        return None

    cmd = build_torchrun_cmd(config)
    print(f"SUBMIT: {config['cell_id']} (estimated {estimated_hours:.2f} hr)")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        print("  [dry-run: not actually executing]")
        return None

    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=int(config["max_runtime_seconds"] + 300),
    )
    elapsed_seconds = time.time() - start
    elapsed_hours = elapsed_seconds / 3600

    # Parse all metrics from training output (only if cell completed successfully)
    if proc.returncode == 0:
        final_loss = parse_final_val_loss(proc.stdout)
        n_active = parse_n_active_params(proc.stdout)
        n_total = parse_n_total_params(proc.stdout)
        flops_per_tok = parse_flops_per_token(proc.stdout)
        num_iters = parse_num_iterations(proc.stdout)
    else:
        final_loss = n_active = n_total = flops_per_tok = num_iters = None

    record = {
        "id": len(m["runs"]) + 1,
        "cell_id": config["cell_id"],
        "phase": config.get("phase", "unknown"),
        "config_file": str(config_path),
        "compute_budget_target_flops": config.get("compute_budget_target_flops"),
        "n_active_params": n_active,
        "n_total_params": n_total,
        "flops_per_token_estimated": flops_per_tok,
        "num_iterations": num_iters,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": elapsed_seconds,
        "wall_clock_hours": elapsed_hours,
        "final_val_loss_joint": final_loss,
        "exit_code": proc.returncode,
        "status": "completed" if proc.returncode == 0 else "failed",
        "architecture_config": config["architecture_config"],
    }
    m["runs"].append(record)
    m["used_hours"] += elapsed_hours
    save_manifest(m)

    status = "DONE" if proc.returncode == 0 else "FAILED"
    print(f"{status}: {config['cell_id']} in {elapsed_hours:.2f} hr, final_loss={final_loss}")
    print(f"Budget remaining: {m['total_budget_hours'] - m['used_hours']:.2f} hr")
    return record


def status() -> None:
    m = load_manifest()
    used = m["used_hours"]
    total = m["total_budget_hours"]
    print(f"Budget: {used:.2f} / {total:.2f} hr used ({100*used/total:.1f}%)")
    print(f"Runs completed: {len([r for r in m['runs'] if r['status'] == 'completed'])}")
    for r in m["runs"]:
        loss = r.get("final_val_loss_joint")
        loss_str = f"loss={loss:.4f}" if loss is not None else "loss=N/A"
        print(f"  [{r['id']}] {r['cell_id']}: {r['status']} ({r['wall_clock_hours']:.2f} hr, {loss_str})")


def budget() -> None:
    m = load_manifest()
    print(json.dumps({
        "used_hours": m["used_hours"],
        "remaining_hours": m["total_budget_hours"] - m["used_hours"],
        "total_hours": m["total_budget_hours"],
    }, indent=2))


def fit() -> None:
    """Run the IsoFLOPs power-law fit on completed Phase 1 cells."""
    import numpy as np

    m = load_manifest()
    p1 = [r for r in m["runs"] if r.get("phase") == "1_fitting" and r["status"] == "completed"]
    if not p1:
        print("No completed Phase 1 cells.")
        return

    by_C: dict[float, list[dict]] = {}
    for r in p1:
        by_C.setdefault(r["compute_budget_target_flops"], []).append(r)

    optimal_points = []
    for C in sorted(by_C):
        cells = sorted(
            by_C[C],
            key=lambda c: c.get("n_active_params") or _estimate_n(c["architecture_config"]),
        )
        if len(cells) < 3:
            print(f"WARNING: C={C:.0e} has only {len(cells)} cells; frontier-style fit expects >=3 widths.")

        best_idx, best = min(enumerate(cells), key=lambda item: item[1]["final_val_loss_joint"])
        if best_idx == 0 or best_idx == len(cells) - 1:
            depth = best["architecture_config"]["num_hidden_layers"]
            print(
                f"WARNING: C={C:.0e} minimum is at endpoint depth d{depth}. "
                "Add an expansion cell before trusting N_opt(C)."
            )

        N = best.get("n_active_params") or _estimate_n(best["architecture_config"])
        optimal_points.append({
            "C": C,
            "N_opt": N,
            "D_opt": C / (6 * N),
            "min_loss": best["final_val_loss_joint"],
            "winning_depth": best["architecture_config"]["num_hidden_layers"],
        })
        print(f"C={C:.0e}: depth={best['architecture_config']['num_hidden_layers']}, "
              f"N={N:.2e}, min_loss={best['final_val_loss_joint']:.4f}")

    if len(optimal_points) < 2:
        print("Need ≥2 compute scales to fit. Run more Phase 1 cells.")
        return

    log_C = np.log([p["C"] for p in optimal_points])
    log_N = np.log([p["N_opt"] for p in optimal_points])
    a, log_k = np.polyfit(log_C, log_N, deg=1)
    k = float(np.exp(log_k))

    fitted = a * log_C + log_k
    residuals = log_N - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((log_N - log_N.mean()) ** 2))
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    max_residual = float(np.max(np.abs(residuals)))

    print(f"\nFitted power law: N_opt(C) = {k:.4e} × C^{a:.4f}")
    print(f"R²: {R2:.4f}")
    print(f"Max residual (log-space): {max_residual:.4f}")
    print(f"Sanity: a + b = {a + (1 - a):.4f} (should be 1.0)")

    if a < 0.4 or a > 0.6:
        print(f"WARNING: slope {a:.3f} outside Chinchilla [0.4, 0.6] range. Document caveat.")
    if R2 < 0.95:
        print(f"WARNING: R² {R2:.3f} < 0.95. Consider adding more compute scales.")


def _estimate_n(arch_config: dict) -> int:
    """Rough estimate from 12 * n_layer * d_model² formula (per assignment §3)."""
    return 12 * arch_config["num_hidden_layers"] * (arch_config["hidden_size"] ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local sweep runner — Stanford-API replacement")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="Submit a cell config")
    s.add_argument("--config", required=True, type=Path)

    s = sub.add_parser("dry-run", help="Show what submit would do without running")
    s.add_argument("--config", required=True, type=Path)

    sub.add_parser("status", help="Show all runs and budget")
    sub.add_parser("budget", help="Print budget JSON")
    sub.add_parser("fit", help="Fit power law from completed Phase 1 cells")
    sub.add_parser("plot", help="Generate all 3 plots (isoflops, scaling-law, predicted-vs-actual)")

    args = parser.parse_args()

    if args.cmd == "submit":
        submit(args.config)
    elif args.cmd == "dry-run":
        submit(args.config, dry_run=True)
    elif args.cmd == "status":
        status()
    elif args.cmd == "budget":
        budget()
    elif args.cmd == "fit":
        fit()
    elif args.cmd == "plot":
        # Run all 3 plot scripts; each is self-contained
        for script in ("plot_isoflops.py", "plot_scaling_law.py", "plot_predicted_vs_actual.py"):
            print(f"\n=== {script} ===")
            rc = subprocess.call([sys.executable, str(REPO_ROOT / "scripts" / script)])
            if rc != 0:
                print(f"  (script returned {rc} — see output above)")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
