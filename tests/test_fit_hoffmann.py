"""Tests for sweep_runner.fit_hoffmann.

Builds a synthetic manifest with known Hoffmann exponents (α=0.34, β=0.28 — Chinchilla),
runs the fit, asserts recovered exponents are within tolerance. Smoke-level — verifies
the L-BFGS+multi-init plumbing converges on well-conditioned data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _build_synthetic_manifest(tmp_path: Path, n_cells: int = 3) -> Path:
    """Create a manifest with n_cells × 16 trajectory points, generated from
    a known Hoffmann L(N, D) = 1.6 + 400/N^0.34 + 800/D^0.28.

    n_cells=3 → 3 unique N values → α identifiable.
    n_cells=2 → 2 unique N values → α non-identifiable (used to test warning).
    """
    E_true, A_true, B_true, alpha_true, beta_true = 1.6, 400.0, 800.0, 0.34, 0.28

    def L(N, D):
        return E_true + A_true / N**alpha_true + B_true / D**beta_true

    cell_specs = [
        ("C1", 1.5e8, 5e18, 20, 1280),
        ("C_mid", 2.5e8, 1.5e19, 24, 1536),
        ("C2", 4.0e8, 3e19, 26, 1664),
    ][:n_cells]

    runs = []
    for cell_idx, (cell_id, N, C, depth, hidden) in enumerate(cell_specs):
        D_final = C / (6 * N)
        traj = [
            {
                "step": step_idx * 100,
                "tokens": int(D_final * step_idx / 16.0),
                "flops": 6 * N * D_final * step_idx / 16.0,
                "val_bpb": float(L(N, D_final * step_idx / 16.0)),
            }
            for step_idx in range(1, 17)
        ]
        runs.append({
            "id": cell_idx + 1,
            "cell_id": cell_id,
            "phase": "1_fitting",
            "status": "completed",
            "compute_budget_target_flops": C,
            "n_active_params": int(N),
            "final_val_loss_joint": float(L(N, D_final)),
            "trajectory": traj,
            "architecture_config": {"num_hidden_layers": depth, "hidden_size": hidden},
            "wall_clock_hours": 1.0,
        })

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "total_budget_hours": 20.0,
        "used_hours": 2.0 * n_cells,
        "runs": runs,
    }, indent=2))
    return manifest_path


def test_recovers_known_exponents_3cells(tmp_path, monkeypatch, capsys):
    """3 unique N values + 48 trajectory points → fit recovers (α, β) within 0.05.
    This is the v3 minimum design (see dev/scaling_law_v3_minimal.md §4)."""
    pytest.importorskip("scipy")
    import scripts.sweep_runner as sr

    manifest_path = _build_synthetic_manifest(tmp_path, n_cells=3)
    monkeypatch.setattr(sr, "MANIFEST", manifest_path)

    sr.fit_hoffmann()
    out = capsys.readouterr().out

    import re
    m = re.search(r"N\^([\d.]+)\s*\+\s*[\d.e+\-]+\s*/\s*D\^([\d.]+)", out)
    assert m is not None, f"fit_hoffmann did not print exponents; output:\n{out}"
    alpha_fit = float(m.group(1))
    beta_fit = float(m.group(2))

    assert abs(alpha_fit - 0.34) < 0.05, f"alpha {alpha_fit} != 0.34 (3-cell fit should recover)"
    assert abs(beta_fit - 0.28) < 0.05, f"beta {beta_fit} != 0.28"
    # Identifiability warning should NOT appear with 3 unique N's
    assert "NON-IDENTIFIABLE" not in out


def test_warns_on_two_unique_N(tmp_path, monkeypatch, capsys):
    """With only 2 unique N values, fit_hoffmann must surface the non-identifiability
    warning. See dev/scaling_law_v3_minimal.md §5 cut #2 revision."""
    pytest.importorskip("scipy")
    import scripts.sweep_runner as sr

    manifest_path = _build_synthetic_manifest(tmp_path, n_cells=2)
    monkeypatch.setattr(sr, "MANIFEST", manifest_path)

    sr.fit_hoffmann()
    out = capsys.readouterr().out

    assert "NON-IDENTIFIABLE" in out, f"warning missing; output:\n{out}"
    assert "C_mid" in out, "warning should point to C_mid as fix"


def test_handles_empty_manifest(tmp_path, monkeypatch, capsys):
    pytest.importorskip("scipy")
    import scripts.sweep_runner as sr

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "total_budget_hours": 20.0, "used_hours": 0.0, "runs": [],
    }))
    monkeypatch.setattr(sr, "MANIFEST", manifest_path)

    sr.fit_hoffmann()
    out = capsys.readouterr().out
    assert "No completed Phase 1 cells" in out


def test_handles_too_few_points(tmp_path, monkeypatch, capsys):
    """Single cell with no trajectory + only endpoint → too few points → graceful skip."""
    pytest.importorskip("scipy")
    import scripts.sweep_runner as sr

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "total_budget_hours": 20.0, "used_hours": 1.0,
        "runs": [{
            "id": 1, "cell_id": "C1", "phase": "1_fitting", "status": "completed",
            "compute_budget_target_flops": 5e18,
            "n_active_params": 150000000,
            "final_val_loss_joint": 1.2,
            "trajectory": [],
            "architecture_config": {"num_hidden_layers": 20, "hidden_size": 1280},
            "wall_clock_hours": 1.0,
        }],
    }))
    monkeypatch.setattr(sr, "MANIFEST", manifest_path)

    sr.fit_hoffmann()
    out = capsys.readouterr().out
    assert "Only 1" in out or "need ≥5" in out
