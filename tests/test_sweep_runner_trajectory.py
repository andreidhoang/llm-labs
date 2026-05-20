"""Tests for sweep_runner.parse_trajectory.

The function extracts per-eval trajectory points from base_train.py stdout so the
manifest can carry (N, D, L) samples instead of just final-loss endpoints. This
is load-bearing for Hoffmann fitting on a 2-cell fit grid (see
dev/scaling_law_v3_minimal.md §5 cut #2).
"""

from scripts.sweep_runner import parse_trajectory


def test_parses_multimodal_traj():
    stdout = """
Step 00100 | Validation mm_bpb: 1.234567 | text: 1.200000 | vision_ctx: 1.500000 | r_actual: 0.300
TRAJ_POINT | step=100 | tokens=51200000 | flops=1.234e+18 | val_bpb=1.234567 | val_text=1.200000 | val_vision=1.500000 | r_actual=0.300
Step 00200 | Validation mm_bpb: 1.123456 | text: 1.100000 | vision_ctx: 1.400000 | r_actual: 0.300
TRAJ_POINT | step=200 | tokens=102400000 | flops=2.468e+18 | val_bpb=1.123456 | val_text=1.100000 | val_vision=1.400000 | r_actual=0.300
"""
    pts = parse_trajectory(stdout)
    assert len(pts) == 2
    assert pts[0]["step"] == 100
    assert pts[0]["tokens"] == 51200000
    assert pts[0]["flops"] == 1.234e18
    assert pts[0]["val_bpb"] == 1.234567
    assert pts[0]["val_text"] == 1.200000
    assert pts[0]["val_vision"] == 1.500000
    assert pts[0]["r_actual"] == 0.300
    assert pts[1]["val_bpb"] == 1.123456
    assert pts[1]["tokens"] == 102400000


def test_parses_text_only_traj():
    stdout = (
        "Step 00050 | Validation bpb: 2.500000\n"
        "TRAJ_POINT | step=50 | tokens=25600000 | flops=6.17e+17 | val_bpb=2.500000\n"
    )
    pts = parse_trajectory(stdout)
    assert len(pts) == 1
    assert pts[0]["step"] == 50
    assert pts[0]["val_bpb"] == 2.5
    # text-only runs do not carry per-modality fields
    assert "val_text" not in pts[0]
    assert "val_vision" not in pts[0]
    assert "r_actual" not in pts[0]


def test_returns_empty_on_legacy_stdout():
    stdout = "just some training output with no TRAJ_POINT lines\nStep 100 | Validation bpb: 1.5\n"
    assert parse_trajectory(stdout) == []


def test_ignores_malformed_traj_line():
    # missing required tokens field — should not match, should not crash
    stdout = "TRAJ_POINT | step=100 | val_bpb=1.5\n"
    assert parse_trajectory(stdout) == []
