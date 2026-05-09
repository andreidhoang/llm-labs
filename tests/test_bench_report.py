from __future__ import annotations

import json
import sys

from scripts import bench_report


def _write_json(path, payload):
    path.write_text(json.dumps(payload))


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_bench_report_generates_markdown(tmp_path, monkeypatch):
    bench_dir = tmp_path / "runs" / "bench"
    bench_dir.mkdir(parents=True)

    _write_json(
        bench_dir / "_meta.json",
        {
            "latest_run": {
                "git_branch": "bench",
                "git_sha": "1234567890abcdef",
                "git_dirty": False,
                "host": "h100-vm",
                "torch_version": "2.8.0",
                "torch_cuda_version": "12.8",
                "nccl_version": "2.26.2",
                "gpu_count": 2,
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "gpu_capability": "sm_90",
                "driver_version": "570.86",
            }
        },
    )
    base = {
        "phase": "phase_0_baseline",
        "n_gpu": 2,
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "device_batch_size": 8,
        "grad_accum_steps": 4,
        "max_seq_len": 2048,
        "compile_mode": "default",
        "compile_fullgraph": False,
        "chunked_ce": False,
        "activation_ckpt": False,
        "fp8": True,
        "metrics": {
            "step_time_ms": 100.0,
            "fwd_bwd_ms": 80.0,
            "optim_step_ms": 20.0,
            "optim_pct_of_step": 20.0,
            "tokens_per_sec_per_gpu": 1000.0,
            "mfu_pct": 40.0,
            "peak_hbm_gb": 50.0,
            "compile_first_step_overhead_s": 10.0,
        },
    }
    faster = {
        **base,
        "phase": "phase_1_chunked_ce",
        "chunked_ce": True,
        "metrics": {
            **base["metrics"],
            "step_time_ms": 80.0,
            "tokens_per_sec_per_gpu": 1250.0,
            "peak_hbm_gb": 42.0,
        },
    }
    _write_json(bench_dir / "phase_0_baseline.json", base)
    _write_json(bench_dir / "phase_1_chunked_ce.json", faster)
    _write_jsonl(
        bench_dir / "phase_0_baseline.jsonl",
        [
            {"step": 0, "step_ms": 101.0, "fwdbwd_ms": 81.0, "optim_ms": 20.0, "peak_hbm_gb": 50.0},
            {"step": 1, "step_ms": 99.0, "fwdbwd_ms": 79.0, "optim_ms": 20.0, "peak_hbm_gb": 50.1},
        ],
    )
    _write_jsonl(
        bench_dir / "phase_1_chunked_ce.jsonl",
        [
            {"step": 0, "step_ms": 81.0, "fwdbwd_ms": 65.0, "optim_ms": 16.0, "peak_hbm_gb": 42.0},
            {"step": 1, "step_ms": 79.0, "fwdbwd_ms": 63.0, "optim_ms": 16.0, "peak_hbm_gb": 42.1},
        ],
    )

    out = bench_dir / "report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench_report", "--bench-dir", str(bench_dir), "--out", str(out), "--no-plots"],
    )
    bench_report.main()

    report = out.read_text()
    assert "# Wallclock Benchmark Report" in report
    assert "phase_1_chunked_ce" in report
    assert "(-20.0%, better)" in report
    assert "highest throughput per GPU" in report
    assert "h100-vm" in report
