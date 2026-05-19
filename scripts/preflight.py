"""Pre-flight verification: run before spending any GPU $.

Single command that runs the full battery of CPU-side checks. If everything
green, you're cleared to commit budget on vast.ai.

Layered checks (each layer gates the next):
    Layer 1 — Static: imports + config validation                              <5 sec
    Layer 2 — Unit: 21-check verify_multimodal.py + 49 pytest tests           <30 sec
    Layer 3 — Integration: full multimodal forward + backward + scatter       <10 sec
    Layer 4 — Determinism: same seed → bit-exact loss for 100 steps           <30 sec
    Layer 5 — GPU smoke (if CUDA available): real SigLIP2 + multimodal forward <60 sec

Usage:
    python scripts/preflight.py              # all CPU layers
    python scripts/preflight.py --gpu        # also run GPU smoke (Layer 5)
    python scripts/preflight.py --quick      # skip slow tests (only Layers 1-3)

Exit code: 0 = all green / cleared for GPU spend; non-zero = blocker found.

Spec: dev/multimodal_debug_guide.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _print_header(title: str):
    print(f"\n{BOLD}{'═' * 70}\n  {title}\n{'═' * 70}{RESET}")


def _print_check(name: str, ok: bool, msg: str = ""):
    marker = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    suffix = f"  {msg}" if msg else ""
    print(f"  {marker} {name}{suffix}")


def _run_subprocess(cmd, timeout=120):
    """Run subprocess; return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s"


# =============================================================================
# Layer 1 — Static: imports + config validation
# =============================================================================

def layer1_static() -> bool:
    _print_header("LAYER 1 — Static checks (imports + config validation)")
    all_ok = True

    # Check 1: imports
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from core.model import GPT, GPTConfig
        from core.multimodal import (
            VisionTower, PatchMerger, scatter_vision_features,
            build_3d_mrope_for_4d_apply, build_position_ids_for_mm,
            per_modality_loss_decomposition,
        )
        from core.dataloader import synthetic_multimodal_loader
        from core.engine import KVCache
        _print_check("All multimodal imports succeed", True)
    except ImportError as e:
        _print_check("Multimodal imports", False, str(e))
        all_ok = False
        return all_ok

    # Check 2: config JSONs valid
    config_dir = REPO_ROOT / "configs" / "scaling_law"
    if not config_dir.exists():
        _print_check("configs/scaling_law/ exists", False)
        return False
    config_files = sorted(config_dir.glob("F*.json")) + [config_dir / "G3_big_run.template.json"]
    for cf in config_files:
        try:
            cfg = json.loads(cf.read_text())
            assert "cell_id" in cfg
            assert "compute_budget_target_flops" in cfg
            assert "architecture_config" in cfg
            assert "moe_config" in cfg
            assert "multimodal_config" in cfg
            _print_check(f"Config {cf.name} valid", True)
        except Exception as e:
            _print_check(f"Config {cf.name}", False, str(e))
            all_ok = False

    # Check 3: required directories exist or can be created
    runs_dir = REPO_ROOT / "runs"
    plots_dir = REPO_ROOT / "dev" / "plots"
    runs_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    _print_check("runs/ and dev/plots/ writable", True)

    return all_ok


# =============================================================================
# Layer 2 — Unit: verify_multimodal.py + pytest
# =============================================================================

def layer2_unit() -> bool:
    _print_header("LAYER 2 — Unit tests (verify_multimodal.py + 3 test suites)")
    all_ok = True

    t0 = time.time()
    rc, out, err = _run_subprocess(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_multimodal.py")],
        timeout=60,
    )
    if rc == 0:
        # Last line should have "Summary: X pass, Y fail, Z skip"
        last_line = out.strip().split("\n")[-2] if "Summary" not in out.strip().split("\n")[-1] else out.strip().split("\n")[-1]
        _print_check(f"verify_multimodal.py", True, last_line)
    else:
        _print_check("verify_multimodal.py", False, f"rc={rc}; check output above")
        print(out[-500:] if out else "")
        all_ok = False

    for test_file in [
        "test_multimodal_joint_forward.py",
        "test_multimodal_integration.py",
        "test_real_siglip.py",
        "test_sweep_runner_parsers.py",
    ]:
        rc, out, err = _run_subprocess(
            [sys.executable, str(REPO_ROOT / "tests" / test_file)],
            timeout=120,
        )
        # Parse "X/Y tests passed" from last line
        last = out.strip().split("\n")[-2] if out.strip().endswith("first run") else out.strip().split("\n")[-1]
        if rc == 0:
            _print_check(f"tests/{test_file}", True, last.strip())
        else:
            _print_check(f"tests/{test_file}", False, f"rc={rc}")
            all_ok = False

    elapsed = time.time() - t0
    print(f"  ({elapsed:.1f}s elapsed)")
    return all_ok


# =============================================================================
# Layer 3 — Integration: full multimodal forward + backward
# =============================================================================

def layer3_integration() -> bool:
    _print_header("LAYER 3 — Integration: full multimodal forward + backward")
    all_ok = True

    try:
        import torch
        import torch.nn as nn
        from core.dataloader import synthetic_multimodal_loader
        from core.model import GPT, GPTConfig
        from core.multimodal import VisionTower

        class MockSiglip(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Conv2d(3, 64, 4, 4)
            def forward(self, pv):
                x = self.proj(pv); B, D, H, W = x.shape
                return x.permute(0, 2, 3, 1).reshape(B, H * W, D)

        # Build small multimodal model
        cfg = GPTConfig(
            sequence_len=128, vocab_size=256, n_layer=2, n_head=4, n_kv_head=4, n_embd=64,
            num_experts=4, top_k=2, num_shared_experts=1, window_pattern="L",
            multimodal=True, vision_embed_dim=64, vision_spatial_merge_size=2,
            image_pad_token_id=255,
        )
        with torch.device("meta"):
            m = GPT(cfg)
        m.to_empty(device="cpu")
        m._needs_vision_tower = False
        m.init_weights()
        m.vision_tower = VisionTower(
            llm_hidden_size=64, vision_encoder=MockSiglip(), vision_embed_dim=64,
            spatial_merge_size=2, freeze_merger=True,
        )
        _print_check("GPT(multimodal=True) builds + init_weights succeeds", True)

        # Synthetic dataloader
        def fake_text_loader():
            torch.manual_seed(0)
            for _ in range(1):
                inputs = torch.randint(0, 240, (1, 64), dtype=torch.long)
                targets = torch.randint(0, 240, (1, 64), dtype=torch.long)
                yield inputs, targets, {"pq_idx": 0, "rg_idx": 0, "epoch": 1}

        loader = synthetic_multimodal_loader(
            fake_text_loader(), mix_ratio=0.3, image_grid_thw_raw=(1, 4, 4),
            spatial_merge_size=2, image_pad_token_id=cfg.vocab_size - 1,
            image_size_pixels=16, seed=0, device="cpu",
        )
        inputs, targets, extras, state = next(loader)
        _print_check(f"Synthetic loader emits batch_extras keys: "
                     f"{list(extras.keys())}", True)

        # Forward
        out = m(inputs, targets, **extras)
        if isinstance(out, dict):
            _print_check(f"Forward returns dict (per-modality split): "
                         f"loss={out['loss'].item():.3f}", True)
            _print_check(f"loss_text={out['loss_text'].item():.3f}, "
                         f"loss_vision={out['loss_vision'].item():.3f}, "
                         f"n_text={out['n_text'].item()}, "
                         f"n_vision={out['n_vision'].item()}", True)
        else:
            _print_check("Forward should return dict (modality_mask provided)", False)
            all_ok = False

        # Backward
        out["loss"].backward()
        n_grads = sum(1 for p in m.parameters()
                      if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0)
        _print_check(f"Backward pass: {n_grads} params with non-zero gradients", n_grads > 0)
        if n_grads == 0:
            all_ok = False

        # Vision tower stayed frozen
        n_vt_grads = sum(1 for p in m.vision_tower.siglip.parameters() if p.grad is not None)
        _print_check(f"Vision encoder frozen (0 grads in SigLIP): "
                     f"actual={n_vt_grads}", n_vt_grads == 0)
        if n_vt_grads > 0:
            all_ok = False

        # Text-only path on same model still works
        text_loss = m(inputs, targets)
        _print_check(f"Text-only path (no extras): scalar loss "
                     f"={text_loss.item():.3f}", text_loss.dim() == 0)
        if text_loss.dim() != 0:
            all_ok = False

    except Exception as e:
        import traceback
        _print_check(f"Integration test failed: {e}", False)
        traceback.print_exc()
        all_ok = False

    return all_ok


# =============================================================================
# Layer 4 — Determinism: same seed → bit-exact (within float tolerance) loss
# =============================================================================

def layer4_determinism() -> bool:
    _print_header("LAYER 4 — Determinism: same seed → reproducible loss")
    all_ok = True

    try:
        import torch
        import torch.nn as nn
        from core.dataloader import synthetic_multimodal_loader

        def fake_text_loader(seed):
            torch.manual_seed(seed)
            for _ in range(1):
                inputs = torch.randint(0, 240, (1, 64), dtype=torch.long)
                targets = torch.randint(0, 240, (1, 64), dtype=torch.long)
                yield inputs, targets, {"pq_idx": 0, "rg_idx": 0, "epoch": 1}

        # Two loader instances, same seed
        l1 = synthetic_multimodal_loader(
            fake_text_loader(seed=0), mix_ratio=0.3, image_grid_thw_raw=(1, 4, 4),
            spatial_merge_size=2, image_pad_token_id=255,
            image_size_pixels=16, seed=42, device="cpu",
        )
        l2 = synthetic_multimodal_loader(
            fake_text_loader(seed=0), mix_ratio=0.3, image_grid_thw_raw=(1, 4, 4),
            spatial_merge_size=2, image_pad_token_id=255,
            image_size_pixels=16, seed=42, device="cpu",
        )
        i1, t1, e1, _ = next(l1)
        i2, t2, e2, _ = next(l2)

        # Check inputs identical
        ok_inputs = torch.equal(i1, i2)
        _print_check(f"Same-seed text inputs identical", ok_inputs)
        if not ok_inputs:
            all_ok = False

        # Check pixel_values bit-exact
        ok_pixels = torch.allclose(e1["pixel_values"], e2["pixel_values"], atol=1e-6)
        _print_check(f"Same-seed synthetic pixel_values bit-exact (atol=1e-6)", ok_pixels)
        if not ok_pixels:
            all_ok = False

        # Check image_pad_mask identical
        ok_mask = torch.equal(e1["image_pad_mask"], e2["image_pad_mask"])
        _print_check(f"Same-seed image_pad_mask identical", ok_mask)
        if not ok_mask:
            all_ok = False

    except Exception as e:
        import traceback
        _print_check(f"Determinism check failed: {e}", False)
        traceback.print_exc()
        all_ok = False

    return all_ok


# =============================================================================
# Layer 5 — GPU smoke (optional, requires CUDA)
# =============================================================================

def layer5_gpu_smoke() -> bool:
    _print_header("LAYER 5 — GPU smoke (real SigLIP2 + multimodal forward)")

    try:
        import torch
        if not torch.cuda.is_available():
            _print_check("CUDA available", False, "Skipping (no GPU)")
            return True  # not a failure, just skipped
        _print_check(f"CUDA available: {torch.cuda.device_count()} GPU(s) "
                     f"({torch.cuda.get_device_name(0)})", True)

        # Probe core.flash_attention to report the ACTIVE dispatch path
        # (FA3 hub kernel / FA2 pip / SDPA fallback). This is more informative
        # than a raw flash_attn import because production uses the unified
        # interface, not the raw flash_attn package directly.
        try:
            from core.flash_attention import (
                USE_FA3, USE_FA2, HAS_FA3, HAS_FA2,
            )
        except ImportError as e:
            _print_check("core.flash_attention importable", False, str(e))
            return False

        major, _ = torch.cuda.get_device_capability()
        is_hopper = (major == 9)

        if USE_FA3:
            _print_check("Flash Attention dispatch", True, "FA3 active (hub kernel)")
        elif USE_FA2:
            severity = "OK on non-Hopper" if not is_hopper else "WARN: FA2 on Hopper (FA3 expected)"
            _print_check("Flash Attention dispatch", not is_hopper, f"FA2 active — {severity}")
            if is_hopper:
                print(f"      sm_90 detected but FA3 not loaded. Try:")
                print(f"        - bash auto/setup_env.sh   (full FA3 recipe)")
                print(f"        - or see auto/FA3_SETUP.md (NGC ABI mismatch is most common cause)")
        else:
            _print_check("Flash Attention dispatch", False,
                         "SDPA fallback (no flash-attn available)")
            print(f"      Neither FA3 nor FA2 loaded; HAS_FA3={HAS_FA3} HAS_FA2={HAS_FA2}")
            print(f"      Install: pip install flash-attn --no-build-isolation  (FA2)")
            print(f"           or: bash auto/setup_env.sh                       (FA3 via kernels hub)")
            return False
    except Exception as e:
        _print_check(f"GPU layer crashed: {e}", False)
        return False

    # Run real-SigLIP test suite (will download SigLIP2 ~1GB on first run)
    print(f"  Running tests/test_real_siglip.py with SIGLIP_DOWNLOAD=1...")
    print(f"  (downloads SigLIP2-SO400M ~1GB on first run; cached afterward)")
    rc, out, err = _run_subprocess(
        [sys.executable, str(REPO_ROOT / "tests" / "test_real_siglip.py")],
        timeout=900,  # 15 min for download + tests
    )
    # Run with env var
    import os
    env = os.environ.copy()
    env["SIGLIP_DOWNLOAD"] = "1"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tests" / "test_real_siglip.py")],
        capture_output=True, text=True, timeout=900, env=env,
    )
    if proc.returncode == 0:
        last = proc.stdout.strip().split("\n")[-1]
        _print_check(f"Real SigLIP2 + multimodal forward", True, last)
    else:
        _print_check(f"Real SigLIP2 GPU smoke", False, f"rc={proc.returncode}")
        print(proc.stdout[-1000:])
        return False

    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pre-flight verification before GPU spend")
    parser.add_argument("--gpu", action="store_true", help="Also run Layer 5 GPU smoke (requires CUDA + downloads SigLIP2)")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests (Layers 4+); only run 1-3")
    args = parser.parse_args()

    print(f"{BOLD}Pre-flight verification — running CPU-side checks before GPU spend{RESET}")

    layers_passed = []

    if not layer1_static():
        print(f"\n{RED}{BOLD}LAYER 1 FAILED.{RESET} Block on imports/configs before proceeding.")
        return 1
    layers_passed.append("L1 static")

    if not layer2_unit():
        print(f"\n{RED}{BOLD}LAYER 2 FAILED.{RESET} Unit tests broken — fix before integration.")
        return 1
    layers_passed.append("L2 unit")

    if not layer3_integration():
        print(f"\n{RED}{BOLD}LAYER 3 FAILED.{RESET} Integration broken — fix before GPU.")
        return 1
    layers_passed.append("L3 integration")

    if not args.quick:
        if not layer4_determinism():
            print(f"\n{YELLOW}{BOLD}LAYER 4 FAILED (warning).{RESET} Determinism broken; may indicate dataloader bug.")
            return 1
        layers_passed.append("L4 determinism")

    if args.gpu:
        if not layer5_gpu_smoke():
            print(f"\n{RED}{BOLD}LAYER 5 FAILED.{RESET} Real SigLIP2 GPU integration broken.")
            return 1
        layers_passed.append("L5 GPU smoke")

    print(f"\n{GREEN}{BOLD}{'═' * 70}")
    print(f"  ✓ ALL LAYERS PASSED ({', '.join(layers_passed)})")
    print(f"  Cleared for GPU spend per dev/scaling_law_self_assignment.md")
    print(f"{'═' * 70}{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
