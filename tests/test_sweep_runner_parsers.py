"""Tests for sweep_runner.py parsers.

Critical: ensures regexes match ACTUAL nanochat training output format.
A regex mismatch here silently fills manifest with None values → fits fail
→ plots fail. Worth dedicated tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.sweep_runner import (  # noqa: E402
    parse_final_val_loss,
    parse_flops_per_token,
    parse_n_active_params,
    parse_n_total_params,
    parse_num_iterations,
)


# Synthetic but format-accurate nanochat output (matches base_train.py print statements)
NANOCHAT_OUTPUT_TEMPLATE = """
GPU: H100-80GB | Peak FLOPS (BF16): 9.89e+14
Compute dtype: torch.bfloat16
Model config:
{
  "sequence_len": 4096,
  "vocab_size": 32000,
  ...
}
Parameter counts:
wte                     : 49,152,000
value_embeds            : 24,576,000
lm_head                 : 49,152,000
transformer_matrices    : 245,123,456
active_transformer_matrices: 87,654,321
scalars                 : 48
moe_inactive            : 157,469,135
total                   : 367,977,504
active_total            : 210,508,369
Estimated FLOPs per token: 1.234e+09
Calculated number of iterations from target FLOPs: 4,860
Total training FLOPs estimate: 5.000e+18
Tokens : Scaling params ratio: 12.50
Step 00100/04860 (2.06%) | loss: 7.234567 | lrm: 0.10
Step 04860 | Validation bpb: 0.715432
Minimum validation bpb: 0.712100
"""


def test_parse_minimum_validation_bpb():
    """Critical: must match nanochat's 'Minimum validation bpb' format."""
    val = parse_final_val_loss(NANOCHAT_OUTPUT_TEMPLATE)
    assert val == 0.7121, f"expected 0.7121, got {val}"


def test_parse_validation_bpb_fallback():
    """If only 'Validation bpb: X' present (no minimum), parse it as fallback."""
    output_no_min = "Step 100 | Validation bpb: 0.823456"
    assert parse_final_val_loss(output_no_min) == 0.823456


def test_parse_val_loss_joint_takes_priority():
    """If multimodal val_loss/joint is present, prefer it over bpb."""
    output_mm = """
Step 100 | Validation bpb: 0.715
Step 100 | val_loss/joint: 0.832
"""
    assert parse_final_val_loss(output_mm) == 0.832


def test_parse_no_loss_returns_none():
    """If no loss line at all, return None (not crash)."""
    assert parse_final_val_loss("Random output with no loss info") is None


def test_parse_active_params():
    val = parse_n_active_params(NANOCHAT_OUTPUT_TEMPLATE)
    assert val == 210_508_369, f"expected 210508369, got {val}"


def test_parse_total_params():
    val = parse_n_total_params(NANOCHAT_OUTPUT_TEMPLATE)
    assert val == 367_977_504


def test_parse_flops_per_token():
    val = parse_flops_per_token(NANOCHAT_OUTPUT_TEMPLATE)
    assert val == 1.234e9, f"expected 1.234e9, got {val}"


def test_parse_num_iterations():
    val = parse_num_iterations(NANOCHAT_OUTPUT_TEMPLATE)
    assert val == 4860


def test_parse_iterations_from_target_data_ratio():
    """Both target-flops and target-param-data-ratio paths print 'Calculated number of iterations'."""
    output = "Calculated number of iterations from target data:param ratio: 12,345"
    val = parse_num_iterations(output)
    assert val == 12345


def test_parsers_handle_extra_whitespace():
    """Tolerant of slight format variations (extra spaces, tabs)."""
    output = "active_total              :    1,000,000\nValidation bpb:0.5\n"
    assert parse_n_active_params(output) == 1_000_000
    assert parse_final_val_loss(output) == 0.5


if __name__ == "__main__":
    import inspect
    tests = [
        (name, obj)
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failures.append((name, e))
            import traceback
            print(f"FAIL  {name}: {e}")
            traceback.print_exc()
    print(f"\n{'=' * 70}\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    sys.exit(0 if not failures else 1)
