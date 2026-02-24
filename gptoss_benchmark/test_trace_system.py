#!/usr/bin/env python3
"""
Quick test to verify the trace & replay system works correctly.
"""

import json
import os
import tempfile
from pathlib import Path

import torch
from test_moe_gemm_a8w4 import test_op_from_trace


def test_trace_replay_system():
    """
    Test that we can:
    1. Create a synthetic trace file
    2. Load and replay it successfully
    """
    # Create a synthetic trace file
    trace_params = {
        "m": 16,
        "n": 256,
        "k": 256,
        "do_gather": False,
        "do_scatter": False,
        "has_y_gammas": False,
        "apply_swiglu": False,
        "fused_quant": False,
        "n_expts_tot": 128,
        "n_expts_act": 4,
        "act_dtype_str": "mxfloat8_e4m3fn",
        "hbm_swizzling": True,
    }

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(trace_params, f, indent=2)
        trace_file = f.name

    try:
        print(f"Created test trace file: {trace_file}")
        print(f"\nTrace contents:")
        print(json.dumps(trace_params, indent=2))

        # Test replay
        print(f"\n{'='*80}")
        print("Testing trace replay...")
        print(f"{'='*80}\n")

        test_op_from_trace(trace_file)

        print(f"\n{'='*80}")
        print("✓ Trace replay test PASSED!")
        print(f"{'='*80}\n")

    finally:
        # Cleanup
        if os.path.exists(trace_file):
            os.remove(trace_file)
            print(f"Cleaned up test trace file: {trace_file}")


if __name__ == "__main__":
    test_trace_replay_system()
