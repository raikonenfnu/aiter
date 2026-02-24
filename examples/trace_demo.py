#!/usr/bin/env python3
"""
Demo script showing how to use the MOE GEMM trace & replay system.

This script demonstrates:
1. Enabling tracing
2. Running a simple MOE GEMM operation
3. Verifying the trace was captured
4. Replaying the trace
"""

import os
import sys
import tempfile
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp


def run_demo():
    """Run a complete trace & replay demonstration."""

    print("="*80)
    print("MOE GEMM A8W4 Trace & Replay Demonstration")
    print("="*80)

    # Create temporary directory for traces
    trace_dir = tempfile.mkdtemp(prefix="moe_trace_demo_")
    print(f"\n1. Created temporary trace directory: {trace_dir}")

    # Enable tracing
    os.environ["MOE_GEMM_TRACE_ENABLE"] = "1"
    os.environ["MOE_GEMM_TRACE_DIR"] = trace_dir
    print(f"2. Enabled tracing (MOE_GEMM_TRACE_ENABLE=1)")

    # Set up a simple MOE GEMM operation
    print(f"\n3. Setting up MOE GEMM operation...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("   WARNING: CUDA not available, demo may not work correctly")
        return

    # Small test case
    m, k, n = 32, 256, 256
    n_expts_tot = 8
    n_expts_act = 2

    print(f"   - Matrix dimensions: m={m}, k={k}, n={n}")
    print(f"   - Experts: {n_expts_tot} total, {n_expts_act} active per token")

    # Create routing data
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    routing_data, gather_idx, scatter_idx = routing(logits, n_expts_act)
    routing_data.gate_scal = None

    # Create input tensors
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w = torch.randn((n_expts_tot, k, n), dtype=torch.bfloat16, device=device)
    bias = torch.randn((n_expts_tot, n), dtype=torch.float32, device=device)

    # Quantize to mxfp
    weight_dtype = torch.uint8  # mxfloat4_e2m1
    w_quant, w_scales = downcast_to_mxfp(w, weight_dtype, axis=1)

    act_dtype = torch.float8_e4m3fn
    x_quant, x_scales = downcast_to_mxfp(x, act_dtype, axis=-1)

    print(f"\n4. Running moe_gemm_a8w4 (this will generate a trace)...")

    # Run the operation (this should generate a trace)
    try:
        result = moe_gemm_a8w4(
            x_quant,
            w_quant,
            x_scales,
            w_scales,
            bias=bias,
            routing_data=routing_data,
            gather_indx=gather_idx,
            scatter_indx=scatter_idx,
            out_dtype=torch.bfloat16,
        )
        print(f"   ✓ Operation completed successfully")
        print(f"   Output shape: {result.shape}")
    except Exception as e:
        print(f"   ✗ Operation failed: {e}")
        return

    # Check for trace files
    print(f"\n5. Checking for generated trace files...")
    trace_files = list(Path(trace_dir).glob("trace_*.json"))

    if not trace_files:
        print(f"   ✗ No trace files found in {trace_dir}")
        print(f"   This might indicate an issue with the tracing system")
        return

    print(f"   ✓ Found {len(trace_files)} trace file(s)")
    for trace_file in trace_files:
        print(f"     - {trace_file.name}")

    # Show trace contents
    import json
    trace_file = trace_files[0]
    print(f"\n6. Trace file contents ({trace_file.name}):")
    with open(trace_file) as f:
        trace_data = json.load(f)
    print(json.dumps(trace_data, indent=2))

    # Disable tracing for replay
    os.environ["MOE_GEMM_TRACE_ENABLE"] = "0"
    print(f"\n7. Disabled tracing (MOE_GEMM_TRACE_ENABLE=0)")

    # Replay the trace
    print(f"\n8. Replaying trace...")
    try:
        from gptoss_benchmark.test_moe_gemm_a8w4 import test_op_from_trace
        test_op_from_trace(str(trace_file), device=device)
        print(f"   ✓ Replay completed successfully")
    except Exception as e:
        print(f"   ✗ Replay failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Cleanup
    print(f"\n9. Cleaning up...")
    import shutil
    shutil.rmtree(trace_dir)
    print(f"   Removed temporary directory: {trace_dir}")

    print(f"\n{'='*80}")
    print("Demo completed successfully!")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"  1. Enable tracing in your production code:")
    print(f"     export MOE_GEMM_TRACE_ENABLE=1")
    print(f"     export MOE_GEMM_TRACE_DIR=/path/to/traces")
    print(f"")
    print(f"  2. Replay traces using:")
    print(f"     python gptoss_benchmark/replay_trace.py /path/to/traces/")
    print(f"")
    print(f"  3. Read the documentation:")
    print(f"     cat gptoss_benchmark/TRACE_REPLAY_README.md")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_demo()
