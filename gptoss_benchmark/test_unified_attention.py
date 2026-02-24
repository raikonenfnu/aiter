# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional
import sys
import argparse

import pytest
import torch
import json
from pathlib import Path

from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils.types import e4m3_dtype

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 64]

DTYPES = [torch.float16, torch.bfloat16]
QDTYPES = [None, e4m3_dtype]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]


# ---------------
# unit tests
# ---------------


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("q_dtype", QDTYPES)
@torch.inference_mode()
def test_op(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    q_dtype: Optional[torch.dtype],
) -> None:
    """Pytest parametrized test wrapper for test_triton_unified_attn."""
    test_triton_unified_attn(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        q_dtype=q_dtype,
    )


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        if sinks is not None:
            s_aux = sinks[:, None, None].repeat_interleave(attn.shape[-2], dim=-2)
            attn = torch.cat((attn, s_aux), dim=-1)
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        if sinks is not None:
            attn = attn[..., :-1]
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("q_dtype", QDTYPES)
def test_triton_unified_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    q_dtype: Optional[torch.dtype],
) -> None:
    """Run unified attention test with given parameters."""
    if q_dtype is not None and q_dtype.itemsize < 2 and block_size < 32:
        pytest.skip("block size must be at least 32 for fp8")

    torch.manual_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=dtype, device="cuda"
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device="cuda"
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cuda"
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    # To prevent overflow we set max num blocks to 2**16
    max_num_blocks = 2**16
    range_num_blocks = min(max_num_blocks, num_blocks)
    block_tables = torch.randint(
        0,
        range_num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )
    sinks = torch.randn(num_query_heads, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    if q_dtype is not None:
        # QKV are drawn from N(0, 1): no need for a fp8 scaling factor
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)

    scale_shape = (num_seqs, num_kv_heads)
    q_descale = None  # Not yet supported
    k_descale = torch.rand(scale_shape, dtype=torch.float32, device="cuda")
    v_descale = torch.rand(scale_shape, dtype=torch.float32, device="cuda")

    unified_attention(
        q=maybe_quantized_query,
        k=maybe_quantized_key_cache,
        v=maybe_quantized_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        sinks=sinks,
    )
    atol, rtol = 1.5e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(
        output, ref_output, atol=atol, rtol=rtol
    ), f"{torch.max(torch.abs(output - ref_output))}"
    print(output)



def load_trace_params(json_path):
    """
    Load trace parameters from a JSON file.

    Args:
        json_path: Path to the JSON file containing traced parameters

    Returns:
        Dictionary of parameters that can be unpacked to run_triton_unified_attn
    """
    with open(json_path, "r") as f:
        params = json.load(f)["configuration"]

    # Parse parameters
    seq_lens = [tuple(x) for x in params["seq_lens"]]
    num_heads = tuple(params["num_heads"])
    head_size = params["head_size"]
    sliding_window = params.get("sliding_window")
    dtype_str = params["dtype"]
    block_size = params["block_size"]
    soft_cap = params.get("soft_cap")
    num_blocks = params["num_blocks"]
    q_dtype_str = params.get("q_dtype")

    # Convert dtype strings to torch dtypes
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float8_e4m3fn": e4m3_dtype,
        "float8_e4m3fnuz": e4m3_dtype,
    }

    dtype = dtype_map.get(dtype_str, torch.float16)
    q_dtype = dtype_map.get(q_dtype_str) if q_dtype_str else None

    return {
        "seq_lens": seq_lens,
        "num_heads": num_heads,
        "head_size": head_size,
        "sliding_window": sliding_window,
        "dtype": dtype,
        "block_size": block_size,
        "soft_cap": soft_cap,
        "num_blocks": num_blocks,
        "q_dtype": q_dtype,
    }


def test_op_from_trace(json_path, verbose=True):
    """
    Run test_triton_unified_attn using parameters loaded from a traced JSON file.

    This allows reproducing real-life cases captured during production runs.

    Usage:
        # First, enable tracing in production:
        # export AITER_TRACE_ATTENTION=1
        # export AITER_TRACE_ATTENTION_DIR=/path/to/traces
        # ... run your workload ...

        # Then replay the trace:
        test_op_from_trace("/path/to/traces/trace_xxxxx.json")

    Args:
        json_path: Path to the JSON trace file
        verbose: Whether to print detailed output (default: True)
    """
    params = load_trace_params(json_path)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Replaying trace from: {json_path}")
        print(f"{'='*80}")
        print(f"Parameters:")
        for key, value in params.items():
            print(f"  {key:20s} = {value}")
        print(f"{'='*80}\n")

    # Call test_triton_unified_attn with loaded parameters
    test_triton_unified_attn(
        seq_lens=params["seq_lens"],
        num_heads=params["num_heads"],
        head_size=params["head_size"],
        sliding_window=params["sliding_window"],
        dtype=params["dtype"],
        block_size=params["block_size"],
        soft_cap=params["soft_cap"],
        num_blocks=params["num_blocks"],
        q_dtype=params["q_dtype"],
    )

    if verbose:
        print(f"\n{'='*80}")
        print(f"Trace replay completed successfully!")
        print(f"{'='*80}\n")


def test_all_traces_in_dir(trace_dir, verbose=False):
    """
    Run run_triton_unified_attn for all trace files in a directory.

    Args:
        trace_dir: Directory containing trace JSON files
        verbose: Whether to print verbose output (default: False)
    """
    trace_path = Path(trace_dir)
    if not trace_path.exists():
        raise ValueError(f"Trace directory does not exist: {trace_dir}")

    trace_files = sorted(trace_path.glob("trace_*.json"))
    if not trace_files:
        print(f"No trace files found in {trace_dir}")
        return

    print(f"\nFound {len(trace_files)} trace file(s) in {trace_dir}\n")

    passed = 0
    failed = 0
    skipped = 0

    for trace_file in trace_files:
        try:
            if verbose:
                print(f"\n{'='*80}")
                print(f"Testing: {trace_file.name}")
                print(f"{'='*80}")
            else:
                print(f"Testing {trace_file.name}...", end=" ", flush=True)

            test_op_from_trace(str(trace_file), verbose=verbose)
            passed += 1

            if not verbose:
                print("✓ PASSED")
        except pytest.skip.Exception as e:
            if verbose:
                print(f"SKIPPED: {e}")
            else:
                print(f"⊘ SKIPPED: {e}")
            skipped += 1
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()
            else:
                print(f"✗ FAILED: {e}")
            failed += 1

    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total:   {len(trace_files)}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"{'='*80}\n")

    return {"total": len(trace_files), "passed": passed, "failed": failed, "skipped": skipped}


def main():
    """
    Main entry point for running trace replays from command line.

    This allows using the test file both as a pytest test suite and as a
    standalone script for replaying production traces.

    Usage:
        # Replay a single trace file
        python test_unified_attention.py --trace /path/to/trace.json

        # Replay all traces in a directory
        python test_unified_attention.py --trace-dir /path/to/traces/

        # Verbose output
        python test_unified_attention.py --trace-dir /path/to/traces/ --verbose
    """
    parser = argparse.ArgumentParser(
        description="Unified Attention Trace Replay Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay a single trace file
  %(prog)s --trace /path/to/trace_1234567890_nh32_hs128_bs16.json

  # Replay all traces in a directory
  %(prog)s --trace-dir ./attention_traces/

  # With verbose output
  %(prog)s --trace-dir ./attention_traces/ --verbose

Note:
  This script can also be used with pytest:
    pytest test_unified_attention.py -v
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--trace",
        type=str,
        metavar="PATH",
        help="Path to a single trace JSON file to replay",
    )
    group.add_argument(
        "--trace-dir",
        type=str,
        metavar="DIR",
        help="Directory containing trace JSON files to replay",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Validate paths
    if args.trace:
        trace_path = Path(args.trace)
        if not trace_path.exists():
            print(f"Error: Trace file does not exist: {args.trace}", file=sys.stderr)
            sys.exit(1)
        if not trace_path.is_file():
            print(f"Error: Path is not a file: {args.trace}", file=sys.stderr)
            sys.exit(1)
        if not trace_path.name.endswith(".json"):
            print(f"Warning: File does not have .json extension: {args.trace}", file=sys.stderr)

    if args.trace_dir:
        trace_dir_path = Path(args.trace_dir)
        if not trace_dir_path.exists():
            print(f"Error: Trace directory does not exist: {args.trace_dir}", file=sys.stderr)
            sys.exit(1)
        if not trace_dir_path.is_dir():
            print(f"Error: Path is not a directory: {args.trace_dir}", file=sys.stderr)
            sys.exit(1)

    # Run replay
    try:
        if args.trace:
            # Replay single trace
            test_op_from_trace(args.trace, verbose=args.verbose)
            print("\n✓ Trace replay completed successfully!\n")
            sys.exit(0)
        else:
            # Replay all traces in directory
            results = test_all_traces_in_dir(args.trace_dir, verbose=args.verbose)

            # Exit with appropriate code
            if results["failed"] > 0:
                sys.exit(1)  # At least one test failed
            elif results["passed"] == 0:
                sys.exit(2)  # No tests passed (all skipped or no tests)
            else:
                sys.exit(0)  # All tests passed

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
