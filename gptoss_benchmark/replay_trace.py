#!/usr/bin/env python3
"""
Script to replay traced moe_gemm_a8w4 calls for debugging and benchmarking.

Usage:
    # Replay a single trace file
    python replay_trace.py /path/to/trace_xxxxx.json

    # Replay all traces in a directory
    python replay_trace.py /path/to/traces/

    # Replay with custom device
    python replay_trace.py /path/to/trace.json --device cuda:1
"""

import argparse
import sys
from pathlib import Path

from test_moe_gemm_a8w4 import test_op_from_trace, test_all_traces_in_dir


def main():
    parser = argparse.ArgumentParser(
        description="Replay traced moe_gemm_a8w4 calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to trace JSON file or directory containing trace files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path does not exist: {args.path}", file=sys.stderr)
        sys.exit(1)

    if path.is_file():
        # Single trace file
        if not path.name.endswith(".json"):
            print(f"Error: File must be a JSON file: {args.path}", file=sys.stderr)
            sys.exit(1)
        test_op_from_trace(str(path), device=args.device)
    elif path.is_dir():
        # Directory of trace files
        test_all_traces_in_dir(str(path), device=args.device)
    else:
        print(f"Error: Invalid path: {args.path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
