# MOE GEMM A8W4 Test Script Usage Guide

## Overview

The `test_moe_gemm_a8w4.py` script can be used in two ways:

1. **As a pytest test suite** - Run parameterized tests with pytest
2. **As a standalone script** - Replay production traces from command line

## Usage

### 1. As a Pytest Test Suite

Run all parameterized tests:

```bash
# Run all tests
pytest test_moe_gemm_a8w4.py -v

# Run specific test combinations
pytest test_moe_gemm_a8w4.py::test_op -v

# Run with specific parameters
pytest test_moe_gemm_a8w4.py -k "m16_n256" -v

# Run with markers
pytest test_moe_gemm_a8w4.py -m "not slow" -v
```

### 2. As a Standalone Script

#### Replay a Single Trace File

```bash
# Basic usage
python test_moe_gemm_a8w4.py --trace /path/to/trace_1234567890_m1024_n3072_k3072.json

# With custom device
python test_moe_gemm_a8w4.py --trace /path/to/trace.json --device cuda:1

# Verbose output (includes parameter details and test output)
python test_moe_gemm_a8w4.py --trace /path/to/trace.json --verbose
```

#### Replay All Traces in a Directory

```bash
# Basic usage
python test_moe_gemm_a8w4.py --trace-dir /path/to/traces/

# With verbose output
python test_moe_gemm_a8w4.py --trace-dir /path/to/traces/ --verbose

# Custom device
python test_moe_gemm_a8w4.py --trace-dir /path/to/traces/ --device cuda:1
```

#### Help

```bash
python test_moe_gemm_a8w4.py --help
```

## Complete Workflow Example

### Step 1: Enable Tracing in Production

```bash
# Set environment variables
export MOE_GEMM_TRACE_ENABLE=1
export MOE_GEMM_TRACE_DIR=/tmp/prod_traces

# Run your production workload
python your_production_script.py

# Check generated traces
ls -lh /tmp/prod_traces/
```

### Step 2: Replay Traces Locally

```bash
# Quick validation (non-verbose)
python test_moe_gemm_a8w4.py --trace-dir /tmp/prod_traces/

# Output:
# Found 5 trace file(s) in /tmp/prod_traces/
#
# Testing trace_1234567890_m1024_n3072_k3072.json... ✓ PASSED
# Testing trace_1234567891_m2048_n3072_k3072.json... ✓ PASSED
# Testing trace_1234567892_m512_n3072_k3072.json... ✓ PASSED
# Testing trace_1234567893_m4096_n3072_k3072.json... ✓ PASSED
# Testing trace_1234567894_m128_n3072_k3072.json... ✓ PASSED
#
# ================================================================================
# Summary:
#   Total:   5
#   Passed:  5
#   Failed:  0
#   Skipped: 0
# ================================================================================
```

```bash
# Detailed debugging (verbose)
python test_moe_gemm_a8w4.py --trace /tmp/prod_traces/trace_1234567890_m1024_n3072_k3072.json --verbose

# Output:
# ================================================================================
# Replaying trace from: /tmp/prod_traces/trace_1234567890_m1024_n3072_k3072.json
# ================================================================================
# Parameters:
#   m                    = 1024
#   n                    = 3072
#   k                    = 3072
#   do_gather            = True
#   do_scatter           = True
#   has_y_gammas         = False
#   apply_swiglu         = True
#   fused_quant          = False
#   n_expts_tot          = 128
#   n_expts_act          = 4
#   act_dtype_str        = mxfloat8_e4m3fn
#   hbm_swizzling        = True
# ================================================================================
#
# [... test output ...]
#
# ================================================================================
# Trace replay completed successfully!
# ================================================================================
```

### Step 3: Use in CI/CD

```bash
#!/bin/bash
# ci_replay_traces.sh

set -e

# Download production traces from storage
echo "Downloading production traces..."
aws s3 sync s3://my-bucket/prod-traces ./prod-traces/

# Run trace replay tests
echo "Running trace replay tests..."
python test_moe_gemm_a8w4.py --trace-dir ./prod-traces/

# Check exit code
if [ $? -eq 0 ]; then
    echo "✓ All trace replays passed!"
    exit 0
else
    echo "✗ Some trace replays failed!"
    exit 1
fi
```

## Exit Codes

When run as a standalone script, the exit codes are:

- `0` - All tests passed successfully
- `1` - At least one test failed, or an error occurred
- `2` - No tests passed (all skipped or no tests found)
- `130` - Interrupted by user (Ctrl+C)

## Command Line Options

| Option | Description |
|--------|-------------|
| `--trace PATH` | Path to a single trace JSON file to replay |
| `--trace-dir DIR` | Directory containing trace JSON files to replay |
| `--device DEVICE` | Device to run on (default: cuda) |
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Show help message |

Note: `--trace` and `--trace-dir` are mutually exclusive.

## Tips

### 1. Filter Specific Traces

```bash
# Only test large matrices (m >= 4000)
for trace in /tmp/prod_traces/trace_*_m[4-9][0-9][0-9][0-9]_*.json; do
    python test_moe_gemm_a8w4.py --trace "$trace"
done
```

### 2. Parallel Execution

```bash
# Run multiple traces in parallel using GNU parallel
ls /tmp/prod_traces/trace_*.json | \
    parallel -j 4 python test_moe_gemm_a8w4.py --trace {}
```

### 3. Continuous Monitoring

```bash
# Watch for new traces and automatically test them
while inotifywait -e create /tmp/prod_traces/; do
    python test_moe_gemm_a8w4.py --trace-dir /tmp/prod_traces/
done
```

### 4. Generate Test Report

```bash
# Capture detailed output to a file
python test_moe_gemm_a8w4.py --trace-dir /tmp/prod_traces/ --verbose > test_report.txt 2>&1

# View summary
tail -20 test_report.txt
```

## Troubleshooting

### Script doesn't recognize --trace argument

Make sure you're running as a script, not with pytest:

```bash
# Correct
python test_moe_gemm_a8w4.py --trace /path/to/trace.json

# Incorrect (pytest doesn't recognize --trace)
pytest test_moe_gemm_a8w4.py --trace /path/to/trace.json
```

### Both pytest and script modes work

Yes! The file supports both modes:

```bash
# Use as pytest (runs parameterized tests)
pytest test_moe_gemm_a8w4.py

# Use as script (replays traces)
python test_moe_gemm_a8w4.py --trace-dir /path/to/traces/
```

### Import errors when running as script

Make sure you're in the correct directory:

```bash
cd /path/to/aiter/gptoss_benchmark
python test_moe_gemm_a8w4.py --trace /path/to/trace.json
```

Or add the project to PYTHONPATH:

```bash
export PYTHONPATH=/path/to/aiter:$PYTHONPATH
python test_moe_gemm_a8w4.py --trace /path/to/trace.json
```
