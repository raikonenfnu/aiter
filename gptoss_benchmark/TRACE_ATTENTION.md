# Unified Attention Tracing & Testing System

Complete guide for tracing real-world `unified_attention` calls and replaying them as tests.

## Overview

This system allows you to:
1. **Trace**: Capture real inference parameters during production runs
2. **Store**: Save parameters as lightweight JSON files
3. **Replay**: Run captured scenarios as reproducible tests

## Quick Start

### 1. Capture Traces

```bash
# Enable tracing
export AITER_TRACE_ATTENTION=1

# Run your inference workload
python your_inference_script.py

# Traces saved to ./attention_traces/
```

### 2. Replay Traces

```bash
cd gptoss_benchmark

# Single trace
python test_unified_attention.py --trace-file ../attention_traces/trace_123.json

# All traces in directory
python test_unified_attention.py --trace-dir ../attention_traces
```

## Files

### Core Implementation
- **`aiter/ops/triton/attention/unified_attention.py`** - Tracing implementation
- **`gptoss_benchmark/test_unified_attention.py`** - Standalone test runner

### Documentation
- **`STANDALONE_USAGE.md`** - Detailed CLI usage guide
- **`TRACE_USAGE.md`** - Comprehensive tracing documentation
- **`README_TRACING.md`** - This file

### Examples
- **`example_trace.json`** - Sample trace file
- **`replay_traces.py`** - Alternative CLI script (legacy)

## Usage Modes

### Mode 1: Standalone Script (Recommended)

Best for batch testing and CI/CD integration.

```bash
# Run single trace
python test_unified_attention.py --trace-file trace.json --verbose

# Run all traces
python test_unified_attention.py --trace-dir ./traces --verbose

# With error details
python test_unified_attention.py --trace-file trace.json --show-errors
```

**Exit codes:**
- `0` = All tests passed
- `1` = One or more tests failed
- `130` = Interrupted by user

### Mode 2: Pytest Integration

Best for test discovery and parameterized testing.

```bash
# Run all parametrized tests
pytest test_unified_attention.py::test_triton_unified_attn -v

# Run specific trace
pytest test_unified_attention.py::test_triton_unified_attn_from_trace \
  --trace-file=trace.json
```

### Mode 3: Python API

Best for programmatic usage.

```python
from test_unified_attention import (
    run_trace_from_json,
    run_all_traces_in_directory
)

# Single trace
run_trace_from_json("./traces/trace_123.json")

# All traces
passed, failed = run_all_traces_in_directory("./traces")
```

## Environment Variables

### Tracing Control

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `AITER_TRACE_ATTENTION` | `0` or `1` | `0` | Enable/disable tracing |
| `AITER_TRACE_ATTENTION_DIR` | path | `./attention_traces` | Where to save traces |

### Example

```bash
# Enable tracing with custom directory
export AITER_TRACE_ATTENTION=1
export AITER_TRACE_ATTENTION_DIR=/data/production_traces

# Run inference
python run_llama.py

# Check traces
ls /data/production_traces/
# trace_1708790400000000_nh32_hs128_bs16.json
# trace_1708790401000000_nh32_hs128_bs16.json
```

## Trace File Format

Each trace captures the exact parameters needed to reproduce a test case:

```json
{
  "seq_lens": [[query_len, kv_len], ...],
  "num_heads": [num_query_heads, num_kv_heads],
  "head_size": 128,
  "sliding_window": null,
  "dtype": "float16",
  "block_size": 16,
  "soft_cap": null,
  "num_blocks": 2048,
  "q_dtype": null
}
```

**File naming:** `trace_{timestamp}_nh{heads}_hs{size}_bs{block}.json`

## Common Workflows

### Workflow 1: Debug Production Issue

```bash
# 1. Enable tracing in staging/production
export AITER_TRACE_ATTENTION=1
python run_production_workload.py

# 2. Find the problematic trace
ls -lt ./attention_traces/ | head

# 3. Reproduce locally
python test_unified_attention.py \
  --trace-file ./attention_traces/trace_problematic.json \
  --verbose --show-errors
```

### Workflow 2: Performance Regression Testing

```bash
# 1. Collect baseline traces
export AITER_TRACE_ATTENTION=1
export AITER_TRACE_ATTENTION_DIR=./baseline_traces
python benchmark_suite.py

# 2. After code changes, validate correctness
python test_unified_attention.py \
  --trace-dir ./baseline_traces \
  --verbose

# 3. If all pass, run performance comparison
# (run your performance benchmarks here)
```

### Workflow 3: CI/CD Integration

```bash
# .github/workflows/test.yml
- name: Test Attention with Golden Traces
  run: |
    cd gptoss_benchmark
    python test_unified_attention.py \
      --trace-dir ../golden_traces \
      --verbose
```

### Workflow 4: Building Golden Test Suite

```bash
# 1. Trace various model configurations
for model in llama-7b llama-13b llama-70b; do
  export AITER_TRACE_ATTENTION=1
  export AITER_TRACE_ATTENTION_DIR=./golden_traces/${model}
  python run_inference.py --model ${model} --prompts test_prompts.txt
done

# 2. Review and curate traces
python test_unified_attention.py --trace-dir ./golden_traces/llama-7b

# 3. Check into version control
git add golden_traces/
git commit -m "Add golden traces for attention tests"
```

## Advanced Usage

### Selective Tracing

```python
# In your code, enable tracing conditionally
import os

# Only trace specific conditions
if should_trace_this_case:
    os.environ["AITER_TRACE_ATTENTION"] = "1"
    unified_attention(...)
    os.environ["AITER_TRACE_ATTENTION"] = "0"
```

### Batch Processing Script

```bash
#!/bin/bash
# test_all_models.sh

MODELS=(
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-2-13b-hf"
  "mistralai/Mistral-7B-v0.1"
)

for model in "${MODELS[@]}"; do
  echo "Testing traces for: $model"
  model_name=$(echo $model | tr '/' '_')

  if python test_unified_attention.py \
       --trace-dir "./traces/${model_name}" \
       --verbose; then
    echo "✓ $model: ALL TESTS PASSED"
  else
    echo "✗ $model: SOME TESTS FAILED"
    exit 1
  fi
done
```

### Filtering Traces

```bash
# Test only specific configurations
for trace in ./traces/trace_*_nh32_*.json; do
  echo "Testing: $trace"
  python test_unified_attention.py --trace-file "$trace"
done

# Test traces from specific time range
find ./traces -name "trace_170879*.json" -exec \
  python test_unified_attention.py --trace-file {} \;
```

## Performance Impact

| Operation | Overhead | Notes |
|-----------|----------|-------|
| Tracing disabled | ~0 ns | Single env var check |
| Tracing enabled | ~1-2 ms | File I/O per call |
| Trace file size | ~200-500 bytes | Very small |
| Test replay | Same as original | Full computation |

## Tips & Best Practices

1. **Organize by scenario**: Create subdirectories for different workloads
   ```
   traces/
   ├── prefill/
   ├── decode/
   ├── mixed_batch/
   └── edge_cases/
   ```

2. **Use descriptive names**: Automatic naming includes key parameters
   - `trace_1708790400_nh32_hs128_bs16.json` tells you the config at a glance

3. **Curate your traces**: Not every trace needs to be a test
   - Keep representative samples
   - Focus on edge cases and common patterns

4. **Version control golden traces**: Check in important test cases
   ```bash
   git add golden_traces/critical_*.json
   ```

5. **CI/CD integration**: Run traces on every PR
   ```yaml
   test:
     script:
       - python test_unified_attention.py --trace-dir golden_traces
   ```

6. **Monitor trace counts**: Too many traces = slow tests
   - Keep golden set small (<100 traces)
   - Use separate directories for exploratory traces

## Troubleshooting

### Trace files not created

```bash
# Check environment variable
echo $AITER_TRACE_ATTENTION  # Should be "1"

# Check directory permissions
ls -ld ./attention_traces/

# Check for errors in logs
grep "AITER_TRACE" your_log_file.txt
```

### Test fails on replay

```bash
# Get full error details
python test_unified_attention.py \
  --trace-file trace.json \
  --show-errors

# Common issues:
# - CUDA OOM: Trace captured on larger GPU
# - Different CUDA version: Numerical differences
# - Missing dependencies: Check requirements
```

### Different results on replay

This is normal for:
- Different CUDA versions
- Different GPU architectures
- FP8 quantization (lower precision)

Tolerances are built-in:
- FP16/BF16: `atol=1.5e-2, rtol=1e-2`
- FP8: `atol=1.5e-1, rtol=1.5e-1`

## FAQ

**Q: Does tracing affect inference performance?**
A: Negligible when disabled (<1ns). ~1-2ms per call when enabled.

**Q: Can I trace in production?**
A: Yes, but use selectively. Enable for short periods to capture specific cases.

**Q: How many traces should I keep?**
A: 10-50 golden traces for CI/CD. More for exploratory analysis.

**Q: Can I modify trace files?**
A: Yes! They're just JSON. Useful for creating synthetic test cases.

**Q: What if my trace is too large?**
A: Traces are tiny (<1KB). If your trace is large, something is wrong.

**Q: Can I trace other operators?**
A: Yes! Copy the pattern from `unified_attention.py` to other ops.

## Summary

```bash
# Complete workflow
export AITER_TRACE_ATTENTION=1                    # Enable
python run_inference.py                           # Capture
python test_unified_attention.py --trace-dir .    # Replay
```

**Three simple steps to production debugging and regression testing!**

---

For more details:
- **CLI usage**: See `STANDALONE_USAGE.md`
- **Tracing details**: See `TRACE_USAGE.md`
- **Implementation**: See `unified_attention.py` and `test_unified_attention.py`
