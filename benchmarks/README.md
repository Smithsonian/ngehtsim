# Benchmarks

This directory contains opt-in runtime benchmarks for representative synthetic-data generation workflows.

The benchmark runner is intentionally not part of the default unit-test suite. Results depend on hardware, Python version, installed dependency versions, and local system state.

## List Scenarios

```bash
MPLBACKEND=Agg python3 benchmarks/benchmark_synthetic_data.py --list-scenarios