# Benchmarks

This directory contains opt-in runtime benchmarks and profiling tools for representative synthetic-data generation workflows.

The benchmark and profiling runners are intentionally not part of the default unit-test suite. Results depend on hardware, Python version, installed dependency versions, and local system state.

## List Scenarios

```bash
MPLBACKEND=Agg python3 benchmarks/benchmark_synthetic_data.py --list-scenarios
```

## Run Timing Benchmarks

Run all scenarios with the default repeat and warmup counts:

```bash
MPLBACKEND=Agg python3 benchmarks/benchmark_synthetic_data.py
```

Run one scenario with more repeats:

```bash
MPLBACKEND=Agg python3 benchmarks/benchmark_synthetic_data.py --repeats 5 --warmups 1 --scenario eht2017_model_clean_reused_generator
```

Benchmark outputs are written to `benchmarks/results/`.

## Profile Synthetic Data Generation

Use the profiler when you need call-stack detail about where time is being spent. The profiler uses the same scenario definitions as the timing benchmark.

Profile the steady-state `make_obs()` phase for the reused-generator scenario:

```bash
MPLBACKEND=Agg python3 benchmarks/profile_synthetic_data.py --scenario eht2017_model_clean_reused_generator --phase make_obs --repeats 3 --warmups 1
```

Profile generator initialization:

```bash
MPLBACKEND=Agg python3 benchmarks/profile_synthetic_data.py --scenario eht2017_model_clean --phase init --repeats 3 --warmups 0
```

Profile initialization and observation generation together:

```bash
MPLBACKEND=Agg python3 benchmarks/profile_synthetic_data.py --scenario eht2017_model_clean --phase full --repeats 3 --warmups 1
```

The profiler writes both a raw `.prof` file and a readable `.txt` summary under `benchmarks/results/`. The `.txt` summary is usually the first file to inspect. The `.prof` file can be opened with tools such as `snakeviz` if desired.
