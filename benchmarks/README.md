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

## Compare Weather Backends

The default is the packaged daily weather backend, preserving the original
benchmark behavior. To compare it with a local external Zarr weather release,
pass its explicit path and select each backend. The runner creates an
independent Zarr store for each backend so that one backend's partition cache
cannot warm another's results.

```bash
WEATHER_STORE=/path/to/ngehtsim-weather-merra2-3hour-v0.1.0.zarr

MPLBACKEND=Agg python3 benchmarks/benchmark_synthetic_data.py \
  --scenario eht2017_model_clean \
  --scenario ngeht_model_clean \
  --repeats 3 \
  --warmups 1 \
  --weather-store "$WEATHER_STORE" \
  --weather-backend legacy \
  --weather-backend zarr-daily \
  --weather-backend zarr-native
```

The JSON result records the backend, release path, and time to open each Zarr
store separately from generator initialization. Zarr benchmark runs require
the optional weather dependency:

```bash
python3 -m pip install "ngehtsim[weather-zarr]"
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

Profile one external-weather backend at a time. For example, profile native
three-hour weather during generator initialization:

```bash
MPLBACKEND=Agg python3 benchmarks/profile_synthetic_data.py \
  --scenario eht2017_model_clean \
  --phase init \
  --repeats 3 \
  --warmups 0 \
  --weather-store "$WEATHER_STORE" \
  --weather-backend zarr-native
```

The profiler writes both a raw `.prof` file and a readable `.txt` summary under `benchmarks/results/`. The `.txt` summary is usually the first file to inspect. The `.prof` file can be opened with tools such as `snakeviz` if desired.
