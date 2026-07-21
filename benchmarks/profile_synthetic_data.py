#!/usr/bin/env python3
"""Profile representative ngehtsim synthetic-data generation workflows."""

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "ngehtsim").exists():
    REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import benchmark_synthetic_data as bench


def default_output_paths(scenario_name, phase):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = "synthetic_data_profile_{0}_{1}_{2}".format(
        scenario_name,
        phase,
        timestamp,
    )
    output_dir = Path("benchmarks") / "results"
    return output_dir / "{0}.prof".format(stem), output_dir / "{0}.txt".format(stem)


def warmup_make_obs(scenario, count):
    if count <= 0:
        return

    if scenario["reuse_generator"]:
        obsgen = bench.make_obs_generator(scenario)
        for _ in range(count):
            input_model = bench.make_source(scenario["input_kind"])
            obsgen.make_obs(input_model, **scenario["make_obs_kwargs"])
        return

    for _ in range(count):
        input_model = bench.make_source(scenario["input_kind"])
        obsgen = bench.make_obs_generator(scenario)
        obsgen.make_obs(input_model, **scenario["make_obs_kwargs"])


def make_init_target(scenario, repeats):
    def target():
        rows = []
        for _ in range(repeats):
            bench.make_obs_generator(scenario)
            rows.append(0)
        return rows

    return target


def make_make_obs_target(scenario, repeats, warmups):
    if scenario["reuse_generator"]:
        obsgen = bench.make_obs_generator(scenario)
        for _ in range(warmups):
            input_model = bench.make_source(scenario["input_kind"])
            obsgen.make_obs(input_model, **scenario["make_obs_kwargs"])

        input_models = [bench.make_source(scenario["input_kind"]) for _ in range(repeats)]

        def target():
            rows = []
            for input_model in input_models:
                obs = obsgen.make_obs(input_model, **scenario["make_obs_kwargs"])
                rows.append(int(len(obs.data)))
            return rows

        return target

    warmup_make_obs(scenario, warmups)
    obs_inputs = [
        (
            bench.make_obs_generator(scenario),
            bench.make_source(scenario["input_kind"]),
        )
        for _ in range(repeats)
    ]

    def target():
        rows = []
        for obsgen, input_model in obs_inputs:
            obs = obsgen.make_obs(input_model, **scenario["make_obs_kwargs"])
            rows.append(int(len(obs.data)))
        return rows

    return target


def make_full_target(scenario, repeats, warmups):
    def target():
        if scenario["reuse_generator"]:
            return [
                result["rows"]
                for result in bench.run_reused_generator_iterations(scenario, repeats, warmups=warmups)
            ]

        for _ in range(warmups):
            bench.run_fresh_iteration(scenario)
        return [
            result["rows"]
            for result in [bench.run_fresh_iteration(scenario) for _ in range(repeats)]
        ]

    return target


def profile_target(target, sort, limit, profile_output, stats_output, metadata):
    profiler = cProfile.Profile()

    start = time.perf_counter()
    rows = profiler.runcall(target)
    elapsed = time.perf_counter() - start

    profile_output.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(profile_output))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats(sort).print_stats(limit)

    header = [
        "scenario: {0}".format(metadata["scenario"]),
        "phase: {0}".format(metadata["phase"]),
        "weather_backend: {0}".format(metadata["weather_backend"]),
        "weather_store_path: {0}".format(metadata["weather_store_path"]),
        "repeats: {0}".format(metadata["repeats"]),
        "warmups: {0}".format(metadata["warmups"]),
        "sort: {0}".format(sort),
        "limit: {0}".format(limit),
        "elapsed_seconds: {0:.6f}".format(elapsed),
        "rows: {0}".format(rows),
        "profile_output: {0}".format(profile_output),
        "",
    ]
    stats_output.write_text("\n".join(header) + stream.getvalue())

    return rows, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="eht2017_model_clean_reused_generator")
    parser.add_argument("--phase", choices=["init", "make_obs", "full"], default="make_obs")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--sort", default="cumtime")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--profile-output", type=Path, default=None)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument(
        "--transform-backend",
        choices=bench.TRANSFORM_BACKENDS,
        default=None,
        help="Override the raster transform backend for the selected scenario.",
    )
    bench.add_weather_backend_arguments(parser)
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")

    if args.list_scenarios:
        for scenario in bench.SCENARIOS:
            print("{0}: {1}".format(scenario["name"], scenario["description"]))
        print("Weather backends: {0}".format(", ".join(bench.WEATHER_BACKENDS)))
        return 0

    if args.weather_backend is not None and len(set(args.weather_backend)) != 1:
        raise SystemExit("The profiler accepts exactly one --weather-backend at a time.")

    weather_backends = bench.prepare_weather_backends(args.weather_backend, args.weather_store)
    scenarios = bench.with_transform_backend(
        bench.select_scenarios([args.scenario]),
        args.transform_backend,
    )
    scenarios = bench.expand_scenarios(scenarios, weather_backends)
    scenario = scenarios[0]
    profile_output, stats_output = default_output_paths(scenario["name"], args.phase)
    if args.profile_output is not None:
        profile_output = args.profile_output
    if args.stats_output is not None:
        stats_output = args.stats_output

    if args.phase == "init":
        target = make_init_target(scenario, args.repeats)
    elif args.phase == "make_obs":
        target = make_make_obs_target(scenario, args.repeats, args.warmups)
    else:
        target = make_full_target(scenario, args.repeats, args.warmups)

    rows, elapsed = profile_target(
        target,
        args.sort,
        args.limit,
        profile_output,
        stats_output,
        {
            "scenario": scenario["name"],
            "phase": args.phase,
            "weather_backend": scenario["weather_backend"],
            "weather_store_path": scenario["weather_store_path"],
            "repeats": args.repeats,
            "warmups": args.warmups,
        },
    )

    print("Profiled {0} [{1}] in {2:.4f}s; rows: {3}".format(
        scenario["name"],
        args.phase,
        elapsed,
        rows,
    ))
    print("Raw profile written to {0}".format(profile_output))
    print("Text summary written to {0}".format(stats_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
