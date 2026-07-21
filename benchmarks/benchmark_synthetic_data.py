#!/usr/bin/env python3
"""Benchmark representative ngehtsim synthetic-data generation workflows."""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "ngehtsim").exists():
    REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

import ehtim as eh
import ngehtsim
import ngehtsim.obs.obs_generator as og


def base_settings(array, dt=6.0, t_rest=1200.0):
    return {
        "source": "M87",
        "array": array,
        "frequency": 230.0,
        "bandwidth": 2.0,
        "month": "Apr",
        "day": "11",
        "year": "2017",
        "weather": "typical",
        "t_start": 0.0,
        "dt": dt,
        "t_int": 600.0,
        "t_rest": t_rest,
        "fringe_finder": ["fringegroups", [5.0, 10.0]],
        "transform_backend": "auto",
        "raster_tolerance": 1.0e-12,
        "random_seed": 12345,
    }


SCENARIOS = [
    {
        "name": "eht2017_model_clean",
        "description": "EHT2017 array, analytic ehtim Model source, no thermal noise or gain corruptions.",
        "settings": base_settings("EHT2017"),
        "input_kind": "model",
        "make_obs_kwargs": {"addnoise": False, "addgains": False},
        "reuse_generator": False,
    },
    {
        "name": "eht2017_model_exact_weather",
        "description": "EHT2017 array, analytic ehtim Model source, exact-date weather, and a fresh generator.",
        "settings": {**base_settings("EHT2017"), "weather": "exact"},
        "input_kind": "model",
        "make_obs_kwargs": {"addnoise": False, "addgains": False},
        "reuse_generator": False,
    },
    {
        "name": "eht2017_model_corruptions",
        "description": "EHT2017 array, analytic ehtim Model source, thermal noise and gain corruptions enabled.",
        "settings": base_settings("EHT2017"),
        "input_kind": "model",
        "make_obs_kwargs": {"addnoise": True, "addgains": True},
        "reuse_generator": False,
    },
    {
        "name": "eht2017_image_clean",
        "description": "EHT2017 array, rasterized ehtim Image source, no thermal noise or gain corruptions.",
        "settings": base_settings("EHT2017"),
        "input_kind": "image",
        "make_obs_kwargs": {"addnoise": False, "addgains": False},
        "reuse_generator": False,
    },
    {
        "name": "eht2017_image_corruptions",
        "description": "EHT2017 array, rasterized ehtim Image source, thermal noise and gain corruptions enabled.",
        "settings": base_settings("EHT2017"),
        "input_kind": "image",
        "make_obs_kwargs": {"addnoise": True, "addgains": True},
        "reuse_generator": False,
    },
    {
        "name": "eht2017_movie_clean",
        "description": "EHT2017 array, time-varying ehtim Movie source, no thermal noise or gain corruptions.",
        "settings": base_settings("EHT2017"),
        "input_kind": "movie",
        "make_obs_kwargs": {"addnoise": False, "addgains": False},
        "reuse_generator": False,
    },
    {
        "name": "eht2017_movie_corruptions",
        "description": "EHT2017 array, time-varying ehtim Movie source, thermal noise and gain corruptions enabled.",
        "settings": base_settings("EHT2017"),
        "input_kind": "movie",
        "make_obs_kwargs": {"addnoise": True, "addgains": True},
        "reuse_generator": False,
    },
    {
        "name": "ngeht_model_clean",
        "description": "ngEHT array, analytic ehtim Model source, no thermal noise or gain corruptions.",
        "settings": base_settings("ngEHT", dt=3.0, t_rest=1800.0),
        "input_kind": "model",
        "make_obs_kwargs": {"addnoise": False, "addgains": False},
        "reuse_generator": False,
    },
    {
        "name": "eht2017_model_clean_reused_generator",
        "description": "Repeated clean EHT2017 Model observations using one obs_generator instance.",
        "settings": base_settings("EHT2017"),
        "input_kind": "model",
        "make_obs_kwargs": {"addnoise": False, "addgains": False},
        "reuse_generator": True,
    },
]

WEATHER_BACKENDS = ("legacy", "zarr-daily", "zarr-native")
TRANSFORM_BACKENDS = ("auto", "direct", "finufft")


def add_weather_backend_arguments(parser):
    """Add optional external-weather benchmark arguments to ``parser``."""

    parser.add_argument(
        "--weather-store",
        type=Path,
        default=None,
        help="Path to a local Zarr weather release.",
    )
    parser.add_argument(
        "--weather-backend",
        action="append",
        choices=WEATHER_BACKENDS,
        default=None,
        help="Weather backend to benchmark. May be passed more than once.",
    )


def prepare_weather_backends(weather_backends, weather_store_path):
    """Create independent runtime weather configurations for each backend."""

    backend_names = weather_backends or ["legacy"]
    backend_names = list(dict.fromkeys(backend_names))
    zarr_backends = {"zarr-daily", "zarr-native"}

    if not zarr_backends.intersection(backend_names):
        return [
            {
                "name": "legacy",
                "store": None,
                "store_path": None,
                "store_init_seconds": 0.0,
                "obs_generator_kwargs": {},
            }
        ]

    if weather_store_path is None:
        raise SystemExit("--weather-store is required for Zarr weather backends.")

    weather_store_path = weather_store_path.expanduser().resolve()
    if not weather_store_path.is_dir():
        raise SystemExit(
            "Zarr weather dataset directory does not exist: {0}".format(weather_store_path)
        )

    try:
        from ngehtsim.weather.zarr_store import ZarrWeatherStore
    except ImportError as exc:
        raise SystemExit(
            "Zarr weather benchmarks require `pip install \"ngehtsim[weather-zarr]\"`."
        ) from exc

    backends = []
    for backend_name in backend_names:
        if backend_name == "legacy":
            backends.append(
                {
                    "name": backend_name,
                    "store": None,
                    "store_path": None,
                    "store_init_seconds": 0.0,
                    "obs_generator_kwargs": {},
                }
            )
            continue

        t0 = time.perf_counter()
        store = ZarrWeatherStore(weather_store_path)
        store_init_seconds = time.perf_counter() - t0
        backends.append(
            {
                "name": backend_name,
                "store": store,
                "store_path": str(weather_store_path),
                "store_init_seconds": store_init_seconds,
                "obs_generator_kwargs": {
                    "weather_store": store,
                    "weather_cadence": (
                        "daily" if backend_name == "zarr-daily" else "native"
                    ),
                },
            }
        )

    return backends


def expand_scenarios(scenarios, weather_backends):
    """Return one executable scenario for every scenario/backend combination."""

    expanded = []
    for scenario in scenarios:
        for backend in weather_backends:
            expanded_scenario = dict(scenario)
            expanded_scenario["settings"] = dict(scenario["settings"])
            expanded_scenario["name"] = "{0}__{1}".format(
                scenario["name"], backend["name"]
            )
            expanded_scenario["description"] = "{0} Weather backend: {1}.".format(
                scenario["description"], backend["name"]
            )
            expanded_scenario["weather_backend"] = backend["name"]
            expanded_scenario["weather_store_path"] = backend["store_path"]
            expanded_scenario["weather_store_init_seconds"] = backend["store_init_seconds"]
            expanded_scenario["_obs_generator_kwargs"] = backend["obs_generator_kwargs"]
            expanded.append(expanded_scenario)
    return expanded


def with_transform_backend(scenarios, transform_backend):
    """Return scenario copies that use an explicit raster transform backend."""

    if transform_backend is None:
        return list(scenarios)

    return [
        {
            **scenario,
            "settings": {
                **scenario["settings"],
                "transform_backend": transform_backend,
            },
        }
        for scenario in scenarios
    ]


def scenario_definition(scenario):
    """Return a JSON-serializable benchmark scenario description."""

    return {key: value for key, value in scenario.items() if not key.startswith("_")}


def make_obs_generator(scenario):
    """Construct an observation generator for one benchmark scenario."""

    return og.obs_generator(
        settings=dict(scenario["settings"]),
        **scenario["_obs_generator_kwargs"],
    )


def make_source(input_kind):
    model = eh.model.Model()
    model = model.add_circ_gauss(F0=1.0, FWHM=40.0 * eh.RADPERUAS)

    if input_kind == "model":
        return model
    if input_kind == "image":
        return model.make_image(160.0 * eh.RADPERUAS, 128)
    if input_kind == "movie":
        image = model.make_image(160.0 * eh.RADPERUAS, 64)
        frame_0 = image.imvec.reshape(image.ydim, image.xdim)
        frame_1 = 1.1 * frame_0
        movie = eh.movie.Movie(
            (frame_0, frame_1),
            times=(0.0, 0.5),
            psize=image.psize,
            ra=image.ra,
            dec=image.dec,
            rf=image.rf,
            source=image.source,
            mjd=image.mjd,
            bounds_error=True,
        )
        for polarization in ("Q", "U", "V"):
            movie.add_pol_movie((np.zeros_like(frame_0), np.zeros_like(frame_1)), polarization)
        return movie

    raise ValueError("Unknown input_kind: {0}".format(input_kind))


def summarize(values):
    if len(values) == 0:
        return {"count": 0}

    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def git_value(*args):
    try:
        return subprocess.check_output(["git"] + list(args), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def metadata():
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "ngehtsim_version": getattr(ngehtsim, "__version__", "unknown"),
        "ehtim_version": getattr(eh, "__version__", "unknown"),
        "git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--short")),
    }


def run_fresh_iteration(scenario):
    input_model = make_source(scenario["input_kind"])

    t0 = time.perf_counter()
    obsgen = make_obs_generator(scenario)
    t1 = time.perf_counter()
    obs = obsgen.make_obs(input_model, **scenario["make_obs_kwargs"])
    t2 = time.perf_counter()

    return {
        "obsgen_init_seconds": t1 - t0,
        "make_obs_seconds": t2 - t1,
        "rows": int(len(obs.data)),
    }


def run_reused_generator_iterations(scenario, repeats, warmups=0):
    t0 = time.perf_counter()
    obsgen = make_obs_generator(scenario)
    t1 = time.perf_counter()
    obsgen_init_seconds = t1 - t0

    for _ in range(warmups):
        input_model = make_source(scenario["input_kind"])
        obsgen.make_obs(input_model, **scenario["make_obs_kwargs"])

    results = []
    for index in range(repeats):
        input_model = make_source(scenario["input_kind"])
        t2 = time.perf_counter()
        obs = obsgen.make_obs(input_model, **scenario["make_obs_kwargs"])
        t3 = time.perf_counter()
        results.append(
            {
                "obsgen_init_seconds": obsgen_init_seconds if index == 0 else 0.0,
                "make_obs_seconds": t3 - t2,
                "rows": int(len(obs.data)),
            }
        )

    return results


def run_scenario(scenario, repeats, warmups):
    print("Running {0}...".format(scenario["name"]), flush=True)

    if scenario["reuse_generator"]:
        raw_results = run_reused_generator_iterations(scenario, repeats, warmups=warmups)
    else:
        for _ in range(warmups):
            run_fresh_iteration(scenario)
        raw_results = [run_fresh_iteration(scenario) for _ in range(repeats)]

    init_seconds = [result["obsgen_init_seconds"] for result in raw_results]
    make_obs_seconds = [result["make_obs_seconds"] for result in raw_results]
    rows = [result["rows"] for result in raw_results]

    result = {
        "name": scenario["name"],
        "description": scenario["description"],
        "settings": scenario["settings"],
        "input_kind": scenario["input_kind"],
        "make_obs_kwargs": scenario["make_obs_kwargs"],
        "reuse_generator": scenario["reuse_generator"],
        "weather_backend": scenario["weather_backend"],
        "weather_store_path": scenario["weather_store_path"],
        "weather_store_init_seconds": scenario["weather_store_init_seconds"],
        "rows": rows,
        "obsgen_init_seconds": summarize(init_seconds),
        "make_obs_seconds": summarize(make_obs_seconds),
        "raw_results": raw_results,
    }

    print(
        "  init median: {0:.4f}s; make_obs median: {1:.4f}s; rows: {2}".format(
            result["obsgen_init_seconds"]["median"],
            result["make_obs_seconds"]["median"],
            rows,
        ),
        flush=True,
    )
    return result


def select_scenarios(names):
    if not names:
        return SCENARIOS

    scenarios_by_name = {scenario["name"]: scenario for scenario in SCENARIOS}
    missing = sorted(set(names) - set(scenarios_by_name))
    if missing:
        raise SystemExit("Unknown scenario(s): {0}".format(", ".join(missing)))

    return [scenarios_by_name[name] for name in names]


def default_output_path():
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return Path("benchmarks") / "results" / "synthetic_data_baseline_{0}.json".format(timestamp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Measured repeats per scenario.")
    parser.add_argument("--warmups", type=int, default=1, help="Unmeasured warmup repeats per scenario.")
    parser.add_argument("--scenario", action="append", help="Scenario name to run. May be passed more than once.")
    parser.add_argument(
        "--transform-backend",
        choices=TRANSFORM_BACKENDS,
        default=None,
        help="Override the raster transform backend for selected scenarios.",
    )
    parser.add_argument("--output", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios and exit.")
    add_weather_backend_arguments(parser)
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")

    if args.list_scenarios:
        for scenario in SCENARIOS:
            print("{0}: {1}".format(scenario["name"], scenario["description"]))
        print("Weather backends: {0}".format(", ".join(WEATHER_BACKENDS)))
        return 0

    weather_backends = prepare_weather_backends(args.weather_backend, args.weather_store)
    scenarios = with_transform_backend(
        select_scenarios(args.scenario),
        args.transform_backend,
    )
    scenarios = expand_scenarios(scenarios, weather_backends)
    output_path = args.output or default_output_path()

    payload = {
        "metadata": metadata(),
        "config": {
            "repeats": args.repeats,
            "warmups": args.warmups,
            "scenario_names": [scenario["name"] for scenario in scenarios],
            "weather_backends": [backend["name"] for backend in weather_backends],
            "transform_backend": args.transform_backend or "auto",
        },
        "scenario_definitions": [scenario_definition(scenario) for scenario in scenarios],
        "scenarios": [run_scenario(scenario, args.repeats, args.warmups) for scenario in scenarios],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("Benchmark results written to {0}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
