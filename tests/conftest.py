#######################################################
# imports

import os

#######################################################
# configuration testing

SLOW_EXAMPLE_TESTS = [
    "test_SYMBA_export.py",
    "test_basic.py",
    "test_flux_calibration.py",
    "test_generate_observation.py",
    "test_generate_observation_flagging.py",
    "test_generate_observation_instrumental_corruptions.py",
    "test_generate_observation_multi-frequency.py",
    "test_generate_observation_space.py",
    "test_generate_observation_with_FPT.py",
    "test_weather.py",
]

OPTIONAL_TESTS = [
    "test_generate_observation_using_ngEHTforecast.py",
]

collect_ignore = []

if os.environ.get("NGEHTSIM_RUN_SLOW_TESTS") != "1":
    collect_ignore.extend(SLOW_EXAMPLE_TESTS)

if os.environ.get("NGEHTSIM_RUN_OPTIONAL_TESTS") != "1":
    collect_ignore.extend(OPTIONAL_TESTS)
