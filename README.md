# ngehtsim

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://smithsonian.github.io/ngehtsim/)
[![Build status](https://github.com/Smithsonian/ngehtsim/actions/workflows/run_unit_tests.yml/badge.svg)](https://github.com/Smithsonian/ngehtsim/actions)
[![Python versions](https://img.shields.io/badge/python-3.11|3.12-blue.svg)](https://github.com/Smithsonian/ngehtsim)
[![Code coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/dpesce/de229c48256d967a1cb8b29dbf575602/raw/covbadge.json)](https://github.com/Smithsonian/ngehtsim/actions)

A set of tools for generating synthetic data for the Event Horizon Telescope ([EHT](https://eventhorizontelescope.org/)), the next-generation Event Horizon Telescope ([ngEHT](https://www.ngeht.org)), and other very long baseline interferometric arrays of radio telescopes.

# Getting started

Details about installing and using ngehtsim can be found in the [online documentation](https://smithsonian.github.io/ngehtsim/).

Note that ngehtsim currently only runs on Unix-based systems (e.g., macOS, Linux) due to dependency restrictions.  Windows installations are thus not currently supported.

## Setting up the environment

ngehtsim requires Python 3.11 or later. It is recommended that you install ngehtsim using a virtual environment, e.g.:

```
    $ git clone https://github.com/Smithsonian/ngehtsim
    $ python -m venv .venv
    $ source .venv/bin/activate
    (.venv) $ python -m pip install --upgrade pip
    (.venv) $ pip install ./ngehtsim
```

There is an optional calibration functionality that requires some additional dependencies; it can be installed using:

```
    (.venv) $ pip install ./ngehtsim[calib]
```

The external versioned weather datasets use Zarr and can be read with:

```
    (.venv) $ pip install ./ngehtsim[weather-zarr]
```

## Checking that it works

There are a number of example scripts contained in the [examples](./examples/) folder.  You can check to make sure your ngehtsim installation is working by running one of these scripts, e.g.:

```
    (.venv) $ cd ./ngehtsim/examples/example_data_generation
    (.venv) $ python ./generate_observation.py
```

# Versioning scheme

For this repository, we attempt to adhere to the `major.minor.patch` [Semantic Versioning](https://semver.org) numbering scheme.

# Licensing

See the [LICENSE](./LICENSE) file for details on the licensing of this software.
