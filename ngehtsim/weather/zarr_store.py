"""Read versioned local Zarr weather datasets.

This module intentionally has no dependency on the legacy binary weather API.
It is an opt-in reader for externally distributed weather releases; callers pass
an explicit local filesystem path to :class:`ZarrWeatherStore`.
"""

from __future__ import annotations

import calendar
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


SCHEMA_VERSION = "0.1.0"
Cadence = Literal["daily", "native"]

_MONTH_NAMES = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
_CADENCES = ("daily", "native")
_SCALAR_ARRAYS = (
    "pwv_mm",
    "wind_speed_m_s",
    "surface_pressure_mbar",
    "surface_temperature_k",
)
_PARTITION_CACHE_SIZE = 64


class WeatherStoreError(ValueError):
    """Raised when a Zarr weather dataset does not meet the supported schema."""


@dataclass(frozen=True)
class WeatherPartition:
    """Weather records for one site, month, and sampling cadence."""

    site: str
    month: int
    cadence: Cadence
    year: np.ndarray
    day: np.ndarray
    time_index: np.ndarray | None
    tau_coefficients: np.ndarray
    tb_coefficients: np.ndarray
    pwv_mm: np.ndarray
    wind_speed_m_s: np.ndarray
    surface_pressure_mbar: np.ndarray
    surface_temperature_k: np.ndarray

    @property
    def record_count(self) -> int:
        """Return the number of weather records in this partition."""

        return len(self.year)


class ZarrWeatherStore:
    """Read a local weather dataset produced by ``ngehtsim-weather-builder``.

    Args:
        path: Local path to the root ``.zarr`` directory.

    Raises:
        ImportError: If the optional Zarr dependency is not installed.
        FileNotFoundError: If ``path`` is not a local directory.
        WeatherStoreError: If the dataset does not match the supported schema.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        if not self.path.is_dir():
            raise FileNotFoundError(
                f"Zarr weather dataset directory does not exist: {self.path}"
            )

        try:
            import zarr
        except ImportError as exc:
            raise ImportError(
                "Zarr weather support requires the optional dependency. Install "
                'ngehtsim with `pip install "ngehtsim[weather-zarr]"`. '
            ) from exc

        self._root = zarr.open_group(self.path, mode="r")
        self._pca_bases: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._partition_cache: OrderedDict[
            tuple[str, int, Cadence], WeatherPartition
        ] = OrderedDict()
        self._validate_root()

    @property
    def attributes(self) -> dict[str, object]:
        """Return a copy of the dataset-level metadata attributes."""

        return dict(self._root.attrs)

    @property
    def dataset_id(self) -> str:
        """Return the builder-supplied immutable dataset identifier."""

        return str(self._root.attrs["dataset_id"])

    @property
    def sites(self) -> tuple[str, ...]:
        """Return all weather site identifiers in deterministic order."""

        return self._sites

    @property
    def frequency_ghz(self) -> np.ndarray:
        """Return the common frequency grid in GHz."""

        return self._frequency_ghz

    def read_partition(
        self, site: str, month: str | int, cadence: Cadence = "daily"
    ) -> WeatherPartition:
        """Read all records for one site, calendar month, and cadence.

        ``month`` accepts the same three-letter abbreviations and numeric values
        as the established binary weather functions (for example ``"Apr"``,
        ``"04"``, and ``4``).
        """

        month_number = _parse_month(month)
        if cadence not in _CADENCES:
            allowed = ", ".join(_CADENCES)
            raise ValueError(f"Unsupported weather cadence {cadence!r}; use {allowed}.")
        if site not in self.sites:
            raise KeyError(f"Weather dataset does not contain site {site!r}.")

        key = (site, month_number, cadence)
        try:
            partition = self._partition_cache.pop(key)
        except KeyError:
            partition = self._read_partition(site, month_number, cadence)
            if len(self._partition_cache) >= _PARTITION_CACHE_SIZE:
                self._partition_cache.popitem(last=False)
        self._partition_cache[key] = partition
        return partition

    def _read_partition(
        self, site: str, month_number: int, cadence: Cadence
    ) -> WeatherPartition:
        """Read and validate one immutable Zarr partition."""

        prefix = f"sites/{site}/months/{month_number:02d}/{cadence}"
        if prefix not in self._root:
            raise WeatherStoreError(
                f"Weather dataset is missing the {cadence} partition for "
                f"{site!r}, month {month_number:02d}."
            )

        group = self._root[prefix]
        required = ["year", "day", "tau_coefficients", "tb_coefficients", *_SCALAR_ARRAYS]
        if cadence == "native":
            required.append("time_index")
        missing = [name for name in required if name not in group]
        if missing:
            raise WeatherStoreError(
                f"Weather partition {prefix!r} is missing arrays: {', '.join(missing)}."
            )
        if cadence == "daily" and "time_index" in group:
            raise WeatherStoreError(
                f"Daily weather partition {prefix!r} must not contain time_index."
            )

        partition = WeatherPartition(
            site=site,
            month=month_number,
            cadence=cadence,
            year=self._read_array(f"{prefix}/year"),
            day=self._read_array(f"{prefix}/day"),
            time_index=(
                self._read_array(f"{prefix}/time_index")
                if cadence == "native"
                else None
            ),
            tau_coefficients=self._read_array(f"{prefix}/tau_coefficients"),
            tb_coefficients=self._read_array(f"{prefix}/tb_coefficients"),
            pwv_mm=self._read_array(f"{prefix}/pwv_mm"),
            wind_speed_m_s=self._read_array(f"{prefix}/wind_speed_m_s"),
            surface_pressure_mbar=self._read_array(f"{prefix}/surface_pressure_mbar"),
            surface_temperature_k=self._read_array(f"{prefix}/surface_temperature_k"),
        )
        self._validate_partition(partition)
        return partition

    def reconstruct_tau_spectra(self, partition: WeatherPartition) -> np.ndarray:
        """Reconstruct opacity spectra for every record in ``partition``."""

        log_tau = self._reconstruct("tau", partition.tau_coefficients)
        return np.power(10.0, log_tau)

    def reconstruct_tb_spectra(self, partition: WeatherPartition) -> np.ndarray:
        """Reconstruct brightness-temperature spectra for every record."""

        return self._reconstruct("tb", partition.tb_coefficients)

    def _validate_root(self) -> None:
        attributes = self._root.attrs
        if attributes.get("schema_version") != SCHEMA_VERSION:
            raise WeatherStoreError(
                "Unsupported Zarr weather schema version "
                f"{attributes.get('schema_version')!r}; expected {SCHEMA_VERSION!r}."
            )
        if not isinstance(attributes.get("dataset_id"), str) or not attributes["dataset_id"]:
            raise WeatherStoreError("Zarr weather dataset has no valid dataset_id attribute.")

        required_paths = (
            "frequency_ghz",
            "pca/tau/mean",
            "pca/tau/components",
            "pca/tb/mean",
            "pca/tb/components",
            "sites",
        )
        missing = [path for path in required_paths if path not in self._root]
        if missing:
            raise WeatherStoreError(
                "Zarr weather dataset is missing required paths: " + ", ".join(missing)
            )
        self._sites = tuple(sorted(self._root["sites"].group_keys()))
        if not self._sites:
            raise WeatherStoreError("Zarr weather dataset does not contain any sites.")

        frequency = self._read_array("frequency_ghz")
        if frequency.ndim != 1 or not len(frequency) or not np.all(np.isfinite(frequency)):
            raise WeatherStoreError("Zarr weather frequency_ghz must be a nonempty finite array.")
        if np.any(np.diff(frequency) <= 0.0):
            raise WeatherStoreError("Zarr weather frequency_ghz must be strictly increasing.")
        self._frequency_ghz = frequency.astype(float, copy=False)

        samples_per_day = attributes.get("native_samples_per_day")
        if not isinstance(samples_per_day, int) or samples_per_day <= 0:
            raise WeatherStoreError(
                "Zarr weather dataset has no valid native_samples_per_day attribute."
            )
        self._native_samples_per_day = samples_per_day

        for quantity in ("tau", "tb"):
            mean = self._read_array(f"pca/{quantity}/mean")
            components = self._read_array(f"pca/{quantity}/components")
            if mean.shape != frequency.shape:
                raise WeatherStoreError(
                    f"Zarr weather PCA mean for {quantity!r} has incompatible shape."
                )
            if components.ndim != 2 or components.shape[1] != len(frequency):
                raise WeatherStoreError(
                    f"Zarr weather PCA components for {quantity!r} have incompatible shape."
                )
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(components)):
                raise WeatherStoreError(
                    f"Zarr weather PCA basis for {quantity!r} contains non-finite values."
                )
            self._pca_bases[quantity] = (mean, components)

    def _validate_partition(self, partition: WeatherPartition) -> None:
        count = partition.record_count
        if count == 0:
            raise WeatherStoreError("Zarr weather partitions must contain at least one record.")

        one_dimensional = (
            partition.year,
            partition.day,
            partition.pwv_mm,
            partition.wind_speed_m_s,
            partition.surface_pressure_mbar,
            partition.surface_temperature_k,
        )
        if any(values.shape != (count,) for values in one_dimensional):
            raise WeatherStoreError("Zarr weather partition scalar arrays have inconsistent shapes.")
        if partition.time_index is not None and partition.time_index.shape != (count,):
            raise WeatherStoreError("Zarr weather partition time_index has an inconsistent shape.")

        expected_tau_shape = (count, self._pca_bases["tau"][1].shape[0])
        expected_tb_shape = (count, self._pca_bases["tb"][1].shape[0])
        if partition.tau_coefficients.shape != expected_tau_shape:
            raise WeatherStoreError("Zarr weather tau coefficients have an inconsistent shape.")
        if partition.tb_coefficients.shape != expected_tb_shape:
            raise WeatherStoreError("Zarr weather Tb coefficients have an inconsistent shape.")

        finite_arrays = (
            partition.tau_coefficients,
            partition.tb_coefficients,
            partition.pwv_mm,
            partition.wind_speed_m_s,
            partition.surface_pressure_mbar,
            partition.surface_temperature_k,
        )
        if not all(np.all(np.isfinite(values)) for values in finite_arrays):
            raise WeatherStoreError("Zarr weather partition contains non-finite values.")
        if np.any((partition.day < 1) | (partition.day > 31)):
            raise WeatherStoreError("Zarr weather partition contains invalid calendar days.")
        month_lengths = np.fromiter(
            (calendar.monthrange(int(year), partition.month)[1] for year in partition.year),
            dtype=int,
            count=count,
        )
        if np.any(partition.day > month_lengths):
            raise WeatherStoreError("Zarr weather partition contains invalid calendar dates.")
        if partition.time_index is not None and np.any(
            (partition.time_index < 0)
            | (partition.time_index >= self._native_samples_per_day)
        ):
            raise WeatherStoreError("Zarr weather partition contains invalid native time indices.")

    def _reconstruct(self, quantity: Literal["tau", "tb"], coefficients: np.ndarray) -> np.ndarray:
        mean, components = self._pca_bases[quantity]
        if coefficients.ndim != 2 or coefficients.shape[1] != components.shape[0]:
            raise WeatherStoreError(
                f"PCA coefficients are incompatible with the {quantity!r} basis."
            )
        return coefficients.astype(float, copy=False) @ components + mean

    def _read_array(self, path: str) -> np.ndarray:
        array = np.asarray(self._root[path][:])
        array.setflags(write=False)
        return array


def _parse_month(month: str | int) -> int:
    if isinstance(month, str) and month in _MONTH_NAMES:
        return _MONTH_NAMES.index(month) + 1

    try:
        month_number = int(month)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Specified month not recognized; use a three-letter abbreviation "
            "(for example Jan or Apr) or a number from 1 to 12."
        ) from exc

    if 1 <= month_number <= 12:
        return month_number
    raise ValueError(
        "Specified month not recognized; use a three-letter abbreviation "
        "(for example Jan or Apr) or a number from 1 to 12."
    )
