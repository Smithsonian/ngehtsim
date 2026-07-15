"""Read versioned local Zarr weather datasets.

This module intentionally has no dependency on the legacy binary weather API.
It is an opt-in reader for externally distributed weather releases; callers pass
an explicit local filesystem path to :class:`ZarrWeatherStore`.
"""

from __future__ import annotations

import calendar
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np


SCHEMA_VERSION = "0.1.0"
Cadence = Literal["daily", "native"]
NativeWeatherForm = Literal["exact", "mean", "median", "good", "bad"]

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
_NATIVE_SUMMARY_CACHE_SIZE = 64
_NATIVE_SUMMARY_FORMS = ("mean", "median", "good", "bad")


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


@dataclass(frozen=True)
class NativeWeatherSamples:
    """Linearly sampled weather values at one or more UTC-hour offsets.

    All fields have a leading dimension matching ``utc_hours``. The spectral
    fields have a second dimension matching :attr:`ZarrWeatherStore.frequency_ghz`.
    """

    utc_hours: np.ndarray
    opacity: np.ndarray
    brightness_temperature: np.ndarray
    pwv_mm: np.ndarray
    wind_speed_m_s: np.ndarray
    surface_pressure_mbar: np.ndarray
    surface_temperature_k: np.ndarray


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
        self._native_summary_cache: OrderedDict[
            tuple[str, int, NativeWeatherForm], dict[str, np.ndarray]
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

    def sample_native(
        self,
        site: str,
        *,
        year: int,
        month: str | int,
        day: int,
        utc_hours: float | np.ndarray,
        form: NativeWeatherForm = "exact",
    ) -> NativeWeatherSamples:
        """Linearly sample three-hourly weather at UTC-hour offsets.

        Args:
            site: Weather site identifier.
            year: Year of the base weather date.
            month: Three-letter or numeric base month.
            day: Day of the base month.
            utc_hours: One or more UTC-hour offsets from the base date. Values
                may cross calendar-day and calendar-month boundaries.
            form: ``"exact"`` samples the selected historical date. Summary
                forms build a month-specific three-hourly climatology before
                interpolation.

        Returns:
            A :class:`NativeWeatherSamples` instance. Results always retain a
            leading sample dimension, including when ``utc_hours`` is scalar.

        Raises:
            WeatherStoreError: If either interpolation endpoint is unavailable.
            ValueError: If an unsupported weather form or invalid date is used.
        """

        if site not in self.sites:
            raise KeyError(f"Weather dataset does not contain site {site!r}.")
        if form != "exact" and form not in _NATIVE_SUMMARY_FORMS:
            allowed = ", ".join(("exact", *_NATIVE_SUMMARY_FORMS))
            raise ValueError(f"Unsupported native weather form {form!r}; use {allowed}.")

        month_number = _parse_month(month)
        try:
            base_date = date(int(year), month_number, int(day))
        except (TypeError, ValueError) as exc:
            raise ValueError("Native weather sampling requires a valid calendar date.") from exc

        hours = np.asarray(utc_hours, dtype=float)
        if hours.ndim == 0:
            hours = hours.reshape(1)
        if hours.ndim != 1 or not np.all(np.isfinite(hours)):
            raise ValueError("utc_hours must be a finite scalar or one-dimensional array.")

        dates, day_hours, absolute_hours = _resolve_utc_hours(base_date, hours)
        if form == "exact":
            sample_hours, values = self._native_exact_values(site, dates, day_hours)
        else:
            sample_hours, values = self._native_summary_values(site, dates, form)

        interpolated = {
            name: _linear_interpolate(
                sample_hours, value, absolute_hours, self._native_time_step_hours
            )
            for name, value in values.items()
        }
        return NativeWeatherSamples(
            utc_hours=_readonly_copy(hours),
            opacity=_readonly_copy(interpolated["opacity"]),
            brightness_temperature=_readonly_copy(interpolated["brightness_temperature"]),
            pwv_mm=_readonly_copy(interpolated["pwv_mm"]),
            wind_speed_m_s=_readonly_copy(interpolated["wind_speed_m_s"]),
            surface_pressure_mbar=_readonly_copy(interpolated["surface_pressure_mbar"]),
            surface_temperature_k=_readonly_copy(interpolated["surface_temperature_k"]),
        )

    def _native_exact_values(
        self, site: str, dates: tuple[date, ...], day_hours: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        endpoint_dates = set(dates)
        endpoint_dates.update(
            item + timedelta(days=1)
            for item, hour in zip(dates, day_hours)
            if hour > 21.0
        )
        partitions = [
            self.read_partition(site, month, cadence="native")
            for month in {item.month for item in endpoint_dates}
        ]
        return self._native_partition_values(partitions, endpoint_dates)

    def _native_summary_values(
        self,
        site: str,
        dates: tuple[date, ...],
        form: NativeWeatherForm,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        sample_hours = []
        samples: dict[str, list[np.ndarray]] = {
            "opacity": [],
            "brightness_temperature": [],
            "pwv_mm": [],
            "wind_speed_m_s": [],
            "surface_pressure_mbar": [],
            "surface_temperature_k": [],
        }
        unique_dates = sorted(set(dates))
        for sample_date in unique_dates:
            current = self._native_month_summary(site, sample_date.month, form)
            next_day = sample_date + timedelta(days=1)
            next_summary = self._native_month_summary(site, next_day.month, form)
            offsets = np.arange(self._native_samples_per_day + 1, dtype=float)
            offsets *= self._native_time_step_hours
            sample_hours.append((_absolute_hour(sample_date) + offsets))
            for name in samples:
                samples[name].append(
                    np.concatenate((current[name], next_summary[name][:1]), axis=0)
                )

        hours = np.concatenate(sample_hours)
        values = {name: np.concatenate(items) for name, items in samples.items()}
        order = np.argsort(hours)
        hours = hours[order]
        values = {name: value[order] for name, value in values.items()}
        unique_hours, indices = np.unique(hours, return_index=True)
        return unique_hours, {name: value[indices] for name, value in values.items()}

    def _native_month_summary(
        self, site: str, month: int, form: NativeWeatherForm
    ) -> dict[str, np.ndarray]:
        key = (site, month, form)
        try:
            values = self._native_summary_cache.pop(key)
        except KeyError:
            partition = self.read_partition(site, month, cadence="native")
            values = self._summarize_native_partition(partition, form)
            if len(self._native_summary_cache) >= _NATIVE_SUMMARY_CACHE_SIZE:
                self._native_summary_cache.popitem(last=False)
        self._native_summary_cache[key] = values
        return values

    def _summarize_native_partition(
        self, partition: WeatherPartition, form: NativeWeatherForm
    ) -> dict[str, np.ndarray]:
        if partition.time_index is None:
            raise WeatherStoreError("Native weather partitions require time_index data.")

        values = {
            "opacity": self.reconstruct_tau_spectra(partition),
            "brightness_temperature": self.reconstruct_tb_spectra(partition),
            "pwv_mm": partition.pwv_mm,
            "wind_speed_m_s": partition.wind_speed_m_s,
            "surface_pressure_mbar": partition.surface_pressure_mbar,
            "surface_temperature_k": partition.surface_temperature_k,
        }
        reducer = _native_reducer(form)
        summary = {name: [] for name in values}
        for time_index in range(self._native_samples_per_day):
            mask = partition.time_index == time_index
            if not np.any(mask):
                raise WeatherStoreError(
                    f"Native weather partition for {partition.site!r}, month "
                    f"{partition.month:02d} is missing time index {time_index}."
                )
            for name, value in values.items():
                summary[name].append(reducer(value[mask], axis=0))
        return {name: np.asarray(items) for name, items in summary.items()}

    def _native_partition_values(
        self, partitions: list[WeatherPartition], endpoint_dates: set[date]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        coordinates = []
        tau_coefficients = []
        tb_coefficients = []
        scalar_values = {name: [] for name in _SCALAR_ARRAYS}
        for partition in partitions:
            if partition.time_index is None:
                raise WeatherStoreError("Native weather partitions require time_index data.")
            mask = np.zeros(partition.record_count, dtype=bool)
            for endpoint_date in endpoint_dates:
                if endpoint_date.month == partition.month:
                    mask |= (
                        (partition.year == endpoint_date.year)
                        & (partition.day == endpoint_date.day)
                    )
            if not np.any(mask):
                continue
            coordinates.append(
                np.fromiter(
                    (
                        _absolute_hour(date(int(year), partition.month, int(day)))
                        + (self._native_time_step_hours * int(time_index))
                        for year, day, time_index in zip(
                            partition.year[mask],
                            partition.day[mask],
                            partition.time_index[mask],
                        )
                    ),
                    dtype=float,
                    count=int(mask.sum()),
                )
            )
            tau_coefficients.append(partition.tau_coefficients[mask])
            tb_coefficients.append(partition.tb_coefficients[mask])
            for name in scalar_values:
                scalar_values[name].append(getattr(partition, name)[mask])

        if not coordinates:
            raise WeatherStoreError("Native weather interpolation endpoints are unavailable.")

        sample_hours = np.concatenate(coordinates)
        values = {
            "opacity": np.power(10.0, self._reconstruct("tau", np.concatenate(tau_coefficients))),
            "brightness_temperature": self._reconstruct("tb", np.concatenate(tb_coefficients)),
            "pwv_mm": np.concatenate(scalar_values["pwv_mm"]),
            "wind_speed_m_s": np.concatenate(scalar_values["wind_speed_m_s"]),
            "surface_pressure_mbar": np.concatenate(scalar_values["surface_pressure_mbar"]),
            "surface_temperature_k": np.concatenate(scalar_values["surface_temperature_k"]),
        }
        order = np.argsort(sample_hours)
        sample_hours = sample_hours[order]
        return sample_hours, {name: value[order] for name, value in values.items()}

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
        time_step_hours = attributes.get("native_time_step_hours")
        if not isinstance(time_step_hours, int) or time_step_hours <= 0:
            raise WeatherStoreError(
                "Zarr weather dataset has no valid native_time_step_hours attribute."
            )
        if samples_per_day * time_step_hours != 24:
            raise WeatherStoreError(
                "Zarr weather native sampling does not cover one 24-hour UTC day."
            )
        self._native_time_step_hours = time_step_hours

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


def _resolve_utc_hours(
    base_date: date, utc_hours: np.ndarray
) -> tuple[tuple[date, ...], np.ndarray, np.ndarray]:
    day_offsets = np.floor(utc_hours / 24.0).astype(int)
    day_hours = utc_hours - (24.0 * day_offsets)
    dates = tuple(base_date + timedelta(days=int(offset)) for offset in day_offsets)
    absolute_hours = np.asarray(
        [_absolute_hour(item) + hour for item, hour in zip(dates, day_hours)], dtype=float
    )
    return dates, day_hours, absolute_hours


def _absolute_hour(value: date) -> float:
    return float(value.toordinal() * 24)


def _linear_interpolate(
    sample_hours: np.ndarray,
    values: np.ndarray,
    target_hours: np.ndarray,
    time_step_hours: int,
) -> np.ndarray:
    if np.any(target_hours < sample_hours[0]) or np.any(target_hours > sample_hours[-1]):
        raise WeatherStoreError("Native weather interpolation endpoints are unavailable.")

    right = np.searchsorted(sample_hours, target_hours, side="left")
    exact = np.zeros(len(target_hours), dtype=bool)
    in_range = right < len(sample_hours)
    exact[in_range] = sample_hours[right[in_range]] == target_hours[in_range]
    result = np.empty((len(target_hours), *values.shape[1:]), dtype=float)
    result[exact] = values[right[exact]]

    interpolate = ~exact
    if not np.any(interpolate):
        return result
    if len(sample_hours) < 2:
        raise WeatherStoreError("Native weather interpolation requires at least two samples.")

    interpolation_right = right[interpolate]
    interpolation_left = interpolation_right - 1
    if np.any(interpolation_left < 0) or np.any(interpolation_right >= len(sample_hours)):
        raise WeatherStoreError("Native weather interpolation endpoints are unavailable.")

    interval = sample_hours[interpolation_right] - sample_hours[interpolation_left]
    if np.any(interval <= 0.0) or np.any(interval > time_step_hours):
        raise WeatherStoreError("Native weather interpolation would cross missing records.")

    weight = (target_hours[interpolate] - sample_hours[interpolation_left]) / interval
    if values.ndim > 1:
        weight = weight.reshape((len(weight),) + (1,) * (values.ndim - 1))
    result[interpolate] = values[interpolation_left] + (
        weight * (values[interpolation_right] - values[interpolation_left])
    )
    return result


def _native_reducer(form: NativeWeatherForm):
    if form == "mean":
        return np.nanmean
    if form == "median":
        return np.nanmedian
    if form == "good":
        return lambda values, axis: np.nanpercentile(values, 15.87, axis=axis)
    if form == "bad":
        return lambda values, axis: np.nanpercentile(values, 84.13, axis=axis)
    raise ValueError(f"Unsupported native weather summary form {form!r}.")


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result
