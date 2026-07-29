"""Declarative station-corruption settings for native simulations.

The native simulator separates station-common voltage gains from differential
two-feed gain ratios.  For a two-feed station with feed gains ``G_A`` and
``G_B``, it realizes a common gain ``G`` and ratio ``R`` as
``G_A = G sqrt(R)`` and ``G_B = G / sqrt(R)``.  Gain phases are generated as
real-valued process variables before being wrapped into complex voltages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RealizationCadence:
    """Grouping rule for a stochastic station-corruption realization.

    Parameters
    ----------
    kind : {"integration", "scan", "track", "interval"}, optional
        ``"integration"`` realizes one value for every distinct integration
        time, ``"scan"`` shares one value throughout every stored scan,
        ``"track"`` shares one value throughout the dataset, and
        ``"interval"`` uses fixed-width time bins.
    interval_seconds : float, optional
        Positive bin width required when ``kind="interval"``.
    origin_mjd : float, optional
        UTC MJD defining the zero edge of ``"interval"`` bins. Omitting it
        anchors bins at the earliest dataset time.

    Notes
    -----
    Scan cadence deliberately requires explicit scan metadata. ngehtsim never
    infers scans from timestamp gaps because that would silently change the
    physical correlation assumed for gain errors.
    """

    kind: str = "track"
    interval_seconds: float | None = None
    origin_mjd: float | None = None

    def __post_init__(self):
        if self.kind not in ("integration", "scan", "track", "interval"):
            raise ValueError(
                "kind must be one of 'integration', 'scan', 'track', or 'interval'."
            )
        if self.kind == "interval":
            if self.interval_seconds is None or not np.isfinite(self.interval_seconds):
                raise ValueError("interval cadence requires a finite interval_seconds value.")
            if self.interval_seconds <= 0.0:
                raise ValueError("interval_seconds must be positive.")
        elif self.interval_seconds is not None:
            raise ValueError("interval_seconds is only valid for interval cadence.")
        if self.origin_mjd is not None and not np.isfinite(self.origin_mjd):
            raise ValueError("origin_mjd must be finite when supplied.")

    @classmethod
    def integration(cls):
        """Return a cadence with one draw per distinct integration time."""

        return cls("integration")

    @classmethod
    def scan(cls):
        """Return a cadence with one draw per stored observation scan."""

        return cls("scan")

    @classmethod
    def track(cls):
        """Return a cadence with one draw over the entire observation."""

        return cls("track")

    @classmethod
    def interval(cls, seconds, origin_mjd=None):
        """Return a cadence using fixed-width bins of ``seconds``."""

        return cls("interval", interval_seconds=float(seconds), origin_mjd=origin_mjd)


def realization_group_ids(dataset, cadence):
    """Return an integer realization group for every native visibility row.

    Parameters
    ----------
    dataset : VisibilityDataset
        Dataset whose UTC timestamps and optional scan metadata define the
        realization groups.
    cadence : RealizationCadence
        Requested grouping rule.

    Returns
    -------
    numpy.ndarray, shape (row,)
        Dense non-negative group IDs. Equal IDs share one stochastic draw for
        a given station.

    Raises
    ------
    ValueError
        If scan cadence is requested without complete, unambiguous scan
        metadata.
    """

    if not isinstance(cadence, RealizationCadence):
        raise TypeError("cadence must be a RealizationCadence instance.")
    time_mjd = np.asarray(dataset.time_mjd, dtype=float)
    if cadence.kind == "track":
        return np.zeros(dataset.row_count, dtype=np.intp)
    if cadence.kind == "integration":
        _, inverse = np.unique(time_mjd, return_inverse=True)
        return inverse.astype(np.intp, copy=False)
    if cadence.kind == "interval":
        if not dataset.row_count:
            return np.zeros(0, dtype=np.intp)
        origin = np.min(time_mjd) if cadence.origin_mjd is None else cadence.origin_mjd
        values = np.floor(((time_mjd - origin) * 86400.0) / cadence.interval_seconds)
        _, inverse = np.unique(values.astype(np.int64), return_inverse=True)
        return inverse.astype(np.intp, copy=False)

    if dataset.scan_start_mjd is None or dataset.scan_stop_mjd is None:
        raise ValueError(
            "scan cadence requires dataset scan metadata; provide scan intervals or "
            "choose integration, track, or interval cadence."
        )
    starts = np.asarray(dataset.scan_start_mjd, dtype=float)
    stops = np.asarray(dataset.scan_stop_mjd, dtype=float)
    memberships = (time_mjd[:, np.newaxis] >= starts[np.newaxis, :]) & (
        time_mjd[:, np.newaxis] <= stops[np.newaxis, :]
    )
    count = np.sum(memberships, axis=1)
    if np.any(count != 1):
        raise ValueError(
            "scan cadence requires every visibility row to belong to exactly one scan."
        )
    return np.argmax(memberships, axis=1).astype(np.intp)


@dataclass(frozen=True)
class GainModel:
    """Stochastic station-common complex voltage gain ``G``.

    Amplitude and phase are sampled separately so they can have different
    realization cadences. The amplitude is a base-10 logarithmic multiplier;
    a zero mean and zero standard deviation therefore leaves it unchanged.

    Parameters
    ----------
    amplitude_sigma_dex, amplitude_mean_dex : float, optional
        Standard deviation and mean of ``log10(abs(G))``.
    phase_distribution : {"none", "uniform", "normal"}, optional
        Distribution about ``phase_mean_rad``. ``"uniform"`` is uniform on
        ``[-pi, pi)``; ``"normal"`` uses ``phase_sigma_rad``.
    phase_sigma_rad : float, optional
        Standard deviation for ``phase_distribution="normal"``.
    amplitude_cadence, phase_cadence : RealizationCadence, optional
        Draw grouping for the two independent processes. Both default to one
        realization per stored scan.
    """

    amplitude_sigma_dex: float = 0.0
    phase_distribution: str = "none"
    amplitude_mean_dex: float = 0.0
    phase_mean_rad: float = 0.0
    phase_sigma_rad: float = 0.0
    amplitude_cadence: RealizationCadence = field(default_factory=RealizationCadence.scan)
    phase_cadence: RealizationCadence = field(default_factory=RealizationCadence.scan)

    def __post_init__(self):
        _validate_gain_fields(self)
        _validate_cadences(self)

    def sample_amplitude(self, rng):
        """Draw one positive voltage-amplitude multiplier."""

        return 10.0 ** (
            self.amplitude_mean_dex
            + (self.amplitude_sigma_dex * rng.normal(0.0, 1.0))
        )

    def sample_phase(self, rng):
        """Draw one unwrapped station-gain phase in radians."""

        return _sample_phase(self, rng)

    def sample(self, rng):
        """Draw one complex gain using independent amplitude and phase draws."""

        return self.sample_amplitude(rng) * np.exp(1.0j * self.sample_phase(rng))


@dataclass(frozen=True)
class GainRatioModel:
    """Stochastic two-feed complex gain ratio ``R = G_A / G_B``.

    Feed ordering can be declared explicitly or inherited from each station's
    local receptor declaration. The native RIME applies the symmetric factors
    ``sqrt(R)`` and ``1/sqrt(R)`` to the ordered feeds, respectively; no
    calibration reference feed or reference station is introduced.

    Parameters
    ----------
    feed_a, feed_b : str or None, optional
        Ordered feed IDs defining the numerator and denominator of
        ``R = G_A / G_B``. Omit both to use the declared receptor order at
        every two-feed station where this model applies. Supplying one feed
        requires supplying the other.
    amplitude_sigma_dex : float, optional
        Standard deviation of ``log10(abs(R))``.
    amplitude_mean_dex : float, optional
        Mean of ``log10(abs(R))``.
    phase_distribution : {"none", "uniform", "normal"}, optional
        Distribution about ``phase_mean_rad`` for the unwrapped ratio phase.
    phase_mean_rad : float, optional
        Mean unwrapped phase of ``R`` in radians.
    phase_sigma_rad : float, optional
        Standard deviation when ``phase_distribution="normal"``.
    amplitude_cadence : RealizationCadence, optional
        Grouping for logarithmic ratio-amplitude draws. The default is one
        realization per track.
    phase_cadence : RealizationCadence, optional
        Grouping for unwrapped ratio-phase draws. The default is one
        realization per track.
    """

    feed_a: str | None = None
    feed_b: str | None = None
    amplitude_sigma_dex: float = 0.0
    phase_distribution: str = "none"
    amplitude_mean_dex: float = 0.0
    phase_mean_rad: float = 0.0
    phase_sigma_rad: float = 0.0
    amplitude_cadence: RealizationCadence = field(default_factory=RealizationCadence.track)
    phase_cadence: RealizationCadence = field(default_factory=RealizationCadence.track)

    def __post_init__(self):
        if (self.feed_a is None) != (self.feed_b is None):
            raise ValueError("feed_a and feed_b must either both be supplied or both be None.")
        if self.feed_a is not None:
            if (
                not isinstance(self.feed_a, str)
                or not isinstance(self.feed_b, str)
                or not self.feed_a
                or not self.feed_b
                or self.feed_a == self.feed_b
            ):
                raise ValueError("feed_a and feed_b must be distinct non-empty feed IDs.")
        _validate_gain_fields(self)
        _validate_cadences(self)

    def sample_log_amplitude(self, rng):
        """Draw one unwrapped ``log10(abs(R))`` realization."""

        return self.amplitude_mean_dex + (self.amplitude_sigma_dex * rng.normal(0.0, 1.0))

    def sample_phase(self, rng):
        """Draw one unwrapped ratio phase in radians."""

        return _sample_phase(self, rng)


@dataclass(frozen=True)
class LeakageModel:
    r"""Stochastic two-feed leakage distribution in the local feed frame.

    For the ordered local feeds A and B, the native RIME applies

    .. math::

       D_{\rm feed} =
       \begin{pmatrix}1 & D_A \\ D_B & 1\end{pmatrix}.

    ``D_A`` is the leakage from feed B into feed A and ``D_B`` is the reverse
    coupling.  Consequently, R/L stations use ordinary ``D_R`` and ``D_L``
    terms, while X/Y stations use ordinary ``D_X`` and ``D_Y`` terms.  The
    common circular sky basis is used only for source coherency and feed
    rotation; it does not define the leakage parameters.

    Parameters
    ----------
    feed_a, feed_b : str or None, optional
        Ordered local feed IDs defining the rows and columns of the leakage
        matrix. Omit both to use a station's declared two-feed order. Supplying
        one feed requires supplying the other.
    leakage_a_mean, leakage_b_mean : complex, optional
        Deterministic complex means of ``D_A`` and ``D_B``. Together with
        ``component_sigma=0`` these define an exact local-feed leakage model.
    component_sigma : float, optional
        Standard deviation assigned independently to the real and imaginary
        parts of each off-diagonal local-feed leakage term.
    cadence : RealizationCadence, optional
        Realization grouping for the complete complex leakage matrix. The
        default is one stable D-term realization per track.
    """

    feed_a: str | None = None
    feed_b: str | None = None
    leakage_a_mean: complex = 0.0j
    leakage_b_mean: complex = 0.0j
    component_sigma: float = 0.0
    cadence: RealizationCadence = field(default_factory=RealizationCadence.track)

    def __post_init__(self):
        if (self.feed_a is None) != (self.feed_b is None):
            raise ValueError("feed_a and feed_b must either both be supplied or both be None.")
        if self.feed_a is not None:
            if (
                not isinstance(self.feed_a, str)
                or not isinstance(self.feed_b, str)
                or not self.feed_a
                or not self.feed_b
                or self.feed_a == self.feed_b
            ):
                raise ValueError("feed_a and feed_b must be distinct non-empty feed IDs.")
        for name in ("leakage_a_mean", "leakage_b_mean"):
            try:
                value = complex(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise TypeError("{0} must be a complex scalar.".format(name)) from exc
            if not np.isfinite(value.real) or not np.isfinite(value.imag):
                raise ValueError("{0} must have finite real and imaginary parts.".format(name))
            object.__setattr__(self, name, value)
        if not np.isfinite(self.component_sigma) or self.component_sigma < 0.0:
            raise ValueError("component_sigma must be finite and non-negative.")
        if not isinstance(self.cadence, RealizationCadence):
            raise TypeError("cadence must be a RealizationCadence instance.")

    def sample(self, rng):
        """Draw the two local-feed complex leakage terms."""

        draw_a = self.component_sigma * (rng.normal(0.0, 1.0) + 1.0j * rng.normal(0.0, 1.0))
        draw_b = self.component_sigma * (rng.normal(0.0, 1.0) + 1.0j * rng.normal(0.0, 1.0))
        return self.leakage_a_mean + draw_a, self.leakage_b_mean + draw_b


@dataclass(frozen=True)
class StationCorruptionModel:
    """Complete native station-effect configuration for one simulation.

    The default realization adds independent thermal noise, opacity
    calibration, feed rotation, weather/solar flagging, and a station-common
    gain with independent amplitude and phase draws per scan. Gain ratios are
    disabled unless explicitly configured for a two-feed station.

    Parameters
    ----------
    thermal_noise : bool, optional
        Add independent complex thermal noise per integration, channel, and
        correlation product.
    opacity_calibrated, feed_rotation : bool, optional
        Select the opacity convention and physical feed-rotation calculation.
    station_gain : GainModel or None, optional
        Common complex station voltage-gain process ``G``. ``None`` disables
        station-common gain corruption.
    station_gain_overrides : mapping, optional
        Mapping ``{station: GainModel or None}`` that replaces the default
        station-common gain model at named stations. ``None`` disables the
        default common gain at that station.
    leakage : LeakageModel or None, optional
        Default local-feed leakage realization for every two-feed station.
        ``None`` disables leakage by default. A default model is skipped for
        single-feed stations; stations with more than two feeds require a
        future general leakage parameterization.
    leakage_overrides : mapping, optional
        Mapping ``{station: LeakageModel or None}`` that replaces the default
        leakage model at named stations. ``None`` disables default leakage at
        that station.
    gain_ratio : GainRatioModel or None, optional
        Default two-feed gain-ratio process. It applies to every two-feed
        station and is skipped for single-feed stations. A model with omitted
        ``feed_a`` and ``feed_b`` uses each station's declared receptor order.
        ``None`` disables gain ratios by default.
    gain_ratio_overrides : mapping, optional
        Mapping ``{station: GainRatioModel or None}`` that replaces the
        default gain-ratio model at named stations. ``None`` disables the
        default ratio at that station.
    flag_wind, flag_daylight, flag_sun : bool, optional
        Enable weather, daytime, and solar-avoidance availability masks.
    """

    thermal_noise: bool = True
    opacity_calibrated: bool = True
    feed_rotation: bool = True
    station_gain: GainModel | None = field(default_factory=GainModel)
    station_gain_overrides: Mapping[str, GainModel | None] = field(default_factory=dict)
    leakage: LeakageModel | None = None
    leakage_overrides: Mapping[str, LeakageModel | None] = field(default_factory=dict)
    gain_ratio: GainRatioModel | None = None
    gain_ratio_overrides: Mapping[str, GainRatioModel | None] = field(default_factory=dict)
    flag_wind: bool = True
    flag_daylight: bool = False
    flag_sun: bool = True

    def __post_init__(self):
        for name in (
            "thermal_noise",
            "opacity_calibrated",
            "feed_rotation",
            "flag_wind",
            "flag_daylight",
            "flag_sun",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError("{0} must be a bool.".format(name))
        if self.station_gain is not None and not isinstance(self.station_gain, GainModel):
            raise TypeError("station_gain must be a GainModel or None.")
        if self.leakage is not None and not isinstance(self.leakage, LeakageModel):
            raise TypeError("leakage must be a LeakageModel or None.")
        if self.gain_ratio is not None and not isinstance(self.gain_ratio, GainRatioModel):
            raise TypeError("gain_ratio must be a GainRatioModel or None.")
        for name, model_type in (
            ("station_gain_overrides", GainModel),
            ("leakage_overrides", LeakageModel),
            ("gain_ratio_overrides", GainRatioModel),
        ):
            normalized = _normalize_overrides(getattr(self, name), model_type, name)
            object.__setattr__(self, name, MappingProxyType(normalized))

    def validate_receptors(self, station_names, receptors):
        """Validate gain-ratio and local-leakage models against receptor inventory.

        Gain ratios are valid only for a station containing exactly two feeds.
        A default ratio is skipped for a single-feed station, which uses only
        its station-common gain. A ratio explicitly configured as an override
        for a single-feed station is rejected. Datasets with more than two
        feeds remain usable without a ratio model but reject a requested ratio
        until a general multi-feed parameterization is introduced. Local-feed
        leakage follows the same two-feed rule; a default leakage model is
        skipped for a single-feed station.
        """

        names = tuple(str(name) for name in station_names)
        unknown_stations = (
            set(self.station_gain_overrides)
            | set(self.leakage_overrides)
            | set(self.gain_ratio_overrides)
        ) - set(names)
        if unknown_stations:
            raise ValueError(
                "Station-corruption overrides reference unknown stations: {0}.".format(
                    ", ".join(sorted(unknown_stations))
                )
            )
        for station in names:
            model = self.gain_ratio_model(station)
            if model is None:
                continue
            station_index = names.index(station)
            feed_ids = tuple(
                receptors.feed_id[index]
                for index in np.flatnonzero(receptors.station_index == station_index)
            )
            if len(feed_ids) == 1:
                if self.gain_ratio_overrides.get(station) is not None:
                    raise ValueError(
                        "Station {0} has one feed and cannot define a gain ratio.".format(
                            station
                        )
                    )
                continue
            if len(feed_ids) != 2:
                raise NotImplementedError(
                    "Gain ratios currently support exactly two feeds per station; "
                    "{0} has {1}.".format(station, len(feed_ids))
                )
            feed_a, feed_b = self.gain_ratio_feed_pair(station, feed_ids)
            if set(feed_ids) != {feed_a, feed_b}:
                raise ValueError(
                    "Gain-ratio feeds for {0} must match its two configured feeds.".format(station)
                )
        for station in names:
            model = self.leakage_model(station)
            if model is None:
                continue
            station_index = names.index(station)
            feed_ids = tuple(
                receptors.feed_id[index]
                for index in np.flatnonzero(receptors.station_index == station_index)
            )
            if len(feed_ids) == 1:
                if self.leakage_overrides.get(station) is not None:
                    raise ValueError(
                        "Station {0} has one feed and cannot define a two-feed leakage matrix.".format(
                            station
                        )
                    )
                continue
            if len(feed_ids) != 2:
                raise NotImplementedError(
                    "Local-feed leakage currently supports exactly two feeds per station; "
                    "{0} has {1}.".format(station, len(feed_ids))
                )
            feed_a, feed_b = self.leakage_feed_pair(station, feed_ids)
            if set(feed_ids) != {feed_a, feed_b}:
                raise ValueError(
                    "Leakage feeds for {0} must match its two configured feeds.".format(station)
                )

    def station_gain_model(self, station):
        """Return the effective station-common gain model for ``station``."""

        return self.station_gain_overrides.get(str(station), self.station_gain)

    def leakage_model(self, station):
        """Return the effective station-frame leakage model for ``station``."""

        return self.leakage_overrides.get(str(station), self.leakage)

    def gain_ratio_model(self, station):
        """Return the effective optional two-feed gain-ratio model for ``station``."""

        return self.gain_ratio_overrides.get(str(station), self.gain_ratio)

    def gain_ratio_feed_pair(self, station, feed_ids):
        """Return the effective ordered gain-ratio feeds for ``station``.

        A generic model with omitted feed names follows the station-local
        receptor declaration order. Explicit model feed IDs are returned
        unchanged. Call :meth:`validate_receptors` before using this method.
        """

        model = self.gain_ratio_model(station)
        if model is None:
            return None
        if model.feed_a is None:
            return tuple(feed_ids)
        return model.feed_a, model.feed_b

    def leakage_feed_pair(self, station, feed_ids):
        """Return the effective ordered local-feed pair for leakage at ``station``."""

        model = self.leakage_model(station)
        if model is None:
            return None
        if model.feed_a is None:
            return tuple(feed_ids)
        return model.feed_a, model.feed_b

    @property
    def has_gain_corruption(self):
        """Whether any common-gain or gain-ratio process is configured."""

        return (
            self.station_gain is not None
            or any(model is not None for model in self.station_gain_overrides.values())
            or self.gain_ratio is not None
            or any(model is not None for model in self.gain_ratio_overrides.values())
        )

    @property
    def has_leakage_corruption(self):
        """Whether any station-frame leakage process is configured."""

        return self.leakage is not None or any(
            model is not None for model in self.leakage_overrides.values()
        )


def _normalize_overrides(mapping, model_type, name):
    """Validate and copy one immutable station-override mapping."""

    if not isinstance(mapping, Mapping):
        raise TypeError("{0} must be a mapping.".format(name))
    normalized = {}
    for station, model in mapping.items():
        if model is not None and not isinstance(model, model_type):
            raise TypeError(
                "Each {0} value must be a {1} or None.".format(
                    name.replace("_overrides", " override"),
                    model_type.__name__,
                )
            )
        normalized[str(station)] = model
    return normalized


def _validate_gain_fields(model):
    for name in ("amplitude_mean_dex", "amplitude_sigma_dex", "phase_mean_rad", "phase_sigma_rad"):
        value = getattr(model, name)
        if not np.isfinite(value):
            raise ValueError("{0} must be finite.".format(name))
    if model.amplitude_sigma_dex < 0.0:
        raise ValueError("amplitude_sigma_dex must be non-negative.")
    if model.phase_sigma_rad < 0.0:
        raise ValueError("phase_sigma_rad must be non-negative.")
    if model.phase_distribution not in ("none", "uniform", "normal"):
        raise ValueError("phase_distribution must be 'none', 'uniform', or 'normal'.")
    if model.phase_distribution != "normal" and model.phase_sigma_rad != 0.0:
        raise ValueError("phase_sigma_rad is only valid for normal phase distribution.")


def _validate_cadences(model):
    for name in ("amplitude_cadence", "phase_cadence"):
        if not isinstance(getattr(model, name), RealizationCadence):
            raise TypeError("{0} must be a RealizationCadence instance.".format(name))


def _sample_phase(model, rng):
    if model.phase_distribution == "none":
        return model.phase_mean_rad
    if model.phase_distribution == "uniform":
        return model.phase_mean_rad + rng.uniform(-np.pi, np.pi)
    return model.phase_mean_rad + (model.phase_sigma_rad * rng.normal(0.0, 1.0))
