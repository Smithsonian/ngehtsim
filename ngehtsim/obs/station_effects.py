"""Declarative station-corruption settings for native simulations.

The native simulator separates effects that act on the common two-component
sky field from gains attached to individual recorded voltage paths.  This is
necessary for mixed-receptor arrays: an X/Y, R/L, or custom feed inventory
cannot be described reliably by historical hand-specific keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class GainModel:
    """Random complex-gain distribution for a station or receptor path.

    Parameters
    ----------
    amplitude_sigma_dex : float, optional
        Standard deviation of the logarithmic gain amplitude in dex. A value
        of zero leaves amplitudes unchanged.
    phase_distribution : {"none", "uniform"}, optional
        ``"uniform"`` draws an independent phase uniformly on ``[-pi, pi)``
        at every station/time sample. ``"none"`` leaves phases unchanged.

    Notes
    -----
    A future time-correlated phase process can extend this compact model
    without changing the station/receptor distinction in the native RIME.
    """

    amplitude_sigma_dex: float = 0.0
    phase_distribution: str = "none"

    def __post_init__(self):
        if (
            not np.isfinite(self.amplitude_sigma_dex)
            or self.amplitude_sigma_dex < 0.0
        ):
            raise ValueError("amplitude_sigma_dex must be finite and non-negative.")
        if self.phase_distribution not in ("none", "uniform"):
            raise ValueError("phase_distribution must be either 'none' or 'uniform'.")

    def sample(self, rng):
        """Draw one complex gain from this model.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random generator used to produce the realization.

        Returns
        -------
        complex
            Drawn gain multiplier.
        """

        amplitude = 10.0 ** (self.amplitude_sigma_dex * rng.normal(0.0, 1.0))
        if self.phase_distribution == "uniform":
            phase = rng.uniform(-np.pi, np.pi)
        else:
            phase = 0.0
        return amplitude * np.exp(1.0j * phase)


@dataclass(frozen=True)
class LeakageModel:
    """Station-frame circular leakage distribution.

    Parameters
    ----------
    component_sigma : float, optional
        Standard deviation assigned independently to the real and imaginary
        parts of each off-diagonal circular-basis leakage term.
    """

    component_sigma: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.component_sigma) or self.component_sigma < 0.0:
            raise ValueError("component_sigma must be finite and non-negative.")


@dataclass(frozen=True)
class StationCorruptionModel:
    """Complete native station-effect configuration for one simulation.

    Parameters
    ----------
    thermal_noise : bool, optional
        Add independent complex thermal noise using each product's propagated
        ``sigma_jy`` uncertainty.
    opacity_calibrated : bool, optional
        When true, retain the established opacity-calibrated visibility and
        uncertainty convention. When false, attenuate the signal instead.
    feed_rotation : bool, optional
        Apply mount and feed-angle rotation in the common circular sky frame.
    common_gain : GainModel or None, optional
        Station-wide gain drawn once per station/time sample and applied to
        every local receptor path. ``None`` disables common gains.
    leakage : LeakageModel or None, optional
        Station-frame circular leakage realization. ``None`` disables leakage.
    path_gain_overrides : mapping, optional
        Nested mapping ``{station: {feed_id: GainModel}}``. Each configured
        gain is applied after the station-frame Jones response to the named
        local voltage path. It supplements ``common_gain``.
    flag_wind, flag_daylight, flag_sun : bool, optional
        Enable weather, daytime, and solar-avoidance availability masks.

    Notes
    -----
    ``StationCorruptionModel()`` preserves the historical native defaults:
    thermal noise, opacity calibration, feed rotation, and a station-common
    0.04-dex gain with uniformly random phase; leakage remains disabled.
    """

    thermal_noise: bool = True
    opacity_calibrated: bool = True
    feed_rotation: bool = True
    common_gain: GainModel | None = field(
        default_factory=lambda: GainModel(
            amplitude_sigma_dex=0.04,
            phase_distribution="uniform",
        )
    )
    leakage: LeakageModel | None = None
    path_gain_overrides: Mapping[str, Mapping[str, GainModel]] = field(
        default_factory=dict
    )
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
        if self.common_gain is not None and not isinstance(self.common_gain, GainModel):
            raise TypeError("common_gain must be a GainModel or None.")
        if self.leakage is not None and not isinstance(self.leakage, LeakageModel):
            raise TypeError("leakage must be a LeakageModel or None.")
        if not isinstance(self.path_gain_overrides, Mapping):
            raise TypeError("path_gain_overrides must be a mapping.")
        normalized = {}
        for station, paths in self.path_gain_overrides.items():
            if not isinstance(paths, Mapping):
                raise TypeError("Each path_gain_overrides station value must be a mapping.")
            normalized_paths = {}
            for feed_id, model in paths.items():
                if not isinstance(model, GainModel):
                    raise TypeError("Each path gain override must be a GainModel.")
                normalized_paths[str(feed_id)] = model
            normalized[str(station)] = MappingProxyType(normalized_paths)
        object.__setattr__(
            self,
            "path_gain_overrides",
            MappingProxyType(normalized),
        )

    def validate_receptors(self, station_names, receptors):
        """Validate path overrides against a resolved native receptor table.

        Parameters
        ----------
        station_names : iterable of str
            Ordered native station names.
        receptors : ReceptorTable
            Resolved station/feed inventory for the native dataset.

        Raises
        ------
        ValueError
            If an override references an unknown station or feed.
        """

        names = tuple(str(name) for name in station_names)
        unknown_stations = set(self.path_gain_overrides) - set(names)
        if unknown_stations:
            raise ValueError(
                "Path gain overrides reference unknown stations: {0}.".format(
                    ", ".join(sorted(unknown_stations))
                )
            )
        for station, paths in self.path_gain_overrides.items():
            station_index = names.index(station)
            valid_feeds = {
                receptors.feed_id[index]
                for index in np.flatnonzero(receptors.station_index == station_index)
            }
            unknown_feeds = set(paths) - valid_feeds
            if unknown_feeds:
                raise ValueError(
                    "Path gain overrides reference unknown feeds for {0}: {1}.".format(
                        station,
                        ", ".join(sorted(unknown_feeds)),
                    )
                )

    def path_gain_model(self, station, feed_id):
        """Return the optional independent gain model for one voltage path.

        Parameters
        ----------
        station : str
            Station name.
        feed_id : str
            Local feed identifier.

        Returns
        -------
        GainModel or None
            Override model, or ``None`` when the path has no independent gain.
        """

        return self.path_gain_overrides.get(str(station), {}).get(str(feed_id))
