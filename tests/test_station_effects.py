"""Unit tests for declarative native station-effect configuration."""

import numpy as np
import pytest

from ngehtsim.obs.station_effects import GainModel, LeakageModel, StationCorruptionModel
from ngehtsim.obs.visibility_dataset import ReceptorTable


class FixedRng:
    """Provide reproducible draws for compact effect-model tests."""

    def normal(self, loc=0.0, scale=1.0, size=None):
        return 1.0 if size is None else np.ones(size, dtype=float)

    def uniform(self, low=0.0, high=1.0, size=None):
        return 0.0 if size is None else np.zeros(size, dtype=float)


def _receptors():
    return ReceptorTable(
        station_index=np.array((0, 0, 1), dtype=int),
        feed_id=("X", "Y", "R"),
        polarization_label=("X", "Y", "R"),
        basis=("LINEAR", "LINEAR", "CIRCULAR"),
    )


def test_gain_model_draws_declared_amplitude_and_phase_distribution():
    gain = GainModel(amplitude_sigma_dex=0.5, phase_distribution="uniform")

    assert gain.sample(FixedRng()) == pytest.approx(10.0 ** 0.5)


def test_station_corruption_model_normalizes_immutable_path_overrides():
    effects = StationCorruptionModel(
        path_gain_overrides={"ALMA": {"X": GainModel(0.02)}},
    )

    assert effects.path_gain_model("ALMA", "X") == GainModel(0.02)
    assert effects.path_gain_model("ALMA", "Y") is None
    with pytest.raises(TypeError):
        effects.path_gain_overrides["ALMA"] = {}


def test_station_corruption_model_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="non-negative"):
        GainModel(amplitude_sigma_dex=-0.01)
    with pytest.raises(ValueError, match="phase_distribution"):
        GainModel(phase_distribution="gaussian")
    with pytest.raises(ValueError, match="non-negative"):
        LeakageModel(component_sigma=-0.01)
    with pytest.raises(TypeError, match="path_gain_overrides"):
        StationCorruptionModel(path_gain_overrides=("ALMA",))
    with pytest.raises(TypeError, match="GainModel"):
        StationCorruptionModel(path_gain_overrides={"ALMA": {"X": 0.02}})


def test_station_corruption_model_validates_station_local_feed_ids():
    effects = StationCorruptionModel(
        path_gain_overrides={"ALMA": {"X": GainModel(0.02)}},
    )
    effects.validate_receptors(("ALMA", "APEX"), _receptors())

    with pytest.raises(ValueError, match="unknown stations"):
        StationCorruptionModel(
            path_gain_overrides={"LMT": {"X": GainModel(0.02)}},
        ).validate_receptors(("ALMA", "APEX"), _receptors())
    with pytest.raises(ValueError, match="unknown feeds"):
        StationCorruptionModel(
            path_gain_overrides={"ALMA": {"R": GainModel(0.02)}},
        ).validate_receptors(("ALMA", "APEX"), _receptors())
