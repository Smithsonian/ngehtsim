"""Unit tests for cadence-aware native station-effect configuration."""

import numpy as np
import pytest

from ngehtsim.obs.station_effects import (
    GainModel,
    GainRatioModel,
    LeakageModel,
    RealizationCadence,
    StationCorruptionModel,
    realization_group_ids,
)
from ngehtsim.obs.visibility_dataset import (
    ReceptorTable,
    StationTable,
    VisibilityDataset,
    standard_products_for_rows,
)


class FixedRng:
    """Provide reproducible draws for compact effect-model tests."""

    def normal(self, loc=0.0, scale=1.0, size=None):
        return 1.0 if size is None else np.ones(size, dtype=float)

    def uniform(self, low=0.0, high=1.0, size=None):
        return 0.0 if size is None else np.zeros(size, dtype=float)


def _dataset(scans=True):
    """Create four two-station rows distributed across two scans."""

    stations = StationTable(
        names=("AA", "BB"),
        position_itrs_m=np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        sefd_r_jy=np.ones(2),
        sefd_l_jy=np.ones(2),
        leakage_r=np.zeros(2, dtype=complex),
        leakage_l=np.zeros(2, dtype=complex),
        feed_rotation_par=np.zeros(2),
        feed_rotation_elev=np.zeros(2),
        feed_rotation_offset_deg=np.zeros(2),
    )
    receptors = ReceptorTable.from_station_labels(2, ("R", "L"), "CIRCULAR")
    antenna1 = np.zeros(4, dtype=int)
    antenna2 = np.ones(4, dtype=int)
    products, row_product_id = standard_products_for_rows(
        receptors,
        antenna1,
        antenna2,
        ("RR", "LL", "RL", "LR"),
    )
    time_mjd = 60000.0 + (np.array((0.0, 10.0, 100.0, 110.0)) / 86400.0)
    data = np.ones((4, 1, 4), dtype=complex)
    values = dict(
        stations=stations,
        receptors=receptors,
        correlation_products=products,
        time_mjd=time_mjd,
        integration_time_s=np.full(4, 10.0),
        antenna1=antenna1,
        antenna2=antenna2,
        uvw_m=np.zeros((4, 3)),
        tau1=np.zeros(4),
        tau2=np.zeros(4),
        channel_frequency_hz=np.array((230.0e9,)),
        channel_bandwidth_hz=np.array((2.0e9,)),
        spectral_window_id=np.array((0,), dtype=int),
        row_product_id=row_product_id,
        visibilities=data,
        sigma_jy=np.ones((4, 1, 4)),
        flags=np.zeros((4, 1, 4), dtype=bool),
        source="TEST",
        ra_hours=12.0,
        dec_degrees=10.0,
    )
    if scans:
        values.update(
            scan_start_mjd=np.array((time_mjd[0] - 1.0e-9, time_mjd[2] - 1.0e-9)),
            scan_stop_mjd=np.array((time_mjd[1] + 1.0e-9, time_mjd[3] + 1.0e-9)),
        )
    return VisibilityDataset(**values)


def test_gain_model_draws_declared_amplitude_and_phase_distribution():
    gain = GainModel(amplitude_sigma_dex=0.5, phase_distribution="uniform")

    assert gain.sample(FixedRng()) == pytest.approx(10.0 ** 0.5)
    assert gain.amplitude_cadence == RealizationCadence.scan()
    assert gain.phase_cadence == RealizationCadence.scan()


def test_gain_ratio_defaults_to_track_and_can_use_station_local_feed_order():
    ratio = GainRatioModel(amplitude_sigma_dex=0.2)
    named_ratio = GainRatioModel("X", "Y")

    assert ratio.amplitude_cadence == RealizationCadence.track()
    assert ratio.phase_cadence == RealizationCadence.track()
    assert ratio.feed_a is None
    assert ratio.feed_b is None
    assert named_ratio.feed_a == "X"
    assert named_ratio.feed_b == "Y"


def test_leakage_model_draws_local_feed_terms_with_explicit_means():
    leakage = LeakageModel(
        "X",
        "Y",
        leakage_a_mean=0.02 + 0.03j,
        leakage_b_mean=-0.04 + 0.01j,
        component_sigma=0.1,
    )

    assert leakage.feed_a == "X"
    assert leakage.feed_b == "Y"
    assert leakage.cadence == RealizationCadence.track()
    assert leakage.sample(FixedRng()) == pytest.approx(
        (0.12 + 0.13j, 0.06 + 0.11j)
    )


def test_realization_cadences_group_rows_without_guessing_scans():
    dataset = _dataset()

    assert np.array_equal(
        realization_group_ids(dataset, RealizationCadence.integration()),
        np.array((0, 1, 2, 3)),
    )
    assert np.array_equal(
        realization_group_ids(dataset, RealizationCadence.scan()),
        np.array((0, 0, 1, 1)),
    )
    assert np.array_equal(
        realization_group_ids(dataset, RealizationCadence.track()),
        np.zeros(4, dtype=int),
    )
    assert np.array_equal(
        realization_group_ids(dataset, RealizationCadence.interval(30.0)),
        np.array((0, 0, 1, 1)),
    )
    with pytest.raises(ValueError, match="scan metadata"):
        realization_group_ids(_dataset(scans=False), RealizationCadence.scan())


def test_station_corruption_model_resolves_immutable_default_and_override_models():
    gain = GainModel(amplitude_sigma_dex=0.01)
    gain_override = GainModel(amplitude_sigma_dex=0.02)
    leakage = LeakageModel(component_sigma=0.01)
    leakage_override = LeakageModel(component_sigma=0.02)
    ratio = GainRatioModel(amplitude_sigma_dex=0.01)
    ratio_override = GainRatioModel("X", "Y", amplitude_sigma_dex=0.02)
    effects = StationCorruptionModel(
        station_gain=gain,
        station_gain_overrides={"ALMA": gain_override, "APEX": None},
        leakage=leakage,
        leakage_overrides={"ALMA": leakage_override, "APEX": None},
        gain_ratio=ratio,
        gain_ratio_overrides={"ALMA": ratio_override, "APEX": None},
    )

    assert effects.station_gain_model("ALMA") == gain_override
    assert effects.station_gain_model("LMT") == gain
    assert effects.station_gain_model("APEX") is None
    assert effects.leakage_model("ALMA") == leakage_override
    assert effects.leakage_model("LMT") == leakage
    assert effects.leakage_model("APEX") is None
    assert effects.gain_ratio_model("ALMA") == ratio_override
    assert effects.gain_ratio_model("LMT") == ratio
    assert effects.gain_ratio_model("APEX") is None
    assert effects.gain_ratio_feed_pair("LMT", ("R", "L")) == ("R", "L")
    assert effects.gain_ratio_feed_pair("ALMA", ("X", "Y")) == ("X", "Y")
    assert effects.leakage_feed_pair("LMT", ("R", "L")) == ("R", "L")
    assert effects.leakage_feed_pair("ALMA", ("X", "Y")) == ("X", "Y")
    assert effects.has_gain_corruption
    assert effects.has_leakage_corruption
    with pytest.raises(TypeError):
        effects.station_gain_overrides["ALMA"] = gain
    with pytest.raises(TypeError):
        effects.leakage_overrides["ALMA"] = leakage
    with pytest.raises(TypeError):
        effects.gain_ratio_overrides["ALMA"] = ratio


def test_station_corruption_model_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="non-negative"):
        GainModel(amplitude_sigma_dex=-0.01)
    with pytest.raises(ValueError, match="phase_distribution"):
        GainModel(phase_distribution="gaussian")
    with pytest.raises(ValueError, match="non-negative"):
        LeakageModel(component_sigma=-0.01)
    with pytest.raises(ValueError, match="both be supplied"):
        LeakageModel("X")
    with pytest.raises(ValueError, match="finite"):
        LeakageModel(leakage_a_mean=np.inf)
    with pytest.raises(ValueError, match="distinct"):
        GainRatioModel("X", "X")
    with pytest.raises(ValueError, match="both be supplied"):
        GainRatioModel("X")
    with pytest.raises(TypeError, match="station_gain_overrides"):
        StationCorruptionModel(station_gain_overrides=("ALMA",))
    with pytest.raises(TypeError, match="GainModel"):
        StationCorruptionModel(station_gain_overrides={"ALMA": 0.02})
    with pytest.raises(TypeError, match="leakage_overrides"):
        StationCorruptionModel(leakage_overrides=("ALMA",))
    with pytest.raises(TypeError, match="LeakageModel"):
        StationCorruptionModel(leakage_overrides={"ALMA": 0.02})
    with pytest.raises(TypeError, match="gain_ratio_overrides"):
        StationCorruptionModel(gain_ratio_overrides=("ALMA",))
    with pytest.raises(TypeError, match="GainRatioModel"):
        StationCorruptionModel(gain_ratio_overrides={"ALMA": 0.02})


def test_station_corruption_model_validates_two_feed_ratio_layouts():
    receptors = ReceptorTable(
        station_index=np.array((0, 0, 1), dtype=int),
        feed_id=("X", "Y", "R"),
        polarization_label=("X", "Y", "R"),
        basis=("LINEAR", "LINEAR", "CIRCULAR"),
    )
    effects = StationCorruptionModel(gain_ratio=GainRatioModel())
    effects.validate_receptors(("ALMA", "APEX"), receptors)

    with pytest.raises(ValueError, match="unknown stations"):
        StationCorruptionModel(
            station_gain_overrides={"LMT": GainModel()},
        ).validate_receptors(("ALMA", "APEX"), receptors)
    with pytest.raises(ValueError, match="one feed"):
        StationCorruptionModel(
            gain_ratio_overrides={"APEX": GainRatioModel("R", "L")},
        ).validate_receptors(("ALMA", "APEX"), receptors)
    with pytest.raises(ValueError, match="must match"):
        StationCorruptionModel(
            gain_ratio=GainRatioModel("R", "L"),
        ).validate_receptors(("ALMA", "APEX"), receptors)

    three_feed_receptors = ReceptorTable(
        station_index=np.array((0, 0, 0), dtype=int),
        feed_id=("R", "L", "X"),
        polarization_label=("R", "L", "X"),
        basis=("CIRCULAR", "CIRCULAR", "LINEAR"),
    )
    with pytest.raises(NotImplementedError, match="exactly two feeds"):
        StationCorruptionModel(gain_ratio=GainRatioModel()).validate_receptors(
            ("ALMA",),
            three_feed_receptors,
        )
    with pytest.raises(NotImplementedError, match="Local-feed leakage"):
        StationCorruptionModel(leakage=LeakageModel()).validate_receptors(
            ("ALMA",),
            three_feed_receptors,
        )
