"""Tests for native re-simulation from imported observation templates."""

from __future__ import annotations

import numpy as np
import pytest
import ehtim as eh

from ngehtsim.obs.observation_template import ObservationTemplate
from ngehtsim.obs.station_effects import (
    GainModel,
    GainRatioModel,
    RealizationCadence,
    StationCorruptionModel,
)
from ngehtsim.obs.receptor_configuration import resolve_receptor_configuration
from ngehtsim.obs.visibility_dataset import (
    StationTable,
    VisibilityDataset,
    receptor_products_for_rows,
)


def _template_dataset():
    """Build a compact circular template with two stored scans."""

    stations = StationTable(
        names=("AA", "BB"),
        position_itrs_m=np.array(((6378137.0, 0.0, 0.0), (0.0, 6378137.0, 0.0))),
        sefd_r_jy=np.array((3000.0, 4000.0)),
        sefd_l_jy=np.array((3000.0, 4000.0)),
        leakage_r=np.zeros(2, dtype=complex),
        leakage_l=np.zeros(2, dtype=complex),
        feed_rotation_par=np.ones(2),
        feed_rotation_elev=np.zeros(2),
        feed_rotation_offset_deg=np.zeros(2),
    )
    configuration = resolve_receptor_configuration(stations.names)
    time_mjd = 60000.0 + np.array((0.0, 20.0, 3600.0, 3620.0)) / 86400.0
    antenna1 = np.zeros(4, dtype=np.intp)
    antenna2 = np.ones(4, dtype=np.intp)
    products, row_product_id = receptor_products_for_rows(
        configuration.receptors,
        antenna1,
        antenna2,
    )
    shape = (4, 1, row_product_id.shape[1])
    return VisibilityDataset(
        stations=stations,
        receptors=configuration.receptors,
        correlation_products=products,
        time_mjd=time_mjd,
        integration_time_s=np.full(4, 10.0),
        antenna1=antenna1,
        antenna2=antenna2,
        uvw_m=np.array(((10.0, 20.0, 30.0), (15.0, 25.0, 35.0),
                        (20.0, 30.0, 40.0), (25.0, 35.0, 45.0))),
        tau1=np.full(4, 0.05),
        tau2=np.full(4, 0.10),
        channel_frequency_hz=np.array((230.0e9,)),
        channel_bandwidth_hz=np.array((2.0e9,)),
        spectral_window_id=np.array((0,), dtype=np.intp),
        row_product_id=row_product_id,
        visibilities=np.zeros(shape, dtype=complex),
        sigma_jy=np.full(shape, 0.25),
        flags=np.zeros(shape, dtype=bool),
        source="M87",
        ra_hours=12.5,
        dec_degrees=12.4,
        scan_start_mjd=np.array((time_mjd[0], time_mjd[2])),
        scan_stop_mjd=np.array((time_mjd[1], time_mjd[3])),
    )


def _model():
    """Build a small analytic source model for native sampling tests."""

    return eh.model.Model().add_circ_gauss(
        F0=1.0,
        FWHM=40.0 * eh.RADPERUAS,
    )


def _clean_effects(**overrides):
    """Return deterministic template effects with no unrelated corruption."""

    values = {
        "thermal_noise": False,
        "opacity_calibrated": True,
        "feed_rotation": False,
        "station_gain": None,
        "leakage": None,
        "flag_wind": False,
        "flag_daylight": False,
        "flag_sun": False,
    }
    values.update(overrides)
    return StationCorruptionModel(**values)


def test_template_substitution_preserves_sampling_uncertainty_and_flags():
    """Unchanged receptor layouts retain exact imported observation metadata."""

    template = ObservationTemplate.from_dataset(_template_dataset())
    result = template.simulate(
        _model(),
        effects=_clean_effects(),
        source_name="Synthetic M87",
        transform_backend="direct",
        random_seed=4,
    )

    output = result.dataset
    original = template.dataset
    for name in (
        "time_mjd",
        "integration_time_s",
        "antenna1",
        "antenna2",
        "uvw_m",
        "sigma_jy",
        "flags",
    ):
        assert np.array_equal(getattr(output, name), getattr(original, name))
    assert output.source == "Synthetic M87"
    assert np.any(np.abs(output.visibilities) > 0.0)


def test_template_station_gain_and_ratio_follow_symmetric_two_feed_model():
    """The recorded feed factors implement G_A=G sqrt(R), G_B=G/sqrt(R)."""

    template = ObservationTemplate.from_dataset(_template_dataset())
    effects = _clean_effects(
        station_gain=GainModel(
            amplitude_mean_dex=0.1,
            phase_mean_rad=0.4,
            phase_distribution="uniform",
            amplitude_cadence=RealizationCadence.scan(),
            phase_cadence=RealizationCadence.scan(),
        ),
        gain_ratio_overrides={
            "AA": GainRatioModel(
                "R",
                "L",
                amplitude_mean_dex=0.2,
                phase_mean_rad=0.6,
                amplitude_cadence=RealizationCadence.track(),
                phase_cadence=RealizationCadence.track(),
            ),
        },
    )
    result = template.simulate(
        _model(),
        effects=effects,
        transform_backend="direct",
        random_seed=5,
    )

    receptors = result.dataset.receptors
    r_index = next(
        index for index, (station, feed) in enumerate(zip(
            receptors.station_index,
            receptors.feed_id,
        )) if result.dataset.stations.names[station] == "AA" and feed == "R"
    )
    l_index = next(
        index for index, (station, feed) in enumerate(zip(
            receptors.station_index,
            receptors.feed_id,
        )) if result.dataset.stations.names[station] == "AA" and feed == "L"
    )
    factors = result.station_terms["gain_ratio_factors"]
    assert np.allclose(factors[:, r_index] * factors[:, l_index], 1.0)
    assert np.allclose(factors[:, r_index] / factors[:, l_index], 10.0 ** 0.2 * np.exp(0.6j))
    assert np.allclose(result.station_terms["common_gain1"][:2], result.station_terms["common_gain1"][0])
    assert not np.allclose(result.station_terms["common_gain1"][:2], result.station_terms["common_gain1"][2:])


def test_template_relayout_supports_mixed_feeds_and_fitseht_round_trip(tmp_path):
    """A requested X/Y station creates a losslessly archivable mixed layout."""

    template = ObservationTemplate.from_dataset(_template_dataset())
    result = template.simulate(
        _model(),
        effects=_clean_effects(),
        station_receptors={"AA": ("X", "Y")},
        transform_backend="direct",
        random_seed=6,
    )

    output = result.dataset
    assert output.receptors.polarization_label == ("X", "Y", "R", "L")
    assert output.product_slot_count == 4
    assert np.allclose(output.sigma_jy, 0.25)
    archive = tmp_path / "mixed.ehtfits"
    output.to_ehtfits(archive)
    restored = VisibilityDataset.from_ehtfits(archive)
    assert restored.receptors.polarization_label == output.receptors.polarization_label
    assert np.array_equal(restored.row_product_id, output.row_product_id)
    assert np.allclose(restored.visibilities, output.visibilities)


def test_template_feed_rotation_requires_explicit_mount_metadata():
    """Template station codes never receive implicit ngehtsim identities."""

    template = ObservationTemplate.from_dataset(_template_dataset())
    effects = _clean_effects(feed_rotation=True)
    with pytest.raises(ValueError, match="requires mount_types or station_resolver"):
        template.simulate(_model(), effects=effects, transform_backend="direct")

    result = template.simulate(
        _model(),
        effects=effects,
        station_resolver={"AA": "ALMA", "BB": "APEX"},
        transform_backend="direct",
    )
    assert np.any(np.abs(result.station_terms["par1"]) > 0.0)


def test_template_scan_cadence_requires_stored_scan_intervals():
    """A gain process cannot silently invent scans from timestamp gaps."""

    template = ObservationTemplate.from_dataset(_template_dataset())
    without_scans = ObservationTemplate.from_dataset(
        VisibilityDataset(
            **{
                name: getattr(template.dataset, name)
                for name in template.dataset.__dataclass_fields__
                if name not in ("scan_start_mjd", "scan_stop_mjd")
            },
        )
    )
    with pytest.raises(ValueError, match="scan cadence requires dataset scan metadata"):
        without_scans.simulate(
            _model(),
            effects=_clean_effects(station_gain=GainModel()),
            transform_backend="direct",
        )
