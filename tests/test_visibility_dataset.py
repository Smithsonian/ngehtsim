"""Tests for the internal visibility dataset and ehtim compatibility adapters."""

import numpy as np
import pytest

from ngehtsim.obs.visibility_dataset import (
    CIRCULAR_CORRELATIONS,
    LINEAR_CIRCULAR_CORRELATIONS,
    StationTable,
    VisibilityDataset,
)


def stations():
    return StationTable(
        names=("ALMA", "APEX", "LMT"),
        position_itrs_m=np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))),
        sefd_r_jy=np.array((100.0, 110.0, 120.0)),
        sefd_l_jy=np.array((101.0, 111.0, 121.0)),
        leakage_r=np.zeros(3, dtype=complex),
        leakage_l=np.zeros(3, dtype=complex),
        feed_rotation_par=np.ones(3),
        feed_rotation_elev=np.zeros(3),
        feed_rotation_offset_deg=np.zeros(3),
    )


def dataset(**overrides):
    values = {
        "stations": stations(),
        "time_mjd": np.array((60000.0, 60000.01)),
        "integration_time_s": np.array((10.0, 10.0)),
        "antenna1": np.array((0, 1)),
        "antenna2": np.array((1, 2)),
        "uvw_m": np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))),
        "tau1": np.array((0.1, 0.2)),
        "tau2": np.array((0.3, 0.4)),
        "channel_frequency_hz": np.array((230.0e9, 231.0e9)),
        "channel_bandwidth_hz": np.array((2.0e9, 2.0e9)),
        "spectral_window_id": np.array((0, 1)),
        "correlation_layouts": (CIRCULAR_CORRELATIONS, LINEAR_CIRCULAR_CORRELATIONS),
        "row_layout_id": np.array((0, 1)),
        "visibilities": np.ones((2, 2, 4), dtype=complex),
        "weights": np.full((2, 2, 4), 4.0),
        "flags": np.zeros((2, 2, 4), dtype=bool),
        "source": "M87",
        "ra_hours": 12.5,
        "dec_degrees": 12.4,
        "scan_start_mjd": np.array((59999.99,)),
        "scan_stop_mjd": np.array((60000.02,)),
    }
    values.update(overrides)
    return VisibilityDataset(**values)


def test_dataset_supports_mixed_layouts_and_multiple_channels():
    data = dataset()

    assert data.row_count == 2
    assert data.channel_count == 2
    assert data.correlation_layouts[data.row_layout_id[0]] == CIRCULAR_CORRELATIONS
    assert data.correlation_layouts[data.row_layout_id[1]] == LINEAR_CIRCULAR_CORRELATIONS
    assert not data.visibilities.flags.writeable
    assert not data.stations.position_itrs_m.flags.writeable


def test_dataset_select_rows_preserves_selected_rows_and_metadata():
    original = dataset()

    selected = original.select_rows(np.array((True, False)))

    assert selected.row_count == 1
    assert np.array_equal(selected.time_mjd, original.time_mjd[:1])
    assert np.array_equal(selected.uvw_m, original.uvw_m[:1])
    assert np.array_equal(selected.visibilities, original.visibilities[:1])
    assert np.array_equal(selected.weights, original.weights[:1])
    assert np.array_equal(selected.flags, original.flags[:1])
    assert selected.stations.names == original.stations.names
    assert np.array_equal(selected.scan_start_mjd, original.scan_start_mjd)
    assert np.array_equal(selected.scan_stop_mjd, original.scan_stop_mjd)
    assert not selected.visibilities.flags.writeable


@pytest.mark.parametrize(
    "row_mask",
    (
        np.array((1, 0), dtype=np.intp),
        np.array((True,)),
        np.array(((True, False),)),
    ),
)
def test_dataset_select_rows_rejects_invalid_masks(row_mask):
    with pytest.raises(ValueError, match="row_mask"):
        dataset().select_rows(row_mask)


def test_dataset_take_rows_reorders_selected_rows():
    original = dataset()

    reordered = original.take_rows(np.array((1, 0), dtype=np.intp))

    assert np.array_equal(reordered.time_mjd, original.time_mjd[::-1])
    assert np.array_equal(reordered.antenna1, original.antenna1[::-1])
    assert np.array_equal(reordered.visibilities, original.visibilities[::-1])


@pytest.mark.parametrize(
    "row_indices",
    (
        np.array((0.0,)),
        np.array(((0, 1),)),
        np.array((2,), dtype=np.intp),
    ),
)
def test_dataset_take_rows_rejects_invalid_indices(row_indices):
    with pytest.raises(ValueError, match="row_indices"):
        dataset().take_rows(row_indices)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"visibilities": np.ones((2, 2, 3), dtype=complex)}, "visibilities"),
        ({"antenna2": np.array((1, 3))}, "antenna2"),
        ({"row_layout_id": np.array((0, 2))}, "row_layout_id"),
        ({"weights": np.zeros((2, 2, 4))}, "weights"),
    ],
)
def test_dataset_rejects_invalid_shapes_and_indices(overrides, message):
    with pytest.raises(ValueError, match=message):
        dataset(**overrides)


def test_ehtim_circular_obsdata_round_trip():
    obs = dataset(
        channel_frequency_hz=np.array((230.0e9,)),
        channel_bandwidth_hz=np.array((2.0e9,)),
        spectral_window_id=np.array((0,)),
        correlation_layouts=(CIRCULAR_CORRELATIONS,),
        row_layout_id=np.array((0, 0)),
        visibilities=np.array(
            [[[1.0 + 1.0j, 2.0 + 1.0j, 3.0 + 1.0j, 4.0 + 1.0j]],
            [[5.0 + 1.0j, 6.0 + 1.0j, 7.0 + 1.0j, 8.0 + 1.0j]]]
        ),
        weights=np.full((2, 1, 4), 4.0),
        flags=np.zeros((2, 1, 4), dtype=bool),
    ).to_ehtim_obsdata()

    restored = VisibilityDataset.from_ehtim_obsdata(obs)

    assert restored.correlation_layouts == (CIRCULAR_CORRELATIONS,)
    assert np.array_equal(restored.antenna1, np.array((0, 1)))
    assert np.array_equal(restored.antenna2, np.array((1, 2)))
    expected_visibilities = np.column_stack(
        (obs.data["rrvis"], obs.data["llvis"], obs.data["rlvis"], obs.data["lrvis"])
    )
    expected_sigma = np.column_stack(
        (obs.data["rrsigma"], obs.data["llsigma"], obs.data["rlsigma"], obs.data["lrsigma"])
    )
    assert np.allclose(restored.visibilities[:, 0], expected_visibilities)
    assert np.allclose(restored.weights[:, 0], 1.0 / np.square(expected_sigma))
    assert restored.source == obs.source
    assert restored.ampcal == obs.ampcal


def test_ehtim_stokes_obsdata_is_converted_to_circular_correlations():
    circular_obs = dataset(
        channel_frequency_hz=np.array((230.0e9,)),
        channel_bandwidth_hz=np.array((2.0e9,)),
        spectral_window_id=np.array((0,)),
        correlation_layouts=(CIRCULAR_CORRELATIONS,),
        row_layout_id=np.array((0, 0)),
        visibilities=np.ones((2, 1, 4), dtype=complex),
        weights=np.full((2, 1, 4), 4.0),
        flags=np.zeros((2, 1, 4), dtype=bool),
    ).to_ehtim_obsdata()

    restored = VisibilityDataset.from_ehtim_obsdata(
        circular_obs.switch_polrep("stokes")
    )

    assert restored.correlation_layouts == (CIRCULAR_CORRELATIONS,)
    assert np.allclose(restored.visibilities[:, 0], 1.0)


def test_ehtim_adapter_rejects_unrepresentable_datasets():
    with pytest.raises(ValueError, match="exactly one spectral channel"):
        dataset().to_ehtim_obsdata()

    mixed = dataset(
        channel_frequency_hz=np.array((230.0e9,)),
        channel_bandwidth_hz=np.array((2.0e9,)),
        spectral_window_id=np.array((0,)),
        correlation_layouts=(LINEAR_CIRCULAR_CORRELATIONS,),
        row_layout_id=np.array((0, 0)),
        visibilities=np.ones((2, 1, 4), dtype=complex),
        weights=np.ones((2, 1, 4)),
        flags=np.zeros((2, 1, 4), dtype=bool),
    )
    with pytest.raises(ValueError, match="circular"):
        mixed.to_ehtim_obsdata()

    flagged = dataset(
        channel_frequency_hz=np.array((230.0e9,)),
        channel_bandwidth_hz=np.array((2.0e9,)),
        spectral_window_id=np.array((0,)),
        correlation_layouts=(CIRCULAR_CORRELATIONS,),
        row_layout_id=np.array((0, 0)),
        visibilities=np.ones((2, 1, 4), dtype=complex),
        weights=np.ones((2, 1, 4)),
        flags=np.ones((2, 1, 4), dtype=bool),
    )
    with pytest.raises(ValueError, match="flagged"):
        flagged.to_ehtim_obsdata()


def test_ehtim_adapter_preserves_rows_across_mjd_boundaries():
    original = dataset(
        channel_frequency_hz=np.array((230.0e9,)),
        channel_bandwidth_hz=np.array((2.0e9,)),
        spectral_window_id=np.array((0,)),
        correlation_layouts=(CIRCULAR_CORRELATIONS,),
        row_layout_id=np.array((0, 0)),
        visibilities=np.ones((2, 1, 4), dtype=complex),
        weights=np.ones((2, 1, 4)),
        flags=np.zeros((2, 1, 4), dtype=bool),
        time_mjd=np.array((60000.99, 60001.01)),
    )

    restored = VisibilityDataset.from_ehtim_obsdata(original.to_ehtim_obsdata())

    assert np.allclose(restored.time_mjd, original.time_mjd)
