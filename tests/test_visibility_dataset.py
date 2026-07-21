"""Tests for native visibility data and optional ehtim adapters."""

from dataclasses import replace

import numpy as np
import pytest

from ngehtsim.obs.visibility_dataset import (
    CIRCULAR_PRODUCT_LABELS,
    CorrelationProductTable,
    ReceptorTable,
    StationTable,
    VisibilityDataset,
    standard_products_for_rows,
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


def circular_dataset(**overrides):
    antenna1 = np.array((0, 1))
    antenna2 = np.array((1, 2))
    receptors = ReceptorTable.from_station_labels(3, ("R", "L"), "CIRCULAR")
    products, row_product_id = standard_products_for_rows(
        receptors,
        antenna1,
        antenna2,
        CIRCULAR_PRODUCT_LABELS,
    )
    values = {
        "stations": stations(),
        "receptors": receptors,
        "correlation_products": products,
        "time_mjd": np.array((60000.0, 60000.01)),
        "integration_time_s": np.array((10.0, 10.0)),
        "antenna1": antenna1,
        "antenna2": antenna2,
        "uvw_m": np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))),
        "tau1": np.array((0.1, 0.2)),
        "tau2": np.array((0.3, 0.4)),
        "channel_frequency_hz": np.array((230.0e9, 231.0e9)),
        "channel_bandwidth_hz": np.array((2.0e9, 2.0e9)),
        "spectral_window_id": np.array((0, 1)),
        "row_product_id": row_product_id,
        "visibilities": np.ones((2, 2, 4), dtype=complex),
        "sigma_jy": np.full((2, 2, 4), 0.5),
        "flags": np.zeros((2, 2, 4), dtype=bool),
        "source": "M87",
        "ra_hours": 12.5,
        "dec_degrees": 12.4,
        "scan_start_mjd": np.array((59999.99,)),
        "scan_stop_mjd": np.array((60000.02,)),
    }
    values.update(overrides)
    return VisibilityDataset(**values)


def mixed_feed_dataset():
    """Return a dataset with a two-by-two and a two-by-three baseline."""

    receptor_table = ReceptorTable(
        station_index=np.array((0, 0, 1, 1, 2, 2, 2)),
        feed_id=("1", "2", "1", "2", "1", "2", "3"),
        polarization_label=("R", "L", "X", "Y", "R", "X", "Y"),
        basis=("CIRCULAR", "CIRCULAR", "LINEAR", "LINEAR", "CIRCULAR", "LINEAR", "LINEAR"),
    )
    products = CorrelationProductTable(
        receptor1_id=np.array((0, 0, 1, 1, 2, 2, 2, 3, 3, 3)),
        receptor2_id=np.array((2, 3, 2, 3, 4, 5, 6, 4, 5, 6)),
    )
    row_product_id = np.array(((0, 1, 2, 3, -1, -1), (4, 5, 6, 7, 8, 9)))
    flags = np.zeros((2, 2, 6), dtype=bool)
    flags[0, :, 4:] = True
    sigma_jy = np.full((2, 2, 6), 0.5)
    sigma_jy[0, :, 4:] = np.nan
    visibilities = np.ones((2, 2, 6), dtype=complex)
    visibilities[0, :, 4:] = np.nan + 1.0j * np.nan
    return VisibilityDataset(
        stations=stations(),
        receptors=receptor_table,
        correlation_products=products,
        time_mjd=np.array((60000.0, 60000.01)),
        integration_time_s=np.array((10.0, 10.0)),
        antenna1=np.array((0, 1)),
        antenna2=np.array((1, 2)),
        uvw_m=np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))),
        tau1=np.array((0.1, 0.2)),
        tau2=np.array((0.3, 0.4)),
        channel_frequency_hz=np.array((230.0e9, 231.0e9)),
        channel_bandwidth_hz=np.array((2.0e9, 2.0e9)),
        spectral_window_id=np.array((0, 1)),
        row_product_id=row_product_id,
        visibilities=visibilities,
        sigma_jy=sigma_jy,
        flags=flags,
        source="M87",
        ra_hours=12.5,
        dec_degrees=12.4,
    )


def test_dataset_supports_mixed_receiver_counts_and_multiple_channels():
    data = mixed_feed_dataset()

    assert data.row_count == 2
    assert data.channel_count == 2
    assert data.product_slot_count == 6
    assert np.array_equal(data.row_product_id[0], np.array((0, 1, 2, 3, -1, -1)))
    assert np.array_equal(data.row_product_id[1], np.array((4, 5, 6, 7, 8, 9)))
    assert np.all(data.flags[~data.sample_present])
    assert np.all(np.isnan(data.sigma_jy[~data.sample_present]))
    assert not data.visibilities.flags.writeable
    assert not data.stations.position_itrs_m.flags.writeable


def test_dataset_select_rows_preserves_selected_rows_and_metadata():
    original = mixed_feed_dataset()

    selected = original.select_rows(np.array((True, False)))

    assert selected.row_count == 1
    assert np.array_equal(selected.time_mjd, original.time_mjd[:1])
    assert np.array_equal(selected.uvw_m, original.uvw_m[:1])
    assert np.array_equal(selected.row_product_id, original.row_product_id[:1])
    assert np.array_equal(selected.visibilities, original.visibilities[:1], equal_nan=True)
    assert np.array_equal(selected.sigma_jy, original.sigma_jy[:1], equal_nan=True)
    assert np.array_equal(selected.flags, original.flags[:1])
    assert selected.stations.names == original.stations.names


def test_dataset_take_rows_reorders_selected_rows():
    original = mixed_feed_dataset()

    reordered = original.take_rows(np.array((1, 0), dtype=np.intp))

    assert np.array_equal(reordered.time_mjd, original.time_mjd[::-1])
    assert np.array_equal(reordered.antenna1, original.antenna1[::-1])
    assert np.array_equal(reordered.row_product_id, original.row_product_id[::-1])
    assert np.array_equal(reordered.visibilities, original.visibilities[::-1], equal_nan=True)


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
        circular_dataset().take_rows(row_indices)


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
        circular_dataset().select_rows(row_mask)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"visibilities": np.ones((2, 2, 3), dtype=complex)}, "visibilities"),
        ({"antenna2": np.array((1, 3))}, "antenna2"),
        ({"row_product_id": np.array(((0, 99, 2, 3), (4, 5, 6, 7)))}, "row_product_id"),
        ({"sigma_jy": np.zeros((2, 2, 4))}, "sigma_jy"),
    ],
)
def test_dataset_rejects_invalid_shapes_and_indices(overrides, message):
    with pytest.raises(ValueError, match=message):
        circular_dataset(**overrides)


def test_dataset_rejects_product_connected_to_wrong_station():
    data = circular_dataset()
    wrong = np.array(data.row_product_id, copy=True)
    wrong[0, 0] = data.row_product_id[1, 0]
    with pytest.raises(ValueError, match="first receptors"):
        circular_dataset(row_product_id=wrong)


def test_ehtim_circular_obsdata_round_trip_uses_sigma_directly():
    obs = circular_dataset(
        channel_frequency_hz=np.array((230.0e9,)),
        channel_bandwidth_hz=np.array((2.0e9,)),
        spectral_window_id=np.array((0,)),
        visibilities=np.array(
            [[[1.0 + 1.0j, 2.0 + 1.0j, 3.0 + 1.0j, 4.0 + 1.0j]],
            [[5.0 + 1.0j, 6.0 + 1.0j, 7.0 + 1.0j, 8.0 + 1.0j]]]
        ),
        sigma_jy=np.full((2, 1, 4), 0.5),
        flags=np.zeros((2, 1, 4), dtype=bool),
    ).to_ehtim_obsdata()

    restored = VisibilityDataset.from_ehtim_obsdata(obs)

    assert np.array_equal(restored.antenna1, np.array((0, 1)))
    assert np.array_equal(restored.antenna2, np.array((1, 2)))
    expected_visibilities = np.column_stack(
        (obs.data["rrvis"], obs.data["llvis"], obs.data["rlvis"], obs.data["lrvis"])
    )
    expected_sigma = np.column_stack(
        (obs.data["rrsigma"], obs.data["llsigma"], obs.data["rlsigma"], obs.data["lrsigma"])
    )
    slots = restored.circular_product_slots()
    rows = np.arange(restored.row_count)[:, np.newaxis]
    assert np.allclose(restored.visibilities[rows, 0, slots], expected_visibilities)
    assert np.allclose(restored.sigma_jy[rows, 0, slots], expected_sigma)
    assert restored.source == obs.source
    assert restored.ampcal == obs.ampcal


def test_ehtim_stokes_obsdata_is_converted_to_standard_circular_products():
    circular_obs = circular_dataset(
        channel_frequency_hz=np.array((230.0e9,)),
        channel_bandwidth_hz=np.array((2.0e9,)),
        spectral_window_id=np.array((0,)),
        visibilities=np.ones((2, 1, 4), dtype=complex),
        sigma_jy=np.full((2, 1, 4), 0.5),
        flags=np.zeros((2, 1, 4), dtype=bool),
    ).to_ehtim_obsdata()

    restored = VisibilityDataset.from_ehtim_obsdata(circular_obs.switch_polrep("stokes"))
    slots = restored.circular_product_slots()
    rows = np.arange(restored.row_count)[:, np.newaxis]

    assert np.allclose(restored.visibilities[rows, 0, slots], 1.0)


def test_ehtim_export_rejects_nonstandard_receptor_basis():
    original = circular_dataset(
        channel_frequency_hz=np.array((230.0e9,)),
        channel_bandwidth_hz=np.array((2.0e9,)),
        spectral_window_id=np.array((0,)),
        visibilities=np.ones((2, 1, 4), dtype=complex),
        sigma_jy=np.ones((2, 1, 4)),
        flags=np.zeros((2, 1, 4), dtype=bool),
    )
    custom_receptors = ReceptorTable(
        station_index=original.receptors.station_index,
        feed_id=original.receptors.feed_id,
        polarization_label=original.receptors.polarization_label,
        basis=("CUSTOM",) * original.receptors.count,
    )
    custom = replace(original, receptors=custom_receptors)

    with pytest.raises(ValueError, match="circular receptor basis"):
        custom.to_ehtim_obsdata()


def test_ehtim_adapter_rejects_unrepresentable_datasets():
    with pytest.raises(ValueError, match="exactly one spectral channel"):
        circular_dataset().to_ehtim_obsdata()

    mixed = mixed_feed_dataset()
    mixed = replace(
        mixed,
        channel_frequency_hz=mixed.channel_frequency_hz[:1],
        channel_bandwidth_hz=mixed.channel_bandwidth_hz[:1],
        spectral_window_id=mixed.spectral_window_id[:1],
        visibilities=mixed.visibilities[:, :1],
        sigma_jy=mixed.sigma_jy[:, :1],
        flags=mixed.flags[:, :1],
    )
    with pytest.raises(ValueError, match="required correlation products"):
        mixed.to_ehtim_obsdata()

    flagged = circular_dataset(
        channel_frequency_hz=np.array((230.0e9,)),
        channel_bandwidth_hz=np.array((2.0e9,)),
        spectral_window_id=np.array((0,)),
        visibilities=np.ones((2, 1, 4), dtype=complex),
        sigma_jy=np.ones((2, 1, 4)),
        flags=np.ones((2, 1, 4), dtype=bool),
    )
    with pytest.raises(ValueError, match="flagged"):
        flagged.to_ehtim_obsdata()


def test_ehtim_adapter_preserves_rows_across_mjd_boundaries():
    original = circular_dataset(
        channel_frequency_hz=np.array((230.0e9,)),
        channel_bandwidth_hz=np.array((2.0e9,)),
        spectral_window_id=np.array((0,)),
        visibilities=np.ones((2, 1, 4), dtype=complex),
        sigma_jy=np.ones((2, 1, 4)),
        flags=np.zeros((2, 1, 4), dtype=bool),
        time_mjd=np.array((60000.99, 60001.01)),
    )

    restored = VisibilityDataset.from_ehtim_obsdata(original.to_ehtim_obsdata())

    assert np.allclose(restored.time_mjd, original.time_mjd)
