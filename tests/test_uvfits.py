"""Tests for native multi-channel UVFITS interchange."""

import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits

from ngehtsim.obs.uvfits import UvfitsError, read_uvfits, write_uvfits
from ngehtsim.obs.visibility_dataset import (
    CIRCULAR_PRODUCT_LABELS,
    CorrelationProductTable,
    LINEAR_PRODUCT_LABELS,
    ReceptorTable,
    StationTable,
    VisibilityDataset,
    standard_products_for_rows,
)


def _stations():
    return StationTable(
        names=("ALMA", "APEX", "LMT"),
        position_itrs_m=np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))),
        sefd_r_jy=np.array((100.0, 110.0, 120.0)),
        sefd_l_jy=np.array((101.0, 111.0, 121.0)),
        leakage_r=np.zeros(3, dtype=complex),
        leakage_l=np.zeros(3, dtype=complex),
        feed_rotation_par=np.zeros(3),
        feed_rotation_elev=np.zeros(3),
        feed_rotation_offset_deg=np.zeros(3),
    )


def _dataset(kind="circular"):
    labels, basis = (
        (CIRCULAR_PRODUCT_LABELS, "CIRCULAR")
        if kind == "circular"
        else (LINEAR_PRODUCT_LABELS, "LINEAR")
    )
    antenna1 = np.array((0, 1))
    antenna2 = np.array((1, 2))
    receptor_labels = ("R", "L") if kind == "circular" else ("X", "Y")
    receptors = ReceptorTable.from_station_labels(3, receptor_labels, basis)
    products, row_product_id = standard_products_for_rows(
        receptors,
        antenna1,
        antenna2,
        labels,
    )
    visibilities = np.arange(32, dtype=float).reshape(2, 4, 4)
    visibilities = visibilities + 1.0j * (100.0 + visibilities)
    sigma_jy = np.full((2, 4, 4), 0.5)
    flags = np.zeros((2, 4, 4), dtype=bool)
    flags[1, 3, 2] = True
    sigma_jy[1, 3, 2] = np.nan
    visibilities[1, 3, 2] = 0.0
    return VisibilityDataset(
        stations=_stations(),
        receptors=receptors,
        correlation_products=products,
        time_mjd=np.array((60000.0, 60000.01)),
        integration_time_s=np.array((10.0, 20.0)),
        antenna1=antenna1,
        antenna2=antenna2,
        uvw_m=np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))),
        tau1=np.array((0.1, 0.2)),
        tau2=np.array((0.3, 0.4)),
        channel_frequency_hz=np.array((230.0e9, 230.001e9, 231.0e9, 231.001e9)),
        channel_bandwidth_hz=np.full(4, 1.0e6),
        spectral_window_id=np.array((0, 0, 1, 1)),
        row_product_id=row_product_id,
        visibilities=visibilities,
        sigma_jy=sigma_jy,
        flags=flags,
        source="M87",
        ra_hours=12.5,
        dec_degrees=12.4,
        scan_start_mjd=np.array((59999.999, 60000.009)),
        scan_stop_mjd=np.array((60000.001, 60000.011)),
    )


@pytest.mark.parametrize("kind", ("circular", "linear"))
def test_native_uvfits_round_trip_preserves_multichannel_data(tmp_path, kind):
    original = _dataset(kind)
    path = tmp_path / "native.uvfits"

    original.to_uvfits(path, array_name="SYNTH")
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0]
        assert primary.header["NAXIS4"] == 2
        assert primary.header["NAXIS5"] == 2
        assert primary.header["CRVAL3"] == (-1.0 if kind == "circular" else -5.0)
        assert primary.header["EQUINOX"] == 2000.0
        assert primary.header["TIMESYS"] == "UTC"
        assert primary.header["BSCALE"] == 1.0
        assert primary.header["BZERO"] == 0.0
        assert [primary.header[f"PSCAL{index}"] for index in range(1, 10)] == [1.0] * 9
        assert [primary.header[f"PZERO{index}"] for index in range(1, 10)] == [0.0] * 9
        assert np.allclose(
            primary.data.par("DATE"),
            original.time_mjd + 2400000.5,
        )
        assert [hdu.name for hdu in hdul] == ["PRIMARY", "AIPS AN", "AIPS FQ", "AIPS NX"]
        expected_feeds = ("R", "L") if kind == "circular" else ("X", "Y")
        assert tuple(hdul["AIPS AN"].data["POLTYA"][:1]) == (expected_feeds[0],)
        assert tuple(hdul["AIPS AN"].data["POLTYB"][:1]) == (expected_feeds[1],)
        antenna = hdul["AIPS AN"]
        mandatory = {
            "EXTNAME", "EXTVER", "ARRAYX", "ARRAYY", "ARRAYZ", "GSTIA0", "DEGPDY",
            "FREQ", "RDATE", "POLARX", "POLARY", "UT1UTC", "DATUTC", "TIMESYS",
            "ARRNAM", "XYZHAND", "FRAME", "NUMORB", "NO_IF", "NOPCAL", "POLTYPE",
            "FREQID",
        }
        assert mandatory <= set(antenna.header)
        assert antenna.header["ARRNAM"] == "SYNTH"
        assert [antenna.header[name] for name in ("ARRAYX", "ARRAYY", "ARRAYZ")] == [0.0] * 3
        assert antenna.header["FRAME"] == "ITRF"
        assert antenna.header["XYZHAND"] == "RIGHT"
        assert antenna.header["NUMORB"] == 0
        assert antenna.header["NOPCAL"] == 0
        assert antenna.header["FREQID"] == 1
        assert antenna.data["ORBPARM"].shape == (3, 0)
        assert antenna.data["POLCALA"].shape == (3, 0)
        assert antenna.data["POLCALB"].shape == (3, 0)
        assert np.allclose(hdul["AIPS FQ"].data["TOTAL BANDWIDTH"][0], [2.0e6, 2.0e6])

    restored = VisibilityDataset.from_uvfits(path)

    assert np.allclose(restored.time_mjd, original.time_mjd)
    assert np.array_equal(restored.antenna1, original.antenna1)
    assert np.array_equal(restored.antenna2, original.antenna2)
    assert np.allclose(restored.uvw_m, original.uvw_m)
    assert np.allclose(restored.channel_frequency_hz, original.channel_frequency_hz)
    assert np.allclose(restored.channel_bandwidth_hz, original.channel_bandwidth_hz)
    assert np.array_equal(restored.spectral_window_id, original.spectral_window_id)
    assert np.array_equal(restored.flags, original.flags)
    assert np.allclose(restored.sigma_jy, original.sigma_jy, equal_nan=True)
    assert np.allclose(restored.visibilities, original.visibilities)
    assert np.allclose(restored.scan_start_mjd, original.scan_start_mjd)
    assert np.allclose(restored.scan_stop_mjd, original.scan_stop_mjd)
    slots = restored.circular_product_slots() if kind == "circular" else restored.linear_product_slots()
    assert slots.shape == (2, 4)


@pytest.mark.parametrize(
    ("array_name", "exception", "message"),
    (
        (None, TypeError, "must be a str"),
        ("", UvfitsError, "must be non-empty"),
        ("NINECHARS", UvfitsError, "at most eight ASCII characters"),
        ("ng" + chr(0x00C9) + "HTsim", UvfitsError, "only ASCII characters"),
    ),
)
def test_native_uvfits_writer_validates_aips_array_name(
    tmp_path,
    array_name,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        _dataset().to_uvfits(tmp_path / "native.uvfits", array_name=array_name)


def test_native_uvfits_reader_loads_checked_in_eht_2017_file_without_ehtim():
    path = "docs/source/EHT2017_tutorial/SR1_M87_2017_096_lo_hops_netcal_StokesI.uvfits"

    dataset = read_uvfits(path)

    assert dataset.row_count == 8645
    assert dataset.channel_count == 1
    assert dataset.circular_product_slots().shape == (8645, 4)
    assert dataset.stations.names == ("AA", "AP", "AZ", "JC", "LM", "PV", "SM", "SR")


def test_native_uvfits_writer_output_is_accepted_by_ehtim(tmp_path):
    ehtim = pytest.importorskip("ehtim")
    path = tmp_path / "native.uvfits"

    _dataset().to_uvfits(path)
    observation = ehtim.obsdata.load_uvfits(
        str(path),
        channel=0,
        IF=0,
        polrep="circ",
    )

    assert len(observation.data) == 2
    assert observation.rf == pytest.approx(230.0e9)


@pytest.mark.parametrize(
    ("row_mask", "scan_index"),
    (
        (np.array((True, False)), 0),
        (np.array((False, True)), 1),
    ),
)
def test_native_uvfits_writer_omits_empty_scans_after_row_filter(
    tmp_path,
    row_mask,
    scan_index,
):
    """NX metadata must follow rows retained for an export boundary subset."""

    original = _dataset()
    filtered = original.select_rows(row_mask)
    path = tmp_path / "filtered.uvfits"

    filtered.to_uvfits(path)

    with fits.open(path, memmap=False) as hdul:
        nx = hdul["AIPS NX"].data
        assert len(nx) == 1
        assert nx["START VIS"].tolist() == [1]
        assert nx["END VIS"].tolist() == [1]
        assert nx["TIME"][0] == pytest.approx(
            0.5 * (original.scan_start_mjd[scan_index] + original.scan_stop_mjd[scan_index])
            - np.floor(filtered.time_mjd.min())
        )

    restored = read_uvfits(path)
    assert np.allclose(restored.scan_start_mjd, original.scan_start_mjd[scan_index:scan_index + 1])
    assert np.allclose(restored.scan_stop_mjd, original.scan_stop_mjd[scan_index:scan_index + 1])


def test_native_uvfits_module_does_not_import_ehtim():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ngehtsim.obs.uvfits; assert 'ehtim' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _mixed_dataset():
    """Create a complete two-feed circular/linear mixed-layout dataset."""

    original = _dataset()
    receptors = ReceptorTable(
        station_index=np.array((0, 0, 1, 1, 2, 2)),
        feed_id=("R", "L", "X", "Y", "R", "L"),
        polarization_label=("R", "L", "X", "Y", "R", "L"),
        basis=("CIRCULAR", "CIRCULAR", "LINEAR", "LINEAR", "CIRCULAR", "CIRCULAR"),
    )
    products = CorrelationProductTable(
        receptor1_id=np.array((0, 0, 1, 1, 2, 2, 3, 3)),
        receptor2_id=np.array((2, 3, 2, 3, 4, 5, 4, 5)),
    )
    return VisibilityDataset(
        stations=original.stations,
        receptors=receptors,
        correlation_products=products,
        time_mjd=original.time_mjd,
        integration_time_s=original.integration_time_s,
        antenna1=original.antenna1,
        antenna2=original.antenna2,
        uvw_m=original.uvw_m,
        tau1=original.tau1,
        tau2=original.tau2,
        channel_frequency_hz=original.channel_frequency_hz,
        channel_bandwidth_hz=original.channel_bandwidth_hz,
        spectral_window_id=original.spectral_window_id,
        row_product_id=np.array(((0, 1, 2, 3), (4, 5, 6, 7))),
        visibilities=original.visibilities,
        sigma_jy=original.sigma_jy,
        flags=original.flags,
        source=original.source,
        ra_hours=original.ra_hours,
        dec_degrees=original.dec_degrees,
    )

def test_native_uvfits_writer_rejects_mixed_receptor_products(tmp_path):
    mixed = _mixed_dataset()

    with pytest.raises(UvfitsError, match="requires exactly"):
        write_uvfits(mixed, tmp_path / "mixed.uvfits")


def test_native_uvfits_force_circular_labels_relabels_mixed_two_feed_layout(tmp_path):
    """Unsafe X/Y-to-R/L export must preserve payloads while changing labels only."""

    mixed = _mixed_dataset()
    path = tmp_path / "mixed-as-circular.uvfits"
    write_uvfits(mixed, path, force_circular_labels=True)

    with fits.open(path, memmap=False) as hdul:
        assert hdul[0].header["CRVAL3"] == -1.0
        assert tuple(hdul["AIPS AN"].data["POLTYA"]) == ("R", "R", "R")
        assert tuple(hdul["AIPS AN"].data["POLTYB"]) == ("L", "L", "L")
        assert any("without a basis conversion" in entry for entry in hdul[0].header["HISTORY"])

    restored = read_uvfits(path)
    forced_slots = np.array(((0, 3, 1, 2), (0, 3, 1, 2)))
    row = np.arange(mixed.row_count)[:, np.newaxis, np.newaxis]
    channel = np.arange(mixed.channel_count)[np.newaxis, :, np.newaxis]
    slot = forced_slots[:, np.newaxis, :]
    assert np.allclose(restored.visibilities, mixed.visibilities[row, channel, slot])
    assert np.allclose(restored.sigma_jy, mixed.sigma_jy[row, channel, slot], equal_nan=True)
    assert np.array_equal(restored.flags, mixed.flags[row, channel, slot])
