"""Tests for native multi-channel UVFITS interchange."""

import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits

from ngehtsim.obs.uvfits import UvfitsError, read_uvfits, write_uvfits
from ngehtsim.obs.visibility_dataset import (
    CIRCULAR_CORRELATIONS,
    LINEAR_CIRCULAR_CORRELATIONS,
    LINEAR_CORRELATIONS,
    StationTable,
    VisibilityDataset,
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


def _dataset(layout=CIRCULAR_CORRELATIONS):
    visibilities = np.arange(32, dtype=float).reshape(2, 4, 4)
    visibilities = visibilities + 1.0j * (100.0 + visibilities)
    weights = np.full((2, 4, 4), 4.0)
    flags = np.zeros((2, 4, 4), dtype=bool)
    flags[1, 3, 2] = True
    weights[1, 3, 2] = 0.0
    visibilities[1, 3, 2] = 0.0
    return VisibilityDataset(
        stations=_stations(),
        time_mjd=np.array((60000.0, 60000.01)),
        integration_time_s=np.array((10.0, 20.0)),
        antenna1=np.array((0, 1)),
        antenna2=np.array((1, 2)),
        uvw_m=np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))),
        tau1=np.array((0.1, 0.2)),
        tau2=np.array((0.3, 0.4)),
        channel_frequency_hz=np.array((230.0e9, 230.001e9, 231.0e9, 231.001e9)),
        channel_bandwidth_hz=np.full(4, 1.0e6),
        spectral_window_id=np.array((0, 0, 1, 1)),
        correlation_layouts=(layout,),
        row_layout_id=np.zeros(2, dtype=np.intp),
        visibilities=visibilities,
        weights=weights,
        flags=flags,
        source="M87",
        ra_hours=12.5,
        dec_degrees=12.4,
        scan_start_mjd=np.array((59999.999, 60000.009)),
        scan_stop_mjd=np.array((60000.001, 60000.011)),
    )


@pytest.mark.parametrize("layout", (CIRCULAR_CORRELATIONS, LINEAR_CORRELATIONS))
def test_native_uvfits_round_trip_preserves_multichannel_data(tmp_path, layout):
    original = _dataset(layout)
    path = tmp_path / "native.uvfits"

    original.to_uvfits(path)
    with fits.open(path, memmap=False) as hdul:
        assert hdul[0].header["NAXIS4"] == 2
        assert hdul[0].header["NAXIS5"] == 2
        assert hdul[0].header["CRVAL3"] == (-1.0 if layout == CIRCULAR_CORRELATIONS else -5.0)
        assert [hdu.name for hdu in hdul] == ["PRIMARY", "AIPS AN", "AIPS FQ", "AIPS NX"]
        expected_feeds = ("R", "L") if layout == CIRCULAR_CORRELATIONS else ("X", "Y")
        assert tuple(hdul["AIPS AN"].data["POLTYA"][:1]) == (expected_feeds[0],)
        assert tuple(hdul["AIPS AN"].data["POLTYB"][:1]) == (expected_feeds[1],)
        assert np.allclose(hdul["AIPS FQ"].data["TOTAL BANDWIDTH"][0], [2.0e6, 2.0e6])

    restored = VisibilityDataset.from_uvfits(path)

    assert restored.correlation_layouts == (layout,)
    assert np.allclose(restored.time_mjd, original.time_mjd)
    assert np.array_equal(restored.antenna1, original.antenna1)
    assert np.array_equal(restored.antenna2, original.antenna2)
    assert np.allclose(restored.uvw_m, original.uvw_m)
    assert np.allclose(restored.channel_frequency_hz, original.channel_frequency_hz)
    assert np.allclose(restored.channel_bandwidth_hz, original.channel_bandwidth_hz)
    assert np.array_equal(restored.spectral_window_id, original.spectral_window_id)
    assert np.array_equal(restored.flags, original.flags)
    assert np.allclose(restored.weights, original.weights)
    assert np.allclose(restored.visibilities, original.visibilities)
    assert np.allclose(restored.scan_start_mjd, original.scan_start_mjd)
    assert np.allclose(restored.scan_stop_mjd, original.scan_stop_mjd)


def test_native_uvfits_reader_loads_the_checked_in_eht_2017_file_without_ehtim():
    path = "docs/source/EHT2017_tutorial/SR1_M87_2017_096_lo_hops_netcal_StokesI.uvfits"

    dataset = read_uvfits(path)

    assert dataset.row_count == 8645
    assert dataset.channel_count == 1
    assert dataset.correlation_layouts == (CIRCULAR_CORRELATIONS,)
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


def test_native_uvfits_writer_rejects_per_row_mixed_layouts(tmp_path):
    original = _dataset()
    mixed = VisibilityDataset(
        stations=original.stations,
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
        correlation_layouts=(CIRCULAR_CORRELATIONS, LINEAR_CIRCULAR_CORRELATIONS),
        row_layout_id=np.array((0, 1), dtype=np.intp),
        visibilities=original.visibilities,
        weights=original.weights,
        flags=original.flags,
        source=original.source,
        ra_hours=original.ra_hours,
        dec_degrees=original.dec_degrees,
    )

    with pytest.raises(UvfitsError, match="per-row mixed"):
        write_uvfits(mixed, tmp_path / "mixed.uvfits")
