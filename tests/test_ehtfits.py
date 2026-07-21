"""Tests for lossless FITS-EHT native visibility interchange."""

import numpy as np
import pytest
from astropy.io import fits

from ngehtsim.obs.ehtfits import EhtfitsError, FORMAT_NAME, FORMAT_VERSION, read_ehtfits
from ngehtsim.obs.visibility_dataset import (
    CorrelationProductTable,
    ReceptorTable,
    StationTable,
    VisibilityDataset,
)


def _stations():
    return StationTable(
        names=("STA1", "STA2", "STA3"),
        position_itrs_m=np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))),
        sefd_r_jy=np.array((100.0, 110.0, 120.0)),
        sefd_l_jy=np.array((101.0, 111.0, 121.0)),
        leakage_r=np.array((0.01j, 0.02j, 0.03j)),
        leakage_l=np.array((0.01, 0.02, 0.03)),
        feed_rotation_par=np.array((1.0, 0.0, 1.0)),
        feed_rotation_elev=np.zeros(3),
        feed_rotation_offset_deg=np.array((0.0, 15.0, -30.0)),
    )


def _mixed_feed_dataset():
    """Two-by-two and two-by-three rows, with an unknown flagged sigma."""

    receptors = ReceptorTable(
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
    visibilities = np.arange(24, dtype=float).reshape(2, 2, 6).astype(complex)
    visibilities += 1.0j * (100.0 + visibilities)
    sigma_jy = np.full((2, 2, 6), 0.5)
    flags = np.zeros((2, 2, 6), dtype=bool)
    visibilities[0, :, 4:] = np.nan + 1.0j * np.nan
    sigma_jy[0, :, 4:] = np.nan
    flags[0, :, 4:] = True
    visibilities[1, 1, 5] = np.nan + 1.0j * np.nan
    sigma_jy[1, 1, 5] = np.nan
    flags[1, 1, 5] = True
    return VisibilityDataset(
        stations=_stations(),
        receptors=receptors,
        correlation_products=products,
        time_mjd=np.array((60000.0, 60000.01)),
        integration_time_s=np.array((10.0, 20.0)),
        antenna1=np.array((0, 1)),
        antenna2=np.array((1, 2)),
        uvw_m=np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))),
        tau1=np.array((0.1, 0.2)),
        tau2=np.array((0.3, 0.4)),
        channel_frequency_hz=np.array((230.0e9, 230.1e9)),
        channel_bandwidth_hz=np.array((2.0e9, 2.0e9)),
        spectral_window_id=np.array((0, 0)),
        row_product_id=row_product_id,
        visibilities=visibilities,
        sigma_jy=sigma_jy,
        flags=flags,
        source="M87",
        ra_hours=12.5,
        dec_degrees=12.4,
        ampcal=False,
        phasecal=False,
        opacitycal=True,
        dcal=False,
        frcal=False,
        scan_start_mjd=np.array((59999.999,)),
        scan_stop_mjd=np.array((60000.011,)),
    )


def test_fits_eht_round_trip_preserves_mixed_receptor_data(tmp_path):
    original = _mixed_feed_dataset()
    path = tmp_path / "mixed.ehtfits"

    original.to_ehtfits(path)
    restored = VisibilityDataset.from_ehtfits(path)

    assert restored.stations.names == original.stations.names
    assert np.array_equal(restored.receptors.station_index, original.receptors.station_index)
    assert restored.receptors.feed_id == original.receptors.feed_id
    assert restored.receptors.polarization_label == original.receptors.polarization_label
    assert restored.receptors.basis == original.receptors.basis
    assert np.array_equal(restored.correlation_products.receptor1_id, original.correlation_products.receptor1_id)
    assert np.array_equal(restored.correlation_products.receptor2_id, original.correlation_products.receptor2_id)
    assert np.array_equal(restored.row_product_id, original.row_product_id)
    assert np.allclose(restored.visibilities, original.visibilities, equal_nan=True)
    assert np.allclose(restored.sigma_jy, original.sigma_jy, equal_nan=True)
    assert np.array_equal(restored.flags, original.flags)
    assert np.allclose(restored.channel_frequency_hz, original.channel_frequency_hz)
    assert np.allclose(restored.scan_start_mjd, original.scan_start_mjd)
    assert restored.ampcal is False
    assert restored.phasecal is False
    assert restored.opacitycal is True
    assert restored.dcal is False
    assert restored.frcal is False


def test_fits_eht_uses_explicit_receptor_product_payloads_and_sigmas(tmp_path):
    path = tmp_path / "mixed.ehtfits"
    _mixed_feed_dataset().to_ehtfits(path)

    with fits.open(path, memmap=False) as hdul:
        assert hdul[0].header["EHTFMT"] == FORMAT_NAME
        assert hdul[0].header["EHTVER"] == FORMAT_VERSION
        assert [hdu.name for hdu in hdul] == [
            "PRIMARY",
            "ARRAY_GEOMETRY",
            "FREQUENCY",
            "SOURCE",
            "EHT_RECEPTORS",
            "EHT_CORRELATION_PRODUCTS",
            "EHT_CHANNELS",
            "EHT_SCANS",
            "UV_DATA",
        ]
        data = hdul["UV_DATA"].data
        assert "WEIGHT" not in data.names
        assert hdul["UV_DATA"].columns["SIGMA"].unit == "JY"
        assert np.array_equal(data["PRODUCT_ID"][0], np.array((1, 2, 3, 4)))
        assert np.array_equal(data["PRODUCT_ID"][1], np.array((5, 6, 7, 8, 9, 10)))
        assert data["NPRODUCT"].tolist() == [4, 6]
        assert len(data["VISIBILITY"][0]) == 16
        assert len(data["SIGMA"][1]) == 12


def test_fits_eht_reader_rejects_non_fits_eht_input(tmp_path):
    path = tmp_path / "not-ehtfits.fits"
    fits.PrimaryHDU().writeto(path)

    with pytest.raises(EhtfitsError, match="not a FITS-EHT"):
        read_ehtfits(path)
