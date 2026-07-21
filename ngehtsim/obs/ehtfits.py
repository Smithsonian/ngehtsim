"""FITS-EHT interchange for native :class:`VisibilityDataset` objects.

FITS-EHT is a project-owned FITS binary-table convention. It deliberately does
not claim FITS-IDI compatibility: it replaces FITS-IDI's global polarization
axis and weight component with explicit station-receptor product metadata and
per-sample ``SIGMA`` values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from ngehtsim.obs.visibility_dataset import (
    CorrelationProductTable,
    ReceptorTable,
    StationTable,
    VisibilityDataset,
)


FORMAT_NAME = "FITS-EHT"
FORMAT_VERSION = "0.1.0"


class EhtfitsError(ValueError):
    """Raised when a FITS-EHT file cannot be represented natively."""


def write_ehtfits(dataset, path, overwrite=False):
    """Write a native dataset as a lossless FITS-EHT file.

    Visibility payloads use variable-length FITS binary-table arrays. Their
    order is channel-major then product-major, with every product ID resolving
    to an ordered pair of station-local receptors.
    """

    if not isinstance(dataset, VisibilityDataset):
        raise TypeError("dataset must be a VisibilityDataset instance.")
    if not dataset.row_count:
        raise EhtfitsError("FITS-EHT output requires at least one visibility row.")

    hdus = [
        _primary_hdu(dataset),
        _array_geometry_hdu(dataset),
        _frequency_hdu(dataset),
        _source_hdu(dataset),
        _receptors_hdu(dataset),
        _correlation_products_hdu(dataset),
        _channels_hdu(dataset),
    ]
    if dataset.scan_start_mjd is not None:
        hdus.append(_scans_hdu(dataset))
    hdus.append(_uv_data_hdu(dataset))
    fits.HDUList(hdus).writeto(Path(path), overwrite=overwrite)


def read_ehtfits(path):
    """Read a FITS-EHT file into a native :class:`VisibilityDataset`."""

    with fits.open(Path(path), memmap=False) as hdul:
        header = hdul[0].header
        if str(header.get("EHTFMT", "")).strip() != FORMAT_NAME:
            raise EhtfitsError("Input is not a FITS-EHT file.")
        version = str(header.get("EHTVER", "")).strip()
        if not version.startswith("0.1."):
            raise EhtfitsError("Unsupported FITS-EHT version: {0!r}.".format(version))

        stations = _read_station_table(_required_table(hdul, "ARRAY_GEOMETRY"))
        receptors = _read_receptor_table(_required_table(hdul, "EHT_RECEPTORS"))
        products = _read_product_table(_required_table(hdul, "EHT_CORRELATION_PRODUCTS"))
        channel_frequency_hz, channel_bandwidth_hz, spectral_window_id = _read_channels(
            _required_table(hdul, "EHT_CHANNELS")
        )
        scans = _read_scans(hdul)
        uv_data = _required_table(hdul, "UV_DATA")
        decoded = _read_uv_data(uv_data, len(channel_frequency_hz))

        return VisibilityDataset(
            stations=stations,
            receptors=receptors,
            correlation_products=products,
            time_mjd=decoded["time_mjd"],
            integration_time_s=decoded["integration_time_s"],
            antenna1=decoded["antenna1"],
            antenna2=decoded["antenna2"],
            uvw_m=decoded["uvw_m"],
            tau1=decoded["tau1"],
            tau2=decoded["tau2"],
            channel_frequency_hz=channel_frequency_hz,
            channel_bandwidth_hz=channel_bandwidth_hz,
            spectral_window_id=spectral_window_id,
            row_product_id=decoded["row_product_id"],
            visibilities=decoded["visibilities"],
            sigma_jy=decoded["sigma_jy"],
            flags=decoded["flags"],
            source=str(header.get("OBJECT", "UNKNOWN")).strip(),
            ra_hours=float(header["OBSRA"]),
            dec_degrees=float(header["OBSDEC"]),
            ampcal=bool(header.get("AMPCAL", True)),
            phasecal=bool(header.get("PHSCAL", True)),
            opacitycal=bool(header.get("OPACAL", True)),
            dcal=bool(header.get("DCAL", True)),
            frcal=bool(header.get("FRCAL", True)),
            scan_start_mjd=scans[0],
            scan_stop_mjd=scans[1],
        )


def _primary_hdu(dataset):
    primary = fits.PrimaryHDU()
    header = primary.header
    header["EHTFMT"] = (FORMAT_NAME, "Project FITS-EHT interchange convention")
    header["EHTVER"] = (FORMAT_VERSION, "FITS-EHT schema version")
    header["OBJECT"] = dataset.source
    header["OBSRA"] = (dataset.ra_hours, "Source right ascension (hours)")
    header["OBSDEC"] = (dataset.dec_degrees, "Source declination (degrees)")
    header["TIMESYS"] = "UTC"
    header["BUNIT"] = "JY"
    header["AMPCAL"] = bool(dataset.ampcal)
    header["PHSCAL"] = bool(dataset.phasecal)
    header["OPACAL"] = bool(dataset.opacitycal)
    header["DCAL"] = bool(dataset.dcal)
    header["FRCAL"] = bool(dataset.frcal)
    header["HISTORY"] = "FITS-EHT is not FITS-IDI compatible."
    header["HISTORY"] = "Visibility uncertainties are stored in UV_DATA SIGMA (Jy)."
    return primary


def _array_geometry_hdu(dataset):
    stations = dataset.stations
    count = len(stations.names)
    columns = fits.ColDefs((
        fits.Column(name="ANTENNA_ID", format="1J", array=np.arange(1, count + 1, dtype=np.int32)),
        fits.Column(name="ANNAME", format="64A", array=np.asarray(stations.names, dtype="S64")),
        fits.Column(name="STABXYZ", format="3D", unit="M", array=stations.position_itrs_m),
        fits.Column(name="SEFDR", format="1D", unit="JY", array=stations.sefd_r_jy),
        fits.Column(name="SEFDL", format="1D", unit="JY", array=stations.sefd_l_jy),
        fits.Column(name="DR_REAL", format="1D", array=stations.leakage_r.real),
        fits.Column(name="DR_IMAG", format="1D", array=stations.leakage_r.imag),
        fits.Column(name="DL_REAL", format="1D", array=stations.leakage_l.real),
        fits.Column(name="DL_IMAG", format="1D", array=stations.leakage_l.imag),
        fits.Column(name="FR_PAR", format="1D", array=stations.feed_rotation_par),
        fits.Column(name="FR_ELEV", format="1D", array=stations.feed_rotation_elev),
        fits.Column(name="FR_OFFSET", format="1D", unit="DEG", array=stations.feed_rotation_offset_deg),
    ))
    return fits.BinTableHDU.from_columns(columns, name="ARRAY_GEOMETRY")


def _frequency_hdu(dataset):
    columns = fits.ColDefs((
        fits.Column(name="FREQ_ID", format="1J", array=np.array((1,), dtype=np.int32)),
        fits.Column(name="CHANNEL_COUNT", format="1J", array=np.array((dataset.channel_count,), dtype=np.int32)),
        fits.Column(name="REFERENCE_HZ", format="1D", unit="HZ", array=np.array((dataset.channel_frequency_hz[0],))),
    ))
    return fits.BinTableHDU.from_columns(columns, name="FREQUENCY")


def _source_hdu(dataset):
    columns = fits.ColDefs((
        fits.Column(name="SOURCE_ID", format="1J", array=np.array((1,), dtype=np.int32)),
        fits.Column(name="SOURCE", format="64A", array=np.array((dataset.source,), dtype="S64")),
        fits.Column(name="RA_HOURS", format="1D", unit="HOUR", array=np.array((dataset.ra_hours,))),
        fits.Column(name="DEC_DEG", format="1D", unit="DEG", array=np.array((dataset.dec_degrees,))),
    ))
    return fits.BinTableHDU.from_columns(columns, name="SOURCE")


def _receptors_hdu(dataset):
    receptors = dataset.receptors
    count = receptors.count
    columns = fits.ColDefs((
        fits.Column(name="RECEPTOR_ID", format="1J", array=np.arange(1, count + 1, dtype=np.int32)),
        fits.Column(name="ANTENNA_ID", format="1J", array=receptors.station_index.astype(np.int32) + 1),
        fits.Column(name="FEED_ID", format="32A", array=np.asarray(receptors.feed_id, dtype="S32")),
        fits.Column(name="POL_LABEL", format="32A", array=np.asarray(receptors.polarization_label, dtype="S32")),
        fits.Column(name="BASIS", format="32A", array=np.asarray(receptors.basis, dtype="S32")),
    ))
    return fits.BinTableHDU.from_columns(columns, name="EHT_RECEPTORS")


def _correlation_products_hdu(dataset):
    products = dataset.correlation_products
    count = products.count
    columns = fits.ColDefs((
        fits.Column(name="PRODUCT_ID", format="1J", array=np.arange(1, count + 1, dtype=np.int32)),
        fits.Column(name="RECEPTOR1_ID", format="1J", array=products.receptor1_id.astype(np.int32) + 1),
        fits.Column(name="RECEPTOR2_ID", format="1J", array=products.receptor2_id.astype(np.int32) + 1),
    ))
    return fits.BinTableHDU.from_columns(columns, name="EHT_CORRELATION_PRODUCTS")


def _channels_hdu(dataset):
    count = dataset.channel_count
    columns = fits.ColDefs((
        fits.Column(name="CHANNEL_ID", format="1J", array=np.arange(1, count + 1, dtype=np.int32)),
        fits.Column(name="FREQUENCY_HZ", format="1D", unit="HZ", array=dataset.channel_frequency_hz),
        fits.Column(name="BANDWIDTH_HZ", format="1D", unit="HZ", array=dataset.channel_bandwidth_hz),
        fits.Column(name="SPECTRAL_WINDOW_ID", format="1J", array=dataset.spectral_window_id.astype(np.int32)),
    ))
    return fits.BinTableHDU.from_columns(columns, name="EHT_CHANNELS")


def _scans_hdu(dataset):
    columns = fits.ColDefs((
        fits.Column(name="START_MJD", format="1D", unit="D", array=dataset.scan_start_mjd),
        fits.Column(name="STOP_MJD", format="1D", unit="D", array=dataset.scan_stop_mjd),
    ))
    return fits.BinTableHDU.from_columns(columns, name="EHT_SCANS")


def _uv_data_hdu(dataset):
    row_count = dataset.row_count
    product_ids = np.empty(row_count, dtype=object)
    visibility = np.empty(row_count, dtype=object)
    sigma_jy = np.empty(row_count, dtype=object)
    flags = np.empty(row_count, dtype=object)
    product_count = np.empty(row_count, dtype=np.int32)

    for row, row_products in enumerate(dataset.row_product_id):
        slots = np.flatnonzero(row_products >= 0)
        product_count[row] = len(slots)
        product_ids[row] = row_products[slots].astype(np.int32) + 1
        row_visibility = dataset.visibilities[row][:, slots]
        row_sigma = dataset.sigma_jy[row][:, slots]
        row_flags = dataset.flags[row][:, slots]
        visibility[row] = np.column_stack((row_visibility.real.ravel(), row_visibility.imag.ravel())).ravel()
        sigma_jy[row] = row_sigma.ravel()
        flags[row] = row_flags.astype(np.uint8).ravel()

    columns = fits.ColDefs((
        fits.Column(name="TIME_MJD", format="1D", unit="D", array=dataset.time_mjd),
        fits.Column(name="INTEGRATION_S", format="1D", unit="S", array=dataset.integration_time_s),
        fits.Column(name="ANTENNA1", format="1J", array=dataset.antenna1.astype(np.int32) + 1),
        fits.Column(name="ANTENNA2", format="1J", array=dataset.antenna2.astype(np.int32) + 1),
        fits.Column(name="UU_M", format="1D", unit="M", array=dataset.uvw_m[:, 0]),
        fits.Column(name="VV_M", format="1D", unit="M", array=dataset.uvw_m[:, 1]),
        fits.Column(name="WW_M", format="1D", unit="M", array=dataset.uvw_m[:, 2]),
        fits.Column(name="TAU1", format="1D", array=dataset.tau1),
        fits.Column(name="TAU2", format="1D", array=dataset.tau2),
        fits.Column(name="NCHANNEL", format="1J", array=np.full(row_count, dataset.channel_count, dtype=np.int32)),
        fits.Column(name="NPRODUCT", format="1J", array=product_count),
        fits.Column(name="PRODUCT_ID", format="QJ()", array=product_ids),
        fits.Column(name="VISIBILITY", format="QD()", unit="JY", array=visibility),
        fits.Column(name="SIGMA", format="QD()", unit="JY", array=sigma_jy),
        fits.Column(name="FLAG", format="QB()", array=flags),
    ))
    return fits.BinTableHDU.from_columns(columns, name="UV_DATA")


def _required_table(hdul, name):
    if name not in hdul or not isinstance(hdul[name], fits.BinTableHDU):
        raise EhtfitsError("FITS-EHT input is missing the {0} table.".format(name))
    return hdul[name].data


def _text(value):
    if isinstance(value, bytes):
        return value.decode("ascii").strip()
    return str(value).strip()


def _required_columns(table, names, extension):
    if table.names is None or any(name not in table.names for name in names):
        raise EhtfitsError("{0} is missing required columns.".format(extension))


def _read_station_table(table):
    _required_columns(
        table,
        ("ANTENNA_ID", "ANNAME", "STABXYZ", "SEFDR", "SEFDL", "DR_REAL", "DR_IMAG", "DL_REAL", "DL_IMAG", "FR_PAR", "FR_ELEV", "FR_OFFSET"),
        "ARRAY_GEOMETRY",
    )
    identifiers = np.asarray(table["ANTENNA_ID"], dtype=np.intp)
    if not np.array_equal(identifiers, np.arange(1, len(identifiers) + 1)):
        raise EhtfitsError("ARRAY_GEOMETRY antenna IDs must be contiguous and one-based.")
    return StationTable(
        names=tuple(_text(value) for value in table["ANNAME"]),
        position_itrs_m=np.asarray(table["STABXYZ"], dtype=float),
        sefd_r_jy=np.asarray(table["SEFDR"], dtype=float),
        sefd_l_jy=np.asarray(table["SEFDL"], dtype=float),
        leakage_r=np.asarray(table["DR_REAL"], dtype=float) + 1.0j * np.asarray(table["DR_IMAG"], dtype=float),
        leakage_l=np.asarray(table["DL_REAL"], dtype=float) + 1.0j * np.asarray(table["DL_IMAG"], dtype=float),
        feed_rotation_par=np.asarray(table["FR_PAR"], dtype=float),
        feed_rotation_elev=np.asarray(table["FR_ELEV"], dtype=float),
        feed_rotation_offset_deg=np.asarray(table["FR_OFFSET"], dtype=float),
    )


def _read_receptor_table(table):
    _required_columns(table, ("RECEPTOR_ID", "ANTENNA_ID", "FEED_ID", "POL_LABEL", "BASIS"), "EHT_RECEPTORS")
    identifiers = np.asarray(table["RECEPTOR_ID"], dtype=np.intp)
    if not np.array_equal(identifiers, np.arange(1, len(identifiers) + 1)):
        raise EhtfitsError("EHT_RECEPTORS IDs must be contiguous and one-based.")
    return ReceptorTable(
        station_index=np.asarray(table["ANTENNA_ID"], dtype=np.intp) - 1,
        feed_id=tuple(_text(value) for value in table["FEED_ID"]),
        polarization_label=tuple(_text(value) for value in table["POL_LABEL"]),
        basis=tuple(_text(value) for value in table["BASIS"]),
    )


def _read_product_table(table):
    _required_columns(table, ("PRODUCT_ID", "RECEPTOR1_ID", "RECEPTOR2_ID"), "EHT_CORRELATION_PRODUCTS")
    identifiers = np.asarray(table["PRODUCT_ID"], dtype=np.intp)
    if not np.array_equal(identifiers, np.arange(1, len(identifiers) + 1)):
        raise EhtfitsError("EHT_CORRELATION_PRODUCTS IDs must be contiguous and one-based.")
    return CorrelationProductTable(
        receptor1_id=np.asarray(table["RECEPTOR1_ID"], dtype=np.intp) - 1,
        receptor2_id=np.asarray(table["RECEPTOR2_ID"], dtype=np.intp) - 1,
    )


def _read_channels(table):
    _required_columns(table, ("CHANNEL_ID", "FREQUENCY_HZ", "BANDWIDTH_HZ", "SPECTRAL_WINDOW_ID"), "EHT_CHANNELS")
    identifiers = np.asarray(table["CHANNEL_ID"], dtype=np.intp)
    if not np.array_equal(identifiers, np.arange(1, len(identifiers) + 1)):
        raise EhtfitsError("EHT_CHANNELS IDs must be contiguous and one-based.")
    return (
        np.asarray(table["FREQUENCY_HZ"], dtype=float),
        np.asarray(table["BANDWIDTH_HZ"], dtype=float),
        np.asarray(table["SPECTRAL_WINDOW_ID"], dtype=np.intp),
    )


def _read_scans(hdul):
    if "EHT_SCANS" not in hdul:
        return None, None
    table = hdul["EHT_SCANS"].data
    _required_columns(table, ("START_MJD", "STOP_MJD"), "EHT_SCANS")
    return np.asarray(table["START_MJD"], dtype=float), np.asarray(table["STOP_MJD"], dtype=float)


def _read_uv_data(table, channel_count):
    required = (
        "TIME_MJD", "INTEGRATION_S", "ANTENNA1", "ANTENNA2", "UU_M", "VV_M", "WW_M", "TAU1", "TAU2",
        "NCHANNEL", "NPRODUCT", "PRODUCT_ID", "VISIBILITY", "SIGMA", "FLAG",
    )
    _required_columns(table, required, "UV_DATA")
    row_count = len(table)
    nproduct = np.asarray(table["NPRODUCT"], dtype=np.intp)
    if np.any(nproduct <= 0):
        raise EhtfitsError("UV_DATA NPRODUCT values must be positive.")
    if not np.all(np.asarray(table["NCHANNEL"], dtype=np.intp) == channel_count):
        raise EhtfitsError("UV_DATA rows must use the file's complete EHT_CHANNELS table.")
    slot_count = int(np.max(nproduct))
    row_product_id = np.full((row_count, slot_count), -1, dtype=np.intp)
    visibilities = np.full((row_count, channel_count, slot_count), np.nan + 1.0j * np.nan, dtype=complex)
    sigma_jy = np.full((row_count, channel_count, slot_count), np.nan, dtype=float)
    flags = np.ones((row_count, channel_count, slot_count), dtype=bool)

    for row in range(row_count):
        count = int(nproduct[row])
        product_ids = np.asarray(table["PRODUCT_ID"][row], dtype=np.intp)
        visibility = np.asarray(table["VISIBILITY"][row], dtype=float)
        sigma = np.asarray(table["SIGMA"][row], dtype=float)
        flag = np.asarray(table["FLAG"][row], dtype=np.uint8)
        sample_count = channel_count * count
        if len(product_ids) != count:
            raise EhtfitsError("UV_DATA PRODUCT_ID length does not match NPRODUCT.")
        if len(visibility) != 2 * sample_count or len(sigma) != sample_count or len(flag) != sample_count:
            raise EhtfitsError("UV_DATA variable-length payload has an inconsistent size.")
        row_product_id[row, :count] = product_ids - 1
        pairs = visibility.reshape(channel_count, count, 2)
        visibilities[row, :, :count] = pairs[..., 0] + 1.0j * pairs[..., 1]
        sigma_jy[row, :, :count] = sigma.reshape(channel_count, count)
        flags[row, :, :count] = flag.reshape(channel_count, count).astype(bool)

    return {
        "time_mjd": np.asarray(table["TIME_MJD"], dtype=float),
        "integration_time_s": np.asarray(table["INTEGRATION_S"], dtype=float),
        "antenna1": np.asarray(table["ANTENNA1"], dtype=np.intp) - 1,
        "antenna2": np.asarray(table["ANTENNA2"], dtype=np.intp) - 1,
        "uvw_m": np.column_stack((
            np.asarray(table["UU_M"], dtype=float),
            np.asarray(table["VV_M"], dtype=float),
            np.asarray(table["WW_M"], dtype=float),
        )),
        "tau1": np.asarray(table["TAU1"], dtype=float),
        "tau2": np.asarray(table["TAU2"], dtype=float),
        "row_product_id": row_product_id,
        "visibilities": visibilities,
        "sigma_jy": sigma_jy,
        "flags": flags,
    }
