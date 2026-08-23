"""Native AIPS UVFITS adapters for :class:`VisibilityDataset`.

The UVFITS random-groups format has one global polarization axis.  It can
therefore represent a uniformly circular or uniformly linear dataset, but not
arbitrary per-baseline mixed-feed correlation layouts.  Mixed data remains
lossless in ``VisibilityDataset`` and is rejected here unless callers
explicitly request the unsafe X/Y-as-R/L compatibility export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.constants import c as SPEED_OF_LIGHT
from astropy.io import fits
from astropy.time import Time

from ngehtsim.obs.visibility_dataset import (
    CIRCULAR_PRODUCT_LABELS,
    LINEAR_PRODUCT_LABELS,
    ReceptorTable,
    StationTable,
    VisibilityDataset,
    standard_products_for_rows,
)


_LIGHT_SPEED_M_S = SPEED_OF_LIGHT.to_value("m / s")
_POLARIZATION_PRODUCTS = {
    -1: "RR",
    -2: "LL",
    -3: "RL",
    -4: "LR",
    -5: "XX",
    -6: "YY",
    -7: "XY",
    -8: "YX",
}


class UvfitsError(ValueError):
    """Raised when a UVFITS file cannot be represented by ``VisibilityDataset``."""


def read_uvfits(path):
    """Read a standard random-groups UVFITS file without using ``ehtim``.

    Circular and linear datasets with a global STOKES axis are supported.  IF
    and frequency axes are flattened into ``VisibilityDataset``'s channel
    axis, preserving an integer spectral-window ID for each IF.

    Parameters
    ----------
    path : str or pathlib.Path
        AIPS random-groups UVFITS file.

    Returns
    -------
    VisibilityDataset
        Native dataset with one global circular or linear receptor layout.

    Raises
    ------
    UvfitsError
        If the file uses unsupported regular axes, polarization codes, scan
        metadata, or cannot supply required row metadata.

    Notes
    -----
    UVFITS weights are an external-format boundary detail. Positive UVFITS
    inverse variances are converted to native ``sigma_jy`` values; invalid or
    flagged values become flagged samples with ``NaN`` uncertainty.
    """

    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0]
        if not isinstance(primary, fits.GroupsHDU):
            raise UvfitsError("UVFITS input must use a random-groups primary HDU.")
        if primary.data is None:
            raise UvfitsError("UVFITS input contains no visibility groups.")

        axes = _regular_axes(primary.header)
        data, frequencies, channel_bandwidths, spectral_window_id, layout = _read_data(
            primary.data,
            primary.header,
            axes,
            hdul,
        )
        station_table, station_index = _read_station_table(hdul)
        time_mjd = _group_parameter(primary.data, "DATE") - 2400000.5
        integration_time_s = _group_parameter(primary.data, "INTTIM")
        if np.any(~np.isfinite(integration_time_s)) or np.any(integration_time_s <= 0.0):
            raise UvfitsError("UVFITS INTTIM values must be finite and positive.")

        baseline = _group_parameter(primary.data, "BASELINE")
        antenna1, antenna2 = _decode_baselines(baseline, station_index)
        if layout == CIRCULAR_PRODUCT_LABELS:
            receptor_labels, basis = ("R", "L"), "CIRCULAR"
        else:
            receptor_labels, basis = ("X", "Y"), "LINEAR"
        receptors = ReceptorTable.from_station_labels(
            len(station_table.names),
            receptor_labels,
            basis,
        )
        correlation_products, row_product_id = standard_products_for_rows(
            receptors,
            antenna1,
            antenna2,
            layout,
        )
        uvw_m = np.column_stack((
            _group_parameter(primary.data, "UU---SIN"),
            _group_parameter(primary.data, "VV---SIN"),
            _optional_group_parameter(primary.data, "WW---SIN", 0.0),
        )) * _LIGHT_SPEED_M_S
        tau1 = _optional_group_parameter(primary.data, "TAU1", 0.0)
        tau2 = _optional_group_parameter(primary.data, "TAU2", 0.0)

        scan_start_mjd, scan_stop_mjd = _read_scans(hdul, time_mjd)
        header = primary.header
        try:
            ra_hours = float(header["OBSRA"]) / 15.0
            dec_degrees = float(header["OBSDEC"])
        except KeyError as exc:
            raise UvfitsError("UVFITS input requires OBSRA and OBSDEC metadata.") from exc

        return VisibilityDataset(
            stations=station_table,
            receptors=receptors,
            correlation_products=correlation_products,
            time_mjd=time_mjd,
            integration_time_s=integration_time_s,
            antenna1=antenna1,
            antenna2=antenna2,
            uvw_m=uvw_m,
            tau1=tau1,
            tau2=tau2,
            channel_frequency_hz=frequencies,
            channel_bandwidth_hz=channel_bandwidths,
            spectral_window_id=spectral_window_id,
            row_product_id=row_product_id,
            visibilities=data["visibilities"],
            sigma_jy=data["sigma_jy"],
            flags=data["flags"],
            source=str(header.get("OBJECT", "UNKNOWN")).strip(),
            ra_hours=ra_hours,
            dec_degrees=dec_degrees,
            ampcal=True,
            phasecal=True,
            opacitycal=True,
            dcal=True,
            frcal=True,
            scan_start_mjd=scan_start_mjd,
            scan_stop_mjd=scan_stop_mjd,
        )


def write_uvfits(
    dataset,
    path,
    overwrite=False,
    force_circular_labels=False,
    array_name="NGEHTSIM",
):
    """Write a uniformly circular or linear ``VisibilityDataset`` as UVFITS.

    The output uses a standard global STOKES axis, a regular frequency axis,
    and optional IF spectral windows.  Rows are time-sorted so an AIPS NX scan
    table can describe native scan metadata correctly.

    Parameters
    ----------
    dataset : VisibilityDataset
        Dataset with a globally uniform standard circular or linear layout.
    path : str or pathlib.Path
        Destination UVFITS file.
    overwrite : bool, optional
        Replace an existing file when ``True``.
    force_circular_labels : bool, optional
        Unsafe compatibility mode for a two-feed mixed R/L and X/Y dataset.
        It writes a circular global STOKES axis and labels every X feed as R
        and every Y feed as L, without transforming any visibility,
        uncertainty, or flag values.  This exists solely for software that
        cannot ingest mixed-feed data and is disabled by default.
    array_name : str, optional
        AIPS array name written as the ``ARRNAM`` keyword in the ``AIPS AN``
        table. It must be non-empty ASCII text of at most eight characters.
        The default identifies a generic ngehtsim-generated array.

    Raises
    ------
    UvfitsError
        If the receptor mapping, channel frequencies, or scan metadata cannot
        be represented by standard UVFITS.

    Notes
    -----
    Native ``sigma_jy`` is converted to UVFITS inverse variance only while
    writing the random-groups payload. Mixed-receptor datasets should use
    FITS-EHT instead. When ``force_circular_labels=True`` is used, the output
    contains a HISTORY warning and is intentionally not physically labelled
    correctly for the linear-feed stations.
    """

    if not isinstance(dataset, VisibilityDataset):
        raise TypeError("dataset must be a VisibilityDataset instance.")
    if not dataset.row_count:
        raise UvfitsError("UVFITS output requires at least one visibility row.")
    if not isinstance(force_circular_labels, bool):
        raise TypeError("force_circular_labels must be a bool.")
    array_name = _aips_array_name(array_name)

    layout, product_slots = _uvfits_layout(
        dataset,
        force_circular_labels=force_circular_labels,
    )
    frequency_grid = _frequency_grid(dataset)
    row_order = np.lexsort((dataset.antenna2, dataset.antenna1, dataset.time_mjd))
    primary = _primary_hdu(
        dataset,
        layout,
        product_slots,
        frequency_grid,
        row_order,
        force_circular_labels=force_circular_labels,
    )
    antenna = _antenna_hdu(
        dataset,
        layout,
        frequency_grid["reference_frequency_hz"],
        len(frequency_grid["channel_indices"]),
        reference_mjd=float(np.floor(np.min(dataset.time_mjd))),
        array_name=array_name,
    )
    frequency = _frequency_hdu(frequency_grid)
    hdus = [primary, antenna, frequency]
    scan = _scan_hdu(dataset, row_order)
    if scan is not None:
        hdus.append(scan)
    fits.HDUList(hdus).writeto(Path(path), overwrite=overwrite)


def _read_data(group_data, header, axes, hdul):
    values, if_count, frequency_count, polarization_codes = _visibility_values(
        group_data,
        header,
        axes,
    )
    layout = _layout_from_stokes_codes(polarization_codes)
    channels, channel_bandwidths, spectral_window_id = _channel_metadata(
        header,
        axes,
        hdul,
        if_count,
        frequency_count,
    )

    row_count = values.shape[0]
    channel_count = if_count * frequency_count
    visibilities = np.zeros((row_count, channel_count, 4), dtype=complex)
    sigma_jy = np.full((row_count, channel_count, 4), np.nan, dtype=float)
    flags = np.ones((row_count, channel_count, 4), dtype=bool)
    real = values[..., 0].reshape(row_count, channel_count, len(polarization_codes))
    imaginary = values[..., 1].reshape(row_count, channel_count, len(polarization_codes))
    if values.shape[-1] == 3:
        raw_weights = values[..., 2].reshape(
            row_count,
            channel_count,
            len(polarization_codes),
        )
    else:
        raw_weights = np.ones_like(real)

    for source_index, product in enumerate(
        _POLARIZATION_PRODUCTS[int(code)] for code in polarization_codes
    ):
        destination_index = layout.index(product)
        valid = (
            np.isfinite(real[..., source_index])
            & np.isfinite(imaginary[..., source_index])
            & np.isfinite(raw_weights[..., source_index])
            & (raw_weights[..., source_index] > 0.0)
        )
        visibilities[..., destination_index] = np.where(
            valid,
            real[..., source_index] + 1.0j * imaginary[..., source_index],
            0.0,
        )
        sigma_values = np.full(valid.shape, np.nan, dtype=float)
        sigma_values[valid] = 1.0 / np.sqrt(raw_weights[..., source_index][valid])
        sigma_jy[..., destination_index] = sigma_values
        flags[..., destination_index] = ~valid

    return {
        "visibilities": visibilities,
        "sigma_jy": sigma_jy,
        "flags": flags,
    }, channels, channel_bandwidths, spectral_window_id, layout


def _regular_axes(header):
    axes = {}
    for number in range(2, int(header["NAXIS"]) + 1):
        name = str(header.get("CTYPE{0}".format(number), "")).strip().upper()
        if name:
            axes[name] = number
    for required in ("COMPLEX", "STOKES", "FREQ"):
        if required not in axes:
            raise UvfitsError("UVFITS input is missing the {0} axis.".format(required))
    return axes


def _visibility_values(group_data, header, axes):
    raw = np.asarray(group_data.data)
    source_axes = []
    destination_axes = []
    if "IF" in axes:
        source_axes.append(_numpy_axis(header, axes["IF"]))
        destination_axes.append(1)
    source_axes.extend((
        _numpy_axis(header, axes["FREQ"]),
        _numpy_axis(header, axes["STOKES"]),
        _numpy_axis(header, axes["COMPLEX"]),
    ))
    if "IF" in axes:
        destination_axes.extend((2, 3, 4))
    else:
        destination_axes.extend((1, 2, 3))
    values = np.moveaxis(raw, source_axes, destination_axes)
    expected_dimensions = 5 if "IF" in axes else 4
    if any(size != 1 for size in values.shape[expected_dimensions:]):
        raise UvfitsError("UVFITS input has unsupported non-singleton regular axes.")
    values = values.reshape(values.shape[:expected_dimensions])
    if "IF" not in axes:
        values = values[:, np.newaxis, ...]
    if values.shape[-1] not in (2, 3):
        raise UvfitsError("UVFITS COMPLEX axis must contain real, imaginary, and optional weight values.")

    stokes = _axis_values(header, axes["STOKES"])
    if not np.allclose(stokes, np.rint(stokes)):
        raise UvfitsError("UVFITS STOKES coordinates must be integral polarization codes.")
    polarization_codes = np.rint(stokes).astype(int)
    return values, values.shape[1], values.shape[2], polarization_codes


def _channel_metadata(header, axes, hdul, if_count, frequency_count):
    frequency_axis = _axis_values(header, axes["FREQ"])
    if len(frequency_axis) != frequency_count:
        raise UvfitsError("UVFITS frequency axis is inconsistent with visibility data.")
    frequency_step = _channel_step(frequency_axis, "UVFITS frequency axis")
    if "AIPS FQ" in hdul:
        fq_data = hdul["AIPS FQ"].data
        if len(fq_data) != 1:
            raise UvfitsError("UVFITS reader currently requires exactly one AIPS FQ row.")
        offsets = _table_vector(fq_data["IF FREQ"][0], if_count, "IF FREQ")
        widths = np.abs(_table_vector(fq_data["CH WIDTH"][0], if_count, "CH WIDTH"))
    else:
        offsets = np.zeros(if_count)
        widths = np.full(if_count, abs(frequency_step))
    if np.any(widths <= 0.0):
        widths = np.full(if_count, abs(frequency_step))
    if np.any(widths <= 0.0):
        raise UvfitsError("UVFITS channel widths must be positive.")

    channels = (offsets[:, np.newaxis] + frequency_axis[np.newaxis, :]).reshape(-1)
    channel_bandwidths = np.repeat(widths, frequency_count)
    spectral_window_id = np.repeat(np.arange(if_count, dtype=np.intp), frequency_count)
    return channels, channel_bandwidths, spectral_window_id


def _layout_from_stokes_codes(codes):
    if not len(codes) or any(int(code) not in _POLARIZATION_PRODUCTS for code in codes):
        raise UvfitsError("UVFITS STOKES axis must describe correlation products, not Stokes values.")
    products = {_POLARIZATION_PRODUCTS[int(code)] for code in codes}
    if products.issubset(set(CIRCULAR_PRODUCT_LABELS)):
        return CIRCULAR_PRODUCT_LABELS
    if products.issubset(set(LINEAR_PRODUCT_LABELS)):
        return LINEAR_PRODUCT_LABELS
    raise UvfitsError("UVFITS input mixes circular and linear correlation products.")


def _read_station_table(hdul):
    if "AIPS AN" not in hdul:
        raise UvfitsError("UVFITS input is missing the AIPS AN antenna table.")
    table = hdul["AIPS AN"].data
    required = ("ANNAME", "STABXYZ", "NOSTA")
    if any(name not in table.names for name in required):
        raise UvfitsError("AIPS AN table is missing required antenna metadata.")
    names = tuple(str(name).strip() for name in table["ANNAME"])
    numbers = np.asarray(table["NOSTA"], dtype=int)
    if len(set(numbers)) != len(numbers):
        raise UvfitsError("AIPS AN antenna numbers must be unique.")
    sefd = np.asarray(table["SEFD"], dtype=float) if "SEFD" in table.names else np.zeros(len(names))
    sefd = np.where(np.isfinite(sefd) & (sefd >= 0.0), sefd, 0.0)
    station_table = StationTable(
        names=names,
        position_itrs_m=np.asarray(table["STABXYZ"], dtype=float),
        sefd_r_jy=sefd,
        sefd_l_jy=sefd,
        leakage_r=np.zeros(len(names), dtype=complex),
        leakage_l=np.zeros(len(names), dtype=complex),
        feed_rotation_par=np.zeros(len(names)),
        feed_rotation_elev=np.zeros(len(names)),
        feed_rotation_offset_deg=np.zeros(len(names)),
    )
    return station_table, {number: index for index, number in enumerate(numbers)}


def _decode_baselines(values, station_index):
    codes = np.rint(np.asarray(values, dtype=float)).astype(int)
    antenna1_number = codes // 256
    antenna2_number = codes % 256
    if np.any(antenna1_number <= 0) or np.any(antenna2_number <= 0):
        raise UvfitsError("UVFITS extended baseline codes are not yet supported.")
    try:
        antenna1 = np.fromiter(
            (station_index[number] for number in antenna1_number),
            dtype=np.intp,
            count=len(codes),
        )
        antenna2 = np.fromiter(
            (station_index[number] for number in antenna2_number),
            dtype=np.intp,
            count=len(codes),
        )
    except KeyError as exc:
        raise UvfitsError("UVFITS BASELINE references an unknown AIPS AN antenna.") from exc
    return antenna1, antenna2


def _read_scans(hdul, time_mjd):
    if "AIPS NX" not in hdul:
        return None, None
    table = hdul["AIPS NX"].data
    required = ("TIME", "TIME INTERVAL")
    if any(name not in table.names for name in required):
        raise UvfitsError("AIPS NX table is missing TIME metadata.")
    reference_mjd = np.floor(np.min(time_mjd))
    center = np.asarray(table["TIME"], dtype=float)
    interval = np.asarray(table["TIME INTERVAL"], dtype=float)
    if np.any(~np.isfinite(center)) or np.any(~np.isfinite(interval)) or np.any(interval < 0.0):
        raise UvfitsError("AIPS NX scan metadata must be finite and non-negative.")
    return (
        reference_mjd + center - (0.5 * interval),
        reference_mjd + center + (0.5 * interval),
    )


def _group_parameter(group_data, name):
    try:
        values = np.asarray(group_data.par(name), dtype=float)
    except (KeyError, AttributeError) as exc:
        raise UvfitsError("UVFITS input is missing the {0} random parameter.".format(name)) from exc
    if np.any(~np.isfinite(values)):
        raise UvfitsError("UVFITS {0} random parameters must be finite.".format(name))
    return values


def _optional_group_parameter(group_data, name, default):
    names = {str(value).strip().upper() for value in group_data.parnames}
    if name not in names:
        return np.full(len(group_data), default, dtype=float)
    return _group_parameter(group_data, name)


def _axis_values(header, number):
    count = int(header["NAXIS{0}".format(number)])
    return (
        float(header["CRVAL{0}".format(number)])
        + (np.arange(count, dtype=float) + 1.0 - float(header["CRPIX{0}".format(number)]))
        * float(header["CDELT{0}".format(number)])
    )


def _numpy_axis(header, number):
    return int(header["NAXIS"]) - number + 1


def _channel_step(values, name):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return 0.0
    step = values[1] - values[0]
    if step == 0.0 or not np.allclose(np.diff(values), step):
        raise UvfitsError("{0} must be regularly spaced.".format(name))
    return step


def _table_vector(values, count, name):
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != count:
        raise UvfitsError("AIPS FQ {0} length does not match the IF axis.".format(name))
    return values


def _uvfits_layout(dataset, force_circular_labels=False):
    if force_circular_labels:
        return CIRCULAR_PRODUCT_LABELS, _forced_circular_product_slots(dataset)
    try:
        return CIRCULAR_PRODUCT_LABELS, dataset.circular_product_slots()
    except ValueError:
        pass
    try:
        return LINEAR_PRODUCT_LABELS, dataset.linear_product_slots()
    except ValueError as exc:
        raise UvfitsError(
            "UVFITS output requires exactly the same circular RR, LL, RL, LR or "
            "linear XX, YY, XY, YX products for every row."
        ) from exc


def _forced_circular_product_slots(dataset):
    """Map complete R/L or X/Y station products onto an unsafe R/L layout."""

    labels = np.asarray(dataset.receptors.polarization_label, dtype=object)
    circular_label = {}
    for station_index, station in enumerate(dataset.stations.names):
        receptors = np.flatnonzero(dataset.receptors.station_index == station_index)
        station_labels = tuple(labels[receptors])
        if len(receptors) != 2 or set(station_labels) not in ({"R", "L"}, {"X", "Y"}):
            raise UvfitsError(
                "force_circular_labels requires every station to have exactly "
                "R/L or X/Y receptor labels; {0!r} has {1}.".format(
                    station,
                    ", ".join(station_labels),
                )
            )
        for receptor in receptors:
            label = str(labels[receptor])
            circular_label[receptor] = {"R": "R", "L": "L", "X": "R", "Y": "L"}[label]

    product_labels = np.asarray(
        tuple(
            circular_label[first] + circular_label[second]
            for first, second in zip(
                dataset.correlation_products.receptor1_id,
                dataset.correlation_products.receptor2_id,
            )
        ),
        dtype=object,
    )
    slots = np.empty((dataset.row_count, len(CIRCULAR_PRODUCT_LABELS)), dtype=np.intp)
    expected = set(CIRCULAR_PRODUCT_LABELS)
    for row, product_ids in enumerate(dataset.row_product_id):
        populated_slots = np.flatnonzero(product_ids >= 0)
        row_labels = tuple(product_labels[product_ids[populated_slots]])
        if len(row_labels) != len(CIRCULAR_PRODUCT_LABELS) or set(row_labels) != expected:
            raise UvfitsError(
                "force_circular_labels requires complete RR, LL, RL, LR-equivalent "
                "products after X-to-R and Y-to-L relabelling."
            )
        for destination, label in enumerate(CIRCULAR_PRODUCT_LABELS):
            slots[row, destination] = populated_slots[row_labels.index(label)]
    return slots


def _frequency_grid(dataset):
    ids = np.unique(dataset.spectral_window_id)
    if not len(ids):
        raise UvfitsError("UVFITS output requires at least one spectral window.")
    channel_indices = []
    offsets = None
    steps = []
    widths = []
    reference_frequency_hz = None
    for window_index, window_id in enumerate(ids):
        indices = np.flatnonzero(dataset.spectral_window_id == window_id)
        indices = indices[np.argsort(dataset.channel_frequency_hz[indices])]
        frequencies = dataset.channel_frequency_hz[indices]
        bandwidths = dataset.channel_bandwidth_hz[indices]
        if not np.allclose(bandwidths, bandwidths[0]):
            raise UvfitsError("Each UVFITS spectral window must have a uniform channel bandwidth.")
        step = _channel_step(frequencies, "VisibilityDataset spectral-window frequencies")
        if len(frequencies) == 1:
            step = bandwidths[0]
        window_offsets = frequencies - frequencies[0]
        if offsets is None:
            offsets = window_offsets
            reference_frequency_hz = frequencies[0]
        elif len(window_offsets) != len(offsets) or not np.allclose(window_offsets, offsets):
            raise UvfitsError("UVFITS output requires matching channel grids in every spectral window.")
        channel_indices.append(indices)
        steps.append(step)
        widths.append(bandwidths[0])
    if not np.allclose(steps, steps[0]):
        raise UvfitsError("UVFITS output requires a common frequency spacing in every spectral window.")
    if_offsets = np.array([
        dataset.channel_frequency_hz[indices[0]] - reference_frequency_hz
        for indices in channel_indices
    ])
    return {
        "channel_indices": tuple(channel_indices),
        "reference_frequency_hz": float(reference_frequency_hz),
        "frequency_step_hz": float(steps[0]),
        "if_offsets_hz": if_offsets,
        "channel_widths_hz": np.asarray(widths, dtype=float),
    }


def _primary_hdu(dataset, layout, product_slots, frequency_grid, row_order,
                 force_circular_labels=False):
    channel_indices = frequency_grid["channel_indices"]
    if_count = len(channel_indices)
    frequency_count = len(channel_indices[0])
    values = np.zeros(
        (dataset.row_count, 1, 1, if_count, frequency_count, 4, 3),
        dtype=np.float32,
    )
    ordered_slots = product_slots[row_order]
    for if_index, channel_index in enumerate(channel_indices):
        row_index = row_order[:, np.newaxis, np.newaxis]
        channel_axis = channel_index[np.newaxis, :, np.newaxis]
        product_axis = ordered_slots[:, np.newaxis, :]
        visibility = dataset.visibilities[row_index, channel_axis, product_axis]
        sigma_jy = dataset.sigma_jy[row_index, channel_axis, product_axis]
        valid = ~dataset.flags[row_index, channel_axis, product_axis]
        values[:, 0, 0, if_index, :, :, 0] = np.where(valid, visibility.real, 0.0)
        values[:, 0, 0, if_index, :, :, 1] = np.where(valid, visibility.imag, 0.0)
        values[:, 0, 0, if_index, :, :, 2] = np.where(valid, 1.0 / np.square(sigma_jy), -1.0)

    reference_mjd = float(np.floor(np.min(dataset.time_mjd)))
    date_jd = reference_mjd + 2400000.5
    baseline = _encode_baselines(dataset.antenna1[row_order], dataset.antenna2[row_order])
    group_data = fits.GroupData(
        values,
        parnames=["UU---SIN", "VV---SIN", "WW---SIN", "BASELINE", "DATE", "DATE", "INTTIM", "TAU1", "TAU2"],
        pardata=(
            dataset.uvw_m[row_order, 0] / _LIGHT_SPEED_M_S,
            dataset.uvw_m[row_order, 1] / _LIGHT_SPEED_M_S,
            dataset.uvw_m[row_order, 2] / _LIGHT_SPEED_M_S,
            baseline,
            np.full(dataset.row_count, date_jd),
            dataset.time_mjd[row_order] - reference_mjd,
            dataset.integration_time_s[row_order],
            dataset.tau1[row_order],
            dataset.tau2[row_order],
        ),
        bitpix=-32,
    )
    primary = fits.GroupsHDU(group_data)
    header = primary.header
    _set_common_header(header, dataset, layout, frequency_grid, reference_mjd)
    if force_circular_labels:
        header["HISTORY"] = (
            "WARNING: X/Y feeds were relabelled as R/L without a basis conversion."
        )
    return primary


def _set_common_header(header, dataset, layout, frequency_grid, reference_mjd):
    """Populate standard primary-HDU metadata for a UVFITS export.

    Random-groups scaling keywords are written explicitly even where their
    FITS defaults are unity and zero.  Several established AIPS-family
    readers require those cards to decode random parameters, in particular
    the two-component ``DATE`` timestamp, reliably.
    """

    stokes_start = -1.0 if layout == CIRCULAR_PRODUCT_LABELS else -5.0
    header["OBSRA"] = dataset.ra_hours * 15.0
    header["OBSDEC"] = dataset.dec_degrees
    header["OBJECT"] = dataset.source
    header["MJD"] = reference_mjd
    header["DATE-OBS"] = Time(reference_mjd, format="mjd", scale="utc").iso[:10]
    header["BUNIT"] = "JY"
    header["EQUINOX"] = 2000.0
    header["TIMESYS"] = "UTC"
    header["BSCALE"] = 1.0
    header["BZERO"] = 0.0
    for parameter_index in range(1, 10):
        header[f"PSCAL{parameter_index}"] = 1.0
        header[f"PZERO{parameter_index}"] = 0.0
    header["CTYPE2"] = "COMPLEX"
    header["CRVAL2"] = 1.0
    header["CDELT2"] = 1.0
    header["CRPIX2"] = 1.0
    header["CTYPE3"] = "STOKES"
    header["CRVAL3"] = stokes_start
    header["CDELT3"] = -1.0
    header["CRPIX3"] = 1.0
    header["CTYPE4"] = "FREQ"
    header["CRVAL4"] = frequency_grid["reference_frequency_hz"]
    header["CDELT4"] = frequency_grid["frequency_step_hz"]
    header["CRPIX4"] = 1.0
    header["CTYPE5"] = "IF"
    header["CRVAL5"] = 1.0
    header["CDELT5"] = 1.0
    header["CRPIX5"] = 1.0
    header["CTYPE6"] = "RA"
    header["CRVAL6"] = dataset.ra_hours * 15.0
    header["CDELT6"] = 1.0
    header["CRPIX6"] = 1.0
    header["CTYPE7"] = "DEC"
    header["CRVAL7"] = dataset.dec_degrees
    header["CDELT7"] = 1.0
    header["CRPIX7"] = 1.0
    header["HISTORY"] = "AIPS SORT ORDER='TB'"


def _antenna_hdu(
    dataset,
    layout,
    reference_frequency_hz,
    if_count,
    *,
    reference_mjd,
    array_name,
):
    """Build a complete AIPS AN table for the native UVFITS writer.

    Global VLBI station positions are expressed in ITRF relative to a
    geocentric array origin. The mandatory orbit and polarization-calibration
    columns have zero repeat counts because ngehtsim does not write those
    optional per-antenna payloads.
    """

    try:
        station_name_lengths = [len(name.encode("ascii")) for name in dataset.stations.names]
    except UnicodeEncodeError as exc:
        raise UvfitsError("UVFITS AIPS AN output requires ASCII station names.") from exc
    if any(length > 8 for length in station_name_lengths):
        raise UvfitsError(
            "UVFITS AIPS AN output supports station names up to eight ASCII characters."
        )
    names = np.asarray(dataset.stations.names, dtype="S8")
    count = len(names)
    first_feed, second_feed = ("R", "L") if layout == CIRCULAR_PRODUCT_LABELS else ("X", "Y")
    columns = fits.ColDefs((
        fits.Column(name="ANNAME", format="8A", array=names),
        fits.Column(name="STABXYZ", format="3D", unit="METERS", array=dataset.stations.position_itrs_m),
        fits.Column(name="ORBPARM", format="0D", array=np.empty((count, 0), dtype=float)),
        fits.Column(name="NOSTA", format="1J", array=np.arange(1, count + 1)),
        fits.Column(name="MNTSTA", format="1J", array=np.zeros(count, dtype=np.int32)),
        fits.Column(name="STAXOF", format="1E", unit="METERS", array=np.zeros(count)),
        fits.Column(name="POLTYA", format="1A", array=np.full(count, first_feed, dtype="S1")),
        fits.Column(name="POLAA", format="1E", unit="DEGREES", array=np.zeros(count)),
        fits.Column(name="POLCALA", format="0E", array=np.empty((count, 0), dtype=np.float32)),
        fits.Column(name="POLTYB", format="1A", array=np.full(count, second_feed, dtype="S1")),
        fits.Column(name="POLAB", format="1E", unit="DEGREES", array=np.full(count, 90.0)),
        fits.Column(name="POLCALB", format="0E", array=np.empty((count, 0), dtype=np.float32)),
        fits.Column(name="SEFD", format="1D", array=dataset.stations.sefd_r_jy),
    ))
    antenna = fits.BinTableHDU.from_columns(columns, name="AIPS AN")
    header = antenna.header
    reference_date = _aips_reference_date(reference_mjd)
    header["EXTVER"] = 1
    header["ARRAYX"] = 0.0
    header["ARRAYY"] = 0.0
    header["ARRAYZ"] = 0.0
    header["GSTIA0"] = reference_date.sidereal_time("mean", "greenwich").degree
    header["DEGPDY"] = 360.98564736629
    header["FREQ"] = reference_frequency_hz
    header["RDATE"] = reference_date.to_value("iso", subfmt="date")
    header["POLARX"] = 0.0
    header["POLARY"] = 0.0
    header["UT1UTC"] = float(reference_date.delta_ut1_utc)
    header["DATUTC"] = 0.0
    header["NO_IF"] = if_count
    header["TIMESYS"] = "UTC"
    header["ARRNAM"] = array_name
    header["XYZHAND"] = "RIGHT"
    header["FRAME"] = "ITRF"
    header["NUMORB"] = 0
    header["NOPCAL"] = 0
    header["POLTYPE"] = "VLBI"
    header["FREQID"] = 1
    return antenna


def _aips_array_name(array_name):
    """Validate an AIPS AN ``ARRNAM`` value and return its stripped form."""

    if not isinstance(array_name, str):
        raise TypeError("array_name must be a str.")
    array_name = array_name.strip()
    if not array_name:
        raise UvfitsError("UVFITS AIPS AN ARRNAM must be non-empty.")
    try:
        encoded = array_name.encode("ascii")
    except UnicodeEncodeError as exc:
        raise UvfitsError("UVFITS AIPS AN ARRNAM must contain only ASCII characters.") from exc
    if len(encoded) > 8:
        raise UvfitsError("UVFITS AIPS AN ARRNAM supports at most eight ASCII characters.")
    return array_name


def _aips_reference_date(reference_mjd):
    """Return the UTC midnight reference date for AIPS AN metadata."""

    timestamp = Time(reference_mjd, format="mjd", scale="utc")
    return Time(timestamp.to_value("iso", subfmt="date"), format="iso", scale="utc")


def _frequency_hdu(frequency_grid):
    count = len(frequency_grid["if_offsets_hz"])
    columns = fits.ColDefs((
        fits.Column(name="FRQSEL", format="1J", array=np.array([1], dtype=np.int32)),
        fits.Column(name="IF FREQ", format="{0}D".format(count), array=np.array([frequency_grid["if_offsets_hz"]])),
        fits.Column(name="CH WIDTH", format="{0}E".format(count), array=np.array([frequency_grid["channel_widths_hz"]], dtype=np.float32)),
        fits.Column(name="TOTAL BANDWIDTH", format="{0}E".format(count), array=np.array([frequency_grid["channel_widths_hz"] * len(frequency_grid["channel_indices"][0])], dtype=np.float32)),
        fits.Column(name="SIDEBAND", format="{0}J".format(count), array=np.ones((1, count), dtype=np.int32)),
    ))
    frequency = fits.BinTableHDU.from_columns(columns, name="AIPS FQ")
    frequency.header["EXTVER"] = 1
    frequency.header["NO_IF"] = count
    return frequency


def _scan_hdu(dataset, row_order):
    """Build an AIPS NX table for scans represented in the output rows.

    Row-filtered datasets retain their original schedule metadata. In
    particular, native fringe selection can remove every visibility in an
    otherwise valid scan. UVFITS ``START VIS`` and ``END VIS`` indices cannot
    describe such an empty scan, so it is omitted from the exported NX table.
    """

    if dataset.scan_start_mjd is None:
        return None
    ordered_time = dataset.time_mjd[row_order]
    reference_mjd = float(np.floor(np.min(ordered_time)))
    starts = dataset.scan_start_mjd
    stops = dataset.scan_stop_mjd
    retained_starts = []
    retained_stops = []
    start_vis = []
    stop_vis = []
    tolerance = 1.0e-10
    for start, stop in zip(starts, stops):
        rows = np.flatnonzero(
            (ordered_time >= start - tolerance) & (ordered_time <= stop + tolerance)
        )
        if not len(rows):
            continue
        retained_starts.append(start)
        retained_stops.append(stop)
        start_vis.append(rows[0] + 1)
        stop_vis.append(rows[-1] + 1)
    if not retained_starts:
        return None
    retained_starts = np.asarray(retained_starts, dtype=float)
    retained_stops = np.asarray(retained_stops, dtype=float)
    center = 0.5 * (retained_starts + retained_stops) - reference_mjd
    interval = retained_stops - retained_starts
    columns = fits.ColDefs((
        fits.Column(name="TIME", format="1D", unit="DAYS", array=center),
        fits.Column(name="TIME INTERVAL", format="1E", unit="DAYS", array=interval),
        fits.Column(name="SOURCE ID", format="1J", array=np.ones(len(retained_starts), dtype=np.int32)),
        fits.Column(name="SUBARRAY", format="1J", array=np.ones(len(retained_starts), dtype=np.int32)),
        fits.Column(name="FREQ ID", format="1J", array=np.ones(len(retained_starts), dtype=np.int32)),
        fits.Column(name="START VIS", format="1J", array=np.asarray(start_vis, dtype=np.int32)),
        fits.Column(name="END VIS", format="1J", array=np.asarray(stop_vis, dtype=np.int32)),
    ))
    scan = fits.BinTableHDU.from_columns(columns, name="AIPS NX")
    scan.header["EXTVER"] = 1
    return scan


def _encode_baselines(antenna1, antenna2):
    antenna1 = np.asarray(antenna1, dtype=int) + 1
    antenna2 = np.asarray(antenna2, dtype=int) + 1
    if np.any(antenna1 > 255) or np.any(antenna2 > 255):
        raise UvfitsError("UVFITS extended baseline codes are not yet supported.")
    return (256 * antenna1) + antenna2
