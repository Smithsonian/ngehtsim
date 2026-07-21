###################################################
# cache key construction

from dataclasses import dataclass

import numpy as np
from astropy.constants import c as SPEED_OF_LIGHT
from astropy.time import Time
import ehtim as eh

from ngehtsim.obs.visibility_dataset import (
    CIRCULAR_PRODUCT_LABELS,
    ReceptorTable,
    StationTable,
    VisibilityDataset,
    standard_products_for_rows,
)


GEOMETRY_CACHE_FIELDS = (
    "sites",
    "ra",
    "dec",
    "rf",
    "bandwidth_hz",
    "t_int",
    "t_rest",
    "t_start",
    "t_stop",
    "mjd",
)


@dataclass(frozen=True)
class GroundGeometry:
    """Ground-station baseline rows independent of ``ehtim.Obsdata``.

    Parameters
    ----------
    time_hours : numpy.ndarray, shape (row,)
        UTC-hour offsets from the observation's reference MJD.
    station1_indices, station2_indices : numpy.ndarray, shape (row,)
        Ordered zero-based station indices for each baseline-time row.
    uvw_m : numpy.ndarray, shape (row, 3)
        Projected baseline coordinates in metres.
    u, v : numpy.ndarray, shape (row,)
        Projected coordinates in wavelengths at the requested observing
        frequency. They are retained for the ehtim boundary adapter.
    """

    time_hours: np.ndarray
    station1_indices: np.ndarray
    station2_indices: np.ndarray
    uvw_m: np.ndarray
    u: np.ndarray
    v: np.ndarray


def _readonly_float_array(values):
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError("Station geometry values must be one-dimensional.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class StationGeometry:
    """Per-row ground-station elevation and parallactic angles in radians.

    Parameters
    ----------
    elevation1_rad, elevation2_rad : array_like, shape (row,)
        Elevation of the first and second stations in radians.
    parallactic_angle1_rad, parallactic_angle2_rad : array_like, shape (row,)
        Parallactic angles of the first and second stations in radians.

    Notes
    -----
    Arrays are copied and stored read-only. The ground-only kernel returns
    ``None`` instead of this class when a station uses the spacecraft
    placeholder coordinate convention.
    """

    elevation1_rad: np.ndarray
    elevation2_rad: np.ndarray
    parallactic_angle1_rad: np.ndarray
    parallactic_angle2_rad: np.ndarray

    def __post_init__(self):
        elevation1_rad = _readonly_float_array(self.elevation1_rad)
        elevation2_rad = _readonly_float_array(self.elevation2_rad)
        parallactic_angle1_rad = _readonly_float_array(self.parallactic_angle1_rad)
        parallactic_angle2_rad = _readonly_float_array(self.parallactic_angle2_rad)
        shape = elevation1_rad.shape
        if any(values.shape != shape for values in (
            elevation2_rad,
            parallactic_angle1_rad,
            parallactic_angle2_rad,
        )):
            raise ValueError("Station geometry arrays must have matching shapes.")
        if not all(np.all(np.isfinite(values)) for values in (
            elevation1_rad,
            elevation2_rad,
            parallactic_angle1_rad,
            parallactic_angle2_rad,
        )):
            raise ValueError("Station geometry arrays must contain finite values.")

        object.__setattr__(self, "elevation1_rad", elevation1_rad)
        object.__setattr__(self, "elevation2_rad", elevation2_rad)
        object.__setattr__(self, "parallactic_angle1_rad", parallactic_angle1_rad)
        object.__setattr__(self, "parallactic_angle2_rad", parallactic_angle2_rad)


def _cache_value(value):
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, list):
        return tuple(_cache_value(item) for item in value)

    if isinstance(value, tuple):
        return tuple(_cache_value(item) for item in value)

    if isinstance(value, dict):
        return tuple(sorted((key, _cache_value(item)) for key, item in value.items()))

    return value


def geometry_cache_key(context):
    return tuple((field, _cache_value(context[field])) for field in GEOMETRY_CACHE_FIELDS)

###################################################
# empty observation construction


def needs_new_empty_observation(obs_empty, cached_key, context):
    return (obs_empty is None) or (cached_key != geometry_cache_key(context))


def _has_space_station(array):
    coordinates = np.column_stack((array.tarr["x"], array.tarr["y"], array.tarr["z"]))
    return np.any(np.all(coordinates == 0.0, axis=1))


def _observation_times(context):
    t_start = float(context["t_start"])
    t_stop = float(context["t_stop"])
    if t_stop < t_start:
        t_stop += 24.0
    return np.arange(t_start, t_stop, float(context["t_rest"]) / 3600.0)


def station_geometry_from_rows(position_itrs_m, time_mjd, antenna1, antenna2,
                               ra_hours, dec_degrees):
    """Calculate ground-station angles from native row and station arrays.

    Returns ``None`` when a row cannot be represented by the ground-only
    geometry kernel, including arrays containing spacecraft placeholders.

    Parameters
    ----------
    position_itrs_m : array_like, shape (station, 3)
        Station ITRS Cartesian coordinates in metres.
    time_mjd : array_like, shape (row,)
        UTC MJD timestamps.
    antenna1, antenna2 : array_like, shape (row,)
        Ordered zero-based station indices.
    ra_hours, dec_degrees : float
        Source right ascension in hours and declination in degrees.

    Returns
    -------
    StationGeometry or None
        Per-row angle arrays, or ``None`` when any station is a spacecraft
        placeholder unsupported by the native ground-only kernel.
    """

    coordinates = np.asarray(position_itrs_m, dtype=float)
    time_mjd = np.asarray(time_mjd, dtype=float)
    antenna1 = np.asarray(antenna1, dtype=np.intp)
    antenna2 = np.asarray(antenna2, dtype=np.intp)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("position_itrs_m must have shape (station, 3).")
    if time_mjd.ndim != 1 or antenna1.shape != time_mjd.shape or antenna2.shape != time_mjd.shape:
        raise ValueError("Station geometry row arrays must have matching one-dimensional shapes.")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(time_mjd)):
        raise ValueError("Station geometry inputs must be finite.")
    if np.any(antenna1 < 0) or np.any(antenna1 >= len(coordinates)):
        raise ValueError("antenna1 contains an out-of-range station index.")
    if np.any(antenna2 < 0) or np.any(antenna2 >= len(coordinates)):
        raise ValueError("antenna2 contains an out-of-range station index.")
    if np.any(np.all(coordinates == 0.0, axis=1)):
        return None

    times_sidereal = Time(
        time_mjd,
        format="mjd",
        scale="utc",
    ).sidereal_time("mean", "greenwich").hour
    ra_rad = float(ra_hours) * (np.pi / 12.0)
    dec_rad = np.deg2rad(float(dec_degrees))
    hour_angle_rotation = np.mod(
        (times_sidereal - float(ra_hours)) * (np.pi / 12.0),
        2.0 * np.pi,
    )
    source_vector = np.array((np.cos(dec_rad), 0.0, np.sin(dec_rad)))

    def station_angles(antenna):
        station_coordinates = coordinates[antenna]
        cosine = np.cos(hour_angle_rotation)
        sine = np.sin(hour_angle_rotation)
        rotated_coordinates = np.column_stack((
            (cosine * station_coordinates[:, 0]) - (sine * station_coordinates[:, 1]),
            (sine * station_coordinates[:, 0]) + (cosine * station_coordinates[:, 1]),
            station_coordinates[:, 2],
        ))
        elevation = 0.5 * np.pi - np.arccos(
            np.sum(rotated_coordinates * source_vector, axis=1)
            / np.linalg.norm(rotated_coordinates, axis=1)
        )
        longitude = np.arctan2(station_coordinates[:, 1], station_coordinates[:, 0])
        latitude = np.arctan2(
            station_coordinates[:, 2],
            np.hypot(station_coordinates[:, 0], station_coordinates[:, 1]),
        )
        hour_angle = np.mod(
            (times_sidereal * (np.pi / 12.0)) + longitude - ra_rad,
            2.0 * np.pi,
        )
        parallactic_angle = np.arctan2(
            np.sin(hour_angle) * np.cos(latitude),
            (np.sin(latitude) * np.cos(dec_rad))
            - (np.cos(latitude) * np.sin(dec_rad) * np.cos(hour_angle)),
        )
        return elevation, parallactic_angle

    elevation1, parallactic_angle1 = station_angles(antenna1)
    elevation2, parallactic_angle2 = station_angles(antenna2)
    return StationGeometry(
        elevation1_rad=elevation1,
        elevation2_rad=elevation2,
        parallactic_angle1_rad=parallactic_angle1,
        parallactic_angle2_rad=parallactic_angle2,
    )


def ground_station_geometry(obs):
    """Calculate station angles for a UTC ground-array ``ehtim.Obsdata``.

    Returns ``None`` for spacecraft-containing arrays or non-UTC observations,
    which retain the established ``ehtim`` metadata path.
    """

    if getattr(obs, "timetype", None) != "UTC":
        return None

    coordinates = np.column_stack((obs.tarr["x"], obs.tarr["y"], obs.tarr["z"]))
    station_index = {str(site): index for index, site in enumerate(obs.tarr["site"])}
    try:
        antenna1 = np.fromiter(
            (station_index[str(site)] for site in obs.data["t1"]),
            dtype=np.intp,
            count=len(obs.data),
        )
        antenna2 = np.fromiter(
            (station_index[str(site)] for site in obs.data["t2"]),
            dtype=np.intp,
            count=len(obs.data),
        )
    except KeyError:
        return None

    return station_geometry_from_rows(
        coordinates,
        np.floor(float(obs.mjd)) + (np.asarray(obs.data["time"], dtype=float) / 24.0),
        antenna1,
        antenna2,
        obs.ra,
        obs.dec,
    )


def ground_geometry(array, context):
    """Generate ground-array baseline rows without ehtim geometry helpers.

    Parameters
    ----------
    array : ehtim.array.Array
        Ground-only telescope array. Spacecraft placeholder coordinates are
        rejected and must use the legacy route.
    context : mapping
        Normalized observation settings containing ``mjd``, ``ra``, ``dec``,
        ``rf``, ``t_start``, ``t_stop``, and ``t_rest``.

    Returns
    -------
    GroundGeometry
        Visible baseline-time rows with UVW coordinates in metres and in
        wavelengths.

    Raises
    ------
    ValueError
        If the array contains a spacecraft placeholder or the requested time
        range produces no baseline-time rows.
    """

    if _has_space_station(array):
        raise ValueError("Ground geometry does not support spacecraft stations.")

    times = _observation_times(context)
    station1, station2 = np.triu_indices(len(array.tarr), k=1)
    if not len(times) or not len(station1):
        raise ValueError("Ground geometry requires at least one timestamp and baseline.")

    row_station1 = np.repeat(station1, len(times))
    row_station2 = np.repeat(station2, len(times))
    row_times = np.tile(times, len(station1))
    time_object = Time(
        np.floor(float(context["mjd"])) + (times / 24.0),
        format="mjd",
        scale="utc",
    )
    sidereal_hours = np.tile(
        time_object.sidereal_time("mean", "greenwich").hour,
        len(station1),
    )
    theta = np.mod((sidereal_hours - float(context["ra"])) * (np.pi / 12.0), 2.0 * np.pi)

    coordinates = np.column_stack((array.tarr["x"], array.tarr["y"], array.tarr["z"]))
    coordinate1 = coordinates[row_station1]
    coordinate2 = coordinates[row_station2]
    cosine = np.cos(theta)
    sine = np.sin(theta)

    def rotate(coordinate):
        return np.column_stack((
            (cosine * coordinate[:, 0]) - (sine * coordinate[:, 1]),
            (sine * coordinate[:, 0]) + (cosine * coordinate[:, 1]),
            coordinate[:, 2],
        ))

    coordinate1 = rotate(coordinate1)
    coordinate2 = rotate(coordinate2)

    declination = np.deg2rad(float(context["dec"]))
    source_vector = np.array((np.cos(declination), 0.0, np.sin(declination)))
    projection_u = np.cross(np.array((0.0, 0.0, 1.0)), source_vector)
    projection_u /= np.linalg.norm(projection_u)
    projection_v = -np.cross(projection_u, source_vector)
    baseline_m = coordinate1 - coordinate2
    uvw_m = np.column_stack((
        baseline_m @ projection_u,
        baseline_m @ projection_v,
        baseline_m @ source_vector,
    ))
    wavelength = SPEED_OF_LIGHT.to_value("m / s") / float(context["rf"])

    elevation1 = np.rad2deg(
        0.5 * np.pi - np.arccos(
            np.sum(coordinate1 * source_vector, axis=1)
            / np.linalg.norm(coordinate1, axis=1)
        )
    )
    elevation2 = np.rad2deg(
        0.5 * np.pi - np.arccos(
            np.sum(coordinate2 * source_vector, axis=1)
            / np.linalg.norm(coordinate2, axis=1)
        )
    )
    visible = (elevation1 > -90.0) & (elevation1 < 90.0)
    visible &= (elevation2 > -90.0) & (elevation2 < 90.0)

    return GroundGeometry(
        time_hours=row_times[visible],
        station1_indices=row_station1[visible],
        station2_indices=row_station2[visible],
        uvw_m=uvw_m[visible],
        u=uvw_m[visible, 0] / wavelength,
        v=uvw_m[visible, 1] / wavelength,
    )


def ground_visibility_template(array, context, geometry=None):
    """Build a native visibility template for a ground-only array.

    The template carries geometry and thermal uncertainties but contains
    zero-valued circular visibilities. Source sampling occurs separately
    through the source-adapter layer.

    Parameters
    ----------
    array : ehtim.array.Array
        Ground-only telescope array.
    context : mapping
        Normalized observation settings including source coordinates, observing
        frequency, bandwidth, integration time, and reference MJD.
    geometry : GroundGeometry, optional
        Precomputed geometry for the same array and context.

    Returns
    -------
    VisibilityDataset
        One-channel circular template with RR, LL, RL, LR products and thermal
        ``sigma_jy`` values computed from station SEFDs.
    """

    if geometry is None:
        geometry = ground_geometry(array, context)

    sefd1r = array.tarr["sefdr"][geometry.station1_indices]
    sefd2r = array.tarr["sefdr"][geometry.station2_indices]
    sefd1l = array.tarr["sefdl"][geometry.station1_indices]
    sefd2l = array.tarr["sefdl"][geometry.station2_indices]
    denominator = 2.0 * float(context["bandwidth_hz"]) * float(context["t_int"])
    sigma = np.column_stack((
        np.sqrt(sefd1r * sefd2r / denominator) / 0.88,
        np.sqrt(sefd1l * sefd2l / denominator) / 0.88,
        np.sqrt(sefd1r * sefd2l / denominator) / 0.88,
        np.sqrt(sefd1l * sefd2r / denominator) / 0.88,
    ))

    scan_half_width_s = 0.5 * float(context["t_int"])
    scan_times = np.unique(geometry.time_hours)
    reference_mjd = float(context["mjd"])
    time_mjd = reference_mjd + (geometry.time_hours / 24.0)
    scan_start_mjd = reference_mjd + (scan_times / 24.0) - (scan_half_width_s / 86400.0)
    scan_stop_mjd = reference_mjd + (scan_times / 24.0) + (scan_half_width_s / 86400.0)

    stations = StationTable.from_ehtim_tarr(array.tarr)
    receptors = ReceptorTable.from_station_labels(
        len(stations.names),
        ("R", "L"),
        "CIRCULAR",
    )
    correlation_products, row_product_id = standard_products_for_rows(
        receptors,
        geometry.station1_indices,
        geometry.station2_indices,
        CIRCULAR_PRODUCT_LABELS,
    )
    return VisibilityDataset(
        stations=stations,
        receptors=receptors,
        correlation_products=correlation_products,
        time_mjd=time_mjd,
        integration_time_s=np.full(len(time_mjd), float(context["t_int"])),
        antenna1=geometry.station1_indices,
        antenna2=geometry.station2_indices,
        uvw_m=geometry.uvw_m,
        tau1=np.zeros(len(time_mjd)),
        tau2=np.zeros(len(time_mjd)),
        channel_frequency_hz=np.array((float(context["rf"]),)),
        channel_bandwidth_hz=np.array((float(context["bandwidth_hz"]),)),
        spectral_window_id=np.array((0,), dtype=np.intp),
        row_product_id=row_product_id,
        visibilities=np.zeros((len(time_mjd), 1, 4), dtype=complex),
        sigma_jy=sigma[:, np.newaxis, :],
        flags=np.zeros((len(time_mjd), 1, 4), dtype=bool),
        source=str(context["ra"]) + ":" + str(context["dec"]),
        ra_hours=float(context["ra"]),
        dec_degrees=float(context["dec"]),
        ampcal=True,
        phasecal=True,
        opacitycal=True,
        dcal=True,
        frcal=True,
        scan_start_mjd=scan_start_mjd,
        scan_stop_mjd=scan_stop_mjd,
    )


def visibility_dataset_elevation_mask(dataset, el_min, el_max):
    """Return the native ground-array elevation-selection mask.

    Parameters
    ----------
    dataset : VisibilityDataset
        Ground-only native dataset.
    el_min, el_max : float
        Exclusive lower and upper elevation limits in degrees.

    Returns
    -------
    numpy.ndarray of bool, shape (row,)
        Rows for which both stations are strictly inside the elevation range.

    Raises
    ------
    ValueError
        If spacecraft stations require the legacy geometry path.
    """

    if not isinstance(dataset, VisibilityDataset):
        raise TypeError("dataset must be a VisibilityDataset instance.")
    if dataset.row_count == 0:
        return np.zeros(0, dtype=bool)
    geometry = station_geometry_from_rows(
        dataset.stations.position_itrs_m,
        dataset.time_mjd,
        dataset.antenna1,
        dataset.antenna2,
        dataset.ra_hours,
        dataset.dec_degrees,
    )
    if geometry is None:
        raise ValueError("Native elevation filtering does not support spacecraft stations.")
    elevation1 = np.rad2deg(geometry.elevation1_rad)
    elevation2 = np.rad2deg(geometry.elevation2_rad)
    return (
        (elevation1 > el_min)
        & (elevation1 < el_max)
        & (elevation2 > el_min)
        & (elevation2 < el_max)
    )


def apply_visibility_dataset_elevation_limits(dataset, el_min, el_max):
    """Return a native dataset restricted to the requested elevation range.

    Parameters
    ----------
    dataset : VisibilityDataset
        Ground-only dataset to filter.
    el_min, el_max : float
        Exclusive elevation limits in degrees.

    Returns
    -------
    VisibilityDataset
        New dataset retaining rows selected by
        :func:`visibility_dataset_elevation_mask`.
    """

    return dataset.select_rows(visibility_dataset_elevation_mask(dataset, el_min, el_max))


def canonicalize_visibility_dataset_rows(dataset):
    """Order native rows by time and station names before stochastic processing."""

    if not isinstance(dataset, VisibilityDataset):
        raise TypeError("dataset must be a VisibilityDataset instance.")
    names = np.asarray(dataset.stations.names)
    order = np.lexsort((
        names[dataset.antenna2],
        names[dataset.antenna1],
        dataset.time_mjd,
    ))
    return dataset.take_rows(order)


def native_visibility_template(template, cached_key, template_cache, array, context,
                               el_min, el_max):
    """Return a cached, elevation-limited native ground visibility template."""

    old_key = cached_key
    cached_key = geometry_cache_key(context)
    if template is None or cached_key != old_key:
        template = ground_visibility_template(array, context)
    if template_cache is None or cached_key != old_key:
        template_cache = {}

    elevation_key = elevation_cache_key(cached_key, el_min, el_max)
    if elevation_key not in template_cache:
        template_cache[elevation_key] = canonicalize_visibility_dataset_rows(
            apply_visibility_dataset_elevation_limits(template, el_min, el_max)
        )
    return template, cached_key, template_cache, template_cache[elevation_key]


def _ground_geometry_obsdata(array, context, geometry):
    """Adapt the native ground visibility template to the ``ehtim`` boundary."""

    return ground_visibility_template(
        array,
        context,
        geometry=geometry,
    ).to_ehtim_obsdata()


def _legacy_empty_observation(array, context):
    return array.obsdata(
        context["ra"],
        context["dec"],
        context["rf"],
        context["bandwidth_hz"],
        context["t_int"],
        context["t_rest"],
        context["t_start"],
        context["t_stop"],
        mjd=context["mjd"],
        polrep="circ",
        tau=0.0,
        timetype="UTC",
        elevmin=-90,
        elevmax=90,
        fix_theta_GMST=False,
    )


def _set_integration_scan_metadata(obs, context):
    """Set scan intervals centered on integrations with widths in seconds."""

    scan_half_width_hours = 0.5 * float(context["t_int"]) / 3600.0
    scan_times = np.unique(obs.data["time"])
    obs.scans = np.column_stack((
        scan_times - scan_half_width_hours,
        scan_times + scan_half_width_hours,
    ))
    return obs


def make_empty_observation(array, context):
    if _has_space_station(array):
        obs = _legacy_empty_observation(array, context)
    else:
        obs = _ground_geometry_obsdata(array, context, ground_geometry(array, context))
    return _set_integration_scan_metadata(obs, context)


def ensure_empty_observation(obs_empty, cached_key, array, context):
    new_key = geometry_cache_key(context)

    if needs_new_empty_observation(obs_empty, cached_key, context):
        return make_empty_observation(array, context), new_key

    return obs_empty, cached_key

###################################################
# elevation limits


def elevation_mask(obs_empty, el_min, el_max):
    els = obs_empty.unpack(["el1", "el2"])

    mask = (obs_empty.data["t1"] == "space") | ((els["el1"] > el_min) & (els["el1"] < el_max))
    mask &= (obs_empty.data["t2"] == "space") | ((els["el2"] > el_min) & (els["el2"] < el_max))

    return mask


def apply_elevation_limits(obs_empty, el_min, el_max):
    obs_limited = obs_empty.copy()
    obs_limited.data = obs_limited.data[elevation_mask(obs_empty, el_min, el_max)]
    return obs_limited

###################################################
# public interface


def elevation_cache_key(cached_key, el_min, el_max):
    return cached_key, _cache_value(el_min), _cache_value(el_max)


def cached_elevation_template(obs_empty, template_cache, cached_key, el_min, el_max):
    key = elevation_cache_key(cached_key, el_min, el_max)
    if key not in template_cache:
        template_cache[key] = apply_elevation_limits(obs_empty, el_min, el_max)
    return template_cache[key]


def observation_template(obs_empty, cached_key, template_cache, array, context, el_min, el_max):
    old_key = cached_key
    obs_empty, cached_key = ensure_empty_observation(obs_empty, cached_key, array, context)

    if template_cache is None or cached_key != old_key:
        template_cache = {}

    obs_limited = cached_elevation_template(
        obs_empty,
        template_cache,
        cached_key,
        el_min,
        el_max,
    )

    return obs_empty, cached_key, template_cache, obs_limited.copy()
