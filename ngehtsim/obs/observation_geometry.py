###################################################
# cache key construction

from dataclasses import dataclass

import numpy as np
from astropy.constants import c as SPEED_OF_LIGHT
from astropy.time import Time
import ehtim as eh


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
    """Ground-station observation rows independent of ``ehtim.Obsdata``."""

    time_hours: np.ndarray
    station1_indices: np.ndarray
    station2_indices: np.ndarray
    u: np.ndarray
    v: np.ndarray


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


def ground_geometry(array, context):
    """Generate ground-array baseline rows without ``ehtim`` geometry helpers."""

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
    wavelength = SPEED_OF_LIGHT.to_value("m / s") / float(context["rf"])
    baseline = (coordinate1 - coordinate2) / wavelength

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
        u=baseline[visible] @ projection_u,
        v=baseline[visible] @ projection_v,
    )


def _ground_geometry_obsdata(array, context, geometry):
    """Adapt internal ground geometry to the established ``ehtim.Obsdata`` boundary."""

    data = np.zeros(len(geometry.time_hours), dtype=eh.const_def.DTPOL_CIRC)
    data["time"] = geometry.time_hours
    data["tint"] = float(context["t_int"])
    data["t1"] = array.tarr["site"][geometry.station1_indices]
    data["t2"] = array.tarr["site"][geometry.station2_indices]
    data["u"] = geometry.u
    data["v"] = geometry.v

    sefd1r = array.tarr["sefdr"][geometry.station1_indices]
    sefd2r = array.tarr["sefdr"][geometry.station2_indices]
    sefd1l = array.tarr["sefdl"][geometry.station1_indices]
    sefd2l = array.tarr["sefdl"][geometry.station2_indices]
    denominator = 2.0 * float(context["bandwidth_hz"]) * float(context["t_int"])
    data["rrsigma"] = np.sqrt(sefd1r * sefd2r / denominator) / 0.88
    data["llsigma"] = np.sqrt(sefd1l * sefd2l / denominator) / 0.88
    data["rlsigma"] = np.sqrt(sefd1r * sefd2l / denominator) / 0.88
    data["lrsigma"] = np.sqrt(sefd1l * sefd2r / denominator) / 0.88

    scan_half_width = 0.5 * float(context["t_rest"])
    scan_times = np.unique(geometry.time_hours)
    scans = np.column_stack((scan_times - scan_half_width, scan_times + scan_half_width))
    return eh.obsdata.Obsdata(
        context["ra"],
        context["dec"],
        context["rf"],
        context["bandwidth_hz"],
        data,
        array.tarr,
        source=str(context["ra"]) + ":" + str(context["dec"]),
        mjd=context["mjd"],
        timetype="UTC",
        polrep="circ",
        ampcal=True,
        phasecal=True,
        opacitycal=True,
        dcal=True,
        frcal=True,
        scantable=scans,
    )


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


def make_empty_observation(array, context):
    if _has_space_station(array):
        return _legacy_empty_observation(array, context)
    return _ground_geometry_obsdata(array, context, ground_geometry(array, context))


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
