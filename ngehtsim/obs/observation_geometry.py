###################################################
# cache key construction

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


def make_empty_observation(array, context):
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
