###################################################
# empty observation construction


def needs_new_empty_observation(obs_empty, rf):
    return (obs_empty is None) or (obs_empty.rf != rf)


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


def ensure_empty_observation(obs_empty, array, context):
    if needs_new_empty_observation(obs_empty, context["rf"]):
        return make_empty_observation(array, context)
    return obs_empty

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


def observation_template(obs_empty, array, context, el_min, el_max):
    obs_empty = ensure_empty_observation(obs_empty, array, context)
    obs_limited = apply_elevation_limits(obs_empty, el_min, el_max)
    return obs_empty, obs_limited
