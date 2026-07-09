###################################################
# imports

import numpy as np
import ehtim as eh

try:
    import ngEHTforecast.fisher as fp
except ImportError:
    fp = None

###################################################
# helpers


def _set_ehtim_metadata(input_model, context):
    input_model.ra = context["ra"]
    input_model.dec = context["dec"]
    input_model.mjd = context["mjd"]
    input_model.source = context["source"]
    input_model.rf = context["rf"]


def _run_quietly(function, verbosity):
    if verbosity <= 0:
        with eh.parloop.HiddenPrints():
            return function()
    return function()

###################################################
# adapters


class EhtimImageAdapter(object):
    def __init__(self, input_model):
        self.input_model = input_model

    def observe(self, obs_empty, context, p=None):
        _set_ehtim_metadata(self.input_model, context)

        obs = _run_quietly(
            lambda: self.input_model.observe_same_nonoise(
                obs_empty,
                ttype=context["ttype"],
                fft_pad_factor=context["fft_pad_factor"],
            ),
            context["verbosity"],
        )

        F0 = self.input_model.total_flux()
        return obs, F0


class EhtimMovieAdapter(object):
    def __init__(self, input_model):
        self.input_model = input_model

    def observe(self, obs_empty, context, p=None):
        _set_ehtim_metadata(self.input_model, context)

        obs = _run_quietly(
            lambda: self.input_model.observe_same_nonoise(
                obs_empty,
                ttype=context["ttype"],
                fft_pad_factor=context["fft_pad_factor"],
                repeat=True,
            ),
            context["verbosity"],
        )

        F0 = np.mean(self.input_model.lightcurve)
        return obs, F0


class EhtimModelAdapter(object):
    def __init__(self, input_model):
        self.input_model = input_model

    def observe(self, obs_empty, context, p=None):
        _set_ehtim_metadata(self.input_model, context)

        obs = _run_quietly(
            lambda: self.input_model.observe_same_nonoise(obs_empty),
            context["verbosity"],
        )

        F0 = np.abs(self.input_model.sample_uv(0.0, 0.0))
        return obs, F0


class FisherForecastAdapter(object):
    def __init__(self, input_model):
        self.input_model = input_model

    def observe(self, obs_empty, context, p=None):
        if p is None:
            raise Exception("When observing an ngEHTforecast model, the parameter vector keyword argument p must be specified!")

        obs = obs_empty.copy()
        obs.source = context["source"]

        if self.input_model.stokes == "I":
            Ivis = self.input_model.visibilities(obs, p, verbosity=context["verbosity"])
            obs = obs.switch_polrep(polrep_out="stokes")
            obs.data["vis"] = Ivis
            obs = obs.switch_polrep(polrep_out="circ")
        else:
            obs = obs.switch_polrep(polrep_out="circ")
            RRvis, LLvis, RLvis, LRvis = self.input_model.visibilities(obs, p, verbosity=context["verbosity"])
            obs.data["rrvis"] = RRvis
            obs.data["llvis"] = LLvis
            obs.data["rlvis"] = RLvis
            obs.data["lrvis"] = LRvis

        dumobs = obs_empty.copy()
        dumdatatable = dumobs.data[0]
        dumdatatable["u"] = 0.0
        dumdatatable["v"] = 0.0
        dumobs.data = dumdatatable
        F0 = np.abs(self.input_model.visibilities(dumobs, p))

        return obs, F0

###################################################
# public interface


def adapter_for(input_model):
    if isinstance(input_model, eh.image.Image):
        return EhtimImageAdapter(input_model)

    if isinstance(input_model, eh.movie.Movie):
        return EhtimMovieAdapter(input_model)

    if isinstance(input_model, eh.model.Model):
        return EhtimModelAdapter(input_model)

    if (fp is not None) and isinstance(input_model, fp.FisherForecast):
        return FisherForecastAdapter(input_model)

    raise TypeError("input_model must be an ehtim Image, Movie, Model, or ngEHTforecast FisherForecast object.")


def observe_source(input_model, obs_empty, context, p=None):
    return adapter_for(input_model).observe(obs_empty, context, p=p)
