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

        def sample_observation():
            # This reproduces ehtim.Model.observe_same_nonoise() without
            # constructing another Obsdata object around a copied data table.
            obs = obs_empty.copy()
            u = obs.data["u"]
            v = obs.data["v"]
            if obs.polrep == "circ":
                # Keep ehtim's sampling order so model implementations with
                # stateful sampling behavior retain the established result.
                obs.data["rrvis"] = self.input_model.sample_uv(u, v, pol="RR")
                obs.data["rlvis"] = self.input_model.sample_uv(u, v, pol="RL")
                obs.data["lrvis"] = self.input_model.sample_uv(u, v, pol="LR")
                obs.data["llvis"] = self.input_model.sample_uv(u, v, pol="LL")
            elif obs.polrep == "stokes":
                obs.data["vis"] = self.input_model.sample_uv(u, v, pol="I")
                obs.data["qvis"] = self.input_model.sample_uv(u, v, pol="Q")
                obs.data["uvis"] = self.input_model.sample_uv(u, v, pol="U")
                obs.data["vvis"] = self.input_model.sample_uv(u, v, pol="V")
            else:
                raise ValueError("Unsupported ehtim observation polarization representation: {0}".format(obs.polrep))

            # Match ehtim.Model.observe_same_nonoise() calibration semantics.
            obs.ampcal = True
            obs.phasecal = True
            obs.opacitycal = True
            obs.dcal = True
            obs.frcal = True
            return obs

        obs = _run_quietly(sample_observation, context["verbosity"])

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
