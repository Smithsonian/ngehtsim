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

        def sample_observation():
            # This reproduces ehtim.Image.observe_same_nonoise() without
            # constructing another Obsdata object around a copied data table.
            obs = obs_empty.copy()
            uv = np.column_stack((obs.data["u"], obs.data["v"]))
            sampled = self.input_model.sample_uv(
                uv,
                polrep_obs=obs.polrep,
                ttype=context["ttype"],
                fft_pad_factor=context["fft_pad_factor"],
                verbose=context["verbosity"] > 0,
            )
            if obs.polrep == "circ":
                obs.data["rrvis"] = sampled[0]
                if sampled[1] is not None:
                    obs.data["llvis"] = sampled[1]
                if sampled[2] is not None:
                    obs.data["rlvis"] = sampled[2]
                    obs.data["lrvis"] = sampled[3]
            elif obs.polrep == "stokes":
                obs.data["vis"] = sampled[0]
                if sampled[1] is not None:
                    obs.data["qvis"] = sampled[1]
                    obs.data["uvis"] = sampled[2]
                    obs.data["vvis"] = sampled[3]
            else:
                raise ValueError("Unsupported ehtim observation polarization representation: {0}".format(obs.polrep))

            # Match ehtim.Image.observe_same_nonoise() metadata semantics.
            obs.source = self.input_model.source
            obs.mjd = self.input_model.mjd
            obs.ampcal = True
            obs.phasecal = True
            obs.opacitycal = True
            obs.dcal = True
            obs.frcal = True
            return obs

        obs = _run_quietly(sample_observation, context["verbosity"])

        F0 = self.input_model.total_flux()
        return obs, F0


class EhtimMovieAdapter(object):
    def __init__(self, input_model):
        self.input_model = input_model

    def observe(self, obs_empty, context, p=None):
        _set_ehtim_metadata(self.input_model, context)

        def sample_observation():
            # This reproduces ehtim.Movie.observe_same_nonoise() without
            # constructing another Obsdata object around each time slice.
            obs = obs_empty.copy()
            obslist = obs_empty.tlist()
            obstimes = np.array([obsdata[0]["time"] for obsdata in obslist])

            if context["ttype"] not in ("direct", "fast", "nfft"):
                raise Exception(
                    "ttype={0}, options for ttype are 'direct', 'fast', 'nfft'".format(
                        context["ttype"]
                    )
                )
            if context["verbosity"] > 0:
                print("Producing clean visibilities from movie with " + context["ttype"] + " FT . . . ")

            if (obstimes < self.input_model.start_hr).any():
                if context["verbosity"] > 0:
                    print(
                        "Some observation times before movie start time %f"
                        % self.input_model.start_hr
                    )
                    print("Looping movie before start\\n")
            if (obstimes > self.input_model.stop_hr).any():
                if context["verbosity"] > 0:
                    print(
                        "Some observation times after movie stop time %f"
                        % self.input_model.stop_hr
                    )
                    print("Looping movie after stop\\n")

            sampled_rows = []
            for obsdata in obslist:
                time = obsdata[0]["time"]
                if self.input_model.bounds_error:
                    if time < self.input_model.start_hr or time > self.input_model.stop_hr:
                        time = self.input_model.start_hr + np.mod(
                            time - self.input_model.start_hr,
                            self.input_model.duration,
                        )

                image = self.input_model.get_image(time)
                uv = np.column_stack((obsdata["u"], obsdata["v"]))
                sampled = image.sample_uv(
                    uv,
                    polrep_obs=obs.polrep,
                    ttype=context["ttype"],
                    fft_pad_factor=context["fft_pad_factor"],
                    verbose=False,
                )

                if obs.polrep == "circ":
                    obsdata["rrvis"] = sampled[0]
                    if sampled[1] is not None:
                        obsdata["llvis"] = sampled[1]
                    if sampled[2] is not None:
                        obsdata["rlvis"] = sampled[2]
                        obsdata["lrvis"] = sampled[3]
                elif obs.polrep == "stokes":
                    obsdata["vis"] = sampled[0]
                    if sampled[1] is not None:
                        obsdata["qvis"] = sampled[1]
                        obsdata["uvis"] = sampled[2]
                        obsdata["vvis"] = sampled[3]
                else:
                    raise ValueError(
                        "Unsupported ehtim observation polarization representation: {0}".format(
                            obs.polrep
                        )
                    )

                sampled_rows.append(obsdata)

            if sampled_rows:
                obs.data = np.hstack(sampled_rows)
            obs.source = self.input_model.source
            obs.mjd = np.floor(obs_empty.mjd)
            obs.ampcal = True
            obs.phasecal = True
            obs.opacitycal = True
            obs.dcal = True
            obs.frcal = True
            return obs

        obs = _run_quietly(sample_observation, context["verbosity"])

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
