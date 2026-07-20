###################################################
# imports

from dataclasses import replace

import numpy as np
import ehtim as eh
from astropy.constants import c as SPEED_OF_LIGHT

from ngehtsim.obs.visibility_dataset import CIRCULAR_CORRELATIONS, VisibilityDataset

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


def _require_native_circular_dataset(dataset):
    if not isinstance(dataset, VisibilityDataset):
        raise TypeError("dataset must be a VisibilityDataset instance.")
    if dataset.channel_count != 1:
        raise ValueError("Native source sampling requires exactly one spectral channel.")
    if any(
        dataset.correlation_layouts[index] != CIRCULAR_CORRELATIONS
        for index in dataset.row_layout_id
    ):
        raise ValueError(
            "Native source sampling requires circular RR, LL, RL, LR correlations."
        )


def _dataset_uv(dataset):
    wavelength = SPEED_OF_LIGHT.to_value("m / s") / dataset.channel_frequency_hz[0]
    return dataset.uvw_m[:, :2] / wavelength


def _with_circular_samples(dataset, sampled, context):
    visibilities = np.array(dataset.visibilities, copy=True)
    visibilities[:, 0, 0] = sampled[0]
    if sampled[1] is not None:
        visibilities[:, 0, 1] = sampled[1]
    if sampled[2] is not None:
        visibilities[:, 0, 2] = sampled[2]
        visibilities[:, 0, 3] = sampled[3]
    return replace(
        dataset,
        visibilities=visibilities,
        source=context["source"],
        ra_hours=context["ra"],
        dec_degrees=context["dec"],
        ampcal=True,
        phasecal=True,
        opacitycal=True,
        dcal=True,
        frcal=True,
    )

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

    def observe_dataset(self, dataset, context):
        """Sample an image onto a native circular single-channel dataset."""

        _require_native_circular_dataset(dataset)

        def sample_dataset():
            sampled = self.input_model.sample_uv(
                _dataset_uv(dataset),
                polrep_obs="circ",
                ttype=context["ttype"],
                fft_pad_factor=context["fft_pad_factor"],
                verbose=context["verbosity"] > 0,
            )
            return _with_circular_samples(
                dataset,
                sampled,
                context,
            )

        sampled_dataset = _run_quietly(sample_dataset, context["verbosity"])
        return sampled_dataset, self.input_model.total_flux()


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

    def observe_dataset(self, dataset, context):
        """Sample a repeating movie onto a native circular dataset."""

        _require_native_circular_dataset(dataset)

        def sample_dataset():
            visibilities = np.array(dataset.visibilities, copy=True)
            uv = _dataset_uv(dataset)
            observation_times = (dataset.time_mjd - float(context["mjd"])) * 24.0

            for time in np.unique(observation_times):
                sample_time = time
                if self.input_model.bounds_error:
                    if time < self.input_model.start_hr or time > self.input_model.stop_hr:
                        sample_time = self.input_model.start_hr + np.mod(
                            time - self.input_model.start_hr,
                            self.input_model.duration,
                        )
                row_mask = observation_times == time
                sampled = self.input_model.get_image(sample_time).sample_uv(
                    uv[row_mask],
                    polrep_obs="circ",
                    ttype=context["ttype"],
                    fft_pad_factor=context["fft_pad_factor"],
                    verbose=False,
                )
                visibilities[row_mask, 0, 0] = sampled[0]
                if sampled[1] is not None:
                    visibilities[row_mask, 0, 1] = sampled[1]
                if sampled[2] is not None:
                    visibilities[row_mask, 0, 2] = sampled[2]
                    visibilities[row_mask, 0, 3] = sampled[3]

            return replace(
                dataset,
                visibilities=visibilities,
                source=context["source"],
                ra_hours=context["ra"],
                dec_degrees=context["dec"],
                ampcal=True,
                phasecal=True,
                opacitycal=True,
                dcal=True,
                frcal=True,
            )

        sampled_dataset = _run_quietly(sample_dataset, context["verbosity"])
        return sampled_dataset, np.mean(self.input_model.lightcurve)


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

    def observe_dataset(self, dataset, context):
        """Sample an analytic model onto a native circular dataset."""

        _require_native_circular_dataset(dataset)

        def sample_dataset():
            uv = _dataset_uv(dataset)
            rr = self.input_model.sample_uv(uv[:, 0], uv[:, 1], pol="RR")
            ll = self.input_model.sample_uv(uv[:, 0], uv[:, 1], pol="LL")
            rl = self.input_model.sample_uv(uv[:, 0], uv[:, 1], pol="RL")
            lr = self.input_model.sample_uv(uv[:, 0], uv[:, 1], pol="LR")
            return _with_circular_samples(
                dataset,
                (rr, ll, rl, lr),
                context,
            )

        sampled_dataset = _run_quietly(sample_dataset, context["verbosity"])
        return sampled_dataset, np.abs(self.input_model.sample_uv(0.0, 0.0))


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


def observe_source_dataset(input_model, dataset, context):
    """Sample a supported source model onto a native visibility dataset."""

    adapter = adapter_for(input_model)
    if not hasattr(adapter, "observe_dataset"):
        raise TypeError(
            "Native VisibilityDataset sampling currently supports ehtim Image, Movie, "
            "and Model inputs only."
        )
    return adapter.observe_dataset(dataset, context)
