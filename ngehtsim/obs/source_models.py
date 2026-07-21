###################################################
# imports

from dataclasses import replace

import numpy as np
import ehtim as eh
from astropy.constants import c as SPEED_OF_LIGHT

from ngehtsim.obs import raster_sampling
from ngehtsim.obs.visibility_dataset import VisibilityDataset

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


def _sample_raster(image, uv, polrep_obs, context, *, native_path):
    """Sample one ehtim raster without mutating its source metadata."""

    backend = raster_sampling.resolve_transform_backend(
        context["transform_backend"],
        native_path=native_path,
    )
    return raster_sampling.sample_ehtim_raster(
        image,
        uv,
        polrep_obs=polrep_obs,
        backend=backend,
        tolerance=context["raster_tolerance"],
    )


def _require_native_circular_dataset(dataset):
    if not isinstance(dataset, VisibilityDataset):
        raise TypeError("dataset must be a VisibilityDataset instance.")
    if dataset.channel_count != 1:
        raise ValueError("Native source sampling requires exactly one spectral channel.")
    try:
        return dataset.circular_product_slots()
    except ValueError as exc:
        raise ValueError(
            "Native source sampling requires exactly circular RR, LL, RL, LR correlations."
        ) from exc


def _dataset_uv(dataset):
    wavelength = SPEED_OF_LIGHT.to_value("m / s") / dataset.channel_frequency_hz[0]
    return dataset.uvw_m[:, :2] / wavelength


def _write_circular_samples(visibilities, slots, rows, sampled):
    """Write RR, LL, RL, LR samples into explicitly mapped product slots."""

    rows = np.asarray(rows, dtype=np.intp)
    visibilities[rows, 0, slots[rows, 0]] = sampled[0]
    if sampled[1] is not None:
        visibilities[rows, 0, slots[rows, 1]] = sampled[1]
    if sampled[2] is not None:
        visibilities[rows, 0, slots[rows, 2]] = sampled[2]
        visibilities[rows, 0, slots[rows, 3]] = sampled[3]


def _with_circular_samples(dataset, sampled, context):
    visibilities = np.array(dataset.visibilities, copy=True)
    _write_circular_samples(
        visibilities,
        dataset.circular_product_slots(),
        np.arange(dataset.row_count),
        sampled,
    )
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
    """Adapt an ``ehtim.Image`` to ngehtsim observation interfaces.

    Parameters
    ----------
    input_model : ehtim.image.Image
        Raster source model. ngehtsim samples it with its native direct or
        FINUFFT transform implementation and never mutates this object.

    Notes
    -----
    :meth:`observe_dataset` samples directly into a native dataset, avoiding
    an intermediate ``ehtim.Obsdata`` data table. It currently requires one
    channel with exactly circular RR, LL, RL, LR products.  The public
    ``transform_backend`` setting controls its Fourier-transform backend.
    """

    def __init__(self, input_model):
        self.input_model = input_model

    def observe(self, obs_empty, context, p=None):
        """Sample the image onto an ehtim observation boundary object.

        Parameters
        ----------
        obs_empty : ehtim.obsdata.Obsdata
            Geometry-only observation rows to populate.
        context : mapping
            Normalized observation settings, including source coordinates,
            frequency, transform backend, and verbosity.
        p : object, optional
            Unused; retained for the common source-adapter interface.

        Returns
        -------
        ehtim.obsdata.Obsdata
            Noise-free source-sampled observation.
        float
            Image total flux density in Jy.
        """
        def sample_observation():
            # The Obsdata boundary retains a direct reference transform.
            obs = obs_empty.copy()
            uv = np.column_stack((obs.data["u"], obs.data["v"]))
            sampled = _sample_raster(
                self.input_model,
                uv,
                obs.polrep,
                context,
                native_path=False,
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

            obs.source = context["source"]
            obs.mjd = context["mjd"]
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
        """Sample an image onto a native circular single-channel dataset.

        Parameters
        ----------
        dataset : VisibilityDataset
            Geometry and correlation layout to populate.
        context : mapping
            Normalized observation settings.

        Returns
        -------
        VisibilityDataset
            Copy of ``dataset`` containing source visibilities and source
            metadata.
        float
            Image total flux density in Jy.

        Raises
        ------
        ValueError
            If the dataset is not exactly a one-channel circular layout or the
            configured native raster transform backend is unsupported.
        """

        _require_native_circular_dataset(dataset)

        def sample_dataset():
            sampled = _sample_raster(
                self.input_model,
                _dataset_uv(dataset),
                "circ",
                context,
                native_path=True,
            )
            return _with_circular_samples(
                dataset,
                sampled,
                context,
            )

        sampled_dataset = _run_quietly(sample_dataset, context["verbosity"])
        return sampled_dataset, self.input_model.total_flux()


class EhtimMovieAdapter(object):
    """Adapt a time-varying ``ehtim.Movie`` to ngehtsim interfaces.

    Parameters
    ----------
    input_model : ehtim.movie.Movie
        Raster movie sampled at each distinct observation timestamp without
        mutating the movie or its generated image frames.

    Notes
    -----
    The movie's own ``bounds_error`` behavior determines whether observation
    times outside its nominal range are looped. Native sampling has the same
    circular single-channel limitation as :class:`EhtimImageAdapter`.
    """

    def __init__(self, input_model):
        self.input_model = input_model

    def observe(self, obs_empty, context, p=None):
        """Sample movie frames onto an ehtim observation boundary object.

        Parameters
        ----------
        obs_empty : ehtim.obsdata.Obsdata
            Geometry-only observation rows to populate.
        context : mapping
            Normalized observation settings. The legacy compatibility route
            resolves ``transform_backend="auto"`` to the direct reference
            transform.
        p : object, optional
            Unused; retained for the common source-adapter interface.

        Returns
        -------
        ehtim.obsdata.Obsdata
            Noise-free observation with the corresponding movie frame sampled
            at every timestamp.
        float
            Mean movie light-curve flux density in Jy.
        """
        def sample_observation():
            # The Obsdata boundary retains the direct reference transform.
            obs = obs_empty.copy()
            obslist = obs_empty.tlist()
            obstimes = np.array([obsdata[0]["time"] for obsdata in obslist])

            if context["verbosity"] > 0:
                print("Producing clean visibilities from movie with direct FT . . . ")

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
                sampled = _sample_raster(
                    image,
                    uv,
                    obs.polrep,
                    context,
                    native_path=False,
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
            obs.source = context["source"]
            obs.mjd = context["mjd"]
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
        """Sample a repeating movie onto a native circular dataset.

        Parameters
        ----------
        dataset : VisibilityDataset
            Geometry and correlation layout to populate.
        context : mapping
            Normalized observation settings. Dataset MJD values determine the
            movie sampling times relative to ``context["mjd"]``. The native
            route resolves ``transform_backend="auto"`` to FINUFFT.

        Returns
        -------
        VisibilityDataset
            Dataset populated from the applicable frame for every timestamp.
        float
            Mean movie light-curve flux density in Jy.

        Raises
        ------
        ValueError
            If the dataset is not exactly a one-channel circular layout or the
            configured raster transform backend is unsupported.
        """

        circular_slots = _require_native_circular_dataset(dataset)

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
                sampled = _sample_raster(
                    self.input_model.get_image(sample_time),
                    uv[row_mask],
                    "circ",
                    context,
                    native_path=True,
                )
                _write_circular_samples(
                    visibilities,
                    circular_slots,
                    np.flatnonzero(row_mask),
                    sampled,
                )

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
    """Adapt an analytic ``ehtim.Model`` to ngehtsim interfaces.

    Parameters
    ----------
    input_model : ehtim.model.Model
        Analytic source model sampled by ehtim's model evaluator.

    Notes
    -----
    Unlike raster Image and Movie models, this path does not use an NFFT
    backend. Native sampling still requires a one-channel circular layout.
    """

    def __init__(self, input_model):
        self.input_model = input_model

    def observe(self, obs_empty, context, p=None):
        """Sample the analytic model onto an ehtim observation boundary object.

        Returns
        -------
        ehtim.obsdata.Obsdata
            Noise-free model-sampled observation.
        float
            Zero-baseline model amplitude in Jy.
        """
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
        """Sample an analytic model onto a native circular dataset.

        Parameters
        ----------
        dataset : VisibilityDataset
            Geometry and correlation layout to populate.
        context : mapping
            Normalized observation settings used to label the output source.

        Returns
        -------
        VisibilityDataset
            Dataset with analytic model samples in its circular product slots.
        float
            Zero-baseline model amplitude in Jy.
        """

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
    """Adapt an optional ``ngEHTforecast.FisherForecast`` source model."""

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
    """Return the source adapter appropriate for a supported input object.

    Parameters
    ----------
    input_model : ehtim Image, Movie, or Model, or ngEHTforecast FisherForecast
        Source object accepted by ngehtsim.

    Returns
    -------
    EhtimImageAdapter, EhtimMovieAdapter, EhtimModelAdapter, or FisherForecastAdapter
        Adapter implementing the appropriate observation route.

    Raises
    ------
    TypeError
        If ``input_model`` is not a supported source type.
    """
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
    """Sample a supported source onto an ehtim observation boundary object.

    Parameters
    ----------
    input_model : supported source object
        Source accepted by :func:`adapter_for`.
    obs_empty : ehtim.obsdata.Obsdata
        Geometry-only observation rows to populate.
    context : mapping
        Normalized observation settings.
    p : object, optional
        Parameter vector required by ``ngEHTforecast.FisherForecast`` models.

    Returns
    -------
    ehtim.obsdata.Obsdata
        Noise-free source-sampled observation.
    float
        Reference total or zero-baseline flux density in Jy.
    """
    return adapter_for(input_model).observe(obs_empty, context, p=p)


def observe_source_dataset(input_model, dataset, context):
    """Sample a supported source model onto a native visibility dataset.

    Parameters
    ----------
    input_model : ehtim Image, Movie, or Model
        Source object supporting native sampling.
    dataset : VisibilityDataset
        Native geometry and correlation layout to populate.
    context : mapping
        Normalized observation settings.

    Returns
    -------
    VisibilityDataset
        Source-sampled native dataset.
    float
        Reference total or zero-baseline flux density in Jy.

    Raises
    ------
    TypeError
        If the source type has no native dataset adapter.
    ValueError
        If the dataset has a currently unsupported channel or correlation
        layout.
    """

    adapter = adapter_for(input_model)
    if not hasattr(adapter, "observe_dataset"):
        raise TypeError(
            "Native VisibilityDataset sampling currently supports ehtim Image, Movie, "
            "and Model inputs only."
        )
    return adapter.observe_dataset(dataset, context)
