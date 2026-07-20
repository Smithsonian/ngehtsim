#######################################################
# imports

import pytest
import ehtim as eh
import numpy as np
import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.observation_geometry as observation_geometry
import ngehtsim.obs.source_models as source_models
from ngehtsim.obs.visibility_dataset import VisibilityDataset

#######################################################
# helpers

COMPACT_OBS_SETTINGS = {
    "source": "M87",
    "sites": ["ALMA", "APEX", "LMT", "SMT"],
    "weather": "typical",
    "t_start": 0.0,
    "dt": 6.0,
    "t_int": 600.0,
    "t_rest": 1200.0,
    "random_seed": 1,
}


def _compact_model():
    model = eh.model.Model()
    return model.add_circ_gauss(F0=1.0, FWHM=40.0 * eh.RADPERUAS)


def _polarized_model():
    model = eh.model.Model()
    model = model.add_circ_gauss(
        F0=0.6,
        FWHM=30.0 * eh.RADPERUAS,
        x0=15.0 * eh.RADPERUAS,
        pol_frac=0.20,
        pol_evpa=30.0 * eh.DEGREE,
        cpol_frac=0.10,
    )
    return model.add_circ_gauss(
        F0=0.4,
        FWHM=45.0 * eh.RADPERUAS,
        x0=-20.0 * eh.RADPERUAS,
        y0=10.0 * eh.RADPERUAS,
        pol_frac=0.15,
        pol_evpa=75.0 * eh.DEGREE,
        cpol_frac=-0.05,
    )


def _polarized_movie():
    grid = np.linspace(-1.0, 1.0, 32)
    x, y = np.meshgrid(grid, grid)
    frame_0 = np.exp(-6.0 * ((x + 0.20) ** 2 + (y - 0.10) ** 2))
    frame_1 = 1.25 * np.exp(-6.0 * ((x - 0.20) ** 2 + (y + 0.10) ** 2))
    frames = (frame_0, frame_1)
    movie = eh.movie.Movie(
        frames,
        times=(0.0, 0.5),
        psize=10.0 * eh.RADPERUAS,
        ra=0.0,
        dec=0.0,
        rf=230.0e9,
        source="M87",
        mjd=57849,
        bounds_error=True,
    )
    movie.add_pol_movie((0.20 * frame_0, 0.15 * frame_1), "Q")
    movie.add_pol_movie((0.05 * frame_0, -0.10 * frame_1), "U")
    movie.add_pol_movie((0.10 * frame_0, -0.05 * frame_1), "V")
    return movie


def _observe_without_corruptions(obsgen, input_model):
    return obsgen.observe(
        input_model,
        addnoise=False,
        addgains=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )


def _empty_observation(obsgen):
    _, _, _, obs_empty = observation_geometry.observation_template(
        None,
        None,
        {},
        obsgen.arr,
        obsgen.geometry_context(),
        el_min=0.0,
        el_max=90.0,
    )
    return obs_empty


def _legacy_model_observe(input_model, obs_empty, context):
    source_models._set_ehtim_metadata(input_model, context)
    obs = input_model.observe_same_nonoise(obs_empty)
    F0 = np.abs(input_model.sample_uv(0.0, 0.0))
    return obs, F0


def _legacy_image_observe(input_model, obs_empty, context):
    source_models._set_ehtim_metadata(input_model, context)
    obs = input_model.observe_same_nonoise(
        obs_empty,
        ttype=context["ttype"],
        fft_pad_factor=context["fft_pad_factor"],
    )
    F0 = input_model.total_flux()
    return obs, F0


def _legacy_movie_observe(input_model, obs_empty, context):
    source_models._set_ehtim_metadata(input_model, context)
    obs = input_model.observe_same_nonoise(
        obs_empty,
        ttype=context["ttype"],
        fft_pad_factor=context["fft_pad_factor"],
        repeat=True,
    )
    F0 = np.mean(input_model.lightcurve)
    return obs, F0

#######################################################
# tests


def test_adapter_for_ehtim_model():
    adapter = source_models.adapter_for(_compact_model())
    assert isinstance(adapter, source_models.EhtimModelAdapter)


def test_adapter_for_ehtim_image():
    image = _compact_model().make_image(160.0 * eh.RADPERUAS, 64)

    adapter = source_models.adapter_for(image)

    assert isinstance(adapter, source_models.EhtimImageAdapter)


def test_adapter_for_ehtim_movie():
    adapter = source_models.adapter_for(_polarized_movie())

    assert isinstance(adapter, source_models.EhtimMovieAdapter)


def test_adapter_rejects_unsupported_model_type():
    with pytest.raises(TypeError, match="input_model must be"):
        source_models.adapter_for(object())


def test_obs_generator_still_accepts_ehtim_model():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)

    obs = _observe_without_corruptions(obsgen, _compact_model())

    assert len(obs.data) > 0


def test_ehtim_model_adapter_does_not_call_observe_same_nonoise(monkeypatch):
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    model = _compact_model()

    def unexpected_legacy_sampler(*args, **kwargs):
        raise AssertionError("The model adapter must sample the cached template directly.")

    monkeypatch.setattr(model, "observe_same_nonoise", unexpected_legacy_sampler)
    obs, F0 = source_models.observe_source(model, _empty_observation(obsgen), obsgen.source_context())

    assert len(obs.data) > 0
    assert F0 > 0.0


def test_ehtim_model_adapter_matches_legacy_full_polarization_sampling():
    direct_generator = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    legacy_generator = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    direct_model = _polarized_model()
    legacy_model = _polarized_model()

    direct, direct_F0 = source_models.observe_source(
        direct_model,
        _empty_observation(direct_generator),
        direct_generator.source_context(),
    )
    legacy, legacy_F0 = _legacy_model_observe(
        legacy_model,
        _empty_observation(legacy_generator),
        legacy_generator.source_context(),
    )

    assert direct.ampcal is legacy.ampcal is True
    assert direct.phasecal is legacy.phasecal is True
    assert direct.opacitycal is legacy.opacitycal is True
    assert direct.dcal is legacy.dcal is True
    assert direct.frcal is legacy.frcal is True
    assert direct_F0 == pytest.approx(legacy_F0)
    for field in ("rrvis", "rlvis", "lrvis", "llvis"):
        assert np.allclose(direct.data[field], legacy.data[field], atol=1.0e-12)


def test_ehtim_image_adapter_does_not_call_observe_same_nonoise(monkeypatch):
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    image = _polarized_model().make_image(160.0 * eh.RADPERUAS, 64)

    def unexpected_legacy_sampler(*args, **kwargs):
        raise AssertionError("The image adapter must sample the cached template directly.")

    monkeypatch.setattr(image, "observe_same_nonoise", unexpected_legacy_sampler)
    obs, F0 = source_models.observe_source(image, _empty_observation(obsgen), obsgen.source_context())

    assert len(obs.data) > 0
    assert F0 > 0.0


def test_ehtim_image_adapter_matches_legacy_full_polarization_sampling():
    direct_generator = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    legacy_generator = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    direct_image = _polarized_model().make_image(160.0 * eh.RADPERUAS, 64)
    legacy_image = _polarized_model().make_image(160.0 * eh.RADPERUAS, 64)

    direct, direct_F0 = source_models.observe_source(
        direct_image,
        _empty_observation(direct_generator),
        direct_generator.source_context(),
    )
    legacy, legacy_F0 = _legacy_image_observe(
        legacy_image,
        _empty_observation(legacy_generator),
        legacy_generator.source_context(),
    )

    assert direct.source == legacy.source
    assert direct.mjd == legacy.mjd
    assert direct.ampcal is legacy.ampcal is True
    assert direct.phasecal is legacy.phasecal is True
    assert direct.opacitycal is legacy.opacitycal is True
    assert direct.dcal is legacy.dcal is True
    assert direct.frcal is legacy.frcal is True
    assert direct_F0 == pytest.approx(legacy_F0)
    for field in ("rrvis", "rlvis", "lrvis", "llvis"):
        assert np.allclose(direct.data[field], legacy.data[field], atol=1.0e-12)


def test_ehtim_image_adapter_samples_native_visibility_dataset():
    direct_generator = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    legacy_generator = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    direct_image = _polarized_model().make_image(160.0 * eh.RADPERUAS, 64)
    legacy_image = _polarized_model().make_image(160.0 * eh.RADPERUAS, 64)
    template = observation_geometry.ground_visibility_template(
        direct_generator.arr,
        direct_generator.geometry_context(),
    )

    direct, direct_F0 = source_models.observe_source_dataset(
        direct_image,
        template,
        direct_generator.source_context(),
    )
    legacy, legacy_F0 = _legacy_image_observe(
        legacy_image,
        template.to_ehtim_obsdata(),
        legacy_generator.source_context(),
    )

    assert direct.source == legacy.source
    assert direct.ra_hours == legacy.ra
    assert direct.dec_degrees == legacy.dec
    assert direct.ampcal is legacy.ampcal is True
    assert direct.phasecal is legacy.phasecal is True
    assert direct.opacitycal is legacy.opacitycal is True
    assert direct.dcal is legacy.dcal is True
    assert direct.frcal is legacy.frcal is True
    assert direct_F0 == pytest.approx(legacy_F0)
    adapted = direct.to_ehtim_obsdata()
    for field in ("rrvis", "llvis", "rlvis", "lrvis"):
        assert np.allclose(adapted.data[field], legacy.data[field], atol=1.0e-12)


def test_ehtim_image_dataset_sampler_avoids_obsdata_conversion(monkeypatch):
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    image = _polarized_model().make_image(160.0 * eh.RADPERUAS, 64)
    template = observation_geometry.ground_visibility_template(
        obsgen.arr,
        obsgen.geometry_context(),
    )

    def unexpected_obsdata_conversion(*args, **kwargs):
        raise AssertionError("Native image sampling must not construct an Obsdata object.")

    def unexpected_legacy_sampler(*args, **kwargs):
        raise AssertionError("Native image sampling must not call observe_same_nonoise.")

    monkeypatch.setattr(
        VisibilityDataset,
        "to_ehtim_obsdata",
        unexpected_obsdata_conversion,
    )
    monkeypatch.setattr(image, "observe_same_nonoise", unexpected_legacy_sampler)
    sampled, F0 = source_models.observe_source_dataset(
        image,
        template,
        obsgen.source_context(),
    )

    assert sampled.row_count == template.row_count
    assert F0 > 0.0


def test_ehtim_movie_adapter_does_not_call_observe_same_nonoise(monkeypatch):
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    movie = _polarized_movie()

    def unexpected_legacy_sampler(*args, **kwargs):
        raise AssertionError("The movie adapter must sample the cached template directly.")

    monkeypatch.setattr(movie, "observe_same_nonoise", unexpected_legacy_sampler)
    obs, F0 = source_models.observe_source(movie, _empty_observation(obsgen), obsgen.source_context())

    assert len(obs.data) > 0
    assert F0 > 0.0


@pytest.mark.parametrize(
    ("polrep", "fields"),
    (
        ("circ", ("rrvis", "rlvis", "lrvis", "llvis")),
        ("stokes", ("vis", "qvis", "uvis", "vvis")),
    ),
)
def test_ehtim_movie_adapter_matches_legacy_full_polarization_sampling(polrep, fields):
    direct_generator = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    legacy_generator = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    direct_movie = _polarized_movie()
    legacy_movie = _polarized_movie()

    direct, direct_F0 = source_models.observe_source(
        direct_movie,
        _empty_observation(direct_generator).switch_polrep(polrep_out=polrep),
        direct_generator.source_context(),
    )
    legacy, legacy_F0 = _legacy_movie_observe(
        legacy_movie,
        _empty_observation(legacy_generator).switch_polrep(polrep_out=polrep),
        legacy_generator.source_context(),
    )

    assert direct.source == legacy.source
    assert direct.mjd == legacy.mjd
    assert direct.ampcal is legacy.ampcal is True
    assert direct.phasecal is legacy.phasecal is True
    assert direct.opacitycal is legacy.opacitycal is True
    assert direct.dcal is legacy.dcal is True
    assert direct.frcal is legacy.frcal is True
    assert direct_F0 == pytest.approx(legacy_F0)
    for field in fields:
        assert np.allclose(direct.data[field], legacy.data[field], atol=1.0e-12)


def test_direct_ehtim_image_sampling_preserves_seeded_corruptions(monkeypatch):
    settings = dict(COMPACT_OBS_SETTINGS)
    settings["fringe_finder"] = ["naive", 0.0]
    direct_generator = og.obs_generator(settings=settings)
    direct = direct_generator.make_obs(
        _polarized_model().make_image(160.0 * eh.RADPERUAS, 64),
        addnoise=True,
        addgains=True,
        addFR=True,
        addleakage=True,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    def legacy_adapter_observe(self, obs_empty, context, p=None):
        return _legacy_image_observe(self.input_model, obs_empty, context)

    monkeypatch.setattr(
        source_models.EhtimImageAdapter,
        "observe",
        legacy_adapter_observe,
    )
    legacy_generator = og.obs_generator(settings=settings)
    legacy = legacy_generator.make_obs(
        _polarized_model().make_image(160.0 * eh.RADPERUAS, 64),
        addnoise=True,
        addgains=True,
        addFR=True,
        addleakage=True,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    assert np.array_equal(direct.data["t1"], legacy.data["t1"])
    assert np.array_equal(direct.data["t2"], legacy.data["t2"])
    for field in (
        "time",
        "u",
        "v",
        "tau1",
        "tau2",
        "rrvis",
        "llvis",
        "rlvis",
        "lrvis",
        "rrsigma",
        "llsigma",
        "rlsigma",
        "lrsigma",
    ):
        assert np.allclose(direct.data[field], legacy.data[field], atol=1.0e-12)


def test_direct_ehtim_movie_sampling_preserves_seeded_corruptions(monkeypatch):
    settings = dict(COMPACT_OBS_SETTINGS)
    settings["fringe_finder"] = ["naive", 0.0]
    direct_generator = og.obs_generator(settings=settings)
    direct = direct_generator.make_obs(
        _polarized_movie(),
        addnoise=True,
        addgains=True,
        addFR=True,
        addleakage=True,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    def legacy_adapter_observe(self, obs_empty, context, p=None):
        return _legacy_movie_observe(self.input_model, obs_empty, context)

    monkeypatch.setattr(
        source_models.EhtimMovieAdapter,
        "observe",
        legacy_adapter_observe,
    )
    legacy_generator = og.obs_generator(settings=settings)
    legacy = legacy_generator.make_obs(
        _polarized_movie(),
        addnoise=True,
        addgains=True,
        addFR=True,
        addleakage=True,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    assert np.array_equal(direct.data["t1"], legacy.data["t1"])
    assert np.array_equal(direct.data["t2"], legacy.data["t2"])
    for field in (
        "time",
        "u",
        "v",
        "tau1",
        "tau2",
        "rrvis",
        "llvis",
        "rlvis",
        "lrvis",
        "rrsigma",
        "llsigma",
        "rlsigma",
        "lrsigma",
    ):
        assert np.allclose(direct.data[field], legacy.data[field], atol=1.0e-12)


def test_direct_ehtim_model_sampling_preserves_seeded_corruptions(monkeypatch):
    settings = dict(COMPACT_OBS_SETTINGS)
    settings["fringe_finder"] = ["naive", 0.0]
    direct_generator = og.obs_generator(settings=settings)
    direct = direct_generator.make_obs(
        _polarized_model(),
        addnoise=True,
        addgains=True,
        addFR=True,
        addleakage=True,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    def legacy_adapter_observe(self, obs_empty, context, p=None):
        return _legacy_model_observe(self.input_model, obs_empty, context)

    monkeypatch.setattr(
        source_models.EhtimModelAdapter,
        "observe",
        legacy_adapter_observe,
    )
    legacy_generator = og.obs_generator(settings=settings)
    legacy = legacy_generator.make_obs(
        _polarized_model(),
        addnoise=True,
        addgains=True,
        addFR=True,
        addleakage=True,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    assert np.array_equal(direct.data["t1"], legacy.data["t1"])
    assert np.array_equal(direct.data["t2"], legacy.data["t2"])
    for field in (
        "time",
        "u",
        "v",
        "tau1",
        "tau2",
        "rrvis",
        "llvis",
        "rlvis",
        "lrvis",
        "rrsigma",
        "llsigma",
        "rlsigma",
        "lrsigma",
    ):
        assert np.allclose(direct.data[field], legacy.data[field], atol=1.0e-12)


def test_obs_generator_still_accepts_ehtim_image():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    image = _compact_model().make_image(160.0 * eh.RADPERUAS, 64)

    obs = _observe_without_corruptions(obsgen, image)

    assert len(obs.data) > 0
