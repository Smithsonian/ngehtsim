#######################################################
# imports

import pytest
import ehtim as eh
import numpy as np
import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.observation_geometry as observation_geometry
import ngehtsim.obs.source_models as source_models
import ngehtsim.obs.raster_sampling as raster_sampling
import ngehtsim.obs.station_observation as station_observation
from ngehtsim.obs.visibility_dataset import VisibilityDataset
from ngehtsim.const_def import default_settings

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


def _position_angle_image():
    """Build an odd-sized image that exercises raster-coordinate conventions."""

    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 35), np.linspace(-1.0, 1.0, 33))
    intensity = np.exp(-3.0 * ((x - 0.2) ** 2 + (y + 0.3) ** 2))
    image = eh.image.Image(
        intensity,
        8.0 * eh.RADPERUAS,
        0.0,
        0.0,
        pa=0.37,
        rf=230.0e9,
        source="M87",
        mjd=57849,
    )
    image.add_pol_image(0.15 * intensity * (1.0 + x), "Q")
    image.add_pol_image(-0.10 * intensity * (1.0 - y), "U")
    image.add_pol_image(0.03 * intensity, "V")
    return image


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
        ttype="direct",
    )
    F0 = input_model.total_flux()
    return obs, F0


def _legacy_movie_observe(input_model, obs_empty, context):
    source_models._set_ehtim_metadata(input_model, context)
    obs = input_model.observe_same_nonoise(
        obs_empty,
        ttype="direct",
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


def test_default_raster_transform_backend_is_auto():
    assert default_settings["transform_backend"] == "auto"
    assert default_settings["raster_tolerance"] == pytest.approx(1.0e-12)
    settings = og.obs_generator(settings=COMPACT_OBS_SETTINGS).settings
    assert settings["transform_backend"] == "auto"
    assert settings["raster_tolerance"] == pytest.approx(1.0e-12)


@pytest.mark.parametrize(
    ("removed_setting", "value"),
    (("ttype", "nfft"), ("fft_pad_factor", 2)),
)
def test_removed_v1_raster_settings_are_rejected(removed_setting, value):
    """Avoid silently accepting settings that no longer affect v2 sampling."""

    with pytest.raises(Exception, match="is not a recognized setting"):
        og.obs_generator(settings={**COMPACT_OBS_SETTINGS, removed_setting: value})


@pytest.mark.parametrize("backend", ("direct", "finufft"))
@pytest.mark.parametrize("polrep", ("circ", "stokes"))
def test_native_raster_sampler_matches_ehtim_direct_with_position_angle(backend, polrep):
    """Preserve ehtim's pulse, centring, position-angle, and polarization rules."""

    image = _position_angle_image()
    uv = np.array(
        (
            (0.0, 0.0),
            (1.4e10, -2.1e10),
            (-3.2e10, 0.8e10),
            (5.6e10, 4.4e10),
        )
    )

    expected = image.sample_uv(uv, polrep_obs=polrep, ttype="direct")
    sampled = raster_sampling.sample_ehtim_raster(
        image,
        uv,
        polrep_obs=polrep,
        backend=backend,
        tolerance=1.0e-12,
    )

    for actual, reference in zip(sampled, expected):
        assert np.allclose(actual, reference, rtol=1.0e-10, atol=1.0e-11)


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


@pytest.mark.parametrize(
    "source_factory",
    (
        lambda: _polarized_model().make_image(160.0 * eh.RADPERUAS, 64),
        _polarized_movie,
    ),
)
def test_native_finufft_raster_sampling_matches_direct(source_factory):
    direct_settings = {**COMPACT_OBS_SETTINGS, "transform_backend": "direct"}
    finufft_settings = {**COMPACT_OBS_SETTINGS, "transform_backend": "finufft"}
    direct_generator = og.obs_generator(settings=direct_settings)
    finufft_generator = og.obs_generator(settings=finufft_settings)
    template = observation_geometry.ground_visibility_template(
        direct_generator.arr,
        direct_generator.geometry_context(),
    )

    direct, direct_F0 = source_models.observe_source_dataset(
        source_factory(),
        template,
        direct_generator.source_context(),
    )
    finufft, finufft_F0 = source_models.observe_source_dataset(
        source_factory(),
        template,
        finufft_generator.source_context(),
    )

    assert finufft_F0 == pytest.approx(direct_F0)
    direct_obs = direct.to_ehtim_obsdata()
    finufft_obs = finufft.to_ehtim_obsdata()
    for field in ("rrvis", "llvis", "rlvis", "lrvis"):
        assert np.allclose(
            finufft_obs.data[field],
            direct_obs.data[field],
            rtol=1.0e-10,
            atol=1.0e-11,
        )


@pytest.mark.parametrize(
    "source_factory",
    (
        lambda: _polarized_model().make_image(160.0 * eh.RADPERUAS, 64),
        _polarized_movie,
    ),
)
def test_ehtim_raster_adapter_rejects_removed_backend_names(source_factory):
    obsgen = og.obs_generator(
        settings={**COMPACT_OBS_SETTINGS, "transform_backend": "nfft"}
    )

    with pytest.raises(ValueError, match="transform_backend='nfft'.*auto.*direct.*finufft"):
        source_models.observe_source(
            source_factory(),
            _empty_observation(obsgen),
            obsgen.source_context(),
        )


def test_native_finufft_accepts_odd_image_dimensions():
    finufft_generator = og.obs_generator(
        settings={**COMPACT_OBS_SETTINGS, "transform_backend": "finufft"}
    )
    odd_image = _compact_model().make_image(160.0 * eh.RADPERUAS, 63)
    template = observation_geometry.ground_visibility_template(
        finufft_generator.arr,
        finufft_generator.geometry_context(),
    )

    observation, F0 = source_models.observe_source_dataset(
        odd_image,
        template,
        finufft_generator.source_context(),
    )

    assert observation.row_count > 0
    assert F0 > 0.0


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


@pytest.mark.parametrize(
    "source_factory",
    (
        lambda: _polarized_model().make_image(160.0 * eh.RADPERUAS, 64),
        _polarized_movie,
    ),
)
def test_native_raster_sampler_never_calls_ehtim_sample_uv(monkeypatch, source_factory):
    """Native raster simulation must remain independent of ehtim's pyNFFT path."""

    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    template = observation_geometry.ground_visibility_template(
        obsgen.arr,
        obsgen.geometry_context(),
    )

    def unexpected_ehtim_sampler(*args, **kwargs):
        raise AssertionError("Native raster sampling must not call ehtim.Image.sample_uv().")

    monkeypatch.setattr(eh.image.Image, "sample_uv", unexpected_ehtim_sampler)
    sampled, F0 = source_models.observe_source_dataset(
        source_factory(),
        template,
        obsgen.source_context(),
    )

    assert sampled.row_count == template.row_count
    assert F0 > 0.0


def test_ehtim_model_adapter_samples_native_visibility_dataset():
    direct_generator = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    legacy_generator = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    direct_model = _polarized_model()
    legacy_model = _polarized_model()
    template = observation_geometry.ground_visibility_template(
        direct_generator.arr,
        direct_generator.geometry_context(),
    )

    direct, direct_F0 = source_models.observe_source_dataset(
        direct_model,
        template,
        direct_generator.source_context(),
    )
    legacy, legacy_F0 = _legacy_model_observe(
        legacy_model,
        template.to_ehtim_obsdata(),
        legacy_generator.source_context(),
    )

    assert direct_F0 == pytest.approx(legacy_F0)
    adapted = direct.to_ehtim_obsdata()
    for field in ("rrvis", "llvis", "rlvis", "lrvis"):
        assert np.allclose(adapted.data[field], legacy.data[field], atol=1.0e-12)


def test_ehtim_model_dataset_sampler_avoids_obsdata_conversion(monkeypatch):
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    template = observation_geometry.ground_visibility_template(
        obsgen.arr,
        obsgen.geometry_context(),
    )

    def unexpected_obsdata_conversion(*args, **kwargs):
        raise AssertionError("Native model sampling must not construct an Obsdata object.")

    monkeypatch.setattr(
        VisibilityDataset,
        "to_ehtim_obsdata",
        unexpected_obsdata_conversion,
    )
    sampled, F0 = source_models.observe_source_dataset(
        _polarized_model(),
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


def test_ehtim_movie_adapter_samples_native_visibility_dataset():
    direct_generator = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    legacy_generator = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    direct_movie = _polarized_movie()
    legacy_movie = _polarized_movie()
    template = VisibilityDataset.from_ehtim_obsdata(
        _empty_observation(direct_generator)
    )

    direct, direct_F0 = source_models.observe_source_dataset(
        direct_movie,
        template,
        direct_generator.source_context(),
    )
    legacy, legacy_F0 = source_models.observe_source(
        legacy_movie,
        _empty_observation(legacy_generator),
        legacy_generator.source_context(),
    )

    assert direct_F0 == pytest.approx(legacy_F0)
    adapted = direct.to_ehtim_obsdata()
    for field in ("rrvis", "llvis", "rlvis", "lrvis"):
        assert np.allclose(adapted.data[field], legacy.data[field], atol=1.0e-12)


def test_ehtim_movie_dataset_sampler_avoids_obsdata_conversion(monkeypatch):
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    template = observation_geometry.ground_visibility_template(
        obsgen.arr,
        obsgen.geometry_context(),
    )

    def unexpected_obsdata_conversion(*args, **kwargs):
        raise AssertionError("Native movie sampling must not construct an Obsdata object.")

    monkeypatch.setattr(
        VisibilityDataset,
        "to_ehtim_obsdata",
        unexpected_obsdata_conversion,
    )
    sampled, F0 = source_models.observe_source_dataset(
        _polarized_movie(),
        template,
        obsgen.source_context(),
    )

    assert sampled.row_count == template.row_count
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


@pytest.mark.parametrize(
    "source_factory",
    (
        lambda: _polarized_model().make_image(160.0 * eh.RADPERUAS, 64),
        _polarized_model,
        _polarized_movie,
    ),
)
def test_obs_generator_routes_supported_sources_through_native_path(monkeypatch, source_factory):
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)

    def unexpected_legacy_path(*args, **kwargs):
        raise AssertionError("Supported ground-array sources must use the native path.")

    monkeypatch.setattr(
        observation_geometry,
        "observation_template",
        unexpected_legacy_path,
    )
    monkeypatch.setattr(source_models, "observe_source", unexpected_legacy_path)
    monkeypatch.setattr(station_observation, "station_terms", unexpected_legacy_path)
    obs = obsgen.observe(
        source_factory(),
        addnoise=False,
        addgains=False,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    assert len(obs.data) > 0


@pytest.mark.parametrize(
    "source_factory",
    (
        lambda: _polarized_model().make_image(160.0 * eh.RADPERUAS, 64),
        _polarized_model,
        _polarized_movie,
    ),
)
def test_native_simulate_does_not_construct_obsdata(monkeypatch, source_factory):
    def unexpected_obsdata_conversion(*args, **kwargs):
        raise AssertionError("simulate() must not construct an ehtim Obsdata object.")

    monkeypatch.setattr(
        VisibilityDataset,
        "to_ehtim_obsdata",
        unexpected_obsdata_conversion,
    )
    result = og.obs_generator(settings=COMPACT_OBS_SETTINGS).simulate(
        source_factory(),
        addnoise=False,
        addgains=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    assert isinstance(result.dataset, VisibilityDataset)


@pytest.mark.parametrize(
    "source_factory",
    (
        lambda: _polarized_model().make_image(160.0 * eh.RADPERUAS, 64),
        _polarized_model,
        _polarized_movie,
    ),
)
def test_native_fpt_selection_stays_native_and_uses_one_readiness_draw(monkeypatch, source_factory):
    settings = dict(COMPACT_OBS_SETTINGS)
    settings["fringe_finder"] = ["fpt", [0.0, 10.0, 86.0, None]]
    readiness_calls = []

    def unready_sites(sites, tech_readiness, rng):
        readiness_calls.append((tuple(sites), tech_readiness))
        return np.array(["ALMA"])

    def unexpected_obsdata_conversion(*args, **kwargs):
        raise AssertionError("Native FPT selection must not construct an ehtim Obsdata object.")

    monkeypatch.setattr(og, "get_unready_sites", unready_sites)
    monkeypatch.setattr(VisibilityDataset, "to_ehtim_obsdata", unexpected_obsdata_conversion)
    source = source_factory()
    original = (source.ra, source.dec, source.mjd, source.source, source.rf)
    result = og.obs_generator(settings=settings).make_dataset(
        source,
        addnoise=False,
        addgains=False,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    names = np.asarray(result.dataset.stations.names)
    touches_alma = (
        (names[result.dataset.antenna1] == "ALMA")
        | (names[result.dataset.antenna2] == "ALMA")
    )
    assert isinstance(result.dataset, VisibilityDataset)
    assert len(readiness_calls) == 1
    assert np.all(result.dataset.flags[touches_alma])
    assert (source.ra, source.dec, source.mjd, source.source, source.rf) == original


def test_native_fpt_make_obs_exports_only_after_selection():
    settings = dict(COMPACT_OBS_SETTINGS)
    settings["fringe_finder"] = ["fpt", [0.0, 10.0, 86.0, None]]

    obs = og.obs_generator(settings=settings).make_obs(
        _compact_model(),
        addnoise=False,
        addgains=False,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    assert len(obs.data) > 0


def test_native_simulation_keeps_terms_in_the_result_not_the_generator():
    settings = dict(COMPACT_OBS_SETTINGS)
    settings["fringe_finder"] = ["naive", 0.0]
    generator = og.obs_generator(settings=settings, weight=1)
    result = generator.make_dataset(
        _polarized_model(),
        addnoise=False,
        addgains=True,
        addFR=True,
        addleakage=True,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    for name in (
        "Tsys1",
        "Tsys2",
        "tau1",
        "tau2",
        "Tb1",
        "Tb2",
        "SEFD1",
        "SEFD2",
        "gainamp1R",
        "gainamp2R",
        "gainamp1L",
        "gainamp2L",
        "gainphase1R",
        "gainphase2R",
        "gainphase1L",
        "gainphase2L",
        "leak1R",
        "leak2R",
        "leak1L",
        "leak2L",
    ):
        assert name in result.station_terms
        assert len(result.station_terms[name]) == result.dataset.row_count
    assert not result.station_terms["tau1"].flags.writeable
    assert not hasattr(generator, "timestamps")
    assert not hasattr(generator, "SEFD1")


def test_native_simulation_is_reproducible_with_a_fixed_seed():
    settings = dict(COMPACT_OBS_SETTINGS)
    settings["fringe_finder"] = ["naive", 0.0]
    first = og.obs_generator(settings=settings).make_dataset(
        _polarized_model(), addnoise=True, addgains=True, addFR=True,
        addleakage=True, flagwind=False, flagday=False, flagsun=False,
    )
    second = og.obs_generator(settings=settings).make_dataset(
        _polarized_model(), addnoise=True, addgains=True, addFR=True,
        addleakage=True, flagwind=False, flagday=False, flagsun=False,
    )

    assert np.array_equal(first.dataset.time_mjd, second.dataset.time_mjd)
    assert np.array_equal(first.dataset.flags, second.dataset.flags)
    assert np.allclose(first.dataset.visibilities, second.dataset.visibilities)
    assert np.allclose(first.dataset.sigma_jy, second.dataset.sigma_jy)


def test_native_simulation_retains_all_flagged_rows_and_exports_an_empty_obsdata():
    station_uptimes = {
        "ALMA": (8.0, 9.0),
        "APEX": (8.0, 9.0),
        "LMT": (8.0, 9.0),
        "SMT": (8.0, 9.0),
    }
    obsgen = og.obs_generator(
        settings=COMPACT_OBS_SETTINGS,
        station_uptimes=station_uptimes,
    )

    result = obsgen.simulate(
        _compact_model(),
        addnoise=False,
        addgains=False,
        addFR=False,
        addleakage=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    assert result.dataset.row_count > 0
    assert not np.any(result.row_mask)
    obs = result.to_ehtim_obsdata()
    assert len(obs.data) == 0
    assert obs.ampcal is True
    assert obs.phasecal is True
    assert obs.opacitycal is True
    assert obs.dcal is True
    assert obs.frcal is True


@pytest.mark.parametrize(
    "source_factory",
    (
        lambda: _polarized_model().make_image(160.0 * eh.RADPERUAS, 64),
        _polarized_model,
        _polarized_movie,
    ),
)
def test_native_source_adapters_do_not_mutate_input_metadata(source_factory):
    source = source_factory()
    original = (source.ra, source.dec, source.mjd, source.source, source.rf)

    og.obs_generator(settings=COMPACT_OBS_SETTINGS).simulate(
        source,
        addnoise=False,
        addgains=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

    assert (source.ra, source.dec, source.mjd, source.source, source.rf) == original


def test_make_obs_exports_updated_station_terms_without_mutating_configuration():
    settings = dict(COMPACT_OBS_SETTINGS)
    settings["fringe_finder"] = ["naive", 0.0]
    generator = og.obs_generator(settings=settings)
    configured_tarr = generator.arr.tarr.copy()

    result = generator.make_dataset(
        _compact_model(),
        addnoise=False,
        addgains=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )
    obs = result.to_ehtim_obsdata()

    assert np.array_equal(generator.arr.tarr, configured_tarr)
    expected_sefd = dict(zip(
        result.dataset.stations.names,
        result.dataset.stations.sefd_r_jy,
    ))
    assert all(
        obs.tarr["sefdr"][index] == pytest.approx(expected_sefd[site])
        for index, site in enumerate(obs.tarr["site"])
    )


def test_obs_generator_still_accepts_ehtim_image():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    image = _compact_model().make_image(160.0 * eh.RADPERUAS, 64)

    obs = _observe_without_corruptions(obsgen, image)

    assert len(obs.data) > 0
