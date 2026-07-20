#######################################################
# imports

import pytest
import ehtim as eh
import numpy as np
import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.observation_geometry as observation_geometry
import ngehtsim.obs.source_models as source_models

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

#######################################################
# tests


def test_adapter_for_ehtim_model():
    adapter = source_models.adapter_for(_compact_model())
    assert isinstance(adapter, source_models.EhtimModelAdapter)


def test_adapter_for_ehtim_image():
    image = _compact_model().make_image(160.0 * eh.RADPERUAS, 64)

    adapter = source_models.adapter_for(image)

    assert isinstance(adapter, source_models.EhtimImageAdapter)


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
