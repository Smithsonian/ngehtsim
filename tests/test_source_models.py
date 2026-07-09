#######################################################
# imports

import pytest
import ehtim as eh
import ngehtsim.obs.obs_generator as og
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


def _observe_without_corruptions(obsgen, input_model):
    return obsgen.observe(
        input_model,
        addnoise=False,
        addgains=False,
        flagwind=False,
        flagday=False,
        flagsun=False,
    )

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


def test_obs_generator_still_accepts_ehtim_image():
    obsgen = og.obs_generator(settings=COMPACT_OBS_SETTINGS)
    image = _compact_model().make_image(160.0 * eh.RADPERUAS, 64)

    obs = _observe_without_corruptions(obsgen, image)

    assert len(obs.data) > 0
