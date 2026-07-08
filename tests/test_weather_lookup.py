#######################################################
# imports

import numpy as np
import ngehtsim.weather.weather as nw

#######################################################
# make sure the month naming/number convention works

def test_integer_month_matches_named_month():
    kwargs = dict(site='ALMA', form='exact', day=11, year=2017)
    assert np.isclose(
        nw.opacity(month=4, freq=230.0, **kwargs),
        nw.opacity(month='Apr', freq=230.0, **kwargs),
    )
    assert np.isclose(
        nw.brightness_temperature(month='04', freq=230.0, **kwargs),
        nw.brightness_temperature(month='Apr', freq=230.0, **kwargs),
    )
