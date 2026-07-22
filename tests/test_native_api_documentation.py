"""Regression checks for the documented native public API."""

from __future__ import annotations

import inspect

from ngehtsim.obs.ehtfits import read_ehtfits, write_ehtfits
from ngehtsim.obs.fringe_selection import FringeRows, fpt_fringe_group_mask, fringe_group_mask, receptor_fringe_snr
from ngehtsim.obs.instrumental_corruptions import (
    apply_circular_corruptions,
    apply_circular_leakage,
    apply_receptor_corruptions,
    receptor_rows_for_station_terms,
)
from ngehtsim.obs.observation_geometry import (
    GroundGeometry,
    StationGeometry,
    apply_visibility_dataset_elevation_limits,
    ground_geometry,
    ground_visibility_template,
    visibility_dataset_elevation_mask,
)
from ngehtsim.obs.simulation_result import SimulationResult
from ngehtsim.obs.source_models import (
    EhtimImageAdapter,
    EhtimModelAdapter,
    EhtimMovieAdapter,
    adapter_for,
    observe_source,
    observe_source_dataset,
)
from ngehtsim.obs.station_observation import (
    station_metadata_for_dataset,
    station_terms_for_dataset,
    template_station_terms_for_dataset,
)
from ngehtsim.obs.station_effects import (
    GainModel,
    GainRatioModel,
    LeakageModel,
    RealizationCadence,
    StationCorruptionModel,
)
from ngehtsim.obs.observation_template import (
    ObservationTemplate,
    read_observation_template,
    simulate_observation_template,
)
from ngehtsim.obs.uvfits import read_uvfits, write_uvfits
from ngehtsim.obs.visibility_dataset import (
    CorrelationProductTable,
    ReceptorTable,
    StationTable,
    VisibilityDataset,
    standard_products_for_rows,
    receptor_products_for_rows,
)
from ngehtsim.obs.receptor_configuration import (
    ReceptorConfiguration,
    configuration_for_dataset,
    resolve_receptor_configuration,
)


DOCUMENTED_NATIVE_API = {
    "StationTable": StationTable,
    "ReceptorTable": ReceptorTable,
    "CorrelationProductTable": CorrelationProductTable,
    "standard_products_for_rows": standard_products_for_rows,
    "receptor_products_for_rows": receptor_products_for_rows,
    "ReceptorConfiguration": ReceptorConfiguration,
    "configuration_for_dataset": configuration_for_dataset,
    "resolve_receptor_configuration": resolve_receptor_configuration,
    "VisibilityDataset": VisibilityDataset,
    "read_ehtfits": read_ehtfits,
    "write_ehtfits": write_ehtfits,
    "read_uvfits": read_uvfits,
    "write_uvfits": write_uvfits,
    "SimulationResult": SimulationResult,
    "GroundGeometry": GroundGeometry,
    "StationGeometry": StationGeometry,
    "ground_geometry": ground_geometry,
    "ground_visibility_template": ground_visibility_template,
    "visibility_dataset_elevation_mask": visibility_dataset_elevation_mask,
    "apply_visibility_dataset_elevation_limits": apply_visibility_dataset_elevation_limits,
    "station_metadata_for_dataset": station_metadata_for_dataset,
    "station_terms_for_dataset": station_terms_for_dataset,
    "template_station_terms_for_dataset": template_station_terms_for_dataset,
    "ObservationTemplate": ObservationTemplate,
    "read_observation_template": read_observation_template,
    "simulate_observation_template": simulate_observation_template,
    "RealizationCadence": RealizationCadence,
    "GainModel": GainModel,
    "GainRatioModel": GainRatioModel,
    "LeakageModel": LeakageModel,
    "StationCorruptionModel": StationCorruptionModel,
    "apply_circular_leakage": apply_circular_leakage,
    "apply_circular_corruptions": apply_circular_corruptions,
    "apply_receptor_corruptions": apply_receptor_corruptions,
    "receptor_rows_for_station_terms": receptor_rows_for_station_terms,
    "FringeRows": FringeRows,
    "fringe_group_mask": fringe_group_mask,
    "fpt_fringe_group_mask": fpt_fringe_group_mask,
    "receptor_fringe_snr": receptor_fringe_snr,
    "EhtimImageAdapter": EhtimImageAdapter,
    "EhtimMovieAdapter": EhtimMovieAdapter,
    "EhtimModelAdapter": EhtimModelAdapter,
    "adapter_for": adapter_for,
    "observe_source": observe_source,
    "observe_source_dataset": observe_source_dataset,
}


def test_native_public_api_has_parameter_documentation():
    """Keep the documented native public surface from silently regressing."""

    missing = []
    for name, object_ in DOCUMENTED_NATIVE_API.items():
        docstring = inspect.getdoc(object_) or ""
        if "Parameters\n----------" not in docstring:
            missing.append(name)
    assert not missing, "Missing parameter documentation: {0}".format(", ".join(missing))
