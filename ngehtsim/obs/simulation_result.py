"""Native observation simulation results and optional ehtim export."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ngehtsim.obs.visibility_dataset import VisibilityDataset


@dataclass(frozen=True)
class SimulationResult:
    """A native synthetic observation and its station-based simulation terms.

    ``dataset`` retains flagged rows. Consumers that need an ehtim-compatible
    view can use :meth:`to_ehtim_obsdata`, which drops those rows at the export
    boundary because ``ehtim.Obsdata`` has no sample-flag representation.
    """

    dataset: VisibilityDataset
    station_terms: Mapping[str, Any]

    def __post_init__(self):
        if not isinstance(self.dataset, VisibilityDataset):
            raise TypeError("dataset must be a VisibilityDataset instance.")
        station_terms = {}
        for name, value in self.station_terms.items():
            if isinstance(value, np.ndarray):
                value = np.array(value, copy=True)
                value.setflags(write=False)
            elif isinstance(value, list):
                value = tuple(value)
            station_terms[name] = value
        object.__setattr__(self, "station_terms", MappingProxyType(station_terms))

    @property
    def row_mask(self):
        """Rows retained after station and fringe-selection flagging."""

        return ~np.any(self.dataset.flags, axis=(1, 2))

    @property
    def unflagged_dataset(self):
        """Return the visibility rows representable by ``ehtim.Obsdata``."""

        return self.dataset.select_rows(self.row_mask)

    def to_ehtim_obsdata(self):
        """Export unflagged circular single-channel data to ``ehtim.Obsdata``.

        The empty-result case is handled only here to preserve the optional
        ehtim boundary; an empty :class:`VisibilityDataset` remains valid.
        """

        unflagged = self.unflagged_dataset
        if unflagged.row_count:
            return unflagged.to_ehtim_obsdata()

        # ehtim refuses to construct an Obsdata from zero rows. Construct a
        # valid temporary export and then expose its representable empty view.
        temporary = replace(self.dataset, flags=np.zeros_like(self.dataset.flags))
        obs = temporary.to_ehtim_obsdata()
        obs.data = obs.data[:0]
        return obs
