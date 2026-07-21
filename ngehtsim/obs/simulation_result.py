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

    Parameters
    ----------
    dataset : VisibilityDataset
        Native simulated visibility data, including rows or individual samples
        excluded by station availability or fringe selection.
    station_terms : mapping of str to object
        Row-aligned station-model products used for this realization, such as
        opacity, SEFD, gains, leakage, and availability information. Arrays
        are copied and made read-only; this mapping is simulation provenance,
        not yet a stable archive schema.
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
        """Rows retained after station and fringe-selection flagging.

        Returns
        -------
        numpy.ndarray of bool, shape (row,)
            ``True`` only when no populated sample in that row is flagged.
        """

        return ~np.any(
            self.dataset.flags & self.dataset.sample_present,
            axis=(1, 2),
        )

    @property
    def unflagged_dataset(self):
        """Return rows whose populated samples are all unflagged.

        Returns
        -------
        VisibilityDataset
            Native subset suitable for an ehtim export only when its channel
            and correlation layout also meet ehtim's constraints.
        """

        return self.dataset.select_rows(self.row_mask)

    def to_ehtim_obsdata(self):
        """Export unflagged circular single-channel data to ``ehtim.Obsdata``.

        The empty-result case is handled only here to preserve the optional
        ehtim boundary; an empty :class:`VisibilityDataset` remains valid.

        Returns
        -------
        ehtim.obsdata.Obsdata
            Circular single-channel boundary object, possibly with zero rows.

        Raises
        ------
        ValueError
            If the unflagged dataset is not representable by ehtim.
        """

        unflagged = self.unflagged_dataset
        if unflagged.row_count:
            return unflagged.to_ehtim_obsdata()

        # ehtim refuses to construct an Obsdata from zero rows. Construct a
        # valid temporary export and then expose its representable empty view.
        temporary_flags = np.array(self.dataset.flags, copy=True)
        temporary_flags[self.dataset.sample_present] = False
        temporary = replace(self.dataset, flags=temporary_flags)
        obs = temporary.to_ehtim_obsdata()
        obs.data = obs.data[:0]
        return obs
