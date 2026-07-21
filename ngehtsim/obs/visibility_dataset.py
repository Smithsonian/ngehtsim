"""Native visibility data structures and optional ``ehtim.Obsdata`` adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from astropy.constants import c as SPEED_OF_LIGHT


CIRCULAR_PRODUCT_LABELS = ("RR", "LL", "RL", "LR")
LINEAR_PRODUCT_LABELS = ("XX", "YY", "XY", "YX")


def _readonly_array(values, dtype=None):
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _require_shape(values, shape, name):
    if values.shape != shape:
        raise ValueError("{0} must have shape {1}, received {2}.".format(name, shape, values.shape))


def _require_finite(values, name):
    if not np.all(np.isfinite(values)):
        raise ValueError("{0} must contain only finite values.".format(name))


def _integer_array(values, name):
    """Return an immutable native-index array without silently truncating values."""

    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("{0} must contain integers.".format(name))
    return _readonly_array(array, dtype=np.intp)


@dataclass(frozen=True)
class StationTable:
    """Station metadata independent of an ``ehtim`` telescope table."""

    names: tuple[str, ...]
    position_itrs_m: np.ndarray
    sefd_r_jy: np.ndarray
    sefd_l_jy: np.ndarray
    leakage_r: np.ndarray
    leakage_l: np.ndarray
    feed_rotation_par: np.ndarray
    feed_rotation_elev: np.ndarray
    feed_rotation_offset_deg: np.ndarray

    def __post_init__(self):
        names = tuple(str(name) for name in self.names)
        if not names:
            raise ValueError("StationTable requires at least one station.")
        if len(set(names)) != len(names):
            raise ValueError("StationTable station names must be unique.")

        count = len(names)
        position_itrs_m = _readonly_array(self.position_itrs_m, dtype=float)
        sefd_r_jy = _readonly_array(self.sefd_r_jy, dtype=float)
        sefd_l_jy = _readonly_array(self.sefd_l_jy, dtype=float)
        leakage_r = _readonly_array(self.leakage_r, dtype=complex)
        leakage_l = _readonly_array(self.leakage_l, dtype=complex)
        feed_rotation_par = _readonly_array(self.feed_rotation_par, dtype=float)
        feed_rotation_elev = _readonly_array(self.feed_rotation_elev, dtype=float)
        feed_rotation_offset_deg = _readonly_array(self.feed_rotation_offset_deg, dtype=float)

        _require_shape(position_itrs_m, (count, 3), "position_itrs_m")
        for name, values in (
            ("sefd_r_jy", sefd_r_jy),
            ("sefd_l_jy", sefd_l_jy),
            ("leakage_r", leakage_r),
            ("leakage_l", leakage_l),
            ("feed_rotation_par", feed_rotation_par),
            ("feed_rotation_elev", feed_rotation_elev),
            ("feed_rotation_offset_deg", feed_rotation_offset_deg),
        ):
            _require_shape(values, (count,), name)
            _require_finite(values, name)
        _require_finite(position_itrs_m, "position_itrs_m")
        if np.any(sefd_r_jy < 0.0) or np.any(sefd_l_jy < 0.0):
            raise ValueError("StationTable SEFD values must be non-negative.")

        object.__setattr__(self, "names", names)
        object.__setattr__(self, "position_itrs_m", position_itrs_m)
        object.__setattr__(self, "sefd_r_jy", sefd_r_jy)
        object.__setattr__(self, "sefd_l_jy", sefd_l_jy)
        object.__setattr__(self, "leakage_r", leakage_r)
        object.__setattr__(self, "leakage_l", leakage_l)
        object.__setattr__(self, "feed_rotation_par", feed_rotation_par)
        object.__setattr__(self, "feed_rotation_elev", feed_rotation_elev)
        object.__setattr__(self, "feed_rotation_offset_deg", feed_rotation_offset_deg)

    @classmethod
    def from_ehtim_tarr(cls, tarr):
        """Create a station table from an ``ehtim`` telescope array."""

        return cls(
            names=tuple(str(name) for name in tarr["site"]),
            position_itrs_m=np.column_stack((tarr["x"], tarr["y"], tarr["z"])),
            sefd_r_jy=tarr["sefdr"],
            sefd_l_jy=tarr["sefdl"],
            leakage_r=tarr["dr"],
            leakage_l=tarr["dl"],
            feed_rotation_par=tarr["fr_par"],
            feed_rotation_elev=tarr["fr_elev"],
            feed_rotation_offset_deg=tarr["fr_off"],
        )

    def to_ehtim_tarr(self):
        """Convert station metadata to an ``ehtim`` telescope array."""

        import ehtim as eh

        tarr = np.empty(len(self.names), dtype=eh.const_def.DTARR)
        tarr["site"] = self.names
        tarr["x"] = self.position_itrs_m[:, 0]
        tarr["y"] = self.position_itrs_m[:, 1]
        tarr["z"] = self.position_itrs_m[:, 2]
        tarr["sefdr"] = self.sefd_r_jy
        tarr["sefdl"] = self.sefd_l_jy
        tarr["dr"] = self.leakage_r
        tarr["dl"] = self.leakage_l
        tarr["fr_par"] = self.feed_rotation_par
        tarr["fr_elev"] = self.feed_rotation_elev
        tarr["fr_off"] = self.feed_rotation_offset_deg
        return tarr


@dataclass(frozen=True)
class ReceptorTable:
    """Station-local receiver signal paths used by a visibility dataset.

    A receptor is one voltage stream, not a two-polarization feed assembly.
    ``receptor_id`` is the zero-based index of a row in this table. The
    ``feed_id`` and ``polarization_label`` fields are descriptive rather than
    inferred calibration instructions, so non-standard receiver arrangements
    remain representable.
    """

    station_index: np.ndarray
    feed_id: tuple[str, ...]
    polarization_label: tuple[str, ...]
    basis: tuple[str, ...]

    def __post_init__(self):
        station_index = _integer_array(self.station_index, "station_index")
        count = len(station_index)
        feed_id = tuple(str(value) for value in self.feed_id)
        polarization_label = tuple(str(value) for value in self.polarization_label)
        basis = tuple(str(value) for value in self.basis)
        if not count:
            raise ValueError("ReceptorTable requires at least one receptor.")
        if len(feed_id) != count or len(polarization_label) != count or len(basis) != count:
            raise ValueError("ReceptorTable fields must contain one value per receptor.")
        if np.any(station_index < 0):
            raise ValueError("ReceptorTable station_index values must be non-negative.")
        if any(not value for value in feed_id):
            raise ValueError("ReceptorTable feed_id values must be non-empty.")
        if any(not value for value in polarization_label):
            raise ValueError("ReceptorTable polarization_label values must be non-empty.")
        if any(not value for value in basis):
            raise ValueError("ReceptorTable basis values must be non-empty.")
        station_feed_pairs = tuple(zip(station_index.tolist(), feed_id))
        if len(set(station_feed_pairs)) != count:
            raise ValueError("ReceptorTable station/feed pairs must be unique.")

        object.__setattr__(self, "station_index", station_index)
        object.__setattr__(self, "feed_id", feed_id)
        object.__setattr__(self, "polarization_label", polarization_label)
        object.__setattr__(self, "basis", basis)

    @property
    def count(self):
        """Return the number of station-local receptor signal paths."""

        return len(self.station_index)

    @classmethod
    def from_station_labels(cls, station_count, labels, basis):
        """Create identical labelled receptors for every station.

        This is a convenience constructor for standard all-circular or
        all-linear arrays. More general arrays should construct a table
        directly with station-local ``feed_id`` values.
        """

        if not isinstance(station_count, (int, np.integer)) or station_count <= 0:
            raise ValueError("station_count must be a positive integer.")
        labels = tuple(str(value) for value in labels)
        if not labels or len(set(labels)) != len(labels) or any(not value for value in labels):
            raise ValueError("labels must contain distinct non-empty values.")
        return cls(
            station_index=np.repeat(np.arange(station_count, dtype=np.intp), len(labels)),
            feed_id=labels * station_count,
            polarization_label=labels * station_count,
            basis=(str(basis),) * (station_count * len(labels)),
        )

    def index_for(self, station_index, polarization_label):
        """Return the unique receptor index for a station and label."""

        matches = np.flatnonzero(
            (self.station_index == station_index)
            & (np.asarray(self.polarization_label, dtype=object) == str(polarization_label))
        )
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one {0!r} receptor at station index {1}.".format(
                    polarization_label,
                    station_index,
                )
            )
        return int(matches[0])


@dataclass(frozen=True)
class CorrelationProductTable:
    """Ordered pairs of receptor IDs that define stored correlations."""

    receptor1_id: np.ndarray
    receptor2_id: np.ndarray

    def __post_init__(self):
        receptor1_id = _integer_array(self.receptor1_id, "receptor1_id")
        receptor2_id = _integer_array(self.receptor2_id, "receptor2_id")
        if receptor1_id.ndim != 1 or receptor2_id.ndim != 1 or receptor1_id.shape != receptor2_id.shape:
            raise ValueError("Correlation product receptor IDs must be matching one-dimensional arrays.")
        if np.any(receptor1_id < 0) or np.any(receptor2_id < 0):
            raise ValueError("Correlation product receptor IDs must be non-negative.")
        pairs = tuple(zip(receptor1_id.tolist(), receptor2_id.tolist()))
        if len(set(pairs)) != len(pairs):
            raise ValueError("Correlation product receptor pairs must be unique.")
        object.__setattr__(self, "receptor1_id", receptor1_id)
        object.__setattr__(self, "receptor2_id", receptor2_id)

    @property
    def count(self):
        """Return the number of distinct correlation-product definitions."""

        return len(self.receptor1_id)


def standard_products_for_rows(receptors, antenna1, antenna2, product_labels):
    """Build explicit receptor-pair products for standard labelled rows.

    ``product_labels`` supplies ordered two-character correlation labels, such
    as ``("RR", "LL", "RL", "LR")``. The returned product table contains
    each unique station/receptor pair once, while ``row_product_id`` maps every
    input row to the relevant ordered product IDs.
    """

    if not isinstance(receptors, ReceptorTable):
        raise TypeError("receptors must be a ReceptorTable.")
    antenna1 = _integer_array(antenna1, "antenna1")
    antenna2 = _integer_array(antenna2, "antenna2")
    if antenna1.ndim != 1 or antenna2.shape != antenna1.shape:
        raise ValueError("antenna1 and antenna2 must be matching one-dimensional arrays.")
    product_labels = tuple(str(value) for value in product_labels)
    if not product_labels or any(len(value) != 2 for value in product_labels):
        raise ValueError("product_labels must contain non-empty two-label products.")

    product_id_by_pair = {}
    first_receptor = []
    second_receptor = []
    row_product_id = np.empty((len(antenna1), len(product_labels)), dtype=np.intp)
    for row, (station1, station2) in enumerate(zip(antenna1, antenna2)):
        for slot, label in enumerate(product_labels):
            receptor1 = receptors.index_for(int(station1), label[0])
            receptor2 = receptors.index_for(int(station2), label[1])
            pair = (receptor1, receptor2)
            product_id = product_id_by_pair.get(pair)
            if product_id is None:
                product_id = len(first_receptor)
                product_id_by_pair[pair] = product_id
                first_receptor.append(receptor1)
                second_receptor.append(receptor2)
            row_product_id[row, slot] = product_id
    return (
        CorrelationProductTable(
            receptor1_id=np.asarray(first_receptor, dtype=np.intp),
            receptor2_id=np.asarray(second_receptor, dtype=np.intp),
        ),
        row_product_id,
    )


@dataclass(frozen=True)
class VisibilityDataset:
    """Native visibility data with explicit station-receptor correlations.

    ``visibilities``, ``sigma_jy``, and ``flags`` have shape ``(row, channel,
    product_slot)``. ``row_product_id`` maps each populated product slot to an
    ordered receptor pair in ``correlation_products``. A value of ``-1`` marks
    padding introduced by the dense in-memory representation; padding is
    always flagged and is never written to a FITS-EHT archive.
    """

    stations: StationTable
    receptors: ReceptorTable
    correlation_products: CorrelationProductTable
    time_mjd: np.ndarray
    integration_time_s: np.ndarray
    antenna1: np.ndarray
    antenna2: np.ndarray
    uvw_m: np.ndarray
    tau1: np.ndarray
    tau2: np.ndarray
    channel_frequency_hz: np.ndarray
    channel_bandwidth_hz: np.ndarray
    spectral_window_id: np.ndarray
    row_product_id: np.ndarray
    visibilities: np.ndarray
    sigma_jy: np.ndarray
    flags: np.ndarray
    source: str
    ra_hours: float
    dec_degrees: float
    ampcal: bool = True
    phasecal: bool = True
    opacitycal: bool = True
    dcal: bool = True
    frcal: bool = True
    scan_start_mjd: np.ndarray | None = None
    scan_stop_mjd: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.stations, StationTable):
            raise TypeError("stations must be a StationTable.")
        if not isinstance(self.receptors, ReceptorTable):
            raise TypeError("receptors must be a ReceptorTable.")
        if not isinstance(self.correlation_products, CorrelationProductTable):
            raise TypeError("correlation_products must be a CorrelationProductTable.")

        time_mjd = _readonly_array(self.time_mjd, dtype=float)
        integration_time_s = _readonly_array(self.integration_time_s, dtype=float)
        antenna1 = _integer_array(self.antenna1, "antenna1")
        antenna2 = _integer_array(self.antenna2, "antenna2")
        uvw_m = _readonly_array(self.uvw_m, dtype=float)
        tau1 = _readonly_array(self.tau1, dtype=float)
        tau2 = _readonly_array(self.tau2, dtype=float)
        channel_frequency_hz = _readonly_array(self.channel_frequency_hz, dtype=float)
        channel_bandwidth_hz = _readonly_array(self.channel_bandwidth_hz, dtype=float)
        spectral_window_id = _integer_array(self.spectral_window_id, "spectral_window_id")
        row_product_id = _integer_array(self.row_product_id, "row_product_id")
        visibilities = _readonly_array(self.visibilities, dtype=complex)
        sigma_jy = _readonly_array(self.sigma_jy, dtype=float)
        flags = _readonly_array(self.flags, dtype=bool)

        row_count = len(time_mjd)
        channel_count = len(channel_frequency_hz)
        if row_product_id.ndim != 2 or row_product_id.shape[0] != row_count:
            raise ValueError("row_product_id must have shape (row, product_slot).")
        product_slot_count = row_product_id.shape[1]
        if row_count and product_slot_count == 0:
            raise ValueError("Visibility rows require at least one product slot.")
        for name, values in (
            ("integration_time_s", integration_time_s),
            ("antenna1", antenna1),
            ("antenna2", antenna2),
            ("tau1", tau1),
            ("tau2", tau2),
        ):
            _require_shape(values, (row_count,), name)
        _require_shape(uvw_m, (row_count, 3), "uvw_m")
        _require_shape(channel_bandwidth_hz, (channel_count,), "channel_bandwidth_hz")
        _require_shape(spectral_window_id, (channel_count,), "spectral_window_id")
        data_shape = (row_count, channel_count, product_slot_count)
        _require_shape(visibilities, data_shape, "visibilities")
        _require_shape(sigma_jy, data_shape, "sigma_jy")
        _require_shape(flags, data_shape, "flags")

        for name, values in (
            ("time_mjd", time_mjd),
            ("integration_time_s", integration_time_s),
            ("uvw_m", uvw_m),
            ("tau1", tau1),
            ("tau2", tau2),
            ("channel_frequency_hz", channel_frequency_hz),
            ("channel_bandwidth_hz", channel_bandwidth_hz),
        ):
            _require_finite(values, name)
        if np.any(integration_time_s <= 0.0):
            raise ValueError("integration_time_s must be positive.")
        if np.any(tau1 < 0.0) or np.any(tau2 < 0.0):
            raise ValueError("tau1 and tau2 must be non-negative.")
        if np.any(channel_frequency_hz <= 0.0) or np.any(channel_bandwidth_hz <= 0.0):
            raise ValueError("Channel frequencies and bandwidths must be positive.")

        station_count = len(self.stations.names)
        if np.any(antenna1 < 0) or np.any(antenna1 >= station_count):
            raise ValueError("antenna1 contains an out-of-range station index.")
        if np.any(antenna2 < 0) or np.any(antenna2 >= station_count):
            raise ValueError("antenna2 contains an out-of-range station index.")
        if np.any(antenna1 == antenna2):
            raise ValueError("Visibility rows must use two distinct stations.")
        if np.any(self.receptors.station_index >= station_count):
            raise ValueError("ReceptorTable references an out-of-range station index.")

        product_count = self.correlation_products.count
        if np.any(row_product_id < -1) or np.any(row_product_id >= product_count):
            raise ValueError("row_product_id contains an out-of-range product index.")
        if product_count and (
            np.any(self.correlation_products.receptor1_id >= self.receptors.count)
            or np.any(self.correlation_products.receptor2_id >= self.receptors.count)
        ):
            raise ValueError("Correlation products reference an out-of-range receptor index.")

        populated_slots = row_product_id >= 0
        if row_count and np.any(np.sum(populated_slots, axis=1) == 0):
            raise ValueError("Every visibility row must contain at least one correlation product.")
        for row, product_ids in enumerate(row_product_id):
            product_ids = product_ids[product_ids >= 0]
            if len(set(product_ids.tolist())) != len(product_ids):
                raise ValueError("A visibility row cannot repeat a correlation product.")
            if len(product_ids):
                product_receptors1 = self.correlation_products.receptor1_id[product_ids]
                product_receptors2 = self.correlation_products.receptor2_id[product_ids]
                if not np.all(self.receptors.station_index[product_receptors1] == antenna1[row]):
                    raise ValueError("Correlation product first receptors must belong to antenna1.")
                if not np.all(self.receptors.station_index[product_receptors2] == antenna2[row]):
                    raise ValueError("Correlation product second receptors must belong to antenna2.")

        sample_present = np.broadcast_to(populated_slots[:, np.newaxis, :], data_shape)
        if np.any(~flags[~sample_present]):
            raise ValueError("Unused product slots must be flagged.")
        if not np.all(np.isnan(sigma_jy[~sample_present])):
            raise ValueError("Unused product slots must have NaN sigma_jy values.")
        unflagged = sample_present & ~flags
        unflagged_sigma = sigma_jy[unflagged]
        if not np.all(np.isfinite(unflagged_sigma)) or np.any(unflagged_sigma <= 0.0):
            raise ValueError("Unflagged visibility sigma_jy values must be finite and positive.")
        flagged_sigma = sigma_jy[sample_present & flags]
        if np.any(np.isinf(flagged_sigma)) or np.any(flagged_sigma <= 0.0):
            raise ValueError("Flagged visibility sigma_jy values must be positive or NaN.")
        if not np.all(np.isfinite(visibilities[unflagged])):
            raise ValueError("Unflagged visibilities must be finite.")

        scan_start_mjd, scan_stop_mjd = _validate_scans(
            self.scan_start_mjd,
            self.scan_stop_mjd,
        )
        for name, value in (("ra_hours", self.ra_hours), ("dec_degrees", self.dec_degrees)):
            if not np.isfinite(value):
                raise ValueError("{0} must be finite.".format(name))

        object.__setattr__(self, "time_mjd", time_mjd)
        object.__setattr__(self, "integration_time_s", integration_time_s)
        object.__setattr__(self, "antenna1", antenna1)
        object.__setattr__(self, "antenna2", antenna2)
        object.__setattr__(self, "uvw_m", uvw_m)
        object.__setattr__(self, "tau1", tau1)
        object.__setattr__(self, "tau2", tau2)
        object.__setattr__(self, "channel_frequency_hz", channel_frequency_hz)
        object.__setattr__(self, "channel_bandwidth_hz", channel_bandwidth_hz)
        object.__setattr__(self, "spectral_window_id", spectral_window_id)
        object.__setattr__(self, "row_product_id", row_product_id)
        object.__setattr__(self, "visibilities", visibilities)
        object.__setattr__(self, "sigma_jy", sigma_jy)
        object.__setattr__(self, "flags", flags)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "ra_hours", float(self.ra_hours))
        object.__setattr__(self, "dec_degrees", float(self.dec_degrees))
        object.__setattr__(self, "scan_start_mjd", scan_start_mjd)
        object.__setattr__(self, "scan_stop_mjd", scan_stop_mjd)

    @property
    def row_count(self):
        """Return the number of baseline-time rows."""

        return len(self.time_mjd)

    @property
    def channel_count(self):
        """Return the number of spectral channels."""

        return len(self.channel_frequency_hz)

    @property
    def product_slot_count(self):
        """Return the dense in-memory product-axis width."""

        return self.row_product_id.shape[1]

    @property
    def sample_present(self):
        """Return a mask for populated, non-padding visibility samples."""

        return np.broadcast_to(
            self.row_product_id[:, np.newaxis, :] >= 0,
            self.visibilities.shape,
        )

    def product_slots(self, product_labels):
        """Return per-row product slots for an exact labelled product set.

        This is intentionally strict. Boundary adapters that require standard
        circular or linear data must reject extra, missing, or ambiguous
        receptor products instead of silently dropping or relabelling them.
        """

        product_labels = tuple(str(value) for value in product_labels)
        if not product_labels or len(set(product_labels)) != len(product_labels):
            raise ValueError("product_labels must contain distinct labels.")
        product_names = np.asarray(
            tuple(
                self.receptors.polarization_label[first]
                + self.receptors.polarization_label[second]
                for first, second in zip(
                    self.correlation_products.receptor1_id,
                    self.correlation_products.receptor2_id,
                )
            ),
            dtype=object,
        )
        output = np.empty((self.row_count, len(product_labels)), dtype=np.intp)
        expected = set(product_labels)
        for row, product_ids in enumerate(self.row_product_id):
            slots = np.flatnonzero(product_ids >= 0)
            labels = tuple(product_names[product_ids[slots]])
            if len(labels) != len(product_labels) or set(labels) != expected:
                raise ValueError(
                    "Dataset rows do not contain exactly the required correlation products: {0}.".format(
                        ", ".join(product_labels)
                    )
                )
            for destination, label in enumerate(product_labels):
                output[row, destination] = slots[labels.index(label)]
        return output

    def circular_product_slots(self):
        """Return slots holding RR, LL, RL, and LR for every row."""

        return self._standard_product_slots(CIRCULAR_PRODUCT_LABELS, "CIRCULAR")

    def linear_product_slots(self):
        """Return slots holding XX, YY, XY, and YX for every row."""

        return self._standard_product_slots(LINEAR_PRODUCT_LABELS, "LINEAR")

    def _standard_product_slots(self, product_labels, expected_basis):
        slots = self.product_slots(product_labels)
        for row, row_slots in enumerate(slots):
            product_ids = self.row_product_id[row, row_slots]
            receptor_ids = np.concatenate((
                self.correlation_products.receptor1_id[product_ids],
                self.correlation_products.receptor2_id[product_ids],
            ))
            bases = np.asarray(self.receptors.basis, dtype=object)[receptor_ids]
            if not np.all(np.char.upper(bases.astype(str)) == expected_basis):
                raise ValueError(
                    "Standard {0} product export requires {0} receptor basis metadata.".format(
                        expected_basis.lower()
                    )
                )
        return slots

    def select_rows(self, row_mask):
        """Return a dataset containing the selected visibility rows."""

        row_mask = np.asarray(row_mask)
        if row_mask.dtype != bool or row_mask.shape != (self.row_count,):
            raise ValueError(
                "row_mask must be a boolean array with one value per visibility row."
            )
        return self.take_rows(np.flatnonzero(row_mask))

    def take_rows(self, row_indices):
        """Return a dataset containing rows selected in the given order."""

        row_indices = np.asarray(row_indices)
        if row_indices.ndim != 1 or not np.issubdtype(row_indices.dtype, np.integer):
            raise ValueError("row_indices must be a one-dimensional integer array.")
        if np.any(row_indices < 0) or np.any(row_indices >= self.row_count):
            raise ValueError("row_indices contains an out-of-range visibility row.")
        return replace(
            self,
            time_mjd=self.time_mjd[row_indices],
            integration_time_s=self.integration_time_s[row_indices],
            antenna1=self.antenna1[row_indices],
            antenna2=self.antenna2[row_indices],
            uvw_m=self.uvw_m[row_indices],
            tau1=self.tau1[row_indices],
            tau2=self.tau2[row_indices],
            row_product_id=self.row_product_id[row_indices],
            visibilities=self.visibilities[row_indices],
            sigma_jy=self.sigma_jy[row_indices],
            flags=self.flags[row_indices],
        )

    @classmethod
    def from_ehtim_obsdata(cls, obs):
        """Convert a standard ``ehtim.Obsdata`` object to a native dataset."""

        import ehtim as eh

        if not isinstance(obs, eh.obsdata.Obsdata):
            raise TypeError("obs must be an ehtim.Obsdata instance.")
        working = obs
        if working.timetype != "UTC":
            working = working.switch_timetype("UTC")
        if working.polrep != "circ":
            working = working.switch_polrep("circ")

        sigma_jy = np.stack(
            (working.data["rrsigma"], working.data["llsigma"], working.data["rlsigma"], working.data["lrsigma"]),
            axis=1,
        )
        if not np.all(np.isfinite(sigma_jy)) or np.any(sigma_jy <= 0.0):
            raise ValueError("ehtim Obsdata sigma values must be finite and positive.")
        scan_start_mjd = None
        scan_stop_mjd = None
        if working.scans is not None:
            scans = np.asarray(working.scans, dtype=float)
            if scans.ndim != 2 or scans.shape[1] != 2:
                raise ValueError("ehtim Obsdata scan table must have shape (scan, 2).")
            scan_start_mjd = working.mjd + (scans[:, 0] / 24.0)
            scan_stop_mjd = working.mjd + (scans[:, 1] / 24.0)

        station_index = {
            str(name): index for index, name in enumerate(working.tarr["site"])
        }
        antenna1 = np.fromiter(
            (station_index[str(name)] for name in working.data["t1"]),
            dtype=np.intp,
        )
        antenna2 = np.fromiter(
            (station_index[str(name)] for name in working.data["t2"]),
            dtype=np.intp,
        )
        receptors = ReceptorTable.from_station_labels(
            len(working.tarr),
            ("R", "L"),
            "CIRCULAR",
        )
        correlation_products, row_product_id = standard_products_for_rows(
            receptors,
            antenna1,
            antenna2,
            CIRCULAR_PRODUCT_LABELS,
        )
        wavelength = SPEED_OF_LIGHT.to_value("m / s") / working.rf
        return cls(
            stations=StationTable.from_ehtim_tarr(working.tarr),
            receptors=receptors,
            correlation_products=correlation_products,
            time_mjd=working.mjd + (working.data["time"] / 24.0),
            integration_time_s=working.data["tint"],
            antenna1=antenna1,
            antenna2=antenna2,
            uvw_m=np.column_stack((working.data["u"] * wavelength, working.data["v"] * wavelength, np.zeros(len(working.data)))),
            tau1=working.data["tau1"],
            tau2=working.data["tau2"],
            channel_frequency_hz=np.array((working.rf,)),
            channel_bandwidth_hz=np.array((working.bw,)),
            spectral_window_id=np.array((0,), dtype=np.intp),
            row_product_id=row_product_id,
            visibilities=np.stack(
                (working.data["rrvis"], working.data["llvis"], working.data["rlvis"], working.data["lrvis"]),
                axis=1,
            )[:, np.newaxis, :],
            sigma_jy=sigma_jy[:, np.newaxis, :],
            flags=np.zeros((len(working.data), 1, 4), dtype=bool),
            source=working.source,
            ra_hours=working.ra,
            dec_degrees=working.dec,
            ampcal=working.ampcal,
            phasecal=working.phasecal,
            opacitycal=working.opacitycal,
            dcal=working.dcal,
            frcal=working.frcal,
            scan_start_mjd=scan_start_mjd,
            scan_stop_mjd=scan_stop_mjd,
        )

    def to_ehtim_obsdata(self):
        """Convert a representable circular single-channel dataset to ``ehtim.Obsdata``."""

        if self.channel_count != 1:
            raise ValueError("ehtim Obsdata output requires exactly one spectral channel.")
        circular_slots = self.circular_product_slots()
        row_index = np.arange(self.row_count)[:, np.newaxis]
        if np.any(self.flags[row_index, 0, circular_slots]):
            raise ValueError("ehtim Obsdata output cannot represent flagged visibility samples.")
        if not self.row_count:
            raise ValueError("ehtim Obsdata output requires at least one visibility row.")

        reference_mjd = int(np.floor(np.min(self.time_mjd)))

        import ehtim as eh

        data = np.zeros(self.row_count, dtype=eh.const_def.DTPOL_CIRC)
        data["time"] = (self.time_mjd - reference_mjd) * 24.0
        data["tint"] = self.integration_time_s
        data["t1"] = tuple(self.stations.names[index] for index in self.antenna1)
        data["t2"] = tuple(self.stations.names[index] for index in self.antenna2)
        data["tau1"] = self.tau1
        data["tau2"] = self.tau2
        wavelength = SPEED_OF_LIGHT.to_value("m / s") / self.channel_frequency_hz[0]
        data["u"] = self.uvw_m[:, 0] / wavelength
        data["v"] = self.uvw_m[:, 1] / wavelength
        circular = self.visibilities[row_index, 0, circular_slots]
        sigma_jy = self.sigma_jy[row_index, 0, circular_slots]
        data["rrvis"] = circular[:, 0]
        data["llvis"] = circular[:, 1]
        data["rlvis"] = circular[:, 2]
        data["lrvis"] = circular[:, 3]
        data["rrsigma"] = sigma_jy[:, 0]
        data["llsigma"] = sigma_jy[:, 1]
        data["rlsigma"] = sigma_jy[:, 2]
        data["lrsigma"] = sigma_jy[:, 3]

        scantable = None
        if self.scan_start_mjd is not None:
            scantable = np.column_stack((
                (self.scan_start_mjd - reference_mjd) * 24.0,
                (self.scan_stop_mjd - reference_mjd) * 24.0,
            ))
        return eh.obsdata.Obsdata(
            self.ra_hours,
            self.dec_degrees,
            self.channel_frequency_hz[0],
            self.channel_bandwidth_hz[0],
            data,
            self.stations.to_ehtim_tarr(),
            scantable=scantable,
            polrep="circ",
            source=self.source,
            mjd=reference_mjd,
            timetype="UTC",
            ampcal=self.ampcal,
            phasecal=self.phasecal,
            opacitycal=self.opacitycal,
            dcal=self.dcal,
            frcal=self.frcal,
        )

    @classmethod
    def from_uvfits(cls, path):
        """Read a native multi-channel UVFITS dataset without using ``ehtim``."""

        from ngehtsim.obs.uvfits import read_uvfits

        return read_uvfits(path)

    def to_uvfits(self, path, overwrite=False):
        """Write this dataset through the native UVFITS adapter."""

        from ngehtsim.obs.uvfits import write_uvfits

        return write_uvfits(self, path, overwrite=overwrite)

    @classmethod
    def from_ehtfits(cls, path):
        """Read a lossless native FITS-EHT visibility dataset."""

        from ngehtsim.obs.ehtfits import read_ehtfits

        return read_ehtfits(path)

    def to_ehtfits(self, path, overwrite=False):
        """Write this dataset as a lossless native FITS-EHT archive."""

        from ngehtsim.obs.ehtfits import write_ehtfits

        return write_ehtfits(self, path, overwrite=overwrite)


def _validate_scans(scan_start_mjd, scan_stop_mjd):
    if scan_start_mjd is None and scan_stop_mjd is None:
        return None, None
    if scan_start_mjd is None or scan_stop_mjd is None:
        raise ValueError("scan_start_mjd and scan_stop_mjd must be provided together.")
    starts = _readonly_array(scan_start_mjd, dtype=float)
    stops = _readonly_array(scan_stop_mjd, dtype=float)
    if starts.ndim != 1 or stops.ndim != 1 or starts.shape != stops.shape:
        raise ValueError("Scan start and stop arrays must be matching one-dimensional arrays.")
    _require_finite(starts, "scan_start_mjd")
    _require_finite(stops, "scan_stop_mjd")
    if np.any(stops < starts):
        raise ValueError("Scan stop times must not precede scan start times.")
    return starts, stops
