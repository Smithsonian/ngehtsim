"""Internal visibility data structures and ``ehtim.Obsdata`` compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from astropy.constants import c as SPEED_OF_LIGHT


CIRCULAR_CORRELATIONS = ("RR", "LL", "RL", "LR")
LINEAR_CORRELATIONS = ("XX", "YY", "XY", "YX")
LINEAR_CIRCULAR_CORRELATIONS = ("XR", "XL", "YR", "YL")
CIRCULAR_LINEAR_CORRELATIONS = ("RX", "RY", "LX", "LY")
_VALID_CORRELATION_PRODUCTS = frozenset(
    first + second for first in "RLXY" for second in "RLXY"
)


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
class VisibilityDataset:
    """Visibility data supporting multiple channels and per-row correlation layouts.

    ``visibilities``, ``weights``, and ``flags`` have shape
    ``(row, channel, correlation)``. Every row has four correlation slots; the
    corresponding labels are selected from ``correlation_layouts`` by
    ``row_layout_id``. This supports circular, linear, and mixed-feed baselines
    without a global polarization representation.
    """

    stations: StationTable
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
    correlation_layouts: tuple[tuple[str, str, str, str], ...]
    row_layout_id: np.ndarray
    visibilities: np.ndarray
    weights: np.ndarray
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
        row_layout_id = _integer_array(self.row_layout_id, "row_layout_id")
        visibilities = _readonly_array(self.visibilities, dtype=complex)
        weights = _readonly_array(self.weights, dtype=float)
        flags = _readonly_array(self.flags, dtype=bool)

        row_count = len(time_mjd)
        channel_count = len(channel_frequency_hz)
        for name, values in (
            ("integration_time_s", integration_time_s),
            ("antenna1", antenna1),
            ("antenna2", antenna2),
            ("tau1", tau1),
            ("tau2", tau2),
            ("row_layout_id", row_layout_id),
        ):
            _require_shape(values, (row_count,), name)
        _require_shape(uvw_m, (row_count, 3), "uvw_m")
        _require_shape(channel_bandwidth_hz, (channel_count,), "channel_bandwidth_hz")
        _require_shape(spectral_window_id, (channel_count,), "spectral_window_id")
        _require_shape(visibilities, (row_count, channel_count, 4), "visibilities")
        _require_shape(weights, (row_count, channel_count, 4), "weights")
        _require_shape(flags, (row_count, channel_count, 4), "flags")

        for name, values in (
            ("time_mjd", time_mjd),
            ("integration_time_s", integration_time_s),
            ("uvw_m", uvw_m),
            ("tau1", tau1),
            ("tau2", tau2),
            ("channel_frequency_hz", channel_frequency_hz),
            ("channel_bandwidth_hz", channel_bandwidth_hz),
            ("weights", weights),
        ):
            _require_finite(values, name)
        unflagged = ~flags
        if not np.all(np.isfinite(visibilities[unflagged])):
            raise ValueError("Unflagged visibilities must be finite.")
        if np.any(integration_time_s <= 0.0):
            raise ValueError("integration_time_s must be positive.")
        if np.any(tau1 < 0.0) or np.any(tau2 < 0.0):
            raise ValueError("tau1 and tau2 must be non-negative.")
        if np.any(channel_frequency_hz <= 0.0) or np.any(channel_bandwidth_hz <= 0.0):
            raise ValueError("Channel frequencies and bandwidths must be positive.")
        if np.any(weights < 0.0) or np.any(weights[unflagged] == 0.0):
            raise ValueError("Unflagged visibility weights must be positive.")
        station_count = len(self.stations.names)
        if np.any(antenna1 < 0) or np.any(antenna1 >= station_count):
            raise ValueError("antenna1 contains an out-of-range station index.")
        if np.any(antenna2 < 0) or np.any(antenna2 >= station_count):
            raise ValueError("antenna2 contains an out-of-range station index.")
        if np.any(antenna1 == antenna2):
            raise ValueError("Visibility rows must use two distinct stations.")

        layouts = tuple(tuple(str(product) for product in layout) for layout in self.correlation_layouts)
        if not layouts:
            raise ValueError("At least one correlation layout is required.")
        if len(set(layouts)) != len(layouts):
            raise ValueError("Correlation layouts must be unique.")
        for layout in layouts:
            if len(layout) != 4 or len(set(layout)) != 4:
                raise ValueError("Every correlation layout must contain four distinct products.")
            if any(product not in _VALID_CORRELATION_PRODUCTS for product in layout):
                raise ValueError("Correlation layouts contain an unsupported product.")
        if np.any(row_layout_id < 0) or np.any(row_layout_id >= len(layouts)):
            raise ValueError("row_layout_id contains an out-of-range layout index.")

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
        object.__setattr__(self, "correlation_layouts", layouts)
        object.__setattr__(self, "row_layout_id", row_layout_id)
        object.__setattr__(self, "visibilities", visibilities)
        object.__setattr__(self, "weights", weights)
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
            row_layout_id=self.row_layout_id[row_indices],
            visibilities=self.visibilities[row_indices],
            weights=self.weights[row_indices],
            flags=self.flags[row_indices],
        )

    @classmethod
    def from_ehtim_obsdata(cls, obs):
        """Convert a standard ``ehtim.Obsdata`` object to an internal dataset."""

        import ehtim as eh

        if not isinstance(obs, eh.obsdata.Obsdata):
            raise TypeError("obs must be an ehtim.Obsdata instance.")
        working = obs
        if working.timetype != "UTC":
            working = working.switch_timetype("UTC")
        if working.polrep != "circ":
            working = working.switch_polrep("circ")

        sigma = np.stack(
            (working.data["rrsigma"], working.data["llsigma"], working.data["rlsigma"], working.data["lrsigma"]),
            axis=1,
        )
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
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
        wavelength = SPEED_OF_LIGHT.to_value("m / s") / working.rf
        return cls(
            stations=StationTable.from_ehtim_tarr(working.tarr),
            time_mjd=working.mjd + (working.data["time"] / 24.0),
            integration_time_s=working.data["tint"],
            antenna1=np.fromiter(
                (station_index[str(name)] for name in working.data["t1"]),
                dtype=np.intp,
            ),
            antenna2=np.fromiter(
                (station_index[str(name)] for name in working.data["t2"]),
                dtype=np.intp,
            ),
            uvw_m=np.column_stack((working.data["u"] * wavelength, working.data["v"] * wavelength, np.zeros(len(working.data)))),
            tau1=working.data["tau1"],
            tau2=working.data["tau2"],
            channel_frequency_hz=np.array((working.rf,)),
            channel_bandwidth_hz=np.array((working.bw,)),
            spectral_window_id=np.array((0,), dtype=np.intp),
            correlation_layouts=(CIRCULAR_CORRELATIONS,),
            row_layout_id=np.zeros(len(working.data), dtype=np.intp),
            visibilities=np.stack(
                (working.data["rrvis"], working.data["llvis"], working.data["rlvis"], working.data["lrvis"]),
                axis=1,
            )[:, np.newaxis, :],
            weights=(1.0 / np.square(sigma))[:, np.newaxis, :],
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
        if any(self.correlation_layouts[index] != CIRCULAR_CORRELATIONS for index in self.row_layout_id):
            raise ValueError("ehtim Obsdata output requires circular RR, LL, RL, LR correlations.")
        if np.any(self.flags):
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
        data["rrvis"] = self.visibilities[:, 0, 0]
        data["llvis"] = self.visibilities[:, 0, 1]
        data["rlvis"] = self.visibilities[:, 0, 2]
        data["lrvis"] = self.visibilities[:, 0, 3]
        data["rrsigma"] = 1.0 / np.sqrt(self.weights[:, 0, 0])
        data["llsigma"] = 1.0 / np.sqrt(self.weights[:, 0, 1])
        data["rlsigma"] = 1.0 / np.sqrt(self.weights[:, 0, 2])
        data["lrsigma"] = 1.0 / np.sqrt(self.weights[:, 0, 3])

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
