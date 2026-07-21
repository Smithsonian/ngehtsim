"""Specification tests for ngehtsim fringe-group and FPT selection."""

from types import SimpleNamespace

import numpy as np
import pytest

import ngehtsim.obs.obs_generator as obs_module
from ngehtsim.obs.fringe_selection import FringeRows, fpt_fringe_group_mask, fringe_group_mask


def rows(station_pairs, snr, time=None, tint=None, rr_sigma=None, ll_sigma=None):
    """Build rows with requested aligned parallel-hand Stokes-I SNR values."""
    count = len(station_pairs)
    snr = np.broadcast_to(np.asarray(snr, dtype=float), (count,))
    time = np.zeros(count, dtype=float) if time is None else np.asarray(time, dtype=float)
    tint = np.full(count, 10.0) if tint is None else np.asarray(tint, dtype=float)
    rr_sigma = np.ones(count) if rr_sigma is None else np.asarray(rr_sigma, dtype=float)
    ll_sigma = np.ones(count) if ll_sigma is None else np.asarray(ll_sigma, dtype=float)
    amplitude = snr * 0.5 * np.hypot(rr_sigma, ll_sigma)
    return FringeRows(
        time=time,
        station1=np.array([pair[0] for pair in station_pairs]),
        station2=np.array([pair[1] for pair in station_pairs]),
        integration_time_s=tint,
        rr=amplitude.astype(complex),
        ll=amplitude.astype(complex),
        rr_sigma=rr_sigma,
        ll_sigma=ll_sigma,
    )


def test_stokes_i_snr_propagates_unequal_parallel_hand_uncertainties():
    selection_rows = FringeRows(
        time=np.array([0.0]),
        station1=np.array(["A"]),
        station2=np.array(["B"]),
        integration_time_s=np.array([10.0]),
        rr=np.array([10.0 + 0.0j]),
        ll=np.array([10.0 + 0.0j]),
        rr_sigma=np.array([3.0]),
        ll_sigma=np.array([4.0]),
    )

    assert np.allclose(selection_rows.stokes_i_snr, [4.0])


def test_fringe_groups_promote_weak_baselines_inside_a_transitive_component():
    selection_rows = rows(
        [("A", "B"), ("B", "C"), ("A", "C")],
        [5.0, 5.0, 0.1],
    )

    assert np.array_equal(
        fringe_group_mask(selection_rows, snr_threshold=5.0, tint_reference_s=10.0),
        [True, True, True],
    )


def test_fringe_groups_keep_disconnected_components_separate():
    selection_rows = rows(
        [("A", "B"), ("C", "D"), ("A", "C")],
        [5.0, 5.0, 0.1],
    )

    assert np.array_equal(
        fringe_group_mask(selection_rows, snr_threshold=5.0, tint_reference_s=10.0),
        [True, True, False],
    )


def test_fringe_groups_are_timestamp_local_and_scale_threshold_by_integration_time():
    selection_rows = rows(
        [("A", "B"), ("B", "C"), ("A", "C"), ("D", "E")],
        [5.0, 5.0, 0.1, 5.0],
        time=[0.0, 1.0, 0.0, 0.0],
        tint=[10.0, 10.0, 10.0, 40.0],
    )

    assert np.array_equal(
        fringe_group_mask(selection_rows, snr_threshold=5.0, tint_reference_s=10.0),
        [True, True, False, False],
    )


def test_fringe_groups_apply_station_availability_before_building_components():
    selection_rows = rows(
        [("A", "B"), ("B", "C"), ("A", "C")],
        [5.0, 5.0, 0.1],
    )

    assert np.array_equal(
        fringe_group_mask(
            selection_rows,
            snr_threshold=5.0,
            tint_reference_s=10.0,
            available_sites=("A", "C"),
        ),
        [False, False, False],
    )


def test_fringe_groups_are_invariant_to_row_order_and_baseline_orientation():
    original = rows(
        [("A", "B"), ("B", "C"), ("A", "C"), ("D", "E")],
        [5.0, 5.0, 0.1, 5.0],
    )
    order = np.array([3, 2, 0, 1])
    shuffled = FringeRows(
        time=original.time[order],
        station1=original.station2[order],
        station2=original.station1[order],
        integration_time_s=original.integration_time_s[order],
        rr=original.rr[order],
        ll=original.ll[order],
        rr_sigma=original.rr_sigma[order],
        ll_sigma=original.ll_sigma[order],
    )

    original_mask = fringe_group_mask(original, snr_threshold=5.0, tint_reference_s=10.0)
    shuffled_mask = fringe_group_mask(shuffled, snr_threshold=5.0, tint_reference_s=10.0)

    assert np.array_equal(shuffled_mask[np.argsort(order)], original_mask)


def test_empty_fringe_rows_are_valid_and_select_nothing():
    empty = rows([], [])

    assert np.array_equal(
        fringe_group_mask(empty, snr_threshold=5.0, tint_reference_s=10.0),
        np.zeros(0, dtype=bool),
    )


@pytest.mark.parametrize(
    ("reference_snr", "expected"),
    [(20.0, True), (19.9, False)],
)
def test_fpt_uses_the_reference_threshold_scaled_by_frequency_ratio(reference_snr, expected):
    target = rows([("A", "B")], [4.0])
    reference = rows([("A", "B")], [reference_snr])

    selected = fpt_fringe_group_mask(
        target,
        reference,
        reference_snr_threshold=20.0,
        tint_reference_s=10.0,
        reference_to_target_ratio=0.25,
    )

    assert np.array_equal(selected, [expected])


def test_fpt_combines_target_and_reference_strong_edges_in_one_station_graph():
    target = rows(
        [("A", "B"), ("B", "C"), ("A", "C")],
        [5.0, 0.1, 0.1],
    )
    reference = rows(
        [("C", "B"), ("D", "E")],
        [20.0, 20.0],
        time=[0.0, 2.0],
    )

    selected = fpt_fringe_group_mask(
        target,
        reference,
        reference_snr_threshold=20.0,
        tint_reference_s=10.0,
        reference_to_target_ratio=0.25,
    )

    assert np.array_equal(selected, [True, True, True])


def test_fpt_does_not_depend_on_matching_reference_row_order_or_count():
    target = rows(
        [("A", "B"), ("B", "C"), ("A", "C")],
        [0.1, 0.1, 0.1],
    )
    reference = rows(
        [("D", "E"), ("C", "B"), ("B", "A")],
        [20.0, 20.0, 20.0],
        time=[2.0, 0.0, 0.0],
    )

    selected = fpt_fringe_group_mask(
        target,
        reference,
        reference_snr_threshold=20.0,
        tint_reference_s=10.0,
        reference_to_target_ratio=0.25,
    )

    assert np.array_equal(selected, [True, True, True])


def test_fpt_respects_target_and_reference_station_availability_independently():
    target = rows([("A", "B")], [0.1])
    reference = rows([("A", "B")], [20.0])

    target_unavailable = fpt_fringe_group_mask(
        target,
        reference,
        reference_snr_threshold=20.0,
        tint_reference_s=10.0,
        reference_to_target_ratio=0.25,
        target_available_sites=("A",),
        reference_available_sites=("A", "B"),
    )
    reference_unavailable = fpt_fringe_group_mask(
        target,
        reference,
        reference_snr_threshold=20.0,
        tint_reference_s=10.0,
        reference_to_target_ratio=0.25,
        target_available_sites=("A", "B"),
        reference_available_sites=("A",),
    )

    assert np.array_equal(target_unavailable, [False])
    assert np.array_equal(reference_unavailable, [False])


def test_fpt_excludes_flagged_rows_from_target_and_reference_graphs():
    target = rows(
        [("A", "B"), ("B", "C"), ("A", "C")],
        [0.1, 0.1, 0.1],
    )
    reference = rows(
        [("A", "B"), ("B", "C")],
        [20.0, 20.0],
    )

    reference_flagged = fpt_fringe_group_mask(
        target,
        reference,
        reference_snr_threshold=20.0,
        tint_reference_s=10.0,
        reference_to_target_ratio=0.25,
        reference_row_available=[True, False],
    )
    target_flagged = fpt_fringe_group_mask(
        target,
        reference,
        reference_snr_threshold=20.0,
        tint_reference_s=10.0,
        reference_to_target_ratio=0.25,
        target_row_available=[True, False, True],
    )

    assert np.array_equal(reference_flagged, [True, False, False])
    assert np.array_equal(target_flagged, [True, False, True])


def test_fpt_can_select_a_native_target_baseline_without_reference_rows():
    target = rows([("A", "B")], [5.0])
    reference = rows([], [])

    assert np.array_equal(
        fpt_fringe_group_mask(
            target,
            reference,
            reference_snr_threshold=20.0,
            tint_reference_s=10.0,
            reference_to_target_ratio=0.25,
        ),
        [True],
    )


def obsdata(selection_rows):
    """Create the narrow Obsdata-like surface consumed by legacy selection."""
    dtype = [
        ("time", float), ("tint", float), ("t1", "U8"), ("t2", "U8"),
        ("rrvis", complex), ("llvis", complex), ("rrsigma", float), ("llsigma", float),
    ]
    data = np.empty(selection_rows.row_count, dtype=dtype)
    data["time"] = selection_rows.time
    data["tint"] = selection_rows.integration_time_s
    data["t1"] = selection_rows.station1
    data["t2"] = selection_rows.station2
    data["rrvis"] = selection_rows.rr
    data["llvis"] = selection_rows.ll
    data["rrsigma"] = selection_rows.rr_sigma
    data["llsigma"] = selection_rows.ll_sigma

    class FakeObsdata:
        def __init__(self, values):
            self.data = values

        def switch_polrep(self, polrep_out):
            assert polrep_out == "circ"
            return self

    return FakeObsdata(data)


def test_legacy_fringegroups_uses_the_shared_order_independent_selector():
    selection_rows = rows(
        [("A", "B"), ("B", "C"), ("A", "C")],
        [5.0, 5.0, 0.1],
    )

    class Generator:
        sites = ("A", "B", "C")
        bands = {"A": "band", "B": "band", "C": "band"}

    assert np.array_equal(
        obs_module.fringegroups(Generator(), obsdata(selection_rows), 5.0, 10.0),
        [True, True, True],
    )


def test_native_fringegroups_uses_parallel_hand_weights_and_shared_selector():
    selection_rows = rows(
        [("A", "B"), ("B", "C"), ("A", "C")],
        [5.0, 5.0, 0.1],
        rr_sigma=[3.0, 3.0, 3.0],
        ll_sigma=[4.0, 4.0, 4.0],
    )
    dataset = SimpleNamespace(
        channel_count=1,
        row_count=selection_rows.row_count,
        stations=SimpleNamespace(names=("A", "B", "C")),
        time_mjd=selection_rows.time,
        integration_time_s=selection_rows.integration_time_s,
        antenna1=np.array([0, 1, 0]),
        antenna2=np.array([1, 2, 2]),
        visibilities=np.column_stack((
            selection_rows.rr,
            selection_rows.ll,
            np.zeros(selection_rows.row_count, dtype=complex),
            np.zeros(selection_rows.row_count, dtype=complex),
        ))[:, np.newaxis, :],
        sigma_jy=np.column_stack((
            selection_rows.rr_sigma,
            selection_rows.ll_sigma,
            np.ones(selection_rows.row_count),
            np.ones(selection_rows.row_count),
        ))[:, np.newaxis, :],
        circular_product_slots=lambda: np.tile(
            np.arange(4, dtype=np.intp),
            (selection_rows.row_count, 1),
        ),
    )

    class Generator:
        sites = ("A", "B", "C")
        bands = {"A": "band", "B": "band", "C": "band"}

    assert np.array_equal(
        obs_module.fringegroups_dataset(Generator(), dataset, 5.0, 10.0),
        [True, True, True],
    )


def test_fpt_wrapper_accepts_independent_target_and_reference_row_sets(monkeypatch):
    target = obsdata(rows(
        [("A", "B"), ("B", "C"), ("A", "C")],
        [5.0, 0.1, 0.1],
    ))
    reference = obsdata(rows(
        [("D", "E"), ("C", "B")],
        [20.0, 20.0],
        time=[2.0, 0.0],
    ))

    class Generator:
        settings = {"frequency": 345.0, "bandwidth": 2.0}
        freq = 345.0e9
        seed = 1
        weather = "mean"
        D_overrides = {}
        surf_rms_overrides = {}
        receiver_configuration_overrides = {}
        bandwidth_overrides = {}
        T_R_overrides = {}
        sideband_ratio_overrides = {}
        lo_freq_overrides = {}
        hi_freq_overrides = {}
        ap_eff_overrides = {}
        wind_loading_overrides = {}
        custom_receivers = {}
        station_uptimes = {}
        weather_store = None
        weather_cadence = "daily"
        sites = ("A", "B", "C", "D", "E")
        bands = {site: "band" for site in sites}
        im = None

    reference_generator = Generator()
    reference_generator.observe_legacy = lambda input_model, **kwargs: reference
    monkeypatch.setattr(obs_module, "obs_generator", lambda *args, **kwargs: reference_generator)

    selected = obs_module.FPT(
        Generator(),
        target,
        snr_ref=20.0,
        tint_ref=10.0,
        freq_ref=86.25,
    )

    assert np.array_equal(selected, [True, True, True])

    assert np.array_equal(
        obs_module.FPT(
            Generator(),
            target,
            snr_ref=20.0,
            tint_ref=10.0,
            freq_ref=86.25,
            unready_sites=("B",),
        ),
        [False, False, False],
    )
