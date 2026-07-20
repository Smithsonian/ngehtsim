"""Pure circular-basis visibility corruption kernels."""

from __future__ import annotations

import numpy as np


def apply_circular_leakage(visibilities, leakage1_r, leakage1_l, leakage2_r,
                           leakage2_l):
    """Apply ``D_1 V D_2^H`` to circular RR, LL, RL, LR visibilities.

    The final axis of ``visibilities`` must use the order ``RR, LL, RL, LR``.
    The result is a new array so every output correlation is calculated from
    the same, unmodified input coherency matrix.
    """

    visibilities = np.asarray(visibilities, dtype=complex)
    if visibilities.ndim < 1 or visibilities.shape[-1] != 4:
        raise ValueError(
            "Circular visibility data must have a final RR, LL, RL, LR axis."
        )

    rr = visibilities[..., 0]
    ll = visibilities[..., 1]
    rl = visibilities[..., 2]
    lr = visibilities[..., 3]
    output = np.empty_like(visibilities)
    output[..., 0] = (
        rr
        + leakage1_r * lr
        + np.conj(leakage2_r) * rl
        + leakage1_r * np.conj(leakage2_r) * ll
    )
    output[..., 1] = (
        ll
        + leakage1_l * rl
        + np.conj(leakage2_l) * lr
        + leakage1_l * np.conj(leakage2_l) * rr
    )
    output[..., 2] = (
        rl
        + leakage1_r * ll
        + np.conj(leakage2_l) * rr
        + leakage1_r * np.conj(leakage2_l) * lr
    )
    output[..., 3] = (
        lr
        + leakage1_l * rr
        + np.conj(leakage2_r) * ll
        + leakage1_l * np.conj(leakage2_r) * rl
    )
    return output
