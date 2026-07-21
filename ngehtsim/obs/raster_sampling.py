"""Native Fourier sampling for ehtim raster source objects.

This module implements the raster-source portion of ngehtsim's native
simulation path.  It deliberately does not call :meth:`ehtim.Image.sample_uv`:
that method's ``"nfft"`` option depends on the unmaintained pyNFFT package.
Instead, the normal native route uses FINUFFT's two-dimensional type-2 NUFFT.
The ``"direct"`` backend is an independent discrete Fourier-transform
implementation used for validation and for legacy output routes.

The implementation reproduces ehtim's documented image conventions: image
values are Jy per pixel, pixels are centred on the image origin, and the
source's pixel pulse is multiplied into every visibility.
"""

from __future__ import annotations

import numpy as np


SUPPORTED_TRANSFORM_BACKENDS = ("auto", "direct", "finufft")
"""Accepted values for the ``transform_backend`` observation setting."""


def resolve_transform_backend(requested_backend, *, native_path):
    """Resolve an observation's raster-transform backend.

    Parameters
    ----------
    requested_backend : {"auto", "direct", "finufft"}
        Value of the public ``transform_backend`` observation setting.
        ``"auto"`` uses FINUFFT for native :class:`VisibilityDataset`
        simulation and the direct reference transform for legacy
        ``ehtim.Obsdata`` routes.
    native_path : bool
        Whether the caller is sampling the native ground-array data path.

    Returns
    -------
    {"direct", "finufft"}
        Concrete backend to execute.

    Raises
    ------
    ValueError
        If the requested value is unknown or if FINUFFT is explicitly
        requested for an ``ehtim.Obsdata`` compatibility route.

    Notes
    -----
    FINUFFT is intentionally restricted to native simulation for this first
    release.  The legacy route remains available for spacecraft geometry and
    other functionality that has not moved to :class:`VisibilityDataset`.
    """

    if requested_backend not in SUPPORTED_TRANSFORM_BACKENDS:
        raise ValueError(
            "transform_backend={0!r}; supported values are 'auto', 'direct', "
            "and 'finufft'.".format(requested_backend)
        )

    if requested_backend == "auto":
        return "finufft" if native_path else "direct"
    if requested_backend == "finufft" and not native_path:
        raise ValueError(
            "transform_backend='finufft' is available only for native "
            "ground-array simulation. Use 'auto' or 'direct' for an "
            "ehtim.Obsdata compatibility route."
        )
    return requested_backend


def sample_ehtim_raster(image, uv, *, polrep_obs, backend, tolerance):
    """Sample an ``ehtim.Image`` with a native Fourier-transform backend.

    Parameters
    ----------
    image : ehtim.image.Image
        Raster image to sample.  The input object is not modified.
    uv : numpy.ndarray
        Array of shape ``(sample_count, 2)`` containing ``(u, v)`` coordinates
        in wavelengths.
    polrep_obs : {"circ", "stokes"}
        Output polarization representation.  Circular output is ordered
        ``(RR, LL, RL, LR)`` and Stokes output is ordered ``(I, Q, U, V)``.
    backend : {"direct", "finufft"}
        Concrete backend returned by :func:`resolve_transform_backend`.
    tolerance : float
        Requested FINUFFT relative accuracy.  It must be finite and strictly
        between ``1e-16`` and ``1``.  The direct backend accepts the value for
        a uniform public interface but does not use it.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Complex visibility arrays, one for each requested correlation or
        Stokes product.  Undefined polarization planes are returned as zeros,
        matching ehtim's default ``zero_empty_pol=True`` behavior.

    Raises
    ------
    ValueError
        If ``uv`` or ``polrep_obs`` is malformed, the requested backend is
        unsupported, or the FINUFFT tolerance is invalid.
    ImportError
        If the FINUFFT backend is requested but its required dependency is not
        installed.
    """

    uv = _validated_uv(uv)
    vectors = _polarization_vectors(image, polrep_obs)
    rotated_uv = _rotate_uv_for_position_angle(image, uv)

    if backend == "direct":
        return _direct_samples(image, rotated_uv, vectors)
    if backend == "finufft":
        return _finufft_samples(image, rotated_uv, vectors, tolerance)
    raise ValueError(
        "backend={0!r}; expected a resolved 'direct' or 'finufft' backend.".format(
            backend
        )
    )


def _validated_uv(uv):
    """Return a finite two-column visibility-coordinate array."""

    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must be a two-dimensional array with shape (sample_count, 2).")
    if not np.all(np.isfinite(uv)):
        raise ValueError("uv coordinates must be finite.")
    return uv


def _polarization_vectors(image, polrep_obs):
    """Return ehtim-compatible source polarization vectors without mutation."""

    if polrep_obs == "circ":
        converted = image.switch_polrep(polrep_out="circ", pol_prim_out="RR")
        products = ("RR", "LL", "RL", "LR")
    elif polrep_obs == "stokes":
        converted = image.switch_polrep(polrep_out="stokes", pol_prim_out="I")
        products = ("I", "Q", "U", "V")
    else:
        raise ValueError(
            "polrep_obs={0!r}; expected 'circ' or 'stokes'.".format(polrep_obs)
        )

    return tuple(np.asarray(converted.get_polvec(product)) for product in products)


def _rotate_uv_for_position_angle(image, uv):
    """Apply ehtim's source-position-angle coordinate rotation."""

    if image.pa == 0.0:
        return uv

    cosine = np.cos(image.pa)
    sine = np.sin(image.pa)
    return np.column_stack(
        (
            cosine * uv[:, 0] - sine * uv[:, 1],
            sine * uv[:, 0] + cosine * uv[:, 1],
        )
    )


def _pulse_factor(image, uv):
    """Evaluate an ehtim image's pixel pulse at all requested coordinates."""

    return np.fromiter(
        (
            image.pulse(2.0 * np.pi * u, 2.0 * np.pi * v, image.psize, dom="F")
            for u, v in uv
        ),
        dtype=np.complex128,
        count=len(uv),
    )


def _direct_samples(image, uv, vectors):
    """Evaluate ehtim's discrete Fourier transform in bounded-size chunks."""

    sample_count = len(uv)
    outputs = [np.zeros(sample_count, dtype=np.complex128) for _ in vectors]
    present = [index for index, vector in enumerate(vectors) if vector.size]
    if not present or sample_count == 0:
        return tuple(outputs)

    x_coordinates = (
        (image.xdim / 2.0 - 0.5 - np.arange(image.xdim)) * image.psize
    )
    y_coordinates = (
        (image.ydim / 2.0 - 0.5 - np.arange(image.ydim)) * image.psize
    )
    vector_matrix = np.column_stack([vectors[index] for index in present])
    max_matrix_elements = 2_000_000
    chunk_size = max(1, max_matrix_elements // (image.xdim * image.ydim))

    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        uv_chunk = uv[start:stop]
        x_phase = np.exp(2.0j * np.pi * np.outer(uv_chunk[:, 0], x_coordinates))
        y_phase = np.exp(2.0j * np.pi * np.outer(uv_chunk[:, 1], y_coordinates))
        transform = (y_phase[:, :, None] * x_phase[:, None, :]).reshape(
            stop - start,
            image.xdim * image.ydim,
        )
        transform *= _pulse_factor(image, uv_chunk)[:, None]
        sampled = transform @ vector_matrix
        for column, index in enumerate(present):
            outputs[index][start:stop] = sampled[:, column]

    return tuple(outputs)


def _finufft_samples(image, uv, vectors, tolerance):
    """Evaluate the image transform with batched FINUFFT type-2 plans."""

    tolerance = _validated_tolerance(tolerance)
    sample_count = len(uv)
    outputs = [np.zeros(sample_count, dtype=np.complex128) for _ in vectors]
    present = [index for index, vector in enumerate(vectors) if vector.size]
    if not present or sample_count == 0:
        return tuple(outputs)

    try:
        import finufft
    except ImportError as exc:
        raise ImportError(
            "transform_backend='finufft' requires the optional FINUFFT "
            "runtime dependency. Reinstall ngehtsim with its declared "
            "dependencies."
        ) from exc

    coefficients = np.ascontiguousarray(
        np.stack(
            [
                vectors[index].reshape(image.ydim, image.xdim).T
                for index in present
            ]
        ),
        dtype=np.complex128,
    )
    plan = finufft.Plan(
        2,
        (image.xdim, image.ydim),
        n_trans=len(present),
        eps=tolerance,
        isign=-1,
        dtype="complex128",
    )
    plan.setpts(2.0 * np.pi * image.psize * uv[:, 0], 2.0 * np.pi * image.psize * uv[:, 1])
    sampled = plan.execute(coefficients)

    phase = np.exp(
        -1.0j
        * np.pi
        * image.psize
        * (
            (image.xdim % 2 == 0) * uv[:, 0]
            + (image.ydim % 2 == 0) * uv[:, 1]
        )
    )
    sampled *= phase[None, :] * _pulse_factor(image, uv)[None, :]
    for row, index in enumerate(present):
        outputs[index] = sampled[row]
    return tuple(outputs)


def _validated_tolerance(tolerance):
    """Validate the FINUFFT tolerance exposed through observation settings."""

    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("raster_tolerance must be a finite floating-point value.") from exc
    if not np.isfinite(tolerance) or not 1.0e-16 < tolerance < 1.0:
        raise ValueError(
            "raster_tolerance must be finite and strictly between 1e-16 and 1."
        )
    return tolerance
