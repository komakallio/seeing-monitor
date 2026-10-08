"""Model fits of star stamps: sub-pixel centroids, widths, and fluxes.

A bin2 star is undersampled: the PSF is about one pixel wide, so a plain center of gravity
shifts with the pixel phase (0.065 pixel peak to peak at a FWHM of 0.8 pixel), and a trail of
several pixels makes a threshold centroid depend on the noise. This module fits each star with
the profile that the optics and the sky produce:

    I(x, y) = b + F * T(x, y)

where `T` is a Gaussian of width `sigma` that is smeared along a line segment (the trail, with
length `L` and direction `phi`) and integrated over each square pixel, so the pixel response is
exact. The fit has five free parameters: the center `(x0, y0)`, the width `sigma`, the flux `F`,
and the local background `b`. The trail comes from the `TrailModel` or from the star's moments
and stays fixed. Because the model integrates over pixels, an undersampled star gives an
unbiased center.

The solver is a batched Gauss-Newton iteration with Marquardt damping and analytic derivatives,
and it fits every star of a batch at once with NumPy. A pixel that is saturated, masked, or
outside the frame gets weight zero, so a star with a clipped core still fits its wings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.survey import _scipy
from seeingmon.survey.geometry import FloatArray

FWHM_PER_SIGMA = 2.0 * float(np.sqrt(2.0 * np.log(2.0)))
_SQRT2 = float(np.sqrt(2.0))
_SQRT_2PI = float(np.sqrt(2.0 * np.pi))

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]
FrameCounts = npt.NDArray[np.float32] | npt.NDArray[np.uint16]

# The sub-position counts that a trail can use, so stars with similar trails share a batch.
_K_BINS = (1, 2, 4, 8, 16, 32, 64)
_HALF_BINS = (3, 4, 5, 6, 8, 10, 13, 17)
_MAX_ELEMENTS = 6_000_000  # the size of the largest Jacobian array that one batch builds


@dataclass(frozen=True, slots=True, eq=False)
class StampFit:
    """The result of fitting N stars. Every array has length N."""

    x: FloatArray
    y: FloatArray
    flux: FloatArray  # the total of the model above the background, in the data units
    background: FloatArray
    sigma: FloatArray  # the Gaussian width in pixels, before the trail and the pixel
    x_error: FloatArray  # 1-sigma errors from the covariance, scaled by the reduced chi-square
    y_error: FloatArray
    chi2_reduced: FloatArray
    n_pixels: IntArray  # the pixels that carried weight
    converged: BoolArray


def _axis_terms(
    edges: FloatArray, centers: FloatArray, sigma: FloatArray
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """The box-integrated Gaussian along one axis, and its derivatives (in single precision).

    `edges` has shape `(N, S + 1)` (the pixel edges), `centers` `(N, K)` (the sub-positions),
    and `sigma` `(N,)`. The results have shape `(N, K, S)`: the profile, its derivative with
    respect to the position, and its derivative with respect to sigma.
    """
    # The pixel edges relative to the sub-positions are small numbers, so single precision keeps
    # 1e-7 of a pixel and halves the memory traffic of the largest arrays.
    u = (edges[:, None, :] - centers[:, :, None]).astype(np.float32)
    s = sigma.astype(np.float32)[:, None, None]
    cdf = _scipy.erf32(u / (s * np.float32(_SQRT2)))
    pdf = np.exp(np.float32(-0.5) * (u / s) ** 2) / (s * np.float32(_SQRT_2PI))
    phi = np.float32(0.5) * (cdf[..., 1:] - cdf[..., :-1])
    d_position = pdf[..., :-1] - pdf[..., 1:]
    z = u * pdf
    d_sigma = -(z[..., 1:] - z[..., :-1]) / s
    return phi, d_position, d_sigma


def stamp_model(
    edges_x: FloatArray,
    edges_y: FloatArray,
    x0: FloatArray,
    y0: FloatArray,
    sigma: FloatArray,
    trail_dx: FloatArray,
    trail_dy: FloatArray,
    n_sub: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """The unit-flux profile `T` and its derivatives with respect to `x0`, `y0`, and `sigma`.

    The arrays have shape `(N, S, S)`, with the row (y) index first. `trail_dx` and `trail_dy`
    are the full trail vector, and the profile averages `n_sub` positions along it.
    """
    offsets = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    xk = x0[:, None] + trail_dx[:, None] * offsets[None, :]
    yk = y0[:, None] + trail_dy[:, None] * offsets[None, :]
    phi_x, dx, ds_x = _axis_terms(edges_x, xk, sigma)
    phi_y, dy, ds_y = _axis_terms(edges_y, yk, sigma)
    scale = 1.0 / n_sub
    # Batched matrix products: (S x K) @ (K x S) for each star.
    phi_y_t = phi_y.transpose(0, 2, 1)
    dy_t = dy.transpose(0, 2, 1)
    ds_y_t = ds_y.transpose(0, 2, 1)
    t = np.matmul(phi_y_t, phi_x).astype(np.float64) * scale
    d_x0 = np.matmul(phi_y_t, dx).astype(np.float64) * scale
    d_y0 = np.matmul(dy_t, phi_x).astype(np.float64) * scale
    d_sigma = (np.matmul(phi_y_t, ds_x) + np.matmul(ds_y_t, phi_x)).astype(np.float64) * scale
    return t, d_x0, d_y0, d_sigma


def _bin_up(values: FloatArray, bins: tuple[int, ...]) -> IntArray:
    """The smallest bin that holds each value (the largest bin for a bigger value)."""
    edges = np.asarray(bins)
    index = np.searchsorted(edges, np.ceil(values), side="left")
    return np.asarray(edges[np.clip(index, 0, len(edges) - 1)], dtype=np.intp)


def _fit_batch(
    data: FrameCounts,
    bad: BoolArray | None,
    x0: FloatArray,
    y0: FloatArray,
    flux0: FloatArray,
    sigma0: FloatArray,
    trail_dx: FloatArray,
    trail_dy: FloatArray,
    noise: FloatArray,
    e_per_adu: float,
    half: int,
    n_sub: int,
    sigma_bounds: tuple[float, float],
    max_iter: int,
    background0: FloatArray | None = None,
    bad_above: float | None = None,
) -> StampFit:
    """Fit the stars of one batch, which share a stamp size and a sub-position count.

    Each star leaves the iteration when its step falls below the tolerance, so the stars that
    converge fast (the bright ones) stop costing time.
    """
    n = x0.shape[0]
    size = 2 * half + 1
    height, width = data.shape
    steps = np.arange(-half, half + 1)
    gx = np.rint(x0).astype(np.intp)[:, None] + steps[None, :]
    gy = np.rint(y0).astype(np.intp)[:, None] + steps[None, :]
    cx = np.clip(gx, 0, width - 1)
    cy = np.clip(gy, 0, height - 1)
    stamp = data[cy[:, :, None], cx[:, None, :]].astype(np.float64)
    valid = ((gy >= 0) & (gy < height))[:, :, None] & ((gx >= 0) & (gx < width))[:, None, :]
    if bad is not None:
        valid &= ~bad[cy[:, :, None], cx[:, None, :]]
    if bad_above is not None:
        valid &= stamp < bad_above
    valid &= np.isfinite(stamp)
    stamp = np.where(valid, stamp, 0.0).reshape(n, size * size)
    valid = valid.reshape(n, size * size)
    edges_x = np.concatenate([gx - 0.5, gx[:, -1:] + 0.5], axis=1).astype(np.float64)
    edges_y = np.concatenate([gy - 0.5, gy[:, -1:] + 0.5], axis=1).astype(np.float64)
    noise_variance = noise.astype(np.float64) ** 2

    # The parameters of every star, in the order x, y, flux, background, sigma.
    params = np.empty((n, 5))
    params[:, 0] = x0
    params[:, 1] = y0
    params[:, 2] = np.maximum(flux0, 3.0 * noise)
    params[:, 3] = 0.0 if background0 is None else background0
    params[:, 4] = np.clip(sigma0, *sigma_bounds)

    def system(
        rows: npt.NDArray[np.intp],
    ) -> tuple[FloatArray, FloatArray, FloatArray, npt.NDArray[np.intp]]:
        """The normal equations of the stars in `rows`: matrix, gradient, chi-square, pixels."""
        m = rows.shape[0]
        p = params[rows]
        t, d_x0, d_y0, d_sigma = stamp_model(
            edges_x[rows], edges_y[rows], p[:, 0], p[:, 1], p[:, 4],
            trail_dx[rows], trail_dy[rows], n_sub,
        )  # fmt: skip
        t = t.reshape(m, -1)
        amplitude = p[:, 2, None]
        residual = stamp[rows] - (p[:, 3, None] + amplitude * t)
        variance = noise_variance[rows, None] + np.clip(amplitude * t, 0.0, None) / e_per_adu
        weight = np.where(valid[rows], 1.0 / variance, 0.0)
        jacobian = np.empty((m, 5, size * size))
        jacobian[:, 0] = amplitude * d_x0.reshape(m, -1)
        jacobian[:, 1] = amplitude * d_y0.reshape(m, -1)
        jacobian[:, 2] = t
        jacobian[:, 3] = 1.0
        jacobian[:, 4] = amplitude * d_sigma.reshape(m, -1)
        weighted = jacobian * weight[:, None, :]
        normal = np.matmul(weighted, jacobian.transpose(0, 2, 1))
        gradient = np.matmul(weighted, residual[:, :, None])[..., 0]
        chi2 = np.sum(residual * residual * weight, axis=1)
        return normal, gradient, chi2, (weight > 0).sum(axis=1)

    identity = np.eye(5)[None, :, :]
    normal_last = np.zeros((n, 5, 5))
    chi2_last = np.zeros(n)
    pixels = np.zeros(n, dtype=np.intp)
    converged = np.zeros(n, dtype=bool)
    active = np.arange(n)
    for iteration in range(max_iter + 1):
        normal, gradient, chi2, counts = system(active)
        normal_last[active] = normal
        chi2_last[active] = chi2
        pixels[active] = counts
        if iteration == max_iter:
            break
        diagonal = np.einsum("npp->np", normal)
        damped = normal + 1e-3 * diagonal[:, :, None] * identity + 1e-12 * identity
        step = np.linalg.solve(damped, gradient[..., None])[..., 0]
        sigma_now = params[active, 4]
        step[:, 0:2] = np.clip(step[:, 0:2], -1.0, 1.0)
        step[:, 4] = np.clip(step[:, 4], -0.5 * sigma_now, 0.5 * sigma_now)
        params[active] += step
        params[active, 4] = np.clip(params[active, 4], *sigma_bounds)
        small = (
            (np.abs(step[:, 0]) < 2e-3)
            & (np.abs(step[:, 1]) < 2e-3)
            & (np.abs(step[:, 4]) < 5e-3 * sigma_now)
        )
        converged[active[small]] = True
        active = active[~small]
        if active.size == 0:
            break

    covariance = np.linalg.inv(normal_last + 1e-12 * identity)
    chi2_reduced = chi2_last / np.maximum(pixels - 5, 1)
    scale = np.sqrt(np.maximum(chi2_reduced, 1.0))
    return StampFit(
        x=params[:, 0].copy(),
        y=params[:, 1].copy(),
        flux=params[:, 2].copy(),
        background=params[:, 3].copy(),
        sigma=params[:, 4].copy(),
        x_error=np.sqrt(np.clip(covariance[:, 0, 0], 0.0, None)) * scale,
        y_error=np.sqrt(np.clip(covariance[:, 1, 1], 0.0, None)) * scale,
        chi2_reduced=chi2_reduced,
        n_pixels=pixels,
        converged=converged,
    )


def fit_stars(
    data: FrameCounts,
    x0: npt.ArrayLike,
    y0: npt.ArrayLike,
    flux0: npt.ArrayLike,
    sigma0: npt.ArrayLike,
    trail_dx: npt.ArrayLike,
    trail_dy: npt.ArrayLike,
    noise: npt.ArrayLike,
    *,
    e_per_adu: float = 1.0,
    bad: BoolArray | None = None,
    sigma_bounds: tuple[float, float] = (0.15, 4.0),
    max_iter: int = 10,
    background0: npt.ArrayLike | None = None,
    bad_above: float | None = None,
) -> StampFit:
    """Fit a profile to each star of a frame.

    `data` is the background-subtracted frame (a small local background term absorbs what the
    subtraction leaves). `x0`, `y0`, `flux0`, and `sigma0` are the starting values. `trail_dx`
    and `trail_dy` give the trail vector of each star in pixels, and `noise` is the rms of the
    background of each star in the units of `data`. `e_per_adu` converts the signal to
    electrons for the Poisson noise. `bad` marks pixels that carry no weight, such as
    saturated pixels and hot pixels. The function sorts the stars into batches of similar size
    and returns the fits in the input order.

    `data` may hold the background too: `background0` gives the starting background of each star
    (the fit then needs no frame that someone subtracted the background from), and `bad_above`
    gives a weight of zero to every pixel of a stamp at or above that level, so that saturated
    pixels need no mask of the whole frame.
    """
    x0a = np.asarray(x0, dtype=np.float64)
    n = x0a.shape[0]
    y0a = np.asarray(y0, dtype=np.float64)
    flux = np.asarray(flux0, dtype=np.float64)
    sigma = np.asarray(sigma0, dtype=np.float64)
    dx = np.asarray(trail_dx, dtype=np.float64)
    dy = np.asarray(trail_dy, dtype=np.float64)
    rms = np.asarray(noise, dtype=np.float64)
    start_background = None if background0 is None else np.asarray(background0, dtype=np.float64)
    length = np.hypot(dx, dy)
    width = np.maximum(sigma, 0.5)
    half = _bin_up(3.0 * width + length / 2.0 + 1.5, _HALF_BINS)
    # Sub-positions spaced by about one width reproduce a smooth line to better than 1e-6.
    n_sub = _bin_up(length / width + 1.0, _K_BINS)

    out = {
        name: np.zeros(n)
        for name in (
            "x",
            "y",
            "flux",
            "background",
            "sigma",
            "x_error",
            "y_error",
            "chi2_reduced",
        )
    }
    n_pixels = np.zeros(n, dtype=np.intp)
    converged = np.zeros(n, dtype=bool)
    for half_value in np.unique(half):
        for sub_value in np.unique(n_sub[half == half_value]):
            members = np.flatnonzero((half == half_value) & (n_sub == sub_value))
            size = 2 * int(half_value) + 1
            per_star = 5 * size * size + 20 * int(sub_value) * size
            chunk = max(1, _MAX_ELEMENTS // per_star)
            for start in range(0, members.size, chunk):
                index = members[start : start + chunk]
                fit = _fit_batch(
                    data,
                    bad,
                    x0a[index],
                    y0a[index],
                    flux[index],
                    sigma[index],
                    dx[index],
                    dy[index],
                    rms[index],
                    e_per_adu,
                    int(half_value),
                    int(sub_value),
                    sigma_bounds,
                    max_iter,
                    None if start_background is None else start_background[index],
                    bad_above,
                )
                out["x"][index] = fit.x
                out["y"][index] = fit.y
                out["flux"][index] = fit.flux
                out["background"][index] = fit.background
                out["sigma"][index] = fit.sigma
                out["x_error"][index] = fit.x_error
                out["y_error"][index] = fit.y_error
                out["chi2_reduced"][index] = fit.chi2_reduced
                n_pixels[index] = fit.n_pixels
                converged[index] = fit.converged
    return StampFit(
        x=out["x"],
        y=out["y"],
        flux=out["flux"],
        background=out["background"],
        sigma=out["sigma"],
        x_error=out["x_error"],
        y_error=out["y_error"],
        chi2_reduced=out["chi2_reduced"],
        n_pixels=n_pixels,
        converged=converged,
    )
