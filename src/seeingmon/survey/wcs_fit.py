"""The pointing fit: a tangent-plane (TAN) camera model fitted to matched stars.

**The camera model.** `CameraAttitude` maps unit vectors in the apparent frame (CIRS, see
`seeingmon.survey.apparent`) to pixels:

    w = R u                      the star in the camera frame
    xi = w_x / w_z, eta = w_y / w_z      the gnomonic (TAN) coordinates, in radians
    x = x_c + xi / s,  y = y_c + parity * eta / s      the pixel, with the plate scale `s` in
                                                       radians per pixel

The camera frame has x to the right, y down (the row index grows downward, as in the arrays),
and z along the boresight toward the sky. `R` is a proper rotation, and `parity` is -1 for an
optical path with a single mirror. The principal point `(x_c, y_c)` is the center of the sensor.
Pixel coordinates are those of the readout mode of the frame: the center of the first pixel is
(0, 0). The model has four free parameters, the three of the rotation and the scale. It is a
rigid TAN WCS written with rotation matrices, so nothing is singular at the pole.

**The fit.** `fit_attitude` starts from an initial attitude (from a plate solver or from the
tracker), matches the catalog stars to the detections by nearest neighbor in the pixel plane
with a shrinking radius, and runs a weighted Gauss-Newton fit with sigma clipping. The weights
come from the centroid errors of the detections plus a floor for what the centroid model does
not capture.

**Conventions for the outputs.**

- The *roll* is the position angle of the direction to the celestial pole in the image,
  measured from image up (the -y axis) toward image left (the -x axis). It is undefined when
  the pole sits within a pixel of the center, and `roll_deg` then returns `None`.
- The *center* is the ICRS direction of the boresight (the astrometric place, with aberration
  and precession removed).
- A *FITS TAN WCS* uses the intermediate coordinates `[xi, eta]` toward the east and the north,
  related to pixel offsets by `CD`. `attitude_from_tan_wcs` and `attitude_to_tan_wcs` convert,
  and the tests check them against `astropy.wcs`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.survey import _scipy
from seeingmon.survey.apparent import ObservationEpoch, astrometric_from_apparent
from seeingmon.survey.geometry import (
    ARCSEC_PER_RAD,
    FloatArray,
    best_rotation,
    exp_so3,
    nearest_rotation,
    tangent_basis,
    vector_to_radec,
)

IntArray = npt.NDArray[np.intp]
BoolArray = npt.NDArray[np.bool_]

# The roll is undefined when the pole lies within this many pixels of the field center.
ROLL_MIN_DISTANCE_PX = 1.0
_Z = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True, slots=True, eq=False)
class CameraAttitude:
    """The camera model: attitude, plate scale, parity, and principal point.

    `rotation` maps CIRS vectors to camera vectors. `scale_rad_px` is the plate scale in
    radians per pixel, and `parity` is +1, or -1 for a mirrored image. `center_px` is the
    principal point in the pixel coordinates of the readout mode.
    """

    rotation: FloatArray
    scale_rad_px: float
    parity: int
    center_px: tuple[float, float]

    def __post_init__(self) -> None:
        if self.parity not in (1, -1):
            raise ValueError("parity must be +1 or -1")
        if not self.scale_rad_px > 0.0:
            raise ValueError("the plate scale must be positive")

    @property
    def scale_arcsec_px(self) -> float:
        return self.scale_rad_px * ARCSEC_PER_RAD

    def camera_vectors(self, vectors: npt.ArrayLike) -> FloatArray:
        """The vectors in the camera frame, shape `(N, 3)`."""
        return np.asarray(np.atleast_2d(vectors), dtype=np.float64) @ self.rotation.T

    def project(self, vectors: npt.ArrayLike) -> tuple[FloatArray, FloatArray, BoolArray]:
        """Pixel positions of CIRS unit vectors, and a mask of those in front of the camera."""
        camera = self.camera_vectors(vectors)
        in_front = camera[:, 2] > 0.05
        safe = np.where(in_front, camera[:, 2], 1.0)
        x = self.center_px[0] + camera[:, 0] / safe / self.scale_rad_px
        y = self.center_px[1] + self.parity * camera[:, 1] / safe / self.scale_rad_px
        return (
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(in_front, dtype=np.bool_),
        )

    def unproject(self, x: npt.ArrayLike, y: npt.ArrayLike) -> FloatArray:
        """CIRS unit vectors for pixel positions, shape `(N, 3)`."""
        xi = (
            np.atleast_1d(np.asarray(x, dtype=np.float64)) - self.center_px[0]
        ) * self.scale_rad_px
        eta = (
            self.parity
            * (np.atleast_1d(np.asarray(y, dtype=np.float64)) - self.center_px[1])
            * self.scale_rad_px
        )
        camera = np.stack([xi, eta, np.ones_like(xi)], axis=-1)
        camera /= np.linalg.norm(camera, axis=-1, keepdims=True)
        return np.asarray(camera @ self.rotation, dtype=np.float64)

    def boresight(self) -> FloatArray:
        """The CIRS unit vector of the boresight (the principal point)."""
        return np.asarray(self.rotation.T @ _Z, dtype=np.float64)

    def pole_pixel(self) -> tuple[float, float] | None:
        """Where the celestial pole of date falls, or `None` if it is behind the camera."""
        x, y, front = self.project(_Z)
        return (float(x[0]), float(y[0])) if front[0] else None

    def pole_distance_px(self) -> float | None:
        """The distance from the field center to the pole, in pixels (`None` if not in front)."""
        pole = self.pole_pixel()
        if pole is None:
            return None
        return float(np.hypot(pole[0] - self.center_px[0], pole[1] - self.center_px[1]))

    def roll_deg(self) -> float | None:
        """The position angle of the direction to the pole, in degrees in (-180, 180].

        The angle runs from image up (-y) toward image left (-x). The value is `None` when the
        pole lies within `ROLL_MIN_DISTANCE_PX` of the field center, where it has no meaning.
        """
        pole = self.pole_pixel()
        if pole is None:
            direction = self.rotation @ _Z  # behind the camera: use the camera-frame direction
            dx, dy = float(direction[0]), float(self.parity * direction[1])
            return float(np.degrees(np.arctan2(-dx, -dy)))
        dx, dy = pole[0] - self.center_px[0], pole[1] - self.center_px[1]
        if np.hypot(dx, dy) < ROLL_MIN_DISTANCE_PX:
            return None
        return float(np.degrees(np.arctan2(-dx, -dy)))

    def center_icrs(self, epoch: ObservationEpoch) -> tuple[float, float]:
        """The ICRS right ascension and declination of the boresight, in degrees."""
        direction = astrometric_from_apparent(self.boresight(), epoch)
        ra, dec = vector_to_radec(direction)
        return float(ra), float(dec)

    def with_rotation(
        self, rotation: FloatArray, scale_rad_px: float | None = None
    ) -> CameraAttitude:
        return CameraAttitude(
            rotation=rotation,
            scale_rad_px=self.scale_rad_px if scale_rad_px is None else scale_rad_px,
            parity=self.parity,
            center_px=self.center_px,
        )


def pixel_center(width_px: int, height_px: int) -> tuple[float, float]:
    """The center of a frame, where the center of the first pixel is (0, 0)."""
    return (width_px - 1) / 2.0, (height_px - 1) / 2.0


# --- Conversions to and from a FITS TAN WCS ------------------------------------------------


def attitude_from_tan_wcs(
    ra_deg: float,
    dec_deg: float,
    cd_matrix: tuple[float, float, float, float],
    *,
    center_px: tuple[float, float],
) -> CameraAttitude:
    """The camera model of a TAN WCS whose reference pixel is `center_px`.

    The attitude is in the sky frame of the WCS, which is ICRS for a solver. `cd_matrix` is
    `(CD1_1, CD1_2, CD2_1, CD2_2)` in degrees per pixel, mapping pixel offsets to the
    intermediate coordinates (east, north). The determinant gives the parity, and the matrix
    gives the plate scale and the roll. A matrix that is not a rotation times a scale, such as
    one with an aspect ratio, is replaced by the nearest one (the scale is the geometric mean).
    """
    cd = np.asarray(cd_matrix, dtype=np.float64).reshape(2, 2) * (np.pi / 180.0)
    determinant = float(np.linalg.det(cd))
    if determinant == 0.0:
        raise ValueError("the CD matrix is singular")
    parity = 1 if determinant > 0.0 else -1
    scale = float(np.sqrt(abs(determinant)))
    flip = np.diag([1.0, float(parity)])
    q_transposed = cd @ flip / scale  # CD = s Q^T P, so Q^T = CD P / s
    u, _, vt = np.linalg.svd(q_transposed)
    q = (u @ vt).T  # the nearest orthogonal matrix, transposed
    east, north, outward = tangent_basis(ra_deg, dec_deg)
    rotation = np.stack(
        [q[0, 0] * east + q[0, 1] * north, q[1, 0] * east + q[1, 1] * north, outward]
    )
    return CameraAttitude(
        rotation=nearest_rotation(rotation), scale_rad_px=scale, parity=parity, center_px=center_px
    )


def attitude_to_tan_wcs(
    attitude: CameraAttitude,
) -> tuple[float, float, tuple[float, float, float, float]]:
    """The TAN WCS of an attitude, in the sky frame of the attitude.

    Returns the reference right ascension and declination in degrees (the boresight) and
    `(CD1_1, CD1_2, CD2_1, CD2_2)` in degrees per pixel.
    """
    boresight = attitude.boresight()
    ra, dec = vector_to_radec(boresight)
    east, north, _ = tangent_basis(float(ra), float(dec))
    # The rows of the rotation are the camera axes. Their components along (east, north) make Q.
    q = np.array(
        [
            [attitude.rotation[0] @ east, attitude.rotation[0] @ north],
            [attitude.rotation[1] @ east, attitude.rotation[1] @ north],
        ]
    )
    cd = attitude.scale_rad_px * q.T @ np.diag([1.0, float(attitude.parity)])
    cd_deg = cd * (180.0 / np.pi)
    return (
        float(ra),
        float(dec),
        (float(cd_deg[0, 0]), float(cd_deg[0, 1]), float(cd_deg[1, 0]), float(cd_deg[1, 1])),
    )


def attitude_from_solver_solution(
    ra_deg: float,
    dec_deg: float,
    cd_matrix: tuple[float, float, float, float],
    *,
    center_px: tuple[float, float],
    catalog_vectors_icrs: npt.ArrayLike,
    apparent_vectors: npt.ArrayLike,
) -> CameraAttitude:
    """The camera model in the apparent frame, from a solver's solution in the catalog frame.

    A solver works with catalog positions (ICRS at the catalog epoch). The camera sees apparent
    places, which differ by precession, nutation, aberration, and proper motion. The two
    patterns agree except for a rotation and a small shift that aberration adds. So the
    function takes catalog stars near the field center, in both frames, and finds the
    rotation `R` that makes `R @ apparent` equal the camera vector that the solver predicts for
    the same star (Wahba's problem). The result is exact up to the differential aberration
    across the field (0.4 pixel in bin2), which the fit then removes.
    """
    solver = attitude_from_tan_wcs(ra_deg, dec_deg, cd_matrix, center_px=center_px)
    targets = np.asarray(catalog_vectors_icrs, dtype=np.float64) @ solver.rotation.T
    sources = np.asarray(apparent_vectors, dtype=np.float64)
    rotation = best_rotation(sources, targets)
    return solver.with_rotation(rotation)


# --- The fit -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FitOptions:
    """Parameters of `fit_attitude`."""

    match_radius_px: tuple[float, ...] = (4.0, 2.0, 1.2)  # one match pass for each radius
    clip_sigma: float = 4.0
    clip_floor_px: float = 0.05  # the clip never goes below this many pixels
    max_iterations: int = 10
    min_stars: int = 4
    sigma_floor_px: float = 0.03  # added in quadrature to each centroid error
    margin_px: float = 8.0  # catalog stars this far outside the frame still match
    fit_scale: bool = True


@dataclass(frozen=True, slots=True, eq=False)
class FitResult:
    """The outcome of a fit.

    `detection_index` and `catalog_index` pair the stars that the final fit used (indices into
    the arrays that were passed in). `residual_x_px` and `residual_y_px` are observed minus
    modeled pixel positions of those pairs. `n_candidates` counts the catalog stars that fell
    inside the frame, and `n_matched` the pairs in the final fit.
    """

    attitude: CameraAttitude
    n_matched: int
    n_candidates: int
    rms_px: float
    rms_arcsec: float
    detection_index: IntArray
    catalog_index: IntArray
    residual_x_px: FloatArray
    residual_y_px: FloatArray
    converged: bool


def _jacobian(
    attitude: CameraAttitude, vectors: FloatArray, fit_scale: bool
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The pixel model, and the derivatives of x and y with respect to the 4 parameters.

    The parameters are a rotation vector `delta` (the update `R <- exp([delta]x) R`) and the
    logarithm of the scale. Returns the modeled `x`, `y`, and the `(N, 2, 4)` Jacobian.
    """
    camera = attitude.camera_vectors(vectors)
    xi = camera[:, 0] / camera[:, 2]
    eta = camera[:, 1] / camera[:, 2]
    scale = attitude.scale_rad_px
    x = attitude.center_px[0] + xi / scale
    y = attitude.center_px[1] + attitude.parity * eta / scale
    jacobian = np.zeros((vectors.shape[0], 2, 4))
    jacobian[:, 0, 0] = -xi * eta / scale
    jacobian[:, 0, 1] = (1.0 + xi**2) / scale
    jacobian[:, 0, 2] = -eta / scale
    jacobian[:, 1, 0] = -attitude.parity * (1.0 + eta**2) / scale
    jacobian[:, 1, 1] = attitude.parity * xi * eta / scale
    jacobian[:, 1, 2] = attitude.parity * xi / scale
    if fit_scale:
        jacobian[:, 0, 3] = -(x - attitude.center_px[0])
        jacobian[:, 1, 3] = -(y - attitude.center_px[1])
    return x, y, jacobian


def _gauss_newton(
    attitude: CameraAttitude,
    vectors: FloatArray,
    x_obs: FloatArray,
    y_obs: FloatArray,
    weights: FloatArray,
    options: FitOptions,
) -> tuple[CameraAttitude, bool]:
    current = attitude
    columns = 4 if options.fit_scale else 3
    for _ in range(options.max_iterations):
        x, y, jacobian = _jacobian(current, vectors, options.fit_scale)
        residual = np.stack([x_obs - x, y_obs - y], axis=1).reshape(-1)
        design = jacobian[:, :, :columns].reshape(-1, columns)
        w = np.repeat(weights, 2)
        normal = design.T @ (design * w[:, None])
        gradient = design.T @ (w * residual)
        try:
            step = np.linalg.solve(normal, gradient)
        except np.linalg.LinAlgError:
            return current, False
        rotation = exp_so3(step[:3]) @ current.rotation
        scale = current.scale_rad_px * (float(np.exp(step[3])) if options.fit_scale else 1.0)
        current = current.with_rotation(nearest_rotation(rotation), scale)
        if float(np.max(np.abs(step[:3]))) < 1e-10 and (
            not options.fit_scale or abs(float(step[3])) < 1e-9
        ):
            return current, True
    return current, False


def _match(
    attitude: CameraAttitude,
    catalog_vectors: FloatArray,
    detections_xy: FloatArray,
    radius_px: float,
    shape: tuple[int, int],
    margin_px: float,
    usable: BoolArray,
) -> tuple[IntArray, IntArray, int]:
    """Pair catalog stars with detections: the nearest detection within the radius.

    Returns the detection and catalog indices of the pairs, and the number of candidate
    catalog stars. A detection that two catalog stars both claim goes to the closer one.
    """
    height, width = shape
    x, y, front = attitude.project(catalog_vectors)
    candidate = (
        front
        & usable
        & (x > -margin_px)
        & (x < width - 1 + margin_px)
        & (y > -margin_px)
        & (y < height - 1 + margin_px)
    )
    rows = np.flatnonzero(candidate)
    if rows.size == 0 or detections_xy.shape[0] == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp), int(rows.size)
    distance, nearest = _scipy.nearest(
        detections_xy, np.column_stack([x[rows], y[rows]]), radius_px
    )
    found = np.isfinite(distance)
    rows, nearest, distance = rows[found], nearest[found], distance[found]
    order = np.argsort(distance, kind="stable")
    rows, nearest = rows[order], nearest[order]
    _, first = np.unique(nearest, return_index=True)  # the closest catalog star for each detection
    return nearest[first], rows[first], int(candidate.sum())


def fit_attitude(
    initial: CameraAttitude,
    catalog_vectors: npt.ArrayLike,
    detections_x: npt.ArrayLike,
    detections_y: npt.ArrayLike,
    detections_error_px: npt.ArrayLike,
    *,
    shape: tuple[int, int],
    options: FitOptions | None = None,
    catalog_usable: npt.ArrayLike | None = None,
) -> FitResult | None:
    """Fit the camera model to the detections, starting from `initial`.

    `catalog_vectors` are the apparent CIRS unit vectors of the catalog stars, `(N, 3)`. The
    detections are pixel positions and a centroid error in pixels for each. `shape` is
    `(height, width)` of the frame, and `catalog_usable` masks catalog stars that must not
    match (for example, the saturated or faint ones). The function returns `None` when
    fewer than `options.min_stars` stars match.
    """
    opts = options or FitOptions()
    vectors = np.asarray(catalog_vectors, dtype=np.float64)
    dx = np.asarray(detections_x, dtype=np.float64)
    dy = np.asarray(detections_y, dtype=np.float64)
    error = np.asarray(detections_error_px, dtype=np.float64)
    detections_xy = np.column_stack([dx, dy])
    usable = (
        np.ones(vectors.shape[0], dtype=bool)
        if catalog_usable is None
        else np.asarray(catalog_usable, dtype=bool)
    )
    attitude = initial
    pairs_d: IntArray = np.zeros(0, dtype=np.intp)
    pairs_c: IntArray = np.zeros(0, dtype=np.intp)
    candidates = 0
    converged = False
    for radius in opts.match_radius_px:
        pairs_d, pairs_c, candidates = _match(
            attitude, vectors, detections_xy, radius, shape, opts.margin_px, usable
        )
        if pairs_d.size < opts.min_stars:
            return None
        weights = 1.0 / (error[pairs_d] ** 2 + opts.sigma_floor_px**2)
        attitude, converged = _gauss_newton(
            attitude, vectors[pairs_c], dx[pairs_d], dy[pairs_d], weights, opts
        )

    # Sigma clipping on the last match: drop the outliers and refit until nothing changes.
    keep = np.ones(pairs_d.size, dtype=bool)
    for _ in range(5):
        x, y, _ = attitude.project(vectors[pairs_c])
        residual = np.hypot(dx[pairs_d] - x, dy[pairs_d] - y)
        # The median of a two-dimensional Gaussian residual is 1.1774 times the width per axis.
        scatter = float(np.median(residual[keep])) / 1.1774
        limit = max(opts.clip_sigma * scatter, opts.clip_floor_px)
        new_keep = residual <= limit
        if new_keep.sum() < opts.min_stars:
            return None
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
        weights = 1.0 / (error[pairs_d[keep]] ** 2 + opts.sigma_floor_px**2)
        attitude, converged = _gauss_newton(
            attitude, vectors[pairs_c[keep]], dx[pairs_d[keep]], dy[pairs_d[keep]], weights, opts
        )
    used_d, used_c = pairs_d[keep], pairs_c[keep]
    x, y, _ = attitude.project(vectors[used_c])
    residual_x, residual_y = dx[used_d] - x, dy[used_d] - y
    rms_px = float(np.sqrt(np.mean(residual_x**2 + residual_y**2)))
    return FitResult(
        attitude=attitude,
        n_matched=int(used_d.size),
        n_candidates=candidates,
        rms_px=rms_px,
        rms_arcsec=rms_px * attitude.scale_arcsec_px,
        detection_index=used_d,
        catalog_index=used_c,
        residual_x_px=residual_x,
        residual_y_px=residual_y,
        converged=converged,
    )
