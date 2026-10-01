"""Vector and rotation helpers for the survey path.

Everything here works on unit vectors and 3 x 3 rotation matrices. Nothing does arithmetic on
right ascension and declination, so nothing is singular at the celestial pole, where the
camera points.

**Conventions.** A rotation matrix `R` is a proper rotation (determinant +1) that changes the
coordinates of a vector from one frame to another: `v_new = R @ v_old`. `rot_x`, `rot_y`, and
`rot_z` return the active rotation of a vector by an angle (right-handed), so
`rot_z(a) @ [1, 0, 0]` is `[cos a, sin a, 0]`. The columns of `R` are the old axes written in
the new frame, and the rows are the new axes written in the old frame.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

ARCSEC_PER_RAD = 180.0 * 3600.0 / np.pi
ARCMIN_PER_RAD = 180.0 * 60.0 / np.pi
MAS_PER_RAD = ARCSEC_PER_RAD * 1000.0
TWO_PI = 2.0 * np.pi


def radec_to_vector(ra_deg: npt.ArrayLike, dec_deg: npt.ArrayLike) -> FloatArray:
    """Unit vectors for equatorial coordinates in degrees, on the last axis of size 3."""
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    cos_dec = np.cos(dec)
    return np.stack([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)], axis=-1)


def vector_to_radec(vector: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Right ascension in [0, 360) and declination in degrees for vectors on the last axis.

    A vector on the pole has an undefined right ascension, and the function returns 0 for it.
    """
    v = np.asarray(vector, dtype=np.float64)
    ra = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0
    dec = np.degrees(np.arctan2(v[..., 2], np.hypot(v[..., 0], v[..., 1])))
    return np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64)


def normalize(vector: npt.ArrayLike) -> FloatArray:
    """Scale vectors on the last axis to unit length."""
    v = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.asarray(v / norm, dtype=np.float64)


def angular_separation(a: npt.ArrayLike, b: npt.ArrayLike) -> FloatArray:
    """The angle in radians between vectors on the last axis.

    The function uses `atan2` of the cross and dot products, so it stays accurate for angles
    near 0 and near pi. The inputs need not be unit vectors.
    """
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    cross = np.linalg.norm(np.cross(va, vb), axis=-1)
    dot = np.sum(va * vb, axis=-1)
    return np.asarray(np.arctan2(cross, dot), dtype=np.float64)


def rot_x(angle_rad: float) -> FloatArray:
    """The active rotation by `angle_rad` about the x axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle_rad: float) -> FloatArray:
    """The active rotation by `angle_rad` about the y axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle_rad: float) -> FloatArray:
    """The active rotation by `angle_rad` about the z axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(vector: npt.ArrayLike) -> FloatArray:
    """The matrix `S` with `S @ w = vector x w`."""
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(rotation_vector: npt.ArrayLike) -> FloatArray:
    """The rotation matrix for a rotation vector (axis times angle in radians), by Rodrigues."""
    rv = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(rv))
    k = skew(rv)
    if angle < 1e-8:  # a series is exact to double precision here
        return np.asarray(np.eye(3) + k + 0.5 * k @ k, dtype=np.float64)
    return np.asarray(
        np.eye(3) + (np.sin(angle) / angle) * k + ((1.0 - np.cos(angle)) / angle**2) * k @ k,
        dtype=np.float64,
    )


def log_so3(rotation: npt.ArrayLike) -> FloatArray:
    """The rotation vector (axis times angle) of a rotation matrix, with an angle up to pi."""
    r = np.asarray(rotation, dtype=np.float64)
    cos_angle = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2.0
    sin_angle = float(np.linalg.norm(vee))
    if sin_angle < 1e-8 and cos_angle > 0.0:
        return np.asarray(vee, dtype=np.float64)  # a small angle: sin(a) ~ a
    if cos_angle < 0.0 and sin_angle < 1e-6:  # an angle near pi: read the axis from R + I
        axis_squared = (np.diag(r) + 1.0) / 2.0
        axis = np.sqrt(np.clip(axis_squared, 0.0, None))
        pivot = int(np.argmax(axis))
        signs = np.sign(r[pivot, :] + r[:, pivot])
        signs[pivot] = 1.0
        axis = axis * np.where(signs == 0.0, 1.0, signs)
        return np.asarray(axis / np.linalg.norm(axis) * np.arccos(cos_angle), dtype=np.float64)
    angle = float(np.arctan2(sin_angle, cos_angle))
    return np.asarray(vee / sin_angle * angle, dtype=np.float64)


def nearest_rotation(matrix: npt.ArrayLike) -> FloatArray:
    """The proper rotation closest to a 3 x 3 matrix, in the Frobenius norm."""
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    correction = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    return np.asarray(u @ correction @ vt, dtype=np.float64)


def best_rotation(
    source: npt.ArrayLike, target: npt.ArrayLike, weights: npt.ArrayLike | None = None
) -> FloatArray:
    """The rotation `R` that minimizes `sum_i w_i |R source_i - target_i|^2` (Wahba's problem).

    `source` and `target` are `(N, 3)` arrays of vectors. The solution comes from the singular
    value decomposition of their weighted cross-covariance (the Kabsch algorithm).
    """
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    w = np.ones(a.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    covariance = (b * w[:, None]).T @ a
    return nearest_rotation(covariance)


def tangent_basis(ra_deg: float, dec_deg: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The east, north, and outward unit vectors at a point of the sphere.

    These are the axes of the intermediate world coordinates of a tangent-plane (TAN) world
    coordinate system with its reference point at `(ra_deg, dec_deg)`. East points toward
    increasing right ascension, and `east x north` is the outward vector. At the pole the
    basis stays well defined for a given right ascension, as the FITS standard needs.
    """
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    sin_ra, cos_ra = np.sin(ra), np.cos(ra)
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    east = np.array([-sin_ra, cos_ra, 0.0])
    north = np.array([-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec])
    outward = np.array([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec])
    return east, north, outward
