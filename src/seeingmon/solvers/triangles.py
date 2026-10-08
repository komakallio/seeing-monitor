"""A plate solver in this process for a camera that looks at the pole: it matches triangles.

The other adapters run a program (astrometry.net, ASTAP). Each call starts a process, reads index
files from disk, and takes seconds on a Pi 4. This solver keeps what it needs in memory and solves
in a few tens of milliseconds, which the alignment view needs while you turn the adjusters.

**The idea.** A triangle of three stars has two side ratios that do not depend on the plate scale
or the roll, so the same triangle in the frame and in the catalog has the same ratios. The solver
builds a table of the triangles of the brightest catalog stars (a few per patch of sky, because the
brightest stars of a frame are the same ones everywhere), and it looks up the triangles of the
brightest detected stars. Each hit gives a similarity transform (scale, roll, shift, and either
parity) from three pairs of points. A transform counts the other bright detections that land on
catalog stars. The transform with the most matches wins, and a least-squares fit of a tangent
plane to all matched stars gives the result.

**What it returns.** The result is an initial solution, as every adapter returns it (see
`seeingmon.solvers.base`): the survey pipeline fits the attitude to all stars afterwards, so an
error of a few pixels does not matter. The catalog positions are ICRS at the catalog epoch, as the
index files of astrometry.net are. A frame with fewer than `MIN_STARS` stars, or a best transform
with fewer than `MIN_MATCHES` matches, gives no solution, and the next solver of the list runs.

**The table.** It holds the triangles with all sides up to `MAX_SIDE_DEG`, from at most
`PER_DISK` stars within `DISK_DEG` of each other. It builds on the first use (call `prepare` to
build it earlier), and it stays. The verification uses a denser list of catalog stars
(`VERIFY_G_LIMIT`), because the field holds only a handful of table stars.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.solvers.base import SolveRequest, SolveResult
from seeingmon.survey import _scipy
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.geometry import (
    ARCSEC_PER_RAD,
    normalize,
    tangent_basis,
    vector_to_radec,
)

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
IntArray = npt.NDArray[np.intp]

NAME = "triangles"
MIN_STARS = 6  # fewer detected stars give no solution
MIN_MATCHES = 6  # the best transform needs this many matched stars
DETECTED_STARS = 12  # the brightest detections whose triangles are looked up
TABLE_G_LIMIT = 10.0  # the catalog stars that can enter the table
PER_DISK = 15  # at most this many table stars within DISK_DEG of a table star
DISK_DEG = 3.0
MAX_SIDE_DEG = 5.0  # the longest side of a table triangle
VERIFY_G_LIMIT = 9.5  # the catalog stars that a transform must meet
RATIO_TOLERANCE = 0.008  # the difference of a side ratio that still counts as the same triangle
MIN_LONGEST = 0.2  # a detected triangle needs a longest side of this share of the frame diagonal
MIN_RATIO = 0.12  # a triangle with a shorter shortest side (against its longest) is too thin
MATCH_RADIUS_PX = 3.0
MAX_CANDIDATES = 6000  # the transforms that the verification keeps, by their fit to the triangle
REFINE_STARS = 80  # the brightest detections that the final fit uses
REFINE_RADIUS_PX = 2.5
REFINE_ROUNDS = 3


@dataclass(frozen=True, slots=True, eq=False)
class Table:
    """The triangles of the brightest catalog stars, and the stars that verify a match."""

    corners: IntArray  # (T, 3) rows of `vectors`, ordered by the opposite side, short to long
    longest: FloatArray  # (T,) the longest side in radians
    vectors: FloatArray  # (A, 3) unit vectors of the table stars
    verify: FloatArray  # (V, 3) unit vectors of the denser verification list
    ratio_tree: Any  # a k-d tree of the side ratios of the triangles
    verify_tree: Any  # a k-d tree of `verify`


@dataclass(frozen=True, slots=True)
class _Transform:
    """A similarity from pixel offsets to a tangent plane at `center`, with the plane's axes."""

    center: FloatArray
    east: FloatArray
    north: FloatArray
    scale: complex  # tangent-plane radians per pixel, with the roll in its phase
    shift: complex
    mirror: bool

    def vectors(self, offsets: FloatArray) -> FloatArray:
        """The unit vectors that the pixel offsets (rows of `(dx, dy)`) map to."""
        z = offsets[:, 0] + 1j * offsets[:, 1]
        plane = self.scale * (np.conj(z) if self.mirror else z) + self.shift
        return normalize(
            self.center + plane.real[:, None] * self.east + plane.imag[:, None] * self.north
        )


def _triples(count: int) -> IntArray:
    """All triples `i < j < k` of `range(count)`, as rows."""
    rows = [
        (i, j, k) for i in range(count) for j in range(i + 1, count) for k in range(j + 1, count)
    ]
    return np.array(rows, dtype=np.intp).reshape(-1, 3)


def _ordered(corners: IntArray, sides: FloatArray) -> tuple[IntArray, FloatArray]:
    """The corners ordered by their opposite side (short to long), and the two side ratios."""
    order = np.argsort(sides, axis=1)
    rows = np.arange(len(corners))[:, None]
    ordered = corners[rows, order]
    s = sides[rows, order]
    return ordered, np.stack([s[:, 0] / s[:, 2], s[:, 1] / s[:, 2]], axis=1)


def _planar_sides(points: FloatArray, corners: IntArray) -> FloatArray:
    """The lengths of the sides opposite the three corners, `(T, 3)`, of triangles in a plane."""
    a, b, c = (points[corners[:, n]] for n in range(3))
    return np.stack(
        [
            np.linalg.norm(b - c, axis=1),
            np.linalg.norm(a - c, axis=1),
            np.linalg.norm(a - b, axis=1),
        ],
        axis=1,
    )


def _angular_sides(vectors: FloatArray, corners: IntArray) -> FloatArray:
    """The same for triangles on the sphere, in degrees."""
    a, b, c = (vectors[corners[:, n]] for n in range(3))

    def angle(u: FloatArray, v: FloatArray) -> FloatArray:
        return np.degrees(np.arccos(np.clip(np.sum(u * v, axis=1), -1.0, 1.0)))

    return np.stack([angle(b, c), angle(a, c), angle(a, b)], axis=1)


def build_table(catalog: CapCatalog) -> Table:
    """The triangle table of a catalog."""
    stars = np.flatnonzero(catalog.g_mag < TABLE_G_LIMIT)
    stars = stars[np.argsort(catalog.g_mag[stars], kind="stable")]
    cos_disk = np.cos(np.radians(DISK_DEG))
    taken = np.zeros((len(stars), 3))
    count = 0
    for vector in catalog.vectors[stars]:
        if count and int(np.count_nonzero(taken[:count] @ vector >= cos_disk)) >= PER_DISK:
            continue
        taken[count] = vector
        count += 1
    vectors = taken[:count].copy()
    near = np.degrees(np.arccos(np.clip(vectors @ vectors.T, -1.0, 1.0))) <= MAX_SIDE_DEG
    parts: list[IntArray] = []
    for i in range(count):
        neighbors = np.flatnonzero(near[i, i + 1 :]) + i + 1
        if neighbors.size < 2:
            continue
        j, k = np.triu_indices(neighbors.size, 1)
        keep = near[neighbors[j], neighbors[k]]
        if keep.any():
            parts.append(
                np.stack(
                    [np.full(int(keep.sum()), i), neighbors[j[keep]], neighbors[k[keep]]], axis=1
                )
            )
    raw = np.concatenate(parts).astype(np.intp) if parts else np.zeros((0, 3), np.intp)
    sides = _angular_sides(vectors, raw)
    corners, ratios = _ordered(raw, sides)
    longest = np.radians(sides.max(axis=1))
    wide = ratios[:, 0] >= MIN_RATIO
    corners, ratios, longest = corners[wide], ratios[wide], longest[wide]
    verify = catalog.vectors[np.flatnonzero(catalog.g_mag < VERIFY_G_LIMIT)]
    return Table(
        corners=corners,
        longest=longest,
        vectors=vectors,
        verify=verify,
        ratio_tree=_scipy.kdtree(ratios),
        verify_tree=_scipy.kdtree(verify),
    )


class TriangleSolver:
    """The solver of this module. It satisfies `seeingmon.solvers.base.PlateSolver`."""

    def __init__(self, catalog: CapCatalog, *, clock: Clock | None = None) -> None:
        self._catalog = catalog
        self._clock = clock or SystemClock()
        self._table: Table | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return NAME

    def prepare(self) -> None:
        """Build the table now, so that the first solve does not pay for it."""
        self._get_table()

    def _get_table(self) -> Table:
        with self._lock:
            if self._table is None:
                self._table = build_table(self._catalog)
            return self._table

    def solve(self, request: SolveRequest) -> SolveResult:
        started = self._clock.monotonic_ns()
        found = self._search(request)
        elapsed = (self._clock.monotonic_ns() - started) / NS_PER_S
        if found is None:
            return SolveResult(solved=False, solver=NAME, elapsed_s=elapsed)
        ra, dec, cd, matched, rms = found
        return SolveResult(
            solved=True,
            solver=NAME,
            elapsed_s=elapsed,
            center_ra_deg=ra,
            center_dec_deg=dec,
            scale_arcsec_px=float(np.sqrt(abs(cd[0] * cd[3] - cd[1] * cd[2])) * 3600.0),
            cd_matrix=cd,
            n_matched=len(matched),
            rms_arcsec=rms,
            matched=matched,
        )

    # --- The search ------------------------------------------------------------------------

    def _search(
        self, request: SolveRequest
    ) -> tuple[float, float, tuple[float, float, float, float], tuple[int, ...], float] | None:
        stars = request.stars
        if len(stars) < MIN_STARS:
            return None
        table = self._get_table()
        order = np.argsort(-stars.flux, kind="stable")
        xy = np.stack([stars.x[order], stars.y[order]], axis=1).astype(np.float64)
        center = np.array([(request.width_px - 1) / 2.0, (request.height_px - 1) / 2.0])
        offsets = xy - center
        top = offsets[:DETECTED_STARS]
        transform = self._best_transform(table, request, top)
        if transform is None:
            return None
        return self._refine(table, request, transform, offsets[:REFINE_STARS], order)

    def _best_transform(
        self, table: Table, request: SolveRequest, top: FloatArray
    ) -> _Transform | None:
        """The similarity, from the triangles of the brightest stars, that meets the most stars."""
        low = request.scale_low_arcsec_px / ARCSEC_PER_RAD
        high = request.scale_high_arcsec_px / ARCSEC_PER_RAD
        diagonal = float(np.hypot(request.width_px, request.height_px))
        triples = _triples(len(top))
        sides = _planar_sides(top, triples)
        corners, ratios = _ordered(triples, sides)
        longest_px = sides.max(axis=1)
        usable = (ratios[:, 0] >= MIN_RATIO) & (longest_px >= MIN_LONGEST * diagonal)
        corners, ratios, longest_px = corners[usable], ratios[usable], longest_px[usable]
        if not len(corners):
            return None
        hits = table.ratio_tree.query_ball_point(ratios, RATIO_TOLERANCE, p=np.inf)
        counts = np.array([len(h) for h in hits], dtype=np.intp)
        if counts.sum() == 0:
            return None
        detected = np.repeat(np.arange(len(corners)), counts)
        rows = np.concatenate([np.asarray(h, dtype=np.intp) for h in hits if len(h)])
        # The longest side fixes the plate scale of a hit, which must lie in the allowed range.
        in_range = table.longest[rows] / longest_px[detected]
        fits = (in_range >= low) & (in_range <= high)
        detected, rows = detected[fits], rows[fits]
        if rows.size == 0:
            return None

        pixels = top[corners[detected]]  # (n, 3, 2)
        d: ComplexArray = pixels[..., 0] + 1j * pixels[..., 1]
        vertices = table.vectors[table.corners[rows]]  # (n, 3, 3)
        centroid = normalize(vertices.sum(axis=1))
        reference = np.where(np.abs(centroid[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
        east = normalize(np.cross(reference, centroid))
        north = np.cross(centroid, east)
        depth = np.einsum("nkc,nc->nk", vertices, centroid)
        t: ComplexArray = (
            np.einsum("nkc,nc->nk", vertices, east) + 1j * np.einsum("nkc,nc->nk", vertices, north)
        ) / depth

        # A similarity from three pairs of points, for either parity (complex least squares).
        d_mean = d.mean(axis=1, keepdims=True)
        t_mean = t.mean(axis=1, keepdims=True)
        dd, tt = d - d_mean, t - t_mean
        norm = np.sum(np.abs(dd) ** 2, axis=1)
        plain = np.sum(tt * np.conj(dd), axis=1) / norm
        flipped = np.sum(tt * dd, axis=1) / norm
        miss_plain = np.sum(np.abs(tt - plain[:, None] * dd) ** 2, axis=1)
        miss_flipped = np.sum(np.abs(tt - flipped[:, None] * np.conj(dd)) ** 2, axis=1)
        mirror = miss_flipped < miss_plain
        a = np.where(mirror, flipped, plain)
        miss = np.where(mirror, miss_flipped, miss_plain)

        magnitude = np.abs(a)
        keep = np.flatnonzero(
            (magnitude >= low) & (magnitude <= high) & (miss <= 3.0 * (6.0 * magnitude) ** 2)
        )
        if keep.size == 0:
            return None
        if keep.size > MAX_CANDIDATES:
            keep = keep[np.argsort(miss[keep])[:MAX_CANDIDATES]]
        a, mirror = a[keep], mirror[keep]
        d_anchor = np.where(mirror, np.conj(d_mean[keep, 0]), d_mean[keep, 0])
        b = t_mean[keep, 0] - a * d_anchor
        c, e, n = centroid[keep], east[keep], north[keep]

        points = top[:, 0] + 1j * top[:, 1]
        mapped = np.where(mirror[:, None], np.conj(points)[None, :], points[None, :])
        plane = a[:, None] * mapped + b[:, None]
        direction = (
            c[:, None, :]
            + plane.real[..., None] * e[:, None, :]
            + plane.imag[..., None] * n[:, None, :]
        )
        direction /= np.linalg.norm(direction, axis=2, keepdims=True)
        distance, _ = table.verify_tree.query(
            direction.reshape(-1, 3), distance_upper_bound=high * MATCH_RADIUS_PX
        )
        close = distance.reshape(direction.shape[:2]) <= (np.abs(a) * MATCH_RADIUS_PX)[:, None]
        score = close.sum(axis=1)
        winner = int(np.argmax(score))
        if score[winner] < MIN_MATCHES:
            return None
        return _Transform(
            center=c[winner],
            east=e[winner],
            north=n[winner],
            scale=complex(a[winner]),
            shift=complex(b[winner]),
            mirror=bool(mirror[winner]),
        )

    def _refine(
        self,
        table: Table,
        request: SolveRequest,
        transform: _Transform,
        offsets: FloatArray,
        order: IntArray,
    ) -> tuple[float, float, tuple[float, float, float, float], tuple[int, ...], float] | None:
        """A tangent plane at the image center, fitted to every star that the transform meets."""
        direction = transform.vectors(offsets)
        middle = transform.vectors(np.zeros((1, 2)))[0]
        chord = REFINE_RADIUS_PX * abs(transform.scale)
        hit = np.zeros(len(offsets), dtype=np.bool_)
        index = np.zeros(len(offsets), dtype=np.intp)
        for _ in range(REFINE_ROUNDS):
            distance, found = table.verify_tree.query(direction, distance_upper_bound=chord)
            hit = np.isfinite(distance)
            index = np.where(hit, found, 0)
            if int(hit.sum()) < MIN_MATCHES:
                return None
            ra, dec = (float(v) for v in vector_to_radec(middle))
            east, north, outward = tangent_basis(ra, dec)
            catalog = table.verify[index[hit]]
            depth = catalog @ outward
            target = np.stack([(catalog @ east) / depth, (catalog @ north) / depth], axis=1)
            design = np.column_stack([offsets[hit], np.ones(int(hit.sum()))])
            solution, *_ = np.linalg.lstsq(design, target, rcond=None)
            matrix, shift = solution[:2].T, solution[2]
            plane = offsets @ matrix.T + shift
            direction = normalize(outward + plane[:, :1] * east + plane[:, 1:] * north)
            middle = normalize(outward + shift[0] * east + shift[1] * north)
            chord = REFINE_RADIUS_PX * float(np.sqrt(abs(np.linalg.det(matrix))))
        # The final plane, centered on the sky point of the image center (the shift is gone).
        ra, dec = (float(v) for v in vector_to_radec(middle))
        east, north, outward = tangent_basis(ra, dec)
        catalog = table.verify[index[hit]]
        depth = catalog @ outward
        target = np.stack([(catalog @ east) / depth, (catalog @ north) / depth], axis=1)
        solution, *_ = np.linalg.lstsq(offsets[hit], target, rcond=None)
        matrix = solution.T
        residual = offsets[hit] @ matrix.T - target
        rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))) * ARCSEC_PER_RAD)
        degrees = matrix * (180.0 / np.pi)
        cd = (
            float(degrees[0, 0]),
            float(degrees[0, 1]),
            float(degrees[1, 0]),
            float(degrees[1, 1]),
        )
        matched = tuple(int(order[i]) for i in np.flatnonzero(hit))
        if len(matched) < MIN_MATCHES:
            return None
        return ra, dec, cd, matched, rms
