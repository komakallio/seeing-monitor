"""Build the cap catalog from the Gaia archive and VizieR, and the solver index from it.

This code runs once, on a machine with a network connection (`seeingmon catalog build`).
The Raspberry Pi never needs it.

**Gaia.** The build sends one asynchronous ADQL job to the Gaia TAP service. A synchronous query
silently truncates its result (35,458 rows came back as 16,385 in a test), and an asynchronous
job allows 3 million rows for an anonymous user. The job starts with `PHASE=RUN`, the server
answers with a redirect to the job, and the build polls the job phase until it completes. The
archive can queue a job for many minutes.

**Tycho-2.** One synchronous query to the VizieR TAP service returns the Tycho-2 stars of the
cap. Tycho-2 holds the stars that Gaia lacks or measures poorly at the bright end (Polaris, for
example) and the V magnitudes of the bright stars. A Tycho-2 star with a Gaia counterpart within
a few arcseconds gives its V magnitude to the Gaia row. A Tycho-2 star without a counterpart
becomes a row of its own, flagged `FLAG_TYCHO_ONLY`, with a G magnitude estimated from V. The
Tycho-2 mean position is at J2000.0. A star with no mean position (flag `X`, such as Polaris)
falls back to its observed position and has no proper motion, and the builder replaces Polaris
with its Hipparcos astrometry.

**Time.** Polling takes a `Clock`, so a test runs without waiting.

**Solver index.** When the astrometry.net tool `build-astrometry-index` is installed, the build
writes the catalog as a FITS table and builds one index file for each scale preset. The index
holds the catalog positions at the catalog epoch (J2016.0).
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.solvers import fitsio
from seeingmon.survey._scipy import nearest as scipy_nearest
from seeingmon.survey.apparent import POLARIS, StarAstrometry
from seeingmon.survey.catalog import (
    FLAG_TYCHO_ONLY,
    GAIA_DR3_EPOCH_JYEAR,
    CapCatalog,
    tycho_source_id,
)
from seeingmon.survey.geometry import (
    ARCSEC_PER_RAD,
    FloatArray,
    angular_separation,
    normalize,
    radec_to_vector,
    vector_to_radec,
)

GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
VIZIER_TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
TYCHO_EPOCH_JYEAR = 2000.0  # the epoch of the Tycho-2 mean positions
POLARIS_HIP = 11767
MAS_PER_RAD = ARCSEC_PER_RAD * 1000.0

Progress = Callable[[str], None]


class CatalogBuildError(Exception):
    """The catalog or the index could not be built."""


@dataclass(frozen=True, slots=True)
class BuildOptions:
    """The parameters of a catalog build. The defaults give the standard 15 degree cap."""

    radius_deg: float = 15.0
    gaia_mag_limit: float = 13.0
    tycho_mag_limit: float = 12.0  # VT
    cap_ra_deg: float = 0.0
    cap_dec_deg: float = 90.0
    gaia_url: str = GAIA_TAP_URL
    vizier_url: str = VIZIER_TAP_URL
    poll_interval_s: float = 10.0
    max_wait_s: float = 7200.0
    request_timeout_s: float = 300.0
    match_radius_arcsec: float = 3.0
    max_records: int = 3_000_000

    def __post_init__(self) -> None:
        if not 0.0 < self.radius_deg <= 90.0:
            raise ValueError("radius_deg must be greater than 0 and at most 90")
        if self.poll_interval_s <= 0 or self.max_wait_s <= 0 or self.request_timeout_s <= 0:
            raise ValueError("the poll interval, the wait limit, and the timeout must be positive")


# --- Queries -----------------------------------------------------------------------------


def gaia_adql(options: BuildOptions) -> str:
    """The ADQL query for the Gaia DR3 stars of the cap."""
    return (
        "SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, bp_rp "
        "FROM gaiadr3.gaia_source "
        f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {options.cap_ra_deg!r}, {options.cap_dec_deg!r}, {options.radius_deg!r})) "
        f"AND phot_g_mean_mag < {options.gaia_mag_limit!r}"
    )


def tycho_adql(options: BuildOptions) -> str:
    """The ADQL query for the Tycho-2 stars of the cap.

    The cone uses the observed position, which every Tycho-2 star has. The mean position at
    J2000.0 and the proper motion are empty for the stars with position flag `X`.
    """
    return (
        "SELECT TYC1, TYC2, TYC3, RAmdeg AS ra_mean, DEmdeg AS dec_mean, "
        '"RA(ICRS)" AS ra_obs, "DE(ICRS)" AS dec_obs, pmRA AS pmra, pmDE AS pmdec, '
        "BTmag AS bt_mag, VTmag AS vt_mag, HIP AS hip "
        'FROM "I/259/tyc2" '
        'WHERE 1 = CONTAINS(POINT(\'ICRS\', "RA(ICRS)", "DE(ICRS)"), '
        f"CIRCLE('ICRS', {options.cap_ra_deg!r}, {options.cap_dec_deg!r}, {options.radius_deg!r})) "
        f"AND VTmag < {options.tycho_mag_limit!r}"
    )


# --- TAP over HTTP -----------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Return a redirect to the caller as an `HTTPError`, so the caller can read `Location`."""

    def redirect_request(  # type: ignore[no-untyped-def]  # urllib's signature has no types
        self, req, fp, code, msg, headers, newurl
    ):
        return None


def _check_url(url: str) -> str:
    if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
        raise CatalogBuildError("a service URL must start with http:// or https://")
    return url


class TapClient:
    """A minimal client for IVOA TAP services, for the two queries of the catalog build."""

    def __init__(
        self,
        clock: Clock,
        *,
        request_timeout_s: float = 300.0,
        retries: int = 3,
        progress: Progress | None = None,
    ) -> None:
        self._clock = clock
        self._timeout = request_timeout_s
        self._retries = retries
        self._progress = progress or (lambda message: None)
        self._opener = urllib.request.build_opener(_NoRedirect)

    def _request(
        self,
        url: str,
        *,
        data: dict[str, str] | None = None,
        retry_wait_s: float = 5.0,
    ) -> tuple[int, dict[str, str], bytes]:
        """One request. Returns the status, the headers, and the body. A redirect is a result."""
        body = None if data is None else urllib.parse.urlencode(data).encode("ascii")
        request = urllib.request.Request(
            _check_url(url),
            data=body,
            headers={"User-Agent": "seeingmon-catalog-build"},
            method="GET" if body is None else "POST",
        )
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            if attempt:
                self._clock.sleep(retry_wait_s)
            try:
                with self._opener.open(request, timeout=self._timeout) as response:
                    return (
                        int(response.status),
                        {key.lower(): value for key, value in response.headers.items()},
                        response.read(),
                    )
            except urllib.error.HTTPError as error:
                try:
                    headers = {key.lower(): value for key, value in error.headers.items()}
                    if error.code in {301, 302, 303, 307, 308}:
                        return int(error.code), headers, b""
                    detail = error.read(2000).decode("utf-8", errors="replace").strip()
                finally:
                    error.close()  # release the socket, even when the request is retried
                if error.code < 500:
                    raise CatalogBuildError(
                        f"the service answered HTTP {error.code}: {detail[:500]}"
                    ) from error
                last_error = error
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                last_error = error
        raise CatalogBuildError(f"cannot reach the service: {last_error}") from last_error

    def sync_query(self, base_url: str, adql: str, *, max_records: int) -> str:
        """Run a synchronous query and return the CSV text."""
        status, _, body = self._request(
            base_url.rstrip("/") + "/sync",
            data={
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "csv",
                "MAXREC": str(max_records),
                "QUERY": adql,
            },
        )
        if status != 200:
            raise CatalogBuildError(f"the synchronous query answered HTTP {status}")
        return body.decode("utf-8-sig")

    def async_query(
        self, base_url: str, adql: str, *, poll_interval_s: float, max_wait_s: float
    ) -> str:
        """Run an asynchronous job, wait for it, and return the CSV text."""
        base = base_url.rstrip("/") + "/async"
        status, headers, body = self._request(
            base,
            data={
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "csv",
                "PHASE": "RUN",
                "QUERY": adql,
            },
        )
        if status in {301, 302, 303, 307, 308} and "location" in headers:
            job_url = urllib.parse.urljoin(base + "/", headers["location"])
        elif status == 200:
            job_url = f"{base}/{_job_id(body)}"
        else:
            raise CatalogBuildError(f"the job request answered HTTP {status} without a job")
        started_ns = self._clock.monotonic_ns()
        while True:
            _, _, phase_body = self._request(job_url + "/phase")
            phase = phase_body.decode("utf-8", errors="replace").strip().upper()
            waited_s = (self._clock.monotonic_ns() - started_ns) / NS_PER_S
            if phase == "COMPLETED":
                break
            if phase in {"ERROR", "ABORTED", "UNKNOWN"}:
                raise CatalogBuildError(
                    f"the job ended in phase {phase}: {self._job_error(job_url)}"
                )
            if waited_s > max_wait_s:
                raise CatalogBuildError(
                    f"the job is still in phase {phase} after {max_wait_s:.0f} s; "
                    f"raise the wait limit or retry later ({job_url})"
                )
            self._progress(f"waiting for the archive job ({phase.lower()}, {waited_s:.0f} s)")
            self._clock.sleep(poll_interval_s)
        status, _, result = self._request(job_url + "/results/result")
        if status != 200:
            raise CatalogBuildError(f"the job result answered HTTP {status}")
        return result.decode("utf-8-sig")

    def _job_error(self, job_url: str) -> str:
        try:
            _, _, body = self._request(job_url + "/error")
        except CatalogBuildError:
            return "no error text"
        return _error_text(body)


def _error_text(body: bytes) -> str:
    """The message of a UWS error summary, or the flattened body when it has no message."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return " ".join(body.decode("utf-8", errors="replace").split())[:500] or "no error text"
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "message" and (element.text or "").strip():
            return " ".join((element.text or "").split())[:500]
    return " ".join("".join(root.itertext()).split())[:500] or "no error text"


def _job_id(body: bytes) -> str:
    """The job ID from the UWS job document that a service returns instead of a redirect."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as error:
        raise CatalogBuildError("the job answer is not valid XML") from error
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "jobId" and element.text:
            return element.text.strip()
    raise CatalogBuildError("the job answer has no job ID")


# --- CSV ---------------------------------------------------------------------------------

_MISSING = {"", "null", "nan", "none", "--"}


def _read_csv(text: str, required: Sequence[str]) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise CatalogBuildError("the table is empty")
    names = {name.strip().lower(): name for name in reader.fieldnames}
    missing = [name for name in required if name not in names]
    if missing:
        raise CatalogBuildError(
            f"the table lacks the columns {', '.join(missing)} (it has {', '.join(names)})"
        )
    return [
        {key.strip().lower(): (value or "").strip() for key, value in row.items() if key}
        for row in reader
    ]


def _floats(rows: list[dict[str, str]], name: str) -> FloatArray:
    values = np.full(len(rows), np.nan)
    for index, row in enumerate(rows):
        text = row.get(name, "")
        if text.lower() not in _MISSING:
            try:
                values[index] = float(text)
            except ValueError as error:
                raise CatalogBuildError(f"column {name} holds {text!r}, not a number") from error
    return values


@dataclass(frozen=True, slots=True)
class GaiaRows:
    source_id: npt.NDArray[np.int64]
    ra_deg: FloatArray
    dec_deg: FloatArray
    pm_ra_mas_yr: FloatArray
    pm_dec_mas_yr: FloatArray
    parallax_mas: FloatArray
    g_mag: FloatArray
    bp_rp: FloatArray


@dataclass(frozen=True, slots=True)
class TychoRows:
    tyc: npt.NDArray[np.int64]  # shape (N, 3)
    ra_deg: FloatArray  # J2000.0 mean position, or the observed position when there is none
    dec_deg: FloatArray
    pm_ra_mas_yr: FloatArray
    pm_dec_mas_yr: FloatArray
    bt_mag: FloatArray
    vt_mag: FloatArray
    hip: npt.NDArray[np.int64]  # 0 where the star has no Hipparcos number


def parse_gaia_csv(text: str) -> GaiaRows:
    """Parse the CSV that `gaia_adql` produces."""
    rows = _read_csv(
        text, ["source_id", "ra", "dec", "pmra", "pmdec", "parallax", "phot_g_mean_mag", "bp_rp"]
    )
    source_id = np.array([int(row["source_id"]) for row in rows], dtype=np.int64)
    return GaiaRows(
        source_id=source_id,
        ra_deg=_floats(rows, "ra"),
        dec_deg=_floats(rows, "dec"),
        pm_ra_mas_yr=_floats(rows, "pmra"),
        pm_dec_mas_yr=_floats(rows, "pmdec"),
        parallax_mas=_floats(rows, "parallax"),
        g_mag=_floats(rows, "phot_g_mean_mag"),
        bp_rp=_floats(rows, "bp_rp"),
    )


def parse_tycho_csv(text: str) -> TychoRows:
    """Parse the CSV that `tycho_adql` produces."""
    rows = _read_csv(
        text,
        [
            "tyc1",
            "tyc2",
            "tyc3",
            "ra_mean",
            "dec_mean",
            "ra_obs",
            "dec_obs",
            "pmra",
            "pmdec",
            "bt_mag",
            "vt_mag",
            "hip",
        ],
    )
    tyc = np.array([[int(row["tyc1"]), int(row["tyc2"]), int(row["tyc3"])] for row in rows])
    tyc = tyc.reshape(len(rows), 3).astype(np.int64)
    ra_mean = _floats(rows, "ra_mean")
    dec_mean = _floats(rows, "dec_mean")
    has_mean = np.isfinite(ra_mean) & np.isfinite(dec_mean)
    ra = np.where(has_mean, ra_mean, _floats(rows, "ra_obs"))
    dec = np.where(has_mean, dec_mean, _floats(rows, "dec_obs"))
    hip_raw = _floats(rows, "hip")
    return TychoRows(
        tyc=tyc,
        ra_deg=ra,
        dec_deg=dec,
        pm_ra_mas_yr=_floats(rows, "pmra"),
        pm_dec_mas_yr=_floats(rows, "pmdec"),
        bt_mag=_floats(rows, "bt_mag"),
        vt_mag=_floats(rows, "vt_mag"),
        hip=np.where(np.isfinite(hip_raw), hip_raw, 0).astype(np.int64),
    )


# --- Merge -------------------------------------------------------------------------------


def propagate(
    ra_deg: FloatArray,
    dec_deg: FloatArray,
    pm_ra_mas_yr: FloatArray,
    pm_dec_mas_yr: FloatArray,
    years: float,
) -> FloatArray:
    """Unit vectors after `years` of proper motion. A NaN proper motion counts as zero.

    The star moves along the east and north vectors, so a star at the pole needs no special
    case. The motion is linear in the tangent plane, which is exact to far below a milliarcsecond
    for 16 years of the proper motions in the cap.
    """
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    sin_ra, cos_ra = np.sin(ra), np.cos(ra)
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    position = np.stack([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec], axis=-1)
    east = np.stack([-sin_ra, cos_ra, np.zeros_like(ra)], axis=-1)
    north = np.stack([-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec], axis=-1)
    mu_ra = np.nan_to_num(pm_ra_mas_yr) / MAS_PER_RAD
    mu_dec = np.nan_to_num(pm_dec_mas_yr) / MAS_PER_RAD
    return normalize(position + years * (mu_ra[:, None] * east + mu_dec[:, None] * north))


def tycho_v_mag(vt_mag: FloatArray, bt_mag: FloatArray) -> FloatArray:
    """Johnson V from Tycho-2 VT and BT: `V = VT - 0.090 (BT - VT)` (Gaia DR3 documentation)."""
    color = np.where(np.isfinite(bt_mag), bt_mag - vt_mag, 0.0)
    return np.asarray(vt_mag - 0.090 * color, dtype=np.float64)


def estimate_g_from_tycho(vt_mag: FloatArray, bt_mag: FloatArray) -> FloatArray:
    """An estimate of Gaia G for a star that has only Tycho-2 photometry.

    The estimate takes BP-RP to be 1.07 times BT-VT (a solar-type star has BP-RP 0.82 and BT-VT
    0.77) and applies the Gaia G-to-V relation. It is good to about 0.1 mag, which suits stars
    that are brighter than the Gaia photometry handles.
    """
    color = np.clip(np.where(np.isfinite(bt_mag), 1.07 * (bt_mag - vt_mag), 0.8), -0.5, 3.0)
    g_minus_v = -0.02704 + 0.01424 * color - 0.2156 * color**2 + 0.01426 * color**3
    return np.asarray(tycho_v_mag(vt_mag, bt_mag) + g_minus_v, dtype=np.float64)


def merge_catalogs(
    gaia: GaiaRows,
    tycho: TychoRows,
    options: BuildOptions,
    *,
    created_utc_s: int = 0,
) -> CapCatalog:
    """Combine the Gaia and Tycho-2 rows into one catalog.

    A Tycho-2 star within `options.match_radius_arcsec` of a Gaia source (after both move to
    the Gaia epoch) gives that source its V magnitude. The other Tycho-2 stars get rows of
    their own. Polaris takes its Hipparcos astrometry. The cap radius applies to the final
    positions.
    """
    tycho = _patch_polaris(tycho)
    years = GAIA_DR3_EPOCH_JYEAR - TYCHO_EPOCH_JYEAR
    tycho_vectors = propagate(
        tycho.ra_deg, tycho.dec_deg, tycho.pm_ra_mas_yr, tycho.pm_dec_mas_yr, years
    )
    tycho_ra, tycho_dec = vector_to_radec(tycho_vectors)
    v_mag = tycho_v_mag(tycho.vt_mag, tycho.bt_mag)

    gaia_vectors = radec_to_vector(gaia.ra_deg, gaia.dec_deg)
    chord = 2.0 * np.sin(np.radians(options.match_radius_arcsec / 3600.0) / 2.0)
    gaia_v = np.full(gaia.source_id.shape, np.nan)
    unmatched = np.ones(tycho.vt_mag.shape, dtype=bool)
    if len(gaia.source_id) and len(tycho.vt_mag):
        distance, nearest = scipy_nearest(gaia_vectors, tycho_vectors, chord)
        matched = np.isfinite(distance)
        unmatched = ~matched
        # Process the brightest Tycho-2 stars last, so that they win when two share a source.
        for index in np.argsort(-v_mag[matched]):
            tycho_index = np.flatnonzero(matched)[index]
            gaia_v[nearest[tycho_index]] = v_mag[tycho_index]

    only = unmatched & np.isfinite(tycho.vt_mag)
    only_ids = np.array(
        [tycho_source_id(*map(int, row)) for row in tycho.tyc[only]], dtype=np.int64
    )
    columns = {
        "source_id": np.concatenate([gaia.source_id, only_ids]),
        "ra_deg": np.concatenate([gaia.ra_deg, tycho_ra[only]]),
        "dec_deg": np.concatenate([gaia.dec_deg, tycho_dec[only]]),
        "g_mag": np.concatenate(
            [gaia.g_mag, estimate_g_from_tycho(tycho.vt_mag[only], tycho.bt_mag[only])]
        ),
        "pm_ra_mas_yr": np.concatenate([gaia.pm_ra_mas_yr, tycho.pm_ra_mas_yr[only]]),
        "pm_dec_mas_yr": np.concatenate([gaia.pm_dec_mas_yr, tycho.pm_dec_mas_yr[only]]),
        "parallax_mas": np.concatenate([gaia.parallax_mas, np.full(int(only.sum()), np.nan)]),
        "bp_rp": np.concatenate([gaia.bp_rp, np.full(int(only.sum()), np.nan)]),
        "v_mag": np.concatenate([gaia_v, v_mag[only]]),
    }
    extra = np.concatenate(
        [
            np.zeros(len(gaia.source_id), dtype=np.uint16),
            np.full(int(only.sum()), FLAG_TYCHO_ONLY, dtype=np.uint16),
        ]
    )
    # Drop Gaia rows without a magnitude (a source can lack G) and anything outside the cap.
    centre = radec_to_vector(options.cap_ra_deg, options.cap_dec_deg)
    separation = np.degrees(
        angular_separation(radec_to_vector(columns["ra_deg"], columns["dec_deg"]), centre)
    )
    keep = np.isfinite(columns["g_mag"]) & (separation <= options.radius_deg + 1e-6)
    return CapCatalog.from_columns(
        source_id=columns["source_id"][keep],
        ra_deg=columns["ra_deg"][keep],
        dec_deg=columns["dec_deg"][keep],
        g_mag=columns["g_mag"][keep],
        pm_ra_mas_yr=columns["pm_ra_mas_yr"][keep],
        pm_dec_mas_yr=columns["pm_dec_mas_yr"][keep],
        parallax_mas=columns["parallax_mas"][keep],
        bp_rp=columns["bp_rp"][keep],
        v_mag=columns["v_mag"][keep],
        extra_flags=extra[keep],
        epoch_jyear=GAIA_DR3_EPOCH_JYEAR,
        cap_ra_deg=options.cap_ra_deg,
        cap_dec_deg=options.cap_dec_deg,
        cap_radius_deg=options.radius_deg,
        gaia_mag_limit=options.gaia_mag_limit,
        tycho_mag_limit=options.tycho_mag_limit,
        created_utc_s=created_utc_s,
        notes="gaia-dr3 and tycho-2",
    )


def _patch_polaris(tycho: TychoRows) -> TychoRows:
    """Replace the Tycho-2 entry of Polaris with its Hipparcos astrometry.

    Tycho-2 lists Polaris with an observed position and no proper motion.
    """
    index = np.flatnonzero(tycho.hip == POLARIS_HIP)
    if index.size == 0:
        return tycho
    star: StarAstrometry = POLARIS
    ra, dec = tycho.ra_deg.copy(), tycho.dec_deg.copy()
    pm_ra, pm_dec = tycho.pm_ra_mas_yr.copy(), tycho.pm_dec_mas_yr.copy()
    ra[index], dec[index] = star.ra_deg, star.dec_deg
    pm_ra[index], pm_dec[index] = star.pm_ra_mas_yr, star.pm_dec_mas_yr
    return TychoRows(
        tyc=tycho.tyc,
        ra_deg=ra,
        dec_deg=dec,
        pm_ra_mas_yr=pm_ra,
        pm_dec_mas_yr=pm_dec,
        bt_mag=tycho.bt_mag,
        vt_mag=tycho.vt_mag,
        hip=tycho.hip,
    )


# --- The whole build ---------------------------------------------------------------------


def build_catalog(
    options: BuildOptions,
    clock: Clock,
    *,
    gaia_csv: str | None = None,
    tycho_csv: str | None = None,
    progress: Progress | None = None,
) -> CapCatalog:
    """Query the archives (or read CSV text that you give) and merge the result.

    Pass `gaia_csv` or `tycho_csv` to skip a query, for example to rebuild from saved answers.
    """
    say = progress or (lambda message: None)
    client = TapClient(clock, request_timeout_s=options.request_timeout_s, progress=say)
    if gaia_csv is None:
        say(f"querying Gaia DR3 (G < {options.gaia_mag_limit:g}, {options.radius_deg:g} degrees)")
        gaia_csv = client.async_query(
            options.gaia_url,
            gaia_adql(options),
            poll_interval_s=options.poll_interval_s,
            max_wait_s=options.max_wait_s,
        )
    gaia = parse_gaia_csv(gaia_csv)
    say(f"Gaia DR3: {len(gaia.source_id)} stars")
    if tycho_csv is None:
        say(f"querying Tycho-2 (VT < {options.tycho_mag_limit:g})")
        tycho_csv = client.sync_query(
            options.vizier_url, tycho_adql(options), max_records=options.max_records
        )
    tycho = parse_tycho_csv(tycho_csv)
    say(f"Tycho-2: {len(tycho.vt_mag)} stars")
    catalog = merge_catalogs(gaia, tycho, options, created_utc_s=clock.utc_ns() // NS_PER_S)
    say(f"catalog: {len(catalog)} stars")
    return catalog


# --- The solver index --------------------------------------------------------------------


def find_index_tool(command: str | Sequence[str]) -> list[str] | None:
    """The command prefix for `build-astrometry-index`, or `None` when it is not installed."""
    parts = [command] if isinstance(command, str) else list(command)
    if not parts:
        return None
    resolved = shutil.which(parts[0])
    return None if resolved is None else [resolved, *parts[1:]]


def build_solver_index(
    catalog: CapCatalog,
    directory: str | os.PathLike[str],
    *,
    command: Sequence[str],
    presets: Sequence[int] = (8, 9, 10, 11, 12),
    id_base: int = 9_000_000,
    timeout_s: float = 3600.0,
    progress: Progress | None = None,
) -> list[Path]:
    """Build one astrometry.net index file for each scale preset, from the catalog.

    The presets name the quad scale (`-P` of `build-astrometry-index`): preset 8 suits images
    about 2 degrees wide, and each step multiplies the size by the square root of 2. The field
    of the reference camera is 4.4 by 3.0 degrees, so 8 to 12 cover it. `command` is the tool
    with its leading arguments. The function returns the index files that it wrote.
    """
    say = progress or (lambda message: None)
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "cap-input.fits"
    fitsio.write_table(
        source,
        {
            "RA": catalog.ra_deg,
            "DEC": catalog.dec_deg,
            "MAG": catalog.g_mag.astype(np.float32),
        },
    )
    written: list[Path] = []
    try:
        for preset in presets:
            target = folder / f"index-cap-{preset:02d}.fits"
            say(f"building {target.name} (scale preset {preset})")
            arguments = [
                *command,
                "-i", str(source),
                "-o", str(target),
                "-P", str(preset),
                "-S", "MAG",
                "-A", "RA",
                "-D", "DEC",
                "-E",
                "-I", str(id_base + preset),
            ]  # fmt: skip
            try:
                result = subprocess.run(
                    arguments,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise CatalogBuildError(f"cannot run the index tool: {error}") from error
            if result.returncode != 0 or not target.is_file():
                tail = " ".join((result.stderr or result.stdout).split())[-300:]
                raise CatalogBuildError(
                    f"the index tool failed for preset {preset} "
                    f"(exit code {result.returncode}): {tail}"
                )
            written.append(target)
    finally:
        source.unlink(missing_ok=True)
    return written
