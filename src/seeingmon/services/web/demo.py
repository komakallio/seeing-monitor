"""Demo mode: the web process over synthetic data, with no camera, no `core`, and no site.

`seeingmon web --demo` builds the same app as `seeingmon web` and feeds it from three synthetic
sources, so that you can look at the UI on a laptop:

- **A store.** `write_demo_store` writes 24 hours of records into a temporary folder through the
  real `Store`: seeing windows, sky quality, pointing, health, and events, and preview images with
  a few FITS frames. The data follow a made-up night (see `sun_elevation_deg`) with two cloud
  passes, a gust of wind, and a short camera fault. Nothing here describes a real place or a real
  measurement. The records carry no zenith angle, and the station is `demo-station`. The pointing
  records put the pole 0.05 degree from the center of the frame, and Polaris circles it at 15
  degrees an hour on a circle of 0.62 degree (see `pole_and_polaris`).
- **A fake `core`.** `DemoCore` is a `FakeCoreClient` whose live view streams the frames of a
  synthetic star field (`StarField`) while the fake scheduler aligns. Start and stop the alignment
  in the UI with the demo token (`DEMO_TOKEN`). The pole starts 0.9 degrees right of and 0.4
  degrees above the field center, drifts through the center and back (`pole_offset_px`), and
  Polaris follows on its orbit, so the numbers and the lines of the overlay move, and the orbit
  is green, amber, and red in turn. The saturation warning comes and goes. In the 20 seconds from
  50 s to 70 s of every drift period, the solver finds no star field (`solution_lost`): the state
  has no solution, no offset, and no sky, and the aim ring comes from the last solution, as it does
  in `core`. The fake `core` also
  holds a dark library of six sets with a model, and it plays a dark session on a short timeline
  (see `DEMO_DARK_SCRIPT`): queued, bias frames, the wait for the cover, dark frames, and the
  build. The camera counts as covered a few seconds into the wait, so the session ends
  `ok`, adds a set at the sensor temperature, and pauses the fake scheduler, so that
  Resume works. A session without the wait for the cover fails, and Pause aborts one. It also
  holds a flat library of two flats (one in use), and it plays a flat session on a short timeline
  (see `DEMO_FLAT_SCRIPT`): queued, the setup, the search for the exposure, the frames (with a
  note that the light drifts), and the combination. The session adds a pending flat, which you
  review on the Flat page, and the fake scheduler pauses at the end. A second set, with the source
  turned, replaces the flat of the first set by a flat of both. Stop or Pause ends a session, and
  Use this flat and Discard work on the library. While the fake scheduler is in `auto` or `safe`,
  the fake `core` also streams a synthetic video of Polaris at about 20 frames a second
  (`PolarisSky`: a star that jitters with the seeing, flickers by a few percent, and sits on a
  noisy sky, through the real stretch and PNG encoder of `core`), and it serves a rolling seeing
  value that varies slowly around the seeing of the demo night (`demo_live_seeing`). Pause the
  fake scheduler or start the alignment, and the video goes quiet.
- **A clock.** `DemoClock` stands still in UTC at `DEMO_NOW_NS`, so the newest record is always
  fresh, and it runs in monotonic time, so the rate limits, the timeouts, and the live view work.

The commands go to the fake `core` only, so the demo token is a fixed word and not a secret.
Nothing in a demo reaches a camera, a heater, or a sink.
"""

from __future__ import annotations

import asyncio
import io
import math
import random
import shutil
import statistics
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from seeingmon.analysis.base import StarState
from seeingmon.clock import NS_PER_S, Clock, ClockStatus, utc_ns_to_iso
from seeingmon.frames import Roi
from seeingmon.records.base import Record
from seeingmon.records.samples import sample_record
from seeingmon.scheduler import activity as words
from seeingmon.services.core.polaris import FrameSlot, PolarisRenderer
from seeingmon.services.web import demo_activity
from seeingmon.services.web.auth import ScryptParams, hash_token
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.contract import (
    ActivityView,
    AimRingView,
    AlignmentFrame,
    AlignmentFrameInfo,
    AlignmentState,
    DarkModelView,
    DarkSetView,
    FaultView,
    FocusHistoryView,
    FocusView,
    HistogramView,
    LastSolutionView,
    LiveSeeingView,
    OffsetView,
    PolarisFrame,
    ReticleView,
    SaturationView,
    SkyView,
    SolvedView,
    TargetView,
    TimingView,
)
from seeingmon.services.web.core_client import FakeCoreClient
from seeingmon.services.web.fake_dark import DarkScript
from seeingmon.services.web.fake_flat import FlatLook, FlatScript
from seeingmon.store.db import Store, StoreReader
from seeingmon.store.layout import DataLayout

if TYPE_CHECKING:
    from fastapi import FastAPI

DEMO_NOW_NS = int(datetime(2026, 10, 1, 3, 0, tzinfo=UTC).timestamp()) * NS_PER_S
DEMO_STATION = "demo-station"
DEMO_PROFILE = "demo-profile"
DEMO_TOKEN = "demo"
HISTORY_S = 24 * 3600
WINDOW_S = 60
FRAME_WIDTH_PX = 2072
FRAME_HEIGHT_PX = 1411
PLATE_SCALE_ARCSEC_PX = 3.82
FRAME_PERIOD_S = 0.5
PREVIEW_SIZE = (640, 436)
IMAGE_SIZE = (480, 327)
FITS_SIZE = (128, 87)
FITS_BLOCK = 2880
HISTOGRAM_BINS = 32
IDLE_POLL_S = 0.2
RAD_PER_ARCSEC = math.pi / 180 / 3600
HOUR_NS = 3600 * NS_PER_S
DAY_NS = 24 * HOUR_NS
DEMO_SENSOR_TEMPERATURE_C = 12.3
# The demo plays a dark session in about 35 seconds: the queue, 5 s of bias frames, a wait
# for the cover (the camera counts as covered after 6 of its 9 seconds), 15 s of dark
# frames, and a build.
DEMO_DARK_SCRIPT = DarkScript(queued_s=4.0, bias_s=5.0, cover_s=9.0, dark_s=15.0, build_s=1.0)
# The demo library: the temperature of each set (degrees C), and its age (days). No set lies
# within 3 C of the sensor temperature, so the library is due until a session adds one.
DEMO_DARK_SETS = (
    (1.4, 90.0),
    (5.1, 66.0),
    (8.3, 47.0),
    (16.9, 25.0),
    (20.2, 12.0),
    (23.8, 3.0),
)
# The demo plays a flat session in about 40 seconds: the queue, 2 s of setup, 4 s of search for the
# exposure, 16 s of frames (the light drifts a little from the middle on), and 3 s of build.
DEMO_FLAT_SCRIPT = FlatScript(
    queued_s=3.0,
    setup_s=2.0,
    exposure_s=4.0,
    capture_s=16.0,
    build_s=3.0,
    warnings=("The light drifts: a frame is 3.4 % above the median level.",),
)
# The demo library: the flat in use (made from two sets, 12 days ago), and an older one from before
# someone cleaned the lens (one more dust shadow).
DEMO_FLATS = (
    (
        47.0,
        FlatLook(
            corner_percent=-9.9,
            shadows=(
                (1210, 802, 2.4, 38.0),
                (2874, 1905, 1.6, 26.0),
                (3420, 420, 1.1, 21.0),
                (610, 2210, 1.3, 24.0),
            ),
        ),
    ),
    (12.0, FlatLook()),
)

# The sky of the live view. The pole starts `POLE_START_DEG` right of and above the field center,
# and it swings through the center and back once in `POLE_DRIFT_PERIOD_S`. Polaris moves on a
# circle of `SKY_COLATITUDE_DEG` around it.
FRAME_CENTER_X = (FRAME_WIDTH_PX - 1) / 2
FRAME_CENTER_Y = (FRAME_HEIGHT_PX - 1) / 2
SKY_COLATITUDE_DEG = 0.62
POLE_START_DEG = (0.9, 0.4)
POLE_DRIFT_PERIOD_S = 150.0
ORBIT_RADIUS_PX = SKY_COLATITUDE_DEG * 3600.0 / PLATE_SCALE_ARCSEC_PX
# The solver finds no star field from `SOLUTION_LOST_FROM_S` to `SOLUTION_LOST_UNTIL_S` of every
# drift period, as it does when a hand moves the camera too fast or a cloud passes.
SOLUTION_LOST_FROM_S = 50.0
SOLUTION_LOST_UNTIL_S = 70.0
LOST_REASON = "the tracker could not match the frame"
# The focus value of the demo is a slow wave with a little noise, and every `FOCUS_SPIKE_EVERY`
# frames a hand touches the telescope for a moment, which inflates the stars (a spike).
FOCUS_POINTS = 120
FOCUS_SPIKE_EVERY = 97
FOCUS_SPIKE_FACTOR = 2.8
# The rule of the spike flag, as `seeingmon.services.core.alignment.focus` states it: a value above
# `SPIKE_FACTOR` times the median of the preceding `SPIKE_WINDOW` values, when at least
# `SPIKE_MIN_PREVIOUS` precede it. A test compares the two.
SPIKE_FACTOR = 2.0
SPIKE_WINDOW = 10
SPIKE_MIN_PREVIOUS = 3
# The right ascension of Polaris in the demo, which only decides where the labels of the grid fall.
POLARIS_RA_DEG = 45.0

# The video of Polaris: a 128 x 128 pixel ROI of the bin1 stream (1.91 arcsec per pixel), shown at
# 20 frames a second while the camera runs at 82. The star is about 1.4 pixels wide (FWHM), and it
# moves by `IMAGE_MOTION_PX_PER_ARCSEC` pixels rms along each axis for each arcsecond of seeing
# (0.48 arcsec of motion per axis for 1 arcsec of seeing, at 1.91 arcsec per pixel). The numbers
# are made up. They show a plausible picture, and they measure nothing.
POLARIS_SIZE = 128
POLARIS_ROI = Roi(2008, 1347, POLARIS_SIZE, POLARIS_SIZE)
POLARIS_PERIOD_S = 0.05
FAST_PLATE_SCALE_ARCSEC_PX = 1.91
CAMERA_FPS = 82.0
IMAGE_MOTION_PX_PER_ARCSEC = 0.25
STAR_SIGMA_PX = 0.6
STAR_PEAK_DN = 18_000.0
FULL_SCALE_DN = 65_520.0
SKY_DN = 480.0
DN_PER_ADU = 16.0
E_PER_ADU = 3.5
READ_NOISE_E = 2.65
TILT_TAU_S = 0.12
SCINTILLATION_TAU_S = 0.08
SCINTILLATION_RMS = 0.035
LIVE_EVERY_S = 2.0
LIVE_SPAN_S = 10.0
LIVE_MIN_SPAN_S = 4.0
DEMO_SEEING_ARCSEC = 1.6

# The pole and Polaris in the pointing history. A rigid mount keeps the pole at one pixel, 47 pixels
# (0.05 degree) from the center of the frame, and the sky turns once in a sidereal day. At the
# newest record Polaris is 75 degrees from straight below the pole, toward the right.
POLE_DISTANCE_PX = 47.0
POLARIS_ANGLE_NOW_DEG = 75.0
SIDEREAL_DEG_PER_HOUR = 360.98564736629 / 24.0


class DemoClock:
    """A clock that stands still in UTC and runs in monotonic time. See the module documentation."""

    def __init__(self, utc_ns: int = DEMO_NOW_NS) -> None:
        self._utc_ns = utc_ns

    def utc_ns(self) -> int:
        return self._utc_ns

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def status(self) -> ClockStatus:
        return ClockStatus(synchronized=True, error_bound_ns=800_000, source="demo")


# --- The night ---------------------------------------------------------------------------------


def sun_elevation_deg(t_utc_ns: int) -> float:
    """The elevation of the Sun in the demo, as a wave of one day.

    The demo night is centered on 22:30 UTC. The Sun is 18 degrees below the horizon at 16:30 and
    at 04:30, and the fast path runs while the Sun is more than 6 degrees below it.
    """
    hours = (t_utc_ns / NS_PER_S % 86_400) / 3600
    return -18.0 - 48.0 * math.cos((hours - 22.5) * 2 * math.pi / 24)


def _hours_before(now_ns: int, hours: float) -> int:
    return now_ns - round(hours * HOUR_NS)


def _cloud_cover(t_utc_ns: int, now_ns: int) -> float:
    """The share of the sky that clouds cover, from 0 to 1. Two passes cross the night."""
    cover = 0.0
    for center_h, half_width_h, peak in ((7.8, 0.45, 0.95), (2.8, 0.35, 0.8)):
        offset_h = (now_ns - t_utc_ns) / HOUR_NS - center_h
        cover = max(cover, peak * math.exp(-((offset_h / half_width_h) ** 2)))
    return cover


def _gust(t_utc_ns: int, now_ns: int) -> bool:
    """Whether a gust of wind shakes the mount, which shows as a vibration line."""
    return 4.62 <= (now_ns - t_utc_ns) / HOUR_NS <= 4.82


def _psd(seeing: float, frequencies: list[float], rng: random.Random) -> list[float]:
    """A power spectrum of image motion: flat at low frequency and steep above a corner."""
    return [
        0.35 * seeing**2 / (1 + (f / 4.0) ** 2) ** (17 / 12) * rng.uniform(0.85, 1.15)
        for f in frequencies
    ]


def demo_seeing_arcsec(t_ns: int) -> float:
    """The seeing of the demo night before the noise, in arcseconds: two slow waves."""
    hours = t_ns / HOUR_NS
    return 1.55 + 0.4 * math.sin(hours / 0.85) + 0.18 * math.sin(hours * 2.9 + 1.3)


def _seeing_record(t_ns: int, now_ns: int, seeing: float, rng: random.Random) -> Record:
    elevation = sun_elevation_deg(t_ns)
    cover = _cloud_cover(t_ns, now_ns)
    gust = _gust(t_ns, now_ns)
    flags: list[str] = []
    quality: dict[str, str] | None = None
    values: dict[str, Any] = {}
    frequencies = [round(0.5 * (80 / 0.5) ** (i / 23), 3) for i in range(24)]
    if elevation > -18.0:
        flags.append("twilight")
    if cover > 0.15:
        flags.append("cloud")
    if gust:
        flags.append("vibration")
        seeing *= 1.5
    if cover > 0.6:
        quality = {
            "seeing_fwhm_arcsec": "too few usable frames",
            "r0_cm": "too few usable frames",
        }
    else:
        r0 = 0.98 * 500e-9 / (seeing * RAD_PER_ARCSEC) * 100
        rms = 0.45 * seeing / 2.355 * 2.0
        values = {
            "seeing_fwhm_arcsec": round(seeing, 3),
            "r0_cm": round(r0, 2),
            "seeing_fwhm_structure_arcsec": round(seeing * rng.uniform(0.94, 1.06), 3),
            "image_motion_rms_x_arcsec": round(rms * rng.uniform(0.95, 1.1), 3),
            "image_motion_rms_y_arcsec": round(rms * rng.uniform(0.85, 1.0), 3),
            "scintillation_index": round(0.003 + 0.002 * rng.random() + 0.01 * cover, 5),
            "width_fwhm_arcsec": round(math.hypot(seeing, 1.1), 3),
            "peak_mean_dn": round(38_000 * (1 - 0.6 * cover) * rng.uniform(0.9, 1.05)),
            "flux_mean_e": round(2.4e6 * (1 - 0.6 * cover) * rng.uniform(0.95, 1.05)),
            "background_mean_dn": round(620 + 900 * max(0.0, 1 + elevation / 18)),
            "saturated_fraction": 0.0,
            "motion_psd_freq_hz": frequencies,
            "motion_psd_x_arcsec2_per_hz": _psd(seeing, frequencies, rng),
            "motion_psd_y_arcsec2_per_hz": _psd(seeing, frequencies, rng),
            "vibration_lines_hz": [11.7, 23.4] if gust else [],
            "sensor_temperature_c": round(-5.0 + 0.1 * rng.random(), 2),
        }
    frames = 5800
    dropped = rng.choice([0, 0, 0, 1, 2])
    return sample_record(
        "seeing_window",
        station_id=DEMO_STATION,
        profile_id=DEMO_PROFILE,
        t_utc_ns=t_ns,
        duration_s=float(WINDOW_S),
        stream_id=1,
        readout_mode="bin1",
        exposure_us=8000,
        gain=120,
        n_frames=frames - dropped,
        n_dropped=dropped,
        valid_fraction=round(0.99 - 0.5 * cover, 3),
        frame_rate_hz=96.7,
        flags=flags,
        quality=quality,
        **values,
    )


def _sky_record(t_ns: int, now_ns: int, rng: random.Random) -> Record:
    elevation = sun_elevation_deg(t_ns)
    cover = _cloud_cover(t_ns, now_ns)
    flags: list[str] = []
    if cover > 0.15:
        flags.append("cloud")
    if elevation > -18.0:
        flags.append("twilight")
    dark = 21.25 - 0.08 * math.sin(t_ns / HOUR_NS * 0.9)
    sky = dark - 6.5 * max(0.0, 1 + elevation / 18) - 0.9 * cover
    return sample_record(
        "sky_quality",
        station_id=DEMO_STATION,
        profile_id=DEMO_PROFILE,
        t_utc_ns=t_ns,
        sky_mag_arcsec2=round(sky + rng.gauss(0, 0.03), 3),
        sky_mag_arcsec2_v=round(sky - 0.15 + rng.gauss(0, 0.03), 3),
        zero_point_mag=round(24.1 - 1.2 * cover, 3),
        transparency=round(max(0.05, 0.93 - 0.9 * cover + rng.gauss(0, 0.01)), 3),
        cloud_fraction=round(min(1.0, cover), 3),
        limiting_mag=round(7.3 - 3.5 * cover - 1.5 * max(0.0, 1 + elevation / 18), 2),
        n_stars_used=round(52 * (1 - 0.85 * cover)),
        flags=flags,
    )


def pole_and_polaris(t_ns: int, now_ns: int, roll_deg: float) -> dict[str, float]:
    """The pixels of the pole and of Polaris in a pointing record, as record fields.

    The pole lies `POLE_DISTANCE_PX` from the center of the frame, in the direction that the roll
    gives (the roll is the position angle of that direction, from image up toward image left).
    Polaris lies `SKY_COLATITUDE_DEG` from the pole, which is `ORBIT_RADIUS_PX` in the image. Its
    angle from straight below the pole toward the right grows by `SIDEREAL_DEG_PER_HOUR`, so
    Polaris turns counterclockwise in the image, as the sky does for a camera that looks north.
    """
    roll = math.radians(roll_deg)
    pole_x = FRAME_CENTER_X - POLE_DISTANCE_PX * math.sin(roll)
    pole_y = FRAME_CENTER_Y - POLE_DISTANCE_PX * math.cos(roll)
    hours_ago = (now_ns - t_ns) / HOUR_NS
    angle = math.radians(POLARIS_ANGLE_NOW_DEG - SIDEREAL_DEG_PER_HOUR * hours_ago)
    return {
        "pole_x_px": round(pole_x, 2),
        "pole_y_px": round(pole_y, 2),
        "polaris_x_px": round(pole_x + ORBIT_RADIUS_PX * math.sin(angle), 2),
        "polaris_y_px": round(pole_y + ORBIT_RADIUS_PX * math.cos(angle), 2),
    }


def _pointing_record(t_ns: int, now_ns: int, rng: random.Random) -> Record:
    hours = (now_ns - t_ns) / HOUR_NS
    offset = 0.55 + 0.25 * math.sin(hours / 3.1) + 0.05 * rng.random() + 0.01 * (24 - hours)
    cover = _cloud_cover(t_ns, now_ns)
    flags = ["few_stars"] if cover > 0.5 else []
    roll = round(12.3 + 0.12 * math.sin(hours / 5.0), 3)
    return sample_record(
        "pointing",
        station_id=DEMO_STATION,
        profile_id=DEMO_PROFILE,
        t_utc_ns=t_ns,
        offset_arcmin=round(offset, 3),
        roll_deg=roll,
        plate_scale_arcsec_px=PLATE_SCALE_ARCSEC_PX,
        attitude=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        solve_rms_arcsec=round(2.9 + 0.4 * rng.random(), 3),
        n_matched=round(41 * (1 - 0.7 * cover)),
        focus_fwhm_px=round(2.35 + 0.1 * rng.random(), 3),
        **pole_and_polaris(t_ns, now_ns, roll),
        readout_mode="bin2",
        solver="demo-solver",
        solve_time_s=round(1.1 + 0.3 * rng.random(), 2),
        flags=flags,
    )


def _health_record(t_ns: int, now_ns: int, index: int) -> Record:
    night = sun_elevation_deg(t_ns) < -6.0
    faulty = 5.55 <= (now_ns - t_ns) / HOUR_NS <= 5.75 or 10.55 <= (now_ns - t_ns) / HOUR_NS <= 10.7
    return sample_record(
        "health",
        station_id=DEMO_STATION,
        profile_id=DEMO_PROFILE,
        t_utc_ns=t_ns,
        state="auto" if night else "safe",
        degraded=faulty,
        components={
            "acquire": "ok",
            "core": "ok",
            "scheduler": "ok",
            "camera": "degraded" if faulty else "ok",
            "store": "ok",
        },
        dark_due=False,
        sensor_temperature_c=-5.0 if night else 3.5,
        heater_duty=0.0 if night else None,
        free_space_gb=round(41.2 - 0.002 * index, 2),
        data_used_gb=round(3.4 + 0.002 * index, 2),
        dropped_total=3 + index // 40,
        queue_depth=0,
        time_synchronized=True,
        time_error_bound_ms=0.8,
        uptime_s=float(86_400 * 6 + index * 300),
        cpu_load_1m=round(0.8 + 0.2 * math.sin(index / 5), 2),
        memory_used_mb=round(612 + 3 * math.sin(index / 9), 1),
    )


def _event(
    now_ns: int, hours_before: float, level: str, kind: str, message: str, **detail: Any
) -> Record:
    return sample_record(
        "event",
        station_id=DEMO_STATION,
        profile_id=DEMO_PROFILE,
        t_utc_ns=_hours_before(now_ns, hours_before),
        level=level,
        kind=kind,
        message=message,
        detail=detail or None,
    )


def demo_events(now_ns: int) -> list[Record]:
    """The events of the demo night: state changes, clouds, a camera fault, and a dark session."""
    return [
        _event(
            now_ns,
            23.8,
            "info",
            "scheduler.state_change",
            "The scheduler entered the safe state: the Sun is too high.",
            state="safe",
        ),
        _event(
            now_ns,
            11.6,
            "info",
            "survey.dark_session",
            "A dark session of 20 frames ended.",
            frames=20,
        ),
        _event(
            now_ns,
            11.5,
            "info",
            "scheduler.state_change",
            "The scheduler entered the auto state.",
            state="auto",
        ),
        _event(
            now_ns,
            10.62,
            "error",
            "camera.fault",
            "The camera did not answer, and the recovery ladder reset the port.",
            failures=2,
        ),
        _event(now_ns, 10.55, "info", "camera.recovered", "The camera answers again.", failures=0),
        _event(now_ns, 8.25, "warning", "scheduler.cloud", "Clouds crossed the star.", active=True),
        _event(now_ns, 7.5, "info", "scheduler.cloud", "The clouds went away.", active=False),
        _event(
            now_ns,
            5.65,
            "warning",
            "camera.dropped_frames",
            "The camera dropped 2% of the frames in a window.",
            fraction=0.02,
        ),
        _event(
            now_ns,
            4.7,
            "warning",
            "fast.vibration",
            "The spectrum shows a vibration line at 11.7 Hz.",
            line_hz=11.7,
        ),
        _event(now_ns, 3.3, "warning", "scheduler.cloud", "Clouds crossed the star.", active=True),
        _event(now_ns, 2.5, "info", "scheduler.cloud", "The clouds went away.", active=False),
        _event(
            now_ns,
            0.4,
            "info",
            "survey.pointing",
            "The pointing survey solved 41 stars.",
            matched=41,
        ),
    ]


def demo_records(now_ns: int = DEMO_NOW_NS, *, seed: int = 2026) -> list[Record]:
    """All the records of the demo, sorted by time. The same seed gives the same records."""
    rng = random.Random(seed)
    start_ns = now_ns - HISTORY_S * NS_PER_S
    records: list[Record] = []
    noise = 0.0
    t_ns = start_ns - start_ns % (WINDOW_S * NS_PER_S) + WINDOW_S * NS_PER_S
    while t_ns < now_ns:
        noise = 0.9 * noise + rng.gauss(0, 0.05)
        if sun_elevation_deg(t_ns) < -6.0:
            seeing = demo_seeing_arcsec(t_ns)
            records.append(_seeing_record(t_ns, now_ns, max(0.75, seeing + noise), rng))
            if (t_ns // NS_PER_S) % 300 == 0:
                records.append(_sky_record(t_ns, now_ns, rng))
            if (t_ns // NS_PER_S) % 600 == 0:
                records.append(_pointing_record(t_ns, now_ns, rng))
        t_ns += WINDOW_S * NS_PER_S
    health_ns = now_ns - 30 * NS_PER_S
    index = 0
    while health_ns > start_ns:
        records.append(_health_record(health_ns, now_ns, 288 - index))
        health_ns -= 300 * NS_PER_S
        index += 1
    records += demo_events(now_ns)
    records.sort(key=lambda record: record.t_utc_ns)
    return records


# --- Images ------------------------------------------------------------------------------------


def pole_offset_px(t_s: float) -> tuple[float, float]:
    """The offset of the pole from the frame center at demo time `t_s`, in pixels (right, down).

    At `t_s` = 0 the pole is `POLE_START_DEG` right of and above the center. It swings through
    the center and back once in `POLE_DRIFT_PERIOD_S`, with a little wobble, as if a hand moved
    the mount. The orbit of Polaris is red while the pole is far out, amber while the circle just
    fits, and green near the center.
    """
    phase = 0.5 * (1.0 + math.cos(2.0 * math.pi * t_s / POLE_DRIFT_PERIOD_S))
    px_per_deg = 3600.0 / PLATE_SCALE_ARCSEC_PX
    right = POLE_START_DEG[0] * px_per_deg * phase + 30.0 * math.sin(t_s / 9.0)
    down = -POLE_START_DEG[1] * px_per_deg * phase + 20.0 * math.sin(t_s / 7.0)
    return right, down


def demo_fwhm_px(seq: int) -> float:
    """The focus value of the demo frame `seq`, in pixels, with the spike of a touched telescope."""
    value = 2.4 + 0.25 * math.sin(seq * FRAME_PERIOD_S / 50) + 0.08 * math.sin(seq * 1.7)
    if seq > 0 and seq % FOCUS_SPIKE_EVERY == 0:
        value *= FOCUS_SPIKE_FACTOR
    return round(value, 4)


def demo_focus_stars(seq: int) -> int:
    """The number of stars that the focus value of the demo frame `seq` rests on."""
    return round(31 + 2 * math.sin(seq * FRAME_PERIOD_S / 7))


def demo_is_spike(previous: list[float], value: float) -> bool:
    """Whether `value` is a spike after the values `previous` (the rule of the helper)."""
    recent = previous[-SPIKE_WINDOW:]
    return len(recent) >= SPIKE_MIN_PREVIOUS and value > SPIKE_FACTOR * statistics.median(recent)


def demo_spike_flags(first: int, last: int) -> list[bool]:
    """The spike flag of each demo frame from `first` to `last` (frame numbers from 1)."""
    start = max(1, first - SPIKE_WINDOW)
    values = [demo_fwhm_px(seq) for seq in range(start, last + 1)]
    flags = [demo_is_spike(values[:offset], value) for offset, value in enumerate(values)]
    return flags[first - start :]


def demo_focus_history(last: int, now_ns: int) -> FocusHistoryView | None:
    """The history of the focus values up to the frame `last`: the newest `FOCUS_POINTS` of them."""
    if last < 1:
        return None
    first = max(1, last - FOCUS_POINTS + 1)
    seqs = list(range(first, last + 1))
    values = [demo_fwhm_px(seq) for seq in seqs]
    return FocusHistoryView(
        session=1,
        index=seqs,
        seq=seqs,
        t_utc_ms=[(now_ns + round(seq * FRAME_PERIOD_S * NS_PER_S)) // 1_000_000 for seq in seqs],
        fwhm_px=values,
        fwhm_arcsec=[round(value * PLATE_SCALE_ARCSEC_PX, 4) for value in values],
        n_stars=[demo_focus_stars(seq) for seq in seqs],
        spike=demo_spike_flags(first, last),
    )


def solution_lost(t_s: float) -> bool:
    """Whether the solver finds no star field at demo time `t_s`."""
    return SOLUTION_LOST_FROM_S <= t_s % POLE_DRIFT_PERIOD_S < SOLUTION_LOST_UNTIL_S


def demo_roll_deg(t_s: float) -> float:
    """The turn of the picture about Polaris at demo time `t_s`, in degrees."""
    return 0.9 * math.sin(t_s / 57)


def demo_aim_ring(roll_deg: float) -> AimRingView:
    """Where Polaris belongs on the reticle for a picture turned by `roll_deg`.

    The ring sits on the circle around the frame center, straight below the center when the roll is
    zero. It depends on the roll only, so a lost solution keeps the ring where the last one put it.
    """
    angle = math.radians(roll_deg)
    return AimRingView(
        x_px=round(FRAME_CENTER_X + ORBIT_RADIUS_PX * math.sin(angle), 3),
        y_px=round(FRAME_CENTER_Y + ORBIT_RADIUS_PX * math.cos(angle), 3),
        source="current frame",
    )


def demo_sky(
    pole_right_px: float,
    pole_down_px: float,
    polaris_angle_deg: float,
    polaris_xy: tuple[float, float] | None = None,
) -> SkyView | None:
    """The sky view for a camera whose pole lies at an offset from the frame center.

    Polaris lies on its orbit, in the image direction `polaris_angle_deg` from straight down.
    `polaris_xy` is the pixel of Polaris, which gives the view its aim ring. The view comes from
    `build_sky_view` through a synthetic camera attitude, so the numbers are the ones that `core`
    computes for a real solution. The attitude needs the survey extra (the model of the camera
    lives there), and a demo without it shows no sky view.
    """
    try:
        from seeingmon.survey.skyview import build_sky_view
        from seeingmon.survey.wcs_fit import CameraAttitude, pixel_center
    except ImportError:
        return None
    scale_rad = PLATE_SCALE_ARCSEC_PX * RAD_PER_ARCSEC
    # The camera-frame direction of the pole, from where it falls in the image.
    pole = np.array([pole_right_px * scale_rad, pole_down_px * scale_rad, 1.0])
    pole /= np.linalg.norm(pole)
    # Polaris is SKY_COLATITUDE_DEG from the pole, toward the image direction given by the angle.
    angle = math.radians(polaris_angle_deg)
    toward = np.array([math.sin(angle), math.cos(angle), 0.0])
    toward -= pole * float(toward @ pole)
    toward /= np.linalg.norm(toward)
    # The rotation takes the CIRS pole to `pole`, and the CIRS direction of Polaris to `toward`.
    ra = math.radians(POLARIS_RA_DEG)
    cirs_z = np.array([0.0, 0.0, 1.0])
    cirs_polaris = np.array([math.cos(ra), math.sin(ra), 0.0])
    cirs = np.stack([cirs_z, cirs_polaris, np.cross(cirs_z, cirs_polaris)], axis=1)
    camera = np.stack([pole, toward, np.cross(pole, toward)], axis=1)
    attitude = CameraAttitude(
        rotation=camera @ cirs.T,
        scale_rad_px=scale_rad,
        parity=1,
        center_px=pixel_center(FRAME_WIDTH_PX, FRAME_HEIGHT_PX),
    )
    geometry = build_sky_view(
        attitude, FRAME_WIDTH_PX, FRAME_HEIGHT_PX, SKY_COLATITUDE_DEG, polaris_xy=polaris_xy
    )
    return SkyView.from_geometry(geometry)


class StarField:
    """A synthetic star field with Polaris on its orbit, as the alignment helper sees it.

    The target is where Polaris belongs when the pole sits at the center of the frame: straight
    below the center, on the orbit. The field has a fixed set of stars, spread over a wider area
    than the frame at the same density, because the mount drifts by up to a degree. `image` draws
    them with a Gaussian profile, shifted by the offset of the mount and turned about Polaris by
    the roll offset, on a sky with noise, and stretches the result like the alignment helper
    does. Positions are in pixels of the frame (`FRAME_WIDTH_PX` by `FRAME_HEIGHT_PX`), and the
    picture has the size that you ask for.
    """

    def __init__(self, seed: int = 11, stars: int = 140) -> None:
        rng = np.random.default_rng(seed)
        self.target_x = FRAME_CENTER_X
        self.target_y = round(FRAME_CENTER_Y + ORBIT_RADIUS_PX, 2)
        margin_x, margin_y = 1000.0, 500.0
        area = (FRAME_WIDTH_PX + 2 * margin_x) * (FRAME_HEIGHT_PX + 2 * margin_y)
        count = round(stars * area / (FRAME_WIDTH_PX * FRAME_HEIGHT_PX))
        magnitude = 5.0 + rng.exponential(1.7, count).clip(0, 7.5)
        flux = 10 ** (-0.4 * (magnitude - 5.0))
        x = np.concatenate(
            ([self.target_x], rng.uniform(20 - margin_x, FRAME_WIDTH_PX + margin_x - 20, count))
        )
        y = np.concatenate(
            ([self.target_y], rng.uniform(20 - margin_y, FRAME_HEIGHT_PX + margin_y - 20, count))
        )
        self._x = x
        self._y = y
        self._flux = np.concatenate(([6.0], flux))
        self.focus_reset_seq = 1  # the best focus value counts the frames from here
        self._best_cache: tuple[int, int, float | None] = (1, 0, None)  # reset, last frame, best

    def reset_focus(self, seq: int) -> None:
        """Restart the best focus value at the frame `seq`. The history stays."""
        self.focus_reset_seq = max(1, seq)

    def best_focus_px(self, last: int) -> float | None:
        """The smallest focus value that is not a spike, from the last reset to the frame `last`."""
        reset, done, best = self._best_cache
        if reset != self.focus_reset_seq or last < done:
            reset = self.focus_reset_seq
            done, best = reset - 1, None
        flags = demo_spike_flags(done + 1, last) if last > done else []
        for seq, spike in zip(range(done + 1, last + 1), flags, strict=True):
            value = demo_fwhm_px(seq)
            if not spike and (best is None or value < best):
                best = value
        self._best_cache = (reset, max(done, last), best)
        return best

    def image(
        self,
        seed: int,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        roll_deg: float = 0.0,
        size: tuple[int, int] = PREVIEW_SIZE,
        sigma_px: float = 1.5,
        saturate: float = 1.0,
    ) -> bytes:
        """A JPEG of the field. `seed` fixes the noise, so the same call gives the same bytes."""
        width, height = size
        rng = np.random.default_rng(seed)
        sky = 8.0 + 3.0 * np.linspace(0, 1, width, dtype=np.float32)[None, :]
        picture = np.broadcast_to(sky, (height, width)).astype(np.float32).copy()
        picture += rng.normal(0.0, 1.1, (height, width)).astype(np.float32)
        angle = math.radians(roll_deg)
        cos, sin = math.cos(angle), math.sin(angle)
        px_x = self._x - self.target_x
        px_y = self._y - self.target_y
        moved_x = self.target_x + dx + px_x * cos - px_y * sin
        moved_y = self.target_y + dy + px_x * sin + px_y * cos
        scale_x, scale_y = width / FRAME_WIDTH_PX, height / FRAME_HEIGHT_PX
        reach = math.ceil(5 * sigma_px)
        for index in range(len(self._x)):
            cx, cy = moved_x[index] * scale_x, moved_y[index] * scale_y
            x0, x1 = max(0, int(cx) - reach), min(width, int(cx) + reach + 1)
            y0, y1 = max(0, int(cy) - reach), min(height, int(cy) + reach + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            xs = np.arange(x0, x1, dtype=np.float32)[None, :]
            ys = np.arange(y0, y1, dtype=np.float32)[:, None]
            profile = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma_px**2))
            amplitude = 90.0 * self._flux[index] * (saturate if index == 0 else 1.0)
            picture[y0:y1, x0:x1] += (amplitude * profile).astype(np.float32)
        stretched = np.arcsinh(np.clip(picture - 6.0, 0, None) / 12.0) / math.asinh(40.0)
        pixels = (np.clip(stretched, 0.0, 1.0) * 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, "JPEG", quality=82, optimize=True)
        return buffer.getvalue()

    def frame(self, seq: int, now_ns: int = DEMO_NOW_NS) -> AlignmentFrame:
        """The live-view frame with this sequence number: the JPEG and the state beside it.

        While the solver finds no star field (`solution_lost`), the state has no `solved`, `offset`,
        and `sky`. It keeps the `reticle`, the focus, and the `aim_ring` and `last_solution` of the
        last frame that had a solution.
        """
        t = seq * FRAME_PERIOD_S
        pole_right, pole_down = pole_offset_px(t)
        roll = demo_roll_deg(t)
        # Polaris sits on its orbit around the pole, straight below it when the roll is zero.
        angle = math.radians(roll)
        dx = pole_right + ORBIT_RADIUS_PX * math.sin(angle)
        dy = pole_down + ORBIT_RADIUS_PX * (math.cos(angle) - 1.0)
        saturation = 0.0003 + 0.0016 * max(0.0, math.sin(t / 90))
        jpeg = self.image(
            seq,
            dx=dx,
            dy=dy,
            roll_deg=roll,
            sigma_px=1.4 + 0.15 * math.sin(t / 50),
            saturate=1.0 + 8 * saturation * 1000,
        )
        distance = math.hypot(dx, dy)
        counts = [round(3.0e6 * math.exp(-0.63 * i)) for i in range(HISTOGRAM_BINS)]
        counts[-1] = round(saturation * FRAME_WIDTH_PX * FRAME_HEIGHT_PX)  # the saturated pixels
        lost = solution_lost(t)
        solution_age_s = round(0.35 + 0.25 * abs(math.sin(t)), 2)
        # The frame of the last solution: a little behind this one, or the last frame before the
        # solver lost the star field.
        if lost:
            cycle_start = t - t % POLE_DRIFT_PERIOD_S
            solved_t = cycle_start + SOLUTION_LOST_FROM_S - FRAME_PERIOD_S
        else:
            solved_t = max(0.0, t - solution_age_s)
        solved_seq = round(solved_t / FRAME_PERIOD_S)
        focus_seq = max(1, max(0, seq - 1) if lost else solved_seq)  # the frame of the latest solve
        focus_px = demo_fwhm_px(focus_seq)
        best_px = self.best_focus_px(focus_seq)
        history = demo_focus_history(focus_seq, now_ns)
        solved_roll = demo_roll_deg(solved_t)
        ring = demo_aim_ring(solved_roll if lost else roll).model_copy(
            update={
                "source": "last solution" if lost else "current frame",
                "age_s": round(t - solved_t, 2),
                "solution_frame_seq": solved_seq,
            }
        )
        frame_t_utc = utc_ns_to_iso(now_ns + round(t * NS_PER_S), digits=3)
        reasons = {
            "solved": LOST_REASON,
            "sky": LOST_REASON,
            "offset": "the offset needs a current solution",
        }
        state = AlignmentState(
            active=True,
            t_utc=frame_t_utc,
            frame=AlignmentFrameInfo(
                seq=seq,
                width_px=FRAME_WIDTH_PX,
                height_px=FRAME_HEIGHT_PX,
                readout_mode="bin2",
                exposure_s=FRAME_PERIOD_S,
                gain=120,
                plate_scale_arcsec_px=PLATE_SCALE_ARCSEC_PX,
            ),
            target=TargetView(x_px=self.target_x, y_px=self.target_y, roll_deg=12.3),
            solved=None
            if lost
            else SolvedView(
                x_px=round(self.target_x + dx, 2),
                y_px=round(self.target_y + dy, 2),
                roll_deg=round(12.3 + roll, 3),
                n_matched=round(40 + 3 * math.sin(t / 9)),
                rms_arcsec=round(2.9 + 0.4 * math.sin(t / 13), 2),
                age_s=solution_age_s,
            ),
            offset=None
            if lost
            else OffsetView(
                dx_px=round(dx, 2),
                dy_px=round(dy, 2),
                distance_px=round(distance, 2),
                dx_arcsec=round(dx * PLATE_SCALE_ARCSEC_PX, 1),
                dy_arcsec=round(dy * PLATE_SCALE_ARCSEC_PX, 1),
                distance_arcsec=round(distance * PLATE_SCALE_ARCSEC_PX, 1),
                roll_deg=round(roll, 3),
            ),
            focus=FocusView(
                fwhm_px=focus_px,
                best_fwhm_px=best_px,
                n_stars=demo_focus_stars(focus_seq),
                fwhm_arcsec=round(focus_px * PLATE_SCALE_ARCSEC_PX, 4),
                best_fwhm_arcsec=(
                    None if best_px is None else round(best_px * PLATE_SCALE_ARCSEC_PX, 4)
                ),
                spike=False if history is None else history.spike[-1],
                frame_seq=focus_seq,
                history=history,
            ),
            histogram=HistogramView(counts=counts, min_dn=0.0, max_dn=65535.0),
            saturation=SaturationView(fraction=round(saturation, 5), warning=saturation > 0.001),
            reticle=ReticleView(
                x_px=FRAME_CENTER_X,
                y_px=FRAME_CENTER_Y,
                radius_px=round(ORBIT_RADIUS_PX, 3),
                polaris_colatitude_deg=SKY_COLATITUDE_DEG,
            ),
            sky=None
            if lost
            else demo_sky(
                pole_right,
                pole_down,
                roll,
                polaris_xy=(round(self.target_x + dx, 3), round(self.target_y + dy, 3)),
            ),
            aim_ring=ring,
            last_solution=LastSolutionView(
                frame_seq=solved_seq,
                t_utc=utc_ns_to_iso(now_ns + round(solved_t * NS_PER_S), digits=3),
                age_s=round(t - solved_t, 2),
                roll_deg=round(12.3 + solved_roll, 3),
                polaris_colatitude_deg=SKY_COLATITUDE_DEG,
                n_matched=round(40 + 3 * math.sin(solved_t / 9)),
                rms_arcsec=round(2.9 + 0.4 * math.sin(solved_t / 13), 2),
                solver="tracker",
            ),
            timing=TimingView(
                frame_seq=seq,
                frame_t_utc=frame_t_utc,
                frame_age_s=round(0.32 + 0.08 * abs(math.sin(t / 3)), 3),
                receive_lag_s=round(0.11 + 0.02 * abs(math.sin(t / 4)), 3),
                preview_s=round(0.19 + 0.05 * abs(math.sin(t / 5)), 3),
                solution_frame_seq=max(0, seq - 1) if lost else solved_seq,
                solve_elapsed_s=round(0.45 + 0.15 * abs(math.sin(t / 6)), 3),
            ),
            quality=reasons if lost else {},
        )
        return AlignmentFrame(state, jpeg)


# --- The video of Polaris ----------------------------------------------------------------------

_ERF = np.vectorize(math.erf, otypes=[np.float64])
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def _smooth_seeing_arcsec(t_s: float) -> float:
    """The seeing of the video at `t_s` seconds: a slow wobble around `DEMO_SEEING_ARCSEC`."""
    wobble = 0.10 * math.sin(2.0 * math.pi * t_s / 47.0) + 0.05 * math.sin(
        2.0 * math.pi * t_s / 13.0 + 1.0
    )
    return DEMO_SEEING_ARCSEC * (1.0 + wobble)


def demo_live_seeing(t_s: float) -> LiveSeeingView | None:
    """The rolling seeing value at `t_s` seconds of demo time, or `None` in the first seconds.

    A new value comes every `LIVE_EVERY_S` seconds, as `core` makes one, and it follows the
    wobble that the video shows, plus the scatter of an estimate over a short span. The first
    value comes after `LIVE_MIN_SPAN_S` seconds, as in `core`.
    """
    step = max(0, math.floor(t_s / LIVE_EVERY_S))
    t_end = step * LIVE_EVERY_S
    if t_end < LIVE_MIN_SPAN_S:
        return None
    rng = random.Random(5000 + step)
    seeing = _smooth_seeing_arcsec(t_end) * (1.0 + rng.gauss(0.0, 0.035))
    span = min(LIVE_SPAN_S, t_end)
    dropped = rng.choice([0, 0, 1, 2, 3])
    frames = round(span * CAMERA_FPS) - dropped
    usable = frames - rng.choice([0, 1, 2, 4])

    def r0_cm(fwhm: float) -> float:
        return round(0.98 * 500e-9 / (fwhm * RAD_PER_ARCSEC) * 100, 2)

    structure = seeing * (1.0 + rng.gauss(0.0, 0.04))
    rms = 0.48 * seeing
    # The second moment of a star that the pixels integrate has the variance of the star plus 1/12.
    width = FWHM_PER_SIGMA * math.sqrt(STAR_SIGMA_PX**2 + 1.0 / 12.0) * FAST_PLATE_SCALE_ARCSEC_PX
    return LiveSeeingView(
        t_utc_ns=DEMO_NOW_NS + round(t_end * NS_PER_S),
        span_s=round(span, 3),
        n_frames=frames,
        n_usable=usable,
        valid_fraction=round(usable / (frames + dropped), 4),
        seeing_fwhm_arcsec=round(seeing, 3),
        seeing_fwhm_structure_arcsec=round(structure, 3),
        r0_cm=r0_cm(seeing),
        r0_structure_cm=r0_cm(structure),
        image_motion_rms_x_arcsec=round(rms * rng.uniform(0.95, 1.1), 3),
        image_motion_rms_y_arcsec=round(rms * rng.uniform(0.85, 1.0), 3),
        width_fwhm_arcsec=round(width, 3),
        stream_id=1,
        readout_mode="bin1",
        exposure_us=2000,
        flags=[],
        quality={},
    )


class PolarisSky:
    """A synthetic video of Polaris: one `FrameSlot` for each call of `next_frame`.

    The star is a Gaussian of 0.6 pixels sigma (1.4 pixels FWHM), integrated over the pixels, on a
    sky of 480 counts. It moves along each axis as a random process that forgets its past in about
    0.12 s, with a spread of `IMAGE_MOTION_PX_PER_ARCSEC` pixels for each arcsecond of seeing, on
    top of a slow drift of a third of a pixel. Its brightness flickers by 3.5% rms in a few
    hundredths of a second, and it drifts by 3% over half a minute. The pixels carry photon noise
    and read noise, and the counts follow a 12-bit ADC in a 16-bit container, like the camera.
    `time_offset_s` is the video time of the first frame. The same seed gives the same video.
    """

    def __init__(
        self,
        seed: int = 17,
        *,
        period_s: float = POLARIS_PERIOD_S,
        time_offset_s: float = 0.0,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._period_s = period_s
        self._offset_s = time_offset_s
        self._index = 0
        self._tilt = self._rng.standard_normal(2)
        self._scintillation = float(self._rng.standard_normal())
        self._edges = np.arange(POLARIS_SIZE + 1, dtype=np.float64) - 0.5
        peak_share = float(_ERF(np.array([0.5 / (STAR_SIGMA_PX * math.sqrt(2.0))]))[0]) ** 2
        self._flux_dn = STAR_PEAK_DN / peak_share  # the flux that gives the peak at a pixel center

    def _weights(self, center: float, sigma: float) -> npt.NDArray[np.float64]:
        scaled = (self._edges - center) / (sigma * math.sqrt(2.0))
        weights: npt.NDArray[np.float64] = np.diff(0.5 * (1.0 + _ERF(scaled)))
        return weights

    def next_frame(self) -> tuple[FrameSlot, float]:
        """The next frame of the video, and its time in seconds."""
        t_s = self._offset_s + self._index * self._period_s
        rng = self._rng
        tilt_memory = math.exp(-self._period_s / TILT_TAU_S)
        self._tilt = tilt_memory * self._tilt + math.sqrt(1.0 - tilt_memory**2) * (
            rng.standard_normal(2)
        )
        flicker_memory = math.exp(-self._period_s / SCINTILLATION_TAU_S)
        self._scintillation = flicker_memory * self._scintillation + math.sqrt(
            1.0 - flicker_memory**2
        ) * float(rng.standard_normal())
        motion_px = IMAGE_MOTION_PX_PER_ARCSEC * _smooth_seeing_arcsec(t_s)
        center = POLARIS_SIZE / 2.0
        x = center + 0.35 * math.sin(2.0 * math.pi * t_s / 41.0) + motion_px * self._tilt[0]
        y = center + 0.30 * math.sin(2.0 * math.pi * t_s / 29.0 + 1.0) + motion_px * self._tilt[1]
        brightness = max(
            0.2,
            1.0
            + SCINTILLATION_RMS * self._scintillation
            + 0.03 * math.sin(2.0 * math.pi * t_s / 31.0),
        )
        sigma = STAR_SIGMA_PX * (1.0 + 0.08 * math.sin(2.0 * math.pi * t_s / 37.0))
        shape = np.outer(self._weights(y, sigma), self._weights(x, sigma))
        star = brightness * self._flux_dn * shape
        electrons = (SKY_DN + star) / DN_PER_ADU * E_PER_ADU
        noisy = rng.poisson(electrons) + rng.normal(0.0, READ_NOISE_E, electrons.shape)
        adu = np.clip(np.rint(noisy / E_PER_ADU), 0, 4095)
        counts = (adu * DN_PER_ADU).astype(np.uint16)
        roi = POLARIS_ROI
        found = StarState(
            found=True,
            x_px=roi.x + x,
            y_px=roi.y + y,
            peak_fraction=float(counts.max()) / FULL_SCALE_DN,
            edge_distance_px=min(x, y, POLARIS_SIZE - 1.0 - x, POLARIS_SIZE - 1.0 - y),
        )
        count = round(t_s * CAMERA_FPS)  # the camera frame that this video frame shows
        slot = FrameSlot(
            data=counts,
            stream_id=1,
            t_utc_ns=DEMO_NOW_NS + round(count / CAMERA_FPS * NS_PER_S),
            mode="bin1",
            exposure_us=2000,
            gain=0,
            adc_bits=12,
            roi=roi,
            star=found,
            count=count,
        )
        self._index += 1
        return slot, t_s


def demo_dark_library(now_ns: int, *, seed: int = 31) -> tuple[list[DarkSetView], DarkModelView]:
    """The six sets of the demo library, the newest first, and the model that they fit.

    The dark current doubles every 6 C, with a little scatter. The numbers are made up.
    """
    rng = random.Random(seed)
    reference_c, rate_ref, doubling_c = 20.0, 0.118, 6.1
    sets: list[DarkSetView] = []
    for temperature, age_days in sorted(DEMO_DARK_SETS, key=lambda item: item[1]):
        when_ns = now_ns - round(age_days * DAY_NS)
        stamp = utc_ns_to_iso(when_ns, digits=0).replace("-", "").replace(":", "")
        scatter = 1.0 + rng.uniform(-0.04, 0.04)
        rate = rate_ref * 2 ** ((temperature - reference_c) / doubling_c) * scatter
        sets.append(
            DarkSetView(
                name=f"dark-{stamp}-bin2-g120.fits",
                t_utc=utc_ns_to_iso(when_ns, digits=0),
                age_days=age_days,
                temperature_c=temperature,
                temperature_spread_c=round(rng.uniform(0.2, 0.6), 2),
                exposure_s=30.0,
                n_frames=9,
                n_bias_frames=9,
                rate_e_per_s=round(rate, 4),
                hot_pixels=rng.randint(160, 240),
            )
        )
    model = DarkModelView(
        reference_c=reference_c,
        rate_ref_e_per_s=rate_ref,
        doubling_c=doubling_c,
        doubling_fitted=True,
        rms_log2=0.07,
        n_sets=len(sets),
    )
    return sets, model


class DemoCore(FakeCoreClient):
    """A fake `core` whose live view streams a synthetic star field while the scheduler aligns.

    The commands work as in `FakeCoreClient`. While the fake scheduler is in `align`, the stream
    yields a frame every `period_s` seconds. In any other state the stream stays open and sends
    nothing, so the UI shows that it waits for frames. The video of Polaris runs in the `auto` and
    `safe` states, one frame every `polaris_period_s` seconds of wall time (the video itself
    advances 50 ms for each frame). The dark library starts with six sets (`demo_dark_library`),
    or empty with `library=False`, and a dark session follows `dark_script` (`DEMO_DARK_SCRIPT`
    by default). The flat library starts with two flats (`DEMO_FLATS`), or empty with
    `library=False`, and a flat session follows `flat_script` (`DEMO_FLAT_SCRIPT` by default).
    Without a dark set, no flat session can start.

    The status plays the activity of a scheduler on a short cycle (see `demo_activity`): the
    phases of `auto` follow the monotonic clock of the fake `core`, and the gate of `safe` opens
    after `GATE_OPEN_S`, which moves the state to `auto`.
    """

    def __init__(
        self,
        clock: Clock | None = None,
        *,
        period_s: float = FRAME_PERIOD_S,
        polaris_period_s: float = POLARIS_PERIOD_S,
        field: StarField | None = None,
        dark_script: DarkScript | None = None,
        flat_script: FlatScript | None = None,
        library: bool = True,
    ) -> None:
        super().__init__(
            clock=clock,
            frames=self._stream,
            polaris=self._polaris_stream,
            instance="demo-core",
            state="auto",
            dark_script=dark_script or DEMO_DARK_SCRIPT,
            flat_script=flat_script or DEMO_FLAT_SCRIPT,
        )
        self.dark.sensor_temperature_c = DEMO_SENSOR_TEMPERATURE_C
        self.flat.sensor_temperature_c = DEMO_SENSOR_TEMPERATURE_C
        if library:
            self.dark.sets, self.dark.model = demo_dark_library(self._clock.utc_ns())
            for index, (age_days, look) in enumerate(DEMO_FLATS):
                newest = index == len(DEMO_FLATS) - 1
                self.flat.seed(age_days=age_days, look=look, active=newest, second_set=newest)
        self._period_s = period_s
        self._polaris_period_s = polaris_period_s
        self._started_ns = self._clock.monotonic_ns()
        self._field = field or StarField()
        self._seq = 0
        self._latest: AlignmentState | None = None
        self._since_mono = self._clock.monotonic_ns()  # when the fake scheduler entered its state

    def alignment_reset_focus(self) -> None:
        """Restart the best focus value at the frame that the stream sends next."""
        super().alignment_reset_focus()
        self._field.reset_focus(self._seq + 1)

    # --- The activity ---

    def _transition(self, state: str, reason: str = "a fake transition") -> None:
        super()._transition(state, reason)
        self._since_mono = self._clock.monotonic_ns()

    def _settle_dark(self) -> None:
        super()._settle_dark()
        if self._state == "safe" and self._elapsed_s() >= demo_activity.GATE_OPEN_S:
            self._transition("auto", "the sky is dark enough")  # the demo sky is dark

    def _elapsed_s(self) -> float:
        """The seconds that the fake scheduler has spent in its state."""
        return max(0.0, (self._clock.monotonic_ns() - self._since_mono) / NS_PER_S)

    def _reason_text(self) -> str | None:
        if self._reason == "a fake transition":
            return demo_activity.STATE_REASONS.get(self._state)
        return words.state_reason_text(self._reason)

    def _activity_view(self, now_ns: int) -> ActivityView | None:
        elapsed, reason = self._elapsed_s(), self._reason_text()
        if self._state == "auto":
            return demo_activity.auto_activity(elapsed, now_ns, reason)
        if self._state == "safe":
            return demo_activity.safe_activity(elapsed, now_ns, reason)
        if self._state == "align":
            return demo_activity.align_activity(elapsed, now_ns, reason)
        if self._state == "commission":
            return demo_activity.commission_activity(elapsed, now_ns, reason, self.dark.task())
        return demo_activity.paused_activity(elapsed, now_ns, reason)

    def _fault_view(self, now_ns: int) -> FaultView:
        if self._state != "auto":
            return FaultView()
        return demo_activity.auto_fault(self._elapsed_s(), now_ns)

    async def _stream(self) -> AsyncIterator[AlignmentFrame]:
        while True:
            if self.state != "align":
                await asyncio.sleep(IDLE_POLL_S)
                continue
            self._seq += 1
            frame = await asyncio.to_thread(self._field.frame, self._seq)
            self._latest = frame.state
            yield frame
            await asyncio.sleep(self._period_s)

    def alignment_state(self) -> AlignmentState:
        self._check()
        if self.state != "align":
            return AlignmentState(active=False)
        return self._latest or AlignmentState(active=True)

    def _video_time_s(self) -> float:
        """The seconds since this fake core started, which is the time of the video."""
        return (self._clock.monotonic_ns() - self._started_ns) / NS_PER_S

    @staticmethod
    def _polaris_frame(sky: PolarisSky, renderer: PolarisRenderer) -> PolarisFrame:
        slot, t_s = sky.next_frame()
        return renderer.render(slot, demo_live_seeing(t_s))

    async def _polaris_stream(self) -> AsyncIterator[PolarisFrame]:
        sky = PolarisSky(time_offset_s=self._video_time_s())
        renderer = PolarisRenderer(scale_for=lambda mode: FAST_PLATE_SCALE_ARCSEC_PX)
        due = time.monotonic()
        while True:
            if self.state not in ("auto", "safe"):  # the camera shows no fast stream otherwise
                await asyncio.sleep(IDLE_POLL_S)
                due = time.monotonic()
                continue
            yield await asyncio.to_thread(self._polaris_frame, sky, renderer)
            due = max(due + self._polaris_period_s, time.monotonic())  # keep the rate, never burst
            await asyncio.sleep(max(0.0, due - time.monotonic()))

    def live_seeing(self) -> LiveSeeingView | None:
        self._check()
        return demo_live_seeing(self._video_time_s())


# --- Files -------------------------------------------------------------------------------------


def _stamp(t_utc_ns: int) -> str:
    moment = datetime.fromtimestamp(t_utc_ns / NS_PER_S, UTC)
    return moment.strftime("%Y%m%dT%H%M%S") + f".{(t_utc_ns // 1_000_000) % 1000:03d}Z"


def fits_bytes(width: int, height: int, seed: int, t_utc_ns: int) -> bytes:
    """A small, valid FITS file: a synthetic 16-bit image with a few cards of header."""
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                   16",
        "NAXIS   =                    2",
        f"NAXIS1  = {width:>20}",
        f"NAXIS2  = {height:>20}",
        "BZERO   =                32768",
        "BSCALE  =                    1",
        f"DATE-OBS= '{utc_ns_to_iso(t_utc_ns, digits=3)}'",
        "COMMENT Synthetic demo frame. It shows no real sky.",
        "END",
    ]
    header = "".join(card.ljust(80) for card in cards).encode("ascii")
    header += b" " * (-len(header) % FITS_BLOCK)
    rng = np.random.default_rng(seed)
    pixels = rng.normal(1200, 25, (height, width)).clip(0, 65535)
    yy, xx = np.mgrid[0:height, 0:width]
    pixels += 30_000 * np.exp(-((xx - width / 2) ** 2 + (yy - height / 2) ** 2) / 8.0)
    stored = (pixels.clip(0, 65535).astype(np.int32) - 32768).astype(">i2")
    data = stored.tobytes()
    data += b"\0" * (-len(data) % FITS_BLOCK)
    return header + data


def write_demo_images(layout: DataLayout, field: StarField, now_ns: int, *, seed: int = 5) -> int:
    """Write a preview every 30 minutes of each dark hour, and a FITS frame for every fourth one.

    Returns the number of previews. The files carry the names that `core` gives them.
    """
    count = 0
    step_ns = 1800 * NS_PER_S
    t_ns = now_ns - HISTORY_S * NS_PER_S
    t_ns += -t_ns % step_ns
    while t_ns < now_ns - 120 * NS_PER_S:
        if sun_elevation_deg(t_ns) < -6.0:
            kind = "survey" if count % 4 == 0 else "preview"
            stamp = _stamp(t_ns)
            folder = layout.previews_dir / stamp[0:4] / stamp[4:6] / stamp[6:8]
            folder.mkdir(parents=True, exist_ok=True)
            cover = _cloud_cover(t_ns, now_ns)
            jpeg = field.image(
                seed + count,
                dx=8 * math.sin(count),
                dy=6 * math.cos(count / 2),
                size=IMAGE_SIZE,
                sigma_px=1.3 + 0.8 * cover,
                saturate=1.0 - 0.8 * cover,
            )
            (folder / f"{kind}-{stamp}.jpg").write_bytes(jpeg)
            if kind == "survey":
                fits_folder = layout.survey_dir / stamp[0:4] / stamp[4:6] / stamp[6:8]
                fits_folder.mkdir(parents=True, exist_ok=True)
                (fits_folder / f"{stamp}.fits").write_bytes(
                    fits_bytes(*FITS_SIZE, seed=seed + count, t_utc_ns=t_ns)
                )
            count += 1
        t_ns += step_ns
    return count


def write_demo_store(layout: DataLayout, now_ns: int = DEMO_NOW_NS, *, seed: int = 2026) -> int:
    """Create the folders, write the records and the images, and return the number of records."""
    layout.create()
    records = demo_records(now_ns, seed=seed)
    with Store.open(layout.db_path) as store:
        store.write_many(records)
    write_demo_images(layout, StarField(), now_ns)
    return len(records)


# --- The app -----------------------------------------------------------------------------------


@dataclass
class DemoApp:
    """The app of `seeingmon web --demo`, its settings, and what to release at the end."""

    app: FastAPI
    settings: WebSettings
    core: DemoCore
    directory: Path
    close: Callable[[], None]


def build_demo(
    settings: WebSettings,
    *,
    profile: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    directory: Path | None = None,
    now_ns: int = DEMO_NOW_NS,
    seed: int = 2026,
    frame_period_s: float = FRAME_PERIOD_S,
    polaris_period_s: float = POLARIS_PERIOD_S,
) -> DemoApp:
    """Build the demo app. `profile` and `config` are what `/profile` and `/config` serve.

    Pass `directory` to keep the data in a folder that you manage. Without it, the demo writes to
    a temporary folder and `close` removes it. The demo token is `DEMO_TOKEN`.
    """
    from seeingmon.services.web.app import create_app
    from seeingmon.services.web.images import ImageStore

    owned = directory is None
    root = Path(tempfile.mkdtemp(prefix="seeingmon-demo-")) if directory is None else directory
    layout = DataLayout(root)
    reader: StoreReader | None = None
    core: DemoCore | None = None

    def close() -> None:
        if core is not None:
            core.close()
        if reader is not None:
            reader.close()
        if owned:
            shutil.rmtree(root, ignore_errors=True)

    try:
        write_demo_store(layout, now_ns, seed=seed)
        reader = StoreReader.open(layout.db_path)
        clock = DemoClock(now_ns)
        core = DemoCore(clock, period_s=frame_period_s, polaris_period_s=polaris_period_s)
        app = create_app(
            settings,
            reader,
            ImageStore(layout, settings.images),
            core,
            clock=clock,
            token_hash=hash_token(DEMO_TOKEN, params=ScryptParams(ln=12, r=8, p=1)),
            profile=profile,
            config=config,
            station_id=DEMO_STATION,
            demo=True,
        )
    except BaseException:
        close()
        raise
    return DemoApp(app, settings, core, root, close)
