"""The fast path on the owner's recordings.

The tests skip when the recordings are not configured. Each test checks a property that real
data must have, with bounds that leave room for another night. `docs/recordings-validation.md`
lists the numbers behind the bounds.

Nothing here prints a path, a file name, header text, or a serial number, and no test reads a
sidecar: the replay gets the readout mode, the exposure, and the gain from options. Failure
messages use fixed words, because pytest shows the values of a failed comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.clock import VirtualClock
from seeingmon.drivers.replay import ReplayFinishedError, create
from seeingmon.fastpath import FastPathConfig, create_fast_analyzer
from seeingmon.fastpath.kernel import FLAG_SATURATED
from seeingmon.frames import PixelFormat, StreamConfig
from seeingmon.profile import Profile, derived
from seeingmon.recordings.ser import SerFile
from seeingmon.records import SeeingWindowRecord
from tests.recordings.real import is_owner_capture, ser_files_or_skip

pytestmark = pytest.mark.recordings

EXPOSURE_US = 10_000
GAIN = 100
BLOCK_S = 10.0
SIDEREAL_ARCSEC_PER_S = 15.04  # the rate of the sky, times the pole distance of Polaris below


@dataclass(frozen=True)
class Capture:
    """One replayed recording: the per-frame rows and the windows that the analyzer closed."""

    rows: npt.NDArray[Any]
    windows: tuple[SeeingWindowRecord, ...]
    plate_scale: float

    @property
    def seconds(self) -> npt.NDArray[np.float64]:
        stamps = self.rows["t_utc_ns"].astype(np.float64)
        return np.asarray((stamps - stamps[0]) * 1e-9, dtype=np.float64)

    @property
    def duration_s(self) -> float:
        return float(self.seconds[-1])

    @property
    def full_windows(self) -> list[SeeingWindowRecord]:
        return [window for window in self.windows if "partial" not in window.flags]


def replay(path: Path, profile: Profile) -> Capture:
    """Replay a capture through the production analyzer, with the default configuration."""
    driver = create(
        profile=None,
        clock=VirtualClock(),
        options={
            "path": path,
            "mode": "bin2",
            "exposure_us": EXPOSURE_US,
            "gain": GAIN,
            "rate": "max",
            "sidecar": False,  # this module reads no sidecar
        },
    )
    driver.open()
    stream = driver.configure(
        StreamConfig("bin2", EXPOSURE_US, GAIN, pixel_format=PixelFormat.RAW8)
    )
    driver.start()
    analyzer = create_fast_analyzer(profile, FastPathConfig(), "recordings")
    analyzer.begin_stream(stream)
    windows: list[SeeingWindowRecord] = []
    while True:
        try:
            frame = driver.read_frame(1.0)
        except ReplayFinishedError:
            break
        windows.extend(analyzer.push(frame).windows)
    windows.extend(analyzer.flush())
    driver.close()
    rows = analyzer.drain_metrics()
    assert rows is not None, "the analyzer produced no per-frame rows"
    plate = derived.plate_scale_arcsec_per_px(profile.mode("bin2"), profile.optics)
    return Capture(rows, tuple(windows), plate)


@pytest.fixture(scope="module")
def captures(recordings_dir: Path, profile: Profile) -> list[Capture]:
    """The owner's captures, replayed once for the whole module, the shortest first."""
    found: list[tuple[int, Path]] = []
    for path in ser_files_or_skip(recordings_dir):
        with SerFile(path) as ser:
            if is_owner_capture(ser):
                found.append((ser.frame_count, path))
    if not found:
        pytest.skip("no recording has the 8-bit mono 320 x 240 layout")
    return [replay(path, profile) for _, path in sorted(found)]


def detrended_blocks(capture: Capture, block_s: float) -> list[tuple[float, float]]:
    """The variance of each axis (arcsec^2) in blocks, after a quadratic fit per block and axis."""
    seconds = capture.seconds
    found = np.isfinite(capture.rows["cx_px"])
    start = 0.0
    out: list[tuple[float, float]] = []
    while start + block_s <= seconds[-1] + 0.5 * block_s * 0.01:
        sel = found & (seconds >= start) & (seconds < start + block_s)
        if int(sel.sum()) > 0.9 * block_s * 97.0:
            variances = []
            for axis in ("cx_px", "cy_px"):
                position = capture.rows[axis][sel].astype(np.float64) * capture.plate_scale
                local = seconds[sel] - seconds[sel].mean()
                residual = position - np.polyval(np.polyfit(local, position, 2), local)
                variances.append(float(residual.var()))
            out.append((variances[0], variances[1]))
        start += block_s
    return out


def test_the_star_is_found_in_nearly_every_frame_and_rarely_saturates(
    captures: list[Capture],
) -> None:
    for capture in captures:
        found = float(np.isfinite(capture.rows["cx_px"]).mean())
        saturated = float(((capture.rows["flags"] & FLAG_SATURATED) != 0).mean())
        assert found > 0.999, "the star is missing from more than 0.1% of the frames"
        assert saturated < 0.005, "more than 0.5% of the frames saturate"
        for window in capture.windows:
            assert "saturated" not in window.flags, "a window carries the saturated flag"


def test_the_centroid_drifts_at_the_sidereal_rate_of_polaris(captures: list[Capture]) -> None:
    """The drift checks the plate scale: the star circles the pole at 15.04 arcsec/s x sin(d).

    The pole distance d of 0.62 degrees gives 0.163 arcsec/s. The bounds leave 15% for the plate
    scale, the pole distance of the night, and a mount that moves.
    """
    expected = SIDEREAL_ARCSEC_PER_S * math.sin(math.radians(0.62))
    for capture in captures:
        found = np.isfinite(capture.rows["cx_px"])
        seconds = capture.seconds[found]
        rate_x = np.polyfit(seconds, capture.rows["cx_px"][found].astype(np.float64), 1)[0]
        rate_y = np.polyfit(seconds, capture.rows["cy_px"][found].astype(np.float64), 1)[0]
        drift = float(np.hypot(rate_x, rate_y)) * capture.plate_scale
        assert 0.85 * expected < drift < 1.15 * expected, "the drift is not the sidereal rate"


def test_the_drop_count_follows_the_gaps_between_the_timestamps(
    captures: list[Capture],
) -> None:
    for capture in captures:
        stamps = capture.rows["t_utc_ns"].astype(np.int64)
        interval = np.diff(stamps)
        period = float(np.median(interval))
        lost = np.where(interval > 1.5 * period, np.rint(interval / period) - 1, 0)
        total = int(lost.sum())
        assert total <= 10, "a capture lost more than 10 frames"
        assert int(capture.rows["dropped_before"].sum()) == total
        assert sum(window.n_dropped for window in capture.windows) == total
        assert sum(window.n_frames for window in capture.windows) == len(capture.rows)


def test_full_windows_last_sixty_seconds_and_only_the_remainder_is_partial(
    captures: list[Capture],
) -> None:
    config = FastPathConfig()
    for capture in captures:
        full = capture.full_windows
        partial = [window for window in capture.windows if "partial" in window.flags]
        assert len(full) >= 1, "a capture of 60 s or more closes at least one full window"
        assert len(partial) <= 1, "only the last window is partial"
        for window in full:
            assert window.duration_s == pytest.approx(config.window_s, abs=0.1)
            assert window.valid_fraction > 0.99
            assert window.r0_cm is not None, "a full window has no seeing estimate"
        for window in partial:
            if window.duration_s < config.min_window_s:
                assert window.r0_cm is None, "a window under the minimum has seeing statistics"


def test_the_motion_is_stationary_between_ten_second_blocks(captures: list[Capture]) -> None:
    """The variance of 10 s blocks scatters by 10 to 30% and has no outlier beyond one burst.

    A stationary Gaussian process with the spectrum of these data scatters by 7% in blocks of
    this length. The recordings scatter by 13%, or 29% with the burst block, because the
    turbulence changes during a capture. The bounds catch a series that is far too steady or
    that jumps, and they let the seeing change.
    """
    for capture in captures:
        blocks = detrended_blocks(capture, BLOCK_S)
        assert len(blocks) >= 5, "a capture has fewer than five 10 s blocks"
        variance = np.asarray([0.5 * (x + y) for x, y in blocks])
        scatter = float(variance.std() / variance.mean())
        assert 0.05 < scatter < 0.6, "the block variances scatter too little or too much"
        assert variance.max() < 4.0 * np.median(variance), "a block is a large outlier"
        assert variance.min() > 0.25 * np.median(variance), "a block is far below the median"


def test_the_spectrum_falls_with_frequency_and_levels_off_below_the_nyquist_limit(
    captures: list[Capture],
) -> None:
    for capture in captures:
        for window in capture.full_windows:
            assert window.motion_psd_freq_hz is not None
            assert window.motion_psd_x_arcsec2_per_hz is not None
            assert window.motion_psd_y_arcsec2_per_hz is not None
            freq = np.asarray(window.motion_psd_freq_hz)
            psd = 0.5 * (
                np.asarray(window.motion_psd_x_arcsec2_per_hz)
                + np.asarray(window.motion_psd_y_arcsec2_per_hz)
            )
            nyquist = 0.5 * float(window.frame_rate_hz or 0.0)
            assert freq[-1] < nyquist, "a bin lies above the Nyquist frequency"
            low = float(psd[freq < 2.0].mean())
            middle = float(psd[(freq > 5.0) & (freq < 10.0)].mean())
            high = psd[freq > 20.0]
            assert low > 2.0 * middle > 0.0, "the spectrum does not fall from 1 Hz to 8 Hz"
            assert low > 5.0 * float(high.mean()), "the spectrum does not fall to the floor"
            assert float(high.max()) < 2.5 * float(high.min()), "the floor above 20 Hz is not flat"


def test_the_two_estimators_agree_within_a_third_and_the_assumptions_are_stored(
    captures: list[Capture],
) -> None:
    """The default wind of 10 m/s puts the structure estimate 17 to 34% above the variance one.

    On average the two agree within 4% at an assumed wind of 2 m/s, and the gap shows how far
    the single frozen layer is from these data. The bound here is the loose one that holds at
    the default.
    """
    for capture in captures:
        for window in capture.full_windows:
            assert window.r0_cm is not None
            assert window.r0_structure_cm is not None
            assert 4.0 < window.r0_cm < 40.0, "r0 is outside the plausible range"
            ratio = window.r0_structure_cm / window.r0_cm
            assert 0.67 < ratio < 1.5, "the two estimators differ by more than a third"
            assert window.assumed_wind_ms == 10.0
            assert window.outer_scale_m == 20.0
            assert window.exposure_correction_factor is not None
            assert window.exposure_correction_factor > 1.2, "the 10 ms exposure is not corrected"
            assert window.zenith_angle_deg is None, "the replay supplies no zenith angle"


def test_the_modeled_centroid_noise_is_a_small_part_of_the_motion(captures: list[Capture]) -> None:
    for capture in captures:
        for window in capture.full_windows:
            assert window.centroid_noise_px is not None
            assert window.image_motion_rms_x_arcsec is not None
            assert window.image_motion_rms_y_arcsec is not None
            noise = (window.centroid_noise_px * capture.plate_scale) ** 2
            motion = 0.5 * (
                window.image_motion_rms_x_arcsec**2 + window.image_motion_rms_y_arcsec**2
            )
            assert noise < 0.15 * motion, "the subtracted noise is over 15% of the variance"


def test_the_pixel_phase_bias_is_small_for_the_defocused_star(captures: list[Capture]) -> None:
    """A star that is 3 to 4 pixels wide has a centroid gain near 1 at every sub-pixel phase.

    The notes predict a gain between 0.53 and 1.48 for an in-focus bin2 star, which makes the
    sub-pixel positions bunch up (a first harmonic of the density of about 0.5). The first
    harmonic of these recordings is 0.08 to 0.11.
    """
    checked = 0
    for capture in captures:
        for axis in ("cx_px", "cy_px"):
            position = capture.rows[axis].astype(np.float64)
            position = position[np.isfinite(position)]
            if float(position.max() - position.min()) < 3.0:
                continue  # the star must cross a few pixels to fill every phase
            harmonic = 2.0 * abs(complex(np.mean(np.exp(2j * np.pi * np.mod(position, 1.0)))))
            assert harmonic < 0.25, "the sub-pixel positions bunch up"
            checked += 1
    assert checked > 0, "no recording moves the star across enough pixels"


def test_the_vibration_burst_of_the_long_capture_is_flagged(captures: list[Capture]) -> None:
    long_captures = [capture for capture in captures if capture.duration_s > 280.0]
    if not long_captures:
        pytest.skip("no capture is long enough to hold the burst at 270 s")
    windows = long_captures[-1].full_windows
    flagged = [window for window in windows if "vibration" in window.flags]
    assert len(flagged) == 1, "exactly one window holds the vibration burst"
    lines = flagged[0].vibration_lines_hz
    assert lines is not None, "the flagged window lists no line"
    assert any(17.3 < line < 17.7 for line in lines), "the line is not near 17.5 Hz"
