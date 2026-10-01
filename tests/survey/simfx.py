"""Simulated cameras for the survey tests: a covered camera and a camera that sees stars.

The sim driver renders a *covered* camera when it has no stars and a sky that is far too faint
to see: every pixel then holds the bias, the dark current at the sensor temperature, the hot
pixels, and the noise. The tests use it for the dark sets. The synthetic site is the sim's own
(latitude 55, longitude 0), which is not a real station.
"""

from __future__ import annotations

import numpy as np

from seeingmon.clock import Clock
from seeingmon.drivers.base import CameraCaps, CameraInfo, RecoveryLevel
from seeingmon.drivers.sim import (
    HotPixelConfig,
    SimDriver,
    SimOptions,
    StarField,
    create,
)
from seeingmon.frames import ActiveStream, Frame, Roi, StreamConfig, StreamKind
from seeingmon.profile import Profile

EMPTY_FIELD = StarField.from_arrays([], [], [])
NO_SKY_MAG = 60.0  # far too faint to give a single electron


def covered_options(
    *,
    ambient_c: float = 15.0,
    sensor_rise_c: float = 4.0,
    hot_pixels_per_mpix: float = 0.0,
    hot_median_e_per_s: float = 3.0,
    seed: int = 1,
) -> SimOptions:
    """Options for a camera with a lens cap on: no stars, no sky, only the sensor."""
    return SimOptions(
        seed=seed,
        stars=EMPTY_FIELD,
        sky_mag_arcsec2=NO_SKY_MAG,
        twilight=False,
        ambient_c=ambient_c,
        sensor_rise_c=sensor_rise_c,
        hot_pixels=HotPixelConfig(
            density_per_mpix=hot_pixels_per_mpix, median_rate_e_per_s=hot_median_e_per_s
        ),
    )


def make_driver(profile: Profile, options: SimOptions, clock: Clock) -> SimDriver:
    """A sim driver for the readout modes of a profile."""
    return create(profile=profile, clock=clock, options=options)


def snapshots(
    driver: SimDriver, mode: str, gain: int, exposure_s: float, count: int
) -> list[Frame]:
    """Take `count` snapshot frames. The driver must be open."""
    driver.configure(
        StreamConfig(mode, max(1, round(exposure_s * 1e6)), gain, kind=StreamKind.SNAPSHOT)
    )
    frames = []
    for _ in range(count):
        driver.start()
        frames.append(driver.read_frame(timeout_s=exposure_s + 30.0))
        driver.stop()
    return frames


def hot_map_truth(driver: SimDriver, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The hot pixels that the driver rendered: columns, rows, and rates in e-/s at 20 C."""
    from seeingmon.drivers.sim.detector import HotPixelMap

    options = driver.options
    params = driver.modes[mode]
    hot = HotPixelMap(params, options.hot_pixels, options.seed)
    return np.asarray(hot.x), np.asarray(hot.y), np.asarray(hot.rate_e_per_s)


class CoverableDriver:
    """A camera with a cover: it shows an uncovered sky until a number of frames have been read.

    The wrapper holds two sim drivers, one with stars and a sky and one covered, and reads from
    the uncovered one while `reads < cover_after_reads`. It lets a test of `seeingmon dark` put
    the cover on in the middle of the wait.
    """

    def __init__(self, uncovered: SimDriver, covered: SimDriver, *, cover_after_reads: int) -> None:
        self._uncovered = uncovered
        self._covered = covered
        self._cover_after = cover_after_reads
        self._current = uncovered
        self.reads = 0

    @property
    def name(self) -> str:
        return "sim"

    def open(self) -> CameraInfo:
        self._covered.open()
        return self._uncovered.open()

    def close(self) -> None:
        self._covered.close()
        self._uncovered.close()

    def capabilities(self) -> CameraCaps:
        return self._uncovered.capabilities()

    def configure(self, config: StreamConfig) -> ActiveStream:
        self._covered.configure(config)
        return self._uncovered.configure(config)

    def start(self) -> None:
        self._current = self._covered if self.reads >= self._cover_after else self._uncovered
        self._current.start()

    def read_frame(self, timeout_s: float) -> Frame:
        frame = self._current.read_frame(timeout_s)
        self.reads += 1
        return frame

    def stop(self) -> None:
        self._current.stop()

    def move_roi(self, x: int, y: int) -> Roi:
        return self._current.move_roi(x, y)

    def read_temperature_c(self) -> float | None:
        return self._current.read_temperature_c()

    def dropped_frames(self) -> int:
        return 0

    def recover(self, level: RecoveryLevel) -> None:
        self._current.recover(level)
