"""A scripted camera that looks at a light panel through a vignetted lens.

The camera follows the protocol of the flat session (`configure` and `take`) and the one of a
`CommissionContext` (`configure`, `start`, `read_frame`, `stop`), so the tests of the session and
the tests of the handler share it. A frame holds the bias, the light of the panel through the lens
(`flatfx.panel_frame`), photon noise, and read noise, in the 16-bit container that the SDK
delivers (14-bit counts in the high bits). The level in the middle is `rate * exposure` counts
above the bias, unless a `light` function says otherwise, so a test sets the brightness of the
panel and knows what the search must find. The camera takes the exposure from the virtual clock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.drivers.base import CameraTimeoutError
from seeingmon.frames import ActiveStream, Frame, StreamConfig
from tests.scheduler.helpers import make_frame
from tests.survey import flatfx as fx

CONTAINER_SHIFT = 2  # 14-bit counts in the high bits of 16


class PanelCamera:
    """A camera in front of a panel. `rate_dn_per_s` is the level above the bias in the middle.

    `light(index, exposure_s)` replaces the rate: it gives the level in counts above the bias for
    the `index`-th frame that the camera takes (counted from 0 over the whole life of the camera).
    `fail_at` makes that frame raise `CameraTimeoutError`. `after_take` runs after each frame, with
    the number of frames taken so far. `read_s` is the time that a frame takes beyond its exposure
    (the read and the transfer), which the virtual clock advances by.
    """

    def __init__(
        self,
        clock: VirtualClock,
        truth: fx.FloatImage,
        *,
        rate_dn_per_s: float = 204_800.0,
        light: Callable[[int, float], float] | None = None,
        bias_level: float = 131.0,
        temperature_c: float | None = 25.0,
        gradient: tuple[float, float] = (0.0, 0.0),
        seed: int = 1,
        mode: str = "bin2",
        adc_bits: int = 14,
        read_s: float = 0.0,
    ) -> None:
        self.clock = clock
        self.read_s = read_s
        self.truth = truth
        self.rate = rate_dn_per_s
        self.light = light
        self.temperature_c = temperature_c
        self.pattern = fx.source_pattern(truth.shape, gradient)
        self.bias = fx.bias_pattern(truth.shape, level=bias_level)
        self.seed = seed
        self.mode = mode
        self.adc_bits = adc_bits
        self.config: StreamConfig | None = None
        self.configures: list[StreamConfig] = []
        self.exposures_s: list[float] = []
        self.fail_at: int | None = None
        self.after_take: Callable[[int], None] | None = None
        self.shape_override: tuple[int, int] | None = None
        self.starts = self.stops = 0

    # --- the protocol of the session ---------------------------------------------------------

    def configure(self, config: StreamConfig) -> None:
        self.config = config
        self.configures.append(config)

    def take(self, exposure_s: float) -> Frame:
        index = len(self.exposures_s)
        if self.fail_at is not None and index == self.fail_at:
            raise CameraTimeoutError("no frame came")
        self.exposures_s.append(exposure_s)
        self.clock.sleep(exposure_s + self.read_s)
        level = self.light(index, exposure_s) if self.light is not None else self.rate * exposure_s
        rng = np.random.default_rng([self.seed, index])
        pixels = fx.panel_frame(
            self.truth,
            self.pattern,
            level=max(level, 0.0),
            bias=self.bias,
            rng=rng,
            shift=CONTAINER_SHIFT,
        )
        if self.shape_override is not None:
            pixels = np.ascontiguousarray(
                pixels[: self.shape_override[0], : self.shape_override[1]]
            )
        config = self.config
        frame = make_frame(
            pixels,
            mode=self.mode,
            gain=config.gain if config is not None else 120,
            exposure_us=round(exposure_s * 1e6),
            adc_bits=self.adc_bits,
            t_utc_ns=self.clock.utc_ns(),
            seq=index,
        )
        frame = replace(frame, temperature_c=self.temperature_c)
        if self.after_take is not None:
            self.after_take(index + 1)
        return frame

    # --- the protocol of a commission context ------------------------------------------------

    def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1

    def read_frame(self, timeout_s: float | None = None) -> Frame:
        assert self.config is not None, "read_frame before configure"
        return self.take(self.config.exposure_us / 1e6)

    def active_stream(self, config: StreamConfig, shape: tuple[int, int]) -> ActiveStream:
        return ActiveStream(1, config, shape, self.adc_bits)


def seconds_of(clock: VirtualClock) -> float:
    return clock.monotonic_ns() / NS_PER_S
