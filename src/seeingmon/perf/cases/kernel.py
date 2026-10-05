"""The `kernel` case: the time per frame of the fast-path kernel in the three modes that matter.

The kernel is `seeingmon.fastpath.kernel.measure_frame`. It subtracts a local background and
computes the intensity-weighted centroid inside a circular aperture, recentered twice, and it
follows the star from frame to frame. The case times single calls on a pool of 64 rendered frames,
so the figure is the time of one frame and its spread. `python -m seeingmon.fastpath.benchmark`
times the same kernel as the mean of batches.

This is the per-frame number that decides whether Rust replaces the metrics: it is the figure to
read first. The architecture estimates 0.2 to 0.4 ms for the first mode on a Pi 4, and the
report compares that estimate with the measurement (`seeingmon perf report --budgets`).

The case also times `seeingmon.fastpath.kernel.search_frame`, the matched filter of one frame of
a search burst, within 20 px of the center of the ROI (the default `[scheduler.search]
radius_px`), as the figure `<mode>.search`. The bursts run while the search looks for Polaris.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from seeingmon.perf.cases.fastmodes import FAST_MODES, REFERENCE_PROFILE, FastMode, pool_frames
from seeingmon.perf.registry import REGISTRY, CaseContext
from seeingmon.perf.report import Measurement

if TYPE_CHECKING:
    import numpy.typing as npt

    from seeingmon.fastpath.kernel import FrameCalibration, KernelParams
    from seeingmon.profile import Profile

_US = 1e6
_ROI_X, _ROI_Y = 100, 200
_SEARCH_RADIUS_PX = 20.0  # the default `[scheduler.search] radius_px`


class Follower:
    """Calls the kernel on the next frame of a pool, starting at the star of the last frame."""

    def __init__(
        self,
        pool: list[npt.NDArray[Any]],
        params: KernelParams,
        calibration: FrameCalibration,
    ) -> None:
        self._pool = pool
        self._params = params
        self._calibration = calibration
        self._next = 0
        self._guess: tuple[float, float] | None = None

    def __call__(self) -> None:
        from seeingmon.fastpath.kernel import measure_frame

        data = self._pool[self._next % len(self._pool)]
        self._next += 1
        result = measure_frame(data, _ROI_X, _ROI_Y, self._params, self._calibration, self._guess)
        self._guess = (result.x, result.y) if result.found else None


class Searcher:
    """Calls the search of a burst frame on the next frame of a pool, around the ROI center."""

    def __init__(
        self,
        pool: list[npt.NDArray[Any]],
        params: KernelParams,
        calibration: FrameCalibration,
    ) -> None:
        self._pool = pool
        self._params = params
        self._calibration = calibration
        self._next = 0
        height, width = pool[0].shape
        self._center = (_ROI_X + 0.5 * (width - 1), _ROI_Y + 0.5 * (height - 1))

    def __call__(self) -> None:
        from seeingmon.fastpath.kernel import search_frame

        data = self._pool[self._next % len(self._pool)]
        self._next += 1
        search_frame(
            data,
            _ROI_X,
            _ROI_Y,
            self._params,
            self._calibration,
            self._center,
            _SEARCH_RADIUS_PX,
        )


def time_kernel(ctx: CaseContext, profile: Profile, mode: FastMode) -> list[Measurement]:
    """Time single calls of the kernel and of the search on the frames of one mode."""
    from seeingmon.fastpath.analyzer import FastPathAnalyzer
    from seeingmon.fastpath.config import FastPathConfig

    params, calibration = FastPathAnalyzer(profile, FastPathConfig()).kernel_setup(
        mode.mode, mode.gain, mode.exposure_us, mode.adc_bits(profile), mode.container_bits
    )
    pool = pool_frames(profile, mode, ctx.smoke)
    height, width = mode.shape_for(ctx.smoke)
    detail: dict[str, Any] = {
        "mode": mode.summary,
        "height_px": height,
        "width_px": width,
        "container_bits": mode.container_bits,
    }
    figures = []
    for name, call in (
        ("kernel", Follower(pool, params, calibration)),
        ("search", Searcher(pool, params, calibration)),
    ):
        stats = ctx.timer(repeats=1500, warmup=200).measure(call).scaled(_US)
        figures.append(
            Measurement(
                f"{mode.key}.{name}",
                "us/frame",
                stats.median,
                stats=stats,
                scale="numpy",
                detail={**detail, "calls": stats.samples},
            )
        )
    return figures


@REGISTRY.case(
    "kernel", summary="Time per frame of the fast-path kernel and the search in three modes"
)
def kernel(ctx: CaseContext) -> list[Measurement]:
    from seeingmon.profile import load_profile

    profile = load_profile(REFERENCE_PROFILE)
    ctx.mark_baseline()
    measurements = [item for mode in FAST_MODES for item in time_kernel(ctx, profile, mode)]
    ctx.note(
        "Each sample is one call of the kernel on one frame, in microseconds. The kernel follows "
        "the star from frame to frame, as the analyzer does. The search figure is one frame of a "
        "search burst, the matched filter within 20 px of the ROI center. The frames come from "
        "seeingmon.fastpath.benchmark.star_frames."
    )
    return measurements
