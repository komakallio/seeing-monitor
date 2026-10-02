"""Measure the frame rates of the camera: the table behind `seeingmon camera rates`.

The rate of a fast stream depends on more than the exposure: on the ROI size, the pixel format, the
USB bandwidth control, and the high-speed mode of the camera. The profile's timing model (a frame
overhead plus a row time) comes from published numbers, so the table measures the real camera and
sets each measurement beside the model. The performance gate and the soak test on a Raspberry Pi
need the table of that Pi.

**The rows.** The table changes one factor at a time around the baseline, which is the fast stream
of the profile (its readout mode, 128 x 128 pixels, 2 ms, RAW16, bandwidth 100, normal speed):

- `exposure`: 0.1 ms to 20 ms. A rate that stays the same shows a readout limit, and a rate that
  follows the exposure shows an exposure limit.
- `roi`: ROI sizes from 32 to 512 pixels.
- `format`: RAW8.
- `bandwidth`: the USB bandwidth control from 40 to 90 percent (100 is the baseline).
- `speed`: the high-speed mode of the camera, at the same ROI sizes.
- `bin2`: the second readout mode, at 0.5 ms so that the readout sets the period (ROI sizes, the
  high-speed mode, and one ROI at 2 ms), and the 320 x 240, 10 ms, RAW8 stream that a recording
  uses.

**The columns.** Each row gives the measured rate, the rate of the model, the median, the standard
deviation (jitter), and the maximum of the frame periods, the frames that the camera dropped, and
the ADC depth of the readout mode (from the profile, because the SDK does not report it).

**The timing fit.** The rows at bandwidth 100 whose exposure is shorter than the frame period give a
straight line: the frame period against the ROI height. Its intercept is `frame_overhead_ms` and
its slope is `row_time_us` of the profile, for each readout mode with and without the high-speed
mode. `fit_timing` computes the lines, and the report prints them next to the values of the
profile.

**The camera stays as it was.** Another program, such as SharpCap, shares the camera and expects
its settings. `run_table` saves every writable control and the geometry before the first row, it
puts back what the rows changed even when a row fails or the run is interrupted, and it closes the
camera. A row that fails does not stop the table.
"""

from __future__ import annotations

import contextlib
import json
import logging
import platform
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seeingmon.clock import Clock, SystemClock, utc_ns_to_iso
from seeingmon.drivers.asi import AsiDriver
from seeingmon.drivers.base import CameraError
from seeingmon.frames import PixelFormat, Roi, StreamConfig
from seeingmon.profile import Profile
from seeingmon.profile.models import ReadoutMode

if TYPE_CHECKING:
    from seeingmon.config import Config

_log = logging.getLogger(__name__)

GROUPS = ("exposure", "roi", "format", "bandwidth", "speed", "bin2")
BASELINE_SIZE_PX = 128
BASELINE_EXPOSURE_US = 2000
BASELINE_BANDWIDTH_PCT = 100
EXPOSURES_US = (100, 500, 1000, 2000, 5000, 10_000, 20_000)
ROI_SIZES_PX = (32, 64, 128, 256, 512)
BANDWIDTHS_PCT = (40, 50, 60, 70, 80, 90, 100)
SPEED_SIZES_PX = ROI_SIZES_PX
BIN2_EXPOSURE_US = 500
BIN2_SIZES_PX = (64, 128, 256, 512)
BIN2_SPEED_SIZES_PX = (64, 128, 256)
BIN2_LONG_EXPOSURE_SIZE_PX = 64  # the ROI that also runs at the baseline exposure
BIN2_RECORDING = (320, 240, 10_000)  # width, height, and exposure of the stream of a recording
MIN_FRAMES = 3  # the shortest measurement: two frame periods and one more
READ_TIMEOUT_S = 2.0  # the longest wait for one frame of a stream in the table


@dataclass(frozen=True, slots=True)
class RowSpec:
    """A row to measure: what differs from the baseline, and the stream to run."""

    group: str
    label: str
    config: StreamConfig


@dataclass(frozen=True, slots=True)
class RateRow:
    """One measured row. The settings are what the camera applied. A row that failed has an
    `error` and no results."""

    group: str
    label: str
    mode: str
    roi_width: int
    roi_height: int
    pixel_format: str
    exposure_us: int
    bandwidth_pct: int | None
    high_speed: bool
    frames: int
    fps: float | None = None
    model_fps: float | None = None
    median_ms: float | None = None
    jitter_ms: float | None = None
    max_ms: float | None = None
    dropped: int | None = None
    adc_bits: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TimingFit:
    """The line through the frame periods of one readout mode, and the profile's values."""

    mode: str
    high_speed: bool
    points: int
    frame_overhead_ms: float
    row_time_us: float
    max_error_pct: float
    profile_frame_overhead_ms: float
    profile_row_time_us: float


@dataclass(frozen=True, slots=True)
class RatesReport:
    """The rows, the timing fits, the conditions of the run, and what could not be restored."""

    rows: list[RateRow]
    fits: list[TimingFit]
    conditions: dict[str, Any]
    restore_problems: list[str]

    @property
    def failed(self) -> bool:
        return bool(self.restore_problems) or any(row.error for row in self.rows)


# --- The plan -------------------------------------------------------------------------------


def _centered(width_px: int, height_px: int, width: int, height: int) -> Roi:
    return Roi((width_px - width) // 2, (height_px - height) // 2, width, height)


def baseline_config(profile: Profile, *, gain: int = 120) -> StreamConfig:
    """The fast stream of the profile: its readout mode, the baseline ROI, and the baseline
    exposure and bandwidth."""
    mode = profile.fast_readout
    return StreamConfig(
        mode=mode.name,
        exposure_us=BASELINE_EXPOSURE_US,
        gain=gain,
        pixel_format=profile.fast_mode.pixel_format or PixelFormat.RAW16,
        roi=_centered(mode.width_px, mode.height_px, BASELINE_SIZE_PX, BASELINE_SIZE_PX),
        bandwidth_pct=BASELINE_BANDWIDTH_PCT,
    )


def plan_rows(
    profile: Profile, *, gain: int = 120, groups: Sequence[str] = GROUPS
) -> list[RowSpec]:
    """The rows of the table, in order. The baseline comes first, and `groups` selects the rest.

    A row that repeats the baseline appears once. A size that exceeds the frame is left out.
    """
    unknown = [name for name in groups if name not in GROUPS]
    if unknown:
        raise ValueError(f"unknown row group {unknown[0]!r}; the groups are {', '.join(GROUPS)}")
    base = baseline_config(profile, gain=gain)
    fast = profile.fast_readout
    rows: list[RowSpec] = [RowSpec("baseline", "baseline", base)]

    def add(group: str, label: str, config: StreamConfig) -> None:
        if group in groups and all(row.config != config for row in rows):
            rows.append(RowSpec(group, label, config))

    def sized(config: StreamConfig, size: int) -> StreamConfig:
        mode = profile.mode(config.mode)
        return replace(config, roi=_centered(mode.width_px, mode.height_px, size, size))

    def inside(size: int, mode: ReadoutMode) -> bool:
        return size <= min(mode.width_px, mode.height_px)

    for exposure_us in EXPOSURES_US:
        add("exposure", f"exposure {exposure_us} us", replace(base, exposure_us=exposure_us))
    for size in ROI_SIZES_PX:
        if inside(size, fast):
            add("roi", f"roi {size}x{size}", sized(base, size))
    add("format", "RAW8", replace(base, pixel_format=PixelFormat.RAW8))
    for bandwidth in BANDWIDTHS_PCT:
        add("bandwidth", f"bandwidth {bandwidth}", replace(base, bandwidth_pct=bandwidth))
    if fast.has_high_speed:
        for size in SPEED_SIZES_PX:
            if inside(size, fast):
                add(
                    "speed",
                    f"high-speed, roi {size}x{size}",
                    replace(sized(base, size), high_speed=True),
                )
    second = next((m for m in profile.readout_modes if m.sdk_bin == 2), None)
    if second is not None and second.name != fast.name:
        # A short exposure, so that the readout and not the exposure sets the frame period.
        bin2 = replace(base, mode=second.name, exposure_us=BIN2_EXPOSURE_US)
        for size in BIN2_SIZES_PX:
            if inside(size, second):
                add("bin2", f"bin2 {size}x{size}, 0.5 ms", sized(bin2, size))
        if inside(BIN2_LONG_EXPOSURE_SIZE_PX, second):
            add(
                "bin2",
                f"bin2 {BIN2_LONG_EXPOSURE_SIZE_PX}x{BIN2_LONG_EXPOSURE_SIZE_PX}, 2 ms",
                sized(replace(bin2, exposure_us=BASELINE_EXPOSURE_US), BIN2_LONG_EXPOSURE_SIZE_PX),
            )
        if second.has_high_speed:
            for size in BIN2_SPEED_SIZES_PX:
                if inside(size, second):
                    add(
                        "bin2",
                        f"bin2 high-speed {size}x{size}, 0.5 ms",
                        replace(sized(bin2, size), high_speed=True),
                    )
        width, height, exposure_us = BIN2_RECORDING
        if inside(max(width, height), second):
            add(
                "bin2",
                f"bin2 {width}x{height}, {exposure_us // 1000} ms, RAW8",
                replace(
                    bin2,
                    exposure_us=exposure_us,
                    pixel_format=PixelFormat.RAW8,
                    roi=_centered(second.width_px, second.height_px, width, height),
                ),
            )
    return rows


# --- One row --------------------------------------------------------------------------------


def _failed(spec: RowSpec, frames: int, error: str) -> RateRow:
    config = spec.config
    roi = config.roi
    return RateRow(
        group=spec.group,
        label=spec.label,
        mode=config.mode,
        roi_width=0 if roi is None else roi.width,
        roi_height=0 if roi is None else roi.height,
        pixel_format=config.pixel_format.name,
        exposure_us=config.exposure_us,
        bandwidth_pct=config.bandwidth_pct,
        high_speed=config.high_speed,
        frames=frames,
        error=error,
    )


def measure_row(driver: AsiDriver, spec: RowSpec, *, frames: int, settle: int) -> RateRow:
    """Run one stream, read `settle` frames and then `frames` frames, and measure the last ones.

    The rate is the reciprocal of the mean frame period, taken from the arrival times of the
    measured frames. A failure of the camera gives a row with an `error` and no numbers.
    """
    if frames < MIN_FRAMES:
        raise ValueError(f"measure at least {MIN_FRAMES} frames")
    try:
        active = driver.configure(spec.config)
        driver.start()
        try:
            received = [driver.read_frame(READ_TIMEOUT_S) for _ in range(settle + frames)]
        finally:
            with contextlib.suppress(CameraError):
                driver.stop()
    except CameraError as error:
        with contextlib.suppress(CameraError):
            driver.stop()
        return _failed(spec, frames, f"{type(error).__name__}: {error}")
    measured = received[settle:]
    periods_s = [(b.t_arrival_ns - a.t_arrival_ns) / 1e9 for a, b in pairwise(measured)]
    config = active.config
    roi = config.roi
    assert roi is not None  # the driver fills the ROI in
    model_period_s = active.frame_period_s
    return RateRow(
        group=spec.group,
        label=spec.label,
        mode=config.mode,
        roi_width=roi.width,
        roi_height=roi.height,
        pixel_format=config.pixel_format.name,
        exposure_us=config.exposure_us,
        bandwidth_pct=config.bandwidth_pct,
        high_speed=config.high_speed,
        frames=frames,
        fps=1.0 / statistics.fmean(periods_s),
        model_fps=None if not model_period_s else 1.0 / model_period_s,
        median_ms=statistics.median(periods_s) * 1e3,
        jitter_ms=statistics.pstdev(periods_s) * 1e3,
        max_ms=max(periods_s) * 1e3,
        dropped=sum(frame.dropped_before for frame in measured),
        adc_bits=active.adc_bits,
    )


# --- The timing fit -------------------------------------------------------------------------


def fit_timing(rows: Sequence[RateRow], profile: Profile) -> list[TimingFit]:
    """Fit the frame period against the ROI height for each readout mode and speed.

    The fit takes the rows at the baseline bandwidth and the profile's pixel format whose exposure
    is well below the frame period, so that the readout sets the period. A mode needs three
    heights.
    """
    wanted_format = (profile.fast_mode.pixel_format or PixelFormat.RAW16).name
    fits: list[TimingFit] = []
    seen: set[tuple[str, bool]] = set()
    for row in rows:
        key = (row.mode, row.high_speed)
        if key in seen:
            continue
        seen.add(key)
        points = {
            r.roi_height: 1.0 / r.fps
            for r in rows
            if (r.mode, r.high_speed) == key
            and r.error is None
            and r.fps
            and r.bandwidth_pct == BASELINE_BANDWIDTH_PCT
            and r.pixel_format == wanted_format
            and r.exposure_us * 1e-6 < 0.8 / r.fps
        }
        if len(points) < 3:
            continue
        heights = sorted(points)
        periods = [points[height] for height in heights]
        slope, intercept = statistics.linear_regression(heights, periods)
        worst = max(
            abs(intercept + slope * height - period) / period
            for height, period in zip(heights, periods, strict=True)
        )
        mode = profile.mode(row.mode)
        current = mode.high_speed_variant() if row.high_speed else mode
        fits.append(
            TimingFit(
                mode=row.mode,
                high_speed=row.high_speed,
                points=len(points),
                frame_overhead_ms=intercept * 1e3,
                row_time_us=slope * 1e6,
                max_error_pct=worst * 100,
                profile_frame_overhead_ms=current.frame_overhead_ms,
                profile_row_time_us=current.row_time_us,
            )
        )
    return fits


# --- The run --------------------------------------------------------------------------------


def run_table(
    driver: AsiDriver,
    profile: Profile,
    *,
    frames: int = 150,
    settle: int = 10,
    gain: int = 120,
    groups: Sequence[str] = GROUPS,
    on_start: Callable[[dict[str, Any]], None] | None = None,
    on_row: Callable[[RateRow], None] | None = None,
    clock: Clock | None = None,
) -> RatesReport:
    """Open the camera, measure every row, put the camera back, and close it.

    `on_start` receives the conditions of the run once the camera is open, and `on_row` receives
    each row as soon as it is measured, so a command can print the table while it grows. The camera
    is restored and closed on every path out: a failing row, an exception from a callback, and an
    interrupt. `clock` gives the time of the report (the system clock by default).
    """
    plan = plan_rows(profile, gain=gain, groups=groups)
    info = driver.open()
    problems: list[str] = []
    rows: list[RateRow] = []
    temperature_c: float | None = None
    conditions: dict[str, Any] = {
        "camera_model": info.model,
        "sdk_version": info.sdk_version,
        "profile": profile.id,
        "frames": frames,
        "settle": settle,
        "gain": gain,
        "utc": utc_ns_to_iso((clock or SystemClock()).utc_ns(), digits=0),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
    }
    try:
        if on_start is not None:
            on_start(conditions)
        settings = driver.save_settings()
        try:
            for spec in plan:
                row = measure_row(driver, spec, frames=frames, settle=settle)
                rows.append(row)
                if on_row is not None:
                    on_row(row)
            temperature_c = driver.read_temperature_c()
        finally:
            problems = driver.restore_settings(settings)
            if problems:
                _log.warning("the camera could not be restored: %s", ", ".join(problems))
    finally:
        driver.close()
    conditions["temperature_c"] = temperature_c
    return RatesReport(rows, fit_timing(rows, profile), conditions, problems)


# --- Output ---------------------------------------------------------------------------------

COLUMNS: tuple[tuple[str, int], ...] = (
    ("setting", 32),
    ("mode", 5),
    ("roi", 9),
    ("exp us", 7),
    ("fmt", 6),
    ("bw", 4),
    ("hs", 3),
    ("fps", 7),
    ("model", 7),
    ("med ms", 7),
    ("jit ms", 7),
    ("max ms", 7),
    ("drops", 5),
    ("bits", 4),
)


def _line(cells: Sequence[str]) -> str:
    parts = []
    for index, ((_, width), cell) in enumerate(zip(COLUMNS, cells, strict=True)):
        parts.append(cell.ljust(width) if index in (0, 1, 4) else cell.rjust(width))
    return "  ".join(parts).rstrip()


def format_header() -> str:
    return _line([title for title, _ in COLUMNS])


def _number(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def format_row(row: RateRow) -> str:
    """One line of the table. A failed row shows its error after the settings."""
    cells = [
        row.label,
        row.mode,
        f"{row.roi_width}x{row.roi_height}",
        str(row.exposure_us),
        row.pixel_format,
        "-" if row.bandwidth_pct is None else str(row.bandwidth_pct),
        "yes" if row.high_speed else "no",
        _number(row.fps, 1),
        _number(row.model_fps, 1),
        _number(row.median_ms, 2),
        _number(row.jitter_ms, 2),
        _number(row.max_ms, 2),
        "-" if row.dropped is None else str(row.dropped),
        "-" if row.adc_bits is None else str(row.adc_bits),
    ]
    line = _line(cells)
    return f"{line}  FAILED {row.error}" if row.error else line


def format_fits(fits: Sequence[TimingFit]) -> list[str]:
    """The fitted timing beside the profile's values, one line for each readout mode and speed.

    A line names the keys of the profile (`frame_overhead_ms_high_speed` for the high-speed mode),
    so that you can copy the values into the readout mode.
    """
    lines = []
    for fit in fits:
        speed = "high-speed" if fit.high_speed else "normal"
        suffix = "_high_speed" if fit.high_speed else ""
        lines.append(
            f"{fit.mode} {speed}: frame_overhead_ms{suffix} = {fit.frame_overhead_ms:.2f} "
            f"(profile {fit.profile_frame_overhead_ms:.2f}), row_time_us{suffix} = "
            f"{fit.row_time_us:.1f} (profile {fit.profile_row_time_us:.1f}), {fit.points} sizes, "
            f"largest error {fit.max_error_pct:.1f}%"
        )
    return lines


def format_title(conditions: Mapping[str, Any]) -> str:
    """The line above the table: the camera, the SDK, and how long each row ran."""
    return (
        f"Frame rates of {conditions['camera_model']} (SDK {conditions['sdk_version']}), "
        f"{conditions['frames']} frames after {conditions['settle']} settle frames, "
        f"gain {conditions['gain']}"
    )


def format_footer(report: RatesReport) -> str:
    """The lines below the table: the sensor temperature, the timing fits, and the restore."""
    lines: list[str] = []
    temperature = report.conditions.get("temperature_c")
    if temperature is not None:
        lines.append(f"Sensor temperature at the end: {temperature:.1f} C")
    if report.fits:
        lines.append("Timing fit at bandwidth 100 (the ROI height against the frame period):")
        lines += [f"  {line}" for line in format_fits(report.fits)]
    if report.restore_problems:
        lines.append(f"NOT restored: {', '.join(report.restore_problems)}")
    else:
        lines.append("The camera is back as it was: every control and the geometry.")
    return "\n".join(lines)


def format_report(report: RatesReport) -> str:
    """The whole report as text: the title, the table, and the footer."""
    return "\n".join(
        [
            format_title(report.conditions),
            "",
            format_header(),
            *(format_row(row) for row in report.rows),
            "",
            format_footer(report),
        ]
    )


def report_to_json(report: RatesReport) -> dict[str, Any]:
    """The report as plain JSON types. It carries no host name, serial number, or path."""
    return {
        "conditions": report.conditions,
        "rows": [asdict(row) for row in report.rows],
        "fits": [asdict(fit) for fit in report.fits],
        "restore_problems": report.restore_problems,
    }


def write_json(report: RatesReport, path: Path) -> None:
    """Write the report as JSON. The folder is created when it is missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report_to_json(report), indent=2) + "\n", encoding="utf-8")


# --- The camera of this machine ---------------------------------------------------------------


def driver_options(config: Config) -> dict[str, Any]:
    """The options of the camera driver, from `[services.acquire.driver_options]`."""
    values = config.effective(redact=False)
    options = values.get("services", {}).get("acquire", {}).get("driver_options", {})
    return dict(options) if isinstance(options, dict) else {}


def create_driver(
    profile: Profile, options: Mapping[str, object], clock: Clock | None = None
) -> AsiDriver:
    """The production driver for the connected camera: the vendor library through `ctypes`."""
    from seeingmon.drivers import asi

    return asi.create(profile=profile, clock=clock or SystemClock(), options=options)
