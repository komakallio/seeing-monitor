"""The frame-rate table: the plan, the measurement, the restore, the timing fit, and the command.

Every test runs on `FakeAsiSdk` and a `VirtualClock`, so a table of 30 rows takes a moment.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import replace
from pathlib import Path

import pytest

from seeingmon.cli import build_parser, main
from seeingmon.clock import utc_ns_to_iso
from seeingmon.drivers.base import CameraDisconnectedError
from seeingmon.frames import PixelFormat, Roi, StreamConfig, StreamKind
from seeingmon.hardware import rates
from seeingmon.hardware.asi.api import AsiControl, AsiErrorCode, AsiLibraryError
from seeingmon.hardware.asi.fake import DEFAULT_TIMING, FakeCameraState, FakeTiming
from seeingmon.hardware.rates import (
    BANDWIDTHS_PCT,
    GROUPS,
    SNAPSHOT_MAX_FRAMES,
    SNAPSHOT_MAX_SETTLE,
    RateRow,
    RowSpec,
    SnapshotFit,
    TimingFit,
    baseline_config,
    fit_snapshots,
    fit_timing,
    format_footer,
    format_header,
    format_report,
    format_row,
    format_snapshot_fits,
    measure_row,
    measure_snapshot_row,
    plan_rows,
    report_to_json,
    run_table,
    write_json,
)
from seeingmon.profile import parse_profile
from tests.hardware.asi_support import Rig, make_rig, reference_profile
from tests.profile.builders import synthetic_data

PROFILE = reference_profile()


def period_s(mode: int, high_speed: bool, rows: int) -> float:
    """The frame period of the fake camera for a ROI of `rows` rows, from its timing table."""
    timing = DEFAULT_TIMING[(mode, high_speed)]
    return timing.overhead_s + rows * timing.row_time_s


BIN1_PERIOD_128_S = period_s(1, False, 128)


@pytest.fixture(autouse=True)
def _own_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read no setting from the environment of the person who runs the tests."""
    for name in [name for name in os.environ if name.upper().startswith("SEEINGMON_")]:
        monkeypatch.delenv(name)


def baseline_spec() -> RowSpec:
    return plan_rows(PROFILE)[0]


def configs(groups: tuple[str, ...]) -> list[StreamConfig]:
    return [row.config for row in plan_rows(PROFILE, groups=groups)[1:]]


class TestPlan:
    def test_the_baseline_is_the_fast_stream_of_the_profile(self) -> None:
        spec = plan_rows(PROFILE)[0]
        assert (spec.group, spec.label) == ("baseline", "baseline")
        config = spec.config
        assert config == baseline_config(PROFILE)
        assert config.mode == PROFILE.fast_mode.mode == "bin1"
        assert config.pixel_format is PixelFormat.RAW16
        assert (config.exposure_us, config.gain, config.bandwidth_pct) == (2000, 120, 100)
        assert config.high_speed is False
        mode = PROFILE.fast_readout
        assert config.roi == Roi((mode.width_px - 128) // 2, (mode.height_px - 128) // 2, 128, 128)

    def test_each_row_of_a_factor_group_changes_one_factor_of_the_baseline(self) -> None:
        base = baseline_config(PROFILE)
        for group, field in (
            ("exposure", "exposure_us"),
            ("format", "pixel_format"),
            ("bandwidth", "bandwidth_pct"),
        ):
            rows = configs((group,))
            assert rows, group
            for config in rows:
                differing = [
                    name
                    for name in (
                        "exposure_us",
                        "pixel_format",
                        "bandwidth_pct",
                        "roi",
                        "high_speed",
                    )
                    if getattr(config, name) != getattr(base, name)
                ]
                assert differing == [field], (group, differing)
        for config in configs(("roi",)):
            assert config.roi is not None
            assert config.roi.width == config.roi.height
            assert replace(config, roi=base.roi) == base  # only the ROI size differs

    def test_the_speed_rows_turn_the_high_speed_mode_on_at_each_roi_size(self) -> None:
        rows = configs(("speed",))
        assert rows
        assert all(config.high_speed for config in rows)
        assert {config.roi.width for config in rows if config.roi} == {32, 64, 128, 256, 512}

    def test_the_bandwidth_rows_run_from_40_to_100(self) -> None:
        values = {c.bandwidth_pct for c in configs(("bandwidth",))} | {100}  # 100 is the baseline
        assert values == set(BANDWIDTHS_PCT) == {40, 50, 60, 70, 80, 90, 100}

    def test_the_bin2_rows_use_the_second_mode_and_include_the_recording_stream(self) -> None:
        rows = plan_rows(PROFILE, groups=("bin2",))[1:]
        assert {row.config.mode for row in rows} == {"bin2"}
        recording = [
            r
            for r in rows
            if r.config.roi and (r.config.roi.width, r.config.roi.height) == (320, 240)
        ]
        assert [r.config.exposure_us for r in recording] == [10_000]
        assert recording[0].config.pixel_format is PixelFormat.RAW8
        assert any(row.config.high_speed for row in rows)
        assert {
            r.config.exposure_us for r in rows if not r.config.roi or r.config.roi.width != 320
        } == {
            500,
            2000,
        }

    def test_the_snapshot_rows_take_single_exposures_of_the_survey_mode_at_full_width(self) -> None:
        rows = plan_rows(PROFILE, groups=("snapshot",))[1:]
        survey = PROFILE.survey_readout
        assert {row.group for row in rows} == {"snapshot"}
        configs_ = [row.config for row in rows]
        assert {config.kind for config in configs_} == {StreamKind.SNAPSHOT}
        assert {config.mode for config in configs_} == {survey.name} == {"bin2"}
        assert {config.exposure_us for config in configs_} == {1000}
        assert {config.bandwidth_pct for config in configs_} == {100}
        assert not any(config.high_speed for config in configs_)
        rois = [config.roi for config in configs_]
        assert all(roi is not None and roi.width == survey.width_px for roi in rois)
        heights = [roi.height for roi in rois if roi is not None]
        assert heights == [64, 256, 1024, survey.height_px]  # the last row is the full frame
        assert all(
            roi is not None and roi.y == (survey.height_px - roi.height) // 2 for roi in rois
        )
        assert [row.label for row in rows][-1] == "snapshot 4144x2822, 1 ms"

    def test_a_snapshot_height_beyond_the_frame_is_left_out(self) -> None:
        small = parse_profile(synthetic_data(), source="the synthetic profile")
        rows = plan_rows(small, groups=("snapshot",))[1:]
        heights = [row.config.roi.height for row in rows if row.config.roi]
        assert small.survey_readout.height_px == 750
        assert heights == [64, 256, 750]  # 1024 does not fit
        assert {row.config.pixel_format for row in rows} == {PixelFormat.RAW8}  # the survey format

    def test_no_row_repeats_another(self) -> None:
        all_rows = plan_rows(PROFILE)
        assert len({row.config for row in all_rows}) == len(all_rows)
        assert len({row.label for row in all_rows}) == len(all_rows)

    def test_the_groups_select_the_rows_and_the_baseline_stays(self) -> None:
        rows = plan_rows(PROFILE, groups=("bandwidth",))
        assert rows[0].label == "baseline"
        assert {row.group for row in rows[1:]} == {"bandwidth"}
        assert plan_rows(PROFILE, groups=())[0].label == "baseline"
        assert len(plan_rows(PROFILE, groups=())) == 1

    def test_an_unknown_group_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="unknown row group 'sleep'"):
            plan_rows(PROFILE, groups=("sleep",))

    def test_a_profile_without_high_speed_values_has_no_speed_rows(self) -> None:
        no_high_speed = {
            "adc_bits_high_speed": None,
            "row_time_us_high_speed": None,
            "frame_overhead_ms_high_speed": None,
        }
        profile = PROFILE.model_copy(
            update={
                "readout_modes": [m.model_copy(update=no_high_speed) for m in PROFILE.readout_modes]
            }
        )
        rows = plan_rows(profile)
        assert not any(row.config.high_speed for row in rows)

    def test_the_groups_are_the_ones_that_the_command_documents(self) -> None:
        assert GROUPS == ("exposure", "roi", "format", "bandwidth", "speed", "bin2", "snapshot")


def spec(**changes: object) -> RowSpec:
    return RowSpec("test", "test row", replace(baseline_config(PROFILE), **changes))  # type: ignore[arg-type]


class TestMeasure:
    def test_a_row_measures_what_the_fake_camera_does(self) -> None:
        rig = make_rig().opened()
        row = measure_row(rig.driver, baseline_spec(), frames=30, settle=5)
        assert row.error is None
        assert row.fps == pytest.approx(1 / BIN1_PERIOD_128_S, rel=1e-6)
        assert row.model_fps == pytest.approx(1 / BIN1_PERIOD_128_S, rel=1e-6)
        assert row.median_ms == pytest.approx(BIN1_PERIOD_128_S * 1e3, rel=1e-6)
        assert row.max_ms == pytest.approx(BIN1_PERIOD_128_S * 1e3, rel=1e-6)
        assert row.jitter_ms == pytest.approx(0.0, abs=1e-6)
        assert (row.dropped, row.adc_bits) == (0, 12)
        assert (row.mode, row.roi_width, row.roi_height) == ("bin1", 128, 128)
        assert (row.pixel_format, row.exposure_us, row.bandwidth_pct) == ("RAW16", 2000, 100)
        assert row.high_speed is False
        assert not rig.sdk.video_active  # the row stops its stream

    def test_a_long_exposure_sets_the_rate(self) -> None:
        row = measure_row(make_rig().opened().driver, spec(exposure_us=20_000), frames=20, settle=2)
        assert row.fps == pytest.approx(50.0, rel=1e-6)
        assert row.model_fps == pytest.approx(50.0, rel=1e-6)

    def test_high_speed_mode_shows_in_the_rate_and_the_bit_depth(self) -> None:
        row = measure_row(make_rig().opened().driver, spec(high_speed=True), frames=20, settle=2)
        assert row.adc_bits == 10
        assert row.fps == pytest.approx(1 / period_s(1, True, 128), rel=1e-6)
        assert row.high_speed is True

    def test_the_row_reports_what_the_camera_applied(self) -> None:
        rig = make_rig(bandwidth_pct=70).opened()
        row = measure_row(
            rig.driver,
            RowSpec("t", "t", replace(baseline_config(PROFILE), bandwidth_pct=None)),
            frames=5,
            settle=1,
        )
        assert row.bandwidth_pct == 70  # the option applied, because the stream asked for none

    def test_the_settle_frames_do_not_count(self) -> None:
        rig = make_rig().opened()
        measure_row(rig.driver, baseline_spec(), frames=10, settle=7)
        reads = [c for c in rig.sdk.calls if c[0] == "get_video_data"]
        # The driver drops one frame after the start, and the row reads 7 + 10 more.
        assert len(reads) == 1 + 7 + 10

    def test_a_row_that_the_camera_refuses_is_reported_and_stops_the_stream(self) -> None:
        rig = make_rig().opened()
        rig.sdk.fail_next("set_roi_format", AsiErrorCode.INVALID_SIZE)
        row = measure_row(rig.driver, baseline_spec(), frames=5, settle=1)
        assert row.error is not None
        assert row.error.startswith("AsiConfigError: set_roi_format failed")
        assert (row.fps, row.dropped) == (None, None)
        assert (row.mode, row.exposure_us, row.pixel_format) == ("bin1", 2000, "RAW16")
        assert not rig.sdk.video_active

    def test_too_few_frames_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            measure_row(make_rig().opened().driver, baseline_spec(), frames=2, settle=0)


SNAPSHOT_FULL_S = 0.001 + 0.27 + 2822 * 75e-6  # the model of the profile: a full frame at 1 ms
SNAPSHOT_POLL_MS = 2.0  # the driver polls the status of an exposure this often at 1 ms


def snapshot_spec(height: int = 2822) -> RowSpec:
    rows = plan_rows(PROFILE, groups=("snapshot",))[1:]
    return next(row for row in rows if row.config.roi and row.config.roi.height == height)


class TestMeasureSnapshots:
    def test_a_row_times_each_exposure_from_the_start_call_to_the_frame(self) -> None:
        rig = make_rig().opened()
        row = measure_snapshot_row(rig.driver, snapshot_spec(), frames=5, settle=1, clock=rig.clock)
        assert row.error is None
        assert row.kind == "snapshot"
        assert row.median_ms is not None
        assert row.max_ms is not None
        assert row.jitter_ms is not None
        # The driver polls the status every 2 ms, so a time can end up to 2 ms after the frame.
        assert row.median_ms == pytest.approx(SNAPSHOT_FULL_S * 1e3, abs=SNAPSHOT_POLL_MS)
        assert row.fps == pytest.approx(1e3 / row.median_ms)
        assert row.model_fps == pytest.approx(1 / SNAPSHOT_FULL_S, rel=1e-6)
        assert row.max_ms >= row.median_ms
        assert row.jitter_ms <= SNAPSHOT_POLL_MS
        assert (row.dropped, row.adc_bits) == (0, 14)
        assert (row.mode, row.roi_width, row.roi_height) == ("bin2", 4144, 2822)
        assert (row.exposure_us, row.bandwidth_pct, row.pixel_format) == (1000, 100, "RAW16")
        assert row.high_speed is False

    def test_each_exposure_is_a_new_start_and_the_settle_exposures_are_taken_too(self) -> None:
        rig = make_rig().opened()
        measure_snapshot_row(rig.driver, snapshot_spec(64), frames=5, settle=2, clock=rig.clock)
        assert len(rig.sdk.calls_named("start_exposure")) == 7
        assert len(rig.sdk.calls_named("get_data_after_exposure")) == 7
        assert not rig.sdk.video_active

    def test_the_row_shows_the_camera_and_not_the_model(self) -> None:
        """A camera that is slower than the model, but within the bound of the driver."""
        rig = make_rig(sdk={"snapshot_timing": {2: FakeTiming(150e-6, 0.5)}}).opened()
        row = measure_snapshot_row(rig.driver, snapshot_spec(), frames=3, settle=0, clock=rig.clock)
        assert row.median_ms == pytest.approx(
            (0.001 + 0.5 + 2822 * 150e-6) * 1e3, abs=SNAPSHOT_POLL_MS
        )
        assert row.model_fps == pytest.approx(1 / SNAPSHOT_FULL_S, rel=1e-6)  # the profile's model

    def test_a_camera_far_slower_than_the_model_gives_a_row_with_the_timeout(self) -> None:
        rig = make_rig(sdk={"snapshot_timing": {2: FakeTiming(75e-6, 5.0)}}).opened()
        row = measure_snapshot_row(rig.driver, snapshot_spec(), frames=3, settle=0, clock=rig.clock)
        assert row.error is not None
        assert row.error.startswith("CameraTimeoutError: the exposure did not finish within 1.5 s")
        assert (row.fps, row.median_ms, row.dropped) == (None, None, None)
        assert row.kind == "snapshot"

    def test_a_row_that_the_camera_refuses_is_reported(self) -> None:
        rig = make_rig().opened()
        rig.sdk.fail_next("set_roi_format", AsiErrorCode.INVALID_SIZE)
        row = measure_snapshot_row(rig.driver, snapshot_spec(), frames=3, settle=0, clock=rig.clock)
        assert row.error is not None
        assert row.error.startswith("AsiConfigError: set_roi_format failed")
        assert (row.fps, row.dropped, row.kind) == (None, None, "snapshot")
        assert (row.mode, row.exposure_us, row.pixel_format) == ("bin2", 1000, "RAW16")

    def test_a_failed_exposure_is_reported(self) -> None:
        rig = make_rig().opened()
        rig.sdk.fail_next_exposure()
        row = measure_snapshot_row(
            rig.driver, snapshot_spec(64), frames=3, settle=0, clock=rig.clock
        )
        assert row.error == "CameraError: the exposure failed"

    def test_too_few_frames_is_an_error(self) -> None:
        rig = make_rig().opened()
        with pytest.raises(ValueError, match="at least 3"):
            measure_snapshot_row(rig.driver, snapshot_spec(), frames=2, settle=0, clock=rig.clock)


def owner_state() -> FakeCameraState:
    """The settings that another program left in the camera."""
    state = FakeCameraState(
        {
            AsiControl.BANDWIDTH_OVERLOAD: 60,
            AsiControl.FLIP: 2,
            AsiControl.OFFSET: 33,
            AsiControl.GAIN: 77,
            AsiControl.EXPOSURE: 12_345,
            AsiControl.HIGH_SPEED_MODE: 1,
        }
    )
    state.automatic.add(AsiControl.EXPOSURE)
    return state


def table(rig: Rig, **options: object) -> rates.RatesReport:
    return run_table(rig.driver, PROFILE, frames=6, settle=1, **options)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def snapshot_report() -> rates.RatesReport:
    """One table of the snapshot group. A snapshot row polls the status of the exposure every 2 ms
    of virtual time for about 0.3 to 0.5 s, so the tests that read the table share one run."""
    rig = make_rig()
    return run_table(rig.driver, PROFILE, frames=6, settle=1, groups=("snapshot",), clock=rig.clock)


class TestRunTable:
    def test_the_camera_is_back_as_it_was_and_closed(self) -> None:
        state = owner_state()
        rig = make_rig(sdk={"state": state})
        before = (dict(state.controls), set(state.automatic))
        report = table(rig, groups=("bandwidth", "speed"))
        assert report.restore_problems == []
        assert not report.failed
        assert (dict(state.controls), set(state.automatic)) == before
        assert rig.sdk.roi == (0, 0, 8288, 5644)  # the geometry, too
        assert not rig.sdk.video_active
        assert [name for name, _ in rig.sdk.calls if name == "close_camera"] == ["close_camera"]
        assert rig.sdk.calls[-1][0] == "close_camera"

    def test_a_failing_row_does_not_stop_the_table_and_the_camera_still_comes_back(self) -> None:
        state = owner_state()
        rig = make_rig(sdk={"state": state})
        before = (dict(state.controls), set(state.automatic))
        rig.sdk.fail_next("set_roi_format", AsiErrorCode.INVALID_SIZE)  # the baseline
        report = table(rig, groups=("bandwidth",))
        errors = [row for row in report.rows if row.error]
        assert [row.label for row in errors] == ["baseline"]
        assert len([row for row in report.rows if not row.error]) == len(BANDWIDTHS_PCT) - 1
        assert report.failed  # a failed row makes the command fail
        assert (dict(state.controls), set(state.automatic)) == before

    def test_an_interrupt_still_restores_and_closes(self) -> None:
        state = owner_state()
        rig = make_rig(sdk={"state": state})
        before = (dict(state.controls), set(state.automatic))
        seen: list[str] = []

        def stop_after_two(row: RateRow) -> None:
            seen.append(row.label)
            if len(seen) == 2:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            table(rig, groups=("bandwidth",), on_row=stop_after_two)
        assert len(seen) == 2
        assert (dict(state.controls), set(state.automatic)) == before
        assert rig.sdk.calls[-1][0] == "close_camera"
        assert not rig.sdk.video_active

    def test_a_camera_that_cannot_be_restored_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rig = make_rig()
        monkeypatch.setattr(rig.driver, "restore_settings", lambda settings: ["gain", "roi"])
        report = table(rig, groups=())
        assert report.restore_problems == ["gain", "roi"]
        assert report.failed
        assert "NOT restored: gain, roi" in format_footer(report)
        assert rig.sdk.calls[-1][0] == "close_camera"  # it still closes the camera

    def test_the_callbacks_see_the_conditions_first_and_then_each_row(self) -> None:
        events: list[str] = []
        table(
            make_rig(),
            groups=("format",),
            on_start=lambda conditions: events.append(f"start {conditions['camera_model']}"),
            on_row=lambda row: events.append(row.label),
        )
        assert events == ["start ZWO ASI294MM (fake)", "baseline", "RAW8"]

    def test_the_report_takes_its_time_from_the_clock_that_it_is_given(self) -> None:
        rig = make_rig()
        started = utc_ns_to_iso(rig.clock.utc_ns(), digits=0)  # the table starts at this time
        report = run_table(rig.driver, PROFILE, frames=6, settle=1, groups=(), clock=rig.clock)
        assert report.conditions["utc"] == started

    def test_the_conditions_name_the_camera_and_the_run_and_nothing_private(self) -> None:
        report = table(make_rig(), groups=())
        conditions = report.conditions
        assert conditions["camera_model"] == "ZWO ASI294MM (fake)"
        assert conditions["sdk_version"] == "1, 41, 0, 0"
        assert (conditions["frames"], conditions["settle"], conditions["gain"]) == (6, 1, 120)
        assert conditions["profile"] == "asi294mm-gs250"
        assert conditions["temperature_c"] == pytest.approx(18.3)
        text = json.dumps(report_to_json(report))
        assert platform.node() not in text

    def test_the_snapshot_group_measures_single_exposures_and_fits_the_snapshot_model(
        self, snapshot_report: rates.RatesReport
    ) -> None:
        report = snapshot_report
        assert not report.failed
        snapshots = [row for row in report.rows if row.kind == "snapshot"]
        assert [row.roi_height for row in snapshots] == [64, 256, 1024, 2822]
        (fit,) = report.snapshot_fits
        assert isinstance(fit, SnapshotFit)
        assert (fit.mode, fit.points, fit.profile_has_model) == ("bin2", 4, True)
        # The status polls round each time up by at most 2 ms.
        assert fit.overhead_s == pytest.approx(0.27, abs=0.003)
        assert fit.row_time_us == pytest.approx(75.0, rel=0.03)
        assert fit.max_error_pct < 1.0
        assert (fit.profile_overhead_s, fit.profile_row_time_us) == (0.27, 75.0)
        assert report.fits == []  # the baseline is one video row, and that is no line

    def test_a_snapshot_row_takes_at_most_a_few_exposures(self) -> None:
        rows = 4  # the snapshot rows of the reference profile
        rig = make_rig()
        run_table(rig.driver, PROFILE, frames=150, settle=10, groups=("snapshot",), clock=rig.clock)
        taken = len(rig.sdk.calls_named("start_exposure"))
        assert taken == rows * (SNAPSHOT_MAX_FRAMES + SNAPSHOT_MAX_SETTLE)

    def test_the_snapshot_rows_do_not_enter_the_video_fit(self) -> None:
        rig = make_rig()
        report = run_table(
            rig.driver,
            PROFILE,
            frames=6,
            settle=1,
            groups=("roi", "speed", "bin2", "snapshot"),
            clock=rig.clock,
        )
        by_key = {(fit.mode, fit.high_speed): fit for fit in report.fits}
        assert set(by_key) == {("bin1", False), ("bin1", True), ("bin2", False), ("bin2", True)}
        for (mode, high_speed), fit in by_key.items():
            timing = DEFAULT_TIMING[(int(mode.removeprefix("bin")), high_speed)]
            assert fit.frame_overhead_ms == pytest.approx(timing.overhead_s * 1e3, rel=1e-3)
            assert fit.row_time_us == pytest.approx(timing.row_time_s * 1e6, rel=1e-3)
        assert [fit.mode for fit in report.snapshot_fits] == ["bin2"]

    def test_the_camera_is_back_as_it_was_after_the_snapshot_rows(self) -> None:
        state = owner_state()
        rig = make_rig(sdk={"state": state})
        before = (dict(state.controls), set(state.automatic))
        report = run_table(
            rig.driver, PROFILE, frames=3, settle=0, groups=("snapshot",), clock=rig.clock
        )
        assert report.restore_problems == []
        assert (dict(state.controls), set(state.automatic)) == before
        assert rig.sdk.roi == (0, 0, 8288, 5644)
        assert rig.sdk.calls[-1][0] == "close_camera"
        # A request for fewer frames than the cap wins: each of the 4 rows took 3 exposures.
        assert len(rig.sdk.calls_named("start_exposure")) == 4 * 3

    def test_nothing_runs_without_a_camera(self) -> None:
        rig = make_rig(sdk={"temperature_c": None})
        rig.sdk.disconnect()
        with pytest.raises(CameraDisconnectedError):
            table(rig, groups=())


class TestFit:
    def test_the_fit_recovers_the_timing_of_the_fake_camera(self) -> None:
        report = table(make_rig(), groups=("roi", "speed", "bin2"))
        by_key = {(fit.mode, fit.high_speed): fit for fit in report.fits}
        assert set(by_key) == {("bin1", False), ("bin1", True), ("bin2", False), ("bin2", True)}
        for (mode, high_speed), fit in by_key.items():
            timing = DEFAULT_TIMING[(int(mode.removeprefix("bin")), high_speed)]
            assert fit.frame_overhead_ms == pytest.approx(timing.overhead_s * 1e3, rel=1e-3)
            assert fit.row_time_us == pytest.approx(timing.row_time_s * 1e6, rel=1e-3)
            assert fit.max_error_pct < 0.1

    def test_the_fake_camera_runs_the_numbers_of_the_profile(self) -> None:
        """The fake and the profile state the same timing, so the fit of a fake run agrees with the
        profile. A refit of the profile from a real camera changes the table of the fake with it."""
        report = table(make_rig(), groups=("roi", "speed", "bin2"))
        for fit in report.fits:
            timing = DEFAULT_TIMING[(int(fit.mode.removeprefix("bin")), fit.high_speed)]
            assert fit.profile_frame_overhead_ms == pytest.approx(timing.overhead_s * 1e3)
            assert fit.profile_row_time_us == pytest.approx(timing.row_time_s * 1e6)

    def row(self, **changes: object) -> RateRow:
        values: dict[str, object] = {
            "group": "roi",
            "label": "x",
            "mode": "bin1",
            "roi_width": 64,
            "roi_height": 64,
            "pixel_format": "RAW16",
            "exposure_us": 2000,
            "bandwidth_pct": 100,
            "high_speed": False,
            "frames": 10,
            "fps": 100.0,
        }
        values.update(changes)
        return RateRow(**values)  # type: ignore[arg-type]

    def test_a_mode_needs_three_heights(self) -> None:
        rows = [self.row(roi_height=64, fps=100.0), self.row(roi_height=128, fps=80.0)]
        assert fit_timing(rows, PROFILE) == []

    def test_a_line_through_three_heights(self) -> None:
        # Periods of 8 ms, 12 ms, and 20 ms at 64, 128, and 256 rows: 4 ms plus 62.5 us a row.
        rows = [
            self.row(roi_height=64, fps=1 / 0.008),
            self.row(roi_height=128, fps=1 / 0.012),
            self.row(roi_height=256, fps=1 / 0.020),
        ]
        (fit,) = fit_timing(rows, PROFILE)
        assert isinstance(fit, TimingFit)
        assert fit.points == 3
        assert fit.frame_overhead_ms == pytest.approx(4.0)
        assert fit.row_time_us == pytest.approx(62.5)
        assert fit.max_error_pct == pytest.approx(0.0, abs=1e-9)

    def test_rows_that_the_exposure_or_the_bandwidth_limit_are_left_out(self) -> None:
        good = [self.row(roi_height=h, fps=1 / (0.004 + h * 62.5e-6)) for h in (64, 128, 256)]
        noise = [
            self.row(roi_height=512, bandwidth_pct=50, fps=5.0),  # not at bandwidth 100
            self.row(roi_height=48, pixel_format="RAW8", fps=5.0),  # not the profile's format
            self.row(roi_height=96, exposure_us=20_000, fps=1 / 0.0201),  # the exposure limits it
            self.row(roi_height=80, error="boom", fps=None),  # a failed row
        ]
        (fit,) = fit_timing([*good, *noise], PROFILE)
        assert fit.points == 3
        assert fit.frame_overhead_ms == pytest.approx(4.0)
        assert fit.row_time_us == pytest.approx(62.5)

    def snapshot_row(self, height: int, readout_s: float, **changes: object) -> RateRow:
        """A snapshot row whose time is the 1 ms exposure plus `readout_s`."""
        values: dict[str, object] = {
            "group": "snapshot",
            "kind": "snapshot",
            "mode": "bin2",
            "roi_width": 4144,
            "roi_height": height,
            "exposure_us": 1000,
            "median_ms": (0.001 + readout_s) * 1e3,
            "fps": 1 / (0.001 + readout_s),
        }
        values.update(changes)
        return self.row(**values)

    def test_a_line_through_the_times_of_single_exposures(self) -> None:
        rows = [self.snapshot_row(h, 0.4 + h * 100e-6) for h in (64, 256, 2822)]
        (fit,) = fit_snapshots(rows, PROFILE)
        assert isinstance(fit, SnapshotFit)
        assert (fit.mode, fit.points, fit.profile_has_model) == ("bin2", 3, True)
        assert fit.overhead_s == pytest.approx(0.4)
        assert fit.row_time_us == pytest.approx(100.0)
        assert fit.max_error_pct == pytest.approx(0.0, abs=1e-9)
        assert (fit.profile_overhead_s, fit.profile_row_time_us) == (0.27, 75.0)

    def test_a_mode_without_a_snapshot_model_shows_the_values_that_the_software_assumes(
        self,
    ) -> None:
        rows = [self.snapshot_row(h, 0.5 + h * 40e-6, mode="bin1") for h in (64, 256, 2048)]
        (fit,) = fit_snapshots(rows, PROFILE)
        assert fit.profile_has_model is False
        assert (fit.profile_overhead_s, fit.profile_row_time_us) == (0.3, 37.6)

    def test_the_snapshot_fit_of_each_mode_is_separate(self) -> None:
        rows = [
            *(self.snapshot_row(h, 0.27 + h * 75e-6) for h in (64, 256, 2822)),
            *(self.snapshot_row(h, 0.5 + h * 40e-6, mode="bin1") for h in (64, 256, 2048)),
        ]
        assert [fit.mode for fit in fit_snapshots(rows, PROFILE)] == ["bin2", "bin1"]

    def test_a_snapshot_mode_needs_three_heights(self) -> None:
        rows = [self.snapshot_row(64, 0.28), self.snapshot_row(256, 0.29)]
        assert fit_snapshots(rows, PROFILE) == []
        assert fit_snapshots([], PROFILE) == []

    def test_a_failed_row_and_a_time_below_the_exposure_are_left_out(self) -> None:
        good = [self.snapshot_row(h, 0.27 + h * 75e-6) for h in (64, 256, 1024)]
        noise = [
            self.snapshot_row(512, 0.3, error="boom", median_ms=None, fps=None),
            self.snapshot_row(128, -0.0005),  # the row took less than its exposure: a bad clock
        ]
        (fit,) = fit_snapshots([*good, *noise], PROFILE)
        assert fit.points == 3
        assert fit.overhead_s == pytest.approx(0.27)

    def test_the_video_fit_leaves_the_snapshot_rows_out(self) -> None:
        rows = [self.snapshot_row(h, 0.27 + h * 75e-6) for h in (64, 256, 1024)]
        assert fit_timing(rows, PROFILE) == []
        video = [
            self.row(roi_height=h, mode="bin2", exposure_us=500, fps=1 / (0.0012 + h * 18.5e-6))
            for h in (64, 128, 256)
        ]
        (fit,) = fit_timing([*rows, *video], PROFILE)  # the snapshot rows come first
        assert (fit.mode, fit.points) == ("bin2", 3)
        assert fit.frame_overhead_ms == pytest.approx(1.2)
        assert fit.row_time_us == pytest.approx(18.5)


class TestOutput:
    def test_a_row_is_one_line_with_the_columns_of_the_header(self) -> None:
        rig = make_rig().opened()
        row = measure_row(rig.driver, baseline_spec(), frames=10, settle=1)
        line = format_row(row)
        assert "\n" not in line
        header = format_header()
        for title in ("setting", "mode", "roi", "exp us", "fmt", "bw", "hs", "fps", "model"):
            assert title in header
        for title in ("med ms", "jit ms", "max ms", "drops", "bits"):
            assert title in header
        for text in ("128x128", "RAW16", "82.1"):
            assert text in line
        assert len(line) == len(header)  # the columns line up

    def test_a_failed_row_shows_its_error(self) -> None:
        rig = make_rig().opened()
        rig.sdk.fail_next("set_roi_format", AsiErrorCode.INVALID_SIZE)
        row = measure_row(rig.driver, baseline_spec(), frames=5, settle=1)
        line = format_row(row)
        assert "FAILED AsiConfigError" in line
        assert line.count("-") >= 5  # the missing numbers

    def test_the_report_has_the_title_the_table_the_fits_and_the_restore_line(self) -> None:
        text = format_report(table(make_rig(), groups=("roi",)))
        assert text.splitlines()[0].startswith("Frame rates of ZWO ASI294MM (fake)")
        assert "roi 64x64" in text
        assert "Timing fit at bandwidth 100" in text
        assert "frame_overhead_ms" in text
        assert text.rstrip().endswith("every control and the geometry.")

    def test_the_json_carries_the_conditions_the_rows_the_fits_and_the_restore(
        self, tmp_path: Path
    ) -> None:
        report = table(make_rig(), groups=("roi",))
        path = tmp_path / "local" / "rates.json"  # the folder does not exist yet
        write_json(report, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {"conditions", "rows", "fits", "snapshot_fits", "restore_problems"}
        assert data["restore_problems"] == []
        assert data["rows"][0]["label"] == "baseline"
        assert data["rows"][0]["kind"] == "video"
        assert data["rows"][0]["fps"] == pytest.approx(report.rows[0].fps)
        assert data["fits"][0]["mode"] == "bin1"
        assert data["snapshot_fits"] == []  # no snapshot rows ran
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_a_snapshot_row_is_one_line_with_the_columns_of_the_header(self) -> None:
        rig = make_rig().opened()
        row = measure_snapshot_row(rig.driver, snapshot_spec(), frames=4, settle=1, clock=rig.clock)
        line = format_row(row)
        assert "\n" not in line
        for text in ("snapshot 4144x2822, 1 ms", "bin2", "4144x2822", "RAW16", "1000", "2.1"):
            assert text in line
        assert len(line) == len(format_header())

    def test_the_snapshot_fit_names_the_keys_of_the_profile(self) -> None:
        fit = SnapshotFit("bin2", 4, 0.2684, 75.44, 0.8, 0.27, 75.0, True)
        (line,) = format_snapshot_fits([fit])
        assert line == (
            "bin2: snapshot_overhead_s = 0.268 (profile 0.270), snapshot_row_time_us = 75.4 "
            "(profile 75.0), 4 sizes, largest error 0.8%"
        )

    def test_a_mode_without_a_model_says_that_the_profile_values_are_assumed(self) -> None:
        fit = SnapshotFit("bin1", 4, 0.5, 40.0, 0.1, 0.3, 37.6, False)
        (line,) = format_snapshot_fits([fit])
        assert "(profile 0.300, no model: assumed)" in line
        assert "(profile 37.6, no model: assumed)" in line

    def test_the_report_prints_the_snapshot_fit_only_when_it_has_one(
        self, snapshot_report: rates.RatesReport
    ) -> None:
        text = format_report(snapshot_report)
        assert "Single-exposure fit (the ROI height against the time beyond the exposure):" in text
        assert "snapshot_overhead_s = 0.27" in text
        assert "snapshot 4144x2822, 1 ms" in text
        assert "Timing fit at bandwidth 100" not in text  # one video row is no line
        assert "Single-exposure fit" not in format_report(table(make_rig(), groups=("roi",)))

    def test_the_json_carries_the_snapshot_rows_and_the_snapshot_fit(
        self, snapshot_report: rates.RatesReport
    ) -> None:
        data = report_to_json(snapshot_report)
        assert [row["kind"] for row in data["rows"]] == ["video", *["snapshot"] * 4]
        (fit,) = data["snapshot_fits"]
        assert fit["mode"] == "bin2"
        assert fit["profile_has_model"] is True
        assert fit["overhead_s"] == pytest.approx(0.27, abs=0.003)
        json.dumps(data)  # plain JSON types only


class TestCommand:
    @pytest.fixture
    def rig(self, monkeypatch: pytest.MonkeyPatch) -> Rig:
        rig = make_rig()
        monkeypatch.setattr(rates, "create_driver", lambda profile, options, clock=None: rig.driver)
        # The command times single exposures on its clock, which must be the clock of the driver.
        monkeypatch.setattr("seeingmon.clock.SystemClock", lambda: rig.clock)
        return rig

    def run(self, tmp_path: Path, *args: str) -> list[str]:
        return ["camera", "rates", "--local-config", str(tmp_path / "none.toml"), *args]

    def test_the_parser_lists_camera_and_rates(self) -> None:
        parser = build_parser()
        assert "camera" in parser.format_help()
        args = parser.parse_args(["camera", "rates", "--frames", "20", "--json"])
        assert (args.command, args.camera_command, args.frames) == ("camera", "rates", 20)
        assert args.json == Path("local") / "camera-rates.json"
        assert (args.settle, args.gain) == (10, 120)

    def test_an_unwritable_json_path_is_an_error_message_after_the_table(
        self, rig: Rig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        blocked = tmp_path / "a-file"
        blocked.write_text("x", encoding="utf-8")  # a folder cannot be made below a file
        code = main(
            self.run(
                tmp_path,
                "--frames",
                "6",
                "--settle",
                "1",
                "--groups",
                "format",
                "--json",
                str(blocked / "rates.json"),
            )
        )
        captured = capsys.readouterr()
        assert code == 1
        assert "baseline" in captured.out  # the table came first
        assert "cannot write" in captured.err
        assert "--json PATH" in captured.err

    def test_the_command_prints_the_table_and_writes_the_json(
        self, rig: Rig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "out" / "rates.json"
        code = main(
            self.run(
                tmp_path,
                "--frames",
                "6",
                "--settle",
                "1",
                "--groups",
                "bandwidth,format",
                "--json",
                str(target),
            )
        )
        out = capsys.readouterr().out
        assert code == 0
        assert out.startswith("Frame rates of ZWO ASI294MM (fake)")
        for label in ("baseline", "RAW8", "bandwidth 40", "bandwidth 90"):
            assert label in out
        assert "exposure 100 us" not in out  # the groups that you did not ask for
        assert "The camera is back as it was" in out
        assert target.is_file()
        assert json.loads(target.read_text(encoding="utf-8"))["restore_problems"] == []
        assert not rig.sdk.video_active

    def test_the_snapshot_group_prints_the_rows_and_the_fit_and_writes_the_json(
        self, rig: Rig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "out" / "snapshots.json"
        code = main(
            self.run(
                tmp_path,
                "--frames",
                "6",
                "--settle",
                "1",
                "--groups",
                "snapshot",
                "--json",
                str(target),
            )
        )
        out = capsys.readouterr().out
        assert code == 0
        for label in ("snapshot 4144x64, 1 ms", "snapshot 4144x2822, 1 ms"):
            assert label in out
        assert "Single-exposure fit" in out
        assert "snapshot_overhead_s = 0.27" in out
        assert "snapshot_row_time_us = 7" in out  # about 75 us
        assert "bandwidth 40" not in out  # the groups that you did not ask for
        data = json.loads(target.read_text(encoding="utf-8"))
        assert len(data["snapshot_fits"]) == 1
        assert not rig.sdk.video_active

    def test_the_help_names_the_snapshot_group_and_the_caps_of_the_snapshot_rows(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["camera", "rates", "--help"])
        text = " ".join(capsys.readouterr().out.split())
        assert "bin2, and snapshot (default: all)" in text
        assert f"a snapshot row takes at most {SNAPSHOT_MAX_FRAMES} exposures" in text
        assert f"a snapshot row drops at most {SNAPSHOT_MAX_SETTLE} exposures" in text

    def test_a_group_that_does_not_exist_is_a_usage_error(
        self, rig: Rig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(self.run(tmp_path, "--groups", "sleep")) == 2
        assert "unknown row group" in capsys.readouterr().err
        assert rig.sdk.calls == []  # it never touched the camera

    def test_too_few_frames_is_a_usage_error(
        self, rig: Rig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(self.run(tmp_path, "--frames", "2")) == 2
        assert "--frames must be at least 3" in capsys.readouterr().err

    def test_a_failed_row_gives_exit_code_1(
        self, rig: Rig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rig.sdk.fail_next("set_roi_format", AsiErrorCode.INVALID_SIZE)
        code = main(self.run(tmp_path, "--frames", "5", "--settle", "0", "--groups", "format"))
        out = capsys.readouterr().out
        assert code == 1
        assert "FAILED AsiConfigError" in out
        assert "RAW8" in out  # the table went on

    def test_a_library_that_is_missing_is_a_message_and_exit_code_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def missing(profile: object, options: object, clock: object = None) -> object:
            raise AsiLibraryError("the ASI library is not installed or not on the search path")

        monkeypatch.setattr(rates, "create_driver", missing)
        assert main(self.run(tmp_path)) == 1
        assert "not installed" in capsys.readouterr().err

    def test_a_camera_that_is_not_there_is_a_message_and_exit_code_1(
        self, rig: Rig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rig.sdk.disconnect()
        assert main(self.run(tmp_path)) == 1
        assert "cannot measure the camera" in capsys.readouterr().err
