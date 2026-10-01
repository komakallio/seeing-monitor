"""The analysis interfaces, their fakes, and a check that each fake satisfies its protocol."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.analysis import (
    NO_STAR,
    FastAnalyzer,
    FastContext,
    MetricsWriter,
    PointingProvider,
    RecordWriter,
    SurveyAnalyzer,
)
from seeingmon.clock import VirtualClock
from seeingmon.frames import ActiveStream, Frame, PixelFormat, Roi, StreamConfig
from seeingmon.records import SeeingWindowRecord
from seeingmon.records.segments import segment_dtype
from seeingmon.testing import (
    FakeCameraDriver,
    FakeFastAnalyzer,
    FakePointingProvider,
    FakeSurveyAnalyzer,
    ListRecordWriter,
)

ROI = Roi(200, 300, 64, 64)
CONFIG = StreamConfig(
    mode="bin1", exposure_us=1_000_000, gain=120, roi=ROI, pixel_format=PixelFormat.RAW16
)


def star_at_center(config: StreamConfig, roi: Roi, seq: int) -> npt.NDArray[np.uint16]:
    data = np.full((roi.height, roi.width), 100, dtype=np.uint16)
    data[roi.height // 2, roi.width // 2] = 5000
    return data


def flat(config: StreamConfig, roi: Roi, seq: int) -> npt.NDArray[np.uint16]:
    return np.full((roi.height, roi.width), 100, dtype=np.uint16)


def stream_of(driver: FakeCameraDriver, config: StreamConfig = CONFIG) -> ActiveStream:
    active = driver.configure(config)
    driver.start()
    return active


def read(driver: FakeCameraDriver, count: int) -> list[Frame]:
    return [driver.read_frame(timeout_s=5.0) for _ in range(count)]


def test_the_fakes_satisfy_the_protocols() -> None:
    assert isinstance(FakeFastAnalyzer(), FastAnalyzer)
    assert isinstance(FakeSurveyAnalyzer(), SurveyAnalyzer)
    assert isinstance(FakePointingProvider(), PointingProvider)
    assert isinstance(ListRecordWriter(), RecordWriter)
    assert isinstance(ListRecordWriter(), MetricsWriter)


class TestFakeFastAnalyzer:
    @pytest.fixture
    def driver(self) -> FakeCameraDriver:
        fake = FakeCameraDriver(VirtualClock(), frame_factory=star_at_center)
        fake.open()
        return fake

    def test_closes_a_window_every_sixty_seconds_of_frame_time(
        self, driver: FakeCameraDriver
    ) -> None:
        analyzer = FakeFastAnalyzer(station_id="s1", profile_id="p1")
        stream = stream_of(driver)
        analyzer.begin_stream(stream)
        windows: list[SeeingWindowRecord] = []
        for frame in read(driver, 140):
            windows += analyzer.push(frame).windows
        windows += analyzer.flush()
        assert [w.n_frames for w in windows] == [60, 60, 20]
        first, _, last = windows
        assert (first.station_id, first.profile_id, first.stream_id) == (
            "s1",
            "p1",
            stream.stream_id,
        )
        assert first.readout_mode == "bin1"
        assert first.exposure_us == 1_000_000
        assert first.duration_s == pytest.approx(60.0)
        assert first.frame_rate_hz == pytest.approx(1.0, rel=0.02)
        assert first.valid_fraction == 1.0
        assert first.flags == []
        assert windows[1].t_utc_ns - first.t_utc_ns == 60 * 1_000_000_000
        assert last.flags == ["partial"]
        assert first.sensor_temperature_c == pytest.approx(18.0)

    def test_tracks_the_star_in_sensor_coordinates(self, driver: FakeCameraDriver) -> None:
        analyzer = FakeFastAnalyzer()
        analyzer.begin_stream(stream_of(driver))
        update = analyzer.push(driver.read_frame(5.0))
        assert update.star.found
        assert (update.star.x_px, update.star.y_px) == (ROI.x + 32, ROI.y + 32)
        assert update.star.edge_distance_px == 32
        assert update.star.peak_fraction == pytest.approx(5000 / 65535)

    def test_a_missing_star_is_reported_and_the_state_resets_on_a_new_stream(self) -> None:
        driver = FakeCameraDriver(VirtualClock(), frame_factory=flat)
        driver.open()
        analyzer = FakeFastAnalyzer()
        analyzer.begin_stream(stream_of(driver))
        assert analyzer.push(driver.read_frame(5.0)).star == NO_STAR

    def test_begin_stream_closes_the_open_window_as_partial(self, driver: FakeCameraDriver) -> None:
        analyzer = FakeFastAnalyzer()
        analyzer.begin_stream(stream_of(driver))
        for frame in read(driver, 5):
            analyzer.push(frame)
        closed = analyzer.begin_stream(stream_of(driver))
        assert [(w.n_frames, w.flags) for w in closed] == [(5, ["partial"])]
        assert analyzer.star == NO_STAR
        assert analyzer.begin_stream(stream_of(driver)) == ()

    def test_a_frame_from_another_stream_starts_that_stream(self, driver: FakeCameraDriver) -> None:
        analyzer = FakeFastAnalyzer()
        first = stream_of(driver)
        analyzer.begin_stream(first)
        for frame in read(driver, 3):
            analyzer.push(frame)
        second = stream_of(driver)
        update = analyzer.push(driver.read_frame(5.0))
        assert [(w.stream_id, w.n_frames) for w in update.windows] == [(first.stream_id, 3)]
        assert second.stream_id != first.stream_id

    def test_the_context_adds_flags_and_values_to_closing_windows(
        self, driver: FakeCameraDriver
    ) -> None:
        analyzer = FakeFastAnalyzer()
        analyzer.begin_stream(stream_of(driver))
        analyzer.set_context(
            FastContext(
                flags=frozenset({"cloud", "heater_on"}), heater_duty=0.25, zenith_angle_deg=35.5
            )
        )
        for frame in read(driver, 3):
            analyzer.push(frame)
        (window,) = analyzer.flush()
        assert window.flags == ["cloud", "heater_on", "partial"]
        assert (window.heater_duty, window.zenith_angle_deg) == (0.25, 35.5)

    def test_dropped_frames_make_a_window_degraded(self, driver: FakeCameraDriver) -> None:
        analyzer = FakeFastAnalyzer()
        analyzer.begin_stream(stream_of(driver))
        frames = read(driver, 10)
        analyzer.push(frames[0])
        analyzer.push(replace(frames[1], dropped_before=4))
        for frame in frames[2:]:
            analyzer.push(frame)
        (window,) = analyzer.flush()
        assert window.n_dropped == 4
        assert window.valid_fraction == pytest.approx(10 / 14)
        assert "degraded" in window.flags

    def test_metrics_use_the_frame_record_dtype_and_drain_once(
        self, driver: FakeCameraDriver
    ) -> None:
        analyzer = FakeFastAnalyzer()
        analyzer.begin_stream(stream_of(driver))
        assert analyzer.drain_metrics() is None
        for frame in read(driver, 7):
            analyzer.push(frame)
        rows = analyzer.drain_metrics()
        assert rows is not None
        assert rows.dtype == segment_dtype("frame")
        assert rows["seq"].tolist() == list(range(7))
        assert rows["cx_px"][0] == ROI.x + 32
        assert rows["peak_dn"][0] == 5000
        assert analyzer.drain_metrics() is None

    def test_windows_validate_as_records(self, driver: FakeCameraDriver) -> None:
        analyzer = FakeFastAnalyzer()
        analyzer.begin_stream(stream_of(driver))
        analyzer.push(driver.read_frame(5.0))
        (window,) = analyzer.flush()
        assert window.record_type == "seeing_window"
        assert type(window).from_row(window.to_row()) == window


class TestFakeSurveyAnalyzer:
    @pytest.fixture
    def frame(self) -> Frame:
        driver = FakeCameraDriver(VirtualClock(), frame_factory=flat)
        driver.open()
        driver.configure(
            StreamConfig(mode="bin2", exposure_us=30_000_000, gain=120, roi=Roi(0, 0, 64, 64))
        )
        driver.start()
        return driver.read_frame(timeout_s=60.0)

    def test_returns_a_survey_frame_record_on_the_next_poll(self, frame: Frame) -> None:
        analyzer = FakeSurveyAnalyzer(station_id="s1", cloud_fraction=0.2)
        assert analyzer.poll() == ()
        analyzer.submit(frame)
        assert analyzer.pending() == 1
        (output,) = analyzer.poll()
        assert analyzer.pending() == 0
        assert (output.t_utc_ns, output.solved, output.cloud_fraction) == (
            frame.t_utc_ns,
            True,
            0.2,
        )
        (record,) = output.records
        assert record.record_type == "survey_frame"
        assert record.station_id == "s1"
        assert analyzer.poll() == ()

    def test_results_wait_for_the_scripted_number_of_polls_and_keep_their_order(
        self, frame: Frame
    ) -> None:
        analyzer = FakeSurveyAnalyzer(polls_until_ready=2)
        analyzer.submit(frame)
        analyzer.submit(replace(frame, seq=1, t_utc_ns=frame.t_utc_ns + 1))
        assert analyzer.poll() == ()
        assert analyzer.poll() == ()
        outputs = analyzer.poll()
        assert [o.t_utc_ns for o in outputs] == [frame.t_utc_ns, frame.t_utc_ns + 1]
        assert analyzer.pending() == 0

    def test_the_outcome_is_fixed_when_the_frame_is_submitted(self, frame: Frame) -> None:
        analyzer = FakeSurveyAnalyzer(solved=False, cloud_fraction=None, polls_until_ready=1)
        analyzer.submit(frame)
        analyzer.solved = True
        assert analyzer.poll() == ()
        (output,) = analyzer.poll()
        assert (output.solved, output.cloud_fraction) == (False, None)


class TestFakePointingProvider:
    def test_without_a_solution_there_is_no_position(self) -> None:
        provider = FakePointingProvider()
        assert provider.polaris_position(0, "bin1") is None

    def test_returns_the_scripted_position_and_drift(self) -> None:
        provider = FakePointingProvider({"bin1": (4000.0, 2800.0)})
        assert provider.polaris_position(10**18, "bin1") == (4000.0, 2800.0)
        assert provider.polaris_position(0, "bin2") is None
        provider.set_solution("bin2", 100.0, 200.0, t0_utc_ns=0, drift_px_per_s=(0.5, -0.25))
        assert provider.polaris_position(4_000_000_000, "bin2") == (102.0, 199.0)

    def test_clear_loses_every_solution(self) -> None:
        provider = FakePointingProvider({"bin1": (1.0, 2.0)})
        provider.clear()
        assert provider.polaris_position(0, "bin1") is None


def test_the_list_writer_keeps_records_and_copies_metrics() -> None:
    writer = ListRecordWriter()
    rows = np.zeros(3, dtype=segment_dtype("frame"))
    writer.write_metrics(4, rows)
    rows["seq"] = 9
    assert writer.metrics[0][0] == 4
    assert writer.metrics[0][1]["seq"].tolist() == [0, 0, 0]
    assert writer.of_type("seeing_window") == []
