"""The rolling seeing value: the ring, the schedule, and the agreement with a stored window.

The live value must be the stored estimate on fewer frames, and nothing else. The tests feed one
set of frames to a live analyzer and to a stored analyzer, and they compare the numbers.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from functools import cache
from typing import Any, TypedDict

import numpy as np
import pytest

from seeingmon.analysis import FastContext
from seeingmon.fastpath import FastPathAnalyzer, FastPathConfig
from seeingmon.fastpath.estimator import EstimatorSettings
from seeingmon.fastpath.live import (
    COLUMNS,
    FLUX,
    LIVE_FIELDS,
    LOST,
    NVX,
    PEAK,
    SATURATED,
    SLOT,
    USABLE,
    WX,
    LiveEstimator,
    LiveRing,
    LiveSeeing,
    LiveStream,
    T,
    X,
    Y,
)
from seeingmon.fastpath.windows import WindowAssembler
from seeingmon.frames import ActiveStream, Frame, PixelFormat, Roi, StreamConfig
from seeingmon.profile import Profile
from tests.fastpath.helpers import (
    BIN1_PERIOD_S,
    EPOCH_NS,
    POLARIS_ELECTRONS_2MS,
    box_integrated_gaussian,
    digitize,
    make_frame,
    synthetic_tilt_arcsec,
)

NS = 1_000_000_000
PERIOD_NS = round(BIN1_PERIOD_S * NS)
ROI = Roi(1000, 2000, 64, 64)
PLATE_SCALE = 1.91
TILT_FRAMES = 2000
CONFIG = FastPathConfig(window_s=60.0, min_window_s=3.0)  # the stored window needs 3 s, not 5 s


class Row(TypedDict):
    t_ns: int
    usable: bool
    saturated: bool
    x: float
    y: float
    width_x: float
    width_y: float
    noise_var_x: float
    noise_var_y: float
    peak: float
    flux: float
    dropped_before: int


def row(index: int = 0, **changes: Any) -> Row:
    """The values of one frame for `LiveRing.add`. Keyword arguments replace fields."""
    fields: Row = {
        "t_ns": EPOCH_NS + index * PERIOD_NS,
        "usable": True,
        "saturated": False,
        "x": 100.0,
        "y": 200.0,
        "width_x": 1.0,
        "width_y": 1.1,
        "noise_var_x": 1e-4,
        "noise_var_y": 1e-4,
        "peak": 5000.0,
        "flux": 14_000.0,
        "dropped_before": 0,
    }
    fields.update(changes)  # type: ignore[typeddict-item]
    return fields


@cache
def tilt_px(r0_m: float, seed: int) -> np.ndarray:
    """The image motion of a star in pixels, from a series with the statistics of the model."""
    arcsec = synthetic_tilt_arcsec(
        r0_m, rate_hz=1.0 / BIN1_PERIOD_S, samples=TILT_FRAMES, seed=seed
    )
    return np.asarray(arcsec / PLATE_SCALE, dtype=np.float64)


def make_frames(
    count: int,
    *,
    r0_m: float = 0.10,
    seed: int = 1,
    lost: dict[int, int] | None = None,
    flux: float = POLARIS_ELECTRONS_2MS,
    mode: str = "bin1",
    stream_id: int = 1,
    exposure_us: int = 2000,
    start: int = 0,
) -> list[Frame]:
    """Frames of a star that moves as the turbulence says.

    `lost` maps the index of a frame to the number of frames that the camera lost just before it.
    """
    rng = np.random.default_rng(seed + 100)
    motion = tilt_px(r0_m, seed)
    assert start + count + sum((lost or {}).values()) < TILT_FRAMES
    frames: list[Frame] = []
    seq = start
    for index in range(count):
        dropped = (lost or {}).get(index, 0)
        seq += dropped
        x, y = 32.0 + motion[seq, 0], 32.0 + motion[seq, 1]
        data = digitize(box_integrated_gaussian((64, 64), x, y, 1.1, flux), rng=rng)
        frames.append(
            make_frame(
                data,
                seq=seq + 1000 * (stream_id - 1),
                stream_id=stream_id,
                t_ns=EPOCH_NS + round(seq * BIN1_PERIOD_S * NS) + 7 * NS * (stream_id - 1),
                dropped_before=dropped,
                roi=ROI,
                mode=mode,
                exposure_us=exposure_us,
                gain=0,
            )
        )
        seq += 1
    return frames


@pytest.fixture(scope="module")
def frames_12s() -> list[Frame]:
    """About 12 seconds of frames with a few lost frames."""
    return make_frames(990, lost={100: 2, 400: 1, 401: 3, 700: 5})


def analyzer_of(profile: Profile, **options: Any) -> FastPathAnalyzer:
    config = CONFIG.model_copy(update=options)
    analyzer = FastPathAnalyzer(profile, config, station_id="live")
    analyzer.set_context(FastContext(zenith_angle_deg=30.0, flags=frozenset({"cloud"})))
    return analyzer


def a_stream() -> LiveStream:
    settings = EstimatorSettings(
        aperture_m=0.05, plate_scale_arcsec_per_px=PLATE_SCALE, exposure_s=0.002
    )
    return LiveStream(1, "bin1", 2000, settings, PLATE_SCALE, True)


# --- The ring --------------------------------------------------------------------------------


class TestRing:
    def test_a_row_holds_the_time_the_slot_the_flags_and_the_values(self) -> None:
        ring = LiveRing(32)
        ring.add(**row(0))
        ring.add(**row(1, usable=False, saturated=True, x=101.5))
        rows = ring.recent(60.0)
        assert rows.shape == (2, COLUMNS)
        assert rows[0, T] == 0.0
        assert rows[1, T] == pytest.approx(BIN1_PERIOD_S, abs=1e-9)
        assert list(rows[:, SLOT]) == [0, 1]
        assert list(rows[:, USABLE]) == [1, 0]
        assert list(rows[:, SATURATED]) == [0, 1]
        assert (rows[1, X], rows[1, Y], rows[1, WX]) == (101.5, 200.0, 1.0)
        assert (rows[0, NVX], rows[0, PEAK], rows[0, FLUX]) == (1e-4, 5000.0, 14_000.0)
        assert ring.base_ns == EPOCH_NS
        assert len(ring) == 2

    def test_a_lost_frame_leaves_an_empty_slot_and_never_a_time_shift(self) -> None:
        ring = LiveRing(64)
        ring.add(**row(0))
        ring.add(**row(1))
        ring.add(**row(4, dropped_before=2))  # the counter says that two frames were lost
        ring.add(**row(5))
        rows = ring.recent(60.0)
        assert list(rows[:, SLOT]) == [0, 1, 4, 5]  # slot = previous + 1 + lost
        assert list(rows[:, LOST]) == [0, 0, 2, 0]

    def test_a_gap_in_time_counts_as_lost_frames_when_the_counter_says_nothing(self) -> None:
        ring = LiveRing(64)
        for index in (0, 1, 2, 3):
            ring.add(**row(index))
        ring.add(**row(7))  # three frames are missing, and `dropped_before` is 0
        assert list(ring.recent(60.0)[:, SLOT]) == [0, 1, 2, 3, 7]
        assert ring.recent(60.0)[-1, LOST] == 3

    def test_the_slots_follow_the_window_assembler_frame_for_frame(self) -> None:
        ring = LiveRing(2048)
        assembler = WindowAssembler(60.0)
        lost = {50: 1, 120: 4, 121: 0, 300: 2}
        index = 0
        for count in range(400):
            index += lost.get(count, 0)
            t_ns = EPOCH_NS + index * PERIOD_NS
            counted = lost.get(count, 0) if count % 2 else 0  # the camera counts half of the drops
            ring.add(**row(0, t_ns=t_ns, dropped_before=counted))
            values = (100.0, 200.0, 1.0, 1.1, 5000.0, 14_000.0, 0.0, 1e-4, 1e-4)
            assembler.add(1, t_ns, counted, "bin1", 2000, 0, None, False, True, False, values)
            index += 1
        (window,) = assembler.flush("end")
        rows = ring.recent(60.0)
        assert np.array_equal(rows[:, SLOT].astype(int), window.slot)
        assert int(rows[:, LOST][1:].sum()) == window.n_dropped

    def test_a_time_that_does_not_advance_starts_a_new_run(self) -> None:
        ring = LiveRing(32)
        for index in range(5):
            ring.add(**row(index))
        ring.add(**row(2))  # the clock stepped back
        assert len(ring) == 1
        assert ring.base_ns == EPOCH_NS + 2 * PERIOD_NS
        ring.add(**row(2))  # the same time again
        assert len(ring) == 1

    def test_the_newest_rows_survive_the_compaction_in_one_block(self) -> None:
        ring = LiveRing(16)
        for index in range(100):
            ring.add(**row(index, peak=float(index)))
        rows = ring.recent(1e9)
        assert 16 <= len(rows) <= 32
        assert rows.base is not None  # a view, not a copy
        assert list(rows[:, PEAK]) == [float(i) for i in range(100 - len(rows), 100)]
        assert np.all(np.diff(rows[:, T]) > 0)
        assert np.all(np.diff(rows[:, SLOT]) == 1)

    def test_recent_gives_the_newest_span(self) -> None:
        ring = LiveRing(512)
        for index in range(400):
            ring.add(**row(index))
        rows = ring.recent(2.0)
        span = rows[-1, T] - rows[0, T]
        assert span <= 2.0
        assert span > 2.0 - 2 * BIN1_PERIOD_S
        assert rows[-1, T] == ring.recent(60.0)[-1, T]
        assert len(ring.recent(0.0)) == 1  # the newest frame alone

    def test_reset_empties_the_ring_and_takes_a_period_hint(self) -> None:
        ring = LiveRing(32)
        ring.add(**row(0))
        ring.reset(BIN1_PERIOD_S)
        assert len(ring) == 0
        assert len(ring.recent(10.0)) == 0
        assert ring.period_s == pytest.approx(BIN1_PERIOD_S)
        ring.reset(None)
        assert ring.period_s is None

    def test_the_period_estimate_follows_the_frames(self) -> None:
        ring = LiveRing(64)
        for index in range(40):
            ring.add(**row(index))
        assert ring.period_s == pytest.approx(BIN1_PERIOD_S, rel=1e-3)

    def test_a_ring_of_a_few_frames_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 16"):
            LiveRing(8)


# --- The schedule ----------------------------------------------------------------------------


def feed_estimator(estimator: LiveEstimator, indices: range) -> list[int]:
    """Add the frames, and return the indices at which an estimate was due."""
    return [index for index in indices if estimator.add(**row(index))]


class TestSchedule:
    @staticmethod
    def config() -> FastPathConfig:
        return FastPathConfig(live_span_s=10.0, live_every_s=2.0, live_min_span_s=4.0)

    def test_nothing_is_due_before_the_minimum_span(self) -> None:
        estimator = LiveEstimator(self.config())
        assert feed_estimator(estimator, range(int(3.9 / BIN1_PERIOD_S))) == []

    def test_the_first_estimate_is_due_when_the_stream_has_run_for_the_minimum_span(self) -> None:
        estimator = LiveEstimator(self.config())
        due = feed_estimator(estimator, range(int(4.5 / BIN1_PERIOD_S)))
        assert due[0] == pytest.approx(4.0 / BIN1_PERIOD_S, abs=1.5)
        assert len(due) > 1  # it stays due until somebody takes the estimate

    def test_after_an_estimate_the_next_one_is_due_two_seconds_later(self) -> None:
        estimator = LiveEstimator(self.config())
        newest = int(4.5 / BIN1_PERIOD_S)
        feed_estimator(estimator, range(newest))
        assert estimator.estimate(a_stream(), frozenset(), None) is not None
        # The estimate ended at the newest frame, so the next is due two seconds after that.
        later = feed_estimator(estimator, range(newest, newest + int(2.5 / BIN1_PERIOD_S)))
        assert later[0] == pytest.approx(newest - 1 + 2.0 / BIN1_PERIOD_S, abs=2.0)

    def test_a_new_stream_starts_the_schedule_again(self) -> None:
        estimator = LiveEstimator(self.config())
        feed_estimator(estimator, range(int(5.0 / BIN1_PERIOD_S)))
        estimator.estimate(a_stream(), frozenset(), None)
        estimator.reset()
        assert len(estimator.ring) == 0
        assert feed_estimator(estimator, range(int(3.0 / BIN1_PERIOD_S))) == []

    def test_a_step_back_in_time_starts_the_schedule_again(self) -> None:
        estimator = LiveEstimator(self.config())
        feed_estimator(estimator, range(int(5.0 / BIN1_PERIOD_S)))
        estimator.estimate(a_stream(), frozenset(), None)
        assert not estimator.add(**row(0))  # the time goes back
        assert len(estimator.ring) == 1

    def test_an_estimate_needs_two_frames(self) -> None:
        estimator = LiveEstimator(self.config())
        assert estimator.estimate(a_stream(), frozenset(), None) is None
        estimator.add(**row(0))
        assert estimator.estimate(a_stream(), frozenset(), None) is None


# --- The agreement with a stored window ------------------------------------------------------


def stored_value(profile: Profile, frames: list[Frame]) -> dict[str, Any]:
    """The estimate of a stored window over exactly these frames, as the record gives it."""
    analyzer = analyzer_of(profile)
    for frame in frames:
        analyzer.push(frame)
    (window,) = analyzer.flush("end")
    return window.model_dump()


def live_values(profile: Profile, frames: list[Frame]) -> list[tuple[int, LiveSeeing]]:
    """The live values, with the number of frames that had been pushed when each one appeared."""
    analyzer = analyzer_of(profile)
    values: list[tuple[int, LiveSeeing]] = []
    last: LiveSeeing | None = None
    for count, frame in enumerate(frames, start=1):
        analyzer.push(frame)
        if analyzer.live is not None and analyzer.live is not last:
            last = analyzer.live
            values.append((count, last))
    return values


class TestAgreementWithAStoredWindow:
    def test_a_live_value_is_the_stored_estimate_of_the_same_frames(
        self, profile: Profile, frames_12s: list[Frame]
    ) -> None:
        values = live_values(profile, frames_12s)
        assert len(values) >= 3  # estimates at about 4, 6, 8, 10, and 12 seconds
        compared = 0
        for count, live in values:
            if live.n_frames != count:
                continue  # the span no longer reaches back to the first frame
            compared += 1
            stored = stored_value(profile, frames_12s[:count])
            assert live.n_frames == stored["n_frames"]
            assert live.span_s == pytest.approx(stored["duration_s"], abs=1e-3)
            assert live.valid_fraction == pytest.approx(stored["valid_fraction"], abs=1e-4)
            for name in LIVE_FIELDS:
                assert getattr(live, name) is not None, name
                assert getattr(live, name) == pytest.approx(stored[name], rel=1e-9), name
            # A live value has no `partial` (no span is short) and no `vibration` (no spectrum).
            assert set(live.flags) == set(stored["flags"]) - {"partial", "vibration"}
            assert (live.stream_id, live.readout_mode, live.exposure_us) == (
                stored["stream_id"],
                stored["readout_mode"],
                stored["exposure_us"],
            )
        assert compared >= 3

    def test_a_value_over_a_full_span_is_the_stored_estimate_of_the_newest_frames(
        self, profile: Profile, frames_12s: list[Frame]
    ) -> None:
        count, live = live_values(profile, frames_12s)[-1]
        assert live.span_s > 9.9  # the span of the newest estimate is full
        first = next(
            index
            for index, frame in enumerate(frames_12s[:count])
            if (frames_12s[count - 1].t_utc_ns - frame.t_utc_ns) / NS <= 10.0
        )
        stored = stored_value(profile, frames_12s[first:count])
        assert live.n_frames == stored["n_frames"]
        for name in LIVE_FIELDS:
            assert getattr(live, name) == pytest.approx(stored[name], rel=1e-9), name

    def test_the_value_is_the_seeing_of_the_simulated_turbulence(
        self, profile: Profile, frames_12s: list[Frame]
    ) -> None:
        live = live_values(profile, frames_12s)[-1][1]
        assert live.seeing_fwhm_arcsec is not None
        assert live.r0_cm is not None
        # r0 is 10 cm along the line of sight, and the context puts the star 30 degrees off the
        # zenith, so the value at the zenith is r0 / cos(30 degrees)^(3/5).
        zenith_r0_cm = 10.0 / np.cos(np.radians(30.0)) ** 0.6
        assert live.r0_cm == pytest.approx(zenith_r0_cm, rel=0.3)  # a span has few samples
        assert live.seeing_fwhm_arcsec == pytest.approx(
            0.98 * 500e-9 / (zenith_r0_cm / 100.0) * 206264.8, rel=0.3
        )

    def test_the_context_flags_and_the_stream_reach_the_value(
        self, profile: Profile, frames_12s: list[Frame]
    ) -> None:
        live = live_values(profile, frames_12s)[0][1]
        assert "cloud" in live.flags
        assert live.readout_mode == "bin1"
        assert live.exposure_us == 2000
        assert live.quality == {}  # every value is there


# --- The analyzer ----------------------------------------------------------------------------


class TestAnalyzer:
    def test_there_is_no_value_before_the_minimum_span_and_then_one_every_two_seconds(
        self, profile: Profile, frames_12s: list[Frame]
    ) -> None:
        analyzer = analyzer_of(profile)
        seen: list[tuple[float, LiveSeeing]] = []
        last: LiveSeeing | None = None
        for frame in frames_12s:
            analyzer.push(frame)
            if analyzer.live is not last and analyzer.live is not None:
                last = analyzer.live
                seen.append(((frame.t_utc_ns - frames_12s[0].t_utc_ns) / NS, last))
        assert seen[0][0] == pytest.approx(4.0, abs=0.05)
        steps = np.diff([when for when, _ in seen])
        assert np.allclose(steps, 2.0, atol=0.05)
        assert analyzer.live is seen[-1][1]

    def test_the_value_is_immutable(self, profile: Profile, frames_12s: list[Frame]) -> None:
        live = live_values(profile, frames_12s)[0][1]
        with pytest.raises(dataclasses.FrozenInstanceError):
            live.span_s = 1.0  # type: ignore[misc]

    def test_a_new_stream_clears_the_value_and_the_ring(
        self, profile: Profile, frames_12s: list[Frame]
    ) -> None:
        analyzer = analyzer_of(profile)
        for frame in frames_12s[:500]:
            analyzer.push(frame)
        assert analyzer.live is not None
        analyzer.push(make_frames(1, stream_id=2)[0])  # a frame of another stream
        live_estimator = analyzer._live
        assert live_estimator is not None
        assert len(live_estimator.ring) == 1
        assert getattr(analyzer, "live") is None  # noqa: B009 - read without a narrowed type

    def test_a_changed_exposure_clears_the_value(
        self, profile: Profile, frames_12s: list[Frame]
    ) -> None:
        analyzer = analyzer_of(profile)
        for frame in frames_12s[:500]:
            analyzer.push(frame)
        assert analyzer.live is not None
        analyzer.push(make_frames(1, exposure_us=8000, start=500)[0])
        assert analyzer.live is None

    def test_begin_stream_clears_the_value(self, profile: Profile, frames_12s: list[Frame]) -> None:
        analyzer = analyzer_of(profile)
        for frame in frames_12s[:500]:
            analyzer.push(frame)
        assert analyzer.live is not None
        analyzer.begin_stream(
            ActiveStream(
                stream_id=9,
                config=StreamConfig("bin1", 2000, 0, roi=ROI, pixel_format=PixelFormat.RAW16),
                frame_shape=(64, 64),
                adc_bits=12,
                frame_period_s=BIN1_PERIOD_S,
            )
        )
        assert analyzer.live is None

    def test_live_enabled_false_keeps_no_ring_and_no_value(
        self, profile: Profile, frames_12s: list[Frame]
    ) -> None:
        analyzer = analyzer_of(profile, live_enabled=False)
        for frame in frames_12s[:500]:
            analyzer.push(frame)
        assert analyzer._live is None
        assert analyzer.live is None

    def test_the_stored_windows_and_the_metrics_do_not_change(
        self, profile: Profile, frames_12s: list[Frame]
    ) -> None:
        on, off = analyzer_of(profile), analyzer_of(profile, live_enabled=False)
        for frame in frames_12s:
            on.push(frame)
            off.push(frame)
        assert [w.model_dump() for w in on.flush()] == [w.model_dump() for w in off.flush()]
        rows_on, rows_off = on.drain_metrics(), off.drain_metrics()
        assert rows_on is not None
        assert rows_off is not None
        assert rows_on.tobytes() == rows_off.tobytes()

    def test_a_failure_of_the_estimate_never_reaches_the_caller(
        self,
        profile: Profile,
        frames_12s: list[Frame],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def explode(*args: object, **kwargs: object) -> None:
            raise RuntimeError("the estimator broke")

        monkeypatch.setattr("seeingmon.fastpath.live.estimate_seeing", explode)
        analyzer = analyzer_of(profile)
        with caplog.at_level(logging.ERROR, logger="seeingmon.fastpath.analyzer"):
            for frame in frames_12s[:700]:
                analyzer.push(frame)  # no exception
        assert analyzer.live is None
        assert analyzer.live_errors >= 1
        assert 1 <= len(caplog.records) <= 3  # the log does not fill up
        assert analyzer.frames_pushed == 700
        monkeypatch.undo()
        for frame in frames_12s[700:]:
            analyzer.push(frame)
        assert analyzer.live is not None  # the next estimate works

    def test_the_value_of_a_sky_with_no_star_says_why_each_number_is_missing(
        self, profile: Profile
    ) -> None:
        analyzer = analyzer_of(profile)
        rng = np.random.default_rng(3)
        for index in range(500):
            data = digitize(np.zeros((64, 64)), rng=rng)
            analyzer.push(make_frame(data, seq=index, t_ns=EPOCH_NS + index * PERIOD_NS, roi=ROI))
        live = analyzer.live
        assert live is not None
        assert live.n_usable == 0
        assert live.valid_fraction == 0.0
        for name in LIVE_FIELDS:
            assert getattr(live, name) is None
            assert live.quality[name]
        assert live.quality["seeing_fwhm_arcsec"] == "too few usable frames"

    def test_a_readout_mode_that_the_profile_does_not_know_gives_notes_and_no_numbers(
        self, profile: Profile
    ) -> None:
        analyzer = analyzer_of(profile)
        for frame in make_frames(500, mode="odd-mode"):
            analyzer.push(frame)
        live = analyzer.live
        assert live is not None
        assert live.seeing_fwhm_arcsec is None
        assert live.quality["seeing_fwhm_arcsec"] == "the readout mode is not in the profile"
        assert live.readout_mode == "odd-mode"

    def test_many_lost_frames_set_degraded_and_a_saturated_star_sets_saturated(
        self, profile: Profile
    ) -> None:
        lost = dict.fromkeys(range(5, 560, 6), 1)  # about one frame in seven is lost
        analyzer = analyzer_of(profile)
        for frame in make_frames(500, lost=lost):
            analyzer.push(frame)
        assert analyzer.live is not None
        assert "degraded" in analyzer.live.flags
        assert analyzer.live.valid_fraction < 0.9
        bright = analyzer_of(profile)
        for frame in make_frames(500, flux=30 * POLARIS_ELECTRONS_2MS):
            bright.push(frame)
        assert bright.live is not None
        assert "saturated" in bright.live.flags

    def test_the_zenith_angle_of_the_context_converts_the_value_like_a_window(
        self, profile: Profile, frames_12s: list[Frame]
    ) -> None:
        with_zenith = analyzer_of(profile)
        line_of_sight = analyzer_of(profile)
        line_of_sight.set_context(FastContext())  # no zenith angle
        for frame in frames_12s[:500]:
            with_zenith.push(frame)
            line_of_sight.push(frame)
        assert with_zenith.live is not None
        assert line_of_sight.live is not None
        assert with_zenith.live.r0_cm is not None
        assert line_of_sight.live.r0_cm is not None
        ratio = with_zenith.live.r0_cm / line_of_sight.live.r0_cm
        assert ratio == pytest.approx(1.0 / np.cos(np.radians(30.0)) ** 0.6, rel=1e-6)
        assert "seeing_fwhm_arcsec" in line_of_sight.live.quality  # a note: no zenith angle


# --- The configuration -----------------------------------------------------------------------


def test_the_live_settings_have_these_defaults() -> None:
    config = FastPathConfig()
    assert (config.live_enabled, config.live_span_s) == (True, 10.0)
    assert (config.live_every_s, config.live_min_span_s) == (2.0, 4.0)


@pytest.mark.parametrize(
    "options",
    [
        {"live_span_s": 0.0},
        {"live_every_s": -1.0},
        {"live_min_span_s": 0.0},
        {"live_span_s": 3.0},  # below the default minimum span of 4 s
    ],
)
def test_a_live_setting_that_cannot_work_is_refused(options: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="live_"):
        FastPathConfig(**options)


# --- The cost --------------------------------------------------------------------------------


def best_of(rounds: int, work: Callable[[], object]) -> float:
    best = float("inf")
    for _ in range(rounds):
        started = time.perf_counter()
        work()
        best = min(best, time.perf_counter() - started)
    return best


def test_adding_a_frame_costs_a_few_microseconds() -> None:
    estimator = LiveEstimator(FastPathConfig())
    rows = [row(index) for index in range(3000)]

    def work() -> None:
        for fields in rows:
            estimator.add(**fields)

    assert best_of(5, work) / len(rows) * 1e6 < 25.0  # about 3 microseconds; the bound is loose


def test_an_estimate_over_ten_seconds_takes_milliseconds() -> None:
    estimator = LiveEstimator(FastPathConfig())
    rng = np.random.default_rng(5)
    motion = rng.normal(0.0, 0.3, (900, 2))
    for index in range(900):
        estimator.add(**row(index, x=100.0 + motion[index, 0], y=200.0 + motion[index, 1]))
    stream = a_stream()

    def work() -> None:
        estimator.estimate(stream, frozenset(), 30.0)

    work()  # the first call fills the caches
    assert best_of(5, work) < 0.05  # about 2 ms; the bound is loose
