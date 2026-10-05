"""The `auto` cycle: a fast window, then a survey step, on a steady cadence.

These scenarios start at night, so the scheduler enters `auto` at once. The fast stream runs one
frame in 2 seconds here (a real one runs at about 90 frames a second), and the analysis window
is 60 seconds, so one 120 second fast period holds 60 frames in two windows of 30 frames.

The first period searches: a burst of three frames at 0 s and one at 15 s find Polaris, and the
fast stream measures from 21 s to the end of that period at 120 s. Every later period measures.
"""

from __future__ import annotations

import itertools

import pytest

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.frames import StreamKind
from tests.scheduler.scenario import TEST_CONFIG, World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")  # the Sun is 40 degrees down, and the sky is dark
CYCLE_S = 180.0


def fast_starts(world: World) -> list[float]:
    """When each fast period on its slot began: the fast streams after the first period."""
    return [t for t in world.fast_starts() if t >= CYCLE_S]


def survey_calls(world: World) -> list[tuple[float, int]]:
    """The survey exposures as (time, exposure in microseconds), in order."""
    return [
        (world.seconds(call.t_utc_ns), call.config.exposure_us)
        for call in world.configures(mode="bin2", video=False)
        if call.config.roi is None
    ]


@pytest.fixture(scope="module")
def steady_world() -> World:
    world = World(start_utc_ns=NIGHT)
    world.run_until(3 * 3600)
    return world


class TestCadence:
    def test_the_scheduler_enters_auto_at_once_under_a_dark_sky(self, steady_world: World) -> None:
        assert steady_world.states_visited() == ["safe", "auto"]
        ((changed_at, _, _),) = steady_world.state_changes()
        assert changed_at < 1.0  # after the first brightness frame

    def test_fast_periods_start_exactly_one_cadence_apart(self, steady_world: World) -> None:
        """The cycle holds its cadence to the millisecond, because the clock is virtual.

        The stated tolerance for a real run is one fast frame period (about 11 ms at 90 frames a
        second) plus the granularity of the clock's sleep. The wait for the slot is a sleep to a
        deadline, so no error accumulates from one cycle to the next.
        """
        starts = fast_starts(steady_world)
        assert len(starts) >= 58
        gaps = [later - earlier for earlier, later in itertools.pairwise(starts)]
        assert all(gap == pytest.approx(CYCLE_S, abs=0.01) for gap in gaps)
        # The search period at the start held the same slot.
        periods = steady_world.period_starts()
        assert periods[0] == pytest.approx(0.0, abs=0.1)
        assert periods[1:] == starts
        assert starts[0] == pytest.approx(CYCLE_S, abs=0.1)

    def test_the_first_period_searches_and_hands_its_rest_to_the_fast_stream(
        self, steady_world: World
    ) -> None:
        bursts = steady_world.burst_starts()
        assert bursts == [pytest.approx(0.0, abs=0.1), pytest.approx(15.0, abs=0.1)]
        (visible,) = steady_world.events("polaris.visible")
        assert steady_world.seconds(visible.t_utc_ns) == pytest.approx(21.0, abs=0.1)
        assert steady_world.fast_starts()[0] == pytest.approx(21.0, abs=0.1)
        assert steady_world.events("polaris.hidden") == []
        status = steady_world.scheduler.status()
        assert status.search is not None
        assert status.search.mode == "measure"

    def test_each_cycle_runs_a_short_and_then_a_long_survey_exposure(
        self, steady_world: World
    ) -> None:
        calls = survey_calls(steady_world)
        exposures = [exposure for _, exposure in calls]
        assert exposures[:6] == [1000, 30_000_000] * 3
        assert len(calls) >= 2 * 59

    def test_the_survey_step_follows_the_fast_period_and_fits_before_the_next_slot(
        self, steady_world: World
    ) -> None:
        starts = steady_world.period_starts()
        calls = survey_calls(steady_world)
        for index, start in enumerate(starts[:50]):
            short_at, _ = calls[2 * index]
            long_at, _ = calls[2 * index + 1]
            assert short_at == pytest.approx(start + 120.0, abs=2.0)  # the period is 120 seconds
            assert long_at == pytest.approx(short_at, abs=1.0)  # the short frame takes milliseconds
            assert long_at + 30.2 < start + CYCLE_S  # the long frame ends before the next slot

    def test_the_survey_steps_keep_the_cadence_to_one_fast_frame_period(
        self, steady_world: World
    ) -> None:
        """The step follows the period, which ends on a frame, so it can slip by one frame."""
        shorts = [t for t, exposure in survey_calls(steady_world) if exposure == 1000]
        gaps = [later - earlier for earlier, later in itertools.pairwise(shorts)]
        assert all(gap == pytest.approx(CYCLE_S, abs=2.0) for gap in gaps)

    def test_each_fast_period_makes_two_full_windows(self, steady_world: World) -> None:
        # The fast stream of the first period runs 99 s: a full window and a shorter one.
        first = [w for w in steady_world.windows() if steady_world.seconds(w.t_utc_ns) < CYCLE_S]
        assert [w.n_frames for w in first] == [30, 20]
        windows = [w for w in steady_world.windows() if steady_world.seconds(w.t_utc_ns) > CYCLE_S]
        assert len(windows) >= 2 * 58
        for window in windows[:100]:
            assert window.n_frames == 30
            assert window.duration_s == pytest.approx(60.0)
            assert window.flags == []
            assert window.n_dropped == 0
            assert window.valid_fraction == 1.0
        starts = [w.t_utc_ns for w in windows[:100]]
        gaps = [(later - earlier) / NS_PER_S for earlier, later in itertools.pairwise(starts)]
        # Two windows of 60 seconds in each cycle of 180 seconds.
        assert gaps[:4] == pytest.approx([60.0, 120.0, 60.0, 120.0], abs=3.0)

    def test_every_fast_period_gets_a_new_stream_so_no_window_spans_two(
        self, steady_world: World
    ) -> None:
        stream_ids = [w.stream_id for w in steady_world.windows()]
        first_windows, second_windows = stream_ids[0::2][:50], stream_ids[1::2][:50]
        assert first_windows == second_windows  # the two windows of one period share a stream
        assert len(set(stream_ids)) >= 59

    def test_the_bursts_wrote_no_window_and_reached_no_analysis_window(
        self, steady_world: World
    ) -> None:
        counters = steady_world.scheduler.status().counters
        assert (counters.search_bursts, counters.search_frames) == (2, 6)
        assert steady_world.fast.frames_measured == 6
        in_windows = sum(w.n_frames for w in steady_world.windows())
        assert in_windows == steady_world.fast.frames_pushed

    def test_the_roi_is_centered_on_polaris_by_the_profile_helpers(
        self, steady_world: World
    ) -> None:
        for call in steady_world.configures(mode="bin1", video=True)[:20]:
            roi = call.config.roi
            assert roi is not None
            assert (roi.width, roi.height) == (32, 32)  # 1 arcmin at 1.91 arcsec per pixel
            x, y = steady_world.star_position(call.t_utc_ns)
            assert roi.contains(x, y)
            assert roi.distance_to_edge(x, y) >= 14  # centered to within a pixel or two

    def test_the_fast_stream_uses_the_fast_settings_of_the_profile(
        self, steady_world: World
    ) -> None:
        for call in steady_world.configures(mode="bin1", video=True)[:5]:
            config = call.config
            assert (config.exposure_us, config.gain, config.kind) == (
                2_000_000,
                0,
                StreamKind.VIDEO,
            )
            assert config.pixel_format.value == 16

    def test_per_frame_metrics_reach_the_writer_for_every_frame(self, steady_world: World) -> None:
        rows = sum(len(batch) for _, batch in steady_world.writer.metrics)
        pushed = steady_world.fast.frames_pushed
        # The open window of the last period has not drained its last rows yet.
        assert pushed - 60 <= rows <= pushed
        metric_streams = {stream for stream, _ in steady_world.writer.metrics}
        assert metric_streams >= {w.stream_id for w in steady_world.windows()}

    def test_the_status_counts_what_happened(self, steady_world: World) -> None:
        counters = steady_world.scheduler.status().counters
        assert counters.fast_periods >= 59
        assert counters.survey_steps >= 59
        assert counters.survey_frames == 2 * counters.survey_steps
        assert counters.windows == len(steady_world.windows())
        assert counters.cadence_overruns == 0
        assert counters.faults == counters.solves_requested == 0
        assert counters.dropped == 0
        assert counters.transitions == 1


class TestRecordsAreWellFormed:
    def test_no_two_events_share_a_key(self, steady_world: World) -> None:
        keys = [event.record_key for event in steady_world.events()]
        assert len(keys) == len(set(keys))

    def test_every_record_survives_a_round_trip_through_its_row(self, steady_world: World) -> None:
        for record in steady_world.writer.records[:300]:
            assert type(record).from_row(record.to_row(), strict=True) == record

    def test_every_record_names_the_station_and_the_profile(self, steady_world: World) -> None:
        assert {r.station_id for r in steady_world.writer.records} == {"test"}
        assert {r.profile_id for r in steady_world.writer.records} == {"asi294mm-gs250"}

    def test_the_records_arrive_in_the_order_of_their_time(self, steady_world: World) -> None:
        """Windows are written as they close, so their start times never go backward."""
        starts = [w.t_utc_ns for w in steady_world.windows()]
        assert starts == sorted(starts)


class TestHighSpeed:
    """The fast stream runs in the high-speed mode of the camera when `[scheduler.fast]` says so."""

    @staticmethod
    def world(*, high_speed: bool) -> World:
        fast = TEST_CONFIG.fast.model_copy(update={"high_speed": high_speed})
        world = World(start_utc_ns=NIGHT, config=TEST_CONFIG.model_copy(update={"fast": fast}))
        world.run_until(1200)
        return world

    def test_the_fast_stream_asks_for_it(self) -> None:
        calls = self.world(high_speed=True).configures(mode="bin1", video=True)
        assert len(calls) >= 3
        assert all(call.config.high_speed for call in calls)

    def test_the_survey_exposures_stay_at_normal_speed(self) -> None:
        calls = self.world(high_speed=True).configures(mode="bin2", video=False)
        assert len(calls) >= 3
        assert not any(call.config.high_speed for call in calls)

    def test_by_default_no_stream_asks_for_it(self, steady_world: World) -> None:
        assert not any(call.config.high_speed for call in steady_world.configures())
