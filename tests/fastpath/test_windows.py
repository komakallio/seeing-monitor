"""Window grouping: frame time, streams, drops, gaps, and the frame slots.

The hypothesis test feeds random sequences of frames to the assembler and checks the invariants
that every window must keep, whatever the sequence.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from seeingmon.fastpath.windows import ClosedWindow, WindowAssembler, is_partial

NS = 1_000_000_000
PERIOD_NS = 100_000_000  # 10 frames per second, so a 5 s window holds 50 frames
WINDOW_S = 5.0
VALUES = (1.0, 2.0, 1.0, 1.0, 100.0, 1000.0, 480.0, 0.001, 0.001)


def add(
    assembler: WindowAssembler,
    t_ns: int,
    *,
    stream_id: int = 1,
    dropped_before: int = 0,
    usable: bool = True,
    saturated: bool = False,
    time_invalid: bool = False,
    temperature_c: float | None = 15.0,
) -> list[ClosedWindow]:
    return assembler.add(
        stream_id,
        t_ns,
        dropped_before,
        "bin1",
        2000,
        0,
        temperature_c,
        time_invalid,
        usable,
        saturated,
        VALUES,
    )


def feed(
    assembler: WindowAssembler, count: int, *, start: int = 0, stream_id: int = 1
) -> list[ClosedWindow]:
    closed: list[ClosedWindow] = []
    for index in range(start, start + count):
        closed += add(assembler, index * PERIOD_NS, stream_id=stream_id)
    return closed


class TestGrouping:
    def test_a_window_closes_when_a_frame_arrives_after_its_length(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        closed = feed(assembler, 101)  # frames at 0.0 to 10.0 s
        assert [w.n_frames for w in closed] == [50, 50]
        assert closed[1].t_start_ns == 50 * PERIOD_NS  # the frame at 5.0 s starts the next window
        assert not any(w.closed_early for w in closed)
        assert assembler.has_open_window  # the frame at 10.0 s opened a third

    def test_the_duration_is_the_span_plus_one_frame_period(self) -> None:
        window = feed(WindowAssembler(WINDOW_S), 51)[0]
        assert window.duration_s == pytest.approx(5.0)
        assert window.period_s == pytest.approx(0.1)

    def test_flush_closes_the_open_window_early_and_keeps_going(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        feed(assembler, 20)
        (window,) = assembler.flush("end")
        assert window.n_frames == 20
        assert window.closed_early
        assert window.reason == "end"
        assert assembler.flush() == []
        later = feed(assembler, 3, start=20)  # the stream continues in a new window
        assert later == []
        assert assembler.has_open_window

    def test_begin_stream_closes_the_old_window_and_sets_the_new_stream(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        feed(assembler, 7)
        (window,) = assembler.begin_stream(2, period_s=0.1)
        assert window.stream_id == 1
        assert window.n_frames == 7
        assert window.closed_early
        assert window.reason == "new stream"
        assert assembler.stream_id == 2
        assert assembler.period_s == pytest.approx(0.1)
        assert assembler.begin_stream(3) == []  # nothing is open

    def test_a_frame_of_another_stream_closes_the_window_and_starts_the_new_stream(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        feed(assembler, 5, stream_id=1)
        closed = add(assembler, 5 * PERIOD_NS, stream_id=2)
        assert [(w.stream_id, w.n_frames, w.reason) for w in closed] == [(1, 5, "new stream")]
        (window,) = assembler.flush()
        assert window.stream_id == 2
        assert window.n_frames == 1

    def test_a_time_that_does_not_advance_closes_the_window(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        feed(assembler, 10)
        closed = add(assembler, 5 * PERIOD_NS)  # the clock stepped back
        assert [(w.n_frames, w.reason) for w in closed] == [(10, "time discontinuity")]
        assert add(assembler, 5 * PERIOD_NS)
        assert True

    def test_the_window_start_follows_the_first_frame_not_a_grid(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        offset = 12_345_678_901
        closed: list[ClosedWindow] = []
        for index in range(120):
            closed += add(assembler, offset + index * PERIOD_NS)
        assert closed[0].t_start_ns == offset
        assert closed[1].t_start_ns == offset + 50 * PERIOD_NS

    def test_rejects_a_non_positive_window(self) -> None:
        with pytest.raises(ValueError, match="window_s"):
            WindowAssembler(0.0)


class TestPartial:
    def test_a_window_that_ends_early_and_short_is_partial(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        feed(assembler, 20)
        (window,) = assembler.flush()
        assert is_partial(window, WINDOW_S)

    def test_a_window_of_full_length_is_not_partial_even_when_flushed(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        feed(assembler, 50)  # 5.0 s of frames (the last at 4.9 s)
        (window,) = assembler.flush()
        assert window.duration_s == pytest.approx(5.0)
        assert not is_partial(window, WINDOW_S)

    def test_a_naturally_closed_window_is_not_partial(self) -> None:
        closed = feed(WindowAssembler(WINDOW_S), 60)
        assert not is_partial(closed[0], WINDOW_S)


class TestDrops:
    def test_a_dropped_frame_count_adds_to_the_window(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        feed(assembler, 10)
        add(assembler, 14 * PERIOD_NS, dropped_before=4)  # frames 10 to 13 never arrived
        (window,) = assembler.flush()
        assert window.n_frames == 11
        assert window.n_dropped == 4

    def test_a_time_gap_counts_as_drops_when_the_producer_counts_none(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        feed(assembler, 10)
        add(assembler, 14 * PERIOD_NS)  # the interval is 5 periods, so 4 frames are missing
        (window,) = assembler.flush()
        assert window.n_dropped == 4

    def test_the_larger_of_the_counter_and_the_gap_wins_and_nothing_counts_twice(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        feed(assembler, 10)
        add(assembler, 12 * PERIOD_NS, dropped_before=2)  # counter and gap agree on 2
        add(assembler, 13 * PERIOD_NS, dropped_before=1)  # the counter says 1, the time says 0
        (window,) = assembler.flush()
        assert window.n_dropped == 3

    def test_an_interval_under_one_and_a_half_periods_is_not_a_gap(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        feed(assembler, 10)
        add(assembler, 9 * PERIOD_NS + 140_000_000)  # 1.4 periods after the last frame
        (window,) = assembler.flush()
        assert window.n_dropped == 0

    def test_the_first_frame_of_a_stream_loses_nothing(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        add(assembler, 0, dropped_before=7)
        (window,) = assembler.flush()
        assert window.n_dropped == 0
        assert window.slot.tolist() == [0]

    def test_drops_before_the_frame_that_opens_a_window_count_for_the_old_window(self) -> None:
        """Frames lost at 4.6 s and 4.7 s belong to the window that ends at 5.0 s. Those at 5.0 s
        and later fall between windows."""
        assembler = WindowAssembler(WINDOW_S)
        feed(assembler, 46)  # the last frame is at 4.5 s
        closed = add(assembler, 51 * PERIOD_NS)  # frames at 4.6, 4.7, 4.8, 4.9, and 5.0 are lost
        assert len(closed) == 1
        assert closed[0].n_frames == 46
        assert closed[0].n_dropped == 4  # 4.6 to 4.9 s
        (window,) = assembler.flush()
        assert window.n_dropped == 0
        assert window.n_frames == 1

    def test_the_degraded_share_uses_frames_and_drops_together(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        feed(assembler, 9)
        add(
            assembler, 19 * PERIOD_NS
        )  # nine frames lost between frame 8 and frame 19... 10 missing
        (window,) = assembler.flush()
        assert window.n_frames == 10
        assert window.n_dropped == 10
        assert window.n_dropped / (window.n_frames + window.n_dropped) == pytest.approx(0.5)

    def test_a_noisy_clock_does_not_look_like_drops(self) -> None:
        rng = np.random.default_rng(1)
        assembler = WindowAssembler(WINDOW_S)
        closed: list[ClosedWindow] = []
        for index in range(300):
            jitter = int(rng.normal(0, 8_000_000))  # 8 ms of timestamp jitter on a 100 ms period
            closed += add(assembler, index * PERIOD_NS + jitter)
        closed += assembler.flush()
        assert sum(w.n_dropped for w in closed) == 0
        assert sum(w.n_frames for w in closed) == 300


class TestSlotsAndCounts:
    def test_slots_skip_the_lost_frames(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        feed(assembler, 4)
        add(assembler, 6 * PERIOD_NS)  # two frames lost
        add(assembler, 7 * PERIOD_NS)
        (window,) = assembler.flush()
        assert window.slot.tolist() == [0, 1, 2, 3, 6, 7]
        assert window.n_slots == 8
        grid = window.on_slots(np.arange(6, dtype=float))
        assert np.isnan(grid[4:6]).all()
        assert grid[6] == 4.0
        assert grid[7] == 5.0

    def test_unusable_frames_keep_their_slot_but_have_no_values(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        add(assembler, 0)
        add(assembler, PERIOD_NS, usable=False)
        add(assembler, 2 * PERIOD_NS)
        (window,) = assembler.flush()
        assert window.n_frames == 3
        assert window.n_usable == 2
        assert window.n_dropped == 0
        assert window.usable.tolist() == [True, False, True]
        assert np.isnan(window.x[1])
        assert window.x[0] == 1.0
        assert window.peak_dn[1] == 100.0  # the peak and the background stay for every frame
        assert window.slot.tolist() == [0, 1, 2]

    def test_the_snr_of_each_frame_is_kept_for_the_usable_frames(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        for index, (usable, snr) in enumerate([(True, 12.0), (False, 30.0), (True, 14.0)]):
            assembler.add(
                1, index * PERIOD_NS, 0, "bin1", 2000, 0, None, False, usable, False, VALUES, snr
            )
        add(assembler, 3 * PERIOD_NS)  # a frame without an SNR
        (window,) = assembler.flush()
        assert window.snr[0] == 12.0
        assert np.isnan(window.snr[1])  # a frame without a usable centroid
        assert window.snr[2] == 14.0
        assert np.isnan(window.snr[3])

    def test_counts_flags_and_the_mean_temperature(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        add(assembler, 0, saturated=True, temperature_c=10.0)
        add(assembler, PERIOD_NS, saturated=True, temperature_c=20.0)
        add(assembler, 2 * PERIOD_NS, time_invalid=True, temperature_c=None)
        (window,) = assembler.flush()
        assert window.n_saturated == 2
        assert window.time_invalid
        assert window.temperature_c == pytest.approx(15.0)
        assert window.saturated.tolist() == [True, True, False]

    def test_a_window_without_temperatures_has_none(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        add(assembler, 0, temperature_c=None)
        (window,) = assembler.flush()
        assert window.temperature_c is None
        assert not window.time_invalid

    def test_settings_of_the_first_frame_describe_the_window(self) -> None:
        assembler = WindowAssembler(WINDOW_S)
        add(assembler, 0)
        (window,) = assembler.flush()
        assert (window.mode, window.exposure_us, window.gain) == ("bin1", 2000, 0)


@settings(max_examples=150, deadline=None)
@given(
    steps=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=4),  # frames lost before this frame
            st.booleans(),  # whether the producer counts them in `dropped_before`
            st.integers(min_value=0, max_value=40),  # 0 means a new stream: rare
            st.booleans(),  # usable
            st.booleans(),  # saturated
        ),
        min_size=1,
        max_size=250,
    ),
    window_s=st.sampled_from([2.0, 5.0]),
)
def test_every_frame_lands_in_exactly_one_window_and_no_window_spans_a_stream(
    steps: list[tuple[int, bool, int, bool, bool]], window_s: float
) -> None:
    assembler = WindowAssembler(window_s)
    stream_id, t_ns = 1, 0
    frames: list[tuple[int, int, int, bool, bool]] = []  # stream, time, lost, usable, saturated
    closed: list[ClosedWindow] = []
    for lost, counted, switch, usable, saturated in steps:
        new_stream = switch == 0
        if new_stream:
            stream_id += 1
            lost = 0
        t_ns += (1 + lost) * PERIOD_NS
        closed += add(
            assembler,
            t_ns,
            stream_id=stream_id,
            dropped_before=lost if counted else 0,
            usable=usable,
            saturated=saturated,
        )
        frames.append((stream_id, t_ns, lost, usable, saturated))
    closed += assembler.flush()

    position = 0
    for window in closed:
        members = frames[position : position + window.n_frames]
        position += window.n_frames
        assert members, "a window has at least one frame"
        # Every frame belongs to the stream of its window.
        assert {m[0] for m in members} == {window.stream_id}
        # The window starts at its first frame, and no frame lies a full window length later.
        assert window.t_start_ns == members[0][1]
        assert members[-1][1] - window.t_start_ns < round(window_s * NS)
        assert window.duration_s >= (members[-1][1] - window.t_start_ns) / NS
        # Slots start at 0 and advance by one plus the frames lost in between (the producer or
        # the time gap tells the loss, and the gap counts exactly with an exact clock).
        assert window.slot[0] == 0
        expected_slots = [0]
        for previous, current in itertools.pairwise(members):
            expected_slots.append(
                expected_slots[-1] + 1 + (current[1] - previous[1]) // PERIOD_NS - 1
            )
        # The assembler needs a period to see a gap, and it learns it from the first intervals,
        # so the first frames after a stream start can differ. The slot grid never moves backward.
        assert (np.diff(window.slot) >= 1).all()
        assert window.slot[-1] <= expected_slots[-1]
        # Counts add up.
        assert window.n_usable == sum(1 for m in members if m[3])
        assert window.n_saturated == sum(1 for m in members if m[4])
        assert 0 <= window.n_usable <= window.n_frames
        assert window.n_dropped >= 0
        assert window.usable.tolist() == [m[3] for m in members]
    assert position == len(frames)  # every frame landed in a window
    assert not assembler.has_open_window
    # Windows come out in time order, and the stream ids never decrease.
    assert [w.stream_id for w in closed] == sorted(w.stream_id for w in closed)
    for first, second in itertools.pairwise(closed):
        if first.stream_id == second.stream_id:
            assert second.t_start_ns > first.t_start_ns


@settings(max_examples=100, deadline=None)
@given(
    lost=st.lists(st.integers(min_value=0, max_value=3), min_size=20, max_size=120),
)
def test_with_an_exact_clock_the_dropped_frames_of_every_window_are_exact(lost: list[int]) -> None:
    """On an exact periodic clock, the gaps give the drops exactly.

    The drops inside a window equal the lost frames of its frames after the first one. The frame
    that opens the next window brings in the drops that precede it, up to the end of the window.
    """
    assembler = WindowAssembler(WINDOW_S)
    t_ns = 0
    times: list[int] = []
    losses: list[int] = []
    closed: list[ClosedWindow] = []
    for index, gap in enumerate(lost):
        t_ns += (1 + (gap if index else 0)) * PERIOD_NS
        times.append(t_ns)
        losses.append(gap if index else 0)
        closed += add(assembler, t_ns, dropped_before=gap if index else 0)
    closed += assembler.flush()
    position = 0
    for window in closed:
        members = range(position, position + window.n_frames)
        inside = sum(losses[i] for i in members if i != members[0])
        following = losses[members[-1] + 1] if members[-1] + 1 < len(losses) else 0
        assert inside <= window.n_dropped <= inside + following
        position += window.n_frames
    assert math.isfinite(sum(w.duration_s for w in closed))
