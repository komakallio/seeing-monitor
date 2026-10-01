from __future__ import annotations

import threading
import time

import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.clock import (
    DEFAULT_START_UTC_NS,
    NS_PER_S,
    Clock,
    ClockStatus,
    ScaledClock,
    SystemClock,
    VirtualClock,
    iso_to_utc_ns,
    sleep_until_utc_ns,
    utc_ns_to_datetime,
    utc_ns_to_iso,
)

YEAR_2100_NS = 4_102_444_800 * NS_PER_S


def test_all_clocks_satisfy_the_protocol() -> None:
    clocks: list[Clock] = [
        SystemClock(),
        VirtualClock(),
        ScaledClock(start_utc_ns=0, origin_real_ns=time.time_ns(), speed=10),
    ]
    assert all(isinstance(clock, Clock) for clock in clocks)


class TestVirtualClock:
    def test_starts_where_it_is_told(self) -> None:
        clock = VirtualClock(start_utc_ns=123 * NS_PER_S, monotonic_start_ns=5)
        assert clock.utc_ns() == 123 * NS_PER_S
        assert clock.monotonic_ns() == 5

    def test_default_start_is_the_start_of_2026(self) -> None:
        assert utc_ns_to_iso(VirtualClock().utc_ns()) == "2026-01-01T00:00:00.000000Z"

    def test_sleep_advances_both_clocks_without_waiting(self) -> None:
        clock = VirtualClock()
        started = time.perf_counter()
        clock.sleep(3600.0)
        assert time.perf_counter() - started < 1.0
        assert clock.utc_ns() == DEFAULT_START_UTC_NS + 3600 * NS_PER_S
        assert clock.monotonic_ns() == 3600 * NS_PER_S

    def test_sleep_of_zero_or_less_does_nothing(self) -> None:
        clock = VirtualClock()
        clock.sleep(0)
        clock.sleep(-5)
        assert clock.utc_ns() == DEFAULT_START_UTC_NS

    def test_cannot_advance_backward(self) -> None:
        with pytest.raises(ValueError, match="backward"):
            VirtualClock().advance(-1)

    def test_advance_to_utc(self) -> None:
        clock = VirtualClock()
        clock.advance_to_utc_ns(DEFAULT_START_UTC_NS + 10)
        assert clock.utc_ns() == DEFAULT_START_UTC_NS + 10
        with pytest.raises(ValueError, match="past"):
            clock.advance_to_utc_ns(DEFAULT_START_UTC_NS)

    def test_a_wall_clock_step_leaves_the_monotonic_clock_alone(self) -> None:
        clock = VirtualClock()
        clock.advance(10)
        clock.step_utc_ns(-4 * NS_PER_S)
        assert clock.monotonic_ns() == 10 * NS_PER_S
        assert clock.utc_ns() == DEFAULT_START_UTC_NS + 6 * NS_PER_S
        clock.advance(1)
        assert clock.utc_ns() == DEFAULT_START_UTC_NS + 7 * NS_PER_S

    def test_status_can_be_set(self) -> None:
        clock = VirtualClock()
        assert clock.status().synchronized is True
        unsynced = ClockStatus(synchronized=False, error_bound_ns=None, source="virtual")
        clock.set_status(unsynced)
        assert clock.status() == unsynced

    def test_concurrent_advances_do_not_lose_time(self) -> None:
        clock = VirtualClock()

        def worker() -> None:
            for _ in range(1000):
                clock.advance_ns(1)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert clock.monotonic_ns() == 4000

    def test_a_simulated_night_runs_in_seconds(self) -> None:
        clock = VirtualClock()
        night_end = clock.utc_ns() + 12 * 3600 * NS_PER_S
        ticks = 0
        started = time.perf_counter()
        while clock.utc_ns() < night_end:
            clock.sleep(1.0)
            ticks += 1
        assert ticks == 12 * 3600
        assert time.perf_counter() - started < 5.0


def test_sleep_until_utc_ns_sleeps_the_difference() -> None:
    clock = VirtualClock()
    sleep_until_utc_ns(clock, clock.utc_ns() + 5 * NS_PER_S)
    assert clock.monotonic_ns() == 5 * NS_PER_S
    sleep_until_utc_ns(clock, clock.utc_ns() - NS_PER_S)  # in the past: no effect
    assert clock.monotonic_ns() == 5 * NS_PER_S


def test_system_clock_tracks_the_operating_system() -> None:
    clock = SystemClock()
    before = time.time_ns()
    now = clock.utc_ns()
    assert before <= now <= time.time_ns()
    first = clock.monotonic_ns()
    assert clock.monotonic_ns() >= first
    assert clock.status() == ClockStatus(synchronized=None, error_bound_ns=None, source="system")


def test_system_clock_reports_a_probed_status() -> None:
    probed = ClockStatus(synchronized=True, error_bound_ns=2_000_000, source="chrony")
    assert SystemClock(status_probe=lambda: probed).status() == probed


class TestScaledClock:
    def test_rejects_a_non_positive_speed(self) -> None:
        with pytest.raises(ValueError, match="speed"):
            ScaledClock(start_utc_ns=0, origin_real_ns=0, speed=0)

    def test_runs_faster_than_real_time(self) -> None:
        clock = ScaledClock(
            start_utc_ns=DEFAULT_START_UTC_NS, origin_real_ns=time.time_ns(), speed=100
        )
        start_utc, start_mono = clock.utc_ns(), clock.monotonic_ns()
        clock.sleep(2.0)  # 20 ms of real time
        assert clock.utc_ns() - start_utc >= 1.5 * NS_PER_S
        assert clock.monotonic_ns() - start_mono >= 1.5 * NS_PER_S

    def test_clocks_with_the_same_parameters_agree(self) -> None:
        origin = time.time_ns()
        a = ScaledClock(start_utc_ns=DEFAULT_START_UTC_NS, origin_real_ns=origin, speed=1)
        b = ScaledClock(start_utc_ns=DEFAULT_START_UTC_NS, origin_real_ns=origin, speed=1)
        assert abs(a.utc_ns() - b.utc_ns()) < NS_PER_S // 10


class TestTimeFormats:
    def test_epoch_and_known_values(self) -> None:
        assert utc_ns_to_iso(0) == "1970-01-01T00:00:00.000000Z"
        assert utc_ns_to_iso(0, digits=0) == "1970-01-01T00:00:00Z"
        assert utc_ns_to_iso(1_000_000_123, digits=9) == "1970-01-01T00:00:01.000000123Z"
        assert iso_to_utc_ns("2026-01-01T00:00:00Z") == DEFAULT_START_UTC_NS

    def test_times_before_the_epoch_format_correctly(self) -> None:
        assert utc_ns_to_iso(-1, digits=9) == "1969-12-31T23:59:59.999999999Z"
        assert iso_to_utc_ns("1969-12-31T23:59:59.999999999Z") == -1

    @given(st.integers(min_value=0, max_value=YEAR_2100_NS))
    def test_nanosecond_round_trip(self, t_ns: int) -> None:
        assert iso_to_utc_ns(utc_ns_to_iso(t_ns, digits=9)) == t_ns

    @given(st.integers(min_value=0, max_value=YEAR_2100_NS))
    def test_microsecond_format_truncates(self, t_ns: int) -> None:
        assert iso_to_utc_ns(utc_ns_to_iso(t_ns)) == t_ns - t_ns % 1000

    def test_accepts_an_explicit_utc_offset_and_short_fractions(self) -> None:
        assert iso_to_utc_ns("2026-01-01T00:00:00.5+00:00") == DEFAULT_START_UTC_NS + NS_PER_S // 2

    @pytest.mark.parametrize(
        "text",
        [
            "2026-01-01T00:00:00",
            "2026-01-01T00:00:00+02:00",
            "2026-01-01 00:00:00Z",
            "2026-01-01T00:00:00.Z",
            "2026-01-01T00:00:00.1234567890Z",
            "not a time",
        ],
    )
    def test_rejects_other_forms(self, text: str) -> None:
        with pytest.raises(ValueError, match=r"UTC|fraction|match|unconverted|time"):
            iso_to_utc_ns(text)

    def test_rejects_bad_digit_counts(self) -> None:
        with pytest.raises(ValueError, match="digits"):
            utc_ns_to_iso(0, digits=10)

    def test_datetime_conversion(self) -> None:
        moment = utc_ns_to_datetime(DEFAULT_START_UTC_NS + 1_500_000)
        assert (moment.year, moment.month, moment.day, moment.microsecond) == (2026, 1, 1, 1500)
        assert moment.utcoffset() is not None
