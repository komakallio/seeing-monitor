"""The clock probe: the rules for the state of the kernel clock, and a real read on Linux."""

from __future__ import annotations

import ctypes
import sys

import pytest

from seeingmon.clock import SystemClock
from seeingmon.services.clockprobe import (
    STA_UNSYNC,
    TIME_ERROR,
    AdjtimexProbe,
    Timex,
    make_probe,
    status_from_kernel,
)
from seeingmon.services.config import ClockSettings

NS_PER_US = 1000


class TestRules:
    def test_a_synchronized_kernel_clock_gives_its_error_bound(self) -> None:
        status = status_from_kernel(state=0, status=0x2001, maxerror_us=1250)
        assert status.synchronized is True
        assert status.error_bound_ns == 1250 * NS_PER_US
        assert status.source == "adjtimex"

    def test_the_unsync_bit_means_not_synchronized(self) -> None:
        assert status_from_kernel(0, STA_UNSYNC | 0x1, 16_000_000).synchronized is False

    def test_the_error_state_means_not_synchronized(self) -> None:
        assert status_from_kernel(TIME_ERROR, 0, 0).synchronized is False

    def test_a_leap_second_in_progress_is_still_synchronized(self) -> None:
        assert status_from_kernel(state=1, status=0, maxerror_us=10).synchronized is True

    def test_a_negative_bound_cannot_happen_and_clamps(self) -> None:
        assert status_from_kernel(0, 0, -5).error_bound_ns == 0


class TestTheProbe:
    def test_it_reports_what_the_call_returns(self) -> None:
        probe = AdjtimexProbe(call=lambda: (0, 0, 200))
        status = probe()
        assert (status.synchronized, status.error_bound_ns) == (True, 200_000)

    def test_a_failed_call_is_unknown_and_logged_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        probe = AdjtimexProbe(call=lambda: None)
        first, second = probe(), probe()
        assert (first.synchronized, first.error_bound_ns) == (None, None)
        assert second == first
        assert caplog.text.count("cannot read the kernel clock state") == 1

    def test_a_missing_libc_is_unknown(self) -> None:
        def broken() -> tuple[int, int, int] | None:
            raise OSError("no libc")

        assert AdjtimexProbe(call=broken)().synchronized is None

    def test_the_system_clock_uses_the_probe(self) -> None:
        clock = SystemClock(status_probe=AdjtimexProbe(call=lambda: (TIME_ERROR, 0, 0)))
        assert clock.status().synchronized is False


@pytest.mark.skipif(sys.platform != "linux", reason="adjtimex is a Linux call")
class TestOnLinux:
    def test_the_structure_has_the_size_of_the_c_structure(self) -> None:
        if ctypes.sizeof(ctypes.c_long) == 8:
            assert ctypes.sizeof(Timex) == 208
        else:
            assert ctypes.sizeof(Timex) == 128

    def test_the_real_call_answers_with_a_status(self) -> None:
        status = AdjtimexProbe()()
        assert status.source == "adjtimex"
        # A runner may or may not be synchronized, but the call itself must work.
        assert status.synchronized is not None
        assert status.error_bound_ns is not None
        assert status.error_bound_ns >= 0


class TestSettings:
    def test_none_means_no_probe(self) -> None:
        assert make_probe("none") is None

    def test_adjtimex_is_asked_for_by_name(self) -> None:
        assert isinstance(make_probe("adjtimex"), AdjtimexProbe)

    def test_auto_picks_adjtimex_on_linux_only(self) -> None:
        probe = make_probe("auto")
        assert isinstance(probe, AdjtimexProbe) == (sys.platform == "linux")

    def test_the_settings_build_a_system_clock_with_the_probe(self) -> None:
        clock = ClockSettings(probe="none").build()
        assert isinstance(clock, SystemClock)
        assert clock.status().synchronized is None  # no probe: unknown
        assert (
            ClockSettings(kind="scaled", start_utc_ns=0, origin_real_ns=0)
            .build()
            .status()
            .synchronized
        )
