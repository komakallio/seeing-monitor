"""The plan of `seeingmon dev --driver asi` on real processes, with the fake SDK and no camera.

`acquire` runs the `asi` driver on `FakeAsiSdk` (the option `fake_sdk` of the factory), `core` runs
on the system clock with the full-sensor profile, and the fast stream takes at most the 2 ms of
`[scheduler.fast] exposure_us`. The tests check that the processes come up with that plan, that
the frames of the camera reach the store, and that the clock runs in real time. They never touch a
real camera, and the vendor library is not needed.

The module carries the `slow` marker, because the processes need a minute, and the clock runs in
real time. Run it with `--slow`.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("scipy", reason="the fast path needs the fast extra")

from seeingmon.clock import NS_PER_S
from seeingmon.services.web.contract import decode_status

from .system import System

pytestmark = pytest.mark.slow

FAKE_SDK_TEMPERATURE_C = 18.3  # the default of `FakeAsiSdk`; the simulator reports 19


@pytest.fixture(scope="module")
def system(tmp_path_factory: pytest.TempPathFactory) -> Iterator[System]:
    built = System(
        tmp_path_factory.mktemp("real"),
        with_web=False,
        acquire_driver="asi",
        log_level="info",  # the log of acquire says what priority its capture thread got
        acquire_overrides={"services": {"acquire": {"driver_options": {"fake_sdk": True}}}},
    )
    built.start_all()
    yield built
    built.stop_all()


class TestTheRealCameraSetup:
    def test_acquire_raises_the_priority_of_its_capture_thread_and_says_so(
        self, system: System
    ) -> None:
        log = system.children["acquire"].spec.log.read_text(encoding="utf-8", errors="replace")
        assert "capture thread priority: " in log
        assert "capture thread priority: disabled" not in log  # the launcher turned it on
        if sys.platform == "win32":
            assert "capture thread priority: highest thread priority" in log
            assert "timer resolution: 1 ms resolution" in log

    def test_core_opens_the_asi_driver_and_the_clock_is_real(self, system: System) -> None:
        assert system.plan.real
        system.wait_for(lambda: bool(system.records("run")), "the run record")
        (run,) = system.records("run")
        assert run.versions["camera_driver"] == "asi"
        system.wait_for(
            lambda: any(h.sensor_temperature_c is not None for h in system.records("health")),
            "a health record with the temperature of the camera",
        )
        temperatures = [
            h.sensor_temperature_c
            for h in system.records("health")
            if h.sensor_temperature_c is not None
        ]
        assert all(t == pytest.approx(FAKE_SDK_TEMPERATURE_C) for t in temperatures)
        newest = max(event.t_utc_ns for event in system.events())
        # The records carry the time of the machine, and not a scaled clock that started elsewhere.
        assert abs(newest - system.plan.origin_real_ns) < 300 * NS_PER_S

    def test_the_fast_stream_takes_the_exposure_of_the_profile(self, system: System) -> None:
        """The fake SDK shows no star, so the stream searches, and no seeing window closes.

        The search bursts read the fast readout mode at the 2 ms of `[scheduler.fast] exposure_us`:
        the frames of the fake SDK stay below the target background, so the adaptive exposure keeps
        the longest. The Sun gates nothing, so the bursts run at any hour, above the search limit as
        probes.
        """

        def a_burst_of_2_ms_frames() -> bool:
            stream = decode_status(system.core_call("status")).scheduler.stream
            return stream is not None and stream.purpose == "search" and stream.exposure_us == 2000

        system.wait_for(a_burst_of_2_ms_frames, "a search burst of 2 ms frames", timeout_s=180)
        status = decode_status(system.core_call("status")).scheduler
        assert status.stream is not None
        assert status.stream.mode == "bin1"  # the fast mode of the real profile
        assert status.state == "auto"
        assert status.counters["frames"] > 0  # frames from the fake SDK reached the scheduler
        assert status.counters["search_frames"] > 0
        assert system.alive() == {"acquire": True, "core": True}
