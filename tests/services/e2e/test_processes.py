"""The three processes together: a run on the simulated sky, and a hard kill of each in turn.

`acquire`, `core`, and `web` run as real subprocesses with the `sim` camera and a scaled clock. The
tests kill one process the way a crash does (`Popen.kill`), start it again, and check that the
system recovers: the other processes keep their data, the killed one finds its place again, the
fault leaves counts and events, and no stretch of the night is missing without a record that says
why. They never assert on the pace of the real clock: they wait, with a generous limit, for the
records that must come, and they read the results from the store and over HTTP.

The tests share one running system and go in order, because starting three processes takes ten
seconds and a night of windows takes more. Each test begins with `system.ensure_running()`. The
module carries the `slow` marker, so run it with `--slow`.
"""

from __future__ import annotations

import sqlite3
import urllib.error
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("scipy", reason="the fast path needs the fast extra")
pytest.importorskip("fastapi", reason="web needs the web extra")

from seeingmon.clock import NS_PER_S
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.rpc import connect_rpc
from seeingmon.services.web.contract import RPC_CHANNEL, decode_status

from .system import System

pytestmark = pytest.mark.slow

WINDOW_S = 8.0
CYCLE_S = 30.0
FAULT_EVENTS = {
    "scheduler.fault",
    "scheduler.recovery_step",
    "scheduler.recovered",
    "scheduler.degraded",
    "acquire.restarted",
    "core.started",
    "core.stopped",
}


@pytest.fixture(scope="module")
def system(tmp_path_factory: pytest.TempPathFactory) -> Iterator[System]:
    built = System(
        tmp_path_factory.mktemp("system"),
        speed=2.0,
        window_s=WINDOW_S,
        core_overrides={
            "scheduler": {
                "fast": {"window_s": WINDOW_S, "exposure_us": 50_000},
                "survey": {"cadence_s": CYCLE_S, "long_exposure_s": 10.0},
                "cloud": {"fast_window_s": WINDOW_S, "survey_cadence_s": CYCLE_S},
                # A camera that is gone for a minute must not park the scheduler for ten.
                "faults": {"degraded_after": 100, "backoff_max_s": 8.0, "slow_retry_s": 8.0},
            }
        },
    )
    built.start_all()
    yield built
    built.stop_all()


def ensure_running(system: System) -> None:
    for name in system.children:
        if not system.children[name].running:
            system.restart(name)


def windows(system: System) -> list[Any]:
    return system.records("seeing_window")


def new_windows_after(system: System, t_utc_ns: int, count: int) -> bool:
    return (
        len([w for w in windows(system) if w.t_utc_ns > t_utc_ns and w.r0_cm is not None]) >= count
    )


def now_ns(system: System) -> int:
    """The simulated time now, from the newest record of the store."""
    times = [
        r.t_utc_ns for kind in ("health", "event", "seeing_window") for r in system.records(kind)
    ]
    return int(max(times))


def core_status(system: System) -> Any:
    key = ConnectionKey.from_text(system.plan.key)
    client, _ = connect_rpc(
        Endpoint.parse(system.plan.core_endpoint), key, {"role": "cli"}, channel=RPC_CHANNEL
    )
    try:
        return decode_status(client.call("status"))
    finally:
        client.close()


def assert_no_silent_gaps(system: System, since_ns: int, *, max_gap_s: float = 90.0) -> None:
    """Every long pause between two windows has a fault, restart, or recovery event in it."""
    found = sorted((w for w in windows(system) if w.t_utc_ns >= since_ns), key=lambda w: w.t_utc_ns)
    events = [e for e in system.events() if e.kind in FAULT_EVENTS]
    for before, after in pairwise(found):
        end_ns = before.t_utc_ns + round(before.duration_s * NS_PER_S)
        gap_s = (after.t_utc_ns - end_ns) / NS_PER_S
        if gap_s <= max_gap_s:
            continue
        low, high = end_ns - 10 * NS_PER_S, after.t_utc_ns + 10 * NS_PER_S
        assert any(low <= e.t_utc_ns <= high for e in events), (
            f"a gap of {gap_s:.0f} s before {after.t_utc_ns} has no event in it"
        )


class TestTheSystem:
    def test_the_three_processes_run_on_the_simulated_sky_and_web_serves_it(
        self, system: System
    ) -> None:
        ensure_running(system)
        system.wait_for(
            lambda: (
                len([w for w in windows(system) if w.r0_cm is not None]) >= 3
                and len(system.records("sky_quality")) >= 1
            ),
            "three windows with r0 and a sky quality record",
        )
        values = [w.r0_cm for w in windows(system) if w.r0_cm is not None]
        assert all(5.0 < value < 15.0 for value in values)  # the truth is 10 cm
        (run,) = system.records("run")
        assert run.versions["camera_driver"] == "sim"
        status = system.get("status")
        assert status["core"]["reachable"] is True
        assert status["station_id"] == "dev"
        latest = system.get("seeing/latest")
        assert latest["t_utc_ns"] > 0
        health = system.get("health")
        assert health["status"] in ("healthy", "degraded")
        assert health["components"]["acquire"] == "ok"

    def test_the_survey_frames_reach_the_images_of_the_web_api(self, system: System) -> None:
        """`core` writes a preview of each long frame, and `web` lists it and serves it."""
        ensure_running(system)

        def kept_frame_listed() -> bool:
            items = (system.get_or_none("images") or {}).get("items", [])
            return any(item["has_fits"] for item in items)

        # The preview comes first and the FITS file a moment later, so wait for the second.
        system.wait_for(kept_frame_listed, "an image with a FITS frame in the web API")
        listing = system.get("images")
        items = listing["items"]
        assert [i["t_utc_ns"] for i in items] == sorted(
            (i["t_utc_ns"] for i in items), reverse=True
        )
        assert all(i["kind"] in {"survey", "event"} for i in items)
        status, headers, body = system.get_bytes("images/latest")
        assert status == 200
        assert headers["content-type"] == "image/jpeg"
        assert body.startswith(b"\xff\xd8\xff")
        newest = system.get(f"images/{items[0]['id']}?format=json")
        assert newest == items[0]
        # The first long frame of a run is kept as a FITS file, and it is the oldest image here.
        kept = [i for i in items if i["has_fits"]]
        assert kept
        status, headers, body = system.get_bytes(f"images/{kept[-1]['id']}?format=fits")
        assert status == 200
        assert headers["content-type"] == "application/fits"
        assert body.startswith(b"SIMPLE  =")
        assert len(body) == kept[-1]["fits_bytes"]
        # The records point at files that the process of `core` wrote.
        refs = [r.image_ref for r in system.records("survey_frame") if r.image_ref]
        assert refs
        data = system.plan.directory / "data"
        assert all((data / ref).is_file() for ref in refs)
        assert all(ref.startswith(("survey/", "previews/")) for ref in refs)


class TestKillingAcquire:
    def test_a_killed_acquire_comes_back_with_the_fault_counted_and_no_silent_gap(
        self, system: System
    ) -> None:
        ensure_running(system)
        before_ns = now_ns(system)
        faults_before = core_status(system).scheduler.counters["faults"]

        system.kill("acquire")
        system.wait_for(
            lambda: any(e.t_utc_ns > before_ns for e in system.events("scheduler.fault")),
            "the fault event of the scheduler",
            timeout_s=120.0,
        )
        system.restart("acquire")  # the supervisor of the unit does this in a real system
        restarted_ns = now_ns(system)
        system.wait_for(
            lambda: any(e.t_utc_ns > before_ns for e in system.events("acquire.restarted")),
            "the event that says acquire restarted",
        )
        system.wait_for(
            lambda: new_windows_after(system, restarted_ns, 2),
            "two windows after the restart of acquire",
        )

        status = core_status(system)
        assert status.scheduler.counters["faults"] > faults_before  # the fault is counted
        assert status.scheduler.counters["recovery_steps"] >= 1
        assert status.scheduler.state != "paused"
        recovered = [e for e in system.events("scheduler.recovered") if e.t_utc_ns > before_ns]
        assert recovered  # and so is the way back
        assert_no_silent_gaps(system, before_ns - 120 * NS_PER_S)
        health = system.get("health")
        assert health["components"]["acquire"] == "ok"


class TestKillingCore:
    def test_a_killed_core_comes_back_with_its_data_and_a_new_run(self, system: System) -> None:
        ensure_running(system)
        system.wait_for(lambda: len(windows(system)) >= 2, "windows before the kill")
        windows_before = len(windows(system))
        starts_before = len(system.events("core.started"))
        runs_before = len(system.records("run"))
        before_ns = now_ns(system)

        system.kill("core")
        assert not system.children["core"].running
        assert system.children["acquire"].running  # nothing else went down with it
        system.restart("core")
        restarted_ns = now_ns(system)
        system.wait_for(
            lambda: len(system.events("core.started")) == starts_before + 1,
            "the start event of the new core",
        )
        system.wait_for(
            lambda: new_windows_after(system, restarted_ns, 2),
            "two windows after the restart of core",
        )

        assert len(windows(system)) >= windows_before  # nothing that was stored is gone
        started = system.events("core.started")
        assert started[-1].detail["instance"] != started[-2].detail["instance"]
        assert len(system.records("run")) == runs_before + 1  # the new run says what it runs
        database = sqlite3.connect(f"file:{system.db_path.as_posix()}?mode=ro", uri=True)
        try:
            assert database.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        finally:
            database.close()
        assert_no_silent_gaps(system, before_ns - 120 * NS_PER_S)
        # web found the new core by itself.
        system.wait_for(lambda: system.get_or_none("status") is not None, "web to answer")
        system.wait_for(
            lambda: system.get("status")["core"]["reachable"] is True, "web to reach the new core"
        )


class TestKillingWeb:
    def test_a_killed_web_comes_back_while_core_goes_on(self, system: System) -> None:
        ensure_running(system)
        assert system.get("status")["core"]["reachable"] is True
        windows_before = len(windows(system))

        system.kill("web")
        with pytest.raises((urllib.error.URLError, OSError)):
            system.get("status", timeout_s=3.0)
        system.wait_for(
            lambda: len(windows(system)) >= windows_before + 2,
            "windows while web is down",
        )  # core does not need web
        system.restart("web")
        system.wait_for(
            lambda: (system.get_or_none("status") or {}).get("core", {}).get("reachable") is True,
            "the restarted web to reach core",
        )
        latest = system.get("seeing/latest")
        assert latest["t_utc_ns"] >= windows(system)[windows_before - 1].t_utc_ns
        assert system.children["acquire"].running
        assert system.children["core"].running
        assert Path(system.plan.directory).is_dir()
