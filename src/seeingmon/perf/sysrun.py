"""A run of the whole system on the simulated sky, measured from outside.

`seeingmon dev` starts `acquire` (with the simulator), `core`, and `web` as processes of one
machine. This module starts the same three processes from the plan of the launcher
(`seeingmon.services.dev.build_plan`), and it reads them from outside, with no hook in the code
that they run:

- **Memory.** The peak resident size of each process, which the system keeps (`VmHWM` on Linux and
  `PeakWorkingSetSize` on Windows), read at every sample and at the end. The survey worker is a
  child of `core`, and the module finds it in the process tree.
- **CPU time.** The CPU time of each process at every sample, and on Linux the CPU time of each
  thread (`seeingmon.perf.procs`). The share of a core in a phase is the CPU time that the
  processes used in the phase, divided by the length of the phase.
- **The state of the scheduler.** The `status` call of `core` says which stream runs and how many
  frames, windows, and survey frames the scheduler has handled. The samples use it to tell the
  phases apart. The call is the one that `web` makes, and it answers at once.

**The phases.** A sample of a phase covers the time between two samples that both fit the phase:

- *fast*: the fast stream runs and no survey frame waits. The first windows after the start do not
  count, because the imports and the first allocations of the analysis happen then;
- *idle*: the scheduler is paused (the `Pause` command, which a person gives with a button in
  `web`), so no frame flows. Whatever `core` still uses is the load that does not belong to the
  frames: the scheduler loop, the store, the health timer, and the answers to `web`.

The difference of the two shares, divided by the frame rate, is what a frame costs `core`: the
receive, the fast path, and the segment append.

**What the run includes.** The simulator renders the frames inside `acquire`, and its cost is far
larger than the cost of `acquire` itself, so the CPU time of `acquire` here includes the simulator.
The `ipc` case measures `acquire` with prebuilt frames, and its figure is the one that the budgets
read. The peak memory of `acquire` includes the simulator too. One client polls `web` every few
seconds, as an open page would.
"""

from __future__ import annotations

import contextlib
import http.client
import os
import socket
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from seeingmon.perf import procs
from seeingmon.perf.load import busy_percent_between, busy_ticks

ROLES = ("acquire", "core", "web", "survey_worker", "other")
THREAD_ROLES = ("acquire", "core", "web")
MIN_FAST_FPS = 30.0  # a stream below this rate is not a fast stream at its steady state
PAUSE_WAIT_S = 90.0
WEB_PATHS = ("status", "seeing/latest", "health")


@dataclass(frozen=True, slots=True)
class RunPlan:
    """What a run does and how long it takes.

    `fast_seconds` and `idle_seconds` are the seconds of each phase to collect. The run waits for
    `survey_steps` survey steps, and for the results of their frames, because a result means that
    the survey worker ran. A step takes two exposures, a short one and a long one, and the steps
    come every 3 minutes at speed 1. A fast interval counts only when the frame rate reaches
    `min_fast_fps`, so a stall does not enter the figures. `max_run_s` ends the sampling when the
    system does not get there, and a note says so.
    """

    sensor: str = "full"
    speed: float = 1.0
    fast_exposure_us: int = 2_000
    polaris_mag: float | None = None
    sample_interval_s: float = 1.0
    poll_interval_s: float = 5.0
    warmup_windows: int = 1
    fast_seconds: float = 60.0
    idle_seconds: float = 30.0
    survey_steps: int = 2
    min_fast_fps: float = MIN_FAST_FPS
    max_run_s: float = 720.0
    ready_timeout_s: float = 120.0


FULL_PLAN = RunPlan()
SMOKE_PLAN = RunPlan(
    sensor="small",
    warmup_windows=0,
    fast_seconds=3.0,
    idle_seconds=2.0,
    survey_steps=0,
    min_fast_fps=1.0,  # a slow runner still gets through, and the smoke run reads no figure's size
    max_run_s=120.0,
)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One sample: the scheduler's counters and the CPU time of each role at one moment.

    `t` is a monotonic time in seconds. `cpu_ns` holds the cumulative CPU time by role, and
    `threads` the cumulative CPU time by thread ID for the roles whose threads the system gives.
    """

    t: float
    state: str
    purpose: str | None
    stream_id: int | None
    frames: int
    windows: int
    survey_steps: int
    survey_results: int
    survey_pending: int
    cpu_ns: Mapping[str, int]
    threads: Mapping[str, Mapping[int, int]] = field(default_factory=dict)


@dataclass(slots=True)
class Phase:
    """The sum of the intervals of one phase: the time, the frames, and the CPU time by role."""

    seconds: float = 0.0
    frames: int = 0
    cpu_ns: dict[str, int] = field(default_factory=dict)
    thread_ns: dict[str, dict[int, int]] = field(default_factory=dict)

    def add(self, first: Snapshot, second: Snapshot) -> None:
        """Add the interval between two samples."""
        self.seconds += second.t - first.t
        self.frames += max(second.frames - first.frames, 0)
        for role, used in second.cpu_ns.items():
            self.cpu_ns[role] = self.cpu_ns.get(role, 0) + max(used - first.cpu_ns.get(role, 0), 0)
        for role, threads in second.threads.items():
            before = first.threads.get(role, {})
            into = self.thread_ns.setdefault(role, {})
            for tid, used in threads.items():
                into[tid] = into.get(tid, 0) + max(used - before.get(tid, 0), 0)

    @property
    def fps(self) -> float:
        """The frame rate of the phase."""
        return self.frames / self.seconds if self.seconds > 0 else 0.0

    def share(self, role: str) -> float | None:
        """The CPU time of a role as a percentage of one core, or `None` without a measurement."""
        if self.seconds <= 0 or role not in self.cpu_ns:
            return None
        return 100.0 * self.cpu_ns[role] / 1e9 / self.seconds

    def top_threads(self, role: str, count: int = 4) -> list[tuple[int, float]]:
        """The busiest threads of a role as `(thread ID, percentage of one core)`."""
        if self.seconds <= 0:
            return []
        threads = self.thread_ns.get(role, {})
        ranked = sorted(threads.items(), key=lambda item: -item[1])[:count]
        return [(tid, 100.0 * used / 1e9 / self.seconds) for tid, used in ranked]


def is_clean_fast(
    first: Snapshot, second: Snapshot, *, warmup_windows: int, min_fps: float = MIN_FAST_FPS
) -> bool:
    """Whether the interval between two samples belongs to the fast phase."""
    return (
        first.state == "auto"
        and second.state == "auto"
        and first.purpose == "fast"
        and second.purpose == "fast"
        and first.stream_id == second.stream_id
        and first.survey_pending == 0
        and second.survey_pending == 0
        and first.survey_steps == second.survey_steps
        and first.windows >= warmup_windows
        and second.frames - first.frames >= min_fps * (second.t - first.t)
    )


def is_idle(first: Snapshot, second: Snapshot) -> bool:
    """Whether the interval between two samples belongs to the idle phase: paused, no frame."""
    return first.state == "paused" and second.state == "paused" and first.frames == second.frames


def split_phases(
    snapshots: list[Snapshot], *, warmup_windows: int, min_fps: float = MIN_FAST_FPS
) -> tuple[Phase, Phase]:
    """The fast phase and the idle phase of a list of samples."""
    fast, idle = Phase(), Phase()
    for first, second in pairwise(snapshots):
        if is_clean_fast(first, second, warmup_windows=warmup_windows, min_fps=min_fps):
            fast.add(first, second)
        elif is_idle(first, second):
            idle.add(first, second)
    return fast, idle


def cost_per_frame_us(fast: Phase, idle: Phase, role: str) -> float | None:
    """The CPU time that a frame costs a role, in microseconds.

    It is the share that the role uses in the fast phase minus the share that it uses when idle,
    divided by the frame rate. Returns `None` without frames or without a share.
    """
    busy, quiet = fast.share(role), idle.share(role)
    if busy is None or quiet is None or fast.fps <= 0:
        return None
    return max(busy - quiet, 0.0) / 100.0 / fast.fps * 1e6


@dataclass(frozen=True, slots=True)
class SystemRun:
    """What a run found: the phases, the peaks, and the facts about the run."""

    plan: RunPlan
    fast: Phase
    idle: Phase
    peaks: Mapping[str, int]
    snapshots: tuple[Snapshot, ...]
    startup_s: float
    sampling_s: float
    survey_steps: int
    survey_results: int
    stream: Mapping[str, Any]
    worker_cpu_ns: int
    logical_cpus: int
    machine_busy_percent: float | None
    own_busy_percent: float | None
    notes: tuple[str, ...] = ()


# --- Reading the processes ----------------------------------------------------------------------


class Sampler:
    """Reads the CPU time and the peak memory of the processes of a run.

    `roots` maps `acquire`, `core`, and `web` to the process IDs that the launcher started. The
    sampler follows each to the process that does the work (a virtual-environment launcher on
    Windows starts the interpreter as a child), and it finds the children of `core`: the survey
    worker, and on Linux the resource tracker of the `multiprocessing` module.
    """

    def __init__(self, roots: Mapping[str, int]) -> None:
        self._roots = dict(roots)
        self._peak_by_pid: dict[int, int] = {}
        self._role_by_pid: dict[int, str] = {}
        self._decided: dict[int, str] = {}
        self._last_cpu: dict[int, int] = {}

    def _classify_child(self, pid: int) -> str | None:
        """The role of a child of `core`, or `None` while the system does not tell yet.

        On Linux, the command line of a process that has just started can read as empty for a
        moment, and a guess then could take the resource tracker for the worker. Windows gives no
        command line, so the program of the process decides there.
        """
        line = procs.command_line(pid)
        if line == "":
            return None
        if line is not None:
            if "spawn_main" in line or "multiprocessing.spawn" in line:
                return "survey_worker"
            if "resource_tracker" in line:
                return "other"
        image = (procs.image_path(pid) or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
        return "survey_worker" if image.startswith("python") else "other"

    def _child_role(self, pid: int) -> str:
        """The role of a child of `core`. A child that is not decided yet counts as `other`."""
        role = self._decided.get(pid)
        if role is None:
            role = self._classify_child(pid)
            if role is None:
                return "other"
            self._decided[pid] = role
        return role

    def processes(self) -> dict[int, str]:
        """The process of each role right now, as `{process ID: role}`."""
        parents = procs.parent_map()
        found: dict[int, str] = {}
        for role, root in self._roots.items():
            real = procs.real_process(root, parents)
            found[real] = role
            if role == "core":
                for child in procs.children_of(real, parents):
                    worker = procs.real_process(child, parents)
                    if worker != real:
                        found[worker] = self._child_role(worker)
        return found

    def read(self) -> tuple[dict[str, int], dict[str, dict[int, int]]]:
        """The cumulative CPU time by role, and by thread for the roles that have threads."""
        cpu: dict[str, int] = {}
        threads: dict[str, dict[int, int]] = {}
        for pid, role in self.processes().items():
            reading = procs.read_process(pid)
            if reading is None:
                continue
            self._role_by_pid[pid] = role
            if reading.peak_rss_bytes is not None:
                self._peak_by_pid[pid] = max(self._peak_by_pid.get(pid, 0), reading.peak_rss_bytes)
            if reading.cpu_ns is not None:
                self._last_cpu[pid] = reading.cpu_ns
            cpu[role] = cpu.get(role, 0) + self._last_cpu.get(pid, 0)
            if role in THREAD_ROLES:
                by_thread = procs.thread_cpu_ns(pid)
                if by_thread:
                    threads[role] = by_thread
        return cpu, threads

    def peaks(self) -> dict[str, int]:
        """The peak resident size by role: the largest process of a role, the sum for `other`."""
        by_role: dict[str, list[int]] = {}
        for pid, peak in self._peak_by_pid.items():
            by_role.setdefault(self._role_by_pid[pid], []).append(peak)
        return {
            role: sum(values) if role == "other" else max(values)
            for role, values in by_role.items()
        }

    def worker_cpu_ns(self) -> int:
        """The CPU time that the survey worker used, in nanoseconds."""
        return sum(
            used
            for pid, used in self._last_cpu.items()
            if self._role_by_pid[pid] == "survey_worker"
        )


# --- Starting and polling -----------------------------------------------------------------------


def free_port() -> int:
    """A port of the loopback interface that nothing listens at now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def clean_environment() -> dict[str, str]:
    """The environment of the run, without the settings of the person who starts it."""
    return {k: v for k, v in os.environ.items() if not k.startswith("SEEINGMON_")}


class WebPoller:
    """A client that asks `web` for a few pages at an interval, as an open page would."""

    def __init__(self, port: int, interval_s: float) -> None:
        self._urls = [f"http://127.0.0.1:{port}/api/v1/{path}" for path in WEB_PATHS]
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="perf-web-poller", daemon=True)
        self.requests = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            for url in self._urls:
                with contextlib.suppress(OSError, http.client.HTTPException, ValueError):
                    with urllib.request.urlopen(url, timeout=5.0) as response:
                        response.read()
                    self.requests += 1


def _fail(message: str, children: Mapping[str, Any]) -> RuntimeError:
    tails = "\n".join(f"--- {name}\n{child.log_tail(15)}" for name, child in children.items())
    return RuntimeError(f"{message}\n{tails}")


def run_system(plan: RunPlan, *, log: Callable[[str], None] | None = None) -> SystemRun:
    """Start the system, sample it through the phases, stop it, and return what it found.

    The system runs in a folder that the function deletes, on a free port of the loopback
    interface, with its own addresses and key, so it never meets another system on the machine.
    Raises `RuntimeError` when a process dies, or when the fast phase never gets a second.
    """
    from seeingmon.scheduler.commands import Pause
    from seeingmon.services.dev import (
        Child,
        DevOptions,
        build_plan,
        wait_for_web,
        wait_until_ready,
    )
    from seeingmon.services.ipc.endpoint import Endpoint
    from seeingmon.services.ipc.keys import ConnectionKey
    from seeingmon.services.ipc.rpc import connect_rpc
    from seeingmon.services.web.contract import (
        METHOD_STATUS,
        METHOD_SUBMIT,
        RPC_CHANNEL,
        decode_result,
        decode_status,
        encode_command,
    )

    say = log or (lambda text: None)
    with tempfile.TemporaryDirectory(prefix="smon-perf-core-", ignore_cleanup_errors=True) as name:
        folder = Path(name)
        options = DevOptions(
            speed=plan.speed,
            port=free_port(),
            sensor=plan.sensor,
            fast_exposure_us=plan.fast_exposure_us,
            polaris_mag=plan.polaris_mag,
        )
        dev_plan = build_plan(
            options,
            directory=folder / "run",
            local_file=folder / "no-owner-settings.toml",
            env=clean_environment(),
        )
        children = {spec.name: Child(spec) for spec in dev_plan.children}
        client: Any = None
        poller: WebPoller | None = None
        stream_facts: dict[str, Any] = {}
        try:
            began = time.monotonic()
            for child_name in ("acquire", "core"):  # core connects to acquire, so acquire is first
                children[child_name].start()
                wait_until_ready(dev_plan, children, plan.ready_timeout_s, (child_name,))
            children["web"].start()
            if not wait_for_web(dev_plan, children["web"], plan.ready_timeout_s):
                raise _fail("web did not start", children)
            startup_s = time.monotonic() - began
            say(f"the three processes are up after {startup_s:.0f} s")
            client, _ = connect_rpc(
                Endpoint.parse(dev_plan.core_endpoint),
                ConnectionKey.from_text(dev_plan.key),
                {"role": "cli"},
                channel=RPC_CHANNEL,
            )
            roots = {
                child_name: child.popen.pid for child_name, child in children.items() if child.popen
            }
            sampler = Sampler(roots)
            poller = WebPoller(dev_plan.port, plan.poll_interval_s)
            poller.start()

            def snapshot() -> Snapshot:
                for child_name, child in children.items():
                    if not child.running:
                        raise _fail(f"{child_name} stopped with code {child.returncode}", children)
                scheduler = decode_status(client.call(METHOD_STATUS)).scheduler
                stream = scheduler.stream
                if stream is not None and stream.purpose == "fast" and not stream_facts:
                    stream_facts.update(
                        mode=stream.mode,
                        exposure_us=stream.exposure_us,
                        gain=stream.gain,
                        roi=None if stream.roi is None else (stream.roi.width, stream.roi.height),
                    )
                cpu, threads = sampler.read()
                counters = scheduler.counters
                return Snapshot(
                    t=time.monotonic(),
                    state=scheduler.state,
                    purpose=None if stream is None else stream.purpose,
                    stream_id=None if stream is None else stream.stream_id,
                    frames=counters.get("frames", 0),
                    windows=counters.get("windows", 0),
                    survey_steps=counters.get("survey_steps", 0),
                    survey_results=counters.get("survey_results", 0),
                    survey_pending=scheduler.survey_pending,
                    cpu_ns=cpu,
                    threads=threads,
                )

            snapshots: list[Snapshot] = []
            notes: list[str] = []
            ticks_before = busy_ticks()
            sampling_began = time.monotonic()
            paused_at: float | None = None
            stage = "collect"
            next_tick = sampling_began
            while True:
                snapshots.append(snapshot())
                latest = snapshots[-1]
                fast, idle = split_phases(
                    snapshots, warmup_windows=plan.warmup_windows, min_fps=plan.min_fast_fps
                )
                elapsed = latest.t - sampling_began
                if stage == "collect":
                    enough = (
                        fast.seconds >= plan.fast_seconds
                        and latest.survey_steps >= plan.survey_steps
                        and latest.survey_pending == 0
                    )
                    if enough or elapsed > plan.max_run_s:
                        if not enough:
                            notes.append(
                                f"the sampling reached its limit of {plan.max_run_s:.0f} s with "
                                f"{fast.seconds:.0f} s of the fast phase and "
                                f"{latest.survey_steps} survey steps"
                            )
                        result = decode_result(
                            client.call(METHOD_SUBMIT, {"command": encode_command(Pause())})
                        )
                        if not result.accepted:
                            raise _fail(f"core refused the pause: {result.message}", children)
                        stage, paused_at = "pausing", latest.t
                        say(f"the fast phase has {fast.seconds:.0f} s, and the scheduler pauses")
                elif stage == "pausing":
                    assert paused_at is not None
                    if latest.t - paused_at > PAUSE_WAIT_S:
                        raise _fail("the scheduler did not pause", children)
                    if idle.seconds > 0:
                        stage = "idle"
                elif idle.seconds >= plan.idle_seconds:
                    break
                next_tick += plan.sample_interval_s
                pause_s = next_tick - time.monotonic()
                if pause_s > 0:
                    time.sleep(pause_s)
            ticks_after = busy_ticks()
            sampling_s = snapshots[-1].t - snapshots[0].t
            fast, idle = split_phases(
                snapshots, warmup_windows=plan.warmup_windows, min_fps=plan.min_fast_fps
            )
            if fast.seconds < 1.0 or fast.frames <= 0:
                raise _fail("the fast stream never reached a steady state", children)
            logical = os.cpu_count() or 1
            machine = (
                None
                if ticks_before is None or ticks_after is None
                else busy_percent_between(ticks_before, ticks_after)
            )
            own_ns = sum(
                max(used - snapshots[0].cpu_ns.get(role, 0), 0)
                for role, used in snapshots[-1].cpu_ns.items()
            )
            own = None if sampling_s <= 0 else 100.0 * own_ns / 1e9 / sampling_s / logical
            return SystemRun(
                plan=plan,
                fast=fast,
                idle=idle,
                peaks=sampler.peaks(),
                snapshots=tuple(snapshots),
                startup_s=startup_s,
                sampling_s=sampling_s,
                survey_steps=snapshots[-1].survey_steps,
                survey_results=snapshots[-1].survey_results,
                stream=stream_facts,
                worker_cpu_ns=sampler.worker_cpu_ns(),
                logical_cpus=logical,
                machine_busy_percent=machine,
                own_busy_percent=own,
                notes=tuple(notes),
            )
        finally:
            if poller is not None:
                poller.stop()
            if client is not None:
                with contextlib.suppress(Exception):
                    client.close("the run is done")
            for child_name in ("web", "core", "acquire"):
                children[child_name].stop()
