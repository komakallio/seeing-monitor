"""The `ipc` case: the cost of moving a frame from `acquire` to `core`.

The case starts `acquire` in its own process (`seeingmon.perf._acquire`), and this process plays
`core`: a `RemoteCameraDriver` opens the session, starts the stream, and reads frames with no
pause. `acquire` runs the production `AcquireService` on a fake camera that replays 64 frames of
a Polaris-like star (32 KB for bin1 at 128 x 128, 8 KB for bin2 at 64 x 64). The fake costs almost
nothing, so the CPU time of the `acquire` process is the cost of `acquire` itself: the capture
thread, the time stamper, the queue, the encoder, and the sender. The case uses two processes, as
production does. The in-process thread pair that the brief allows for a runner where a subprocess
is fragile is not needed, because the end-to-end tests of the services lane start `acquire` the
same way on every runner.

The camera paces itself on the real clock. The case runs

- **nominal**: bin1 at 98 frames per second, the rate of the budget. It repeats the run and
  reports the median of the runs and their spread. The CPU time of each process, per frame and as a
  share of one core, is the figure to compare with the 10% budget of `acquire`. The detail of the
  figure splits the CPU time of `acquire` among its threads;
- **stress**: bin1 at four times the nominal rate. The processes work harder, so the CPU clock
  (coarse on Windows) reads them more accurately, and the drop counters show whether the layer
  keeps up;
- **bin2**: 64 x 64 at 360 frames per second, for the receive cost of the second fast mode.

The CPU time of the `core` side is what receiving costs `core`: the stream reader thread and the
decoder. It adds to the fast path in the 25% budget, so the budgets include it in a second row.
An unthrottled source is not a useful test: the capture thread then outruns the sender, the queue
drops frames, and the figure shows how the interpreter shares its lock between threads. The vendor
SDK and the USB transfer cost something too, and the harness does not measure them.

**Read the figure with care.** The cost per frame is dominated by thread wake-ups: the capture
thread sleeps until the next frame, the sender wakes for each frame, and the reader of `core`
wakes for each message. The cost of a wake-up depends on the operating system and, in a virtual
machine, on the hypervisor, so this case transfers to a Pi 4 less well than the others do. The
`calibration` case has a thread hand-off workload that measures the same effect on each machine.
"""

from __future__ import annotations

import json
import queue
import secrets
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

from seeingmon.perf.cases.fastmodes import FAST_MODES, REFERENCE_PROFILE, FastMode, pool_frames
from seeingmon.perf.memory import process_cpu_ns
from seeingmon.perf.registry import REGISTRY, CaseContext
from seeingmon.perf.report import Measurement
from seeingmon.perf.runner import child_environment
from seeingmon.perf.timing import TimingStats

if TYPE_CHECKING:
    from seeingmon.drivers.base import CameraDriver
    from seeingmon.services.ipc.endpoint import Endpoint

NOMINAL_HZ = 98.0
BIN2_HZ = 360.0
STRESS_FACTOR = 4.0
_COMMAND_TIMEOUT_S = 60.0
_ERROR_LINES = 12
_FLOOR = 1e-3  # microseconds, for a CPU time below the resolution of the CPU clock


class AcquireProcess:
    """The `acquire` helper process: a line-based control channel on its standard streams."""

    def __init__(self, endpoint: Endpoint, key_text: str, pool_path: Path) -> None:
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "seeingmon.perf._acquire",
                "--address",
                str(endpoint),
                "--pool",
                str(pool_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=child_environment(),
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._errors: deque[str] = deque(maxlen=_ERROR_LINES)
        threading.Thread(target=self._read, name="acquire-output", daemon=True).start()
        threading.Thread(target=self._drain, name="acquire-errors", daemon=True).start()
        assert self._process.stdin is not None
        self._process.stdin.write(key_text + "\n")  # the key goes through the pipe, not argv
        self._process.stdin.flush()

    def _read(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._lines.put(line.rstrip("\n"))
        self._lines.put(None)

    def _drain(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._errors.append(line.rstrip("\n"))

    def error_tail(self) -> str:
        """The last lines that the process wrote to its standard error."""
        return " | ".join(self._errors)

    def send(self, command: str) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()

    def expect(self, prefix: str, timeout_s: float = _COMMAND_TIMEOUT_S) -> str:
        """The next line of output, which must start with `prefix`."""
        try:
            line = self._lines.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError(f"acquire did not answer {prefix!r} in {timeout_s:.0f} s") from None
        if line is None or not line.startswith(prefix):
            raise RuntimeError(
                f"acquire answered {line!r} when {prefix!r} was due; it wrote: {self.error_tail()}"
            )
        return line

    def close(self) -> None:
        """Stop the process, and drop it after a few seconds if it does not stop."""
        try:
            if self._process.poll() is None:
                self.send("quit")
        except OSError:
            pass
        try:
            self._process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10.0)
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
                stream.close()


def new_endpoint(folder: Path) -> Endpoint:
    """An endpoint of the platform's own family, as production uses it."""
    from seeingmon.services.ipc.endpoint import Endpoint

    text = (
        f"seeingmon-perf-{uuid.uuid4().hex[:12]}"
        if sys.platform == "win32"
        else str(folder / "acquire.sock")
    )
    return Endpoint.parse(text)


@dataclass(frozen=True, slots=True)
class RunResult:
    """One run of the pair: the frames that `core` read, and the CPU time of both sides."""

    received: int
    wall_s: float
    core_cpu_ns: int
    acquire: dict[str, Any]

    @property
    def sent(self) -> int:
        return int(self.acquire["frames_sent"])

    @property
    def acquire_us(self) -> float:
        return _per_frame_us(float(self.acquire["cpu_ns"]), self.sent)

    @property
    def core_us(self) -> float:
        return _per_frame_us(float(self.core_cpu_ns), self.received)

    def thread_us(self) -> dict[str, float]:
        """The CPU time per frame of `acquire`: capture thread, sender thread, and the rest."""
        threads: dict[str, int] = self.acquire["threads_cpu_ns"]
        capture = sum(used for name, used in threads.items() if "capture" in name)
        sender = sum(used for name, used in threads.items() if "sender" in name)
        rest = sum(threads.values()) - capture - sender
        per_frame = 1.0 / max(self.sent, 1) / 1e3
        return {
            "capture_us": round(capture * per_frame, 1),
            "sender_us": round(sender * per_frame, 1),
            "other_us": round(max(rest, 0) * per_frame, 1),
        }


def _per_frame_us(cpu_ns: float, frames: int) -> float:
    return max(cpu_ns / max(frames, 1) / 1e3, _FLOOR)


class Session:
    """The `acquire` process and the driver that reads from it, for several runs."""

    def __init__(self, folder: Path, pool_path: Path) -> None:
        from seeingmon.services.ipc.keys import ConnectionKey
        from seeingmon.services.remote import RemoteCameraDriver

        self._endpoint = new_endpoint(folder)
        key_text = secrets.token_urlsafe(32)
        self._helper = AcquireProcess(self._endpoint, key_text, pool_path)
        self._key = ConnectionKey.from_text(key_text)
        self._driver_class = RemoteCameraDriver
        self._driver: CameraDriver | None = None

    def __enter__(self) -> Session:
        try:
            self._helper.expect("ready")
            self._driver = self._driver_class(
                self._endpoint, self._key, connect_timeout_s=30.0, rpc_timeout_s=30.0
            )
            self._driver.open()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._driver is not None:
                self._driver.close()
        finally:
            self._helper.close()

    def run(self, mode: FastMode, rate_hz: float, seconds: float) -> RunResult:
        """Stream `mode` at `rate_hz` for `seconds`, reading with no pause."""
        from seeingmon.drivers.base import CameraTimeoutError
        from seeingmon.frames import Roi, StreamConfig

        driver = self._driver
        assert driver is not None
        height, width = mode.shape
        config = StreamConfig(
            mode.mode, round(1e6 / rate_hz), mode.gain, roi=Roi(100, 200, width, height)
        )
        driver.configure(config)
        self._helper.send("mark")
        self._helper.expect("marked")
        received = 0
        wall_start = time.perf_counter_ns()
        deadline_ns = wall_start + round(seconds * 1e9)
        driver.start()
        cpu_start = process_cpu_ns()
        while time.perf_counter_ns() < deadline_ns:
            try:
                driver.read_frame(5.0)
            except CameraTimeoutError:
                break
            received += 1
        cpu_ns = process_cpu_ns() - cpu_start
        wall_ns = time.perf_counter_ns() - wall_start
        driver.stop()
        self._helper.send("report")
        report: dict[str, Any] = json.loads(self._helper.expect("{"))
        if received == 0 or int(report["frames_sent"]) == 0:
            raise RuntimeError("no frame crossed the connection")
        return RunResult(received, wall_ns / 1e9, cpu_ns, report)


def _stats(values: list[float]) -> TimingStats:
    return TimingStats.from_samples(values)


def _detail(
    rate_hz: float, runs: list[RunResult], extra: dict[str, float | int | str]
) -> dict[str, float | int | str]:
    last = runs[-1]
    detail: dict[str, float | int | str] = {
        "rate_hz": rate_hz,
        "runs": len(runs),
        "frames_per_run": round(statistics.median(run.sent for run in runs)),
        "received_fps": round(statistics.median(run.received / run.wall_s for run in runs), 1),
        "dropped_at_queue": sum(int(run.acquire["dropped_queue"]) for run in runs),
        "dropped_by_gap": sum(int(run.acquire["dropped_gap"]) for run in runs),
        "queue_peak_frames": int(last.acquire["queue_peak_frames"]),
    }
    detail.update(extra)
    return detail


def figures(
    prefix: str,
    rate_hz: float,
    runs: list[RunResult],
    *,
    with_acquire: bool = True,
    threads: bool = False,
) -> list[Measurement]:
    """The measurements of the runs of one stream: CPU per frame and the share at the rate."""
    share = rate_hz / 1e4  # percent of one core for a cost in microseconds at this rate
    extra: dict[str, float | int | str] = {}
    if threads:
        split = [run.thread_us() for run in runs]
        for key in ("capture_us", "sender_us", "other_us"):
            extra[key] = round(statistics.median(item[key] for item in split), 1)
    detail = _detail(rate_hz, runs, extra)
    out: list[Measurement] = []
    sides = [("core_rx", [run.core_us for run in runs])]
    if with_acquire:
        sides.insert(0, ("acquire", [run.acquire_us for run in runs]))
    for side, costs in sides:
        per_frame = _stats(costs)
        out.append(
            Measurement(
                f"{prefix}{side}.cpu_per_frame",
                "us/frame",
                per_frame.median,
                per_frame,
                "interpreter",
                dict(detail),
            )
        )
        shares = per_frame.scaled(share)
        out.append(
            Measurement(
                f"{prefix}{side}.share",
                "percent",
                shares.median,
                shares,
                "interpreter",
                {**detail, "share_at_hz": rate_hz},
            )
        )
    return out


@REGISTRY.case("ipc", summary="Cost of moving a frame from acquire to core, in CPU time per frame")
def ipc(ctx: CaseContext) -> list[Measurement]:
    import numpy as np

    from seeingmon.profile import load_profile

    profile = load_profile(REFERENCE_PROFILE)
    bin1, bin2 = FAST_MODES[0], FAST_MODES[1]
    ctx.mark_baseline()
    repeats = ctx.pick(3, 1)
    seconds = ctx.pick(4.0, 0.4)
    measurements: list[Measurement] = []
    with tempfile.TemporaryDirectory(prefix="smon-", ignore_cleanup_errors=True) as name:
        folder = Path(name)
        pool_path = folder / "pool.npz"
        pools: dict[str, Any] = {
            f"f{mode.shape[0]}x{mode.shape[1]}": np.stack(pool_frames(profile, mode))
            for mode in (bin1, bin2)
        }
        np.savez(pool_path, **pools)
        with Session(folder, pool_path) as session:
            session.run(bin1, NOMINAL_HZ, seconds / 2)  # a warm-up that is not reported
            nominal = [session.run(bin1, NOMINAL_HZ, seconds) for _ in range(repeats)]
            measurements.extend(figures("", NOMINAL_HZ, nominal, threads=True))
            peak = nominal[-1].acquire["peak_rss_bytes"]
            measurements.append(
                Measurement(
                    "acquire.peak_rss",
                    "bytes",
                    float(peak or 1),
                    None,
                    "memory",
                    {"process": "acquire"},
                )
            )
            second = [session.run(bin2, BIN2_HZ, ctx.pick(3.0, 0.4))]
            measurements.extend(figures("bin2.", BIN2_HZ, second, with_acquire=False))
            if not ctx.smoke:
                stress = [session.run(bin1, NOMINAL_HZ * STRESS_FACTOR, 3.0)]
                measurements.extend(figures("stress.", NOMINAL_HZ, stress))
    ctx.note(
        "acquire runs in its own process with a fake camera that replays 64 frames, and the CPU "
        "time of that process is the cost of acquire. This process reads the stream the way core "
        "does. A share is the CPU time of a frame times the nominal rate (98 fps for bin1, 360 fps "
        "for the bin2 row). The nominal figure is the median of the runs. The stress run is four "
        "times faster, and its share still uses the nominal rate."
    )
    return measurements
