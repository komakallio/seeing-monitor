"""The `ipc` case: the cost of moving a frame from `acquire` to `core`.

The case starts `acquire` in its own process (`seeingmon.perf._acquire`), and this process plays
`core`: a `RemoteCameraDriver` opens the session, starts the stream, and reads frames with no
pause. `acquire` runs the production `AcquireService` on a fake camera that replays 64 frames of
a Polaris-like star (bin1, 128 x 128, 16 bit, 32 KB each). The fake costs almost nothing, so the
CPU time of the `acquire` process is the cost of `acquire` itself: the capture thread, the time
stamper, the queue, the encoder, and the sender. The case uses two processes, as production does.
The in-process thread pair that the brief allows for a runner where a subprocess is fragile is
not needed, because the end-to-end tests of the services lane start `acquire` the same way on
every runner.

The camera paces itself on the real clock, and the case runs two rates:

- **nominal**, the rate of the budget (98 frames per second): the CPU time of each process, per
  frame and as a share of one core, is the figure to compare with the 10% budget of `acquire`;
- **stress**, four times the nominal rate: the processes work harder, so the CPU clock (coarse on
  Windows) reads them more accurately, and the drop counters show whether the layer keeps up.

An unthrottled source is not a useful test: the capture thread then outruns the sender, the
queue drops frames, and the figure shows how the interpreter shares its lock between threads.
The vendor SDK and the USB transfer cost something too, and the harness does not measure them.
"""

from __future__ import annotations

import json
import queue
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seeingmon.perf.cases.fastmodes import FAST_MODES, REFERENCE_PROFILE, pool_frames
from seeingmon.perf.memory import process_cpu_ns
from seeingmon.perf.registry import REGISTRY, CaseContext
from seeingmon.perf.report import Measurement
from seeingmon.perf.runner import child_environment

if TYPE_CHECKING:
    from seeingmon.services.ipc.endpoint import Endpoint

NOMINAL_HZ = 98.0
STRESS_FACTOR = 4.0
_COMMAND_TIMEOUT_S = 60.0
_ERROR_LINES = 12
_FLOOR = 1e-3  # microseconds, for a CPU time below the resolution of the CPU clock


class AcquireProcess:
    """The `acquire` helper process: a line-based control channel on its standard streams."""

    def __init__(self, endpoint: Endpoint, key_text: str, pool_path: Path, rate_hz: float) -> None:
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "seeingmon.perf._acquire",
                "--address",
                str(endpoint),
                "--pool",
                str(pool_path),
                "--rate",
                str(rate_hz),
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


def run_pair(pool_path: Path, folder: Path, rate_hz: float, seconds: float) -> dict[str, Any]:
    """Run the pair for `seconds`, and return the CPU times and the counters of both sides."""
    from seeingmon.drivers.base import CameraTimeoutError
    from seeingmon.frames import Roi, StreamConfig
    from seeingmon.services.ipc.keys import ConnectionKey
    from seeingmon.services.remote import RemoteCameraDriver

    endpoint = new_endpoint(folder)
    key_text = secrets.token_urlsafe(32)
    helper = AcquireProcess(endpoint, key_text, pool_path, rate_hz)
    try:
        helper.expect("ready")
        driver = RemoteCameraDriver(
            endpoint,
            ConnectionKey.from_text(key_text),
            connect_timeout_s=30.0,
            rpc_timeout_s=30.0,
        )
        try:
            driver.open()
            driver.configure(StreamConfig("bin1", 2000, 0, roi=Roi(100, 200, 128, 128)))
            helper.send("mark")
            helper.expect("marked")
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
            helper.send("report")
            report: dict[str, Any] = json.loads(helper.expect("{"))
        finally:
            driver.close()
    finally:
        helper.close()
    if received == 0 or report["frames_sent"] == 0:
        raise RuntimeError("no frame crossed the connection")
    return {
        "received": received,
        "wall_s": wall_ns / 1e9,
        "core_cpu_ns": cpu_ns,
        "acquire": report,
    }


def _per_frame_us(cpu_ns: float, frames: int) -> float:
    return max(cpu_ns / frames / 1e3, _FLOOR)


def run_figures(prefix: str, rate_hz: float, result: dict[str, Any]) -> list[Measurement]:
    """The measurements of one run: CPU per frame and the share at the rate, for both sides."""
    acquire = result["acquire"]
    sent, received = int(acquire["frames_sent"]), int(result["received"])
    detail: dict[str, float | int | str] = {
        "rate_hz": rate_hz,
        "frames_received": received,
        "frames_sent": sent,
        "frames_captured": int(acquire["frames_captured"]),
        "dropped_at_queue": int(acquire["dropped_queue"]),
        "dropped_by_gap": int(acquire["dropped_gap"]),
        "queue_peak_frames": int(acquire["queue_peak_frames"]),
        "received_fps": round(received / result["wall_s"], 1),
        "wall_s": round(result["wall_s"], 2),
        "processes": "acquire in a child process, core side in this process",
    }
    acquire_us = _per_frame_us(float(acquire["cpu_ns"]), sent)
    core_us = _per_frame_us(float(result["core_cpu_ns"]), received)
    share = NOMINAL_HZ / 1e4  # percent of one core for a cost in microseconds at 98 fps
    return [
        Measurement(
            f"{prefix}acquire.cpu_per_frame", "us/frame", acquire_us, None, "interpreter", detail
        ),
        Measurement(
            f"{prefix}acquire.share",
            "percent",
            acquire_us * share,
            None,
            "interpreter",
            dict(detail),
        ),
        Measurement(
            f"{prefix}core_rx.cpu_per_frame", "us/frame", core_us, None, "interpreter", dict(detail)
        ),
        Measurement(
            f"{prefix}core_rx.share", "percent", core_us * share, None, "interpreter", dict(detail)
        ),
    ]


@REGISTRY.case("ipc", summary="Cost of moving a frame from acquire to core, in CPU time per frame")
def ipc(ctx: CaseContext) -> list[Measurement]:
    import numpy as np

    from seeingmon.profile import load_profile

    profile = load_profile(REFERENCE_PROFILE)
    mode = FAST_MODES[0]
    ctx.mark_baseline()
    measurements: list[Measurement] = []
    with tempfile.TemporaryDirectory(prefix="smon-", ignore_cleanup_errors=True) as name:
        folder = Path(name)
        pool_path = folder / "pool.npy"
        np.save(pool_path, np.stack(pool_frames(profile, mode, ctx.smoke)))
        nominal = run_pair(pool_path, folder, NOMINAL_HZ, ctx.pick(5.0, 0.5))
        measurements.extend(run_figures("", NOMINAL_HZ, nominal))
        measurements.append(
            Measurement(
                "acquire.peak_rss",
                "bytes",
                float(nominal["acquire"]["peak_rss_bytes"] or 1),
                None,
                "memory",
                {"process": "acquire"},
            )
        )
        if not ctx.smoke:
            stress = run_pair(pool_path, folder, NOMINAL_HZ * STRESS_FACTOR, 3.0)
            measurements.extend(run_figures("stress.", NOMINAL_HZ * STRESS_FACTOR, stress))
    ctx.note(
        "acquire runs in its own process with a fake camera that replays 64 frames, and the CPU "
        "time of that process is the cost of acquire. This process reads the stream the way core "
        "does. The shares are the CPU time of a frame times the nominal rate of 98 fps. The "
        "stress run is four times faster, and its CPU clock reads a busier process."
    )
    return measurements
