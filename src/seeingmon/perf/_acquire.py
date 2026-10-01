"""The `acquire` side of the `ipc` case, in a process of its own.

    python -m seeingmon.perf._acquire --address ADDRESS --pool FILE [--burst N]

The process runs an `AcquireService` with a fake camera that replays the frames of a pool (a NumPy
`.npz` file that the case wrote, with one array of frames for each ROI size, such as `f128x128`).
The fake costs almost nothing, so the CPU time that the process uses is the cost of `acquire`
itself: the capture thread, the time stamper, the queue, and the sender. The fake has no readout
time, so the exposure of a stream is its frame period, and the frame rate is the inverse of the
exposure.

**Bursts.** A paced camera wakes the capture thread for every frame, and each frame then wakes the
sender, so the CPU time per frame includes the cost of the wake-ups. With `--burst N`, the camera
delivers N frames at a time: the capture thread sleeps for N frame periods, and then it reads N
frames with no pause. The frames keep their regular time stamps, and the rate stays the same. The
wake-ups are shared by N frames, so two runs with different bursts tell the work from the
wake-ups (see `seeingmon.perf.cases.ipc`).

The process reads commands from its standard input, one per line, and answers on its standard
output:

- the first line of the input is the connection key, which never appears on a command line;
- `ready <endpoint>` is the first answer, after the service listens;
- `mark` records the CPU time of the process and of each thread, and the frame counters, and
  answers `marked`;
- `report` answers one line of JSON with the CPU time of the process and of each thread and the
  counters since the mark, and the peak memory of the process;
- `quit` stops the service and ends the process.

The process imports no SciPy and no simulator, so its memory is that of a real `acquire`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from typing import Any

from seeingmon.clock import SystemClock
from seeingmon.perf.memory import peak_rss_bytes, process_cpu_ns
from seeingmon.perf.threadcpu import threads_cpu_ns


class BurstClock(SystemClock):
    """The system clock for a camera that delivers `burst` frames at a time.

    The fake camera calls `sleep` once per frame, with the frame period. The first call of a burst
    sleeps for `burst` periods, and the other calls return at once. A frame of a burst gets the
    time that it would have had if the camera had delivered it alone: the `i`-th frame of a burst
    of `n` is `n - 1 - i` periods older than the moment of the read. The time stamps then follow a
    regular line, as the stamper of `acquire` expects, and the capture thread wakes once per burst.
    """

    def __init__(self, burst: int) -> None:
        super().__init__()
        if burst < 1:
            raise ValueError("a burst has at least one frame")
        self._burst = burst
        self._position = 0  # the number of frames that the current burst has returned
        self._period_s = 0.0

    def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self._period_s = seconds
        if self._position == 0:
            time.sleep(seconds * self._burst)
        self._position = (self._position + 1) % self._burst

    def _shift_ns(self) -> int:
        index = (self._position - 1) % self._burst  # the position of the frame just read
        return round((self._burst - 1 - index) * self._period_s * 1e9)

    def utc_ns(self) -> int:
        return super().utc_ns() - self._shift_ns()

    def monotonic_ns(self) -> int:
        return super().monotonic_ns() - self._shift_ns()


def _counters(service: Any) -> dict[str, int]:
    health = service.health()
    return {
        "frames_captured": int(health.frames_captured),
        "frames_sent": int(health.frames_sent),
        "dropped_queue": int(health.dropped_queue),
        "dropped_gap": int(health.dropped_gap),
        "flow_stalls": int(health.flow_stalls),
        "queue_peak_frames": int(health.queue_peak_frames),
    }


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run acquire with a fake camera for a benchmark.")
    parser.add_argument("--address", required=True, help="the socket path or pipe of the service")
    parser.add_argument("--pool", required=True, help="the NumPy file with the frames to replay")
    parser.add_argument(
        "--burst", type=int, default=1, help="frames that the camera delivers at a time"
    )
    args = parser.parse_args(argv)

    import numpy as np

    from seeingmon.services.acquire.service import AcquireService
    from seeingmon.services.config import AcquireSettings, ServicesConfig
    from seeingmon.services.ipc.endpoint import Endpoint
    from seeingmon.services.ipc.keys import ConnectionKey
    from seeingmon.testing import FakeCameraDriver

    key = ConnectionKey.from_text(sys.stdin.readline())
    with np.load(args.pool) as archive:
        pools = {name: archive[name] for name in archive.files}

    def next_frame(config: Any, roi: Any, seq: int) -> Any:
        pool = pools[f"f{roi.height}x{roi.width}"]
        return pool[seq % len(pool)]

    clock = BurstClock(args.burst)
    # The fake has no readout time, so the exposure of a stream is its frame period.
    driver = FakeCameraDriver(clock, overhead_s=0.0, row_time_s=0.0, frame_factory=next_frame)
    settings = ServicesConfig(
        acquire=AcquireSettings(
            time_source="stamp",  # fit the arrival times, as the real driver's frames need
            raise_priority=False,
        ),
    )
    service = AcquireService(
        driver,
        clock,
        Endpoint.parse(args.address),
        key,
        settings,
        priority_hook=lambda: "disabled",
    )
    started = service.start()
    _say(f"ready {started}")

    marked: dict[str, Any] = {}
    for line in sys.stdin:
        command = line.strip()
        if command == "mark":
            marked = {
                "cpu_ns": process_cpu_ns(),
                "wall_ns": time.perf_counter_ns(),
                "threads": threads_cpu_ns(),
            }
            marked.update(_counters(service))
            _say("marked")
        elif command == "report":
            now = _counters(service)
            report: dict[str, Any] = {
                "cpu_ns": process_cpu_ns() - marked.get("cpu_ns", 0),
                "wall_ns": time.perf_counter_ns() - marked.get("wall_ns", 0),
                "peak_rss_bytes": peak_rss_bytes(),
            }
            report.update({name: value - marked.get(name, 0) for name, value in now.items()})
            report["queue_peak_frames"] = now["queue_peak_frames"]
            before: dict[str, int] = marked.get("threads", {})
            report["threads_cpu_ns"] = {
                name: used - before.get(name, 0) for name, used in threads_cpu_ns().items()
            }
            _say(json.dumps(report))
        elif command == "quit":
            break
    service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
