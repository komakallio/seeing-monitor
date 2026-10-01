"""The `acquire` side of the `ipc` case, in a process of its own.

    python -m seeingmon.perf._acquire --address ADDRESS --pool FILE [--virtual-clock]

The process runs an `AcquireService` with a fake camera that replays the frames of a pool (a NumPy
`.npz` file that the case wrote, with one array of frames for each ROI size, such as `f128x128`).
The fake costs almost nothing, so the CPU time that the process uses is the cost of `acquire`
itself: the capture thread, the time stamper, the queue, and the sender. The fake has no readout
time, so the exposure of a stream is its frame period, and the frame rate is the inverse of the
exposure.

With `--virtual-clock`, the fake camera does not sleep: it advances a virtual clock instead, so the
capture thread runs as fast as it can and never waits. The queue then drops the frames that the
sender cannot take, and the CPU time of each thread, per frame that it handled, is the cost of the
work without the wake-ups of a paced stream.

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

from seeingmon.perf.memory import peak_rss_bytes, process_cpu_ns
from seeingmon.perf.threadcpu import threads_cpu_ns


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
        "--virtual-clock", action="store_true", help="never sleep: run the camera as fast as it can"
    )
    args = parser.parse_args(argv)

    import numpy as np

    from seeingmon.clock import Clock, SystemClock, VirtualClock
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

    clock: Clock = VirtualClock() if args.virtual_clock else SystemClock()
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
