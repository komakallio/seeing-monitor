"""The `acquire` side of the `ipc` case, in a process of its own.

    python -m seeingmon.perf._acquire --address ADDRESS --pool FILE --rate HZ [--rows N]

The process runs an `AcquireService` with a fake camera that replays the frames of a pool (a NumPy
file that the case wrote), so the CPU time that it uses is the cost of `acquire` itself: the
capture thread, the time stamper, the queue, and the sender. It reads commands from its standard
input, one per line, and answers on its standard output:

- the first line of the input is the connection key, which never appears on a command line;
- `ready <endpoint>` is the first answer, after the service listens;
- `mark` records the CPU time of the process and the frame counters, and answers `marked`;
- `report` answers one line of JSON with the CPU time and the counters since the mark, and the
  peak memory of the process;
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
    parser.add_argument("--rate", type=float, required=True, help="the frame rate, in hertz")
    args = parser.parse_args(argv)

    import numpy as np

    from seeingmon.clock import SystemClock
    from seeingmon.services.acquire.service import AcquireService
    from seeingmon.services.config import AcquireSettings, ServicesConfig
    from seeingmon.services.ipc.endpoint import Endpoint
    from seeingmon.services.ipc.keys import ConnectionKey
    from seeingmon.testing import FakeCameraDriver

    key = ConnectionKey.from_text(sys.stdin.readline())
    pool = np.load(args.pool)

    def next_frame(config: Any, roi: Any, seq: int) -> Any:
        return pool[seq % len(pool)]

    clock = SystemClock()
    # The fake paces itself: every frame takes `overhead_s` on the clock, and a row costs nothing.
    driver = FakeCameraDriver(
        clock, overhead_s=1.0 / args.rate, row_time_s=0.0, frame_factory=next_frame
    )
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
            marked = {"cpu_ns": process_cpu_ns(), "wall_ns": time.perf_counter_ns()}
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
            _say(json.dumps(report))
        elif command == "quit":
            break
    service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
