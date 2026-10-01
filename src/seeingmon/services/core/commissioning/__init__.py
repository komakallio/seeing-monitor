"""Commissioning in `core`: the burst and replay handlers, and the client of the commands.

The scheduler owns the queue of commissioning tasks and the sweep. `core` registers the other two
handlers with `Scheduler.register_handler`:

- `burst` (`seeingmon.services.core.commissioning.burst`) records raw frames to a SER file with a
  JSON sidecar in the `bursts` folder of the data layout, and pins the burst.
- `replay` (`seeingmon.services.core.commissioning.replay`) feeds a recording through the
  production analysis into a separate store under the data directory.

The commands `seeingmon burst`, `sweep`, and `replay` queue a task in the running `core` through
the RPC (`seeingmon.services.core.commissioning.client`), and they have a `--standalone` mode that
runs the task against the configured driver without `core`, for bench work
(`seeingmon.services.core.commissioning.standalone`).
"""

from __future__ import annotations
