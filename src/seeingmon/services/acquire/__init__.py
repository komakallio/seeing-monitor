"""The `acquire` process: it owns one camera driver and streams its frames to `core`.

`acquire` is small on purpose. It has a capture thread that reads frames and stamps them, a
control thread that serves the driver calls, a sender thread that streams frames, and a watchdog
thread. It does no analysis. A hang in the vendor SDK ends the process, and systemd starts it
again.

- `seeingmon.services.acquire.timing`: `TimeStamper` turns arrival times into frame times.
- `seeingmon.services.acquire.drops`: `DropAccountant` counts lost frames.
- `seeingmon.services.acquire.queue`: `FrameQueue`, the bounded queue that drops its oldest frame.
- `seeingmon.services.acquire.service`: `AcquireService`, the process itself.
"""

from __future__ import annotations
