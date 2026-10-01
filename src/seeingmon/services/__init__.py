"""The `acquire`, `core`, and `web` processes and the local connections between them.

The package has four parts:

- `seeingmon.services.config` holds the `[services]` configuration section.
- `seeingmon.services.ipc` is the connection layer: authenticated local connections, a typed
  request-response RPC, and a binary stream channel. It sends and receives bytes only, and no
  pickle crosses a process boundary.
- `seeingmon.services.acquire` is the process that owns the camera driver.
- `seeingmon.services.remote` holds `RemoteCameraDriver`, a `CameraDriver` that talks to
  `acquire`, so `core` runs the same scheduler code as the tests do.

Importing this package loads nothing heavy. The commands (`seeingmon acquire`, `seeingmon core`,
and `seeingmon web`) are registered in `seeingmon.services.cli`.
"""

from __future__ import annotations
