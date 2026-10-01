"""Recordings: SER files, their sidecars, and the tools that inspect them.

- `seeingmon.recordings.ser` reads and writes SER (version 3) video files.
- `seeingmon.recordings.sidecar` parses SharpCap settings sidecars and reads and writes the
  JSON sidecar of a burst.
- `seeingmon.drivers.replay` feeds a recording to the rest of the system as a camera.

This package imports nothing at load time, so `seeingmon --help` stays fast. Import the
module you need.
"""

from __future__ import annotations
