"""The alignment helper of `core`: a live view and a quick solve while the scheduler aligns.

In the `align` state the scheduler streams a bin2 view and hands every frame to the helper. The
helper does two jobs at different speeds:

- **The live view** (`seeingmon.services.core.alignment.preview`, `helper`). Each frame becomes a
  small stretched JPEG, and the helper pushes the newest one to every open live-view stream. It
  never queues: a frame that the encoder has not taken when the next one arrives is dropped, and a
  stream whose window is full skips the frame. The target latency is under 1.5 seconds.
- **The quick solve** (`solve`). About once a second, a worker solves the newest frame: it detects
  the stars, matches them to the catalog from the latest pointing solution, and falls back to the
  plate solver. The result gives the position of Polaris, the roll, the offset from the target, and
  the focus value (the median FWHM of the unsaturated stars).

`state` merges both into the `AlignmentState` that `web` reads (see
`seeingmon.services.web.contract`). Importing this package loads nothing heavy.
"""

from __future__ import annotations
