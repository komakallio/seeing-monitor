"""The alignment helper of `core`: a live view and a quick solve while the scheduler aligns.

In the `align` state the scheduler streams a bin2 view and hands every frame to the helper. The
helper does two jobs at different speeds, and neither waits for the other:

- **The live view** (`seeingmon.services.core.alignment.preview`, `helper`). Each frame becomes a
  small stretched JPEG, and the helper pushes the newest one to every open live-view stream. It
  never queues: a frame that the encoder has not taken when the next one arrives is dropped, and a
  stream whose window is full skips the frame. The target latency is under 1.5 seconds.
- **The quick solve** (`solve`, `worker`). A solver takes the newest frame as soon as it has
  finished the previous solve: it detects the stars, matches them to the catalog from the latest
  pointing solution, and falls back to the plate solver. The result gives the position of Polaris,
  the roll, the offset from the target, and the focus value (the median FWHM of the unsaturated
  stars). The detector holds the GIL for seconds, so by default the solve runs in a worker process
  (`worker`) and never stalls the live view.

`state` merges both into the `AlignmentState` that `web` reads (see
`seeingmon.services.web.contract`), and it says which frame each part comes from and how old it is
(`TimingView`). Importing this package loads nothing heavy.
"""

from __future__ import annotations
