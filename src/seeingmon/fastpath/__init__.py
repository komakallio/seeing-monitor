"""The fast path: per-frame star metrics, windows, and the seeing estimator.

The package turns the frames of the fast stream into per-frame metrics and into one
`seeing_window` record for each window of frame time (60 s by default). Start with
`create_fast_analyzer`, which builds the `FastPathAnalyzer` that the scheduler feeds, and read
`FastPathConfig` for the settings. The modules, in the order that the data flows through them:

- `kernel`: background, centroid, widths, flux, and flags of one frame (or a stack).
- `windows`: groups frames into windows of frame time and counts the frames that a window lost.
- `estimator` and `models`: the image-motion variance, `r0`, and the seeing, with the
  corrections for outer scale, exposure, detrending, and the finite aperture of the centroid.
- `spectrum` and `scintillation`: the Welch spectrum, the vibration lines, and the scintillation.
- `analyzer`: ties them together behind the `FastAnalyzer` interface.
- `benchmark`: microseconds per frame (`python -m seeingmon.fastpath.benchmark`).

The kernel needs only NumPy. The seeing models need SciPy (the `fast` extra) once, when they build
a table of corrections.
"""

from __future__ import annotations

from seeingmon.fastpath.analyzer import ALGORITHM_REVISION, FastPathAnalyzer, create_fast_analyzer
from seeingmon.fastpath.config import FastPathConfig

__all__ = ["ALGORITHM_REVISION", "FastPathAnalyzer", "FastPathConfig", "create_fast_analyzer"]
