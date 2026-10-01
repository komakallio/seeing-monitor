"""The fast path: per-frame star metrics, windows, and the seeing estimator.

The package turns the frames of the fast stream into per-frame metrics and into one
`seeing_window` record for each window of frame time (60 s by default).
"""

from __future__ import annotations
