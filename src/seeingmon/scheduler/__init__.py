"""The scheduler: one owner of the camera that shares its time between the measurement modes.

See the "Scheduler" section of `docs/architecture.md`. The modules are:

- `config`: the `[scheduler]` table and the observing site.
- `ephemeris`: the Sun's elevation, for the daylight gate and the twilight flag.
- `levels`: the steps of the recovery ladder.
"""

from __future__ import annotations
