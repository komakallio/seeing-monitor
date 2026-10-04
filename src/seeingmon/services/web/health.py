"""The health of the system, as `GET /health` and `GET /status` report it.

`evaluate_health` is a pure function. It reads the newest `health` record that `core` wrote, and
it adds what only the web process knows: whether the store opened, and whether `core` answers.

**Levels.** The status is one of three words, and `GET /health` answers 200 for the first two
and 503 for the third, so an external watchdog can poll it.

- `healthy`: the newest health record is fresh, every component is `ok`, and no flag is set.
- `degraded`: the system runs, and something needs a look. A component is `degraded`, the record
  says that the system is degraded, a flag is set (`low_space`, `sink_backlog`, `time_invalid`),
  or `core` does not answer although its last record is fresh.
- `failed`: the store cannot be read, no health record exists, the newest record is older than
  `max_age_s` (so `core` stopped writing), or a component is `failed`. The camera ladder reports
  `failed` when the camera failed repeatedly.

A component state that this module does not know counts as `degraded`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from seeingmon.clock import NS_PER_S

HEALTHY = "healthy"
DEGRADED = "degraded"
FAILED = "failed"
SEVERITY = {HEALTHY: 0, DEGRADED: 1, FAILED: 2}
COMPONENT_LEVEL = {"ok": HEALTHY, "degraded": DEGRADED, "failed": FAILED}


@dataclass(frozen=True, slots=True)
class HealthReport:
    """The verdict and its evidence.

    `age_s` is the age of the newest health record, or `None` when there is none. `quality` says
    why a value is missing, as the API does everywhere. Its entry `components`, which comes from the
    record, says why a component is not `ok`, such as `camera: the camera is not connected`.
    """

    status: str
    reasons: tuple[str, ...]
    components: dict[str, str]
    age_s: float | None
    quality: dict[str, str] | None = None
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def http_status(self) -> int:
        """200 for a healthy or degraded system, and 503 for a failed one."""
        return 503 if self.status == FAILED else 200


def evaluate_health(
    *,
    now_ns: int,
    record: Mapping[str, Any] | None,
    store_ok: bool,
    core_ok: bool | None,
    max_age_s: float,
) -> HealthReport:
    """Judge the system. `record` is the newest health record as `Record.to_row` gives it.

    Pass `core_ok=None` when the caller did not ask `core`.
    """
    components: dict[str, str] = {"web": "ok"}
    if not store_ok:
        components["store"] = "failed"
        return HealthReport(
            FAILED,
            ("store_unreadable",),
            components,
            None,
            {"age_s": "the store cannot be read"},
        )
    if record is None:
        return HealthReport(
            FAILED,
            ("no_health_record",),
            components,
            None,
            {"age_s": "core has not written a health record yet"},
        )
    age_s = max(0.0, (now_ns - int(record["t_utc_ns"])) / NS_PER_S)
    components.update({str(name): str(state) for name, state in record["components"].items()})
    components["web"] = "ok"  # this process answers
    level = HEALTHY
    reasons: list[str] = []

    def raise_to(new: str, reason: str) -> None:
        nonlocal level
        reasons.append(reason)
        if SEVERITY[new] > SEVERITY[level]:
            level = new

    if age_s > max_age_s:
        raise_to(FAILED, "health_stale")
    for name in sorted(components):
        state = COMPONENT_LEVEL.get(components[name], DEGRADED)
        if state != HEALTHY:
            raise_to(state, f"component_{state}:{name}")
    if record["degraded"] and level == HEALTHY:
        raise_to(DEGRADED, "system_degraded")
    flags = tuple(str(flag) for flag in record.get("flags") or ())
    for flag in flags:
        raise_to(DEGRADED, f"flag:{flag}")
    if core_ok is False:
        components["core_link"] = "degraded"
        raise_to(DEGRADED, "core_unreachable")
    elif core_ok is True:
        components["core_link"] = "ok"
    note = (record.get("quality") or {}).get("components")
    quality = {"components": str(note)} if note else None
    return HealthReport(level, tuple(reasons), components, age_s, quality, flags)
