"""The report: what a run measured, as plain data that survives a round trip through JSON.

A `Report` holds the environment facts and one `CaseResult` for each case. A case result holds
`Measurement` figures, each with a unit and, for a timed quantity, the statistics of its repeats.
The `schema_version` changes whenever a reader of an old file could misread it, and `read_report`
refuses a version that it does not know.

A figure carries a `scale`, the class of code that it times. The estimate for a Raspberry Pi 4
multiplies the figure by the range of its class (see `seeingmon.perf.scaling`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from seeingmon.perf.environment import Environment
from seeingmon.perf.timing import TimingStats

SCHEMA_VERSION = 1

Status = Literal["ok", "skipped", "failed"]
STATUSES: tuple[Status, ...] = ("ok", "skipped", "failed")

# The classes of code that the Raspberry Pi 4 estimate scales differently.
SCALES = ("none", "numpy", "interpreter", "scheduler", "memory")

DetailValue = float | int | str


class ReportError(ValueError):
    """A report file that this version of the harness cannot read."""


@dataclass(frozen=True, slots=True)
class Measurement:
    """One figure of a case.

    `value` is the headline number, in `unit`. For a timed quantity it is the median of the
    repeats, and `stats` holds the whole summary in the same unit. `scale` names the class of
    code for the Raspberry Pi 4 estimate. `detail` holds the facts that explain the figure, such
    as the number of frames.
    """

    name: str
    unit: str
    value: float
    stats: TimingStats | None = None
    scale: str = "none"
    detail: dict[str, DetailValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("a measurement needs a name and a unit")
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError(f"the measurement {self.name!r} is not a finite, non-negative number")
        if self.scale not in SCALES:
            raise ValueError(f"the scale of {self.name!r} must be one of {', '.join(SCALES)}")
        for key, item in self.detail.items():
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"the detail {key!r} of {self.name!r} is not finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "value": self.value,
            "stats": None if self.stats is None else self.stats.to_dict(),
            "scale": self.scale,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Measurement:
        try:
            stats = data.get("stats")
            return cls(
                name=str(data["name"]),
                unit=str(data["unit"]),
                value=float(data["value"]),
                stats=None if stats is None else TimingStats.from_dict(stats),
                scale=str(data.get("scale", "none")),
                detail=dict(data.get("detail") or {}),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"a measurement is incomplete or malformed ({error})") from None


@dataclass(frozen=True, slots=True)
class CaseResult:
    """The outcome of one case.

    `status` is `ok`, `skipped` (the case could not run, and `reason` says why), or `failed` (the
    case raised, and `reason` holds the error). `baseline_rss_bytes` is the resident size after the
    case imported its code and before it did its work, and `peak_rss_bytes` is the peak of the
    process that ran the case. `system_busy_percent` is how busy the machine was just before the
    case started.
    """

    name: str
    status: Status
    reason: str | None = None
    duration_s: float = 0.0
    cpu_s: float | None = None
    baseline_rss_bytes: int | None = None
    peak_rss_bytes: int | None = None
    system_busy_percent: float | None = None
    measurements: tuple[Measurement, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"the status of {self.name!r} must be one of {', '.join(STATUSES)}")
        if self.status != "ok" and not self.reason:
            raise ValueError(f"the case {self.name!r} is {self.status} and needs a reason")

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def measurement(self, name: str) -> Measurement | None:
        """The figure with this name, or `None`."""
        for item in self.measurements:
            if item.name == name:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "duration_s": self.duration_s,
            "cpu_s": self.cpu_s,
            "baseline_rss_bytes": self.baseline_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "system_busy_percent": self.system_busy_percent,
            "measurements": [item.to_dict() for item in self.measurements],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CaseResult:
        try:
            return cls(
                name=str(data["name"]),
                status=_status(data["status"]),
                reason=None if data.get("reason") is None else str(data["reason"]),
                duration_s=float(data.get("duration_s", 0.0)),
                cpu_s=None if data.get("cpu_s") is None else float(data["cpu_s"]),
                baseline_rss_bytes=_optional_int(data.get("baseline_rss_bytes")),
                peak_rss_bytes=_optional_int(data.get("peak_rss_bytes")),
                system_busy_percent=(
                    None
                    if data.get("system_busy_percent") is None
                    else float(data["system_busy_percent"])
                ),
                measurements=tuple(Measurement.from_dict(item) for item in data["measurements"]),
                notes=tuple(str(note) for note in data.get("notes", ())),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"a case result is incomplete or malformed ({error})") from None


def _status(value: object) -> Status:
    for status in STATUSES:
        if value == status:
            return status
    raise ValueError(f"unknown case status {value!r}")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("a byte count must be a number")
    return int(value)


@dataclass(frozen=True, slots=True)
class Report:
    """A run of the harness: its label, the environment, and the results of the cases.

    `label` names the machine class of the run, such as `dev` or `pi4`. `smoke` is true for a run
    with the tiny sizes of `--smoke`, whose figures say nothing about speed.
    """

    label: str
    smoke: bool
    created_utc: str
    environment: Environment
    cases: tuple[CaseResult, ...]
    schema_version: int = SCHEMA_VERSION

    def case(self, name: str) -> CaseResult | None:
        """The result of the case with this name, or `None`."""
        for result in self.cases:
            if result.name == name:
                return result
        return None

    def measurement(self, case: str, name: str) -> Measurement | None:
        """The figure `name` of the case `case`, or `None` when the case or the figure is absent."""
        result = self.case(case)
        return None if result is None or not result.ok else result.measurement(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "smoke": self.smoke,
            "created_utc": self.created_utc,
            "environment": self.environment.to_dict(),
            "cases": [result.to_dict() for result in self.cases],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Report:
        """Read a report from `to_dict`. Raises `ReportError` for an unknown schema version or a
        malformed report."""
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ReportError(
                f"the report has schema version {version!r}, and this software reads version "
                f"{SCHEMA_VERSION}"
            )
        try:
            return cls(
                label=str(data["label"]),
                smoke=bool(data["smoke"]),
                created_utc=str(data["created_utc"]),
                environment=Environment.from_dict(dict(data["environment"])),
                cases=tuple(CaseResult.from_dict(item) for item in data["cases"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ReportError(f"the report is incomplete or malformed ({error})") from None


def write_report(path: Path | str, report: Report) -> Path:
    """Write a report as JSON. Creates the missing folders, and returns the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report.to_dict(), indent=2, allow_nan=False)
    target.write_text(text + "\n", encoding="utf-8", newline="\n")
    return target


def read_report(path: Path | str) -> Report:
    """Read a report from a JSON file. Raises `ReportError` when the file is not a report."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ReportError(f"cannot read the report: {error.strerror or error}") from None
    except ValueError as error:
        raise ReportError(f"the file is not valid JSON ({error})") from None
    if not isinstance(data, dict):
        raise ReportError("the file does not hold a report")
    return Report.from_dict(data)
