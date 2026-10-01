"""The `memory` case: what importing each part of the software costs in resident memory.

Each figure is a fresh Python process that imports one set of modules and then reads its own
resident size. The sets are what a process of the system imports:

| Set | What it loads |
|---|---|
| `interpreter` | Python and the memory reader. This is the floor. |
| `numpy` | NumPy. |
| `core_types` | The frame and record types, with pydantic. |
| `fastpath` | The fast-path analyzer, with SciPy. |
| `store` | The SQLite store and the segment writer. |
| `survey` | The survey analyzer and pipeline, with SEP and pyerfa. |
| `astropy` | `astropy.coordinates` and `astropy.time`, which the tests and some tools use. |
| `web` | FastAPI, uvicorn, and Pillow, which the `web` process loads. |
| `core_imports` | The modules that the `core` process imports. |

The figures answer what astropy and FastAPI cost on a machine with 2 GB. The peak memory of the
workloads comes from the other cases, and `seeingmon perf report --budgets` adds them up.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

from seeingmon.perf.registry import REGISTRY, CaseContext
from seeingmon.perf.report import Measurement
from seeingmon.perf.runner import child_environment

IMPORT_SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("interpreter", ()),
    ("numpy", ("numpy",)),
    ("core_types", ("seeingmon.frames", "seeingmon.records")),
    ("fastpath", ("seeingmon.fastpath.analyzer",)),
    ("store", ("seeingmon.store.db", "seeingmon.store.segments")),
    ("survey", ("seeingmon.survey.analyzer",)),
    ("astropy", ("astropy.coordinates", "astropy.time")),
    ("web", ("fastapi", "uvicorn", "PIL.Image")),
    (
        "core_imports",
        (
            "seeingmon.fastpath.analyzer",
            "seeingmon.store.wiring",
            "seeingmon.scheduler.scheduler",
            "seeingmon.survey.analyzer",
        ),
    ),
)
SMOKE_SETS = ("interpreter", "numpy", "core_types", "web")  # the budgets read `web`

_SCRIPT = """
import importlib, json, sys
missing = []
for name in json.loads(sys.argv[1]):
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
from seeingmon.perf.memory import read_memory
reading = read_memory()
print(json.dumps({
    "rss": None if reading is None else reading.rss_bytes,
    "peak": None if reading is None else reading.peak_rss_bytes,
    "missing": missing,
}))
"""
_TIMEOUT_S = 120.0


@dataclass(frozen=True, slots=True)
class ImportBaseline:
    """The memory of a fresh process after its imports, in bytes, and the modules it lacks."""

    rss: int | None
    peak: int | None
    missing: tuple[str, ...]


def import_baseline(modules: tuple[str, ...]) -> ImportBaseline:
    """Import `modules` in a fresh process, and read its resident size and its peak."""
    completed = subprocess.run(
        [sys.executable, "-c", _SCRIPT, json.dumps(list(modules))],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=child_environment(),
        timeout=_TIMEOUT_S,
        check=True,
    )
    data = json.loads(completed.stdout.strip().splitlines()[-1])
    return ImportBaseline(data["rss"], data["peak"], tuple(data["missing"]))


@REGISTRY.case("memory", summary="Resident memory after importing each part of the software")
def memory(ctx: CaseContext) -> list[Measurement]:
    ctx.mark_baseline()
    measurements: list[Measurement] = []
    for label, modules in IMPORT_SETS:
        if ctx.smoke and label not in SMOKE_SETS:
            continue
        found = import_baseline(modules)
        if found.missing:
            ctx.note(f"{label}: not measured, because {', '.join(found.missing)} is missing.")
            continue
        size = found.rss or found.peak
        if not size:
            ctx.note(f"{label}: not measured, because this platform gives no memory reading.")
            continue
        detail: dict[str, float | int | str] = {"modules": ", ".join(modules) or "none"}
        if found.peak:
            detail["peak_bytes"] = found.peak
        measurements.append(
            Measurement(f"baseline.{label}", "bytes", float(size), None, "memory", detail)
        )
    if not measurements:
        raise RuntimeError("no import set could be measured")
    ctx.note(
        "Each figure is the resident size of a fresh process after it imported the modules of the "
        "set. Windows counts the working set, and Linux counts the resident set."
    )
    return measurements
