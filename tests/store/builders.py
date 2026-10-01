"""Small builders for the records that the storage tests write."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from seeingmon.records.samples import sample_record
from seeingmon.records.seeing import SeeingWindowRecord
from seeingmon.records.segments import segment_dtype
from seeingmon.records.survey import StarListRecord
from seeingmon.records.system import EventRecord, HealthRecord

NS_PER_S = 1_000_000_000
NS_PER_DAY = 86_400 * NS_PER_S
T0 = 1_767_225_600 * NS_PER_S  # 2026-01-01T00:00:00Z
STATION = "station-1"


def make_event(t_utc_ns: int = T0, message: str = "An event.", **overrides: Any) -> EventRecord:
    return sample_record(EventRecord, t_utc_ns=t_utc_ns, message=message, **overrides)


def make_health(t_utc_ns: int = T0, **overrides: Any) -> HealthRecord:
    return sample_record(HealthRecord, t_utc_ns=t_utc_ns, **overrides)


def make_window(t_utc_ns: int = T0, **overrides: Any) -> SeeingWindowRecord:
    return sample_record(SeeingWindowRecord, t_utc_ns=t_utc_ns, **overrides)


def make_star_list(t_utc_ns: int = T0, **overrides: Any) -> StarListRecord:
    return sample_record(StarListRecord, t_utc_ns=t_utc_ns, **overrides)


FRAME_DTYPE = segment_dtype("frame")


def make_rows(
    t_utc_ns: int,
    count: int,
    *,
    step_ns: int = NS_PER_S,
    seq0: int = 0,
    seed: int = 1,
) -> npt.NDArray[Any]:
    """Build `count` rows of the `frame` segment layout, `step_ns` apart, with seeded values."""
    rng = np.random.default_rng(seed)
    rows = np.zeros(count, dtype=FRAME_DTYPE)
    rows["t_utc_ns"] = t_utc_ns + np.arange(count, dtype=np.int64) * step_ns
    rows["seq"] = np.arange(seq0, seq0 + count)
    rows["t_err_us"] = 5
    for name in ("cx_px", "cy_px", "width_x_px", "width_y_px", "flux_e", "bg_dn"):
        rows[name] = rng.uniform(1.0, 100.0, count).astype(np.float32)
    rows["peak_dn"] = rng.integers(1, 60_000, count)
    rows["flags"] = rng.integers(0, 4, count)
    rows["dropped_before"] = rng.integers(0, 3, count)
    return rows
