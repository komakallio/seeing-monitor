"""Small builders for the records that the storage tests write."""

from __future__ import annotations

from typing import Any

from seeingmon.records.samples import sample_record
from seeingmon.records.seeing import SeeingWindowRecord
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
