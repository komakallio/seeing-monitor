"""The verdict of `evaluate_health`: healthy, degraded, or failed."""

from __future__ import annotations

from typing import Any

import pytest

from seeingmon.clock import NS_PER_S
from seeingmon.services.web.health import DEGRADED, FAILED, HEALTHY, evaluate_health
from tests.services.web.seed import NOW_NS

MAX_AGE_S = 180.0


def record(age_s: float = 30.0, **fields: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "t_utc_ns": NOW_NS - round(age_s * NS_PER_S),
        "state": "auto",
        "degraded": False,
        "components": {"acquire": "ok", "core": "ok", "camera": "ok"},
        "flags": [],
    }
    values.update(fields)
    return values


def judge(rec: dict[str, Any] | None, *, store_ok: bool = True, core_ok: bool | None = True) -> Any:
    return evaluate_health(
        now_ns=NOW_NS, record=rec, store_ok=store_ok, core_ok=core_ok, max_age_s=MAX_AGE_S
    )


def test_a_fresh_record_with_every_component_ok_is_healthy() -> None:
    report = judge(record())
    assert report.status == HEALTHY
    assert report.reasons == ()
    assert report.http_status == 200
    assert report.age_s == 30.0
    assert report.components == {
        "web": "ok",
        "acquire": "ok",
        "core": "ok",
        "camera": "ok",
        "core_link": "ok",
    }


def test_the_web_component_is_always_ok_because_the_process_answers() -> None:
    report = judge(record(components={"web": "failed", "core": "ok"}))
    assert report.components["web"] == "ok"
    assert report.status == HEALTHY


def test_a_degraded_component_makes_the_system_degraded_and_the_answer_200() -> None:
    report = judge(record(components={"acquire": "degraded", "core": "ok"}))
    assert report.status == DEGRADED
    assert report.reasons == ("component_degraded:acquire",)
    assert report.http_status == 200


def test_a_failed_component_makes_the_system_failed_and_the_answer_503() -> None:
    report = judge(record(components={"camera": "failed", "core": "ok"}, degraded=True))
    assert report.status == FAILED
    assert report.reasons == ("component_failed:camera",)
    assert report.http_status == 503


def test_the_worst_component_decides() -> None:
    report = judge(record(components={"a": "degraded", "b": "failed", "c": "ok"}))
    assert report.status == FAILED
    assert report.reasons == ("component_degraded:a", "component_failed:b")


def test_an_unknown_component_state_counts_as_degraded() -> None:
    report = judge(record(components={"sensor": "unplugged"}))
    assert report.status == DEGRADED
    assert report.reasons == ("component_degraded:sensor",)


def test_the_degraded_flag_alone_makes_the_system_degraded() -> None:
    report = judge(record(degraded=True))
    assert report.status == DEGRADED
    assert report.reasons == ("system_degraded",)


@pytest.mark.parametrize("flag", ["low_space", "sink_backlog", "time_invalid"])
def test_a_flag_makes_the_system_degraded(flag: str) -> None:
    report = judge(record(flags=[flag]))
    assert report.status == DEGRADED
    assert report.reasons == (f"flag:{flag}",)
    assert report.flags == (flag,)


def test_a_record_older_than_the_limit_means_that_core_stopped_and_the_system_failed() -> None:
    report = judge(record(age_s=MAX_AGE_S + 1))
    assert report.status == FAILED
    assert report.reasons == ("health_stale",)
    assert report.http_status == 503


def test_a_record_exactly_at_the_limit_is_still_fresh() -> None:
    assert judge(record(age_s=MAX_AGE_S)).status == HEALTHY


def test_a_record_from_the_future_has_an_age_of_zero() -> None:
    report = judge(record(age_s=-50))
    assert report.age_s == 0.0
    assert report.status == HEALTHY


def test_no_record_means_failed() -> None:
    report = judge(None)
    assert report.status == FAILED
    assert report.reasons == ("no_health_record",)
    assert report.age_s is None
    assert report.quality == {"age_s": "core has not written a health record yet"}
    assert report.http_status == 503


def test_an_unreadable_store_means_failed() -> None:
    report = judge(record(), store_ok=False)
    assert report.status == FAILED
    assert report.reasons == ("store_unreadable",)
    assert report.components["store"] == "failed"
    assert report.quality == {"age_s": "the store cannot be read"}


def test_a_silent_core_with_a_fresh_record_is_degraded() -> None:
    report = judge(record(), core_ok=False)
    assert report.status == DEGRADED
    assert report.reasons == ("core_unreachable",)
    assert report.components["core_link"] == "degraded"


def test_a_silent_core_with_a_stale_record_is_failed() -> None:
    report = judge(record(age_s=1000), core_ok=False)
    assert report.status == FAILED
    assert set(report.reasons) == {"health_stale", "core_unreachable"}


def test_a_core_that_was_not_asked_leaves_no_link_component() -> None:
    report = judge(record(), core_ok=None)
    assert "core_link" not in report.components
    assert report.status == HEALTHY


def test_the_reasons_name_each_cause_once() -> None:
    report = judge(record(components={"core": "degraded"}, degraded=True, flags=["low_space"]))
    assert report.reasons == ("component_degraded:core", "flag:low_space")


def test_the_note_of_the_record_says_why_a_component_is_not_ok() -> None:
    note = "camera: the camera is not connected"
    report = judge(
        record(
            components={"camera": "failed", "core": "ok"},
            degraded=True,
            quality={"components": note, "queue_depth": "acquire did not answer"},
        )
    )
    assert report.status == FAILED
    assert report.quality == {"components": note}  # only the note of the components passes on


def test_a_record_without_a_note_leaves_the_quality_empty() -> None:
    assert judge(record()).quality is None
    assert judge(record(quality=None)).quality is None
    assert judge(record(quality={"queue_depth": "acquire did not answer"})).quality is None
