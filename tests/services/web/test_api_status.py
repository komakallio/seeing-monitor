"""`/status` and `/health`: the state of every component, and the 200 and 503 verdicts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.records.samples import sample_record
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.contract import (
    ActivityView,
    CoreStatus,
    FaultView,
    RoiView,
    StreamView,
)
from seeingmon.services.web.core_client import CoreUnavailableError, FakeCoreClient
from seeingmon.store.db import Store, StoreReader
from tests.services.web.client import TestClient
from tests.services.web.seed import NOW_NS, common

API = "/api/v1"
WINDOWS_ERROR = "Cannot open C:\\Users\\someone\\data\\db.sqlite"  # repo-check: allow


class PrivateCore(FakeCoreClient):
    """A fake core whose status carries what the API must not pass on."""

    def status(self) -> CoreStatus:
        status = super().status()
        scheduler = status.scheduler.model_copy(
            update={
                "sun_elevation_deg": -25.5,
                "state_reason": "The sky is dark at 192.0.2.50",
                "fault": FaultView(
                    failures=2,
                    good_frames=1,
                    last_error=WINDOWS_ERROR,
                    next_attempt_utc_ns=NOW_NS + 30 * NS_PER_S,
                    next_step="reopen",
                ),
            }
        )
        return status.model_copy(update={"scheduler": scheduler})


def write_health(writer: Store, age_s: float, **fields: Any) -> None:
    values: dict[str, Any] = {
        "state": "auto",
        "degraded": False,
        "components": {"acquire": "ok", "core": "ok"},
        "dark_due": False,
    }
    values.update(fields)
    writer.write(
        sample_record("health", **common(NOW_NS - round(age_s * NS_PER_S) - 1000, **values))
    )


# --- /status ---------------------------------------------------------------------------------


def test_the_status_has_the_documented_shape(client: TestClient) -> None:
    response = client.get(f"{API}/status")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "now",
        "api_version",
        "software_version",
        "station_id",
        "demo",
        "health",
        "core",
        "scheduler",
        "data",
        "ui",
        "quality",
    }
    assert body["now"] == "2026-10-01T03:00:00.000000Z"
    assert body["api_version"] == "1.0.0"
    assert body["station_id"] == "test-station"
    assert body["demo"] is False
    assert body["quality"] is None


def test_the_status_reports_the_scheduler_and_the_link_to_core(client: TestClient) -> None:
    body = client.get(f"{API}/status").json()
    assert body["core"] == {"reachable": True, "instance": "fake-core"}
    scheduler = body["scheduler"]
    assert scheduler["state"] == "auto"
    assert scheduler["t_utc"] == "2026-10-01T03:00:00.000000Z"
    assert scheduler["degraded"] is False
    assert scheduler["stream"]["purpose"] == "fast"
    assert scheduler["queued_tasks"] == 0
    assert scheduler["fault"]["failures"] == 0


def test_the_status_gives_the_age_of_the_newest_record_of_each_type(client: TestClient) -> None:
    data = client.get(f"{API}/status").json()["data"]
    assert set(data) == {"seeing_window", "sky_quality", "pointing", "health", "event"}
    assert data["health"] == {"t_utc": "2026-10-01T02:59:30.000000Z", "age_s": 30.0}
    assert data["seeing_window"]["t_utc"] == "2026-10-01T02:21:00.000000Z"
    assert data["seeing_window"]["age_s"] == 39 * 60.0
    assert data["event"]["age_s"] == (60 - 4) * 60.0


def test_the_status_of_an_empty_store_has_null_times(empty_client: TestClient) -> None:
    data = empty_client.get(f"{API}/status").json()["data"]
    assert data["seeing_window"] == {"t_utc": None, "age_s": None}


def test_the_status_tells_the_ui_what_it_needs(client: TestClient) -> None:
    assert client.get(f"{API}/status").json()["ui"] == {
        "refresh_s": 10.0,
        "token_required_for_reads": False,
        "commands_enabled": True,
        "alignment_max_fps": 30.0,
        "alignment_stall_s": 0.3,
    }


def test_the_status_says_that_commands_are_off_without_a_token_hash(
    make_app: Callable[..., FastAPI], open_client: Callable[..., TestClient], seeded: Store
) -> None:
    client = open_client(make_app(token_hash=None))
    assert client.get(f"{API}/status").json()["ui"]["commands_enabled"] is False


def test_the_status_follows_the_clock(client: TestClient, clock: VirtualClock) -> None:
    clock.advance(120)
    body = client.get(f"{API}/status").json()
    assert body["now"] == "2026-10-01T03:02:00.000000Z"
    assert body["data"]["health"]["age_s"] == 150.0


def test_a_silent_core_leaves_the_scheduler_null_with_a_reason(
    client: TestClient, core: FakeCoreClient
) -> None:
    core.fail_with = CoreUnavailableError("core does not answer")
    body = client.get(f"{API}/status").json()
    assert body["core"] == {"reachable": False, "instance": None}
    assert body["scheduler"] is None
    assert body["quality"] == {"scheduler": "core does not answer"}
    assert body["health"]["status"] == "degraded"
    assert "core_unreachable" in body["health"]["reasons"]


def test_the_status_leaves_out_the_sun_and_cleans_the_text_of_core(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
) -> None:
    client = open_client(make_app(core=PrivateCore(clock=clock)))
    response = client.get(f"{API}/status")
    text = response.text
    assert "sun_elevation" not in text
    assert "25.5" not in text
    assert "someone" not in text
    assert "192.0.2.50" not in text
    fault = response.json()["scheduler"]["fault"]
    assert fault["last_error"] == "Cannot open <hidden>"
    assert fault["next_attempt"] == "2026-10-01T03:00:30.000000Z"
    assert fault["next_step"] == "reopen"
    assert response.json()["scheduler"]["state_reason"] == "The sky is dark at <hidden>"


def test_the_status_has_no_activity_when_core_gives_none(client: TestClient) -> None:
    assert client.get(f"{API}/status").json()["scheduler"]["activity"] is None


def test_the_status_carries_the_activity_with_iso_times(
    client: TestClient, core: FakeCoreClient
) -> None:
    core.activity = ActivityView(
        state="auto",
        phase="fast",
        label="Fast stream: seeing windows",
        since_utc_ns=NOW_NS - 20 * NS_PER_S,
        ends_utc_ns=NOW_NS + 120 * NS_PER_S,
        next_label="Survey step: a 1 ms and a 30 s frame",
        next_utc_ns=NOW_NS + 121 * NS_PER_S,
        cadence_s=180.0,
        detail="Windows of 20 s: 4 of 7 closed",
        reason="the sky is dark enough",
    )
    activity = client.get(f"{API}/status").json()["scheduler"]["activity"]
    assert activity == {
        "state": "auto",
        "phase": "fast",
        "label": "Fast stream: seeing windows",
        "since_utc": "2026-10-01T02:59:40.000000Z",
        "ends_utc": "2026-10-01T03:02:00.000000Z",
        "next_label": "Survey step: a 1 ms and a 30 s frame",
        "next_utc": "2026-10-01T03:02:01.000000Z",
        "cadence_s": 180.0,
        "detail": "Windows of 20 s: 4 of 7 closed",
        "reason": "the sky is dark enough",
    }


def test_an_activity_without_the_optional_values_has_them_null(
    client: TestClient, core: FakeCoreClient
) -> None:
    core.activity = ActivityView(
        state="paused", phase="paused", label="Paused: nothing runs", since_utc_ns=NOW_NS
    )
    activity = client.get(f"{API}/status").json()["scheduler"]["activity"]
    assert activity["ends_utc"] is None
    assert activity["next_label"] is None
    assert activity["next_utc"] is None
    assert activity["cadence_s"] is None
    assert activity["detail"] is None
    assert activity["reason"] is None


def test_the_activity_cleans_the_text_that_may_name_a_private_place(
    client: TestClient, core: FakeCoreClient
) -> None:
    core.activity = ActivityView(
        state="safe",
        phase="camera_fault",
        label="Camera fault: nothing answers at \\\\.\\pipe\\private-name",
        since_utc_ns=NOW_NS,
        next_label="Recovery step: reopen the camera at 192.0.2.50",
        detail=WINDOWS_ERROR,
        reason="cannot reach http://host.example/path",
    )
    response = client.get(f"{API}/status")
    activity = response.json()["scheduler"]["activity"]
    assert activity["label"] == "Camera fault: nothing answers at <hidden>"
    assert activity["next_label"] == "Recovery step: reopen the camera at <hidden>"
    assert activity["detail"] == "Cannot open <hidden>"
    assert activity["reason"] == "cannot reach <hidden>"
    for private in ("someone", "192.0.2.50", "host.example", "private-name"):
        assert private not in response.text


def test_the_status_reports_a_stream_with_its_roi(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
) -> None:
    class Streaming(FakeCoreClient):
        def status(self) -> CoreStatus:
            status = super().status()
            stream = StreamView(
                stream_id=3,
                purpose="fast",
                mode="bin1",
                exposure_us=2000,
                gain=0,
                roi=RoiView(x=1, y=2, width=128, height=64),
            )
            scheduler = status.scheduler.model_copy(update={"stream": stream})
            return status.model_copy(update={"scheduler": scheduler})

    client = open_client(make_app(core=Streaming(clock=clock)))
    stream = client.get(f"{API}/status").json()["scheduler"]["stream"]
    assert stream["roi"] == {"x": 1, "y": 2, "width": 128, "height": 64}


def test_the_status_never_carries_the_scheduler_fields_that_it_does_not_name(
    client: TestClient,
) -> None:
    scheduler = client.get(f"{API}/status").json()["scheduler"]
    assert set(scheduler) == {
        "t_utc",
        "state",
        "state_reason",
        "state_since",
        "last_transition",
        "degraded",
        "stream",
        "cloud",
        "cloud_fraction",
        "twilight",
        "background_fraction",
        "sensor_temperature_c",
        "counters",
        "fault",
        "queued_tasks",
        "survey_pending",
        "alignment_idle_s",
        "activity",
    }


# --- /health ---------------------------------------------------------------------------------


def test_a_healthy_system_answers_200(client: TestClient) -> None:
    response = client.get(f"{API}/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "now": "2026-10-01T03:00:00.000000Z",
        "age_s": 30.0,
        "reasons": [],
        "components": {
            "web": "ok",
            "acquire": "ok",
            "core": "ok",
            "scheduler": "ok",
            "camera": "ok",
            "core_link": "ok",
        },
        "flags": [],
        "quality": None,
    }
    assert response.headers["cache-control"] == "no-store"


def test_a_degraded_system_still_answers_200(
    client: TestClient, seeded: Store, clock: VirtualClock
) -> None:
    write_health(seeded, 5, components={"acquire": "degraded", "core": "ok"}, flags=["low_space"])
    body = client.get(f"{API}/health")
    assert body.status_code == 200
    assert body.json()["status"] == "degraded"
    assert set(body.json()["reasons"]) == {"component_degraded:acquire", "flag:low_space"}


def test_a_failed_component_answers_503(client: TestClient, seeded: Store) -> None:
    write_health(seeded, 5, components={"acquire": "ok", "camera": "failed"}, degraded=True)
    response = client.get(f"{API}/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "failed"
    assert body["reasons"] == ["component_failed:camera"]
    assert body["components"]["camera"] == "failed"


def test_a_record_that_gets_too_old_turns_the_system_failed(
    client: TestClient, clock: VirtualClock
) -> None:
    assert client.get(f"{API}/health").status_code == 200
    clock.advance(150)  # the record is 180 s old, which is the limit
    assert client.get(f"{API}/health").status_code == 200
    clock.advance(1)
    response = client.get(f"{API}/health")
    assert response.status_code == 503
    assert response.json()["reasons"] == ["health_stale"]
    assert response.json()["age_s"] == 181.0


def test_a_store_without_a_health_record_answers_503(empty_client: TestClient) -> None:
    response = empty_client.get(f"{API}/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "failed"
    assert body["reasons"] == ["no_health_record"]
    assert body["age_s"] is None
    assert body["quality"] == {"age_s": "core has not written a health record yet"}


def test_an_unreadable_store_answers_503(client: TestClient, reader: StoreReader) -> None:
    reader.close()
    response = client.get(f"{API}/health")
    assert response.status_code == 503
    assert response.json()["reasons"] == ["store_unreadable"]
    assert response.json()["components"]["store"] == "failed"


def test_a_silent_core_with_a_fresh_record_answers_200_degraded(
    client: TestClient, core: FakeCoreClient
) -> None:
    core.fail_with = CoreUnavailableError("core does not answer")
    response = client.get(f"{API}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["components"]["core_link"] == "degraded"


def test_the_limit_of_the_age_comes_from_the_settings(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
) -> None:
    strict = WebSettings.model_validate({"health_max_age_s": 10.0})
    client = open_client(make_app(settings=strict))
    assert client.get(f"{API}/health").status_code == 503  # the record is 30 s old


def test_health_needs_no_token_by_default(client: TestClient) -> None:
    assert client.get(f"{API}/health").status_code == 200


def test_the_health_answer_has_a_json_body_even_when_it_is_503(
    client: TestClient, reader: StoreReader
) -> None:
    reader.close()
    response = client.get(f"{API}/health")
    assert response.headers["content-type"] == "application/json"
    assert set(response.json()) == {
        "status",
        "now",
        "age_s",
        "reasons",
        "components",
        "flags",
        "quality",
    }
