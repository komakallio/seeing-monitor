"""The OpenAPI description: it is current, it is consistent, and the live answers match it."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from seeingmon.clock import VirtualClock
from seeingmon.records.api_schema import api_schema, schema_name
from seeingmon.scheduler.commands import QueueDark, QueueFlat
from seeingmon.services.web.api import router
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.contract import DarkModelView, PolarisFrame, PolarisState
from seeingmon.services.web.core_client import FakeCoreClient
from seeingmon.services.web.fake_dark import DarkScript
from seeingmon.services.web.fake_flat import FlatScript
from seeingmon.services.web.openapi import COMMAND, render_openapi
from seeingmon.services.web.schemas import SERVED_RECORD_TYPES
from seeingmon.store.db import Store
from tests.records.jsonschema_lite import InvalidError, validate
from tests.services.conftest import wait_until
from tests.services.web.client import TestClient
from tests.services.web.helpers import bearer, dark_set, live_seeing_view, polaris_state, tiny_png
from tests.services.web.seed import write_fits, write_preview

API = "/api/v1"
METHODS = ("get", "post", "put", "patch", "delete")


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(render_openapi())
    return parsed


def operations(document: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (method, path, operation)
        for path, item in document["paths"].items()
        for method, operation in item.items()
        if method in METHODS
    ]


def references(value: Any) -> list[str]:
    """Every `$ref` target in a JSON value."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$ref" and isinstance(item, str):
                found.append(item)
            else:
                found += references(item)
    elif isinstance(value, list):
        for item in value:
            found += references(item)
    return found


# --- The committed file ----------------------------------------------------------------------


def test_the_committed_description_is_current(repo_root: Path) -> None:
    committed = (repo_root / "docs" / "openapi.json").read_text(encoding="utf-8")
    assert committed == render_openapi(), f"docs/openapi.json is stale. Run `{COMMAND}`."


def test_the_description_is_stable_between_two_renderings() -> None:
    assert render_openapi() == render_openapi()


def test_the_committed_file_ends_with_one_newline_and_uses_unix_line_ends(repo_root: Path) -> None:
    raw = (repo_root / "docs" / "openapi.json").read_bytes()
    assert raw.endswith(b"}\n")
    assert b"\r" not in raw


# --- The structure ---------------------------------------------------------------------------


def test_the_description_is_openapi_3_1(document: dict[str, Any]) -> None:
    assert document["openapi"].startswith("3.1")
    assert document["info"]["title"] == "Seeing monitor API"
    assert document["info"]["version"] == "1.0.0"


def test_every_reference_resolves(document: dict[str, Any]) -> None:
    schemas = document["components"]["schemas"]
    for target in references(document):
        assert target.startswith("#/components/schemas/"), target
        assert target.removeprefix("#/components/schemas/") in schemas, target


def test_the_default_validation_error_of_fastapi_is_not_in_the_description(
    document: dict[str, Any],
) -> None:
    assert "HTTPValidationError" not in document["components"]["schemas"]
    assert "ValidationError" not in document["components"]["schemas"]
    assert "HTTPValidationError" not in json.dumps(document)


def test_every_operation_has_an_id_a_summary_a_description_and_a_tag(
    document: dict[str, Any],
) -> None:
    ids = []
    for method, path, operation in operations(document):
        where = f"{method.upper()} {path}"
        ids.append(operation["operationId"])
        assert operation["summary"], where
        assert len(operation.get("description", "")) > 20, where
        assert operation["tags"], where
        assert re.fullmatch(r"[a-z][a-z0-9_]*", operation["operationId"]), where
    assert len(ids) == len(set(ids))


def test_every_post_names_the_token_and_the_documented_failures(document: dict[str, Any]) -> None:
    posts = [(path, op) for method, path, op in operations(document) if method == "post"]
    assert len(posts) == 13
    for path, operation in posts:
        assert operation["security"] == [{"bearerAuth": []}], path
        # The reset of the best focus value takes no body and is no scheduler command.
        required = {"200", "401", "403", "429"}
        if not path.endswith("/alignment/focus/reset"):
            required |= {"409", "413", "422"}
        assert required <= set(operation["responses"]), path
    assert document["components"]["securitySchemes"]["bearerAuth"]["scheme"] == "bearer"


def test_every_delete_names_the_token_and_the_documented_failures(document: dict[str, Any]) -> None:
    deletes = [(path, op) for method, path, op in operations(document) if method == "delete"]
    assert [path for path, _ in deletes] == [f"{API}/flat/{{version}}"]
    for path, operation in deletes:
        assert operation["security"] == [{"bearerAuth": []}], path
        assert {"200", "401", "403", "404", "409", "422", "429"} <= set(operation["responses"]), (
            path
        )


def test_a_read_route_does_not_claim_a_token(document: dict[str, Any]) -> None:
    for method, path, operation in operations(document):
        if method == "get":
            assert "security" not in operation, path


def test_every_error_response_uses_the_one_error_shape(document: dict[str, Any]) -> None:
    for _method, path, operation in operations(document):
        for status, response in operation["responses"].items():
            if status.startswith(("4", "5")) and status not in {"409", "503"}:
                schema = response["content"]["application/json"]["schema"]
                assert schema == {"$ref": "#/components/schemas/ErrorResponse"}, (path, status)


def test_every_route_of_the_app_is_described_and_every_description_has_a_route(
    app: Any, document: dict[str, Any]
) -> None:
    routes = {
        (method.lower(), route.path)
        for route in router.routes
        if isinstance(route, APIRoute)
        for method in route.methods or ()
    }
    described = {(method, path) for method, path, _ in operations(document)}
    assert described == routes


def test_the_documented_endpoints_are_the_ones_of_the_architecture(
    document: dict[str, Any],
) -> None:
    paths = set(document["paths"])
    required = {
        "/status",
        "/health",
        "/seeing",
        "/seeing/latest",
        "/sky",
        "/sky/latest",
        "/pointing",
        "/pointing/latest",
        "/images/latest",
        "/images/{image_id}",
        "/events",
        "/profile",
        "/config",
        "/commands/burst",
        "/commands/sweep",
        "/commands/replay",
        "/mode",
        "/alignment/start",
        "/alignment/stop",
        "/alignment/state",
        "/alignment/focus/reset",
        "/alignment/rapid-focus/start",
        "/alignment/rapid-focus/stop",
        "/seeing/live",
        "/polaris/frame",
        "/dark",
        "/commands/dark",
        "/flat",
        "/flat/session",
        "/flat/session/stop",
        "/flat/{version}/activate",
        "/flat/{version}",
        "/flat/{version}/image",
    }
    assert {f"{API}{path}" for path in required} <= paths


def test_the_history_routes_document_their_parameters(document: dict[str, Any]) -> None:
    for path in ("seeing", "sky", "pointing"):
        parameters = {p["name"]: p for p in document["paths"][f"{API}/{path}"]["get"]["parameters"]}
        assert set(parameters) == {"from", "to", "step", "limit", "cursor", "fields"}
        assert parameters["limit"]["schema"]["anyOf"][0]["maximum"] == 10000
        step = document["components"]["schemas"]["Step"]
        assert step["enum"] == ["raw", "1m", "10m", "1h"]


# --- The records come from the declarations --------------------------------------------------


@pytest.mark.parametrize("record_type", SERVED_RECORD_TYPES)
def test_a_record_schema_is_the_generated_one_plus_the_iso_time(
    document: dict[str, Any], record_type: str
) -> None:
    generated = api_schema([record_type])["components"]["schemas"][schema_name(record_type)]
    served = document["components"]["schemas"][schema_name(record_type)]
    assert set(served["properties"]) == set(generated["properties"]) | {"t_utc"}
    for name, schema in generated["properties"].items():
        assert served["properties"][name] == schema
    assert served["required"] == [*generated["required"], "t_utc"]
    assert served["x-record-type"] == record_type
    assert served["properties"]["t_utc"]["format"] == "date-time"


@pytest.mark.parametrize("record_type", SERVED_RECORD_TYPES)
def test_a_history_row_makes_every_field_optional_and_adds_the_sample_count(
    document: dict[str, Any], record_type: str
) -> None:
    row = document["components"]["schemas"][f"{schema_name(record_type)}Row"]
    record = document["components"]["schemas"][schema_name(record_type)]
    assert row["required"] == ["t_utc_ns", "t_utc"]
    assert set(row["properties"]) == set(record["properties"]) | {"n_samples"}


@pytest.mark.parametrize("component", ["Pointing", "PointingRow"])
def test_the_pointing_schemas_document_the_pole_pixel(
    document: dict[str, Any], component: str
) -> None:
    properties = document["components"]["schemas"][component]["properties"]
    for name in ("pole_x_px", "pole_y_px"):
        assert properties[name]["type"] == ["number", "null"]
        assert properties[name]["x-unit"] == "px"
        assert "celestial pole" in properties[name]["description"]
    assert document["components"]["schemas"]["Pointing"]["required"][-3:] == [
        "pole_x_px",
        "pole_y_px",
        "t_utc",
    ]


def test_the_quality_schema_is_the_generated_one(document: dict[str, Any]) -> None:
    generated = api_schema(["seeing_window"])["components"]["schemas"]["Quality"]
    assert document["components"]["schemas"]["Quality"] == generated


# --- The live answers match the description --------------------------------------------------


def documented_schema(
    document: dict[str, Any], method: str, path: str, status: int
) -> dict[str, Any]:
    response = document["paths"][path][method]["responses"][str(status)]
    return response["content"]["application/json"]["schema"]  # type: ignore[no-any-return]


CASES: list[tuple[str, str, str, int, dict[str, Any]]] = [
    # (method, documented path, request path, status, request options)
    ("get", "/api/v1", "/api/v1", 200, {}),
    ("get", "/api/v1/status", "/api/v1/status", 200, {}),
    ("get", "/api/v1/health", "/api/v1/health", 200, {}),
    ("get", "/api/v1/seeing/latest", "/api/v1/seeing/latest", 200, {}),
    ("get", "/api/v1/sky/latest", "/api/v1/sky/latest", 200, {}),
    ("get", "/api/v1/pointing/latest", "/api/v1/pointing/latest", 200, {}),
    ("get", "/api/v1/seeing", "/api/v1/seeing", 200, {}),
    ("get", "/api/v1/seeing", "/api/v1/seeing", 200, {"params": {"step": "10m"}}),
    ("get", "/api/v1/seeing", "/api/v1/seeing", 200, {"params": {"fields": "r0_cm"}}),
    ("get", "/api/v1/sky", "/api/v1/sky", 200, {"params": {"step": "1h"}}),
    ("get", "/api/v1/pointing", "/api/v1/pointing", 200, {"params": {"step": "1m"}}),
    ("get", "/api/v1/events", "/api/v1/events", 200, {}),
    ("get", "/api/v1/images", "/api/v1/images", 200, {}),
    ("get", "/api/v1/alignment/state", "/api/v1/alignment/state", 200, {}),
    ("get", "/api/v1/seeing/live", "/api/v1/seeing/live", 404, {}),
    ("get", "/api/v1/dark", "/api/v1/dark", 200, {}),
    ("get", "/api/v1/flat", "/api/v1/flat", 200, {}),
    ("get", "/api/v1/profile", "/api/v1/profile", 200, {}),
    ("get", "/api/v1/config", "/api/v1/config", 200, {}),
    ("post", "/api/v1/commands/burst", "/api/v1/commands/burst", 200, {"json": {}}),
    ("post", "/api/v1/alignment/stop", "/api/v1/alignment/stop", 409, {}),
    ("post", "/api/v1/commands/dark", "/api/v1/commands/dark", 200, {"json": {}}),
    ("post", "/api/v1/commands/dark", "/api/v1/commands/dark", 422, {"json": {"frames": 2}}),
    ("post", "/api/v1/flat/session", "/api/v1/flat/session", 422, {"json": {}}),  # no dark set
    ("post", "/api/v1/flat/session", "/api/v1/flat/session", 422, {"json": {"frames": 7}}),
    ("post", "/api/v1/flat/session/stop", "/api/v1/flat/session/stop", 409, {}),
    ("post", "/api/v1/flat/{version}/activate", "/api/v1/flat/flat-00000000/activate", 404, {}),
    ("post", "/api/v1/flat/{version}/activate", "/api/v1/flat/short/activate", 422, {}),
    ("delete", "/api/v1/flat/{version}", "/api/v1/flat/flat-00000000", 404, {}),
    ("get", "/api/v1/flat/{version}/image", "/api/v1/flat/flat-00000000/image", 404, {}),
    ("post", "/api/v1/mode", "/api/v1/mode", 401, {"no_token": True, "json": {"mode": "auto"}}),
    ("post", "/api/v1/mode", "/api/v1/mode", 422, {"json": {"mode": "reboot"}}),
    ("get", "/api/v1/seeing", "/api/v1/seeing", 422, {"params": {"limit": 0}}),
    ("get", "/api/v1/images/{image_id}", "/api/v1/images/preview-20200101T000000.000Z", 404, {}),
]


EMPTY_CASES: list[tuple[str, str, str, int, dict[str, Any]]] = [
    ("get", "/api/v1/images/latest", "/api/v1/images/latest", 404, {}),
    ("get", "/api/v1/seeing/latest", "/api/v1/seeing/latest", 404, {}),
    ("get", "/api/v1/health", "/api/v1/health", 503, {}),
]


@pytest.mark.parametrize(
    ("method", "path", "url", "status", "options"),
    CASES,
    ids=[f"{m}-{u}-{s}-{i}" for i, (m, _, u, s, _) in enumerate(CASES)],
)
def test_the_live_answer_matches_the_documented_schema(
    client: TestClient,
    document: dict[str, Any],
    method: str,
    path: str,
    url: str,
    status: int,
    options: dict[str, Any],
) -> None:
    check_answer(client, document, method, path, url, status, options)


@pytest.mark.parametrize(
    ("method", "path", "url", "status", "options"),
    EMPTY_CASES,
    ids=[f"{m}-{u}-{s}-{i}" for i, (m, _, u, s, _) in enumerate(EMPTY_CASES)],
)
def test_the_live_answer_of_an_empty_store_matches_the_documented_schema(
    empty_client: TestClient,
    document: dict[str, Any],
    method: str,
    path: str,
    url: str,
    status: int,
    options: dict[str, Any],
) -> None:
    check_answer(empty_client, document, method, path, url, status, options)


def check_answer(
    chosen: TestClient,
    document: dict[str, Any],
    method: str,
    path: str,
    url: str,
    status: int,
    options: dict[str, Any],
) -> None:
    headers = {} if options.get("no_token") or method == "get" else bearer()
    kwargs: dict[str, Any] = {"headers": headers}
    if "json" in options:
        kwargs["json"] = options["json"]
    if "params" in options:
        kwargs["params"] = options["params"]
    response = getattr(chosen, method)(url, **kwargs)
    assert response.status_code == status, response.text
    schema = documented_schema(document, method, path, status)
    try:
        validate(response.json(), schema, document)
    except InvalidError as error:  # name the case, because the message names only the path
        raise AssertionError(f"{method.upper()} {url} {status}: {error}") from None


def test_a_populated_dark_library_matches_the_documented_schema(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
    document: dict[str, Any],
) -> None:
    core = FakeCoreClient(clock=clock, dark_script=DarkScript(queued_s=0.0, bias_s=10.0))
    core.dark.sets = [dark_set("dark-a", 12.0, 3.0), dark_set("dark-b", 4.0, 40.0)]
    core.dark.model = DarkModelView(
        reference_c=20.0, rate_ref_e_per_s=0.12, doubling_c=6.0, doubling_fitted=True, n_sets=2
    )
    chosen = open_client(make_app(core=core))
    core.submit(QueueDark(frames=5, bias_frames=4))
    clock.advance(2)  # the session runs its bias frames
    answer = chosen.get(f"{API}/dark")
    assert answer.json()["task"]["phase"] == "bias"
    schema = documented_schema(document, "get", f"{API}/dark", 200)
    validate(answer.json(), schema, document)
    components = document["components"]["schemas"]
    assert set(components["DarkLibraryResponse"]["properties"]) == {
        "mode",
        "gain",
        "exposure_s",
        "sensor_temperature_c",
        "status",
        "model",
        "sets",
        "task",
        "quality",
    }


def test_the_rolling_seeing_value_matches_the_documented_schema(
    client: TestClient, core: FakeCoreClient, clock: VirtualClock, document: dict[str, Any]
) -> None:
    core.live = live_seeing_view(
        t_utc_ns=clock.utc_ns() - 3_000_000_000,
        r0_cm=None,
        seeing_fwhm_arcsec=None,
        flags=["cloud"],
        quality={"r0_cm": "too few usable frames", "seeing_fwhm_arcsec": "too few usable frames"},
    )
    answer = client.get(f"{API}/seeing/live")
    assert answer.status_code == 200
    schema = documented_schema(document, "get", f"{API}/seeing/live", 200)
    assert schema == {"$ref": "#/components/schemas/LiveSeeingResponse"}
    validate(answer.json(), schema, document)
    properties = document["components"]["schemas"]["LiveSeeingResponse"]["properties"]
    assert set(answer.json()) == set(properties)
    assert {"t_utc", "age_s", "t_utc_ns", "span_s", "seeing_fwhm_arcsec"} <= set(properties)


def test_the_state_of_a_polaris_frame_is_documented_and_the_header_matches_it(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
    document: dict[str, Any],
) -> None:
    async def source() -> AsyncIterator[PolarisFrame]:
        yield PolarisFrame(polaris_state(0), tiny_png())
        await asyncio.Event().wait()

    chosen = open_client(make_app(core=FakeCoreClient(clock=clock, polaris=source)))
    assert wait_until(lambda: chosen.get(f"{API}/polaris/frame").status_code == 200)
    answer = chosen.get(f"{API}/polaris/frame")
    state = json.loads(answer.headers["x-frame-state"])
    schemas = document["components"]["schemas"]
    validate(state, {"$ref": "#/components/schemas/PolarisState"}, document)
    assert set(schemas["PolarisState"]["properties"]) == set(PolarisState.model_fields)
    assert set(schemas["PolarisState"]["properties"]) == set(state)
    assert {"PolarisStar", "PolarisStretch", "LiveSeeingView", "RoiView"} <= set(schemas)
    assert schemas["PolarisState"]["properties"]["image_type"]["const"] == "image/png"
    operation = document["paths"][f"{API}/polaris/frame"]["get"]
    ok = operation["responses"]["200"]
    assert set(ok["content"]) == {"image/png"}
    assert set(ok["headers"]) == {"X-Frame-Seq", "X-Frame-State"}
    reference = ok["headers"]["X-Frame-State"]["content"]["application/json"]["schema"]
    assert reference == {"$ref": "#/components/schemas/PolarisState"}
    assert "204" in operation["responses"]
    assert operation["tags"] == ["live"]
    assert document["paths"][f"{API}/seeing/live"]["get"]["tags"] == ["live"]


def flat_core(clock: VirtualClock) -> FakeCoreClient:
    core = FakeCoreClient(
        clock=clock,
        flat_script=FlatScript(queued_s=0.0, setup_s=1.0, exposure_s=2.0, capture_s=10.0),
    )
    core.dark.sets = [dark_set("dark-a", 12.0, 3.0)]
    core.flat.seed(age_days=30.0, active=True)
    core.flat.seed(age_days=1.0, state="pending", second_set=True)
    return core


def test_a_populated_flat_library_matches_the_documented_schema(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
    document: dict[str, Any],
) -> None:
    core = flat_core(clock)
    chosen = open_client(make_app(core=core))
    core.submit(QueueFlat(frames=16))
    clock.advance(5.0)  # the session takes its frames
    answer = chosen.get(f"{API}/flat")
    body = answer.json()
    assert body["task"]["phase"] == "capture"
    assert body["flats"][0]["second_set"] is True  # the two-set fields are documented too
    validate(body, documented_schema(document, "get", f"{API}/flat", 200), document)
    components = document["components"]["schemas"]
    assert set(components["FlatLibraryResponse"]["properties"]) == {
        "mode",
        "gain",
        "sensor_temperature_c",
        "active_version",
        "pending_version",
        "flat_file_pinned",
        "library_overrides",
        "blocker",
        "flats",
        "session",
        "task",
        "quality",
    }
    assert set(components["FlatResponse"]["properties"]) >= {
        "version",
        "state",
        "corner_percent",
        "vignetting",
        "shadow_items",
        "optics_tilt",
        "agreement",
        "warnings",
        "image_url",
    }


def test_the_answers_of_the_flat_commands_match_the_documented_schemas(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
    document: dict[str, Any],
) -> None:
    core = flat_core(clock)
    generous = WebSettings.model_validate({"rate_limit": {"commands_per_window": 100}})
    chosen = open_client(make_app(settings=generous, core=core))
    pending = core.flat.flats[0].version
    old = core.flat.flats[1].version  # in use until the pending flat takes over

    def check(method: str, path: str, url: str, status: int, **kwargs: Any) -> None:
        response = getattr(chosen, method)(url, headers=bearer(), **kwargs)
        assert response.status_code == status, response.text
        validate(response.json(), documented_schema(document, method, path, status), document)

    session = f"{API}/flat/session"
    action = f"{API}/flat/{{version}}"
    check("post", session, session, 200, json={"frames": 16})
    check("post", session, session, 409, json={"frames": 16})  # a session is queued or running
    check(
        "post", f"{action}/activate", f"{API}/flat/{pending}/activate", 409
    )  # and holds the library
    check("post", f"{session}/stop", f"{session}/stop", 200)
    check("post", f"{action}/activate", f"{API}/flat/{pending}/activate", 200)
    check("delete", action, f"{API}/flat/{pending}", 409)  # the flat in use
    check("delete", action, f"{API}/flat/{old}", 200)


def test_the_image_of_a_flat_is_documented_as_a_jpeg_with_a_pattern_for_the_version(
    document: dict[str, Any],
) -> None:
    operation = document["paths"][f"{API}/flat/{{version}}/image"]["get"]
    assert set(operation["responses"]["200"]["content"]) == {"image/jpeg"}
    (parameter,) = operation["parameters"]
    assert (parameter["name"], parameter["in"], parameter["required"]) == ("version", "path", True)
    assert parameter["schema"]["pattern"] == r"^flat-[0-9a-f]{8}$"


def test_an_image_answer_matches_the_documented_schema(
    client: TestClient, layout: Any, document: dict[str, Any]
) -> None:
    write_preview(layout, "20261001T020000.000Z")
    write_fits(layout, "20261001T020000.000Z")
    listed = client.get(f"{API}/images")
    validate(listed.json(), documented_schema(document, "get", f"{API}/images", 200), document)
    item = client.get(f"{API}/images/latest", params={"format": "json"})
    schema = {"$ref": "#/components/schemas/ImageItem"}
    validate(item.json(), schema, document)
    assert listed.json()["items"][0] == item.json()
    content = document["paths"][f"{API}/images/latest"]["get"]["responses"]["200"]["content"]
    assert set(content) == {"image/jpeg", "application/fits", "application/json"}


def test_the_openapi_route_serves_the_same_document(client: TestClient) -> None:
    response = client.get(f"{API}/openapi.json")
    assert response.status_code == 200
    assert response.json() == json.loads(render_openapi())
