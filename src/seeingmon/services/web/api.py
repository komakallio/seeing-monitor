"""The routes of the REST API v1.

`router` holds the routes of `/api/v1`. The handlers are plain functions, which FastAPI runs in its
thread pool, because the store and the link to `core` block. The frame routes of the live views are
`async`, because they share the event loop with the hubs that feed them.

The router does not know any one application. Each handler takes the `WebContext` of its app through
the `Ctx` dependency, which reads `app.state.ctx` (see `seeingmon.services.web.app`). So one router
serves every app, and a test that builds many apps builds the routes once.

Every read route takes `authorize_read`, and every `POST` route takes `authorize_command` (see
`seeingmon.services.web.context` for the rule). A response is JSON, except the images and the live
frame. The record routes build their schemas from the record declarations
(`seeingmon.records.api_schema`, through `seeingmon.services.web.schemas`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Depends, Path, Query, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response

import seeingmon
from seeingmon.clock import NS_PER_S
from seeingmon.scheduler.commands import (
    CancelTask,
    Command,
    CommandResult,
    RejectReason,
    StopAlignment,
    StopRapidFocus,
)
from seeingmon.services.web.context import WebContext
from seeingmon.services.web.contract import (
    ActivityView,
    AlignmentFrame,
    AlignmentState,
    DarkLibraryView,
    FlatActionView,
    FlatLibraryView,
    PolarisFrame,
    SchedulerView,
)
from seeingmon.services.web.data import (
    InvalidQueryError,
    Step,
    StoreUnavailableError,
    TimeRange,
    decode_token,
    encode_token,
    iso,
)
from seeingmon.services.web.errors import ApiError
from seeingmon.services.web.health import HealthReport
from seeingmon.services.web.images import ImageInfo, ImageKey, parse_image_id
from seeingmon.services.web.live import (
    RAPID_LISTS,
    RAPID_PATH,
    FrameHub,
    HistoryCursor,
    Subscription,
)
from seeingmon.services.web.models import (
    FLAT_VERSION_PATTERN,
    ActivityResponse,
    AlignmentStartRequest,
    ApiInfo,
    BurstRequest,
    CommandResponse,
    CoreLink,
    DarkLibraryResponse,
    DarkModelResponse,
    DarkRequest,
    DarkSetResponse,
    DarkStatusResponse,
    DarkTaskResponse,
    ErrorResponse,
    EventLevel,
    FaultStatusView,
    FlatActionResponse,
    FlatLibraryResponse,
    FlatSessionRequest,
    FocusResetResponse,
    HealthResponse,
    ImageFormat,
    ImageItem,
    ImageList,
    LiveSeeingResponse,
    ModeRequest,
    Order,
    RapidFocusStartRequest,
    RecordTime,
    ReplayRequest,
    SchedulerStatusResponse,
    StatusResponse,
    StreamStatus,
    SweepRequest,
    UiSettings,
)
from seeingmon.services.web.privacy import scrub_json, scrub_text
from seeingmon.services.web.schemas import json_response

_log = logging.getLogger(__name__)

API_VERSION = "1.0.0"
PREFIX = "/api/v1"

# The record routes: the path, the record type, and the name of the OpenAPI component.
SERIES = (
    ("seeing", "seeing_window", "SeeingWindow"),
    ("sky", "sky_quality", "SkyQuality"),
    ("pointing", "pointing", "Pointing"),
)

IMMUTABLE = "public, max-age=86400, immutable"
SECURITY: list[dict[str, list[str]]] = [{"bearerAuth": []}]
ERROR_TEXT = {
    401: "The token is missing or wrong.",
    403: "The server has no token configured, so it refuses every command.",
    404: "The resource does not exist.",
    409: "The scheduler cannot do this in its current state.",
    413: "The request body is too large.",
    422: "A parameter or the body is not valid.",
    429: "The client sent too many requests.",
    502: "The core process answered something that the web process cannot use.",
    503: "The store or the core process is not available.",
}
FORMAT_QUERY = Query(description="`jpeg` returns the preview, `fits` the frame, `json` the data.")
WS_CLOSE_POLICY = 1008
WS_CLOSE_TRY_LATER = 1013
WS_AUTH_TIMEOUT_S = 5.0


def get_context(request: Request) -> WebContext:
    """The context of the app that handles the request."""
    context = request.app.state.ctx
    assert isinstance(context, WebContext)
    return context


Ctx = Annotated[WebContext, Depends(get_context)]


def authorize_read(request: Request, ctx: Ctx) -> None:
    """The dependency of a read route. It checks the token when the settings ask for one."""
    ctx.authorize_read(request)


def authorize_command(request: Request, ctx: Ctx) -> None:
    """The dependency of a `POST` route. It checks the token and the rate limit."""
    ctx.authorize_command(request)


READ = [Depends(authorize_read)]
WRITE = [Depends(authorize_command)]
router = APIRouter(prefix=PREFIX)


def errors(*codes: int) -> dict[int | str, dict[str, Any]]:
    """The `responses` entries for the error statuses of a route."""
    return {code: {"model": ErrorResponse, "description": ERROR_TEXT[code]} for code in codes}


def _age_s(now_ns: int, t_ns: int | None) -> float | None:
    return None if t_ns is None else round(max(0.0, (now_ns - t_ns) / NS_PER_S), 3)


def health_response(report: HealthReport, now_ns: int) -> HealthResponse:
    return HealthResponse(
        status=report.status,
        now=iso(now_ns),
        age_s=None if report.age_s is None else round(report.age_s, 3),
        reasons=list(report.reasons),
        components=report.components,
        flags=list(report.flags),
        quality=report.quality,
    )


def _optional_iso(t_ns: int | None) -> str | None:
    return None if t_ns is None else iso(t_ns)


def _optional_scrub(text: str | None) -> str | None:
    return None if text is None else scrub_text(text)


def activity_response(view: ActivityView) -> ActivityResponse:
    """The activity as the API serves it: ISO times, and no text from a private place."""
    return ActivityResponse(
        state=view.state,
        phase=view.phase,
        label=scrub_text(view.label),
        since_utc=iso(view.since_utc_ns),
        ends_utc=_optional_iso(view.ends_utc_ns),
        next_label=_optional_scrub(view.next_label),
        next_utc=_optional_iso(view.next_utc_ns),
        cadence_s=view.cadence_s,
        detail=_optional_scrub(view.detail),
        reason=_optional_scrub(view.reason),
    )


def scheduler_response(view: SchedulerView) -> SchedulerStatusResponse:
    """The scheduler status as the API serves it: ISO times, and no text from a private place."""
    stream = None
    if view.stream is not None:
        roi = view.stream.roi
        stream = StreamStatus(
            stream_id=view.stream.stream_id,
            purpose=view.stream.purpose,
            mode=view.stream.mode,
            exposure_us=view.stream.exposure_us,
            gain=view.stream.gain,
            roi=None
            if roi is None
            else {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height},
        )
    fault = view.fault
    return SchedulerStatusResponse(
        t_utc=iso(view.t_utc_ns),
        state=view.state,
        state_reason=scrub_text(view.state_reason),
        state_since=iso(view.state_since_utc_ns),
        last_transition=None
        if view.last_transition_utc_ns is None
        else iso(view.last_transition_utc_ns),
        degraded=view.degraded,
        stream=stream,
        cloud=view.cloud,
        cloud_fraction=view.cloud_fraction,
        twilight=view.twilight,
        background_fraction=view.background_fraction,
        sensor_temperature_c=view.sensor_temperature_c,
        counters=dict(view.counters),
        fault=FaultStatusView(
            failures=fault.failures,
            good_frames=fault.good_frames,
            last_error=None if fault.last_error is None else scrub_text(fault.last_error),
            next_attempt=_optional_iso(fault.next_attempt_utc_ns),
            next_step=fault.next_step,
            cause=fault.cause,
            reason=_optional_scrub(fault.reason),
            since=_optional_iso(fault.since_utc_ns),
        ),
        queued_tasks=view.queued_tasks,
        survey_pending=view.survey_pending,
        alignment_idle_s=view.alignment_idle_s,
        activity=None if view.activity is None else activity_response(view.activity),
    )


def dark_response(view: DarkLibraryView) -> DarkLibraryResponse:
    """The dark library as the API serves it. Free text from `core` goes through `scrub_text`."""
    quality: dict[str, str] = {}
    if view.sensor_temperature_c is None:
        quality["sensor_temperature_c"] = "core reports no sensor temperature"
    if view.model is None:
        quality["model"] = "core has no dark model yet"
    task = view.task
    return DarkLibraryResponse(
        mode=view.mode,
        gain=view.gain,
        exposure_s=view.exposure_s,
        sensor_temperature_c=view.sensor_temperature_c,
        status=DarkStatusResponse(
            due=view.status.due,
            reason=scrub_text(view.status.reason),
            tolerance_c=view.status.tolerance_c,
            max_age_days=view.status.max_age_days,
            gap_c=view.status.gap_c,
            nearest_name=view.status.nearest_name,
            newest_age_days=view.status.newest_age_days,
        ),
        model=None
        if view.model is None
        else DarkModelResponse(
            reference_c=view.model.reference_c,
            rate_ref_e_per_s=view.model.rate_ref_e_per_s,
            doubling_c=view.model.doubling_c,
            doubling_fitted=view.model.doubling_fitted,
            rms_log2=view.model.rms_log2,
            n_sets=view.model.n_sets,
        ),
        sets=[DarkSetResponse(**item.model_dump()) for item in view.sets],
        task=DarkTaskResponse(
            **{
                **task.model_dump(),
                "message": scrub_text(task.message),
                "reason": scrub_text(task.reason),
                "summary": scrub_text(task.summary),
            }
        ),
        quality=quality or None,
    )


def flat_response(view: FlatLibraryView) -> FlatLibraryResponse:
    """The flat library as the API serves it. Free text from `core` goes through `scrub_json`."""
    quality: dict[str, str] = {}
    if view.sensor_temperature_c is None:
        quality["sensor_temperature_c"] = "core reports no sensor temperature"
    data = scrub_json(view.model_dump(mode="json"))
    for flat in data["flats"]:
        flat["image_url"] = f"{PREFIX}/flat/{flat['version']}/image" if flat["has_image"] else None
    data["quality"] = quality or None
    return FlatLibraryResponse.model_validate(data)


def flat_action_reply(view: FlatActionView) -> JSONResponse:
    """The answer to an activation or a deletion: a body when it worked, an error when not."""
    message = scrub_text(view.message)
    if view.ok:
        body = FlatActionResponse(message=message, version=view.version or "")
        return JSONResponse(body.model_dump(mode="json"))
    if view.reason == "unknown":
        raise ApiError(404, "unknown_flat", message)
    codes = {"active": "flat_in_use", "busy": "session_busy", "invalid": "flat_unusable"}
    raise ApiError(409, codes.get(view.reason or "", "conflict"), message)


def build_status(ctx: WebContext) -> StatusResponse:
    """Collect the state of every component for `GET /status`."""
    now_ns = ctx.clock.utc_ns()
    quality: dict[str, str] = {}
    core_status = ctx.probe_core()
    if core_status is None:
        quality["scheduler"] = "core does not answer"
    report = ctx.judge(core_status)
    times: dict[str, RecordTime] = {}
    for record_type in ("seeing_window", "sky_quality", "pointing", "health", "event"):
        t_ns: int | None = None
        try:
            record = ctx.data.latest(record_type)
            t_ns = None if record is None else int(record["t_utc_ns"])
        except StoreUnavailableError:
            quality[record_type] = "the store cannot be read"
        times[record_type] = RecordTime(
            t_utc=None if t_ns is None else iso(t_ns), age_s=_age_s(now_ns, t_ns)
        )
    web = ctx.settings
    return StatusResponse(
        now=iso(now_ns),
        api_version=API_VERSION,
        software_version=seeingmon.__version__,
        station_id=ctx.station_id,
        demo=ctx.demo,
        health=health_response(report, now_ns),
        core=CoreLink(
            reachable=core_status is not None,
            instance=None if core_status is None else core_status.instance,
        ),
        scheduler=None if core_status is None else scheduler_response(core_status.scheduler),
        data=times,
        ui=UiSettings(
            refresh_s=web.ui_refresh_s,
            token_required_for_reads=web.require_token_for_reads,
            commands_enabled=ctx.verifier.enabled,
            alignment_max_fps=web.live.max_fps,
            alignment_stall_s=web.live.stall_s,
            polaris_max_fps=web.live.polaris_max_fps,
            polaris_stall_s=web.live.stall_s,
        ),
        quality=quality or None,
    )


def image_item(info: ImageInfo) -> ImageItem:
    base = f"{PREFIX}/images/{info.id}"
    return ImageItem(
        id=info.id,
        kind=info.key.kind,
        t_utc=iso(info.key.t_utc_ns),
        t_utc_ns=info.key.t_utc_ns,
        size_bytes=info.size_bytes,
        has_fits=info.has_fits,
        fits_bytes=info.fits_bytes,
        preview_url=base,
        fits_url=f"{base}?format=fits" if info.has_fits else None,
    )


def command_reply(ctx: WebContext, command: Command) -> JSONResponse:
    """Send a command to the scheduler, and turn its answer into the response of the API."""
    return result_reply(type(command).__name__, ctx.core.submit(command))


def result_reply(name: str, result: CommandResult) -> JSONResponse:
    """Turn the answer of the scheduler to the command `name` into the response of the API."""
    _log.info("command %s was %s", name, "accepted" if result.accepted else "rejected")
    message = scrub_text(result.message)
    if result.accepted:
        body = CommandResponse(
            accepted=True,
            message=message,
            state=result.state,
            reason=None,
            task_id=result.task_id,
        )
        return JSONResponse(body.model_dump(mode="json"))
    if result.reason is RejectReason.INVALID:
        raise ApiError(422, "invalid_command", message)
    if result.reason is RejectReason.CLOSED:
        raise ApiError(503, "core_unavailable", message, headers={"Retry-After": "5"})
    body = CommandResponse(
        accepted=False,
        message=message,
        state=result.state,
        reason=None if result.reason is None else result.reason.value,
        task_id=None,
    )
    return JSONResponse(body.model_dump(mode="json"), status_code=409)


def _serve_image(
    ctx: WebContext, info: ImageInfo, image_format: ImageFormat, *, latest: bool
) -> Any:
    if image_format is ImageFormat.JSON:
        return JSONResponse(image_item(info).model_dump(mode="json"))
    cache = "no-cache" if latest else IMMUTABLE
    if image_format is ImageFormat.JPEG:
        path = ctx.images.preview_path(info.key)
        if path is None:
            raise ApiError(404)
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": cache})
    fits = ctx.images.fits_path(info.key) if info.has_fits else None
    if fits is None:
        raise ApiError(404, "no_fits", "This image has no FITS frame.")
    return FileResponse(
        fits,
        media_type="application/fits",
        filename=f"{info.key.stamp}.fits",
        headers={"Cache-Control": cache},
    )


def _image_cursor(text: str) -> ImageKey:
    token = decode_token(text)
    image_id = token.get("k") if len(token) == 1 else None
    if not isinstance(image_id, str):
        raise InvalidQueryError("cursor is not a cursor that this server gave")
    return parse_image_id(image_id)


# --- Routes ----------------------------------------------------------------------------------


@router.get(
    "",
    operation_id="get_api_info",
    summary="Describe the API",
    tags=["status"],
    response_model=ApiInfo,
)
def api_info() -> ApiInfo:
    """Return the version of the API and the software, and where the OpenAPI description is."""
    return ApiInfo(
        api_version=API_VERSION,
        software_version=seeingmon.__version__,
        openapi=f"{PREFIX}/openapi.json",
    )


# --- Status and health ---


@router.get(
    "/status",
    operation_id="get_status",
    summary="Get the state of every component",
    tags=["status"],
    response_model=StatusResponse,
    responses=errors(401, 429),
    dependencies=READ,
)
def get_status(ctx: Ctx) -> JSONResponse:
    """Return the health, the scheduler, the link to `core`, and the age of the newest records.

    A value that the server cannot find is `null`, and `quality` says why. The status leaves
    out the Sun's elevation, because a series of elevations would show the site.
    """
    return JSONResponse(build_status(ctx).model_dump(mode="json"))


@router.get(
    "/health",
    operation_id="get_health",
    summary="Get the health verdict",
    tags=["status"],
    response_model=HealthResponse,
    responses={
        200: {"model": HealthResponse, "description": "The system is healthy or degraded."},
        503: {"model": HealthResponse, "description": "The system has failed."},
        **errors(401, 429),
    },
    dependencies=READ,
)
def get_health(ctx: Ctx) -> JSONResponse:
    """Judge the system from the newest health record and from the link to `core`.

    The answer is 200 when the status is `healthy` or `degraded`, and 503 when it is `failed`.
    A failed system has an unreadable store, no health record, a record older than the limit
    (so `core` stopped writing), or a failed component. An external watchdog can poll this
    route.
    """
    now_ns = ctx.clock.utc_ns()
    report = ctx.judge(ctx.probe_core())
    return JSONResponse(
        health_response(report, now_ns).model_dump(mode="json"),
        status_code=report.http_status,
    )


# --- Records ---


@router.get(
    "/events",
    operation_id="get_events",
    summary="List events",
    tags=["records"],
    response_model=None,
    responses={
        200: json_response("EventPage", "A page of events."),
        **errors(401, 422, 429, 503),
    },
    dependencies=READ,
)
def get_events(
    ctx: Ctx,
    from_: Annotated[
        str | None,
        Query(
            alias="from",
            description="The start of the range, included. Default: 24 hours before `to`.",
        ),
    ] = None,
    to: Annotated[
        str | None, Query(description="The end of the range, excluded. Default: now.")
    ] = None,
    level: Annotated[
        EventLevel, Query(description="The lowest severity to return.")
    ] = EventLevel.INFO,
    kind: Annotated[
        str | None,
        Query(
            max_length=64,
            pattern=r"^[a-z][a-z0-9_.]*$",
            description="Return only events whose kind starts with this text.",
        ),
    ] = None,
    order: Annotated[Order, Query(description="`desc` returns the newest first.")] = Order.DESC,
    limit: Annotated[int | None, Query(ge=1, le=10_000, description="Items per page.")] = None,
    cursor: Annotated[
        str | None, Query(max_length=128, description="The `next_cursor` of a page.")
    ] = None,
) -> JSONResponse:
    """Return events, newest first by default. Their text has no paths or addresses."""
    time_range = ctx.data.resolve_range(from_, to)
    page = ctx.data.events(
        time_range=time_range,
        limit=ctx.data.resolve_limit(limit),
        cursor=cursor,
        min_level=level.value,
        kind=kind,
        descending=order is Order.DESC,
    )
    return JSONResponse(_page_body("event", page, ctx.clock.utc_ns()))


# --- Reference ---


@router.get(
    "/profile",
    operation_id="get_profile",
    summary="Get the hardware profile",
    tags=["reference"],
    response_model=dict[str, Any],
    responses=errors(401, 404, 429),
    dependencies=READ,
)
def get_profile(ctx: Ctx) -> JSONResponse:
    """Return the hardware profile with its derived values.

    Each readout mode carries its own plate scale, field of view, and sampling, so a plate
    scale always names its mode.
    """
    if ctx.profile is None:
        raise ApiError(404, "not_found", "The server has no profile to show.")
    return JSONResponse(ctx.profile)


@router.get(
    "/config",
    operation_id="get_config",
    summary="Get the effective configuration",
    tags=["reference"],
    response_model=dict[str, Any],
    responses=errors(401, 404, 429),
    dependencies=READ,
)
def get_config(ctx: Ctx) -> JSONResponse:
    """Return the configuration without secrets, site data, paths, hosts, and addresses.

    The response keeps the sections that describe how the station computes, such as the
    scheduler and the fast path, and it leaves out the sections that describe one installation.
    """
    if ctx.config_view is None:
        raise ApiError(404, "not_found", "The server has no configuration to show.")
    return JSONResponse(ctx.config_view)


# --- Images ---


@router.get(
    "/images",
    operation_id="list_images",
    summary="List the images",
    tags=["images"],
    response_model=ImageList,
    responses=errors(401, 422, 429),
    dependencies=READ,
)
def list_images(
    ctx: Ctx,
    limit: Annotated[int | None, Query(ge=1, le=1000, description="Images per page.")] = None,
    cursor: Annotated[
        str | None, Query(max_length=128, description="The `next_cursor` of a page.")
    ] = None,
) -> JSONResponse:
    """List the preview images, newest first."""
    settings = ctx.images.settings
    count = settings.default_list_limit if limit is None else limit
    if count > settings.max_list_limit:
        raise InvalidQueryError(f"limit must be between 1 and {settings.max_list_limit}")
    before = None if cursor is None else _image_cursor(cursor)
    infos, more = ctx.images.recent(count, before)
    body = ImageList(
        now=iso(ctx.clock.utc_ns()),
        items=[image_item(info) for info in infos],
        next_cursor=encode_token({"k": infos[-1].id}) if more and infos else None,
    )
    return JSONResponse(body.model_dump(mode="json"))


image_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "The image: a JPEG preview, a FITS frame, or its description.",
        "content": {
            "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
            "application/fits": {"schema": {"type": "string", "format": "binary"}},
            "application/json": {"schema": {"$ref": "#/components/schemas/ImageItem"}},
        },
    },
    **errors(401, 404, 422, 429),
}


@router.get(
    "/images/latest",
    operation_id="get_latest_image",
    summary="Get the newest image",
    tags=["images"],
    response_model=None,
    responses=image_responses,
    dependencies=READ,
)
def get_latest_image(
    ctx: Ctx,
    format: Annotated[ImageFormat, FORMAT_QUERY] = ImageFormat.JPEG,
) -> Any:
    """Return the newest image. A client that shows it should ask again after a while."""
    info = ctx.images.latest()
    if info is None:
        raise ApiError(404, "no_data", "The server holds no image yet.")
    return _serve_image(ctx, info, format, latest=True)


@router.get(
    "/images/{image_id}",
    operation_id="get_image",
    summary="Get an image",
    tags=["images"],
    response_model=None,
    responses=image_responses,
    dependencies=READ,
)
def get_image(
    ctx: Ctx,
    image_id: Annotated[str, Path(max_length=64, description="An ID from `GET /images`.")],
    format: Annotated[ImageFormat, FORMAT_QUERY] = ImageFormat.JPEG,
) -> Any:
    """Return one image. The ID names a file, so it never holds a path: a bad ID is a 404."""
    try:
        info = ctx.images.get(image_id)
    except InvalidQueryError:
        raise ApiError(404) from None
    if info is None:
        raise ApiError(404)
    return _serve_image(ctx, info, format, latest=False)


# --- Commands ---

command_responses: dict[int | str, dict[str, Any]] = {
    200: {"model": CommandResponse, "description": "The scheduler accepted the command."},
    409: {"model": CommandResponse, "description": ERROR_TEXT[409]},
    **errors(401, 403, 413, 422, 429, 502, 503),
}


@router.post(
    "/commands/burst",
    operation_id="post_burst",
    summary="Queue a burst",
    tags=["commands"],
    response_model=CommandResponse,
    responses=command_responses,
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def post_burst(body: BurstRequest, ctx: Ctx) -> JSONResponse:
    """Queue a burst: record raw frames to a SER file with a JSON sidecar.

    The burst runs at the next cycle boundary, and its result is pinned.
    """
    return command_reply(ctx, body.to_command())


@router.post(
    "/commands/sweep",
    operation_id="post_sweep",
    summary="Queue a sweep",
    tags=["commands"],
    response_model=CommandResponse,
    responses=command_responses,
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def post_sweep(body: SweepRequest, ctx: Ctx) -> JSONResponse:
    """Queue a sweep: a short fast window for every cell of a grid of settings.

    An empty axis uses the configured default. The scheduler checks the values against the
    hardware profile.
    """
    return command_reply(ctx, body.to_command())


@router.post(
    "/commands/replay",
    operation_id="post_replay",
    summary="Queue a replay",
    tags=["commands"],
    response_model=CommandResponse,
    responses=command_responses,
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def post_replay(body: ReplayRequest, ctx: Ctx) -> JSONResponse:
    """Queue a replay of a recording through the production analysis."""
    return command_reply(ctx, body.to_command())


@router.post(
    "/mode",
    operation_id="post_mode",
    summary="Pause or resume the scheduler",
    tags=["commands"],
    response_model=CommandResponse,
    responses=command_responses,
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def post_mode(body: ModeRequest, ctx: Ctx) -> JSONResponse:
    """Pause the scheduler, or resume it. A resumed scheduler enters `safe` first."""
    return command_reply(ctx, body.to_command())


@router.post(
    "/alignment/start",
    operation_id="post_alignment_start",
    summary="Start the alignment helper",
    tags=["alignment"],
    response_model=CommandResponse,
    responses=command_responses,
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def post_alignment_start(ctx: Ctx, body: AlignmentStartRequest | None = None) -> JSONResponse:
    """Start the alignment stream, or keep it alive when it already runs.

    The command preempts every other mode. Send it again to restart the idle timer.
    """
    return command_reply(ctx, (body or AlignmentStartRequest()).to_command())


@router.post(
    "/alignment/stop",
    operation_id="post_alignment_stop",
    summary="Stop the alignment helper",
    tags=["alignment"],
    response_model=CommandResponse,
    responses=command_responses,
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def post_alignment_stop(ctx: Ctx) -> JSONResponse:
    """End the alignment stream. The scheduler goes back to `safe` and checks the sky."""
    return command_reply(ctx, StopAlignment())


@router.post(
    "/alignment/focus/reset",
    operation_id="post_alignment_focus_reset",
    summary="Restart the best focus value",
    tags=["alignment"],
    response_model=FocusResetResponse,
    responses=errors(401, 403, 429, 502, 503),
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def post_alignment_focus_reset(ctx: Ctx) -> JSONResponse:
    """Restart the best focus value of the alignment, for example after you refocus.

    The best value is the smallest value of the session that is not a spike. A reset forgets it, and
    the next value that is not a spike becomes the best. The history of the values stays, so the
    curve before and after the refocus shows together. The call needs no running alignment: it
    changes nothing then.
    """
    ctx.core.alignment_reset_focus()
    return JSONResponse(
        FocusResetResponse(reset=True, message="the best focus value restarted").model_dump(
            mode="json"
        )
    )


@router.post(
    "/alignment/rapid-focus/start",
    operation_id="post_alignment_rapid_focus_start",
    summary="Start the rapid focus mode on Polaris",
    tags=["alignment"],
    response_model=CommandResponse,
    responses=command_responses,
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def post_alignment_rapid_focus_start(
    ctx: Ctx, body: RapidFocusStartRequest | None = None
) -> JSONResponse:
    """Switch the alignment to the rapid focus mode, or keep that mode alive when it runs.

    The camera then streams a small window of the fast readout mode around Polaris, and `core`
    measures the width of the star in arcseconds on every frame. The readings come with the
    video of Polaris, in `rapid_focus` of its state. The mode is offered only while the
    alignment runs, the stars are coarsely focused, and `core` has located Polaris:
    `GET /alignment/state` says so in `rapid_focus.available`, and `rapid_focus.reason` says
    in words what is missing. The request carries no position, because `core` takes it from
    the solution of the alignment frames. A start while the mode runs restarts its idle timer,
    and a new exposure or gain changes the stream. The scheduler ends the mode on its own
    after a time without use, or when the star leaves the window, and the alignment then shows
    the whole frame again. A refusal is `409` with the reason `not_aligning`, `not_available`
    (the message says what is missing), `camera_fault`, or `degraded`.
    """
    request = body or RapidFocusStartRequest()
    return result_reply(
        "StartRapidFocus", ctx.core.rapid_focus_start(request.exposure_us, request.gain)
    )


@router.post(
    "/alignment/rapid-focus/stop",
    operation_id="post_alignment_rapid_focus_stop",
    summary="Stop the rapid focus mode",
    tags=["alignment"],
    response_model=CommandResponse,
    responses=command_responses,
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def post_alignment_rapid_focus_stop(ctx: Ctx) -> JSONResponse:
    """End the rapid focus mode. The alignment shows the whole frame again, and it keeps running.

    A stop while the mode does not run is accepted and changes nothing. The answer is `409`
    with the reason `not_aligning` when no alignment runs.
    """
    return command_reply(ctx, StopRapidFocus())


# --- Dark ---


@router.post(
    "/commands/dark",
    operation_id="post_dark",
    summary="Queue a dark session",
    tags=["commands", "dark"],
    response_model=CommandResponse,
    responses=command_responses,
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def post_dark(body: DarkRequest, ctx: Ctx) -> JSONResponse:
    """Queue a dark session: record bias and dark frames with the camera covered.

    Cover the camera before you send it. With `wait_for_cover`, the session waits until short
    test frames are dark, which proves that the camera is covered. The session adds a set to the
    dark library, and with `pause_after` it pauses the scheduler at the end, so that nothing
    records data while the camera may still be covered. `Resume` (`POST /mode`) continues, and
    `Pause` ends a running session as `aborted`. The session starts at the next step of the
    scheduler, after a survey exposure in progress, and a paused scheduler or a running alignment
    holds it back until you resume or the alignment ends. Only one session may be queued or
    running (`409` with the reason `busy`). `GET /dark` shows its progress.
    """
    limits = ctx.settings.requests
    problems = []
    if body.exposure_s is not None and body.exposure_s > limits.max_dark_exposure_s:
        problems.append(
            {
                "field": "body.exposure_s",
                "message": f"The exposure may be at most {limits.max_dark_exposure_s:g} seconds",
                "type": "less_than_equal",
            }
        )
    if len(body.label) > limits.max_label_chars:
        problems.append(
            {
                "field": "body.label",
                "message": f"The label may have at most {limits.max_label_chars} characters",
                "type": "string_too_long",
            }
        )
    if problems:
        raise ApiError(422, details=problems)
    return command_reply(ctx, body.to_command())


@router.get(
    "/dark",
    operation_id="get_dark",
    summary="Get the dark library and the dark session",
    tags=["dark"],
    response_model=DarkLibraryResponse,
    responses=errors(401, 429, 502, 503),
    dependencies=READ,
)
def get_dark(ctx: Ctx) -> JSONResponse:
    """Return the dark sets, whether the library is due for a new set, the dark model, the sensor
    temperature, and the progress of the latest dark session.

    Poll it while a session is queued or running. It answers `503 core_unavailable` when `core`
    does not answer, because the library lives there.
    """
    return JSONResponse(dark_response(ctx.core.dark_library()).model_dump(mode="json"))


# --- Flat ---

FlatVersion = Annotated[
    str,
    Path(pattern=FLAT_VERSION_PATTERN, description="The version of a flat, from `GET /flat`."),
]
flat_action_responses: dict[int | str, dict[str, Any]] = {
    409: {
        "model": ErrorResponse,
        "description": "The flat is in use, a flat session is queued or running, or the flat does "
        "not fit the frame of the survey.",
    },
    **errors(401, 403, 404, 422, 429, 502, 503),
}


@router.get(
    "/flat",
    operation_id="get_flat",
    summary="Get the flat library and the flat session",
    tags=["flat"],
    response_model=FlatLibraryResponse,
    responses=errors(401, 429, 502, 503),
    dependencies=READ,
)
def get_flat(ctx: Ctx) -> JSONResponse:
    """Return the flats of the library with the numbers of their reports, the flat in use, the
    flat that waits for a decision, the first set that waits for a second set, and the progress of
    the latest flat session.

    Poll it while a session is queued or running. It answers `503 core_unavailable` when `core`
    does not answer, because the library lives there.
    """
    return JSONResponse(flat_response(ctx.core.flat_library()).model_dump(mode="json"))


@router.post(
    "/flat/session",
    operation_id="post_flat_session",
    summary="Queue a flat session",
    tags=["commands", "flat"],
    response_model=CommandResponse,
    responses=command_responses,
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def post_flat_session(body: FlatSessionRequest, ctx: Ctx) -> JSONResponse:
    """Queue a flat session: take frames of a light source over the aperture, and add a pending
    flat to the library.

    Cover the front of the guide scope with a uniform light source before you send it. The
    session finds the exposure, takes the frames, combines them with the bias of the dark library,
    and adds the flat. It needs a dark set (`422` without one), and a second set (`set_number` 2)
    needs the first set of a session (`422` without one). With `pause_after`, it pauses the
    scheduler at the end, so that nothing records data while the light source may still cover the
    camera. `Resume` (`POST /mode`) continues. The session starts at the next step of the
    scheduler, after a survey exposure in progress, and a paused scheduler or a running alignment
    holds it back until you resume or the alignment ends. Only one session may be queued or
    running (`409` with the reason `busy`). `GET /flat` shows its progress.
    """
    return command_reply(ctx, body.to_command())


@router.post(
    "/flat/session/stop",
    operation_id="post_flat_session_stop",
    summary="Stop the flat session",
    tags=["commands", "flat"],
    response_model=CommandResponse,
    responses=command_responses,
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def post_flat_session_stop(ctx: Ctx) -> JSONResponse:
    """Remove a flat session that waits, or stop the one that runs at its next frame.

    The library stays as it was. The scheduler still pauses at the end when the session asked for
    it. The answer is `409` with the reason `no_task` when no flat session is queued or running.
    """
    return command_reply(ctx, CancelTask(kind="flat"))


@router.post(
    "/flat/{version}/activate",
    operation_id="post_flat_activate",
    summary="Use a flat",
    tags=["flat"],
    response_model=FlatActionResponse,
    responses={
        200: {"model": FlatActionResponse, "description": "The survey uses the flat."},
        413: {"model": ErrorResponse, "description": ERROR_TEXT[413]},
        **flat_action_responses,
    },
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def post_flat_activate(version: FlatVersion, ctx: Ctx) -> JSONResponse:
    """Make a flat the one that the survey divides by.

    The survey worker picks it up from its next frame, with no restart. The active flat of the
    library wins over the setting `flat_file`. Using a pending flat approves it and ends the
    session that made it. The answer is `404` for an unknown flat, and `409` while a flat session
    is queued or running, or when the flat does not fit the frame of the survey.
    """
    return flat_action_reply(ctx.core.flat_activate(version))


@router.delete(
    "/flat/{version}",
    operation_id="delete_flat",
    summary="Delete a flat",
    tags=["flat"],
    response_model=FlatActionResponse,
    responses={
        200: {"model": FlatActionResponse, "description": "The flat is deleted."},
        **flat_action_responses,
    },
    dependencies=WRITE,
    openapi_extra={"security": SECURITY},
)
def delete_flat(version: FlatVersion, ctx: Ctx) -> JSONResponse:
    """Delete a flat that the survey does not use, with its report and its preview.

    Deleting a pending flat discards it and ends the session that made it. The answer is `404`
    for an unknown flat, and `409` for the flat in use, or while a flat session is queued or
    running.
    """
    return flat_action_reply(ctx.core.flat_delete(version))


@router.get(
    "/flat/{version}/image",
    operation_id="get_flat_image",
    summary="Get the preview of a flat",
    tags=["flat"],
    response_model=None,
    response_class=Response,
    responses={
        200: {
            "description": "The preview as a JPEG, stretched to plus and minus 10 percent around "
            "1: a lighter pixel gets more light than the median pixel.",
            "content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}},
        },
        **errors(401, 404, 422, 429, 502, 503),
    },
    dependencies=READ,
)
def get_flat_image(version: FlatVersion, ctx: Ctx) -> Response:
    """Return the preview image of a flat. A version never changes its picture, so a client may
    keep it."""
    jpeg = ctx.core.flat_image(version)
    if jpeg is None:
        raise ApiError(404, "unknown_flat", "There is no preview of a flat with that name.")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": IMMUTABLE})


# --- Alignment ---


@router.get(
    "/alignment/state",
    operation_id="get_alignment_state",
    summary="Get the state of the alignment helper",
    tags=["alignment"],
    response_model=AlignmentState,
    responses=errors(401, 429, 502, 503),
    dependencies=READ,
)
def get_alignment_state(ctx: Ctx) -> JSONResponse:
    """Return the offset, the rotation, the focus, the histogram, and the saturation warning.

    Positions are in pixels of the frame that `frame` describes. `active` is false outside the
    alignment mode.
    """
    return JSONResponse(ctx.core.alignment_state().model_dump(mode="json"))


@router.get(
    "/alignment/frame",
    operation_id="get_alignment_frame",
    summary="Poll for the newest live-view frame",
    tags=["alignment"],
    response_model=None,
    response_class=Response,
    responses={
        200: {
            "description": "The newest frame, as a JPEG. `X-Frame-Seq` numbers it.",
            "content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}},
        },
        204: {"description": "No frame is newer than `after`."},
        **errors(401, 422, 429),
    },
    dependencies=READ,
)
async def get_alignment_frame(
    ctx: Ctx,
    after: Annotated[int, Query(ge=0, description="The `X-Frame-Seq` of the last frame.")] = 0,
) -> Response:
    """Return the newest live-view frame, for a client that cannot use the WebSocket.

    Each call keeps the stream to `core` open for a few seconds. Call it about twice a second
    while the page shows the live view.
    """
    ctx.hub.touch()
    latest = ctx.hub.latest
    if latest is None or latest.seq <= after:
        return Response(status_code=204)
    return Response(
        latest.frame.jpeg,
        media_type="image/jpeg",
        headers={"X-Frame-Seq": str(latest.seq), "Cache-Control": "no-store"},
    )


@router.websocket("/alignment/stream")
async def alignment_stream(websocket: WebSocket) -> None:
    # The live view pushes the newest frame. Each frame is a text message with its state,
    # `{"type": "state", "state": {...}}`, and then a binary message with the JPEG. The text
    # message `{"type": "idle"}` says that no frame arrived for `stall_s` seconds, and
    # `{"type": "error", "code": ...}` says that the stream to core broke. The focus history of
    # the state holds the whole history in the first message of a viewer (`reset` true) and the new
    # points in the next ones (`reset` false). With
    # `require_token_for_reads`, the client sends `{"type": "auth", "token": "..."}` first.
    ctx = websocket.app.state.ctx
    await _serve_stream(ctx, websocket, ctx.hub, ctx.settings.live.max_fps, _alignment_describer)


# --- The video of Polaris and the rolling seeing value ---


@router.get(
    "/seeing/live",
    operation_id="get_seeing_live",
    summary="Get the rolling seeing value",
    tags=["live"],
    response_model=LiveSeeingResponse,
    responses=errors(401, 404, 429, 502, 503),
    dependencies=READ,
)
def get_seeing_live(ctx: Ctx) -> JSONResponse:
    """Return the seeing of the newest seconds of the fast stream, as a provisional value.

    `core` repeats the estimate every few seconds and does not store it. The estimate uses the
    estimator of the stored windows on the newest `span_s` seconds of frames, so it jitters more
    than a stored value does. `t_utc` is the end of the span and `age_s` is how old the value is:
    `age_s` keeps growing between the fast periods of a cycle, when no new frames arrive. The
    answer is `404 no_data` when `core` has no value, which is the case until a fast period has
    gathered enough frames.
    """
    live = ctx.core.live_seeing()
    if live is None:
        raise ApiError(
            404,
            "no_data",
            "Core has no live seeing value: the fast stream is not running, or it has not gathered "
            "enough frames yet.",
        )
    age_s = round(max(0.0, (ctx.clock.utc_ns() - live.t_utc_ns) / NS_PER_S), 3)
    body = LiveSeeingResponse(**live.model_dump(), t_utc=iso(live.t_utc_ns), age_s=age_s)
    return JSONResponse(body.model_dump(mode="json"))


POLARIS_FRAME_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": (
            "The newest frame, as a PNG image. `X-Frame-Seq` numbers it, and `X-Frame-State` holds "
            "its state."
        ),
        "headers": {
            "X-Frame-Seq": {
                "description": "The sequence number of the frame. Pass it as `after`.",
                "schema": {"type": "integer"},
            },
            "X-Frame-State": {
                "description": (
                    "The state of the frame as one line of JSON, with the sequence number in `seq`."
                ),
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/PolarisState"}}
                },
            },
        },
        "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}},
    },
    204: {"description": "No frame is newer than `after`."},
    **errors(401, 422, 429),
}


@router.get(
    "/polaris/frame",
    operation_id="get_polaris_frame",
    summary="Poll for the newest frame of the video of Polaris",
    tags=["live"],
    response_model=None,
    response_class=Response,
    responses=POLARIS_FRAME_RESPONSES,
    dependencies=READ,
)
async def get_polaris_frame(
    ctx: Ctx,
    after: Annotated[int, Query(ge=0, description="The `X-Frame-Seq` of the last frame.")] = 0,
) -> Response:
    """Return the newest frame of the video of Polaris, for a client that cannot use the WebSocket.

    The image is a lossless 8-bit grayscale PNG with the size of the ROI, so a page can magnify it
    without interpolation. The state of the same frame travels in the header `X-Frame-State`, so
    one request gets both. Each call keeps the stream to `core` open for a few seconds. Call it as
    often as `ui.polaris_max_fps` says while the page shows the video. A value of `after` above
    the newest number means that the server restarted, and the call returns the newest frame.
    While the rapid focus mode runs, the header leaves out `rapid_focus.readings`, which are too
    large for a header: poll `GET /alignment/state` for them.
    """
    hub = ctx.polaris_hub
    hub.touch()
    latest = hub.latest
    if latest is None or latest.seq == after:
        return Response(status_code=204)
    state, image = _polaris_parts(latest.frame)
    rapid = state.get("rapid_focus")
    if isinstance(rapid, dict):
        rapid["readings"] = None
    return Response(
        image,
        media_type="image/png",
        headers={
            "X-Frame-Seq": str(latest.seq),
            "X-Frame-State": json.dumps(state, separators=(",", ":")),
            "Cache-Control": "no-store",
        },
    )


@router.websocket("/polaris/stream")
async def polaris_stream(websocket: WebSocket) -> None:
    # The video pushes the newest frame, with the protocol of `/alignment/stream`: a text message
    # `{"type": "state", "state": {...}}` and then a binary message with the PNG image. The text
    # message `{"type": "idle"}` says that no frame arrived for `stall_s` seconds, and
    # `{"type": "error", "code": ...}` says that the stream to core broke. The readings of
    # the rapid focus mode in the state hold all of them in the first message of a viewer
    # (`reset` true) and the new ones in the next messages (`reset` false). With
    # `require_token_for_reads`, the client sends `{"type": "auth", "token": "..."}` first.
    ctx = websocket.app.state.ctx
    await _serve_stream(
        ctx, websocket, ctx.polaris_hub, ctx.settings.live.polaris_max_fps, _polaris_describer
    )


def _add_series(path: str, record_type: str, name: str) -> None:
    """Add the `latest` route and the history route of one record type."""

    @router.get(
        f"/{path}/latest",
        operation_id=f"get_{path}_latest",
        summary=f"Get the newest {record_type} record",
        tags=["records"],
        response_model=None,
        responses={
            200: json_response(name, f"The newest `{record_type}` record."),
            **errors(401, 404, 429, 503),
        },
        dependencies=READ,
    )
    def latest(ctx: Ctx) -> JSONResponse:
        record = ctx.data.latest(record_type)
        if record is None:
            raise ApiError(404, "no_data", f"The store holds no {record_type} record yet.")
        return JSONResponse(record)

    latest.__doc__ = (
        f"Return the newest `{record_type}` record. A missing value is `null`, and `quality` says "
        "why. A field that the server withholds is `null` with the note `withheld`."
    )

    @router.get(
        f"/{path}",
        operation_id=f"get_{path}_history",
        summary=f"Get the history of {record_type} records",
        tags=["records"],
        response_model=None,
        responses={
            200: json_response(f"{name}Page", f"A page of `{record_type}` items."),
            **errors(401, 422, 429, 503),
        },
        dependencies=READ,
    )
    def history(
        ctx: Ctx,
        from_: Annotated[
            str | None,
            Query(
                alias="from",
                description="The start of the range, included. Default: 24 hours before `to`.",
            ),
        ] = None,
        to: Annotated[
            str | None, Query(description="The end of the range, excluded. Default: now.")
        ] = None,
        step: Annotated[
            Step, Query(description="The bucket width. `raw` returns the stored records.")
        ] = Step.RAW,
        limit: Annotated[int | None, Query(ge=1, le=10_000, description="Items per page.")] = None,
        cursor: Annotated[
            str | None, Query(max_length=128, description="The `next_cursor` of a page.")
        ] = None,
        fields: Annotated[
            str | None,
            Query(
                max_length=1024,
                description="A comma-separated list of the fields to return. Default: all fields "
                "except the arrays of numbers.",
            ),
        ] = None,
    ) -> JSONResponse:
        cls = ctx.data.record_class(record_type)
        time_range = ctx.data.resolve_range(from_, to)
        page = ctx.data.history(
            record_type,
            time_range=time_range,
            step=step,
            limit=ctx.data.resolve_limit(limit),
            cursor=cursor,
            wanted=ctx.data.parse_fields(cls, fields),
        )
        return JSONResponse(_page_body(record_type, page, ctx.clock.utc_ns()))

    history.__doc__ = (
        f"Return the `{record_type}` records between `from` (included) and `to` (excluded), "
        "oldest first.\n\n"
        "With `step`, each item is a bucket that starts at a multiple of the step in UTC. A "
        "numeric field is the mean of the bucket, and a list of flags is the union of the flags, "
        "so a flag that applied to any record shows. `n_samples` counts the records of the "
        "bucket. Pass `next_cursor` back as `cursor` to read the next page."
    )


for _path, _record_type, _name in SERIES:
    _add_series(_path, _record_type, _name)


def _page_body(record_type: str, page: Any, now_ns: int) -> dict[str, Any]:
    time_range: TimeRange = page.time_range
    return {
        "record_type": record_type,
        "step": page.step.value,
        "from": iso(time_range.start_ns),
        "to": iso(time_range.end_ns),
        "now": iso(now_ns),
        "items": page.items,
        "next_cursor": page.next_cursor,
    }


# --- The live views --------------------------------------------------------------------------

F = TypeVar("F")
# What a hub frame sends to one client: the JSON state, and the bytes of the image.
Describe = Callable[[F], tuple[dict[str, Any], bytes]]


def _alignment_describer() -> Describe[AlignmentFrame]:
    """The `describe` function of one alignment viewer, which keeps what the viewer has received."""
    cursor = HistoryCursor()  # the focus history that this viewer has received

    def describe(frame: AlignmentFrame) -> tuple[dict[str, Any], bytes]:
        state = frame.state.model_dump(mode="json")
        cursor.delta(state)
        return state, frame.jpeg

    return describe


def _polaris_parts(frame: PolarisFrame) -> tuple[dict[str, Any], bytes]:
    return frame.state.model_dump(mode="json"), frame.image


def _polaris_describer() -> Describe[PolarisFrame]:
    """The `describe` function of one viewer of the video of Polaris.

    It keeps what the viewer has received of the readings of the rapid focus mode.
    """
    cursor = HistoryCursor(RAPID_PATH, RAPID_LISTS)

    def describe(frame: PolarisFrame) -> tuple[dict[str, Any], bytes]:
        state, image = _polaris_parts(frame)
        cursor.delta(state)
        return state, image

    return describe


async def _serve_stream(
    ctx: WebContext,
    websocket: WebSocket,
    hub: FrameHub[F],
    max_fps: float,
    make_describe: Callable[[], Describe[F]],
) -> None:
    """Serve one WebSocket viewer of a live view. See `alignment_stream` and `polaris_stream`.

    The two views share the protocol: the token, the limit of the viewers (`max_clients` for each
    hub), and the messages. `max_fps` caps the frames that this viewer gets, and `make_describe`
    gives this viewer its function that splits a frame into its JSON state and its image.
    """
    client = ctx.client_key(websocket)
    needs_token = ctx.settings.require_token_for_reads
    authorized = not needs_token
    header = websocket.headers.get("authorization")
    if needs_token and header is not None:
        try:
            ctx.check_token(client, header)
            authorized = True
        except ApiError:
            await websocket.close(code=WS_CLOSE_POLICY)
            return
    await websocket.accept()
    if not authorized and not await _read_token(ctx, websocket, client):
        await websocket.close(code=WS_CLOSE_POLICY)
        return
    if hub.viewers >= ctx.settings.live.max_clients:
        # The close comes after the handshake on purpose. A close before it turns into an HTTP
        # 403, which a browser reports as code 1006 (a failed connection). The page would take
        # "busy" for "broken" and fall back to polling, which the limit does not cover.
        await websocket.close(code=WS_CLOSE_TRY_LATER)
        return
    describe = make_describe()
    async with hub.subscribe() as subscription:
        sender = asyncio.ensure_future(
            _send_frames(ctx, websocket, hub, subscription, max_fps, describe)
        )
        receiver = asyncio.ensure_future(_drain(websocket))
        done, pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in (*done, *pending):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
    with contextlib.suppress(Exception):
        await websocket.close()


async def _read_token(ctx: WebContext, websocket: WebSocket, client: str) -> bool:
    """Read the first message of a client that must send its token. Returns whether it is right."""
    try:
        text = await asyncio.wait_for(websocket.receive_text(), WS_AUTH_TIMEOUT_S)
        message = json.loads(text)
        if not isinstance(message, dict) or message.get("type") != "auth":
            return False
        token = message.get("token")
        if not isinstance(token, str):
            return False
        ctx.check_token(client, f"Bearer {token}")
    except Exception:  # a bad message, a wrong token, a timeout, or a client that left
        return False
    return True


async def _drain(websocket: WebSocket) -> None:
    """Read and ignore what the client sends, so that a disconnect ends the stream at once."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


async def _send_frames(
    ctx: WebContext,
    websocket: WebSocket,
    hub: FrameHub[F],
    subscription: Subscription[F],
    max_fps: float,
    describe: Describe[F],
) -> None:
    stall_s = ctx.settings.live.stall_s
    min_interval_s = 1.0 / max_fps
    clock = ctx.clock
    last_sent_ns: int | None = None
    while True:
        update = await subscription.next_update(stall_s)
        if update is None:
            await websocket.send_text(json.dumps({"type": "idle"}))
            continue
        if update.error_changed and update.error is not None:
            await websocket.send_text(json.dumps({"type": "error", "code": update.error}))
        if update.frame is None:
            continue
        frame = update.frame
        if last_sent_ns is not None:
            wait_s = min_interval_s - (clock.monotonic_ns() - last_sent_ns) / NS_PER_S
            if wait_s > 0:
                await hub.sleep(wait_s)
                frame = subscription.newest() or frame  # the newest frame wins
        state, image = describe(frame.frame)
        await websocket.send_text(json.dumps({"type": "state", "state": state}))
        await websocket.send_bytes(image)
        last_sent_ns = clock.monotonic_ns()


__all__ = ["API_VERSION", "PREFIX", "build_status", "router"]
