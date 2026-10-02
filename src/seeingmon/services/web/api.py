"""The routes of the REST API v1.

`router` holds the routes of `/api/v1`. The handlers are plain functions, which FastAPI runs in its
thread pool, because the store and the link to `core` block. The alignment routes are `async`,
because they share the event loop with the live view.

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
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response

import seeingmon
from seeingmon.clock import NS_PER_S
from seeingmon.scheduler.commands import Command, RejectReason, StopAlignment
from seeingmon.services.web.context import WebContext
from seeingmon.services.web.contract import AlignmentState, SchedulerView
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
from seeingmon.services.web.live import Subscription
from seeingmon.services.web.models import (
    AlignmentStartRequest,
    ApiInfo,
    BurstRequest,
    CommandResponse,
    CoreLink,
    ErrorResponse,
    EventLevel,
    FaultStatusView,
    HealthResponse,
    ImageFormat,
    ImageItem,
    ImageList,
    ModeRequest,
    Order,
    RecordTime,
    ReplayRequest,
    SchedulerStatusResponse,
    StatusResponse,
    StreamStatus,
    SweepRequest,
    UiSettings,
)
from seeingmon.services.web.privacy import scrub_text
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
            next_attempt=None
            if fault.next_attempt_utc_ns is None
            else iso(fault.next_attempt_utc_ns),
            next_step=fault.next_step,
        ),
        queued_tasks=view.queued_tasks,
        survey_pending=view.survey_pending,
        alignment_idle_s=view.alignment_idle_s,
    )


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
    name = type(command).__name__
    result = ctx.core.submit(command)
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
    # `{"type": "error", "code": ...}` says that the stream to core broke. With
    # `require_token_for_reads`, the client sends `{"type": "auth", "token": "..."}` first.
    await _serve_stream(websocket.app.state.ctx, websocket)


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


# --- The live view ---------------------------------------------------------------------------


async def _serve_stream(ctx: WebContext, websocket: WebSocket) -> None:
    """Serve one WebSocket viewer of the alignment stream. See `alignment_stream`."""
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
    if ctx.hub.viewers >= ctx.settings.live.max_clients:
        # The close comes after the handshake on purpose. A close before it turns into an HTTP
        # 403, which a browser reports as code 1006 (a failed connection). The page would take
        # "busy" for "broken" and fall back to polling, which the limit does not cover.
        await websocket.close(code=WS_CLOSE_TRY_LATER)
        return
    async with ctx.hub.subscribe() as subscription:
        sender = asyncio.ensure_future(_send_frames(ctx, websocket, subscription))
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


async def _send_frames(ctx: WebContext, websocket: WebSocket, subscription: Subscription) -> None:
    live = ctx.settings.live
    min_interval_s = 1.0 / live.max_fps
    clock = ctx.clock
    last_sent_ns: int | None = None
    while True:
        update = await subscription.next_update(live.stall_s)
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
                await ctx.hub.sleep(wait_s)
                frame = subscription.newest() or frame  # the newest frame wins
        await websocket.send_text(
            json.dumps({"type": "state", "state": frame.frame.state.model_dump(mode="json")})
        )
        await websocket.send_bytes(frame.frame.jpeg)
        last_sent_ns = clock.monotonic_ns()


__all__ = ["API_VERSION", "PREFIX", "build_status", "router"]
