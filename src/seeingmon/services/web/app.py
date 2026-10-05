"""The FastAPI application of the web process.

`create_app(settings, store, images, core)` wires the pieces together and returns the app:

- the REST API under `/api/v1` (`seeingmon.services.web.api`), with its OpenAPI description at
  `/api/v1/openapi.json`,
- the static UI at `/` (`seeingmon/services/web/static`, plain HTML, CSS, and JavaScript with no
  external asset),
- the middleware for the host check (the outermost layer), the security headers, the cache headers,
  the body limit, and compression.

The app takes everything it needs as arguments, so a test builds it with a temporary store and a
`FakeCoreClient`, and `seeingmon web --demo` builds it with synthetic data. Nothing here opens a
file or a connection by itself. `store` reads the database through a read-only connection (a
`StoreReader`, or the `ReopeningReader` that `seeingmon web` uses), and `images` reads the preview
and FITS files. Neither can write.
"""

from __future__ import annotations

import contextlib
import mimetypes
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from starlette.routing import Match, Mount
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from seeingmon.clock import Clock, SystemClock
from seeingmon.services.web.api import API_VERSION, PREFIX, router
from seeingmon.services.web.auth import TokenVerifier
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.context import WebContext
from seeingmon.services.web.core_client import CoreClient
from seeingmon.services.web.data import StoreData, StoreSource
from seeingmon.services.web.errors import register_error_handlers
from seeingmon.services.web.images import ImageStore
from seeingmon.services.web.live import AlignmentHub, PolarisHub, Sleep
from seeingmon.services.web.middleware import (
    AllowedHosts,
    BodyLimit,
    SecurityHeaders,
    SelectiveGZip,
)
from seeingmon.services.web.privacy import public_config, scrub_json
from seeingmon.services.web.schemas import polaris_components, record_components

STATIC_DIR = Path(__file__).resolve().parent / "static"

# The type of a static file must not depend on the type registry of the operating system. On
# Windows, the registry can map `.js` to `application/javascript` or even to `text/plain`, and
# Python 3.11 and 3.13 disagree on `.js` too. With `X-Content-Type-Options: nosniff`, a browser
# refuses to run a script that arrives as `text/plain`, so the app sets the types that it serves.
STATIC_TYPES = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "text/javascript",
    ".svg": "image/svg+xml",
    ".json": "application/json",
}
for _suffix, _type in STATIC_TYPES.items():
    mimetypes.add_type(_type, _suffix)
OPENAPI_URL = f"{PREFIX}/openapi.json"

DESCRIPTION = """\
The REST API of the seeing monitor: a fixed camera that points at Polaris, and a Raspberry Pi that
measures the atmospheric seeing and the sky quality unattended.

**Versioning.** The API lives under `/api/v1`. Within `v1`, a change only adds a field or an
endpoint. A breaking change creates `/api/v2`, and `v1` stays for at least one release.

**Access.** Reads are open on the LAN by default, and a setting can require the token for reads
too. Every `POST` and `DELETE` needs the token as `Authorization: Bearer <token>`. The server
refuses every command when no token is configured. A client that sends too many requests gets 429
with a `Retry-After` header.

**Hosts.** The server answers a request only when its `Host` header names an allowed host: the
loopback names, the addresses that the server listens on, and the entries of the `allowed_hosts`
setting. A server that listens on a wildcard address also admits any IP address and the names of
its device. Any other host gets `400 host_not_allowed`, and the message names the setting. A
WebSocket handshake with an `Origin` header needs an allowed host there too. This rule does not
change who may read or send commands.

**Values.** Times are ISO 8601 UTC strings. A field name carries its unit, such as
`seeing_fwhm_arcsec`. A missing value is `null`, and the `quality` object says why. Every error
has the same shape: `{"error": {"code": ..., "message": ..., "details": ...}}`.

**History.** A history route takes `from` (included) and `to` (excluded), and `step` (`raw`, `1m`,
`10m`, or `1h`). A numeric field of a bucket is the mean, and a list of flags is the union. A page
holds at most `limit` items, and `next_cursor` points to the next page.
"""

TAGS = [
    {"name": "status", "description": "The health and the state of every component."},
    {"name": "records", "description": "Seeing, sky quality, pointing, and events."},
    {"name": "images", "description": "The preview images and the FITS frames."},
    {"name": "commands", "description": "Commands for the scheduler. Each needs the token."},
    {"name": "alignment", "description": "The alignment helper and its live view."},
    {
        "name": "live",
        "description": "The live video of Polaris and the rolling seeing value of the fast stream.",
    },
    {
        "name": "dark",
        "description": "The dark library, and the dark session that adds a set to it.",
    },
    {
        "name": "flat",
        "description": "The flat library, and the flat session that adds a flat to it.",
    },
    {"name": "reference", "description": "The hardware profile and the configuration."},
]


class UiMount(Mount):
    """Mount the static UI at `/`, but leave every `/api` path to the API router.

    A plain mount at `/` matches every path, so a request with the wrong method for an API route
    would reach the static files and get a 404 instead of the 405 that names the problem.
    """

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        if scope["type"] == "http" and (
            scope["path"] == "/api" or scope["path"].startswith("/api/")
        ):
            return Match.NONE, {}
        return super().matches(scope)


def _install_openapi(app: FastAPI) -> None:
    """Build the OpenAPI document from the routes, and add the schemas of the records."""

    def build() -> dict[str, Any]:
        if app.openapi_schema is not None:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            summary=app.summary,
            description=app.description,
            routes=app.routes,
            tags=TAGS,
        )
        components = schema.setdefault("components", {})
        components.setdefault("schemas", {}).update(record_components())
        components["schemas"].update(polaris_components())
        components["securitySchemes"] = {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "description": (
                    "The API token. The server stores only its hash. Every POST and DELETE needs "
                    "it, and a read needs it when the server sets `require_token_for_reads`."
                ),
            }
        }
        app.openapi_schema = schema
        return schema

    app.openapi = build  # type: ignore[method-assign]


def build_context(
    settings: WebSettings,
    store: StoreSource,
    images: ImageStore,
    core: CoreClient,
    *,
    clock: Clock | None = None,
    token_hash: str | None = None,
    profile: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    station_id: str | None = None,
    demo: bool = False,
    sleep: Sleep | None = None,
) -> WebContext:
    """Wire the readers, the client of `core`, the access rule, and the live view together.

    `token_hash` is the stored hash of the API token (see `seeingmon.services.web.auth`), or
    `None`, which makes the server refuse every command. `profile` is the output of
    `profile_summary`, and `config` is the output of `Config.effective(redact=True,
    omit_site=True)`. The context serves the profile as it is and the configuration through
    `public_config`, which drops the sections that describe the installation. `station_id` limits
    the records to one station. `sleep` replaces `asyncio.sleep` in the alignment hub, for tests.

    Raises `AuthConfigError` for a malformed `token_hash` and `ConfigError` for a setting that
    names a field that no record has.
    """
    clock = clock or SystemClock()
    hub_options: dict[str, Any] = {"idle_s": settings.live.idle_s}
    if sleep is not None:
        hub_options["sleep"] = sleep
    return WebContext(
        settings=settings,
        data=StoreData(store, settings, clock, station_id=station_id),
        images=images,
        core=core,
        clock=clock,
        verifier=TokenVerifier(token_hash, clock),
        hub=AlignmentHub(core, clock, **hub_options),
        polaris_hub=PolarisHub(core, clock, **hub_options),
        profile=None if profile is None else scrub_json(profile),
        config_view=None if config is None else public_config(config),
        station_id=station_id,
        demo=demo,
    )


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await app.state.ctx.hub.close()
    await app.state.ctx.polaris_hub.close()


def create_app(
    settings: WebSettings,
    store: StoreSource,
    images: ImageStore,
    core: CoreClient,
    *,
    clock: Clock | None = None,
    token_hash: str | None = None,
    profile: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    station_id: str | None = None,
    demo: bool = False,
    sleep: Sleep | None = None,
) -> FastAPI:
    """Build the application. The arguments are those of `build_context`.

    The context sits in `app.state.ctx`, and the routes read it from there on every request. So
    one app can serve a new context: a test that builds hundreds of contexts builds the routes
    once, which saves most of its time.
    """
    app = FastAPI(
        title="Seeing monitor API",
        version=API_VERSION,
        summary="Seeing, sky quality, pointing, images, and commands.",
        description=DESCRIPTION,
        openapi_url=OPENAPI_URL,
        docs_url=None,
        redoc_url=None,
        lifespan=_lifespan,
    )
    app.state.ctx = build_context(
        settings,
        store,
        images,
        core,
        clock=clock,
        token_hash=token_hash,
        profile=profile,
        config=config,
        station_id=station_id,
        demo=demo,
        sleep=sleep,
    )
    register_error_handlers(app)
    app.include_router(router)
    app.router.routes.append(
        UiMount("/", app=StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    )
    app.add_middleware(
        SelectiveGZip,
        skip_prefixes=(
            f"{PREFIX}/images",
            f"{PREFIX}/alignment/frame",
            f"{PREFIX}/polaris/frame",
        ),
    )
    app.add_middleware(BodyLimit, max_bytes=settings.max_body_bytes)
    app.add_middleware(SecurityHeaders)
    # The last middleware added is the outermost. The host check runs before anything else.
    app.add_middleware(AllowedHosts, allowed=lambda: app.state.ctx.allowed_hosts)
    _install_openapi(app)
    return app
