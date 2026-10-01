"""The state that the routes of the web process share, and the access rule.

`WebContext` holds the pieces that `create_app` wires together: the data readers, the client of
`core`, the token verifier, the rate limiters, and the alignment hub. It also decides who may do
what. The access rule has two levels:

- **Reads** are open on the LAN. With `require_token_for_reads`, a read needs the token too.
- **Every `POST`** needs the token as `Authorization: Bearer <token>`. Without a token hash in the
  configuration, the server refuses every command with `403 commands_disabled`.

**Order of the checks.** A command first counts against the per-client limit of commands, so a
flood gets `429` before the server does any work. The server then refuses the command if no token
is configured. Then it checks the failure limit of the client: a client that failed the token check
too often gets `429` and no check at all, so nobody can guess the token at the speed of the
network. Last, it checks the token. A missing or malformed token gets `401` without counting
against the failure limit, because it guesses nothing. A wrong token gets the same `401` and counts.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from starlette.requests import HTTPConnection, Request

from seeingmon.clock import Clock
from seeingmon.services.web.auth import RateLimiter, TokenVerifier, bearer_token
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.contract import CoreStatus
from seeingmon.services.web.core_client import CoreClient, CoreError
from seeingmon.services.web.data import StoreData, StoreUnavailableError
from seeingmon.services.web.errors import ApiError
from seeingmon.services.web.health import HealthReport, evaluate_health
from seeingmon.services.web.images import ImageStore
from seeingmon.services.web.live import AlignmentHub

_log = logging.getLogger(__name__)

BEARER_CHALLENGE = {"WWW-Authenticate": 'Bearer realm="seeing-monitor"'}


class WebContext:
    """Everything that a route needs, and the access rule."""

    def __init__(
        self,
        *,
        settings: WebSettings,
        data: StoreData,
        images: ImageStore,
        core: CoreClient,
        clock: Clock,
        verifier: TokenVerifier,
        hub: AlignmentHub,
        profile: Mapping[str, Any] | None,
        config_view: Mapping[str, Any] | None,
        station_id: str | None,
        demo: bool,
    ) -> None:
        self.settings = settings
        self.data = data
        self.images = images
        self.core = core
        self.clock = clock
        self.verifier = verifier
        self.hub = hub
        self.profile = profile
        self.config_view = config_view
        self.station_id = station_id
        self.demo = demo
        limits = settings.rate_limit
        self.command_limiter = RateLimiter(
            clock,
            limit=limits.commands_per_window,
            window_s=limits.window_s,
            max_clients=limits.max_clients,
        )
        self.failure_limiter = RateLimiter(
            clock,
            limit=limits.auth_failures_per_window,
            window_s=limits.auth_failure_window_s,
            max_clients=limits.max_clients,
        )

    # --- Access ----------------------------------------------------------------------------

    @staticmethod
    def client_key(connection: HTTPConnection) -> str:
        """The remote address of a client, which the rate limits count by."""
        return connection.client.host if connection.client else "unknown"

    @staticmethod
    def _too_many(retry_after_s: int) -> ApiError:
        return ApiError(429, headers={"Retry-After": str(retry_after_s)})

    def check_token(self, client: str, header: str | None) -> None:
        """Check a token. Raises `ApiError` 429 for a client that failed too often, and 401."""
        blocked = self.failure_limiter.check(client)
        if blocked is not None:
            raise self._too_many(blocked)
        token = bearer_token(header)
        if token is None:
            raise ApiError(401, headers=BEARER_CHALLENGE)
        if not self.verifier.verify(token):
            self.failure_limiter.record(client)
            _log.warning("a client sent a token that does not match")
            raise ApiError(401, headers=BEARER_CHALLENGE)

    def authorize_read(self, request: Request) -> None:
        """The dependency of a read route. It checks the token only when reads need one."""
        if self.settings.require_token_for_reads:
            self.check_token(self.client_key(request), request.headers.get("authorization"))

    def authorize_command(self, request: Request) -> None:
        """The dependency of a `POST` route. See the module documentation for the order."""
        client = self.client_key(request)
        wait = self.command_limiter.acquire(client)
        if wait is not None:
            raise self._too_many(wait)
        if not self.verifier.enabled:
            raise ApiError(
                403,
                "commands_disabled",
                "No API token is configured, so the server refuses every command.",
            )
        self.check_token(client, request.headers.get("authorization"))

    # --- Health and status -----------------------------------------------------------------

    def probe_core(self) -> CoreStatus | None:
        """The status of `core`, or `None` when it does not answer."""
        try:
            return self.core.status()
        except CoreError:
            return None

    def judge(self, core_status: CoreStatus | None) -> HealthReport:
        """Judge the health. Pass what `probe_core` returned: `None` means that core is silent."""
        store_ok = True
        record = None
        try:
            record = self.data.latest("health")
        except StoreUnavailableError:
            store_ok = False
        return evaluate_health(
            now_ns=self.clock.utc_ns(),
            record=record,
            store_ok=store_ok,
            core_ok=core_status is not None,
            max_age_s=self.settings.health_max_age_s,
        )
