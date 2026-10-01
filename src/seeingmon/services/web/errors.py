"""The errors of the REST API, and the one JSON shape that every error response has.

Every error response is `{"error": {"code": "<word>", "message": "<sentence>", "details": [...]}}`.
`details` appears for an invalid request and lists one entry for each invalid field. A response
never carries a stack trace, the value that a client sent, or text from another process. An error
that the server did not expect gets the code `internal_error` and a generic sentence, and the
server logs the real error.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from seeingmon.services.web.core_client import CoreProtocolError, CoreUnavailableError
from seeingmon.services.web.data import InvalidQueryError, StoreUnavailableError

_log = logging.getLogger(__name__)

CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "invalid_request",
    429: "rate_limited",
    500: "internal_error",
    502: "core_error",
    503: "unavailable",
}
MESSAGES = {
    400: "The request is not valid.",
    401: "The request needs a valid token.",
    403: "The request is not allowed.",
    404: "The resource does not exist.",
    405: "The method is not allowed for this resource.",
    409: "The request conflicts with the state of the system.",
    413: "The request body is too large.",
    422: "The request is not valid.",
    429: "Too many requests. Wait and try again.",
    500: "The server hit an unexpected error.",
    502: "The core process answered something that the web process cannot use.",
    503: "The service is not available.",
}


class ApiError(StarletteHTTPException):
    """An error with a status, a code, and a sentence for the client.

    It derives from the Starlette exception, so FastAPI passes it through the places where it
    would otherwise turn an unknown exception into a 400.
    """

    def __init__(
        self,
        status_code: int,
        code: str | None = None,
        message: str | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        details: list[dict[str, str]] | None = None,
    ) -> None:
        text = message or MESSAGES.get(status_code, "The request failed.")
        super().__init__(status_code, detail=text, headers=dict(headers) if headers else None)
        self.code = code or CODES.get(status_code, "error")
        self.message = text
        self.details = details


def error_body(
    code: str, message: str, details: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    """The JSON of an error response."""
    return {"error": {"code": code, "message": message, "details": details}}


def error_response(error: ApiError) -> JSONResponse:
    return JSONResponse(
        error_body(error.code, error.message, error.details),
        status_code=error.status_code,
        headers=error.headers,
    )


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(exc)


async def _http_exception(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    status = exc.status_code
    return error_response(ApiError(status, headers=exc.headers))


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    details = [
        {
            "field": ".".join(str(part) for part in item["loc"]),
            "message": str(item["msg"]),
            "type": str(item["type"]),
        }
        for item in exc.errors()
    ]
    return error_response(ApiError(422, details=details))


async def _invalid_query(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, InvalidQueryError)
    return error_response(ApiError(422, "invalid_request", str(exc)))


async def _store_unavailable(request: Request, exc: Exception) -> JSONResponse:
    return error_response(
        ApiError(
            503,
            "store_unavailable",
            "The store cannot be read.",
            headers={"Retry-After": "5"},
        )
    )


async def _core_unavailable(request: Request, exc: Exception) -> JSONResponse:
    return error_response(
        ApiError(
            503,
            "core_unavailable",
            "The core process does not answer.",
            headers={"Retry-After": "5"},
        )
    )


async def _core_protocol(request: Request, exc: Exception) -> JSONResponse:
    return error_response(ApiError(502, "core_error"))


async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    # The server logs the traceback when it gets the re-raised exception, so log one line here.
    _log.error("an unexpected error ended a request: %s", type(exc).__name__)
    return error_response(ApiError(500))


def register_error_handlers(app: FastAPI) -> None:
    """Install the handlers that give every error the same JSON shape."""
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_exception)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(InvalidQueryError, _invalid_query)
    app.add_exception_handler(StoreUnavailableError, _store_unavailable)
    app.add_exception_handler(CoreUnavailableError, _core_unavailable)
    app.add_exception_handler(CoreProtocolError, _core_protocol)
    app.add_exception_handler(Exception, _unexpected)
