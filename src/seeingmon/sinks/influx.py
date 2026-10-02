"""The InfluxDB sink: it writes rows as line protocol over HTTP.

**Layout.** `seeingmon.records.sink_mapping` decides the layout, and this module applies it. A
record type is a measurement with the name of the type. The tags are `station` and `profile`. The
timestamp of a point is `t_utc_ns`, in nanoseconds. Every other field is a field of the point:

| Field kind | Line protocol |
|---|---|
| `int` | `42i` |
| `float` | `1.5`, always with a decimal point or an exponent, so a field never changes type |
| `bool` | `true` or `false` |
| `str` | `"text"`, with `"` and `\\` escaped |
| `json` (a list or a dict) | the JSON text, as a string |
| `bytes` | the base64 text, as a string |

A `None` value leaves its field out of the point, and so does a float that is not finite. A new
revision of a result has the same measurement, tags, and timestamp as the old one, so InfluxDB
replaces the point and merges the fields. A field that a revision leaves out stays at its old
value.

**Tags.** InfluxDB cannot carry a newline in a tag, and it cannot tell a backslash before a
separator from an escape. The sink replaces a backslash, a newline, a carriage return, and a NUL
character in a tag value with `_`. A station ID and a profile ID never hold them in practice.

**Versions.** Version 2 posts to `/api/v2/write?org=&bucket=&precision=ns` with the header
`Authorization: Token <token>`. InfluxDB 3 accepts the same call. Version 1 posts to
`/write?db=&precision=ns` (and `rp=` when you set a retention policy), with HTTP basic
authentication when you give a user name and a password.

**Errors.** The sink sends one request for each batch and waits at most `timeout_s`. A network
error, a timeout, a status of 5xx, 408, or 429 raises `SinkError(retryable=True)`: the forwarder
keeps the cursor and retries. Any other status raises `SinkError(retryable=False)`: the forwarder
parks the sink. That includes a redirect, because following one could turn a write into a read.
One reply is a success although it is an error status: a batch that the server refuses only
because its points lie beyond the retention policy of the bucket. Retrying cannot change that, so
the sink logs it and goes on. No error message holds the token, the password, or the address.
"""

from __future__ import annotations

import base64
import http.client
import json
import logging
import math
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from seeingmon.records.base import FieldKind, Record, get_record_type
from seeingmon.records.sink_mapping import sink_mapping
from seeingmon.sinks.base import SinkError, StoredRow
from seeingmon.sinks.config import InfluxSinkConfig

logger = logging.getLogger(__name__)

_TAG_UNSAFE = re.compile(r"[\\\n\r\x00]")
_ERROR_BODY_BYTES = 2000
_MESSAGE_CHARS = 200


# --- line protocol ------------------------------------------------------------------------------


def escape_measurement(text: str) -> str:
    """Escape a measurement name: a comma and a space."""
    return text.replace(",", "\\,").replace(" ", "\\ ")


def escape_key(text: str) -> str:
    """Escape a tag key, a tag value, or a field key: a comma, an equals sign, and a space."""
    return text.replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")


def sanitize_tag_value(text: str) -> str:
    """Replace the characters that a tag value cannot carry (see the module documentation)."""
    return _TAG_UNSAFE.sub("_", text)


def quote_string(text: str) -> str:
    """Quote a string field value: a backslash and a double quote get a backslash."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def encode_field(kind: FieldKind, value: Any) -> str | None:
    """Encode one field value, or return `None` when the point leaves the field out."""
    if value is None:
        return None
    if kind == "bool":
        return "true" if value else "false"
    if kind == "int":
        return f"{int(value)}i"
    if kind == "float":
        number = float(value)
        return repr(number) if math.isfinite(number) else None
    if kind == "json":
        return quote_string(json.dumps(value, separators=(",", ":"), ensure_ascii=False))
    return quote_string(str(value))  # `str`, and `bytes` as base64 text


def format_point(record: str | type[Record], values: Mapping[str, Any]) -> str:
    """Build the line of one point from the values of a stored row.

    `record` is a record type name or class. `values` holds the `Record.to_row` values that every
    read of the store returns. Raises `KeyError` when the row has no `t_utc_ns`.
    """
    mapping = sink_mapping(record).influx
    tags = {}
    for tag, field in mapping.tags.items():
        text = values.get(field)
        if isinstance(text, str) and text:
            tags[tag] = sanitize_tag_value(text)
    head = [escape_measurement(mapping.measurement)]
    head += [f"{escape_key(tag)}={escape_key(tags[tag])}" for tag in sorted(tags)]
    fields = []
    for item in mapping.fields:
        encoded = encode_field(item.kind, values.get(item.name))
        if encoded is not None:
            fields.append(f"{escape_key(item.name)}={encoded}")
    if not fields:
        raise ValueError(f"a {mapping.measurement} row has no field to send")
    return f"{','.join(head)} {','.join(fields)} {int(values[mapping.time_field])}"


def encode_batch(record: str | type[Record], rows: Sequence[StoredRow]) -> bytes:
    """Build the body of one write request: one line for each row, with a final newline."""
    lines = [format_point(record, row.values) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


# --- the HTTP client ----------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect. `urllib` would turn a redirected POST into a GET."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def make_opener(
    *, verify_tls: bool = True, use_environment_proxies: bool = True
) -> urllib.request.OpenerDirector:
    """Build the HTTP opener of a sink: no redirects, and optionally no certificate check.

    The opener honors the proxy variables of the environment (`HTTPS_PROXY`) unless you pass
    `use_environment_proxies=False`.
    """
    handlers: list[Any] = [_NoRedirect()]
    if not use_environment_proxies:
        handlers.append(urllib.request.ProxyHandler({}))
    if not verify_tls:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        handlers.append(urllib.request.HTTPSHandler(context=context))
    return urllib.request.build_opener(*handlers)


def _read_reply(error: urllib.error.HTTPError) -> bytes:
    """Read the start of an error reply, and close it. A broken connection gives no body."""
    try:
        return error.read(_ERROR_BODY_BYTES)
    except (OSError, http.client.HTTPException):
        return b""
    finally:
        error.close()


def _short(text: str, limit: int = _MESSAGE_CHARS) -> str:
    return " ".join(text.split())[:limit]


def _server_message(body: bytes, limit: int = _MESSAGE_CHARS) -> str:
    """Read the message of an InfluxDB error reply: JSON with `message` or `error`, or text."""
    text = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except ValueError:
        return _short(text, limit)
    if isinstance(parsed, dict):
        for key in ("message", "error"):
            if isinstance(parsed.get(key), str):
                return _short(parsed[key], limit)
    return _short(text, limit)


def error_reply_message(error: urllib.error.HTTPError, *, limit: int = _MESSAGE_CHARS) -> str:
    """Read an error reply of InfluxDB, close it, and return the message that the server gave.

    The message is the `message` or `error` text of a JSON reply, or else the start of the body, on
    one line and at most `limit` characters (200 by default). A broken connection gives an empty
    message. The SQM-LE reader of `seeingmon.hardware.sqm_influx` uses this function for its own
    error replies. It asks for a longer message, cleans it, and then shortens it.
    """
    return _server_message(_read_reply(error), limit)


_HINTS = {
    401: "check the token or the credentials",
    403: "the token or the user may not write to this bucket or database",
    404: "check the endpoint, the organization, and the bucket or the database",
    413: "the request was too large; lower max_batch_rows",
}


class InfluxSink:
    """Writes rows to InfluxDB. Build it from an `InfluxSinkConfig`, or let `build_sinks` do it.

    Pass the secrets that the configuration holds (`token`, or `password` for version 1), after
    `resolve_credential` read them. Pass `opener` to replace the HTTP client, for example with
    `make_opener(use_environment_proxies=False)` in a test.
    """

    def __init__(
        self,
        name: str,
        config: InfluxSinkConfig,
        *,
        token: str | None = None,
        password: str | None = None,
        opener: urllib.request.OpenerDirector | None = None,
    ) -> None:
        self._name = name
        self._timeout_s = config.timeout_s
        self._max_batch_rows = config.max_batch_rows
        self._record_types = None if config.record_types is None else frozenset(config.record_types)
        self._url = self.write_url(config)
        self._headers = {
            "Content-Type": "text/plain; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": "seeingmon",
        }
        if config.version == 2:
            if token:
                self._headers["Authorization"] = f"Token {token}"
        elif config.username and password:
            raw = f"{config.username}:{password}".encode()
            self._headers["Authorization"] = "Basic " + base64.b64encode(raw).decode("ascii")
        self._opener = opener if opener is not None else make_opener(verify_tls=config.verify_tls)

    def __repr__(self) -> str:
        return f"InfluxSink({self._name!r})"

    @staticmethod
    def write_url(config: InfluxSinkConfig) -> str:
        """The URL that a batch goes to, including the query. It holds no secret."""
        if config.version == 2:
            query = {"org": config.org, "bucket": config.bucket, "precision": "ns"}
            return f"{config.endpoint}/api/v2/write?{urllib.parse.urlencode(query)}"
        query = {"db": config.database, "precision": "ns"}
        if config.retention_policy:
            query["rp"] = config.retention_policy
        return f"{config.endpoint}/write?{urllib.parse.urlencode(query)}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_batch_rows(self) -> int:
        return self._max_batch_rows

    def accepts(self, record_type: str) -> bool:
        if self._record_types is not None:
            return record_type in self._record_types
        try:
            return get_record_type(record_type).storage == "table"
        except KeyError:
            return False

    def send(self, record_type: str, rows: Sequence[StoredRow]) -> None:
        """Write a batch. Returns when the server accepted every point, or raises `SinkError`."""
        body = encode_batch(record_type, rows)
        request = urllib.request.Request(self._url, data=body, headers=self._headers, method="POST")
        try:
            with self._opener.open(request, timeout=self._timeout_s) as response:
                status = response.status
                response.read()
        except urllib.error.HTTPError as exc:
            self._handle_error(exc.code, _read_reply(exc))
            return
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
            OSError,
        ) as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            kind = type(reason if isinstance(reason, BaseException) else exc).__name__
            raise SinkError(
                f"the InfluxDB endpoint of sink {self._name} is not reachable ({kind})",
                retryable=True,
            ) from None
        if not 200 <= status < 300:
            raise SinkError(
                f"the InfluxDB endpoint of sink {self._name} answered HTTP {status}, "
                "and a write should answer 204",
                retryable=False,
            )

    def _handle_error(self, status: int, body: bytes) -> None:
        message = _server_message(body)
        if status in (400, 422) and "beyond retention policy" in message.lower():
            logger.warning(
                "sink %s: InfluxDB dropped points that lie beyond the retention policy", self._name
            )
            return  # the server will never take them, so retrying only blocks the sink
        retryable = status >= 500 or status in (408, 429)
        hint = "" if retryable else _HINTS.get(status, "")
        if 300 <= status < 400:
            hint = "the server redirected the write; set the endpoint to the final address"
        text = f"the InfluxDB endpoint of sink {self._name} answered HTTP {status}"
        if message:
            text += f": {message}"
        if hint:
            text += f" ({hint})"
        raise SinkError(text, retryable=retryable)
