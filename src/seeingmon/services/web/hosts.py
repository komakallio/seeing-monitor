"""Host names and addresses: the rule that decides which `Host` and `Origin` a request may carry.

The web process answers only the hosts that its settings allow. A browser sends the host of the
address bar in the `Host` header, so a request that reaches the server through a name that the owner
did not list (a DNS rebinding attack, or a typo) gets an error. A WebSocket handshake also carries
an `Origin` header, and the same rule applies to its host, which stops a page from another site from
opening the live view in a visitor's browser.

**Entries.** An entry of `allowed_hosts` is an IPv4 address, an IPv6 address, or a DNS name. A
wildcard is an error, because a wildcard would void the rule. The functions compare an entry in
lowercase, without a trailing dot, and without the brackets of an IPv6 literal. They compare an IPv6
address in its canonical form, so `2001:DB8:0::5` equals `2001:db8::5`.

**Headers.** `host_of_header` reads a `Host` value: `name`, `name:port`, `192.0.2.5:8080`, or
`[2001:db8::5]:8080`. Starlette's `TrustedHostMiddleware` splits at the first colon, which breaks on
an IPv6 literal, so this module reads the brackets itself. The port never counts.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable
from urllib.parse import urlsplit

LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
MAX_HOST_CHARS = 253
MAX_PORT = 65535

_LABEL = re.compile(r"[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?")
_PORT = re.compile(r"[0-9]{0,5}")


def _address_or_none(text: str) -> str | None:
    """The canonical text of an IP address, or `None` when `text` is not one."""
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return None


def _name_or_none(text: str) -> str | None:
    """The text of a DNS name in lowercase, or `None` when it is not a valid name.

    One trailing dot (the root label) is dropped. A name that has only digits and dots is not a
    name, because a client could read it as an address in a short form such as `127.1`.
    """
    name = text[:-1] if text.endswith(".") else text
    name = name.lower()
    if not name or len(name) > MAX_HOST_CHARS:
        return None
    labels = name.split(".")
    if not all(_LABEL.fullmatch(label) for label in labels):
        return None
    if all(label.isdigit() for label in labels):
        return None
    return name


def _valid_port(text: str) -> bool:
    return _PORT.fullmatch(text) is not None and (not text or int(text) <= MAX_PORT)


def normalize_entry(value: str) -> str:
    """The canonical form of an entry of the settings. Raises `ValueError` for anything else.

    The messages never show the value, because the configuration layer must not echo one.
    """
    text = value.strip()
    if "*" in text or "?" in text:
        raise ValueError("a host is an exact name or address: wildcards are not allowed")
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text.isascii():
        raise ValueError("a host is an IP address or an ASCII DNS name, such as the punycode form")
    address = _address_or_none(text)
    if address is not None:
        if ipaddress.ip_address(address).is_unspecified:
            raise ValueError("a host is one name or address, not 0.0.0.0 or ::")
        return address
    name = _name_or_none(text)
    if name is None:
        raise ValueError(
            "a host is an IP address or a DNS name of letters, digits, hyphens, and dots, "
            "without a port or a path"
        )
    return name


def normalize_entries(values: Iterable[str]) -> tuple[str, ...]:
    """Normalize a list of entries, keep the order, and drop the repeats."""
    return tuple(dict.fromkeys(normalize_entry(value) for value in values))


def host_of_header(value: str) -> str | None:
    """The host of a `Host` header in canonical form, or `None` for a malformed value.

    The function drops the port. It reads `[addr]:port` for an IPv6 literal, and it refuses an
    unbracketed IPv6 literal, which a `Host` header never holds.
    """
    text = value.strip()
    if not text or not text.isascii() or not text.isprintable() or len(text) > MAX_HOST_CHARS + 8:
        return None
    if text.startswith("["):
        close = text.find("]")
        if close < 0:
            return None
        literal, rest = text[1:close], text[close + 1 :]
        if rest and not (rest.startswith(":") and _valid_port(rest[1:])):
            return None
        return _address_or_none(literal)
    host, colon, port = text.partition(":")
    if colon and not _valid_port(port):
        return None
    return _address_or_none(host) or _name_or_none(host)


def host_of_origin(value: str) -> str | None:
    """The canonical host of an `Origin` header, or `None` when it names no host.

    `Origin: null` (a sandboxed page, or a file) names no host, and neither does a scheme that is
    not `http` or `https`.
    """
    text = value.strip()
    if not text.isascii() or not text.isprintable() or " " in text:
        return None  # urlsplit would silently drop a tab or a newline inside the value
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    if parts.path not in ("", "/") or parts.query or parts.fragment or "@" in parts.netloc:
        return None
    return host_of_header(parts.netloc)


def allowed_set(
    bind_address: str, extra_bind_addresses: Iterable[str], allowed_hosts: Iterable[str]
) -> frozenset[str]:
    """The hosts that a request may name: the list, the loopback names, and the bind addresses."""
    return frozenset(
        (
            *LOOPBACK_NAMES,
            normalize_entry(bind_address),
            *(normalize_entry(address) for address in extra_bind_addresses),
            *(normalize_entry(host) for host in allowed_hosts),
        )
    )
