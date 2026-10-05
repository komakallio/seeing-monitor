"""The addresses of this device: wildcards, the loopback address, and the addresses to show."""

from __future__ import annotations

import socket
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from seeingmon.services.web import netaddr
from seeingmon.services.web.netaddr import (
    connect_address,
    device_addresses,
    device_names,
    is_wildcard,
)

LINK_LOCAL = "169.254.7.7"  # repo-check: allow
LAPTOP_MDNS = "my-laptop.local"  # repo-check: allow
PI_MDNS = "pi.local"  # repo-check: allow


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("0.0.0.0", True),
        ("::", True),
        ("[::]", True),
        ("0:0:0:0:0:0:0:0", True),
        (" 0.0.0.0 ", True),
        ("127.0.0.1", False),
        ("192.0.2.5", False),
        ("::1", False),
        ("localhost", False),
        ("0.0.0", False),
        ("", False),
    ],
)
def test_a_wildcard_is_the_unspecified_address_in_any_spelling(
    address: str, expected: bool
) -> None:
    assert is_wildcard(address) is expected


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("0.0.0.0", "127.0.0.1"),
        ("::", "::1"),
        ("[::]", "::1"),
        ("localhost", "127.0.0.1"),
        ("LOCALHOST", "127.0.0.1"),
        ("127.0.0.1", "127.0.0.1"),
        ("192.0.2.5", "192.0.2.5"),
        ("2001:db8::5", "2001:db8::5"),
    ],
)
def test_a_program_reaches_a_wildcard_listener_on_the_loopback_address(
    address: str, expected: str
) -> None:
    assert connect_address(address) == expected


def fake_sockets(
    *,
    route: str | None,
    resolved: Sequence[str] | OSError,
    hostname: str = "my-laptop",
) -> SimpleNamespace:
    """Stand in for the `socket` module of `netaddr`: no test may depend on the real network."""

    class Route:
        def __init__(self, family: int, kind: int) -> None:
            self.family = family
            self.kind = kind

        def __enter__(self) -> Route:
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def connect(self, target: tuple[str, int]) -> None:
            if route is None:
                raise OSError("Network is unreachable")

        def getsockname(self) -> tuple[str, int]:
            return (route or "", 50000)

    def getaddrinfo(host: str, port: object, family: int) -> list[tuple[object, ...]]:
        if isinstance(resolved, OSError):
            raise resolved
        return [(family, socket.SOCK_STREAM, 6, "", (address, 0)) for address in resolved]

    return SimpleNamespace(
        socket=Route,
        getaddrinfo=getaddrinfo,
        gethostname=lambda: hostname,
        AF_INET=socket.AF_INET,
        SOCK_DGRAM=socket.SOCK_DGRAM,
    )


def test_the_addresses_of_the_device_start_with_the_one_of_the_default_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = ["192.0.2.9", "198.51.100.2", "192.0.2.9"]
    monkeypatch.setattr(netaddr, "socket", fake_sockets(route="198.51.100.2", resolved=resolved))
    assert device_addresses() == ("198.51.100.2", "192.0.2.9")


def test_the_addresses_of_the_device_leave_out_the_ones_that_nobody_can_reach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = ["127.0.1.1", LINK_LOCAL, "0.0.0.0", "224.0.0.1", "not-an-address", "192.0.2.9"]
    monkeypatch.setattr(netaddr, "socket", fake_sockets(route="127.0.0.1", resolved=resolved))
    assert device_addresses() == ("192.0.2.9",)


def test_a_device_with_no_route_still_gives_the_addresses_of_its_host_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(netaddr, "socket", fake_sockets(route=None, resolved=["192.0.2.9"]))
    assert device_addresses() == ("192.0.2.9",)


def test_a_device_whose_host_name_does_not_resolve_gives_the_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sockets = fake_sockets(route="198.51.100.2", resolved=socket.gaierror("no such host"))
    monkeypatch.setattr(netaddr, "socket", sockets)
    assert device_addresses() == ("198.51.100.2",)


def test_a_device_on_no_network_gives_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    sockets = fake_sockets(route=None, resolved=socket.gaierror("no such host"))
    monkeypatch.setattr(netaddr, "socket", sockets)
    assert device_addresses() == ()


def test_the_real_system_gives_only_addresses_that_a_client_can_use() -> None:
    for address in device_addresses():
        assert not address.startswith(("127.", "169.254.", "0.", "224."))


@pytest.mark.parametrize(
    ("hostname", "expected"),
    [
        ("my-laptop", ("my-laptop", LAPTOP_MDNS)),
        ("My-Laptop", ("my-laptop", LAPTOP_MDNS)),
        ("pi.example.org.", ("pi.example.org", "pi", PI_MDNS)),
        ("  ", ()),
        ("", ()),
    ],
)
def test_the_names_of_the_device_are_its_host_name_its_first_label_and_the_mdns_name(
    monkeypatch: pytest.MonkeyPatch, hostname: str, expected: tuple[str, ...]
) -> None:
    monkeypatch.setattr(netaddr, "socket", fake_sockets(route=None, resolved=[], hostname=hostname))
    assert device_names() == expected


def test_a_system_with_no_host_name_gives_no_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> str:
        raise OSError("no host name")

    sockets = fake_sockets(route=None, resolved=[])
    sockets.gethostname = broken
    monkeypatch.setattr(netaddr, "socket", sockets)
    assert device_names() == ()
