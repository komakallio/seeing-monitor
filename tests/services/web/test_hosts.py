"""The host rule: how entries and headers are read, and which hosts a request may name."""

from __future__ import annotations

import pytest

from seeingmon.services.web.hosts import (
    LOOPBACK_NAMES,
    allowed_set,
    host_of_header,
    host_of_origin,
    normalize_entries,
    normalize_entry,
)

LONG_LABEL = "a" * 64
LONG_NAME = ".".join(["a" * 60] * 5)


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("pi.example", "pi.example"),
        ("PI.Example", "pi.example"),
        ("pi.example.", "pi.example"),
        ("  pi.example  ", "pi.example"),
        ("pi", "pi"),
        ("my_pi", "my_pi"),
        ("xn--bcher-kva.example", "xn--bcher-kva.example"),
        ("192.0.2.5", "192.0.2.5"),
        ("2001:db8::5", "2001:db8::5"),
        ("2001:DB8:0::5", "2001:db8::5"),
        ("[2001:db8::5]", "2001:db8::5"),
        ("::1", "::1"),
        ("localhost", "localhost"),
    ],
)
def test_an_entry_is_read_in_its_canonical_form(entry: str, expected: str) -> None:
    assert normalize_entry(entry) == expected


@pytest.mark.parametrize(
    "entry",
    [
        "*",
        "*.example",
        "pi.*",
        "p?.example",
        "",
        "   ",
        ".",
        "0.0.0.0",
        "::",
        "pi.example:8080",
        "http://pi.example",
        "pi.example/path",
        "user@pi.example",
        "pi example",
        "a..b",
        "pi.example..",
        "-pi.example",
        "pi-.example",
        LONG_LABEL + ".example",
        LONG_NAME,
        "127.1",
        "1.2.3",
        "192.0.2.5.",
        "bücher.example",
        "[2001:db8::5",
        "[not an address]",
    ],
)
def test_a_bad_entry_is_an_error(entry: str) -> None:
    with pytest.raises(ValueError, match="host"):
        normalize_entry(entry)


def test_an_error_never_shows_the_entry() -> None:
    for entry in ("secret-name.example:80", "*.secret-name.example", "sécret-name.example"):
        with pytest.raises(ValueError, match="host") as raised:
            normalize_entry(entry)
        assert "secret" not in str(raised.value)


def test_the_entries_of_a_list_keep_their_order_and_lose_their_repeats() -> None:
    entries = ["B.example", "a.example.", "b.EXAMPLE", "192.0.2.5", "A.example"]
    assert normalize_entries(entries) == ("b.example", "a.example", "192.0.2.5")


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("localhost", "localhost"),
        ("localhost:8080", "localhost"),
        ("LOCALHOST:8080", "localhost"),
        ("localhost.", "localhost"),
        ("localhost.:8080", "localhost"),
        ("localhost:", "localhost"),
        ("127.0.0.1", "127.0.0.1"),
        ("127.0.0.1:8080", "127.0.0.1"),
        ("192.0.2.5:80", "192.0.2.5"),
        ("[::1]", "::1"),
        ("[::1]:8080", "::1"),
        ("[2001:db8::5]:8080", "2001:db8::5"),
        ("[2001:DB8:0::5]:8080", "2001:db8::5"),
        ("[2001:db8::5]:", "2001:db8::5"),
        ("pi.example:65535", "pi.example"),
        ("  pi.example:8080  ", "pi.example"),
    ],
)
def test_a_host_header_gives_the_host_without_the_port(header: str, expected: str) -> None:
    assert host_of_header(header) == expected


@pytest.mark.parametrize(
    "header",
    [
        "",
        "   ",
        "[::1",
        "[::1]x",
        "[::1]8080",
        "[::1]:abc",
        "[::1]:70000",
        "[not-an-address]:80",
        "[]",
        "::1",
        "2001:db8::5",
        "localhost:abc",
        "localhost:70000",
        "localhost:80:80",
        "local host",
        "pi.example\r\nx: y",
        "pi.\texample",
        "pi.example\x00",
        "user@pi.example",
        "pi.example/path",
        "127.1",
        "bücher.example",
        "-bad.example",
        "a" * 300,
        ":8080",
    ],
)
def test_a_malformed_host_header_gives_nothing(header: str) -> None:
    assert host_of_header(header) is None


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("http://localhost", "localhost"),
        ("http://localhost:8080", "localhost"),
        ("https://Pi.Example:8443", "pi.example"),
        ("http://pi.example./", "pi.example"),
        ("http://192.0.2.5:8080", "192.0.2.5"),
        ("http://[2001:db8::5]:8080", "2001:db8::5"),
        ("http://[::1]", "::1"),
        ("  http://localhost:8080  ", "localhost"),
    ],
)
def test_an_origin_gives_its_host(origin: str, expected: str) -> None:
    assert host_of_origin(origin) == expected


@pytest.mark.parametrize(
    "origin",
    [
        "",
        "null",
        "file://",
        "file:///home/x",
        "ftp://localhost",
        "chrome-extension://abcdef",  # repo-check: allow
        "http://",
        "http:localhost",
        "//localhost",
        "localhost:8080",
        "http://user@localhost",
        "http://localhost/path",
        "http://localhost?x=1",
        "http://localhost#x",
        "http://local\nhost",  # repo-check: allow
        "http://local\thost",  # repo-check: allow
        "http://local host",  # repo-check: allow
        "http://[::1",
        "http://localhost:port",
        "http://bücher.example",  # repo-check: allow
    ],
)
def test_an_origin_without_a_usable_host_gives_nothing(origin: str) -> None:
    assert host_of_origin(origin) is None


def test_the_allowed_set_holds_the_loopback_names_without_any_setting() -> None:
    assert allowed_set("127.0.0.1", (), ()) == frozenset(LOOPBACK_NAMES)


def test_the_allowed_set_adds_the_bind_addresses_and_the_list() -> None:
    allowed = allowed_set("192.0.2.5", ["2001:DB8::5", "192.0.2.6"], ["Pi.Example.", "pi"])
    assert allowed == frozenset(
        {*LOOPBACK_NAMES, "192.0.2.5", "2001:db8::5", "192.0.2.6", "pi.example", "pi"}
    )


def test_the_set_of_a_localhost_bind_address_is_the_loopback_names() -> None:
    assert allowed_set("localhost", (), ()) == frozenset(LOOPBACK_NAMES)


def test_an_entry_that_does_not_validate_fails_the_set() -> None:
    with pytest.raises(ValueError, match="wildcards"):
        allowed_set("127.0.0.1", (), ["*.example"])
