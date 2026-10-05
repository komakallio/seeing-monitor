"""The addresses of this device: wildcards, the address that reaches a listener, and what to show.

The web process may listen on a wildcard address (`0.0.0.0` for every IPv4 interface, `::` for every
IPv6 interface), so that the UI answers on whatever address the device has. A phone hotspot hands
out a new address at each connection, and a Raspberry Pi may sit on a LAN or behind a VPN gateway
that forwards to its LAN address. The helpers here serve that:

- `is_wildcard` tells a wildcard from the address of one interface.
- `connect_address` gives the address that a program on this device connects to, to reach a listener
  that is bound to an address (the loopback address, for a wildcard).
- `device_addresses` lists the IPv4 addresses that other devices can use to reach this one, so that
  the launcher can show them.
- `device_names` lists the names of this device, which the host rule allows when the process
  listens on a wildcard.

The module uses the standard library only, so the launcher imports it without the web libraries.
"""

from __future__ import annotations

import ipaddress
import socket

WILDCARD_V4 = "0.0.0.0"
WILDCARD_V6 = "::"
WILDCARDS = (WILDCARD_V4, WILDCARD_V6)
LOOPBACK_V4 = "127.0.0.1"
LOOPBACK_V6 = "::1"

# A documentation address (RFC 5737) that no network routes. Connecting a UDP socket to it sends
# nothing, and the system names the address of the interface that the default route uses.
_ROUTE_PROBE = ("192.0.2.1", 9)


def _canonical(address: str) -> str | None:
    """The canonical text of an IP address, with or without brackets, or `None`."""
    text = address.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return None


def is_wildcard(address: str) -> bool:
    """Whether `address` is `0.0.0.0` or `::`, in any spelling."""
    return _canonical(address) in WILDCARDS


def connect_address(address: str) -> str:
    """The address that a program on this device connects to, to reach a listener on `address`.

    A wildcard gives the loopback address of its family, and `localhost` gives the IPv4 loopback
    address. Any other address gives itself.
    """
    if address.strip().lower() == "localhost":
        return LOOPBACK_V4
    canonical = _canonical(address)
    if canonical == WILDCARD_V4:
        return LOOPBACK_V4
    if canonical == WILDCARD_V6:
        return LOOPBACK_V6
    return address


def device_addresses() -> tuple[str, ...]:
    """The IPv4 addresses that other devices can use to reach this one, the likeliest first.

    The first is the address of the interface that the default route uses, which is the network
    that the device is on right now. The others are the addresses that the host name resolves to.
    Loopback, link-local, and multicast addresses stay out. The function sends no packet, and it
    returns what it found, which is nothing when the device has no network.
    """
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(_ROUTE_PROBE)
            found.append(str(probe.getsockname()[0]))
    except OSError:
        pass  # no route: the device is on no network
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        infos = []
    found.extend(str(info[4][0]) for info in infos)
    usable: list[str] = []
    for text in found:
        canonical = _canonical(text)
        if canonical is None:
            continue
        address = ipaddress.ip_address(canonical)
        if (
            address.is_loopback
            or address.is_link_local
            or address.is_unspecified
            or address.is_multicast
        ):
            continue
        usable.append(canonical)
    return tuple(dict.fromkeys(usable))


def device_names() -> tuple[str, ...]:
    """The host name of this device, its first label, and that label under `.local`.

    `.local` is the name that multicast DNS gives a device on a LAN. The names come from the
    system and cost no lookup. An empty tuple means that the system has no host name.
    """
    try:
        name = socket.gethostname().strip().rstrip(".").lower()
    except OSError:
        return ()
    if not name:
        return ()
    label = name.split(".")[0]
    return tuple(dict.fromkeys(item for item in (name, label, f"{label}.local") if item))
