"""Addresses for the services tests: every test that starts a server gets a fresh one.

The default addresses (`seeingmon-core` and `seeingmon-acquire` as pipe names on Windows) belong to
the machine, so two tests, two runs, or a test and a running system collide on them. On Windows the
collision shows up as an access-denied error. A test passes an address from `unique_address`
through the settings (`[services] core_address`), and never relies on the default.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from seeingmon.services.ipc.endpoint import Endpoint


def unique_address(role: str) -> str:
    """The text for `[services] <role>_address`: a pipe name on Windows, a short path elsewhere.

    The name holds 12 random hex digits. A Unix socket path holds at most 107 bytes, so the path
    sits directly in the temporary directory and not in the long directory of a test.
    """
    token = uuid.uuid4().hex[:12]
    if os.name == "nt":
        return f"seeingmon-test-{token}-{role}"
    return str(Path(tempfile.gettempdir()) / f"smon-{token}-{role}.sock")


def unique_endpoint(role: str) -> Endpoint:
    """A fresh endpoint of the native family for `role`."""
    return Endpoint.parse(unique_address(role))
