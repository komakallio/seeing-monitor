"""Import FastAPI's `TestClient` in a way that the test run accepts.

Starlette 1.7 prefers `httpx2` for its test client, and it warns when it falls back to `httpx`
(which the dev group installs). The test run turns every warning into an error, and the warning
comes from the import itself, so the import happens here, once, with that one message ignored. When
the dev group moves to `httpx2`, this module needs no change, and the filter has no effect.
"""

from __future__ import annotations

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r"Using `httpx` with `starlette\.testclient`")
    from fastapi.testclient import TestClient

__all__ = ["TestClient"]
