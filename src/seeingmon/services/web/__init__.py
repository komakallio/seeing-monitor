"""The `web` process: the REST API v1, the static UI, the alignment live view, and the demo mode.

`web` faces the LAN, so it gets read-only access to the store and the images, no camera access,
and it sends commands to `core` over the local connection layer (`seeingmon.services.ipc`).
`core` is the only writer.

Importing this package stays cheap: FastAPI, uvicorn, and Pillow load only when you use the
modules that need them. The modules are:

- `config`: the `[web]` and `[auth]` configuration sections.
- `auth`: the token hash, the bearer-token check, and the per-client rate limiter.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from seeingmon.services.web.config import AuthSettings, WebSettings

# The module that defines each public name.
_EXPORTS: dict[str, str] = {
    "AuthSettings": "config",
    "WebSettings": "config",
}

__all__ = ["AuthSettings", "WebSettings"]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
