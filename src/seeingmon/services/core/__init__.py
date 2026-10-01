"""The `core` process: the scheduler, the analysis, the store, and the commissioning tools.

`core` is the composition root of the system. It reads the configuration and the profile, opens
the store, and builds the scheduler with the analyzers and the camera. The camera is a
`RemoteCameraDriver` that talks to `acquire`. `core` is the only writer of the store, and it
serves the commands and the live view to `web` over the local connection layer.

Importing this package stays cheap. Each module loads what it needs when you use it:

- `settings`: the `[services.core]` tables and the `[alignment]` section.
- `app`: `CoreApp`, which wires and runs everything, and `run_core`, the entry of
  `seeingmon core`.
- `health`: the `run` and `health` records.
- `events`: the hardware events of `acquire` and of the local hardware, as `event` records.
- `escalation`: the recovery steps above the driver (restart `acquire`, reboot, power cycle).
- `survey_worker`: the survey analysis in a low-priority process.
- `rpc`: the RPC methods that `web` and the commands call, and the live-view stream.
- `alignment`: the alignment helper (preview frames, the quick solve, the state).
- `commissioning`: the burst and replay handlers, and the client of the commands.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from seeingmon.services.core.settings import (
        AlignmentSettings,
        CoreSettings,
        EscalationSettings,
        SurveyWorkerSettings,
    )

# The module that defines each public name.
_EXPORTS: dict[str, str] = {
    "AlignmentSettings": "settings",
    "CoreSettings": "settings",
    "EscalationSettings": "settings",
    "SurveyWorkerSettings": "settings",
}

__all__ = ["AlignmentSettings", "CoreSettings", "EscalationSettings", "SurveyWorkerSettings"]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
