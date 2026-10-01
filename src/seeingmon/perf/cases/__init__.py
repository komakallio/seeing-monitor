"""The cases of the performance harness.

Importing this package registers every case in `seeingmon.perf.registry.REGISTRY`, in the order of
`CASE_MODULES`: `calibration`, then the cases for the budgets of the architecture, then the
placeholder for the `core` process. Each module imports only the harness at the top. A case
imports the code that it measures when it runs.
"""

from __future__ import annotations

import importlib

CASE_MODULES = (
    "calibration",
    "kernel",
    "fastpath",
    "ipc",
    "survey",
    "store",
    "memory",
    "core_sim",
)

for _module in CASE_MODULES:
    importlib.import_module(f"{__name__}.{_module}")
