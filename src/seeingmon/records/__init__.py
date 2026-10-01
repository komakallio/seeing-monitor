"""Record declarations and the generators that build on them.

Import the record classes and helpers from here, for example
`from seeingmon.records import RECORD_TYPES, Record, get_record_type`. The package loads each
name on first use, so importing `seeingmon.records` (and `seeingmon --help`) stays fast.

The declarations live in `base` (the `Record` base class, `quantity`, and the registry) and in
one module for each owner: `seeing`, `survey`, `reference`, and `system`. The generators are
`sqlite_schema`, `api_schema`, `quantity_reference`, `sink_mapping`, and `segments`. See the
module documentation of `base` for how to declare and change a record type.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from seeingmon.records.base import (
        KEY_FIELDS,
        RECORD_TYPES,
        FieldSpec,
        Record,
        Storage,
        field_specs,
        get_record_type,
        quantity,
    )
    from seeingmon.records.seeing import FrameRecord, SeeingWindowRecord

# The module that defines each public name.
_EXPORTS: dict[str, str] = {
    "KEY_FIELDS": "base",
    "RECORD_TYPES": "base",
    "FieldSpec": "base",
    "Record": "base",
    "Storage": "base",
    "field_specs": "base",
    "get_record_type": "base",
    "quantity": "base",
    "FrameRecord": "seeing",
    "SeeingWindowRecord": "seeing",
}

__all__ = [
    "KEY_FIELDS",
    "RECORD_TYPES",
    "FieldSpec",
    "FrameRecord",
    "Record",
    "SeeingWindowRecord",
    "Storage",
    "field_specs",
    "get_record_type",
    "quantity",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
