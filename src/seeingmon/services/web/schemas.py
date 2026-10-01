"""The OpenAPI components of the records that the API serves.

The record schemas come from `seeingmon.records.api_schema`, which generates them from the record
declarations. This module adds what the REST API adds to a record, and the page types.

- Each served record type has a component with its name (`SeeingWindow`, `SkyQuality`, `Pointing`,
  and `Event`). It is the generated schema plus `t_utc`, the ISO 8601 time. `GET .../latest` serves
  it.
- Each also has a `...Row` component: the item of a history page. Every field is optional, because
  a request can name the fields that it wants. A count is a number and not an integer, because the
  mean of a bucket is not whole. `n_samples` tells how many records a bucket combines.
- Each also has a `...Page` component: `items` and `next_cursor`, with the range that the page
  answers.
"""

from __future__ import annotations

import copy
from typing import Any

from seeingmon.records.api_schema import api_schema, schema_name
from seeingmon.services.web.data import SETTING_FIELDS

SERVED_RECORD_TYPES = ("seeing_window", "sky_quality", "pointing", "event")
PAGE_STEPS = ["raw", "1m", "10m", "1h"]

TIME_SCHEMA: dict[str, Any] = {
    "type": "string",
    "format": "date-time",
    "description": (
        "The start time as an ISO 8601 UTC string with six fractional digits. The integer "
        "`t_utc_ns` is too large for a JavaScript number to hold exactly, so use this field there."
    ),
    "examples": ["2026-10-01T21:00:00.000000Z"],
}
KEEP_INTEGER = SETTING_FIELDS | {"t_utc_ns"}
SAMPLES_SCHEMA: dict[str, Any] = {
    "type": "integer",
    "minimum": 1,
    "description": "The number of records that this item combines. A raw item combines none.",
}


def component_ref(name: str) -> dict[str, str]:
    """A `$ref` to a component schema."""
    return {"$ref": f"#/components/schemas/{name}"}


def _widen(schema: dict[str, Any]) -> None:
    """Let a field of a row hold the mean of integers, which is a number and not an integer."""
    kind = schema.get("type")
    if kind == "integer":
        schema["type"] = "number"
    elif isinstance(kind, list) and "integer" in kind:
        schema["type"] = ["number" if item == "integer" else item for item in kind]
    else:
        return
    schema["format"] = "double"


def record_components() -> dict[str, Any]:
    """The component schemas of the served records, their rows, and their pages."""
    generated = api_schema(SERVED_RECORD_TYPES)["components"]["schemas"]
    components: dict[str, Any] = {"Quality": generated["Quality"]}
    for record_type in SERVED_RECORD_TYPES:
        name = schema_name(record_type)
        record = copy.deepcopy(generated[name])
        record["properties"]["t_utc"] = copy.deepcopy(TIME_SCHEMA)
        record["required"] = [*record["required"], "t_utc"]
        components[name] = record

        row = copy.deepcopy(record)
        row["title"] = f"{name}Row"
        row["description"] = (
            f"An item of the history of `{record_type}` records. Without `fields`, an item holds "
            "every field except the arrays of numbers. With `fields`, it holds the named fields. "
            f"{record['description']}"
        )
        for field, schema in row["properties"].items():
            if field not in KEEP_INTEGER:
                _widen(schema)
        row["properties"]["n_samples"] = copy.deepcopy(SAMPLES_SCHEMA)
        row["required"] = ["t_utc_ns", "t_utc"]
        components[f"{name}Row"] = row

        components[f"{name}Page"] = {
            "type": "object",
            "title": f"{name}Page",
            "description": f"A page of `{record_type}` items, oldest first unless noted.",
            "required": ["record_type", "step", "from", "to", "now", "items", "next_cursor"],
            "properties": {
                "record_type": {"type": "string", "const": record_type},
                "step": {"type": "string", "enum": PAGE_STEPS},
                "from": copy.deepcopy(TIME_SCHEMA) | {"description": "The start of the range."},
                "to": copy.deepcopy(TIME_SCHEMA)
                | {"description": "The end of the range, which the range excludes."},
                "now": copy.deepcopy(TIME_SCHEMA) | {"description": "The time of the server."},
                "items": {"type": "array", "items": component_ref(f"{name}Row")},
                "next_cursor": {
                    "type": ["string", "null"],
                    "description": "Pass it as `cursor` to read the next page, or `null`.",
                },
            },
        }
    return components


def json_response(name: str, description: str) -> dict[str, Any]:
    """The `responses` entry of a route whose body is the component `name`."""
    return {
        "description": description,
        "content": {"application/json": {"schema": component_ref(name)}},
    }
