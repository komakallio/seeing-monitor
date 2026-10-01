"""JSON Schema for the records, as OpenAPI 3.1 components, generated from the declarations.

A schema describes a record as `Record.to_row` returns it and as the REST API serves it. Field
names equal the declared names, so they carry the unit (`seeing_fwhm_arcsec`). Every field is
present in a response. A missing value is `null`, and the `quality` object says why. The schema
adds these extensions:

- `x-unit` on a field with a unit, such as `arcsec` or `mag/arcsec^2`.
- `x-codes` on a field with documented codes, a map from each code to its meaning.
- `x-record-type`, `x-storage`, `x-retention-days`, and `x-key` on a record.

The component for a record is named after the record type in CamelCase: `seeing_window` becomes
`SeeingWindow`. `t_utc_ns` is an integer number of nanoseconds, which a JavaScript number cannot
hold exactly, so the REST API also serves an ISO 8601 time next to it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, get_args, get_origin

from seeingmon.records.base import (
    KEY_FIELDS,
    RECORD_TYPES,
    FieldSpec,
    Record,
    base_field_specs,
    doc_paragraphs,
    field_specs,
    resolve_record_type,
)

QUALITY_REF = "#/components/schemas/Quality"

# How a client reads the `quality` object. The field definition says what it is.
QUALITY_NOTE = (
    "A client reads a `null` value together with this map, which gives the reason as a short "
    "code, such as `too_few_frames`."
)


def schema_name(record: str | type[Record]) -> str:
    """The component name of a record type: `seeing_window` becomes `SeeingWindow`."""
    return "".join(part.capitalize() for part in resolve_record_type(record).record_type.split("_"))


def _scalar(annotation: Any) -> dict[str, Any]:
    if annotation is Any:
        return {}
    kinds: dict[Any, dict[str, Any]] = {
        bool: {"type": "boolean"},
        int: {"type": "integer", "format": "int64"},
        float: {"type": "number", "format": "double"},
        str: {"type": "string"},
        bytes: {"type": "string", "contentEncoding": "base64"},
    }
    return dict(kinds[annotation])


def _constraints(schema: dict[str, Any], constraints: Any, *, array: bool, text: bool) -> None:
    keywords = {
        "ge": "minimum",
        "gt": "exclusiveMinimum",
        "le": "maximum",
        "lt": "exclusiveMaximum",
        "pattern": "pattern",
    }
    for name, keyword in keywords.items():
        if name in constraints:
            schema[keyword] = constraints[name]
    if array:
        length = ("minItems", "maxItems")
    elif text:
        length = ("minLength", "maxLength")
    else:
        length = ("minProperties", "maxProperties")
    if "min_length" in constraints:
        schema[length[0]] = constraints["min_length"]
    if "max_length" in constraints:
        schema[length[1]] = constraints["max_length"]


def field_schema(spec: FieldSpec) -> dict[str, Any]:
    """The schema of one field, with its definition, its unit, and its codes."""
    if spec.name == "quality":
        return {
            "anyOf": [{"$ref": QUALITY_REF}, {"type": "null"}],
            "description": spec.definition,
        }
    origin = get_origin(spec.annotation)
    if origin is list:
        schema: dict[str, Any] = {"type": "array", "items": _scalar(get_args(spec.annotation)[0])}
        _constraints(schema, spec.constraints, array=True, text=False)
        if spec.codes is not None:
            schema["items"]["enum"] = sorted(spec.codes)
    elif origin is dict:
        schema = {"type": "object"}
        value = _scalar(get_args(spec.annotation)[1])
        if value:
            schema["additionalProperties"] = value
        _constraints(schema, spec.constraints, array=False, text=False)
    else:
        schema = _scalar(spec.annotation)
        _constraints(schema, spec.constraints, array=False, text=True)
        if spec.codes is not None:
            schema["enum"] = sorted(spec.codes)
    if spec.nullable:
        schema["type"] = [schema["type"], "null"]
        if "enum" in schema:
            schema["enum"] = [*schema["enum"], None]
    schema["description"] = spec.definition
    if spec.examples:
        schema["examples"] = list(spec.examples)
    if spec.unit is not None:
        schema["x-unit"] = spec.unit
    if spec.codes is not None:
        schema["x-codes"] = dict(sorted(spec.codes.items()))
    return schema


def record_schema(record: str | type[Record]) -> dict[str, Any]:
    """The schema of one record type. Every field is required, because a response always has it."""
    cls = resolve_record_type(record)
    specs = field_specs(cls)
    schema: dict[str, Any] = {
        "type": "object",
        "title": schema_name(cls),
        "description": "\n\n".join(doc_paragraphs(cls)),
        "properties": {spec.name: field_schema(spec) for spec in specs},
        "required": [spec.name for spec in specs],
        "x-record-type": cls.record_type,
        "x-storage": cls.storage,
        "x-key": [*KEY_FIELDS],
    }
    if cls.retention_days is not None:
        schema["x-retention-days"] = cls.retention_days
    return schema


def quality_schema() -> dict[str, Any]:
    """The schema of the `quality` object that every record shares."""
    definition = next(spec.definition for spec in base_field_specs() if spec.name == "quality")
    return {
        "type": "object",
        "title": "Quality",
        "description": f"{definition} {QUALITY_NOTE}",
        "additionalProperties": {"type": "string"},
    }


def api_schema(record_types: Iterable[str | type[Record]] | None = None) -> dict[str, Any]:
    """The OpenAPI components for the given record types, or for every record type.

    The result is `{"components": {"schemas": {...}}}`. The `Quality` schema comes first, and the
    records follow in declaration order.
    """
    classes = (
        list(RECORD_TYPES.values())
        if record_types is None
        else [resolve_record_type(record) for record in record_types]
    )
    schemas: dict[str, Any] = {"Quality": quality_schema()}
    for cls in classes:
        schemas[schema_name(cls)] = record_schema(cls)
    return {"components": {"schemas": schemas}}
