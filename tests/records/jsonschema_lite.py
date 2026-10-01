"""A small JSON Schema validator for the keywords that the API schema generator emits.

The tests use it to check that every record row validates against the generated schema. It
supports `$ref` to `#/components/schemas/...`, `anyOf`, `type`, `enum`, `properties`,
`required`, `additionalProperties`, `items`, the numeric and length bounds, `pattern`, and
`contentEncoding: base64`. It ignores `format`, `description`, `title`, and the `x-` extensions.
`KEYWORDS` lists everything that the validator knows, so a test can fail when the generator emits
a keyword that this module would silently skip.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any

KEYWORDS = frozenset(
    {
        "$ref",
        "anyOf",
        "type",
        "enum",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
        "pattern",
        "contentEncoding",
        "format",
        "description",
        "title",
        "examples",
    }
)

_PREFIX = "#/components/schemas/"


class InvalidError(Exception):
    """The instance does not match the schema. The message names the path."""


def unknown_keywords(schema: Any) -> set[str]:
    """The keywords in a schema that neither `KEYWORDS` nor an `x-` extension covers."""
    found: set[str] = set()
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in ("properties",):
                for sub in value.values():
                    found |= unknown_keywords(sub)
            elif key in ("additionalProperties", "items"):
                found |= unknown_keywords(value)
            elif key == "anyOf":
                for sub in value:
                    found |= unknown_keywords(sub)
            elif key not in KEYWORDS and not key.startswith("x-"):
                found.add(key)
    return found


def _is_type(instance: Any, name: str) -> bool:
    if name == "null":
        return instance is None
    if name == "boolean":
        return isinstance(instance, bool)
    if name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if name == "number":
        return isinstance(instance, int | float) and not isinstance(instance, bool)
    if name == "string":
        return isinstance(instance, str)
    if name == "array":
        return isinstance(instance, list)
    if name == "object":
        return isinstance(instance, dict)
    raise ValueError(f"unknown type {name}")


def validate(
    instance: Any, schema: dict[str, Any], document: dict[str, Any], path: str = "$"
) -> None:
    """Raise `InvalidError` when `instance` does not match `schema`. `document` resolves `$ref`."""
    if "$ref" in schema:
        name = schema["$ref"].removeprefix(_PREFIX)
        validate(instance, document["components"]["schemas"][name], document, path)
        return
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate(instance, option, document, path)
            except InvalidError:
                continue
            break
        else:
            raise InvalidError(f"{path}: matches none of the anyOf options")
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(instance, name) for name in names):
            raise InvalidError(f"{path}: {instance!r} is not of type {names}")
    if "enum" in schema and instance not in schema["enum"]:
        raise InvalidError(f"{path}: {instance!r} is not one of {schema['enum']}")
    if isinstance(instance, int | float) and not isinstance(instance, bool):
        _check_number(instance, schema, path)
    if isinstance(instance, str):
        _check_string(instance, schema, path)
    if isinstance(instance, list):
        _check_array(instance, schema, document, path)
    if isinstance(instance, dict):
        _check_object(instance, schema, document, path)


def _check_number(value: float, schema: dict[str, Any], path: str) -> None:
    if "minimum" in schema and value < schema["minimum"]:
        raise InvalidError(f"{path}: {value} is below {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        raise InvalidError(f"{path}: {value} is above {schema['maximum']}")
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        raise InvalidError(f"{path}: {value} is not above {schema['exclusiveMinimum']}")
    if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
        raise InvalidError(f"{path}: {value} is not below {schema['exclusiveMaximum']}")


def _check_string(value: str, schema: dict[str, Any], path: str) -> None:
    if "minLength" in schema and len(value) < schema["minLength"]:
        raise InvalidError(f"{path}: the text is shorter than {schema['minLength']}")
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        raise InvalidError(f"{path}: the text is longer than {schema['maxLength']}")
    if "pattern" in schema and not re.search(schema["pattern"], value):
        raise InvalidError(f"{path}: {value!r} does not match {schema['pattern']}")
    if schema.get("contentEncoding") == "base64":
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            raise InvalidError(f"{path}: not base64") from None


def _check_array(
    value: list[Any], schema: dict[str, Any], document: dict[str, Any], path: str
) -> None:
    if "minItems" in schema and len(value) < schema["minItems"]:
        raise InvalidError(f"{path}: fewer than {schema['minItems']} items")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        raise InvalidError(f"{path}: more than {schema['maxItems']} items")
    if "items" in schema:
        for index, item in enumerate(value):
            validate(item, schema["items"], document, f"{path}[{index}]")


def _check_object(
    value: dict[str, Any], schema: dict[str, Any], document: dict[str, Any], path: str
) -> None:
    for name in schema.get("required", []):
        if name not in value:
            raise InvalidError(f"{path}: missing {name}")
    properties = schema.get("properties", {})
    extra = schema.get("additionalProperties", True)
    for name, item in value.items():
        if name in properties:
            validate(item, properties[name], document, f"{path}.{name}")
        elif extra is False:
            raise InvalidError(f"{path}: unexpected {name}")
        elif isinstance(extra, dict):
            validate(item, extra, document, f"{path}.{name}")
    if "minProperties" in schema and len(value) < schema["minProperties"]:
        raise InvalidError(f"{path}: fewer than {schema['minProperties']} properties")
    if "maxProperties" in schema and len(value) > schema["maxProperties"]:
        raise InvalidError(f"{path}: more than {schema['maxProperties']} properties")
