"""Hypothesis strategies that build records from their declarations.

A strategy draws every field from its declared type, bounds, codes, and pattern, so a new
field of any record type joins the property tests with no change here. A few record types have
rules that connect fields (a star list needs as many bytes as its rows). `_REPAIRS` enforces
them. `minimal_record` builds the smallest valid record without Hypothesis.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from datetime import date
from typing import Any, get_args, get_origin

from hypothesis import strategies as st

from seeingmon.records.base import (
    RECORD_TYPES,
    FieldSpec,
    Record,
    field_specs,
    resolve_record_type,
)

ALL_RECORD_TYPES: list[type[Record]] = list(RECORD_TYPES.values())

# Text that SQLite and JSON both store without change: no surrogates and no NUL character.
_CHARACTERS = st.characters(codec="utf-8", exclude_characters="\x00")
TEXT = st.text(alphabet=_CHARACTERS, max_size=12)
NAMES = st.text(alphabet=_CHARACTERS, min_size=1, max_size=8)
FINITE_FLOATS = st.floats(allow_nan=False, allow_infinity=False)

_JSON_LEAVES: st.SearchStrategy[Any] = (
    st.none() | st.booleans() | st.integers(-(2**53), 2**53) | FINITE_FLOATS | TEXT
)
JSON_VALUES: st.SearchStrategy[Any] = st.recursive(
    _JSON_LEAVES,
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(NAMES, inner, max_size=3),
    max_leaves=8,
)

# Fields with a format that the declared type does not show.
_FORMATTED: dict[str, Any] = {"night": "2026-01-01", "kind": "scheduler.state_change"}


def type_id(cls: type[Record]) -> str:
    """A test ID for a parametrized record class."""
    return cls.record_type


def _bounds(constraints: dict[str, Any]) -> tuple[Any, Any, bool, bool]:
    low = constraints.get("ge", constraints.get("gt"))
    high = constraints.get("le", constraints.get("lt"))
    return low, high, "gt" in constraints, "lt" in constraints


def _integers(constraints: dict[str, Any]) -> st.SearchStrategy[int]:
    low, high, exclude_low, exclude_high = _bounds(constraints)
    low = -(2**63) if low is None else int(low) + int(exclude_low)
    high = 2**63 - 1 if high is None else int(high) - int(exclude_high)
    return st.integers(low, high)


def _floats(constraints: dict[str, Any], *, width: int = 64) -> st.SearchStrategy[float]:
    low, high, exclude_low, exclude_high = _bounds(constraints)
    return st.floats(
        min_value=low,
        max_value=high,
        exclude_min=exclude_low and low is not None,
        exclude_max=exclude_high and high is not None,
        allow_nan=False,
        allow_infinity=False,
        width=width,  # type: ignore[arg-type]
    )


def _element(annotation: Any) -> st.SearchStrategy[Any]:
    if annotation is Any:
        return JSON_VALUES
    return {
        bool: st.booleans(),
        int: st.integers(-(2**53), 2**53),
        float: FINITE_FLOATS,
        str: TEXT,
    }[annotation]


def _value(spec: FieldSpec) -> st.SearchStrategy[Any]:
    constraints = dict(spec.constraints)
    if spec.codes is not None:
        code = st.sampled_from(sorted(spec.codes))
        if spec.annotation is str:
            return code
        return st.lists(code, unique=True, max_size=len(spec.codes))
    if spec.kind == "bool":
        return st.booleans()
    if spec.kind == "int":
        return _integers(constraints)
    if spec.kind == "float":
        return _floats(constraints, width={"f2": 16, "f4": 32}.get(spec.dtype or "", 64))
    if spec.kind == "str":
        if "pattern" in constraints:
            return st.from_regex(constraints["pattern"], fullmatch=True)
        return st.text(
            alphabet=_CHARACTERS,
            min_size=constraints.get("min_length", 0),
            max_size=max(constraints.get("max_length", 12), constraints.get("min_length", 0)),
        )
    if spec.kind == "bytes":
        return st.binary(max_size=64)
    args = get_args(spec.annotation)
    if get_origin(spec.annotation) is list:
        return st.lists(
            _element(args[0]),
            min_size=constraints.get("min_length", 0),
            max_size=constraints.get("max_length", 4),
        )
    return st.dictionaries(NAMES, _element(args[1]), max_size=3)


def field_strategy(spec: FieldSpec) -> st.SearchStrategy[Any]:
    """The values that a field accepts, including `None` for a nullable field."""
    value = _value(spec)
    return st.none() | value if spec.nullable else value


def _repair_seeing_window(values: dict[str, Any], draw: st.DrawFn) -> None:
    freq = values.get("motion_psd_freq_hz")
    for name in ("motion_psd_x_arcsec2_per_hz", "motion_psd_y_arcsec2_per_hz"):
        if freq is None:
            values[name] = None
        elif values.get(name) is not None:
            values[name] = draw(st.lists(FINITE_FLOATS, min_size=len(freq), max_size=len(freq)))


def _repair_star_rows(values: dict[str, Any], draw: st.DrawFn) -> None:
    columns = draw(st.lists(NAMES, min_size=1, max_size=4, unique=True))
    count = draw(st.integers(0, 6))
    floats = draw(
        st.lists(
            st.floats(width=32, allow_nan=False, allow_infinity=False),
            min_size=count * len(columns),
            max_size=count * len(columns),
        )
    )
    values["columns"] = columns
    values["n_stars"] = count
    values["data"] = struct.pack(f"<{len(floats)}f", *floats)


_REPAIRS: dict[str, Callable[[dict[str, Any], st.DrawFn], None]] = {
    "seeing_window": _repair_seeing_window,
    "star_list": _repair_star_rows,
    "star_epoch": _repair_star_rows,
}


@st.composite
def record_values(draw: st.DrawFn, record: str | type[Record]) -> dict[str, Any]:
    """Draw a dict of constructor arguments for a record type."""
    cls = resolve_record_type(record)
    specs = field_specs(cls)
    names = [spec.name for spec in specs]
    values: dict[str, Any] = {}
    for spec in specs:
        if spec.name == "quality":
            values["quality"] = draw(
                st.none() | st.dictionaries(st.sampled_from(names), TEXT, max_size=3)
            )
        elif spec.name == "night":
            values["night"] = draw(st.dates().map(date.isoformat))
        else:
            values[spec.name] = draw(field_strategy(spec))
    repair = _REPAIRS.get(cls.record_type)
    if repair is not None:
        repair(values, draw)
    return values


def records(record: str | type[Record]) -> st.SearchStrategy[Record]:
    """A strategy for valid records of a type (a name or a class)."""
    cls = resolve_record_type(record)
    return record_values(cls).map(lambda values: cls(**values))


def _minimal(spec: FieldSpec) -> Any:
    constraints = spec.constraints
    if spec.name in _FORMATTED:
        return _FORMATTED[spec.name]
    if spec.codes is not None:
        return sorted(spec.codes)[0] if spec.annotation is str else []
    if spec.kind == "bool":
        return False
    if spec.kind == "int":
        if "ge" in constraints:
            return int(constraints["ge"])
        return int(constraints["gt"]) + 1 if "gt" in constraints else 0
    if spec.kind == "float":
        if "gt" in constraints:
            high = constraints.get("le", constraints.get("lt"))
            low = constraints["gt"]
            return (low + high) / 2 if high is not None else low + 1.0
        return float(constraints.get("ge", 0.0))
    if spec.kind == "str":
        return "x"
    if spec.kind == "bytes":
        return b""
    return {} if get_origin(spec.annotation) is dict else []


def minimal_values(record: str | type[Record]) -> dict[str, Any]:
    """The constructor arguments for the smallest valid record: required fields only."""
    specs = field_specs(resolve_record_type(record))
    return {spec.name: _minimal(spec) for spec in specs if spec.required}


def minimal_record(record: str | type[Record]) -> Record:
    """The smallest valid record of a type, without Hypothesis."""
    cls = resolve_record_type(record)
    return cls(**minimal_values(cls))
