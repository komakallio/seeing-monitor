"""Sample records for tests and fakes.

`sample_record("seeing_window")` builds a valid record of a type, so a test that needs a record
does not have to fill every required field. A sample sets the required fields only, and the
optional fields stay `None`. A field takes its declared example when it has one. Otherwise it
takes the simplest value that its type and bounds allow: zero or the lowest allowed number,
`"x"` for text, an empty list or dict, `False`, or the first documented code. Pass keyword
arguments to set or replace fields:

    window = sample_record("seeing_window", n_frames=5400, seeing_fwhm_arcsec=1.2)

A sample is for tests. The values are valid, and they are not realistic measurements.
"""

from __future__ import annotations

from typing import Any, TypeVar, get_origin, overload

from seeingmon.records.base import FieldSpec, Record, field_specs, resolve_record_type

R = TypeVar("R", bound=Record)


def _simplest(spec: FieldSpec) -> Any:
    constraints = spec.constraints
    if spec.codes is not None:
        return next(iter(spec.codes)) if spec.annotation is str else []
    if spec.kind == "bool":
        return False
    if spec.kind == "int":
        low = int(constraints.get("ge", int(constraints["gt"]) + 1 if "gt" in constraints else 0))
        return max(low, 0) if constraints.get("le", 0) >= 0 else low
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


def sample_values(record: str | type[Record], **overrides: Any) -> dict[str, Any]:
    """Return the constructor arguments of a sample: the required fields, then the overrides."""
    values = {
        spec.name: spec.examples[0] if spec.examples else _simplest(spec)
        for spec in field_specs(resolve_record_type(record))
        if spec.required
    }
    values.update(overrides)
    return values


@overload
def sample_record(record: type[R], **overrides: Any) -> R: ...


@overload
def sample_record(record: str, **overrides: Any) -> Record: ...


def sample_record(record: str | type[Record], **overrides: Any) -> Record:
    """Build a valid record of a type (a name or a class), with the overrides applied."""
    cls = resolve_record_type(record)
    return cls(**sample_values(cls, **overrides))
