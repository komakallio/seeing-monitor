"""The declarations may only grow: a stored database and the segment files depend on them."""

from __future__ import annotations

import subprocess
import sys

import pytest

from seeingmon.records.base import RECORD_TYPES, Record, field_specs
from tests.records.baseline import BASELINE
from tests.records.strategies import ALL_RECORD_TYPES, type_id

# The type, nullability, and unit of the fields that every record has.
BASE_FIELDS = (
    ("station_id", "str", False, None),
    ("t_utc_ns", "int", False, "ns"),
    ("revision", "int", False, None),
    ("profile_id", "str", False, None),
    ("provenance", "dict[str, str]", False, None),
    ("quality", "dict[str, str]", True, None),
)

# The record types that the architecture lists in its data model.
ARCHITECTURE_RECORD_TYPES = {
    "frame": ("segment", 7),
    "seeing_window": ("table", None),
    "survey_frame": ("table", None),
    "sky_quality": ("table", None),
    "pointing": ("table", None),
    "star_list": ("table", 365),
    "star_epoch": ("table", None),
    "reference": ("table", None),
    "health": ("table", None),
    "event": ("table", None),
    "run": ("table", None),
}


def summary(cls: type[Record], *, base: bool) -> list[tuple[str, str, bool, str | None]]:
    return [
        (spec.name, spec.type_name, spec.nullable, spec.unit)
        for spec in field_specs(cls)
        if spec.base == base
    ]


def test_the_record_types_of_the_architecture_are_declared() -> None:
    declared = {name: (cls.storage, cls.retention_days) for name, cls in RECORD_TYPES.items()}
    for name, expected in ARCHITECTURE_RECORD_TYPES.items():
        assert declared.get(name) == expected, name
    assert list(RECORD_TYPES)[: len(ARCHITECTURE_RECORD_TYPES)] == list(ARCHITECTURE_RECORD_TYPES)


def test_the_order_of_the_registry_does_not_depend_on_the_import_order() -> None:
    # The generated files list the record types in registry order, so a lane that imports its
    # own declaration module first must not change that order.
    code = (
        "import seeingmon.records.system, seeingmon.records.reference\n"
        "from seeingmon.records.base import RECORD_TYPES\n"
        "print(','.join(RECORD_TYPES))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    names = result.stdout.strip().split(",")
    assert names[: len(ARCHITECTURE_RECORD_TYPES)] == list(ARCHITECTURE_RECORD_TYPES)


@pytest.mark.parametrize("cls", ALL_RECORD_TYPES, ids=type_id)
def test_every_record_has_the_base_fields_first(cls: type[Record]) -> None:
    assert summary(cls, base=True) == list(BASE_FIELDS)
    assert [spec.name for spec in field_specs(cls)[: len(BASE_FIELDS)]] == [
        name for name, *_ in BASE_FIELDS
    ]


@pytest.mark.parametrize("name", sorted(BASELINE))
def test_a_declaration_keeps_the_fields_that_it_shipped_with(name: str) -> None:
    assert name in RECORD_TYPES, f"{name} was removed"
    shipped = list(BASELINE[name])
    now = summary(RECORD_TYPES[name], base=False)
    assert now[: len(shipped)] == shipped, (
        f"{name} changed a shipped field. Append new fields at the end of the class, and never "
        "rename, remove, retype, or reorder a field."
    )
