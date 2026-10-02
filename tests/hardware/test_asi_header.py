"""The binding against the vendor header, and the parser and the comparison that do it.

The check against a real header needs a copy of `ASICamera2.h`, which the repository never holds.
Set `SEEINGMON_ASI__HEADER_PATH` to the file, and run

    python -m pytest tests/hardware/test_asi_header.py -s

on the platform that will use the library: the Raspberry Pi has its own header, and its own `long`.
The test skips without the variable. The tests of the parser and of the comparison run everywhere,
on a small header that this file holds.
"""

from __future__ import annotations

import ctypes
import os
from enum import IntEnum
from pathlib import Path
from typing import cast

import pytest

from tests.hardware.asi_header import (
    HEADER_ENV,
    CType,
    Header,
    HeaderError,
    Report,
    build_structure,
    compare_enum,
    compare_functions,
    compare_struct,
    compare_with_binding,
    parse_header,
    read_header,
    to_ctypes,
    tokenize,
)

DEMO = """\
/* A synthetic header, in the style of a vendor header. */
#ifndef DEMO_H
#define DEMO_H

#ifdef _WINDOWS
	#define DEMO_API __declspec(dllexport)
#else
	#define DEMO_API
#endif

#define DEMO_ID_MAX 256

typedef enum COLOR_KIND { // the colors
	COLOR_RED = 0,
	COLOR_GREEN,   /* implicit: 1 */
	COLOR_BLUE = 5,
	COLOR_NONE = -1,
} COLOR_KIND;

typedef enum SIZE_TAG { SMALL = 0x10, LARGE } SIZE_ALIAS;

typedef struct _DEMO_INFO
{
	char Name[16]; //the name
	int Id;
	long Width;
	unsigned char Flags[4];
	COLOR_KIND Color;
	double Scale;
	char Unused[8];
} DEMO_INFO;

typedef DEMO_INFO DEMO_ALIAS;

#ifndef __cplusplus
#define COLOR_KIND int
#endif

#ifdef __cplusplus
extern "C" {
#endif

/***** the functions *****/
DEMO_API  int DemoCount();
DEMO_API int DemoOpen(int id);
DEMO_API COLOR_KIND DemoGet(int  id, long *plValue, COLOR_KIND *pColor);
DEMO_API  int DemoFill(int iCameraID, unsigned char* pBuffer, long lSize, int iWaitms);
DEMO_API char* DemoVersion();
DEMO_API int DemoInfo(DEMO_INFO *pInfo, int iIndex);
DEMO_API int DemoVoid(void);
DEMO_API int DemoUnnamed(int, long);

#ifdef __cplusplus
}
#endif

#endif
"""


@pytest.fixture(scope="module")
def demo() -> Header:
    return parse_header(DEMO)


class TestParser:
    def test_comments_and_preprocessor_lines_are_gone(self) -> None:
        tokens = tokenize(DEMO)
        assert "#" not in tokens
        assert "DEMO_ID_MAX" not in tokens  # a preprocessor line
        assert "implicit" not in tokens  # a comment
        assert "the" not in tokens
        assert tokens[:4] == ["typedef", "enum", "COLOR_KIND", "{"]

    def test_enumerators_take_implicit_explicit_hex_and_negative_values(self, demo: Header) -> None:
        assert demo.enums["COLOR_KIND"].members == {
            "COLOR_RED": 0,
            "COLOR_GREEN": 1,
            "COLOR_BLUE": 5,
            "COLOR_NONE": -1,
        }
        assert demo.enums["SIZE_ALIAS"].members == {"SMALL": 16, "LARGE": 17}  # keyed by the alias

    def test_a_structure_keeps_the_order_the_types_and_the_array_lengths(
        self, demo: Header
    ) -> None:
        fields = demo.structs["DEMO_INFO"].fields
        assert [name for name, _ in fields] == [
            "Name", "Id", "Width", "Flags", "Color", "Scale", "Unused",
        ]  # fmt: skip
        assert dict(fields) == {
            "Name": CType("char", 0, 16),
            "Id": CType("int"),
            "Width": CType("long"),
            "Flags": CType("unsigned char", 0, 4),
            "Color": CType("COLOR_KIND"),
            "Scale": CType("double"),
            "Unused": CType("char", 0, 8),
        }

    def test_a_typedef_of_a_name_is_an_alias(self, demo: Header) -> None:
        assert demo.aliases == {"DEMO_ALIAS": "DEMO_INFO"}

    def test_prototypes_have_their_return_type_and_argument_types(self, demo: Header) -> None:
        def types(name: str) -> tuple[str, list[str]]:
            function = demo.functions[name]
            return str(function.returns), [str(param) for param in function.params]

        assert types("DemoCount") == ("int", [])
        assert types("DemoOpen") == ("int", ["int"])
        assert types("DemoGet") == ("COLOR_KIND", ["int", "long *", "COLOR_KIND *"])
        assert types("DemoFill") == ("int", ["int", "unsigned char *", "long", "int"])
        assert types("DemoVersion") == ("char *", [])
        assert types("DemoInfo") == ("int", ["DEMO_INFO *", "int"])
        assert types("DemoVoid") == ("int", [])  # `(void)` has no arguments
        assert types("DemoUnnamed") == ("int", ["int", "long"])  # names are optional
        assert len(demo.functions) == 8  # the extern "C" block and its closing brace do not matter

    def test_a_construct_that_it_does_not_know_is_an_error(self) -> None:
        with pytest.raises(HeaderError):
            parse_header("typedef enum { A = ")
        with pytest.raises(HeaderError, match="no type or no name"):
            parse_header("typedef struct { ; } X;")

    def test_it_reads_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "demo.h"
        path.write_text(DEMO, encoding="utf-8")
        assert set(read_header(path).functions) == set(parse_header(DEMO).functions)


class TestCTypes:
    def test_c_types_map_to_the_platform_types(self, demo: Header) -> None:
        assert to_ctypes(CType("int"), demo) is ctypes.c_int
        assert to_ctypes(CType("long"), demo) is ctypes.c_long  # 4 bytes on Windows, 8 on Linux
        assert to_ctypes(CType("COLOR_KIND"), demo) is ctypes.c_int  # an enumeration is an int
        assert to_ctypes(CType("double"), demo) is ctypes.c_double
        assert to_ctypes(CType("char", 0, 16), demo) is ctypes.c_char * 16
        assert to_ctypes(CType("int", 1), demo) is ctypes.POINTER(ctypes.c_int)
        assert to_ctypes(CType("COLOR_KIND", 1), demo) is ctypes.POINTER(ctypes.c_int)
        assert to_ctypes(CType("unsigned char", 1), demo) is ctypes.POINTER(ctypes.c_ubyte)
        assert to_ctypes(CType("char", 1), demo) is ctypes.c_char_p
        assert to_ctypes(CType("void", 1), demo) is ctypes.c_void_p
        assert to_ctypes(CType("void"), demo) is None

    def test_an_alias_resolves_to_its_structure(self, demo: Header) -> None:
        structure = to_ctypes(CType("DEMO_ALIAS"), demo)
        assert issubclass(structure, ctypes.Structure)
        assert [name for name, _ in structure._fields_] == [
            "Name", "Id", "Width", "Flags", "Color", "Scale", "Unused",
        ]  # fmt: skip

    def test_a_type_that_it_does_not_know_is_an_error(self, demo: Header) -> None:
        with pytest.raises(HeaderError, match="MYSTERY"):
            to_ctypes(CType("MYSTERY"), demo)

    def test_the_built_structure_has_the_layout_of_the_platform(self, demo: Header) -> None:
        built = build_structure(demo.structs["DEMO_INFO"], demo)
        assert built.Width.offset % ctypes.sizeof(ctypes.c_long) == 0
        assert ctypes.sizeof(built) % ctypes.alignment(ctypes.c_double) == 0


# --- The comparison, on a small binding -----------------------------------------------------


class Color(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 5
    NONE = -1


class DemoInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 16),
        ("ident", ctypes.c_int),
        ("width", ctypes.c_long),
        ("flags", ctypes.c_ubyte * 4),
        ("color", ctypes.c_int),
        ("scale", ctypes.c_double),
        ("reserved", ctypes.c_char * 8),
    ]


FIELD_NAMES = {
    "Name": "name",
    "Id": "ident",
    "Width": "width",
    "Flags": "flags",
    "Color": "color",
    "Scale": "scale",
    "Unused": "reserved",
}
BUFFER = ctypes.c_void_p
TABLE: dict[str, tuple[object, list[object]]] = {
    "DemoCount": (ctypes.c_int, []),
    "DemoOpen": (ctypes.c_int, [ctypes.c_int]),
    "DemoGet": (
        ctypes.c_int,
        [ctypes.c_int, ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_int)],
    ),
    "DemoFill": (ctypes.c_int, [ctypes.c_int, BUFFER, ctypes.c_long, ctypes.c_int]),
    "DemoVersion": (ctypes.c_char_p, []),
    "DemoInfo": (ctypes.c_int, [ctypes.POINTER(DemoInfo), ctypes.c_int]),
}


def color_enum(**members: int) -> type[IntEnum]:
    """A binding with these members, to test the comparison on a binding that differs."""
    return cast("type[IntEnum]", IntEnum("Color", members))


def enum_report(header: Header, binding: type[IntEnum]) -> Report:
    report = Report()
    compare_enum(header, "COLOR_KIND", binding, lambda member: "COLOR_" + member, report)
    return report


def struct_report(header: Header, binding: type[ctypes.Structure]) -> Report:
    report = Report()
    compare_struct(header, "DEMO_INFO", binding, FIELD_NAMES, {"DEMO_INFO": DemoInfo}, report)
    return report


def function_report(header: Header, table: dict[str, tuple[object, list[object]]]) -> Report:
    report = Report()
    compare_functions(header, table, {"DEMO_INFO": DemoInfo}, report)
    return report


class TestComparison:
    def test_a_binding_that_matches_has_no_problems(self, demo: Header) -> None:
        assert enum_report(demo, Color).problems == []
        assert struct_report(demo, DemoInfo).problems == []
        assert function_report(demo, TABLE).problems == []

    def test_a_changed_enumerator_value_is_a_problem(self, demo: Header) -> None:
        wrong = color_enum(RED=0, GREEN=1, BLUE=6, NONE=-1)
        (problem,) = enum_report(demo, wrong).problems
        assert "Color.BLUE is 6" in problem
        assert "COLOR_BLUE the value 5" in problem

    def test_a_member_that_the_header_lacks_is_a_problem(self, demo: Header) -> None:
        wrong = color_enum(RED=0, GREEN=1, BLUE=5, NONE=-1, PURPLE=7)
        (problem,) = enum_report(demo, wrong).problems
        assert "the header has no COLOR_PURPLE" in problem

    def test_a_member_that_only_the_header_has_is_reported_and_is_no_problem(
        self, demo: Header
    ) -> None:
        fewer = color_enum(RED=0, GREEN=1, NONE=-1)
        report = enum_report(demo, fewer)
        assert report.problems == []
        assert report.extras == ["COLOR_KIND: COLOR_BLUE = 5"]

    def test_a_missing_enumeration_is_a_problem(self, demo: Header) -> None:
        report = Report()
        compare_enum(demo, "NO_SUCH_ENUM", Color, lambda member: member, report)
        assert report.problems == ["the header has no enumeration NO_SUCH_ENUM"]

    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            (  # a field type that differs. A short is no `int` on any platform.
                [("name", ctypes.c_char * 16), ("ident", ctypes.c_short)],
                "Id is int",
            ),
            (  # an array length that differs
                [("name", ctypes.c_char * 15)],
                "Name is char[16]",
            ),
            (  # the fields in another order
                [("ident", ctypes.c_int), ("name", ctypes.c_char * 16)],
                "field 0",
            ),
        ],
        ids=["type", "length", "order"],
    )
    def test_a_structure_that_differs_is_a_problem(
        self, demo: Header, fields: list[tuple[str, object]], expected: str
    ) -> None:
        rest = list(DemoInfo._fields_)[len(fields) :]
        wrong = type("Wrong", (ctypes.Structure,), {"_fields_": [*fields, *rest]})
        problems = struct_report(demo, wrong).problems
        assert problems
        assert any(expected in problem for problem in problems), problems

    def test_a_structure_with_another_field_count_is_a_problem(self, demo: Header) -> None:
        short = type("Short", (ctypes.Structure,), {"_fields_": list(DemoInfo._fields_)[:-1]})
        (problem, *_) = struct_report(demo, short).problems
        assert "7 fields in the header and 6 in the binding" in problem

    def test_a_field_that_the_header_names_differently_is_a_problem(self, demo: Header) -> None:
        renamed = {**FIELD_NAMES, "Id": "identifier"}
        report = Report()
        compare_struct(demo, "DEMO_INFO", DemoInfo, renamed, {"DEMO_INFO": DemoInfo}, report)
        assert any("field 1: the header has Id" in problem for problem in report.problems)

    def test_a_missing_structure_is_a_problem(self, demo: Header) -> None:
        report = Report()
        compare_struct(demo, "NO_SUCH", DemoInfo, FIELD_NAMES, {}, report)
        assert report.problems == ["the header has no structure NO_SUCH"]

    def test_a_function_that_differs_is_a_problem(self, demo: Header) -> None:
        wrong = dict(TABLE)
        wrong["DemoOpen"] = (ctypes.c_short, [ctypes.c_int])  # the wrong return type
        wrong["DemoFill"] = (ctypes.c_int, [ctypes.c_int, BUFFER, ctypes.c_short, ctypes.c_int])
        wrong["DemoGet"] = (ctypes.c_int, [ctypes.c_int])  # the wrong number of arguments
        wrong["DemoMissing"] = (ctypes.c_int, [])  # a function that the header lacks
        problems = function_report(demo, wrong).problems
        assert any("DemoOpen returns int" in problem for problem in problems)
        assert any("DemoFill argument 3 is long" in problem for problem in problems)
        assert any("DemoGet takes 3 arguments in the header" in problem for problem in problems)
        assert any("the header has no function DemoMissing" in problem for problem in problems)

    def test_a_buffer_can_be_a_void_pointer_or_an_unsigned_char_pointer(self, demo: Header) -> None:
        table = dict(TABLE)
        table["DemoFill"] = (
            ctypes.c_int,
            [ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_long, ctypes.c_int],
        )
        assert function_report(demo, table).problems == []

    def test_functions_that_the_binding_does_not_call_are_reported_as_extras(
        self, demo: Header
    ) -> None:
        extras = function_report(demo, TABLE).extras
        assert extras == ["function DemoVoid", "function DemoUnnamed"]


# --- The check against the real header ------------------------------------------------------


def vendor_header() -> Header:
    value = os.environ.get(HEADER_ENV)
    if not value or not Path(value).is_file():
        pytest.skip(f"set {HEADER_ENV} to the vendor header ASICamera2.h to compare it")
    return read_header(Path(value))


class TestVendorHeader:
    """The binding against `ASICamera2.h`. These tests skip without the file."""

    def test_the_enumerations_match(self) -> None:
        header = vendor_header()
        report = Report()
        from tests.hardware.asi_header import ENUMERATIONS

        for name, binding, member_name in ENUMERATIONS:
            compare_enum(header, name, binding, member_name, report)
        assert report.problems == []
        for extra in report.extras:
            print(f"the header has more than the binding lists: {extra}")

    def test_the_structures_match_field_for_field_and_in_layout(self) -> None:
        header = vendor_header()
        report = compare_with_binding(header)
        problems = [
            p for p in report.problems if p.startswith(("ASI_CAMERA_INFO", "ASI_CONTROL_CAPS"))
        ]
        assert problems == []

    def test_the_function_signatures_match(self) -> None:
        header = vendor_header()
        report = Report()
        from seeingmon.hardware.asi.ctypes_api import _FUNCTIONS
        from tests.hardware.asi_header import STRUCTURES

        structs = {name: binding for name, binding, _ in STRUCTURES}
        compare_functions(header, _FUNCTIONS, structs, report)
        assert report.problems == []
        print(f"the binding calls {len(_FUNCTIONS)} of {len(header.functions)} functions")

    def test_the_whole_binding_matches(self) -> None:
        report = compare_with_binding(vendor_header())
        assert report.problems == []
