"""Read the parts of the vendor header that the binding mirrors, and compare them with the binding.

The vendor describes its C interface in `ASICamera2.h`. The binding (`seeingmon.hardware.asi`)
copies what it needs in its own words: the enumerations, the two structures that the SDK fills, and
the argument types of the functions that it calls. A new SDK release can change any of them, and a
wrong structure layout does not fail loudly: it reads garbage. This module parses the header and
compares it with the binding, so that a person runs the check against the header of a release, on
the platform that will use it (the Raspberry Pi has its own header and its own `long`). On a
platform where `long` and `int` have the same size, `ctypes` makes `c_int` an alias of `c_long`, and
the check cannot tell them apart there.

The repository never holds the header. `test_asi_header` reads the file that
`SEEINGMON_ASI__HEADER_PATH` names, and it skips without one.

**The parser** handles the C that the header uses: comments, preprocessor lines (which it skips),
`extern "C"` blocks, `typedef enum` and `typedef struct` with their tags and aliases, enumerators
with implicit and explicit values, fields with array lengths, and function prototypes with or
without parameter names. It is not a C parser.

**The comparison** checks, for each enumeration that the binding mirrors, that every member of the
binding exists in the header under its mapped name with the same value. For each structure, it
checks the field count, the order, the type of each field (after the mapping of C types to
`ctypes`, with `long` as the platform's `c_long`), and the layout of a structure that it builds from
the header. For each function in the binding's table, it checks the name, the return type, and the
types of the arguments. A member, a field, or a function that the header has and the binding lacks
is not a failure: it goes in `Report.extras`, because a newer SDK adds controls.
"""

from __future__ import annotations

import ctypes
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from seeingmon.hardware.asi.api import (
    AsiControl,
    AsiErrorCode,
    AsiExposureStatus,
    AsiImageType,
)
from seeingmon.hardware.asi.ctypes_api import (
    _FUNCTIONS,
    _CameraInfoStruct,
    _ControlCapsStruct,
)

HEADER_ENV = "SEEINGMON_ASI__HEADER_PATH"


class HeaderError(ValueError):
    """The header has a construct that this parser does not understand."""


@dataclass(frozen=True, slots=True)
class CType:
    """A C type as the header writes it: a base name, a pointer depth, and an array length."""

    name: str
    pointers: int = 0
    array: int | None = None

    def __str__(self) -> str:
        text = self.name + " " + "*" * self.pointers if self.pointers else self.name
        return text if self.array is None else f"{text}[{self.array}]"


@dataclass(frozen=True, slots=True)
class EnumDef:
    name: str
    members: dict[str, int]


@dataclass(frozen=True, slots=True)
class StructDef:
    name: str
    fields: tuple[tuple[str, CType], ...]


@dataclass(frozen=True, slots=True)
class FunctionDef:
    name: str
    returns: CType
    params: tuple[CType, ...]


@dataclass(slots=True)
class Header:
    enums: dict[str, EnumDef] = field(default_factory=dict)
    structs: dict[str, StructDef] = field(default_factory=dict)
    functions: dict[str, FunctionDef] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)


# --- The parser -----------------------------------------------------------------------------

_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
_TOKEN = re.compile(r'"[^"\n]*"|[A-Za-z_]\w*|0[xX][0-9A-Fa-f]+|\d+|[{}()\[\];,=*\-]')
_BASE_WORDS = frozenset({"int", "long", "short", "char", "float", "double", "void", "unsigned"})
_IGNORED_WORDS = frozenset({"const", "struct", "enum", "extern"})


def _noise(word: str) -> bool:
    """A word that carries no type: a qualifier, or the export macro (`ASICAMERA_API`)."""
    return word in _IGNORED_WORDS or word.endswith("_API")


def tokenize(text: str) -> list[str]:
    """The tokens of the header, without comments and preprocessor lines."""
    lines = _COMMENT.sub(" ", text).splitlines()
    return _TOKEN.findall("\n".join(line for line in lines if not line.lstrip().startswith("#")))


def _number(token: str) -> int:
    return int(token, 16) if token.lower().startswith("0x") else int(token)


def _ctype(words: Sequence[str], pointers: int, array: int | None = None) -> CType:
    named = [word for word in words if not _noise(word)]
    if not named:
        raise HeaderError("a declaration has no type")
    return CType(" ".join(named), pointers, array)


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self, offset: int = 0) -> str:
        position = self.index + offset
        return self.tokens[position] if position < len(self.tokens) else ""

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if not token or (expected is not None and token != expected):
            raise HeaderError(f"expected {expected!r}, found {token!r} near token {self.index}")
        self.index += 1
        return token

    def until(self, stop: str) -> list[str]:
        found: list[str] = []
        while self.peek() and self.peek() != stop:
            found.append(self.take())
        self.take(stop)
        return found

    def parse(self) -> Header:
        header = Header()
        while self.index < len(self.tokens):
            token = self.peek()
            if token == "typedef":
                self.take()
                self.typedef(header)
            elif token == "extern" and self.peek(1).startswith('"'):
                self.take()
                self.take()
                self.take("{")  # the matching "}" is skipped below
            elif token in ("}", ";"):
                self.take()
            else:
                self.function(header)
        return header

    # typedef enum / struct / alias

    def typedef(self, header: Header) -> None:
        kind = self.peek()
        if kind in ("enum", "struct"):
            self.take()
            if self.peek() != "{":
                self.take()  # the tag
            self.take("{")
            if kind == "enum":
                members = self.enum_body()
                alias = self.take()
                header.enums[alias] = EnumDef(alias, members)
            else:
                fields = self.struct_body()
                alias = self.take()
                header.structs[alias] = StructDef(alias, tuple(fields))
            self.take(";")
            return
        words = self.until(";")
        if len(words) == 2:
            header.aliases[words[1]] = words[0]

    def enum_body(self) -> dict[str, int]:
        members: dict[str, int] = {}
        value = -1
        while self.peek() != "}":
            name = self.take()
            if self.peek() == "=":
                self.take()
                sign = -1 if self.peek() == "-" else 1
                if sign < 0:
                    self.take()
                value = sign * _number(self.take())
            else:
                value += 1
            members[name] = value
            if self.peek() == ",":
                self.take()
        self.take("}")
        return members

    def struct_body(self) -> list[tuple[str, CType]]:
        fields: list[tuple[str, CType]] = []
        while self.peek() != "}":
            declaration = self.until(";")
            if len(declaration) < 2:
                raise HeaderError(f"a field of a structure has no type or no name: {declaration}")
            array: int | None = None
            if "[" in declaration:
                open_at = declaration.index("[")
                array = _number(declaration[open_at + 1])
                declaration = declaration[:open_at]
            name = declaration[-1]
            type_tokens = declaration[:-1]
            pointers = type_tokens.count("*")
            fields.append(
                (name, _ctype([word for word in type_tokens if word != "*"], pointers, array))
            )
        self.take("}")
        return fields

    # function prototypes

    def function(self, header: Header) -> None:
        statement = self.until(";")
        if "(" not in statement:
            return  # not a prototype
        open_at = statement.index("(")
        head = [word for word in statement[:open_at] if not _noise(word)]
        name = head[-1]
        returns = _ctype([w for w in head[:-1] if w != "*"], head[:-1].count("*"))
        inside = statement[open_at + 1 : len(statement) - 1 - statement[::-1].index(")")]
        params: list[CType] = []
        group: list[str] = []
        for word in [*inside, ","]:
            if word != ",":
                group.append(word)
                continue
            if group and group != ["void"]:
                params.append(self.parameter(group))
            group = []
        header.functions[name] = FunctionDef(name, returns, tuple(params))

    @staticmethod
    def parameter(group: list[str]) -> CType:
        pointers = group.count("*")
        words = [word for word in group if word != "*" and not _noise(word)]
        if len(words) > 1 and words[-1] not in _BASE_WORDS:
            words = words[:-1]  # the last word is the name of the parameter
        return _ctype(words, pointers)


def parse_header(text: str) -> Header:
    """Parse the enumerations, the structures, the aliases, and the prototypes of a header."""
    return _Parser(tokenize(text)).parse()


def read_header(path: Path) -> Header:
    return parse_header(path.read_text(encoding="utf-8", errors="replace"))


# --- C types as ctypes ----------------------------------------------------------------------

_BASE_CTYPES: Mapping[str, Any] = {
    "char": ctypes.c_char,
    "unsigned char": ctypes.c_ubyte,
    "int": ctypes.c_int,
    "unsigned int": ctypes.c_uint,
    "long": ctypes.c_long,
    "unsigned long": ctypes.c_ulong,
    "short": ctypes.c_short,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
}


def resolve(name: str, header: Header) -> str:
    """Follow the `typedef` aliases to the name of a base type, an enumeration, or a structure."""
    seen: set[str] = set()
    while name in header.aliases and name not in seen:
        seen.add(name)
        name = header.aliases[name]
    return name


def to_ctypes(ctype: CType, header: Header, structs: Mapping[str, Any] | None = None) -> Any:
    """The `ctypes` type that the platform's C compiler would give `ctype`.

    An enumeration is a C `int`. A structure maps to the class in `structs` under its name, or else
    to a class that the function builds from the header. `char *` is `c_char_p` and `void *` is
    `c_void_p`.
    """
    structs = structs or {}
    name = resolve(ctype.name, header)
    if name in header.enums:
        base: Any = ctypes.c_int
    elif name in _BASE_CTYPES:
        base = _BASE_CTYPES[name]
    elif name in structs:
        base = structs[name]
    elif name in header.structs:
        base = build_structure(header.structs[name], header, structs)
    elif name == "void":
        if ctype.pointers == 0:
            return None
        base = None
    else:
        raise HeaderError(f"the header uses a type that this check does not know: {name!r}")
    if base is ctypes.c_char and ctype.pointers == 1 and ctype.array is None:
        return ctypes.c_char_p
    if base is None and ctype.pointers == 1:
        return ctypes.c_void_p
    for _ in range(ctype.pointers):
        base = ctypes.POINTER(base)
    return base if ctype.array is None else base * ctype.array


def build_structure(
    struct: StructDef, header: Header, structs: Mapping[str, Any] | None = None
) -> type[ctypes.Structure]:
    """A `ctypes` structure with the fields of the header, in the layout of this platform."""
    fields = [(name, to_ctypes(ctype, header, structs)) for name, ctype in struct.fields]
    return type(struct.name, (ctypes.Structure,), {"_fields_": fields})


# --- The comparison -------------------------------------------------------------------------


@dataclass(slots=True)
class Report:
    """What the comparison found. `problems` are differences, and `extras` are header entries that
    the binding does not list, which a newer SDK adds."""

    problems: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)


def _error_name(member: str) -> str:
    renames = {
        "INVALID_FILE_FORMAT": "INVALID_FILEFORMAT",
        "INVALID_IMAGE_TYPE": "INVALID_IMGTYPE",
        "OUT_OF_BOUNDARY": "OUTOF_BOUNDARY",
        "GPS_VERSION_ERROR": "GPS_VER_ERR",
        "GPS_PARAMETER_OUT_OF_RANGE": "GPS_PARAM_OUT_OF_RANGE",
        "GPS_FPGA_ERROR": "GPS_FPGA_ERR",
    }
    return "ASI_SUCCESS" if member == "SUCCESS" else "ASI_ERROR_" + renames.get(member, member)


def _control_name(member: str) -> str:
    renames = {
        "BANDWIDTH_OVERLOAD": "BANDWIDTHOVERLOAD",
        "AUTO_MAX_EXPOSURE": "AUTO_MAX_EXP",
        "COOLER_POWER_PERCENT": "COOLER_POWER_PERC",
        "TARGET_TEMPERATURE": "TARGET_TEMP",
    }
    return "ASI_" + renames.get(member, member)


ENUMERATIONS: tuple[tuple[str, type[IntEnum], Callable[[str], str]], ...] = (
    ("ASI_ERROR_CODE", AsiErrorCode, _error_name),
    ("ASI_IMG_TYPE", AsiImageType, lambda member: "ASI_IMG_" + member),
    ("ASI_CONTROL_TYPE", AsiControl, _control_name),
    ("ASI_EXPOSURE_STATUS", AsiExposureStatus, lambda member: "ASI_EXP_" + member),
)

# The header's field names, mapped to the binding's.
CAMERA_INFO_FIELDS: Mapping[str, str] = {
    "Name": "name",
    "CameraID": "camera_id",
    "MaxHeight": "max_height",
    "MaxWidth": "max_width",
    "IsColorCam": "is_color",
    "BayerPattern": "bayer_pattern",
    "SupportedBins": "supported_bins",
    "SupportedVideoFormat": "supported_formats",
    "PixelSize": "pixel_size_um",
    "MechanicalShutter": "mechanical_shutter",
    "ST4Port": "st4_port",
    "IsCoolerCam": "is_cooled",
    "IsUSB3Host": "is_usb3_host",
    "IsUSB3Camera": "is_usb3_camera",
    "ElecPerADU": "electrons_per_adu",
    "BitDepth": "bit_depth",
    "IsTriggerCam": "is_trigger_camera",
    "Unused": "reserved",
}
CONTROL_CAPS_FIELDS: Mapping[str, str] = {
    "Name": "name",
    "Description": "description",
    "MaxValue": "max_value",
    "MinValue": "min_value",
    "DefaultValue": "default_value",
    "IsAutoSupported": "is_auto_supported",
    "IsWritable": "is_writable",
    "ControlType": "control_type",
    "Unused": "reserved",
}
STRUCTURES: tuple[tuple[str, type[ctypes.Structure], Mapping[str, str]], ...] = (
    ("ASI_CAMERA_INFO", _CameraInfoStruct, CAMERA_INFO_FIELDS),
    ("ASI_CONTROL_CAPS", _ControlCapsStruct, CONTROL_CAPS_FIELDS),
)

# A buffer that the SDK fills can be declared as `unsigned char *` or as `void *` on the binding's
# side, because the binding passes a buffer of bytes.
_EQUIVALENT: Mapping[Any, tuple[Any, ...]] = {
    ctypes.POINTER(ctypes.c_ubyte): (ctypes.c_void_p,),
}


def compare_enum(
    header: Header,
    header_name: str,
    binding: type[IntEnum],
    header_member: Callable[[str], str],
    report: Report,
) -> None:
    definition = header.enums.get(header_name)
    if definition is None:
        report.problems.append(f"the header has no enumeration {header_name}")
        return
    mapped = set()
    for member in binding:
        wanted = header_member(member.name)
        mapped.add(wanted)
        if wanted not in definition.members:
            report.problems.append(
                f"{binding.__name__}.{member.name} ({int(member)}): the header has no {wanted}"
            )
        elif definition.members[wanted] != int(member):
            report.problems.append(
                f"{binding.__name__}.{member.name} is {int(member)}, "
                f"and the header gives {wanted} the value {definition.members[wanted]}"
            )
    report.extras.extend(
        f"{header_name}: {name} = {value}"
        for name, value in definition.members.items()
        if name not in mapped and not name.endswith("_END")
    )


def compare_struct(
    header: Header,
    header_name: str,
    binding: type[ctypes.Structure],
    names: Mapping[str, str],
    structs: Mapping[str, Any],
    report: Report,
) -> None:
    definition = header.structs.get(header_name)
    if definition is None:
        report.problems.append(f"the header has no structure {header_name}")
        return
    fields = list(binding._fields_)
    if len(fields) != len(definition.fields):
        report.problems.append(
            f"{header_name} has {len(definition.fields)} fields in the header and "
            f"{len(fields)} in the binding"
        )
    for index, (header_field, ctype) in enumerate(definition.fields[: len(fields)]):
        binding_name, binding_type = fields[index][0], fields[index][1]
        if names.get(header_field) != binding_name:
            report.problems.append(
                f"{header_name} field {index}: the header has {header_field}, "
                f"and the binding has {binding_name}"
            )
        expected = to_ctypes(ctype, header, structs)
        if binding_type is not expected:
            report.problems.append(
                f"{header_name}.{header_field} is {ctype} in the header "
                f"({getattr(expected, '__name__', expected)}), "
                f"and the binding has {getattr(binding_type, '__name__', binding_type)}"
            )
    if len(fields) == len(definition.fields):
        built = build_structure(definition, header, structs)
        if ctypes.sizeof(built) != ctypes.sizeof(binding):
            report.problems.append(
                f"{header_name} is {ctypes.sizeof(built)} bytes in this platform's layout of the "
                f"header, and {ctypes.sizeof(binding)} bytes in the binding"
            )
        for (header_field, _), spec in zip(definition.fields, fields, strict=True):
            binding_name = spec[0]
            if getattr(built, header_field).offset != getattr(binding, binding_name).offset:
                report.problems.append(
                    f"{header_name}.{header_field} starts at byte "
                    f"{getattr(built, header_field).offset} in the header's layout, and at byte "
                    f"{getattr(binding, binding_name).offset} in the binding's"
                )


def _same_type(binding: Any, expected: Any) -> bool:
    return binding is expected or binding in _EQUIVALENT.get(expected, ())


def compare_functions(
    header: Header,
    table: Mapping[str, tuple[Any, list[Any]]],
    structs: Mapping[str, Any],
    report: Report,
) -> None:
    for name, (restype, argtypes) in table.items():
        definition = header.functions.get(name)
        if definition is None:
            report.problems.append(f"the header has no function {name}")
            continue
        expected_return = to_ctypes(definition.returns, header, structs)
        if not _same_type(restype, expected_return):
            report.problems.append(
                f"{name} returns {definition.returns} in the header, and the binding declares "
                f"{getattr(restype, '__name__', restype)}"
            )
        if len(argtypes) != len(definition.params):
            report.problems.append(
                f"{name} takes {len(definition.params)} arguments in the header, "
                f"and the binding declares {len(argtypes)}"
            )
            continue
        for position, (argtype, ctype) in enumerate(
            zip(argtypes, definition.params, strict=True), start=1
        ):
            expected = to_ctypes(ctype, header, structs)
            if not _same_type(argtype, expected):
                report.problems.append(
                    f"{name} argument {position} is {ctype} in the header, and the binding "
                    f"declares {getattr(argtype, '__name__', argtype)}"
                )
    report.extras.extend(f"function {name}" for name in header.functions if name not in table)


def compare_with_binding(header: Header) -> Report:
    """Compare the header with the enumerations, structures, and functions of the binding."""
    report = Report()
    structs = {name: binding for name, binding, _ in STRUCTURES}
    for header_name, binding, member_name in ENUMERATIONS:
        compare_enum(header, header_name, binding, member_name, report)
    for header_name, binding_struct, names in STRUCTURES:
        compare_struct(header, header_name, binding_struct, names, structs, report)
    compare_functions(header, _FUNCTIONS, structs, report)
    return report
