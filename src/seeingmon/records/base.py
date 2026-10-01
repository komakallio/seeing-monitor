"""The record base class, the field helper, and the registry of record types.

Every result that the system produces is an immutable record. You declare a record type
once, as a subclass of `Record`, and everything else comes from that declaration: the
SQLite schema (`sqlite_schema`), the sink mappings (`sink_mapping`), the API schema
(`api_schema`), the segment layout (`segments`), and the quantity reference
(`quantity_reference`, written to `docs/quantities.md`).

**Key.** A record is identified by `(station_id, record_type, t_utc_ns, revision)`. A sink
upserts by this key, so sending a record twice changes nothing. Tables are append-only:
every stored row gets a `row_id` that only grows, and the `row_id` is the sink cursor. A
correction is a new record with the next `revision`, never an update in place.

**Values.** A missing value is `None`, and `quality` says why. API field names carry their
unit (`seeing_fwhm_arcsec`). A record rejects NaN and infinity, because neither is valid
JSON and SQLite stores NaN as `NULL`.

**Declare a record type.**

```python
class ExampleRecord(Record):
    \"\"\"One sentence about what the record describes.\"\"\"

    record_type: ClassVar[str] = "example"
    storage: ClassVar[Storage] = "table"  # or "segment"
    retention_days: ClassVar[int | None] = None  # None keeps the records forever

    count: int = quantity(ge=0, definition="The number of examples.")
    value_arcsec: float | None = quantity(
        unit="arcsec", default=None, definition="The size of the example, in arcseconds."
    )
```

Declare every field with `quantity`. It needs a one-sentence definition and a unit unless the
quantity has none. The name of a field ends with its unit (`_arcsec`, `_px`, `_hz`). A field
whose type includes `None` must have a default. The supported types are `int`, `float`,
`bool`, `str`, `bytes`, `list` of `int`, `float`, `str`, or `bool`, and `dict` from `str` to
`int`, `float`, `str`, `bool`, or `Any` (any JSON value). The class registers itself when you
import its module. Add the module name to `DECLARATION_MODULES` so that the registry finds it
without an explicit import.

**Change a record type.** Add a field at the end of the class, and make it optional or give
it a default. The generated migration adds the column to an existing database. Never remove,
rename, or retype a field, because the generators refuse the change. Add a new field and stop
writing the old one. To allow a new value in a field that declares `codes`, add the code and its
meaning to the map. Then regenerate `docs/quantities.md` (`seeingmon records reference`).

**Rows.** `Record.to_row` returns a dict of JSON-compatible values (`bytes` become base64
text), and `Record.from_row` reverses it. `from_row` is lenient, because a stored row can
come from newer software: it ignores keys that the declaration does not know and it does not
check the documented codes. The constructor is strict. It accepts NumPy scalars and arrays,
but it rejects a `float` for an `int` field and a string for a number.
"""

from __future__ import annotations

import base64
import binascii
import enum
import functools
import importlib
import inspect
import json
import re
import types
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Self,
    TypeAlias,
    Union,
    get_args,
    get_origin,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    model_validator,
)
from pydantic_core import PydanticUndefined

Storage: TypeAlias = Literal["table", "segment"]
FieldKind: TypeAlias = Literal["int", "float", "bool", "str", "bytes", "json"]

STORAGE_KINDS: tuple[str, ...] = ("table", "segment")

# The columns that identify a record, together with the record type (the table name).
KEY_FIELDS: tuple[str, ...] = ("station_id", "t_utc_ns", "revision")

# Modules that declare record types. The registry imports them on first use. Keep one module
# per owner, so that lanes extend their own file and never edit another lane's declarations.
DECLARATION_MODULES: tuple[str, ...] = ("seeing", "survey", "reference", "system")

# The storage types for the per-row fields of a segment record: NumPy codes, little-endian.
_INT_DTYPES: dict[str, tuple[int, int]] = {
    "u1": (0, 2**8 - 1),
    "u2": (0, 2**16 - 1),
    "u4": (0, 2**32 - 1),
    "u8": (0, 2**64 - 1),
    "i1": (-(2**7), 2**7 - 1),
    "i2": (-(2**15), 2**15 - 1),
    "i4": (-(2**31), 2**31 - 1),
    "i8": (-(2**63), 2**63 - 1),
}
# The largest finite value of the narrow floating-point dtypes. A float64 field needs no bound.
_FLOAT_LIMITS: dict[str, float] = {"f2": 65_504.0, "f4": 3.4028234663852886e38}
_FLOAT_DTYPES: frozenset[str] = frozenset({"f2", "f4", "f8"})
SEGMENT_DTYPES: frozenset[str] = frozenset(_INT_DTYPES) | _FLOAT_DTYPES

# SQLite stores integers in 8 bytes, so a larger value cannot be stored.
_SQLITE_INT_MIN, _SQLITE_INT_MAX = _INT_DTYPES["i8"]

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,62}")  # safe as a table, column, or measurement name
_JSON_SCALARS: tuple[type, ...] = (bool, int, float, str)
_PLAIN_TYPES: frozenset[type] = frozenset({type(None), bool, int, float, str, bytes})
_CONSTRAINT_NAMES: tuple[str, ...] = (
    "ge",
    "gt",
    "le",
    "lt",
    "min_length",
    "max_length",
    "pattern",
)


def quantity(
    *,
    definition: str,
    unit: str | None = None,
    default: Any = PydanticUndefined,
    default_factory: Callable[[], Any] | None = None,
    dtype: str | None = None,
    codes: Mapping[str, str] | None = None,
    ge: float | None = None,
    gt: float | None = None,
    le: float | None = None,
    lt: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
    example: Any = PydanticUndefined,
) -> Any:
    """Declare a field of a record, with its definition and unit.

    Without `default` or `default_factory`, the field is required. A field whose type includes
    `None` must pass `default=None` (or another default).

    Args:
        definition: One sentence that says what the field means. The quantity reference
            and the API schema publish it.
        unit: The unit as a short ASCII string, such as `arcsec`, `px`, `Hz`, or `degC`.
            Leave it out for a quantity without a unit, such as a count, a fraction, or text.
        default: The default value.
        default_factory: A function that returns the default, for a list or a dict.
        dtype: The storage type of the field in a segment file, as a NumPy code such as
            `f4` or `u2`. Only the fields of a `segment` record set it. A field without a
            dtype lives in the segment header, once for every row of the segment.
        codes: The allowed values of a `str` field, or of the items of a `list[str]` field,
            as a map from each code to its meaning. The constructor rejects other values.
        ge: The minimum value, inclusive.
        gt: The minimum value, exclusive.
        le: The maximum value, inclusive.
        lt: The maximum value, exclusive.
        min_length: The minimum length of a `str`, `bytes`, `list`, or `dict`.
        max_length: The maximum length.
        pattern: A regular expression that a `str` value must match.
        example: A typical value. The API schema publishes it, and `sample_record` uses it for
            a required field. A field whose value needs a format that the type does not show,
            such as a date or a dotted code, needs an example.
    """
    text = " ".join(definition.split())
    if not text:
        raise ValueError("a field needs a definition")
    if default is not PydanticUndefined and default_factory is not None:
        raise ValueError("pass either default or default_factory, not both")
    if dtype is not None and dtype not in SEGMENT_DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; use one of {sorted(SEGMENT_DTYPES)}")
    if codes is not None and not codes:
        raise ValueError("codes must not be empty")

    extra: dict[str, Any] = {}
    if unit is not None:
        extra["unit"] = unit
    if dtype is not None:
        extra["dtype"] = dtype
    if codes is not None:
        extra["codes"] = dict(codes)

    constraints: dict[str, Any] = {
        "ge": ge,
        "gt": gt,
        "le": le,
        "lt": lt,
        "min_length": min_length,
        "max_length": max_length,
        "pattern": pattern,
    }
    if dtype in _INT_DTYPES and gt is None and lt is None:
        low, high = _INT_DTYPES[dtype]
        constraints["ge"] = low if ge is None else ge
        constraints["le"] = high if le is None else le
    kwargs = {key: value for key, value in constraints.items() if value is not None}
    if example is not PydanticUndefined:
        kwargs["examples"] = [example]
    if default_factory is not None:
        kwargs["default_factory"] = default_factory
    else:
        kwargs["default"] = default
    return Field(description=text, json_schema_extra=extra or None, **kwargs)


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """What the declaration says about one field. Build it with `field_specs`.

    `kind` is the storage class of the value: `json` stands for a list or a dict. `nullable`
    means that the type includes `None`. `has_default` means that the constructor needs no
    value. `dtype` is set for the per-row fields of a segment record. `constraints` holds the
    bounds, lengths, and pattern that the declaration sets (`ge`, `gt`, `le`, `lt`,
    `min_length`, `max_length`, and `pattern`). `examples` holds the declared example, if any.
    """

    name: str
    annotation: Any
    kind: FieldKind
    nullable: bool
    has_default: bool
    default: Any
    unit: str | None
    definition: str
    codes: Mapping[str, str] | None
    dtype: str | None
    constraints: Mapping[str, Any]
    examples: tuple[Any, ...]
    base: bool

    @property
    def required(self) -> bool:
        """Whether the constructor needs a value for the field."""
        return not self.has_default

    @property
    def type_name(self) -> str:
        """The type without `None`, such as `float` or `list[str]`."""
        return _type_name(self.annotation)


def _type_name(annotation: Any) -> str:
    if annotation is Any:
        return "Any"
    args = get_args(annotation)
    if not args and isinstance(annotation, type):
        return annotation.__name__
    origin = get_origin(annotation)
    name = getattr(origin, "__name__", str(origin))
    return f"{name}[{', '.join(_type_name(arg) for arg in args)}]"


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Split `X | None` into `X` and a flag. Any other union is an error."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = get_args(annotation)
        rest = tuple(arg for arg in args if arg is not type(None))
        if len(rest) != 1:
            raise TypeError("use one type, or one type or None")
        return rest[0], len(rest) != len(args)
    return annotation, False


def _kind_of(annotation: Any) -> FieldKind:
    scalars: dict[Any, FieldKind] = {
        bool: "bool",
        int: "int",
        float: "float",
        str: "str",
        bytes: "bytes",
    }
    if annotation in scalars:
        return scalars[annotation]
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list and len(args) == 1 and args[0] in _JSON_SCALARS:
        return "json"
    if (
        origin is dict
        and len(args) == 2
        and args[0] is str
        and (args[1] in _JSON_SCALARS or args[1] is Any)
    ):
        return "json"
    raise TypeError(
        "unsupported type; use int, float, bool, str, bytes, a list of int, float, str, or "
        "bool, or a dict from str to int, float, str, bool, or Any"
    )


def _constraints_of(info: Any) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for item in info.metadata:
        for name in _CONSTRAINT_NAMES:
            value = getattr(item, name, None)
            if value is not None:
                found[name] = value
    return found


def _build_specs(cls: type[Record]) -> tuple[FieldSpec, ...]:
    specs: list[FieldSpec] = []
    for name, info in cls.model_fields.items():
        try:
            inner, nullable = _unwrap_optional(info.annotation)
            kind = _kind_of(inner)
        except TypeError as exc:
            raise TypeError(f"{cls.__name__}.{name}: {exc} (got {info.annotation!r})") from None
        extra: dict[str, Any] = (
            dict(info.json_schema_extra) if isinstance(info.json_schema_extra, dict) else {}
        )
        has_default = not info.is_required()
        specs.append(
            FieldSpec(
                name=name,
                annotation=inner,
                kind=kind,
                nullable=nullable,
                has_default=has_default,
                default=info.get_default(call_default_factory=True) if has_default else None,
                unit=extra.get("unit"),
                definition=(info.description or "").strip(),
                codes=extra.get("codes"),
                dtype=extra.get("dtype"),
                constraints=_constraints_of(info),
                examples=tuple(info.examples or ()),
                base=name in Record.model_fields,
            )
        )
    return tuple(specs)


@functools.cache
def _specs_for(cls: type[Record]) -> tuple[FieldSpec, ...]:
    return _build_specs(cls)


def _plain(value: Any) -> Any:
    """Turn NumPy scalars and arrays, enum members, and sequences into plain Python values."""
    if type(value) in _PLAIN_TYPES:
        return value
    if type(value).__module__ == "numpy":  # a scalar or an array. Both have `tolist`.
        return _plain(value.tolist())
    if isinstance(value, enum.Enum):
        return _plain(value.value)
    if isinstance(value, bytearray | memoryview):
        return bytes(value)
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


class Record(BaseModel):
    """An immutable result of the system, keyed by `(station_id, record_type, t_utc_ns, revision)`.

    The module documentation explains how to declare a record type. Subclasses set
    `record_type`, `storage`, and `retention_days`. Pass `register=False` in the class
    statement to build a variant of a record type that stays out of the registry.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    record_type: ClassVar[str]
    storage: ClassVar[Storage] = "table"
    retention_days: ClassVar[int | None] = None

    station_id: str = quantity(
        min_length=1,
        example="station-1",
        definition="The ID of the station that produced the record, from the local configuration.",
    )
    t_utc_ns: int = quantity(
        unit="ns",
        dtype="i8",
        example=1_767_225_600_000_000_000,  # 2026-01-01T00:00:00Z
        definition=(
            "The start of the interval that the record describes, or the instant for a point "
            "record, in nanoseconds since the Unix epoch, in UTC."
        ),
    )
    revision: int = quantity(
        default=0,
        ge=0,
        definition=(
            "The revision of the result for this station and time, which is 0 for the first "
            "result and grows by one for each reprocessed result."
        ),
    )
    profile_id: str = quantity(
        min_length=1,
        example="profile-1",
        definition="The ID of the hardware profile that was active when the record was produced.",
    )
    provenance: dict[str, str] = quantity(
        example={"algo": "fast-1"},
        definition=(
            "The versions that produced the result, as a map from a component to a version "
            "string, such as the algorithm revision (`algo`) and the calibration versions."
        ),
    )
    quality: dict[str, str] | None = quantity(
        default=None,
        definition=(
            "The reason that a value is missing or uncertain, as a map from the field name to "
            "a short reason, or `null` when every field is present and trusted."
        ),
    )

    def __init_subclass__(cls, *, register: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @classmethod
    def __pydantic_init_subclass__(cls, *, register: bool = True, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if not hasattr(cls, "record_type"):
            return  # a helper class that declares no record type
        if register and "record_type" not in cls.__dict__:
            return  # a subclass that inherits the type and adds nothing to declare
        _validate_declaration(cls)
        if register:
            RECORD_TYPES.register(cls)

    @model_validator(mode="before")
    @classmethod
    def _convert_plain(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {key: _plain(value) for key, value in data.items()}
        return data

    @model_validator(mode="after")
    def _check_values(self, info: ValidationInfo) -> Self:
        cls = type(self)
        if not hasattr(cls, "record_type"):
            raise TypeError("Record is a base class. Declare a subclass with a record_type.")
        lenient = bool(info.context and info.context.get("lenient"))
        specs = field_specs(cls)
        for spec in specs:
            value = getattr(self, spec.name)
            if value is None:
                continue
            if spec.kind == "int" and not _SQLITE_INT_MIN <= value <= _SQLITE_INT_MAX:
                raise ValueError(f"{spec.name} does not fit in 8 bytes")
            limit = _FLOAT_LIMITS.get(spec.dtype or "")
            if spec.kind == "float" and limit is not None and abs(value) > limit:
                raise ValueError(f"{spec.name} does not fit in {spec.dtype}")
            if spec.kind == "json":
                try:
                    json.dumps(value, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{spec.name} must hold only JSON values: {exc}") from None
            if spec.codes is not None and not lenient:
                used = [value] if isinstance(value, str) else value
                unknown = [code for code in used if code not in spec.codes]
                if unknown:
                    raise ValueError(
                        f"{spec.name} has unknown code(s) {unknown}; "
                        f"the declared codes are {sorted(spec.codes)}"
                    )
        if self.quality and not lenient:
            unknown_names = sorted(set(self.quality) - {spec.name for spec in specs})
            if unknown_names:
                raise ValueError(
                    f"quality names fields that the record does not have: {unknown_names}"
                )
        return self

    @property
    def record_key(self) -> tuple[str, str, int, int]:
        """The key of the record: `(station_id, record_type, t_utc_ns, revision)`."""
        return (self.station_id, self.record_type, self.t_utc_ns, self.revision)

    def __hash__(self) -> int:
        """Hash the key, so that records with list or dict fields still work in a set.

        Two records that are equal have the same key, so they have the same hash.
        """
        return hash(self.record_key)

    def to_row(self) -> dict[str, Any]:
        """Return the record as a dict of JSON-compatible values, keyed by the declared names.

        A missing value is `None`. A `bytes` value becomes base64 text. `json.dumps` accepts
        the result, and the caller can change it without changing the record.
        """
        row = self.model_dump()
        for spec in field_specs(type(self)):
            if spec.kind == "bytes" and row[spec.name] is not None:
                row[spec.name] = base64.b64encode(row[spec.name]).decode("ascii")
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any], *, strict: bool = False) -> Self:
        """Build a record from the output of `to_row`, or from a stored row.

        Base64 text becomes `bytes`. By default, the method ignores keys that the declaration
        does not know (such as a column that newer software added) and it does not check the
        documented codes, so that a row from newer software still loads. Pass `strict=True`
        to reject unknown keys and unknown codes. Raises `pydantic.ValidationError` (a
        `ValueError`) when a value has the wrong type.
        """
        specs = field_specs(cls)
        names = {spec.name for spec in specs}
        if strict:
            unknown = sorted(set(row) - names)
            if unknown:
                raise ValueError(f"{cls.__name__} has no field named {unknown}")
        values = {name: row[name] for name in names if name in row}
        for spec in specs:
            value = values.get(spec.name)
            if spec.kind == "bytes" and isinstance(value, str):
                try:
                    values[spec.name] = base64.b64decode(value, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError(f"{spec.name} is not valid base64: {exc}") from None
        return cls.model_validate(values, context=None if strict else {"lenient": True})


def _validate_declaration(cls: type[Record]) -> None:
    """Check a record class when you declare it, so that a mistake fails at import."""
    name = cls.__name__
    if not _IDENTIFIER.fullmatch(cls.record_type):
        raise ValueError(f"{name}.record_type must be lowercase letters, digits, and underscores")
    if cls.storage not in STORAGE_KINDS:
        raise ValueError(f"{name}.storage must be one of {STORAGE_KINDS}, not {cls.storage!r}")
    retention = cls.retention_days
    if retention is not None and (isinstance(retention, bool) or retention < 1):
        raise ValueError(f"{name}.retention_days must be a positive number of days or None")

    row_fields = 0
    for spec in field_specs(cls):
        where = f"{name}.{spec.name}"
        if not _IDENTIFIER.fullmatch(spec.name):
            raise ValueError(f"{where}: use lowercase letters, digits, and underscores")
        if not spec.definition:
            raise ValueError(f"{where}: declare the field with quantity() and a definition")
        if spec.nullable and not spec.has_default:
            raise ValueError(f"{where}: a field that can be None needs a default, such as None")
        if spec.unit is not None and spec.kind in ("bool", "str", "bytes"):
            raise ValueError(f"{where}: a {spec.kind} field has no unit")
        if spec.codes is not None and spec.annotation not in (str, list[str]):
            raise ValueError(f"{where}: codes apply to str and list[str] fields only")
        _check_examples(cls, spec)
        if spec.dtype is None:
            continue
        if spec.dtype in _INT_DTYPES and spec.kind != "int":
            raise ValueError(f"{where}: dtype {spec.dtype} needs an int field")
        if spec.dtype in _FLOAT_DTYPES and spec.kind != "float":
            raise ValueError(f"{where}: dtype {spec.dtype} needs a float field")
        if cls.storage == "table" and not spec.base:
            raise ValueError(f"{where}: only the fields of a segment record take a dtype")
        row_fields += 1
    # A segment holds the time of each row and at least one metric.
    if cls.storage == "segment" and row_fields < 2:
        raise ValueError(f"{name}: a segment record needs per-row fields with a dtype")


def _check_examples(cls: type[Record], spec: FieldSpec) -> None:
    """Check that each declared example is a valid value of its field."""
    if not spec.examples:
        return
    info = cls.model_fields[spec.name]
    config = ConfigDict(strict=True, allow_inf_nan=False)
    adapter: TypeAdapter[Any] = TypeAdapter(Annotated[info.annotation, info], config=config)
    for example in spec.examples:
        where = f"{cls.__name__}.{spec.name}"
        try:
            adapter.validate_python(example)
        except ValidationError as exc:
            raise ValueError(f"{where}: the example {example!r} is not valid: {exc}") from None
        if spec.codes is not None:
            used = [example] if isinstance(example, str) else example
            unknown = [code for code in used if code not in spec.codes]
            if unknown:
                raise ValueError(f"{where}: the example {example!r} is not a declared code")


def field_specs(record: str | type[Record]) -> tuple[FieldSpec, ...]:
    """Return the field declarations of a record type in declaration order.

    The base fields come first (`station_id`, `t_utc_ns`, `revision`, `profile_id`,
    `provenance`, and `quality`). `record` is a record type name or a record class.
    """
    return _specs_for(resolve_record_type(record))


def doc_paragraphs(record: str | type[Record]) -> list[str]:
    """Return the paragraphs of the docstring of a record type, each one on a single line.

    The generated files describe a record with its docstring. This drops the line breaks that
    wrap the text in the source.
    """
    text = inspect.cleandoc(resolve_record_type(record).__doc__ or "")
    return [" ".join(block.split()) for block in text.split("\n\n") if block.strip()]


def base_field_specs() -> tuple[FieldSpec, ...]:
    """Return the declarations of the fields that every record has, in order."""
    return _specs_for(Record)


class RecordRegistry(Mapping[str, "type[Record]"]):
    """The record types by name. The first read imports the declaration modules.

    The order is the same on every run, whatever the import order was: first the order of
    `DECLARATION_MODULES`, then the order of the class statements in a module. The generated
    files depend on that order.
    """

    def __init__(self) -> None:
        self._types: dict[str, type[Record]] = {}
        self._sequence: dict[str, int] = {}
        self._loaded = False

    def register(self, cls: type[Record]) -> None:
        """Add a record class. Raises `ValueError` when another class has the same name."""
        existing = self._types.get(cls.record_type)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"record type {cls.record_type!r} is already declared by {existing.__name__}"
            )
        self._types[cls.record_type] = cls
        self._sequence.setdefault(cls.record_type, len(self._sequence))

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            for module in DECLARATION_MODULES:
                importlib.import_module(f"seeingmon.records.{module}")
        except BaseException:
            self._loaded = False
            raise

    def _position(self, name: str) -> tuple[int, int]:
        module = self._types[name].__module__.removeprefix("seeingmon.records.")
        rank = (
            DECLARATION_MODULES.index(module)
            if module in DECLARATION_MODULES
            else len(DECLARATION_MODULES)
        )
        return rank, self._sequence[name]

    def __getitem__(self, name: str) -> type[Record]:
        self._load()
        try:
            return self._types[name]
        except KeyError:
            known = ", ".join(self)
            raise KeyError(f"unknown record type {name!r}; the known types are {known}") from None

    def __iter__(self) -> Iterator[str]:
        self._load()
        return iter(sorted(self._types, key=self._position))

    def __len__(self) -> int:
        self._load()
        return len(self._types)


RECORD_TYPES = RecordRegistry()


def get_record_type(name: str) -> type[Record]:
    """Return the record class with the given `record_type`. Raises `KeyError` if unknown."""
    return RECORD_TYPES[name]


def resolve_record_type(record: str | type[Record]) -> type[Record]:
    """Accept a record type name or a record class, and return the class."""
    if isinstance(record, str):
        return get_record_type(record)
    candidate: object = record
    if (
        isinstance(candidate, type)
        and issubclass(candidate, Record)
        and hasattr(candidate, "record_type")
    ):
        return candidate
    raise TypeError(f"expected a record type name or a Record subclass, not {record!r}")
