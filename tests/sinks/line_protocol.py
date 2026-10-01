"""A small reference parser of InfluxDB line protocol, for the tests of the encoder.

It follows the documented grammar and nothing else, so that a test can check the encoder against
an independent reading of the format:

    <measurement>[,<tag>=<value>...] <field>=<value>[,<field>=<value>...] [<timestamp>]

- In a measurement, a backslash escapes a comma or a space. In a tag key, a tag value, and a field
  key, a backslash escapes a comma, an equals sign, or a space. A backslash before any other
  character is a literal backslash.
- A string field value is in double quotes. Inside it, `\\"` is a quote and `\\\\` is a backslash.
  A string may hold a space, a comma, a newline, and an equals sign.
- An integer ends in `i`. `true` and `false` are booleans. Any other number is a float.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HEAD_ESCAPES = {",", "=", " "}


@dataclass(frozen=True)
class Point:
    measurement: str
    tags: dict[str, str]
    fields: dict[str, Any]
    timestamp: int | None


class ParseError(ValueError):
    pass


def _read_escaped(text: str, start: int, stops: str, escapes: set[str]) -> tuple[str, int]:
    """Read up to an unescaped character in `stops`. Return the text and the index of the stop."""
    out: list[str] = []
    i = start
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text) and text[i + 1] in escapes:
            out.append(text[i + 1])
            i += 2
            continue
        if char in stops:
            break
        out.append(char)
        i += 1
    return "".join(out), i


def _read_string(text: str, start: int) -> tuple[str, int]:
    """Read a quoted string that starts at `start` (the opening quote). Return the end index."""
    out: list[str] = []
    i = start + 1
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text) and text[i + 1] in {'"', "\\"}:
            out.append(text[i + 1])
            i += 2
            continue
        if char == '"':
            return "".join(out), i + 1
        out.append(char)
        i += 1
    raise ParseError("a string field is not closed")


def _decode_scalar(token: str) -> Any:
    if token in ("true", "false"):
        return token == "true"
    if token.endswith("i"):
        return int(token[:-1])
    return float(token)


def parse_line(line: str) -> Point:
    """Parse one line of line protocol. Raises `ParseError` for a malformed line."""
    measurement, i = _read_escaped(line, 0, ", ", {",", " "})
    tags: dict[str, str] = {}
    while i < len(line) and line[i] == ",":
        key, i = _read_escaped(line, i + 1, "=", HEAD_ESCAPES)
        if i >= len(line) or line[i] != "=":
            raise ParseError("a tag has no value")
        value, i = _read_escaped(line, i + 1, ", ", HEAD_ESCAPES)
        tags[key] = value
    if i >= len(line) or line[i] != " ":
        raise ParseError("the line has no field set")
    i += 1
    fields: dict[str, Any] = {}
    while True:
        key, i = _read_escaped(line, i, "=", HEAD_ESCAPES)
        if i >= len(line) or line[i] != "=":
            raise ParseError("a field has no value")
        i += 1
        if i < len(line) and line[i] == '"':
            fields[key], i = _read_string(line, i)
        else:
            end = i
            while end < len(line) and line[end] not in ", ":
                end += 1
            fields[key] = _decode_scalar(line[i:end])
            i = end
        if i < len(line) and line[i] == ",":
            i += 1
            continue
        break
    timestamp = None
    if i < len(line):
        if line[i] != " ":
            raise ParseError("the field set ends badly")
        timestamp = int(line[i + 1 :])
    return Point(measurement, tags, fields, timestamp)


def split_lines(body: str) -> list[str]:
    """Split a request body into lines. A newline inside a quoted string does not end a line.

    Quotes count only in the field set, after the first unescaped space. A tag value may hold one.
    """
    lines: list[str] = []
    current: list[str] = []
    quoted = False
    in_fields = False
    i = 0
    while i < len(body):
        char = body[i]
        if char == "\\" and i + 1 < len(body):
            current.append(char)
            current.append(body[i + 1])
            i += 2
            continue
        if char == " " and not in_fields:
            in_fields = True
        elif char == '"' and in_fields:
            quoted = not quoted
        if char == "\n" and not quoted:
            lines.append("".join(current))
            current = []
            in_fields = False
        else:
            current.append(char)
        i += 1
    if current:
        lines.append("".join(current))
    return lines
