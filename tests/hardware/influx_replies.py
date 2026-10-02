"""Replies that a real InfluxDB gives to a query, for the tests of the SQM-LE reader.

Version 1 answers `GET /query` with JSON, and version 2 answers `POST /api/v2/query` with annotated
CSV. The builders follow the output that those servers give: the version 1 timestamp is an integer
of nanoseconds (`epoch=ns`), and the version 2 timestamp is RFC 3339 with the trailing zeros of
the fraction cut off, as the Go server writes it. The names and values are made up.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from seeingmon.clock import NS_PER_S, utc_ns_to_iso

V1_EMPTY = '{"results":[{"statement_id":0}]}'  # no series: no point matched
V2_EMPTY = ""  # Flux sends an empty body when no table has a row
MAGNITUDE = 21.43
TEMPERATURE = 3.4


def v1_reply(
    rows: Sequence[Sequence[object]],
    *,
    columns: Sequence[str] = ("time", "mag", "temp"),
    measurement: str = "sqm",
) -> str:
    """The JSON of `GET /query` for rows of `[time_ns, magnitude, temperature]`."""
    series = [{"name": measurement, "columns": list(columns), "values": [list(r) for r in rows]}]
    return json.dumps({"results": [{"statement_id": 0, "series": series}]})


def go_time(t_utc_ns: int) -> str:
    """RFC 3339 with the fraction that Go writes: up to nine digits, no trailing zeros."""
    stamp = utc_ns_to_iso(t_utc_ns, digits=9)
    head, _, fraction = stamp[:-1].partition(".")
    fraction = fraction.rstrip("0")
    return f"{head}.{fraction}Z" if fraction else f"{head}Z"


def v2_csv(
    rows: Iterable[tuple[str, int, float]],
    *,
    unit: str = "roof",
    separate_tables: bool = False,
    line_end: str = "\r\n",
) -> str:
    """The annotated CSV of `POST /api/v2/query`, with one table for each row.

    Each row is `(field, time_ns, value)`, the way `last()` gives one table for each field. By
    default the tables share one header, as InfluxDB writes tables of the same schema. With
    `separate_tables`, each table has its own annotations and header, after an empty line.
    """
    start, stop = "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"
    annotations = [
        "#group,false,false,true,true,false,false,true,true,true",
        "#datatype,string,long,dateTime:RFC3339,dateTime:RFC3339,dateTime:RFC3339,double,"
        "string,string,string",
        "#default,_result,,,,,,,,",
        ",result,table,_start,_stop,_time,_value,_field,_measurement,unit",
    ]
    lines: list[str] = [] if separate_tables else list(annotations)
    for table, (field, t_utc_ns, value) in enumerate(rows):
        if separate_tables:
            lines += ([""] if table else []) + annotations
        lines.append(f",,{table},{start},{stop},{go_time(t_utc_ns)},{value},{field},sqm,{unit}")
    lines.append("")  # InfluxDB ends the reply with an empty line
    return line_end.join(lines) + line_end


def ago(now_ns: int, seconds: float) -> int:
    """The time stamp `seconds` before `now_ns`."""
    return now_ns - round(seconds * NS_PER_S)
