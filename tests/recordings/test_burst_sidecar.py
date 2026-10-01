"""The JSON sidecar of a burst."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from seeingmon.frames import PixelFormat, Roi, StreamConfig, StreamKind, TimeQuality
from seeingmon.recordings.sidecar import (
    SIDECAR_SCHEMA_VERSION,
    BurstSidecar,
    SidecarError,
    burst_sidecar_path,
    read_burst_sidecar,
    write_burst_sidecar,
)

FULL = BurstSidecar(
    profile_id="asi294mm-gs250",
    stream=StreamConfig(
        mode="bin2",
        exposure_us=10_000,
        gain=100,
        pixel_format=PixelFormat.RAW8,
        roi=Roi(16, 8, 320, 240),
        kind=StreamKind.VIDEO,
        offset=10,
        bandwidth_pct=72,
        high_speed=True,
    ),
    adc_bits=14,
    time_quality=TimeQuality.ESTIMATED,
    frame_period_s=0.0102188,
    frame_count=5874,
    start_utc_ns=1_767_268_800_123_456_700,
    temperature_c=20.5,
)
MINIMAL = BurstSidecar(
    profile_id="p1",
    stream=StreamConfig(mode="bin1", exposure_us=2000, gain=0),
    adc_bits=12,
    time_quality=TimeQuality.FITTED,
)


def tampered(tmp_path: Path, mutate: Callable[[dict[str, Any]], object]) -> Path:
    """Write `FULL`, change its JSON, and return the path of the changed file."""
    path = tmp_path / "burst.json"
    write_burst_sidecar(path, FULL)
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestRoundTrip:
    @pytest.mark.parametrize("sidecar", [FULL, MINIMAL], ids=["full", "minimal"])
    def test_what_you_write_is_what_you_read(self, tmp_path: Path, sidecar: BurstSidecar) -> None:
        path = tmp_path / "burst.json"
        write_burst_sidecar(path, sidecar)
        assert read_burst_sidecar(path) == sidecar

    def test_the_file_holds_these_fields_and_no_others(self, tmp_path: Path) -> None:
        path = tmp_path / "burst.json"
        write_burst_sidecar(path, FULL)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {
            "schema_version",
            "profile_id",
            "stream",
            "adc_bits",
            "time_quality",
            "frame_period_s",
            "frame_count",
            "start_utc_ns",
            "temperature_c",
        }
        assert set(data["stream"]) == {
            "mode",
            "exposure_us",
            "gain",
            "pixel_format",
            "kind",
            "roi",
            "offset",
            "bandwidth_pct",
            "high_speed",
        }
        assert set(data["stream"]["roi"]) == {"x", "y", "width", "height"}
        assert data["schema_version"] == SIDECAR_SCHEMA_VERSION == 1
        assert data["stream"]["pixel_format"] == "RAW8"
        assert data["stream"]["kind"] == "video"
        assert data["time_quality"] == "ESTIMATED"

    def test_the_text_is_stable_and_complete(self, tmp_path: Path) -> None:
        path = tmp_path / "burst.json"
        write_burst_sidecar(path, MINIMAL)
        text = path.read_text(encoding="utf-8")
        assert text.endswith("}\n")
        assert "\r" not in text
        assert text.index('"adc_bits"') < text.index('"profile_id"') < text.index('"stream"')
        assert list(tmp_path.iterdir()) == [path]  # no temporary file is left behind

    def test_writing_replaces_an_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "burst.json"
        write_burst_sidecar(path, FULL)
        write_burst_sidecar(path, MINIMAL)
        assert read_burst_sidecar(path) == MINIMAL

    def test_the_path_follows_the_ser_file(self, tmp_path: Path) -> None:
        assert burst_sidecar_path(tmp_path / "burst-001.ser") == tmp_path / "burst-001.json"
        assert burst_sidecar_path(str(tmp_path / "a.b.SER")).name == "a.b.json"

    def test_unknown_keys_are_ignored(self, tmp_path: Path) -> None:
        def add(data: dict[str, Any]) -> None:
            data["future_field"] = {"anything": [1, 2, 3]}
            data["stream"]["future_setting"] = True

        assert read_burst_sidecar(tampered(tmp_path, add)) == FULL

    def test_numbers_may_be_integers_where_floats_are_expected(self, tmp_path: Path) -> None:
        def mutate(data: dict[str, Any]) -> None:
            data["frame_period_s"] = 1
            data["temperature_c"] = 20

        sidecar = read_burst_sidecar(tampered(tmp_path, mutate))
        assert (sidecar.frame_period_s, sidecar.temperature_c) == (1.0, 20.0)


class TestNoPrivateValues:
    @pytest.mark.parametrize("profile_id", ["asi294mm-gs250", "A", "a.b_c-1", "x" * 64])
    def test_ordinary_profile_ids_are_accepted(self, profile_id: str) -> None:
        assert (
            BurstSidecar(profile_id, MINIMAL.stream, 14, TimeQuality.EXACT).profile_id == profile_id
        )

    @pytest.mark.parametrize(
        "profile_id",
        [
            "",
            "x" * 65,
            "a/b",
            "a\\b",
            "a b",
            "a@b",
            ".hidden",
            "-x",
            "name:port",
            "caf\N{LATIN SMALL LETTER E WITH ACUTE}",
        ],
    )
    def test_profile_ids_cannot_carry_paths_or_addresses(self, profile_id: str) -> None:
        with pytest.raises(SidecarError, match="profile ID"):
            BurstSidecar(profile_id, MINIMAL.stream, 14, TimeQuality.EXACT)

    def test_a_reader_applies_the_same_rule(self, tmp_path: Path) -> None:
        def mutate(data: dict[str, Any]) -> None:
            data["profile_id"] = "a/b"

        with pytest.raises(SidecarError, match="profile ID"):
            read_burst_sidecar(tampered(tmp_path, mutate))

    def test_the_file_text_has_no_separators_or_host_characters(self, tmp_path: Path) -> None:
        path = tmp_path / "burst.json"
        write_burst_sidecar(path, FULL)
        text = path.read_text(encoding="utf-8")
        assert not any(char in text for char in "\\@")
        assert "/" not in text


class TestInvalidSidecars:
    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda d: d.pop("schema_version"), "'schema_version' is missing"),
            (lambda d: d.update(schema_version=0), "1 or more"),
            (lambda d: d.update(schema_version=SIDECAR_SCHEMA_VERSION + 1), "supports up to 1"),
            (lambda d: d.update(schema_version="1"), "must be an integer"),
            (lambda d: d.update(schema_version=True), "must be an integer"),
            (lambda d: d.update(schema_version=1.0), "must be an integer"),
            (lambda d: d.pop("profile_id"), "'profile_id' is missing"),
            (lambda d: d.update(profile_id=7), "must be a string"),
            (lambda d: d.pop("stream"), "'stream' is missing"),
            (lambda d: d.update(stream=[]), "must be an object"),
            (lambda d: d["stream"].pop("mode"), "'mode' is missing"),
            (lambda d: d["stream"].update(exposure_us="10"), "must be an integer"),
            (lambda d: d["stream"].update(exposure_us=0), "stream settings are not valid"),
            (lambda d: d["stream"].update(gain=-1), "stream settings are not valid"),
            (lambda d: d["stream"].update(mode=""), "stream settings are not valid"),
            (lambda d: d["stream"].update(pixel_format="RAW12"), "one of RAW8, RAW16"),
            (lambda d: d["stream"].update(kind="movie"), "stream settings are not valid"),
            (lambda d: d["stream"].update(high_speed="yes"), "must be true or false"),
            (lambda d: d["stream"].update(roi={"x": 0}), "'y' is missing"),
            (lambda d: d["stream"]["roi"].update(width=0), "stream settings are not valid"),
            (lambda d: d.update(adc_bits=0), "adc_bits"),
            (lambda d: d.update(adc_bits=17), "adc_bits"),
            (lambda d: d.update(time_quality="GREAT"), "one of INVALID, ESTIMATED, FITTED, EXACT"),
            (lambda d: d.update(frame_period_s=0), "frame_period_s"),
            (lambda d: d.update(frame_period_s="fast"), "must be a number"),
            (lambda d: d.update(frame_period_s=True), "must be a number"),
            (lambda d: d.update(frame_count=-1), "frame_count"),
            (lambda d: d.update(frame_count=1.5), "must be an integer"),
            (lambda d: d.update(temperature_c=float("nan")), "must be a number"),
            (lambda d: d.update(start_utc_ns="now"), "must be an integer"),
        ],
    )
    def test_a_bad_field_is_named(
        self, tmp_path: Path, mutate: Callable[[dict[str, Any]], object], message: str
    ) -> None:
        path = tampered(tmp_path, mutate)
        with pytest.raises(SidecarError, match=message):
            read_burst_sidecar(path)

    @pytest.mark.parametrize("text", ["not json", "[]", "7", "null", "", "{"])
    def test_text_that_is_not_a_json_object(self, tmp_path: Path, text: str) -> None:
        path = tmp_path / "burst.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(SidecarError, match=r"JSON|object"):
            read_burst_sidecar(path)

    def test_bytes_that_are_not_text(self, tmp_path: Path) -> None:
        path = tmp_path / "burst.json"
        path.write_bytes(b"\xff\xfe\x00{")
        with pytest.raises(SidecarError, match="not UTF-8"):
            read_burst_sidecar(path)

    def test_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SidecarError, match="cannot read the sidecar"):
            read_burst_sidecar(tmp_path / "missing.json")

    def test_messages_never_quote_the_content_or_the_path(self, tmp_path: Path) -> None:
        folder = tmp_path / "distinctive-folder-name"
        folder.mkdir()
        path = folder / "burst.json"
        path.write_text('{"schema_version": 1, "profile_id": "SECRETVALUE/x"}', encoding="utf-8")
        errors: list[SidecarError] = []
        for target in (path, folder / "missing.json"):
            with pytest.raises(SidecarError) as raised:
                read_burst_sidecar(target)
            errors.append(raised.value)
        with pytest.raises(SidecarError) as unwritable:
            write_burst_sidecar(folder / "no-such-folder" / "burst.json", FULL)
        errors.append(unwritable.value)
        for error in errors:
            assert "SECRETVALUE" not in str(error)
            assert "distinctive-folder-name" not in str(error)
            assert error.__cause__ is None

    def test_a_failed_write_leaves_nothing_behind(self, tmp_path: Path) -> None:
        with pytest.raises(SidecarError, match="cannot write the sidecar"):
            write_burst_sidecar(tmp_path / "missing-folder" / "burst.json", FULL)
        assert list(tmp_path.iterdir()) == []
