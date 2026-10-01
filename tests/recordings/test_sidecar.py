"""The SharpCap settings sidecar parser, on synthetic text that follows the real layout."""

from __future__ import annotations

import codecs
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.recordings.sidecar import (
    SidecarError,
    SidecarInfo,
    decode_sidecar_bytes,
    is_private_key,
    is_private_value,
    parse_decimal_comma,
    parse_duration_ms,
    parse_julian_date,
    parse_sharpcap_sidecar,
    parse_utc,
    read_sharpcap_sidecar,
    sharpcap_sidecar_path,
)
from tests.recordings.synthetic import FAKE_SERIAL
from tests.recordings.synthetic import SHARPCAP_SIDECAR as REAL_LAYOUT

START_NS = 1_767_268_800_123_456_700  # 2026-01-01T12:00:00.1234567Z


class TestRealLayout:
    def test_every_line_parses_without_a_problem(self) -> None:
        info = parse_sharpcap_sidecar(REAL_LAYOUT)
        assert info.problems == ()
        assert info.dropped == 2  # the serial number and the free-text notes
        assert len(info.entries) == REAL_LAYOUT.count("\n") - 3  # header, serial, notes

    def test_the_typed_accessors(self) -> None:
        info = parse_sharpcap_sidecar(REAL_LAYOUT)
        assert info.camera_model == "ZWO ASI294MM"
        assert info.exposure_us == 10_000
        assert info.gain == 100
        assert info.fps == 100.0
        assert info.sensor_temperature_c == 20.5
        assert info.start_utc_ns == START_NS
        assert info.end_utc_ns == START_NS + 30 * 1_000_000_000
        assert info.duration_s == 30.0
        assert info.frame_count == 3000
        assert info.resolution == (320, 240)
        assert info.binning == 1
        assert info.read_mode == "11 Megapixel"
        assert info.colour_space == "MONO8"
        assert info.utc_offset_s == 10_800
        assert info.readout_mode == "bin2"

    def test_other_values_are_kept_as_written(self) -> None:
        info = parse_sharpcap_sidecar(REAL_LAYOUT)
        assert info.get("Turbo USB") == "72(Auto)"
        assert info.get("frame rate limit") == "Maximum"
        assert info.get("Exposure/Gain Shift") == "0"
        assert info.get("Flip (after dark/flat)") == "None"
        assert info.get("Output Format") == "SER file (*.ser)"
        assert info.get("Trail Width") == "3"
        assert info.get("No such key") is None
        assert info.values["Display MidTone Point"] == "0,5"

    def test_the_serial_number_and_the_notes_do_not_survive(self) -> None:
        info = parse_sharpcap_sidecar(REAL_LAYOUT)
        assert "CameraSerialNumber" not in info.values
        assert "Notes" not in info.values
        assert info.get("CameraSerialNumber") is None
        shown = repr(info) + " ".join(info.problems)
        assert FAKE_SERIAL not in shown
        assert all(FAKE_SERIAL not in e.value and "Serial" not in e.key for e in info.entries)

    def test_the_julian_dates_agree_with_the_utc_stamps(self) -> None:
        info = parse_sharpcap_sidecar(REAL_LAYOUT)
        julian = parse_julian_date(info.get("JDStartCapture") or "")
        assert info.start_utc_ns is not None
        assert abs(julian - info.start_utc_ns) < 100_000_000  # the Julian date has 6 decimals

    def test_windows_line_endings_and_a_byte_order_mark(self) -> None:
        crlf = "\N{ZERO WIDTH NO-BREAK SPACE}" + REAL_LAYOUT.replace("\n", "\r\n")
        info = parse_sharpcap_sidecar(crlf)
        assert info.problems == ()
        assert info.camera_model == "ZWO ASI294MM"
        assert info.exposure_us == 10_000

    def test_the_repr_shows_counts_only(self) -> None:
        shown = repr(parse_sharpcap_sidecar(REAL_LAYOUT))
        assert shown == "<SidecarInfo 53 entries, 2 dropped, 0 problems>"


class TestTolerance:
    def test_the_colon_separator_and_comments(self) -> None:
        text = (
            "# written by hand\n"
            "Exposure: 10,0000ms\n"
            "Gain: 100\n"
            "\n"
            "StartCapture: 2026-01-01T12:00:00.1234567Z\n"
            "; another comment\n"
            "// and another\n"
            "ActualFrameRate = 97,8567fps\n"
        )
        info = parse_sharpcap_sidecar(text)
        assert info.problems == ()
        assert (info.exposure_us, info.gain, info.fps) == (10_000, 100, 97.8567)
        assert info.start_utc_ns == START_NS

    def test_the_equals_sign_wins_when_it_comes_first(self) -> None:
        info = parse_sharpcap_sidecar("StartCapture=2026-01-01T12:00:00.1234567Z\nNote2=a: b\n")
        assert info.values["StartCapture"] == "2026-01-01T12:00:00.1234567Z"
        assert info.values["Note2"] == "a: b"

    def test_lines_that_do_not_parse_become_problems_without_their_text(self) -> None:
        text = "just words without a separator\n=orphan value\nGain=100\n"
        info = parse_sharpcap_sidecar(text)
        assert info.problems == ("line 1: no key and value", "line 2: no key")
        assert info.gain == 100
        assert "words" not in " ".join(info.problems)
        assert "orphan" not in " ".join(info.problems)

    def test_an_empty_or_garbage_file(self) -> None:
        empty = parse_sharpcap_sidecar("")
        assert empty.entries == ()
        assert empty.exposure_us is None
        assert empty.readout_mode is None
        assert empty.camera_model is None
        garbage = parse_sharpcap_sidecar(
            "\x00\x01\x02\n"
            "\N{LATIN SMALL LETTER N WITH TILDE}=\N{LATIN SMALL LETTER O WITH DIAERESIS}\n"
            "[[[\n]]]\n"
        )
        assert isinstance(garbage, SidecarInfo)

    def test_unreadable_values_give_none_and_name_the_setting(self) -> None:
        text = (
            "Exposure=lots\nGain=high\nTemperature=warm\nStartCapture=yesterday\n"
            "ActualFrameRate=fast\nFrameCount=many\nResolution=big\nBinning=none\n"
            "TimeZone=east\nDuration=long\nEndCapture=later\n"
        )
        info = parse_sharpcap_sidecar(text)
        assert info.exposure_us is None
        assert info.gain is None
        assert info.sensor_temperature_c is None
        assert info.start_utc_ns is None
        assert info.fps is None
        assert info.frame_count is None
        assert info.resolution is None
        assert info.binning is None
        assert info.utc_offset_s is None
        assert info.duration_s is None
        assert info.end_utc_ns is None
        assert "the exposure value cannot be read" in info.problems
        assert "the start time value cannot be read" in info.problems
        assert len(info.problems) == 11
        joined = " ".join(info.problems)
        assert "lots" not in joined
        assert "yesterday" not in joined

    def test_missing_keys_are_not_problems(self) -> None:
        info = parse_sharpcap_sidecar("Gain=100\n")
        assert info.problems == ()
        assert info.exposure_us is None

    def test_keys_match_without_regard_to_case_and_spacing(self) -> None:
        info = parse_sharpcap_sidecar(
            "EXPOSURE TIME = 1,5s\nsensor temperature=5.5 C\nGAIN=7\nActual Frame Rate=50fps\n"
        )
        assert info.exposure_us == 1_500_000
        assert info.sensor_temperature_c == 5.5
        assert info.gain == 7
        assert info.fps == 50.0

    def test_similar_keys_do_not_stand_in_for_the_real_ones(self) -> None:
        info = parse_sharpcap_sidecar(
            "Exposure/Gain Shift=3\nAuto Exp Max Gain=285\nAuto Exp Max Exp MS=30000\n"
            "Frame Rate Limit=100\nAuto Exp Target Brightness=100\n"
        )
        assert info.exposure_us is None
        assert info.gain is None
        assert info.fps is None

    def test_the_first_of_a_duplicate_key_wins(self) -> None:
        info = parse_sharpcap_sidecar("[A]\nGain=1\n[B]\nGain=2\n")
        assert info.gain == 1
        assert info.values["Gain"] == "1"
        assert [e.section for e in info.entries] == ["A", "B"]
        assert info.section == "A"

    def test_the_julian_date_stands_in_for_a_missing_utc_stamp(self) -> None:
        info = parse_sharpcap_sidecar(
            "JDStartCapture=2461042,000001\nJDEndCapture=2461042,000349\n"
        )
        assert info.start_utc_ns == 1_767_268_800_086_400_000
        assert info.end_utc_ns == 1_767_268_830_153_600_000

    def test_temperature_units(self) -> None:
        for text, expected in (
            ("18,3", 18.3),
            ("18,3C", 18.3),
            ("18,3 \N{DEGREE SIGN}C", 18.3),
            ("18,3\N{DEGREE CELSIUS}", 18.3),
            ("-5,5 C", -5.5),
            ("32 F", 0.0),
            ("273,15 K", 0.0),
        ):
            info = parse_sharpcap_sidecar(f"Temperature={text}\n")
            assert info.sensor_temperature_c == pytest.approx(expected), text
        assert parse_sharpcap_sidecar("Temperature=5 parsecs\n").sensor_temperature_c is None

    def test_a_time_zone_in_hours_or_hours_and_minutes(self) -> None:
        for text, seconds in (
            ("+3,00", 10_800),
            ("-5,5", -19_800),
            ("+05:30", 19_800),
            ("-08:00", -28_800),
        ):
            assert parse_sharpcap_sidecar(f"TimeZone={text}\n").utc_offset_s == seconds
        assert parse_sharpcap_sidecar("TimeZone=+99\n").utc_offset_s is None

    def test_resolution_and_binning_forms(self) -> None:
        assert parse_sharpcap_sidecar("Resolution=4144 x 2822\n").resolution == (4144, 2822)
        assert parse_sharpcap_sidecar(
            "Capture Area=640\N{MULTIPLICATION SIGN}480\n"
        ).resolution == (640, 480)
        assert parse_sharpcap_sidecar("Binning=2x2\n").binning == 2
        assert parse_sharpcap_sidecar("Binning=Bin 3\n").binning == 3
        assert parse_sharpcap_sidecar("Binning=0\n").binning is None

    @given(st.text())
    def test_it_never_raises(self, text: str) -> None:
        info = parse_sharpcap_sidecar(text)
        accessors = (
            info.exposure_us,
            info.gain,
            info.fps,
            info.sensor_temperature_c,
            info.start_utc_ns,
            info.end_utc_ns,
            info.duration_s,
            info.frame_count,
            info.resolution,
            info.binning,
            info.read_mode,
            info.colour_space,
            info.utc_offset_s,
            info.readout_mode,
            info.camera_model,
        )
        assert len(accessors) == 15
        assert repr(info).startswith("<SidecarInfo ")


class TestPrivacy:
    @pytest.mark.parametrize(
        "key",
        [
            "CameraSerialNumber",
            "Serial Number",
            "Serial",
            "SerialNo",
            "serial_number",
            "S/N",
            "S.N.",
            "SN",
            "Camera SN",
            "CameraSN",
            "GUID",
            "CameraGuid",
            "UUID",
            "DeviceID",
            "Device Id",
            "ID",
            "Camera ID",
            "UniqueId",
            "USBID",
            "Notes",
            "Comment",
            "Output File",
            "Folder",
            "Observer",
            "UserName",
            "Latitude",
            "Site Location",
        ],
    )
    def test_private_keys_are_dropped(self, key: str) -> None:
        assert is_private_key(key)
        info = parse_sharpcap_sidecar(f"{key}=SECRETVALUE\nGain=100\n")
        assert info.dropped == 1
        assert [e.key for e in info.entries] == ["Gain"]
        assert "SECRETVALUE" not in repr(info)

    @pytest.mark.parametrize(
        "key",
        [
            "Gain",
            "Exposure",
            "Binning",
            "Valid Range",
            "Width",
            "Solid Fill",
            "Hybrid",
            "Idle Time",
            "Snapshot",
            "Transparency",
            "Pan",
            "Display Black Point",
            "Auto Exp Max Exp MS",
            "Planet/Disk Stabilization",
            "Flip (after dark/flat)",
            "DisplayStretchEnable",
            "SharpCapVersion",
            "JDStartCapture",
            "Turbo USB",
        ],
    )
    def test_ordinary_keys_survive(self, key: str) -> None:
        assert not is_private_key(key)

    def test_private_looking_values_are_dropped_under_any_key(self) -> None:
        text = (
            "Camera=00000000-0000-0000-0000-000000000000\n"
            "Device=0000000000AB\n"
            "Origin=ZWO camera, serial 123\n"
            "Source=" + "C" + ":\\" + "captures\\one.ser\n"
            "Target=" + "/ho" + "me/someone/one.ser\n"
            "Gain=100\n"
        )
        info = parse_sharpcap_sidecar(text)
        assert info.dropped == 5
        assert [e.key for e in info.entries] == ["Gain"]

    def test_ordinary_values_survive(self) -> None:
        for value in ("11 Megapixel", "72(Auto)", "2026-01-01T12:00:00.1234567Z", "4.1.99999.0"):
            assert not is_private_value(value)

    def test_a_private_section_header_is_not_kept(self) -> None:
        info = parse_sharpcap_sidecar("[Camera serial SECRETVALUE]\nGain=1\n")
        assert info.camera_model is None
        assert info.entries[0].section == ""

    def test_the_serial_is_absent_from_everything_the_object_exposes(self) -> None:
        text = "CameraSerialNumber=SECRETVALUE\nBadLine SECRETVALUE\nExposure=SECRETVALUE\n"
        info = parse_sharpcap_sidecar(text)
        assert "SECRETVALUE" not in repr(info) + " ".join(info.problems)
        assert not any(entry.key == "CameraSerialNumber" for entry in info.entries)
        assert info.problems == ("line 2: no key and value", "the exposure value cannot be read")


class TestReading:
    def test_the_path_of_the_sidecar(self, tmp_path: Path) -> None:
        assert (
            sharpcap_sidecar_path(tmp_path / "capture_01.ser")
            == tmp_path / "capture_01.CameraSettings.txt"
        )
        assert sharpcap_sidecar_path(tmp_path / "Capture.SER").name == "Capture.CameraSettings.txt"
        assert sharpcap_sidecar_path(str(tmp_path / "a.b.ser")).name == "a.b.CameraSettings.txt"

    def test_utf_8_with_and_without_a_byte_order_mark(self, tmp_path: Path) -> None:
        for name, payload in (
            ("a", REAL_LAYOUT.encode()),
            ("b", codecs.BOM_UTF8 + REAL_LAYOUT.encode()),
        ):
            path = tmp_path / f"{name}.CameraSettings.txt"
            path.write_bytes(payload)
            assert read_sharpcap_sidecar(path).exposure_us == 10_000

    def test_utf_16_with_a_byte_order_mark(self) -> None:
        raw = codecs.BOM_UTF16_LE + REAL_LAYOUT.encode("utf-16-le")
        assert parse_sharpcap_sidecar(decode_sidecar_bytes(raw)).exposure_us == 10_000

    def test_windows_1252_text(self) -> None:
        text = "Temperature=18,3 \N{DEGREE SIGN}C\nNote2=caf\N{LATIN SMALL LETTER E WITH ACUTE}\n"
        raw = text.encode("cp1252")
        info = parse_sharpcap_sidecar(decode_sidecar_bytes(raw))
        assert info.sensor_temperature_c == 18.3
        assert info.values["Note2"] == "caf\N{LATIN SMALL LETTER E WITH ACUTE}"

    def test_a_missing_file_raises_a_sidecar_error_that_omits_the_path(
        self, tmp_path: Path
    ) -> None:
        folder = tmp_path / "distinctive-folder-name"
        folder.mkdir()
        with pytest.raises(SidecarError, match="cannot read the sidecar") as raised:
            read_sharpcap_sidecar(folder / "missing.CameraSettings.txt")
        assert "distinctive-folder-name" not in str(raised.value)
        assert raised.value.__cause__ is None


class TestReadoutMode:
    @pytest.mark.parametrize(
        ("read_mode", "binning", "expected"),
        [
            ("11 Megapixel", "1", "bin2"),
            ("11MP", "1", "bin2"),
            ("11,7 Megapixels", "1", "bin2"),
            ("11 Megapixel", None, "bin2"),  # a missing binning counts as 1
            ("11 Megapixel", "2", None),
            ("47 Megapixel", "1", "bin1"),
            ("46,8 Megapixel", "1", "bin1"),
            ("47 Megapixel", "2", "bin2"),
            ("47 Megapixel", "3", None),
            ("Fast", "1", None),
            ("11 Megapixel", "many", None),
            ("1 Megapixel", "1", None),
        ],
    )
    def test_the_heuristic(self, read_mode: str, binning: str | None, expected: str | None) -> None:
        text = f"Read Mode={read_mode}\n" + ("" if binning is None else f"Binning={binning}\n")
        assert parse_sharpcap_sidecar(text).readout_mode == expected

    def test_no_read_mode_gives_no_answer(self) -> None:
        assert parse_sharpcap_sidecar("Binning=1\n").readout_mode is None


class TestValueHelpers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("97,8567", 97.8567),
            ("97.8567", 97.8567),
            ("1 234,5", 1234.5),
            ("1\N{NO-BREAK SPACE}234,5", 1234.5),
            ("1.234,5", 1234.5),
            ("1,234.5", 1234.5),
            ("-3,5", -3.5),
            ("+3,00", 3.0),
            ("1e3", 1000.0),
            (",5", 0.5),
            ("  7 ", 7.0),
            ("0", 0.0),
        ],
    )
    def test_decimal_comma(self, text: str, expected: float) -> None:
        assert parse_decimal_comma(text) == expected

    @pytest.mark.parametrize(
        "text", ["", "abc", "97,8567fps", "nan", "inf", "1,2,3", "--1", "1.2.3", "1 23"]
    )
    def test_decimal_comma_rejects(self, text: str) -> None:
        with pytest.raises(ValueError, match="not a number"):
            parse_decimal_comma(text)

    @given(st.floats(min_value=0, max_value=1e9, allow_nan=False))
    def test_decimal_comma_reads_what_python_prints(self, value: float) -> None:
        assert parse_decimal_comma(repr(value).replace(".", ",")) == value

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("10,0000ms", 10.0),
            ("10 ms", 10.0),
            ("1,5s", 1500.0),
            ("250us", 0.25),
            ("250 \N{MICRO SIGN}s", 0.25),
            ("250 \N{GREEK SMALL LETTER MU}s", 0.25),
            ("2min", 120_000.0),
            ("10", 10.0),
            ("0,5 MS", 0.5),
        ],
    )
    def test_duration(self, text: str, expected: float) -> None:
        assert parse_duration_ms(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["ms", "10 parsecs", "-1ms", "", "ten ms"])
    def test_duration_rejects(self, text: str) -> None:
        with pytest.raises(ValueError, match=r"duration|number"):
            parse_duration_ms(text)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2026-01-01T12:00:00.1234567Z", START_NS),
            ("2026-01-01 12:00:00Z", 1_767_268_800_000_000_000),
            ("2026-01-01T12:00:00,5Z", 1_767_268_800_500_000_000),
            ("2026-01-01T12:00:00.123456789012Z", 1_767_268_800_123_456_789),
            ("2026-01-01T12:00:00+00:00", 1_767_268_800_000_000_000),
            ("1970-01-01T00:00:00Z", 0),
            ("1969-12-31T23:59:59.9999999Z", -100),
            ("  2026-01-01T12:00:00z ", 1_767_268_800_000_000_000),
        ],
    )
    def test_utc(self, text: str, expected: int) -> None:
        assert parse_utc(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "2026-01-01T12:00:00",  # no Z: it could be local time
            "2026-01-01T12:00:00+03:00",
            "2026-13-01T12:00:00Z",
            "2026-02-30T12:00:00Z",
            "2026-01-01T12:00:60Z",
            "12:00:00Z",
            "",
        ],
    )
    def test_utc_rejects(self, text: str) -> None:
        with pytest.raises(ValueError, match="UTC"):
            parse_utc(text)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2440587,5", 0),
            ("2440587.5", 0),
            ("2461042,0", 1_767_268_800_000_000_000),
            ("2461042,000001", 1_767_268_800_086_400_000),
            ("2460000.5", 1_677_283_200_000_000_000),
        ],
    )
    def test_julian_date(self, text: str, expected: int) -> None:
        assert parse_julian_date(text) == expected

    @pytest.mark.parametrize("text", ["", "abc", "12", "1e9", "99999999"])
    def test_julian_date_rejects(self, text: str) -> None:
        with pytest.raises(ValueError, match="Julian"):
            parse_julian_date(text)

    def test_helper_errors_do_not_quote_the_input(self) -> None:
        for helper in (parse_decimal_comma, parse_duration_ms, parse_utc, parse_julian_date):
            with pytest.raises(ValueError) as raised:  # noqa: PT011
                helper("SECRETVALUE")
            assert "SECRETVALUE" not in str(raised.value)
