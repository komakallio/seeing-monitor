"""The second layer that keeps private values out of the API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from seeingmon.config import load_config
from seeingmon.services.web.privacy import (
    HIDDEN,
    PUBLIC_SECTIONS,
    public_config,
    scrub_json,
    scrub_text,
    withhold_sun_elevation,
)

WINDOWS_PATH = "C:\\Users\\someone\\data\\db.sqlite"  # repo-check: allow
UNC_PATH = "\\\\fileserver\\share\\folder"  # repo-check: allow
HOME_NOTE = "see /home/someone/x/y"  # repo-check: allow


@pytest.mark.parametrize(
    ("text", "kept", "gone"),
    [
        (f"Cannot write {WINDOWS_PATH} now", "Cannot write ", "someone"),
        (f"Cannot read {UNC_PATH}", "Cannot read ", "fileserver"),
        ("Opened /home/someone/data/x.sqlite", "Opened ", "someone"),  # repo-check: allow
        ("Opened /var/lib/seeingmon/db", "Opened ", "seeingmon"),
        ("Opened ~/data/x", "Opened ", "data"),
        ("Posted to https://plug.example.org/api/cycle?key=abc", "Posted to ", "plug.example"),
        ("Posted to http://192.0.2.9:8080/x", "Posted to ", "192.0.2.9"),
        ("Connected to 192.0.2.7.", "Connected to ", "192.0.2.7"),
        ("Connected to 192.0.2.7, then 198.51.100.4", "Connected to ", "198.51.100.4"),
        ("The path is /run/user/1000/seeingmon/core.sock", "The path is ", "core.sock"),
    ],
)
def test_text_loses_its_paths_urls_and_addresses(text: str, kept: str, gone: str) -> None:
    cleaned = scrub_text(text)
    assert HIDDEN in cleaned
    assert cleaned.startswith(kept)
    assert gone not in cleaned


@pytest.mark.parametrize(
    "text",
    [
        "The scheduler entered the auto state.",
        "Window of 60 s, 5400 frames, 12/13 usable.",
        "Ratio 3/4 and 10/12/2026",
        "n/a",
        "Exposure 2 ms, gain 0, mode bin1.",
        "Version 1.2.3 of the catalog",
        "Offset 0.5 arcmin",
        "The star moved by 1.5 px / 2 s",
    ],
)
def test_ordinary_text_stays_as_it_is(text: str) -> None:
    assert scrub_text(text) == text


def test_scrub_json_cleans_every_string_and_leaves_keys_and_numbers() -> None:
    value = {
        "path /var/lib/x": ["Opened /var/lib/seeingmon/db", 5, None, True],
        "nested": {"url": "https://example.org/a", "count": 2.5},
    }
    cleaned = scrub_json(value)
    assert cleaned == {
        "path /var/lib/x": [f"Opened {HIDDEN}", 5, None, True],
        "nested": {"url": HIDDEN, "count": 2.5},
    }
    assert value["nested"]["url"] == "https://example.org/a"  # type: ignore[index]


def test_scrub_json_stops_at_a_depth_limit() -> None:
    deep: Any = "end"
    for _ in range(50):
        deep = [deep]
    assert HIDDEN in str(scrub_json(deep))


def test_every_value_that_gives_the_suns_elevation_is_withheld() -> None:
    value = {
        "sun_elevation_deg": -18.73,
        "nested": {"dark_sun_deg": -19.1, "sun_altitude_deg": -5.0, "frames": 5},
        "list": [{"first_visible_sun_deg": 4.2}],
        "sky_mag_arcsec2": 20.5,
        "sun_elevation_limit_deg": -3.0,  # a setting, not a measurement at a time
        "sunset_count": 1,
    }
    assert withhold_sun_elevation(value) == {
        "sun_elevation_deg": None,
        "nested": {"dark_sun_deg": None, "sun_altitude_deg": None, "frames": 5},
        "list": [{"first_visible_sun_deg": None}],
        "sky_mag_arcsec2": 20.5,
        "sun_elevation_limit_deg": -3.0,
        "sunset_count": 1,
    }
    assert value["sun_elevation_deg"] == -18.73  # the input stays as it was
    assert withhold_sun_elevation(None) is None


EFFECTIVE: dict[str, Any] = {
    "profile": "test-profile",
    "station_id": "test-station",
    "scheduler": {"fast": {"window_s": 120.0}, "survey": {"cadence_s": 180.0}},
    "fastpath": {"window_s": 60.0},
    "survey": {"catalog_path": "<redacted>", "solvers": ["astrometry.net"], "dut1_s": 0.0},
    "store": {"retention": {"previews_days": 7.0}},
    "services": {
        "acquire": {"driver": "asi", "driver_options": {"serial": "SN123456", "gain": 1}},
        "connect_timeout_s": 5.0,
    },
    "web": {"port": 8080, "rate_limit": {"window_s": 60.0}},
    "paths": {"data_dir": "<redacted>"},
    "replay": {"recordings_dir": "<redacted>"},
    "auth": {"token_hash": "<redacted>"},
    "sinks": {"lab": {"kind": "influx", "org": "my-org", "bucket": "b", "endpoint": "<redacted>"}},
    "power": {"route": "http", "http": {"url": "<redacted>"}},
    "heater": {"enabled": True, "pins": {"heater": {"chip": "gpiochip0", "line": 4}}},
    "sqm": {"enabled": True, "host": "<redacted>"},
    "site": {"latitude_deg": 12.3456, "longitude_deg": 65.4321, "elevation_m": 77.0},
}


def test_the_public_configuration_keeps_only_the_public_sections() -> None:
    public = public_config(EFFECTIVE)
    assert set(public) == set(PUBLIC_SECTIONS) & set(EFFECTIVE)
    for hidden in ("paths", "replay", "auth", "sinks", "power", "heater", "sqm", "site"):
        assert hidden not in public


def test_the_public_configuration_keeps_the_values_that_describe_the_computation() -> None:
    public = public_config(EFFECTIVE)
    assert public["scheduler"]["fast"]["window_s"] == 120.0
    assert public["fastpath"] == {"window_s": 60.0}
    assert public["store"]["retention"]["previews_days"] == 7.0
    assert public["services"]["connect_timeout_s"] == 5.0
    assert public["profile"] == "test-profile"


def test_the_public_configuration_drops_the_options_of_the_driver_where_a_serial_may_sit() -> None:
    public = public_config(EFFECTIVE)
    assert public["services"]["acquire"] == {"driver": "asi"}
    assert "SN123456" not in str(public)


def test_the_public_configuration_hides_a_value_that_looks_like_a_path_or_an_address() -> None:
    public = public_config({"scheduler": {"note": HOME_NOTE, "peer": "http://192.0.2.1/a", "n": 1}})
    assert public["scheduler"] == {"note": f"see {HIDDEN}", "peer": HIDDEN, "n": 1}


def test_the_public_configuration_does_not_change_its_input() -> None:
    before = repr(EFFECTIVE)
    public_config(EFFECTIVE)
    assert repr(EFFECTIVE) == before


def test_a_real_effective_configuration_has_no_private_value_in_its_public_form(
    tmp_path: Path,
) -> None:
    local = tmp_path / "local.toml"
    local.write_text(
        "[site]\nlatitude_deg = 12.3456\nlongitude_deg = 65.4321\nelevation_m = 77.0\n"
        '[paths]\ndata_dir = "a-private-folder"\n'
        '[auth]\ntoken_hash = "a-private-hash"\n'
        '[sqm]\nenabled = true\nhost = "sqm.example.org"\n'
        '[services]\nconnection_key = "a-private-connection-key-value"\n',
        encoding="utf-8",
    )
    config = load_config(local_file=local, env={})
    text = str(public_config(config.effective(redact=True, omit_site=True)))
    for private in (
        "12.3456",
        "65.4321",
        "77.0",
        "a-private-folder",
        "a-private-hash",
        "sqm.example.org",
        "a-private-connection-key-value",
    ):
        assert private not in text
