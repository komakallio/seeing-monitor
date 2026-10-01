"""The configuration files in the repository follow the rules of the layers."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from seeingmon import paths
from seeingmon.config import Config, load_config

TOP_LEVEL_KEYS = {"profile", "station_id"}  # the keys that belong to no lane


def read(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def leaf_paths(table: dict[str, Any], prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """The key paths of every non-table value, such as `("scheduler", "window_s")`."""
    found = []
    for key, value in table.items():
        if isinstance(value, dict):
            found += leaf_paths(value, (*prefix, key))
        else:
            found.append((*prefix, key))
    return found


@pytest.fixture
def config_dir(repo_root: Path) -> Path:
    return repo_root / "config"


def test_the_packaged_config_directory_is_the_repository_one(config_dir: Path) -> None:
    assert paths.config_dir() == config_dir


def test_the_default_file_holds_only_the_keys_that_belong_to_no_lane(config_dir: Path) -> None:
    """A lane adds its defaults in `config/default.d/<lane>.toml`, never in `default.toml`."""
    assert set(read(config_dir / "default.toml")) == TOP_LEVEL_KEYS


def test_the_default_profile_is_the_reference_profile(config_dir: Path) -> None:
    assert read(config_dir / "default.toml")["profile"] == "asi294mm-gs250"


def test_no_two_default_files_define_the_same_key(config_dir: Path) -> None:
    """Each lane owns its file, so a key that appears twice means two lanes collide."""
    files = [config_dir / "default.toml", *sorted((config_dir / "default.d").glob("*.toml"))]
    owners: dict[tuple[str, ...], str] = {}
    for path in files:
        for key in leaf_paths(read(path)):
            assert key not in owners, (
                f"{'.'.join(key)} is defined in {owners[key]} and in {path.name}; "
                "each lane keeps its own keys"
            )
            owners[key] = path.name


def test_every_default_file_parses(config_dir: Path) -> None:
    for path in (config_dir / "default.d").glob("*.toml"):
        read(path)


# --- The template for the local file ------------------------------------------------------------


@pytest.fixture
def template(config_dir: Path) -> dict[str, Any]:
    return read(config_dir / "local.example.toml")


def test_the_template_has_the_documented_keys(template: dict[str, Any]) -> None:
    assert set(template) == {"station_id", "profile", "site", "paths", "replay", "auth"}
    assert set(template["site"]) == {"latitude_deg", "longitude_deg", "elevation_m"}
    assert set(template["paths"]) == {"data_dir"}
    assert set(template["replay"]) == {"recordings_dir"}
    assert set(template["auth"]) == {"token_hash"}


def test_the_template_holds_no_real_site_location(template: dict[str, Any]) -> None:
    """The placeholders are 0.0, which is no site of ours. A real coordinate never goes here."""
    assert template["site"] == {"latitude_deg": 0.0, "longitude_deg": 0.0, "elevation_m": 0.0}


def test_the_template_values_are_placeholders(template: dict[str, Any]) -> None:
    for section, key in [
        ("paths", "data_dir"),
        ("replay", "recordings_dir"),
        ("auth", "token_hash"),
    ]:
        value = template[section][key]
        assert value.startswith("<"), f"{section}.{key} is not a placeholder"
        assert value.endswith(">"), f"{section}.{key} is not a placeholder"


def test_the_template_is_a_valid_local_layer(config_dir: Path) -> None:
    config = load_config(
        config_dir=config_dir, local_file=config_dir / "local.example.toml", env={}
    )
    assert config.station_id == "my-station"
    assert config.profile.id == "asi294mm-gs250"
    assert config.effective()["auth"] == {"token_hash": "<redacted>"}


def test_the_template_names_the_documented_environment_scheme(config_dir: Path) -> None:
    text = (config_dir / "local.example.toml").read_text(encoding="utf-8")
    assert "SEEINGMON_<SECTION>__<KEY>" in text


def test_the_loaded_defaults_are_a_config(config_dir: Path, tmp_path: Path) -> None:
    config = load_config(local_file=tmp_path / "none.toml", env={})
    assert isinstance(config, Config)
    assert config.station_id == "unset"
