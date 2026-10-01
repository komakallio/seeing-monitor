"""The `[web]` and `[auth]` configuration sections."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from seeingmon.config import ConfigError, load_config
from seeingmon.services.web.config import AuthSettings, WebSettings

HASH = "$scrypt$ln=10,r=8,p=1$c2FsdHNhbHRzYWx0$ZGlnZXN0ZGlnZXN0ZGlnZXN0"


def read_defaults(repo_root: Path) -> dict[str, object]:
    with (repo_root / "config" / "default.d" / "web.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_the_default_file_and_the_model_agree(repo_root: Path) -> None:
    """The model declares the defaults of the TOML file, so a change in one shows in the other."""
    table = read_defaults(repo_root)["web"]
    assert WebSettings.model_validate(table) == WebSettings()


def test_the_default_file_keeps_every_key_under_the_web_table(repo_root: Path) -> None:
    assert set(read_defaults(repo_root)) == {"web"}


def test_the_default_bind_address_is_the_loopback_interface() -> None:
    """A fresh install must not listen on the LAN until the owner sets an address."""
    assert WebSettings().bind_address == "127.0.0.1"


def test_reads_are_open_by_default() -> None:
    assert WebSettings().require_token_for_reads is False


def test_the_zenith_angle_is_withheld_by_default() -> None:
    assert WebSettings().withhold_fields == ("zenith_angle_deg",)


@pytest.mark.parametrize("address", ["192.0.2.7", "::1", "2001:db8::5", "localhost", " 192.0.2.7 "])
def test_a_bind_address_names_one_interface(address: str) -> None:
    assert WebSettings(bind_address=address).bind_address in {
        "192.0.2.7",
        "::1",
        "2001:db8::5",
        "localhost",
    }


@pytest.mark.parametrize(
    "address", ["0.0.0.0", "::", "", "not-an-address", "192.0.2", "host.example"]
)
def test_a_wildcard_or_unparsable_bind_address_is_an_error(address: str) -> None:
    with pytest.raises(ValueError, match="bind_address"):
        WebSettings(bind_address=address)


def test_an_unknown_key_is_an_error() -> None:
    with pytest.raises(ValueError, match="bind_adress"):
        WebSettings.model_validate({"bind_adress": "127.0.0.1"})


def test_a_default_page_larger_than_the_maximum_is_an_error() -> None:
    with pytest.raises(ValueError, match="default_limit"):
        WebSettings.model_validate({"paging": {"default_limit": 600, "max_limit": 500}})


def test_a_default_image_list_larger_than_the_maximum_is_an_error() -> None:
    with pytest.raises(ValueError, match="default_list_limit"):
        WebSettings.model_validate({"images": {"default_list_limit": 50, "max_list_limit": 10}})


def test_withheld_fields_must_look_like_field_names() -> None:
    with pytest.raises(ValueError, match="withhold_fields"):
        WebSettings(withhold_fields=("Not A Field",))


def test_the_section_reads_from_the_layers(tmp_path: Path) -> None:
    local = tmp_path / "local.toml"
    local.write_text('[web]\nbind_address = "192.0.2.9"\nport = 9000\n', encoding="utf-8")
    config = load_config(local_file=local, env={"SEEINGMON_WEB__RATE_LIMIT__WINDOW_S": "30"})
    web = config.section("web", WebSettings)
    assert web.bind_address == "192.0.2.9"
    assert web.port == 9000
    assert web.rate_limit.window_s == 30.0
    assert web.core.rpc_timeout_s == 5.0


def test_a_bad_section_names_the_key_and_not_the_value(tmp_path: Path) -> None:
    local = tmp_path / "local.toml"
    local.write_text('[web]\nbind_address = "0.0.0.0"\n', encoding="utf-8")
    config = load_config(local_file=local, env={})
    with pytest.raises(ConfigError) as raised:
        config.section("web", WebSettings)
    assert "bind_address" in str(raised.value)


# --- The token hash --------------------------------------------------------------------------


def test_no_token_hash_is_found_without_a_source() -> None:
    assert AuthSettings().load_token_hash(env={}) is None


def test_the_hash_comes_from_the_configured_value() -> None:
    assert AuthSettings(token_hash=HASH).load_token_hash(env={}) == HASH


def test_the_hash_comes_from_a_systemd_credential(tmp_path: Path) -> None:
    (tmp_path / "seeingmon-token-hash").write_text(f"{HASH}\n", encoding="utf-8")
    found = AuthSettings().load_token_hash(env={"CREDENTIALS_DIRECTORY": str(tmp_path)})
    assert found == HASH


def test_the_hash_comes_from_a_named_file(tmp_path: Path) -> None:
    path = tmp_path / "hash.txt"
    path.write_text(f"  {HASH}  \n", encoding="utf-8")
    assert AuthSettings(token_hash_file=str(path)).load_token_hash(env={}) == HASH


def test_the_configured_value_wins_over_the_credential_and_the_file(tmp_path: Path) -> None:
    (tmp_path / "seeingmon-token-hash").write_text("from-credential", encoding="utf-8")
    other = tmp_path / "file.txt"
    other.write_text("from-file", encoding="utf-8")
    settings = AuthSettings(token_hash="from-value", token_hash_file=str(other))
    assert settings.load_token_hash(env={"CREDENTIALS_DIRECTORY": str(tmp_path)}) == "from-value"


def test_the_credential_wins_over_the_file(tmp_path: Path) -> None:
    (tmp_path / "seeingmon-token-hash").write_text("from-credential", encoding="utf-8")
    other = tmp_path / "file.txt"
    other.write_text("from-file", encoding="utf-8")
    settings = AuthSettings(token_hash_file=str(other))
    assert settings.load_token_hash(env={"CREDENTIALS_DIRECTORY": str(tmp_path)}) == (
        "from-credential"
    )


def test_a_missing_token_hash_file_is_an_error_without_a_path_in_the_message(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "private-name.txt"
    with pytest.raises(ConfigError) as raised:
        AuthSettings(token_hash_file=str(missing)).load_token_hash(env={})
    assert "private-name" not in str(raised.value)
    assert "token hash file" in str(raised.value)


def test_an_empty_token_hash_file_means_no_hash(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("\n", encoding="utf-8")
    assert AuthSettings(token_hash_file=str(path)).load_token_hash(env={}) is None


def test_a_huge_token_hash_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "huge.txt"
    path.write_text("x" * 10_000, encoding="utf-8")
    with pytest.raises(ConfigError, match="too large"):
        AuthSettings(token_hash_file=str(path)).load_token_hash(env={})


def test_the_hash_stays_out_of_repr_and_the_effective_configuration(tmp_path: Path) -> None:
    settings = AuthSettings(token_hash=HASH)
    assert HASH not in repr(settings)
    local = tmp_path / "local.toml"
    local.write_text(f'[auth]\ntoken_hash = "{HASH}"\n', encoding="utf-8")
    effective = load_config(local_file=local, env={}).effective()
    assert HASH not in str(effective)
    assert effective["auth"] == {"token_hash": "<redacted>"}
