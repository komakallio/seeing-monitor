"""The `[web]` and `[auth]` configuration sections."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from seeingmon.config import ConfigError, load_config
from seeingmon.services.web import config as web_config
from seeingmon.services.web.config import AuthSettings, WebSettings

HASH = "$scrypt$ln=10,r=8,p=1$c2FsdHNhbHRzYWx0$ZGlnZXN0ZGlnZXN0ZGlnZXN0"
MDNS_NAME = "my-pi.local"  # repo-check: allow
MULTICAST_V6 = "ff02::1"  # repo-check: allow


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


@pytest.mark.parametrize(
    "address", ["192.0.2.7", "::1", "2001:db8::5", "localhost", " 192.0.2.7 ", "0.0.0.0", "::"]
)
def test_a_bind_address_names_an_interface_or_a_wildcard(address: str) -> None:
    assert WebSettings(bind_address=address).bind_address in {
        "192.0.2.7",
        "::1",
        "2001:db8::5",
        "localhost",
        "0.0.0.0",
        "::",
    }


@pytest.mark.parametrize("address", ["", "not-an-address", "192.0.2", "host.example", "*"])
def test_an_unparsable_bind_address_is_an_error(address: str) -> None:
    with pytest.raises(ValueError, match="bind_address"):
        WebSettings(bind_address=address)


def test_the_default_needs_no_host_list_and_listens_on_one_address() -> None:
    settings = WebSettings()
    assert settings.allowed_hosts == ()
    assert settings.extra_bind_addresses == ()
    assert settings.listen_addresses() == ("127.0.0.1",)
    assert {"localhost", "127.0.0.1", "::1"} <= settings.allowed_host_set()


def test_the_allowed_hosts_are_stored_in_their_canonical_form() -> None:
    settings = WebSettings(
        allowed_hosts=("Pi.Example.", "2001:DB8:0::5", "pi.example", "192.0.2.5")
    )
    assert settings.allowed_hosts == ("pi.example", "2001:db8::5", "192.0.2.5")


@pytest.mark.parametrize("entry", ["*.example", "*", "pi.example:8080", "0.0.0.0", "::", "", "a b"])
def test_a_bad_allowed_host_is_an_error(entry: str) -> None:
    with pytest.raises(ValueError, match="allowed_hosts"):
        WebSettings(allowed_hosts=(entry,))


def test_an_allowed_host_error_in_the_layers_names_the_key_and_not_the_value(
    tmp_path: Path,
) -> None:
    local = tmp_path / "local.toml"
    local.write_text('[web]\nallowed_hosts = ["*.private-name.example"]\n', encoding="utf-8")
    with pytest.raises(ConfigError) as raised:
        load_config(local_file=local, env={}).section("web", WebSettings)
    assert "allowed_hosts" in str(raised.value)
    assert "private-name" not in str(raised.value)


def test_the_allowed_host_set_adds_the_list_and_every_bind_address() -> None:
    settings = WebSettings(
        bind_address="192.0.2.5",
        extra_bind_addresses=("2001:db8::5",),
        allowed_hosts=("pi.example",),
    )
    assert settings.allowed_host_set() == frozenset(
        {"localhost", "127.0.0.1", "::1", "192.0.2.5", "2001:db8::5", "pi.example"}
    )


@pytest.mark.parametrize(
    "address", ["192.0.2.7", "::1", "2001:db8::5", "localhost", " 192.0.2.7 ", "0.0.0.0", "::"]
)
def test_an_extra_bind_address_reads_like_the_bind_address(address: str) -> None:
    settings = WebSettings(extra_bind_addresses=(address,))
    assert settings.extra_bind_addresses in {
        ("192.0.2.7",),
        ("::1",),
        ("2001:db8::5",),
        ("localhost",),
        ("0.0.0.0",),
        ("::",),
    }


@pytest.mark.parametrize("address", ["", "not-an-address", "192.0.2", "host.example", "*"])
def test_an_unparsable_extra_bind_address_is_an_error(address: str) -> None:
    with pytest.raises(ValueError, match="extra_bind_addresses"):
        WebSettings(extra_bind_addresses=("192.0.2.7", address))


def test_the_listen_addresses_are_distinct_and_keep_their_order() -> None:
    settings = WebSettings(
        bind_address="192.0.2.5",
        extra_bind_addresses=("2001:db8::5", "192.0.2.5", "localhost", "127.0.0.1", "::1"),
    )
    assert settings.listen_addresses() == ("192.0.2.5", "2001:db8::5", "127.0.0.1", "::1")


def test_localhost_listens_on_the_ipv4_loopback_address() -> None:
    assert WebSettings(bind_address="localhost").listen_addresses() == ("127.0.0.1",)


def test_a_wildcard_listens_on_its_own() -> None:
    assert WebSettings(bind_address="0.0.0.0").listen_addresses() == ("0.0.0.0",)
    assert WebSettings(bind_address="::").listen_addresses() == ("::",)


def test_a_wildcard_leaves_out_the_addresses_of_its_family() -> None:
    """A wildcard socket and a socket of its family cannot share one port."""
    both = WebSettings(
        bind_address="0.0.0.0",
        extra_bind_addresses=("127.0.0.1", "192.0.2.5", "::1", "2001:db8::5", "localhost"),
    )
    assert both.listen_addresses() == ("0.0.0.0", "::1", "2001:db8::5")
    everything = WebSettings(
        bind_address="192.0.2.5", extra_bind_addresses=("::", "0.0.0.0", "2001:db8::5", "::1")
    )
    assert everything.listen_addresses() == ("::", "0.0.0.0")


def test_a_wildcard_bind_admits_every_address_and_the_names_of_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_config, "device_names", lambda: ("my-pi", MDNS_NAME))
    rule = WebSettings(bind_address="0.0.0.0", allowed_hosts=("pi.example",)).allowed_host_set()
    assert rule.any_address is True
    for admitted in (
        "localhost",
        "127.0.0.1",
        "::1",
        "pi.example",
        "my-pi",
        MDNS_NAME,
        "192.0.2.77",
        "2001:db8::77",
        "203.0.113.9",
    ):
        assert admitted in rule
    for refused in ("evil.example", "0.0.0.0", "::", "224.0.0.1", MULTICAST_V6):
        assert refused not in rule


def test_an_address_bind_admits_that_address_and_not_every_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_config, "device_names", lambda: ("my-pi",))
    rule = WebSettings(bind_address="192.0.2.5").allowed_host_set()
    assert rule.any_address is False
    assert "192.0.2.5" in rule
    assert "192.0.2.77" not in rule
    assert "my-pi" not in rule


def test_the_new_keys_read_from_the_layers(tmp_path: Path) -> None:
    local = tmp_path / "local.toml"
    local.write_text(
        '[web]\nbind_address = "192.0.2.9"\nextra_bind_addresses = ["2001:db8::9"]\n'
        'allowed_hosts = ["pi.example", "192.0.2.9"]\n',
        encoding="utf-8",
    )
    web = load_config(local_file=local, env={}).section("web", WebSettings)
    assert web.listen_addresses() == ("192.0.2.9", "2001:db8::9")
    assert web.allowed_hosts == ("pi.example", "192.0.2.9")


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
    local.write_text('[web]\nbind_address = "private-host.example"\n', encoding="utf-8")
    config = load_config(local_file=local, env={})
    with pytest.raises(ConfigError) as raised:
        config.section("web", WebSettings)
    assert "bind_address" in str(raised.value)
    assert "private-host" not in str(raised.value)


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


def test_the_video_of_polaris_has_its_own_cadence_and_shares_the_other_live_settings() -> None:
    live = WebSettings().live
    assert live.polaris_max_fps == 20.0
    assert live.max_fps == 2.0  # the alignment preview keeps its own cadence
    assert (live.max_clients, live.stall_s, live.idle_s) == (4, 5.0, 10.0)


@pytest.mark.parametrize("value", [0, -1.0, 60.5])
def test_the_cadence_of_the_video_has_bounds(value: float) -> None:
    with pytest.raises(ValueError, match="polaris_max_fps"):
        WebSettings.model_validate({"live": {"polaris_max_fps": value}})


def test_a_local_file_sets_the_cadence_of_the_video(tmp_path: Path) -> None:
    local = tmp_path / "local.toml"
    local.write_text("[web.live]\npolaris_max_fps = 12.5\n", encoding="utf-8")
    web = load_config(local_file=local, env={}).section("web", WebSettings)
    assert web.live.polaris_max_fps == 12.5
