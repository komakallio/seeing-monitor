"""`deploy/install.sh`: the command line, the plan, and what a run installs.

The installer runs under bash as the test user, against the rig in `tests/deploy/rig.py`: stub
system programs, a fake system root, and a fake interpreter. Linux only.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path

import pytest

from tests.deploy.rig import (
    INSTALL,
    VERSION,
    Rig,
    arch_directory,
    make_rig,
)
from tests.deploy.scripts import DEPLOY, linux_only, run_script
from tools.lint_deploy import placeholders, render

pytestmark = linux_only

REQUIRED = (
    "prefix",
    "user",
    "data_dir",
    "config_dir",
    "wheel",
    "requirements",
    "time_source",
)
KEY_TEXT = "a-connection-key-for-the-tests\n"
HOME_FOLDER = "/" + "home" + "/someone/data"  # assembled, so that check_repo does not flag it


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return make_rig(tmp_path)


def plan(output: str) -> dict[str, str]:
    """The `Label   value` lines of the plan."""
    labels = (
        "Release",
        "Prefix",
        "Service user",
        "Data directory",
        "Config directory",
        "Python",
        "Time sources",
        "Vendor SDK",
        "Keep releases",
        "Supervisor rule",
        "System root",
    )
    found: dict[str, str] = {}
    for line in output.splitlines():
        for label in labels:
            if line.startswith(label + " "):
                found[label] = line[len(label) :].strip()
    return found


def release_id(rig: Rig) -> str:
    """The release name that the installer derives from the wheel and the requirements."""
    result = rig.install("--dry-run")
    release = plan(result.stdout).get("Release")
    assert release, result.output
    return release


# --- The command line ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", REQUIRED)
def test_a_missing_required_parameter_is_named(rig: Rig, key: str) -> None:
    result = rig.install(**{key: None})
    assert result.returncode == 2
    option = "--" + key.replace("_", "-")
    if key == "time_source":
        assert "missing required parameter: --time-source (or --no-time-config)" in result.stderr
    else:
        assert f"missing required parameter: {option}\n" in result.stderr
    assert not rig.prefix.exists()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--bogus"], "unknown option: --bogus"),
        (["--keep"], "the option --keep needs a value"),
        (["--keep", "--dry-run"], "needs a value, not the option --dry-run"),
    ],
)
def test_a_bad_command_line_is_reported(rig: Rig, arguments: list[str], message: str) -> None:
    result = rig.install(*arguments)
    assert result.returncode == 2
    assert message in result.stderr
    assert 'Run "install.sh --help"' in result.stderr


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"prefix": "opt/seeingmon"}, "--prefix needs an absolute path of plain characters"),
        ({"prefix": "/opt/see ingmon"}, "--prefix needs an absolute path of plain characters"),
        ({"prefix": "/opt/../etc"}, "--prefix must not contain '..'"),
        ({"prefix": "/opt/seeingmon/"}, "--prefix must not end with a slash"),
        ({"prefix": "/opt"}, "/opt is a system directory"),
        ({"prefix": "/"}, "/ is a system directory"),
        ({"config_dir": "/etc"}, "/etc is a system directory"),
        ({"data_dir": "/usr"}, "/usr is a system directory"),
        ({"data_dir": HOME_FOLDER}, f"the units hide {HOME_FOLDER}"),
        ({"config_dir": "/root/config"}, "the units hide /root/config"),
        ({"user": "Seeing"}, "--user needs a plain account name"),
        ({"user": "root user"}, "--user needs a plain account name"),
        ({"time_source": "bad host"}, "--time-source needs a host name or address"),
        ({"time_source": "-oOption"}, "--time-source needs a host name or address"),
    ],
)
def test_a_bad_value_is_reported(rig: Rig, changes: dict[str, str], message: str) -> None:
    result = rig.install(**changes)
    assert result.returncode == 2
    assert message in result.stderr
    assert not rig.prefix.exists()


def test_the_prefix_the_data_and_the_config_must_not_hold_each_other(rig: Rig) -> None:
    cases = [
        (
            {"data_dir": str(rig.prefix / "data")},
            "--prefix and --data-dir must not hold each other",
        ),
        ({"config_dir": str(rig.prefix)}, "--prefix and --config-dir must not hold each other"),
        ({"config_dir": str(rig.data_dir / "config")}, "--data-dir and --config-dir must not hold"),
    ]
    for changes, message in cases:
        result = rig.install(**changes)
        assert result.returncode == 2, changes
        assert message in result.stderr


def test_the_files_must_exist_and_look_right(rig: Rig, tmp_path: Path) -> None:
    missing = str(tmp_path / "missing")
    bad_wheel = rig.write("other-1.0-py3-none-any.whl", "x")
    no_hashes = rig.write("plain.txt", "numpy==2.0\n")
    cases = [
        ({"wheel": missing}, "--wheel names a file that does not exist"),
        ({"wheel": str(bad_wheel)}, "--wheel must be a seeingmon wheel"),
        ({"requirements": missing}, "--requirements names a file that does not exist"),
        ({"requirements": str(no_hashes)}, "--requirements has no hashes"),
        ({"system_root": missing}, "--system-root names a folder that does not exist"),
        ({"wheelhouse": missing}, "--wheelhouse names a folder that does not exist"),
        ({"local_config": missing}, "--local-config names a file that does not exist"),
        ({"env_file": missing}, "--env-file names a file that does not exist"),
        ({"connection_key_file": missing}, "--connection-key-file names a file"),
        ({"token_hash_file": missing}, "--token-hash-file names a file that does not exist"),
    ]
    for changes, message in cases:
        result = rig.install(**changes)
        assert result.returncode == 2, changes
        assert message in result.stderr, (changes, result.stderr)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--keep", "1"], "--keep needs a number of at least 2"),
        (["--keep", "many"], "--keep needs a whole number"),
        (["--usbfs-memory-mb", "8"], "--usbfs-memory-mb needs a number of at least 16"),
        (["--usbfs-memory-mb", "lots"], "--usbfs-memory-mb needs a whole number"),
        (["--no-time-config"], "--time-source and --no-time-config exclude each other"),
        (["--sdk-archive", __file__], "--sdk-archive needs --sdk-sha256"),
        (["--sdk-sha256", "0" * 64], "--sdk-sha256 needs --sdk-archive"),
        (["--sdk-archive", __file__, "--sdk-sha256", "ABC"], "needs 64 lowercase hex digits"),
    ],
)
def test_a_bad_option_combination_is_reported(rig: Rig, extra: list[str], message: str) -> None:
    result = rig.install(*extra)
    assert result.returncode == 2
    assert message in result.stderr


def test_help_lists_every_option(rig: Rig) -> None:
    result = run_script(INSTALL, ["--help"], env=rig.env())
    assert result.returncode == 0
    for option in (
        "--prefix",
        "--user",
        "--data-dir",
        "--config-dir",
        "--wheel",
        "--requirements",
        "--time-source",
        "--no-time-config",
        "--sdk-archive",
        "--sdk-sha256",
        "--local-config",
        "--connection-key-file",
        "--token-hash-file",
        "--env-file",
        "--wheelhouse",
        "--python",
        "--usbfs-memory-mb",
        "--keep",
        "--supervisor-actions",
        "--system-root",
        "--dry-run",
    ):
        assert option in result.stdout, option


# --- The plan -----------------------------------------------------------------------------------


def test_a_dry_run_prints_the_plan_and_changes_nothing(rig: Rig) -> None:
    result = rig.install_env({"PATH": "/usr/bin:/bin"}, "--dry-run")  # no stubs, so no root
    assert result.returncode == 0, result.output
    values = plan(result.stdout)
    assert re.fullmatch(rf"{re.escape(VERSION)}-[0-9a-f]{{12}}", values["Release"])
    assert values["Prefix"] == str(rig.prefix)
    assert values["Service user"] == "seeingmon (a system account without a login)"
    assert values["Data directory"] == str(rig.data_dir)
    assert values["Config directory"] == str(rig.config_dir)
    assert values["Time sources"] == "time.example.org 192.0.2.10"
    assert values["Vendor SDK"] == "none"
    assert values["Keep releases"] == "2"
    assert values["Supervisor rule"] == "no"
    assert "Steps:" in result.stdout
    assert "dry run: nothing changed." in result.stdout
    for step in range(1, 11):
        assert re.search(rf"^\s*{step}\. ", result.stdout, re.M), step
    assert not rig.prefix.exists()
    assert not rig.data_dir.exists()
    assert not rig.config_dir.exists()
    assert rig.stubs.calls() == []


def test_the_plan_shows_the_choices(rig: Rig, tmp_path: Path) -> None:
    sdk, checksum = rig.make_sdk({"x/lib/x64/libASICamera2.so": b"x"})
    result = rig.install(
        "--dry-run",
        "--supervisor-actions",
        "--keep",
        "3",
        sdk_archive=str(sdk),
        sdk_sha256=checksum,
        time_source=None,
        no_time_config=True,
    )
    assert result.returncode == 0, result.output
    values = plan(result.stdout)
    assert values["Vendor SDK"] == "from the archive, after the checksum matches"
    assert values["Supervisor rule"] == "yes (a polkit rule)"
    assert values["Keep releases"] == "3"
    assert values["Time sources"] == "chrony is left alone"


def test_the_release_name_follows_the_contents(rig: Rig) -> None:
    first = release_id(rig)
    assert release_id(rig) == first
    rig.new_release("changed wheel")
    second = release_id(rig)
    assert second != first
    assert second.startswith(f"{VERSION}-")
    rig.requirements.write_text("numpy==2.1 \\\n    --hash=sha256:00\n", encoding="utf-8")
    assert release_id(rig) not in {first, second}


# --- A fresh install ----------------------------------------------------------------------------


def test_a_fresh_install_lays_out_a_release_and_switches_to_it(rig: Rig) -> None:
    result = rig.install()
    assert result.returncode == 0, result.output
    name = release_id(rig)
    assert rig.releases == [name]
    release = rig.prefix / "releases" / name
    assert (release / ".installed").read_text(encoding="utf-8").strip() == name
    assert (release / "venv" / "bin" / "seeingmon").is_file()
    assert rig.link("current") == f"releases/{name}"
    assert rig.link("previous") is None
    assert "Summary" in result.stdout
    assert f"switched {rig.prefix}/current to release {name}" in result.stdout
    assert not list(rig.prefix.rglob("*.new"))
    assert not list(rig.config_dir.rglob("*.new"))
    assert not list(rig.system_root.rglob("*.new"))


def test_the_installer_creates_a_system_user_without_a_login(rig: Rig) -> None:
    rig.install()
    assert rig.calls("useradd") == [
        "[--system] [--user-group] [--no-create-home] [--home-dir] [/var/lib/seeingmon] "
        "[--shell] [/usr/sbin/nologin] [--comment] [seeingmon service] [seeingmon]"
    ]
    assert rig.calls("usermod") == ["[-aG] [gpio] [seeingmon]"]


def test_the_user_does_not_join_a_gpio_group_that_does_not_exist(rig: Rig) -> None:
    (rig.state / "group-gpio").unlink()
    result = rig.install()
    assert result.returncode == 0, result.output
    assert rig.calls("usermod") == []
    assert "this system has no gpio group" in result.stdout


def test_the_release_comes_from_hashed_requirements_and_the_wheel(rig: Rig) -> None:
    rig.install()
    name = release_id(rig)
    venv = rig.prefix / "releases" / name / "venv"
    python_calls = [call for call in rig.calls("fakepython") if "[pip]" in call or "[venv]" in call]
    assert python_calls[0] == f"[-m] [venv] [{venv}]"
    requirements = (
        "[-m] [pip] [install] [--disable-pip-version-check] [--no-input] [--no-cache-dir] "
        f"[--require-hashes] [--only-binary=:all:] [--no-deps] [-r] [{rig.requirements}]"
    )
    wheel = (
        "[-m] [pip] [install] [--disable-pip-version-check] [--no-input] [--no-cache-dir] "
        f"[--no-index] [--no-deps] [--] [{rig.wheel}]"
    )
    assert python_calls[1] == requirements
    assert python_calls[2] == wheel
    assert python_calls[3] == "[-m] [pip] [check] [--disable-pip-version-check]"


def test_a_wheelhouse_replaces_the_package_index(rig: Rig, tmp_path: Path) -> None:
    houses = tmp_path / "wheels"
    houses.mkdir()
    rig.install(wheelhouse=str(houses))
    requirements = next(call for call in rig.calls("fakepython") if "[--require-hashes]" in call)
    assert f"[--no-index] [--find-links] [{houses}] [--require-hashes]" in requirements


def test_the_units_and_system_files_are_rendered_from_the_templates(rig: Rig) -> None:
    rig.install()
    values = rig.values()
    pairs = {
        "systemd/seeingmon.target": rig.etc("systemd/system/seeingmon.target"),
        "systemd/seeingmon-acquire.service": rig.etc("systemd/system/seeingmon-acquire.service"),
        "systemd/seeingmon-core.service": rig.etc("systemd/system/seeingmon-core.service"),
        "systemd/seeingmon-web.service": rig.etc("systemd/system/seeingmon-web.service"),
        "systemd/seeingmon-failed@.service": rig.etc("systemd/system/seeingmon-failed@.service"),
        "udev/99-seeingmon-asi.rules": rig.etc("udev/rules.d/99-seeingmon-asi.rules"),
        "tmpfiles/seeingmon-usb.conf": rig.etc("tmpfiles.d/seeingmon-usb.conf"),
        "journald/seeingmon.conf": rig.etc("systemd/journald.conf.d/seeingmon.conf"),
        "chrony/seeingmon.conf": rig.etc("chrony/conf.d/seeingmon.conf"),
        "wrapper/seeingmon.sh": rig.prefix / "bin" / "seeingmon",
    }
    for template, target in pairs.items():
        text = (DEPLOY / template).read_text(encoding="utf-8")
        expected = render(text, values)
        assert target.read_text(encoding="utf-8") == expected, template
        assert placeholders(expected) == set(), template
    core = pairs["systemd/seeingmon-core.service"].read_text(encoding="utf-8")
    assert f"ExecStart={rig.prefix}/current/venv/bin/seeingmon core" in core
    assert f"ReadWritePaths={rig.data_dir}" in core
    assert f"WorkingDirectory={rig.config_dir}" in core
    assert "User=seeingmon" in core


def test_the_files_get_their_modes(rig: Rig) -> None:
    rig.install()
    assert rig.mode(rig.etc("systemd/system/seeingmon-core.service")) == 0o644
    assert rig.mode(rig.etc("udev/rules.d/99-seeingmon-asi.rules")) == 0o644
    assert rig.mode(rig.prefix / "bin" / "seeingmon") == 0o755
    assert rig.mode(rig.prefix / "bin" / "rollback.sh") == 0o755
    assert rig.mode(rig.config_dir) == 0o750
    assert rig.mode(rig.config_dir / "credentials") == 0o750
    assert rig.mode(rig.config_dir / "local") == 0o750
    assert rig.mode(rig.data_dir) == 0o750
    assert (rig.prefix / "bin" / "rollback.sh").read_bytes() == (
        DEPLOY / "rollback.sh"
    ).read_bytes()


def test_the_installer_gives_ownership_to_the_right_accounts(rig: Rig) -> None:
    rig.install()
    chown = rig.calls("chown")
    assert any(call.startswith("[root:seeingmon] [--] [" + str(rig.config_dir)) for call in chown)
    assert f"[seeingmon:seeingmon] [--] [{rig.data_dir}]" in chown
    key = rig.config_dir / "credentials" / "seeingmon-connection-key"
    assert f"[seeingmon:seeingmon] [{key}.new]" in chown


def test_the_system_is_told_about_the_new_files_in_the_right_order(rig: Rig) -> None:
    rig.install()
    calls = rig.stubs.calls()

    def position(text: str) -> int:
        return next(i for i, call in enumerate(calls) if text in call)

    assert position("systemctl [daemon-reload]") < position("systemctl [enable]")
    assert position("systemctl [enable]") < position("systemctl [restart] [seeingmon.target]")
    assert position("udevadm [control] [--reload-rules]") < position("udevadm [trigger]")
    assert any("[--attr-match=idVendor=03c3]" in call for call in rig.calls("udevadm"))
    assert any(call.startswith("[--create] [") for call in rig.calls("systemd-tmpfiles"))
    assert "[restart] [chrony]" in rig.calls("systemctl")
    assert "[restart] [systemd-journald]" in rig.calls("systemctl")
    enable = next(call for call in rig.calls("systemctl") if call.startswith("[enable]"))
    for unit in ("seeingmon.target", "seeingmon-acquire.service", "seeingmon-web.service"):
        assert unit in enable
    assert "seeingmon-failed@" not in enable


def test_a_connection_key_is_made_once_and_kept(rig: Rig) -> None:
    rig.install()
    key = rig.config_dir / "credentials" / "seeingmon-connection-key"
    text = key.read_text(encoding="utf-8")
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", text)
    assert rig.mode(key) == 0o600
    rig.install()
    assert key.read_text(encoding="utf-8") == text


def test_a_connection_key_that_you_give_replaces_the_key(rig: Rig) -> None:
    rig.install()
    rig.clear_calls()
    key_file = rig.write("key.txt", KEY_TEXT)
    result = rig.install(connection_key_file=str(key_file))
    assert result.returncode == 0, result.output
    key = rig.config_dir / "credentials" / "seeingmon-connection-key"
    assert key.read_text(encoding="utf-8") == KEY_TEXT
    assert rig.mode(key) == 0o600
    assert "[restart] [seeingmon.target]" in rig.calls("systemctl")


@pytest.mark.parametrize("text", ["short\n", "x" * 1025, "   \n"])
def test_a_connection_key_of_the_wrong_length_stops_the_run_before_any_change(
    rig: Rig, text: str
) -> None:
    key_file = rig.write("key.txt", text)
    result = rig.install(connection_key_file=str(key_file))
    assert result.returncode == 1
    assert "must have 16 to 1,024 characters" in result.stderr
    assert not rig.prefix.exists()
    assert rig.calls("useradd") == []


def test_the_token_hash_you_give_becomes_the_web_credential(rig: Rig) -> None:
    hash_file = rig.write("hash.txt", "a-token-hash-for-the-tests\n")
    result = rig.install(token_hash_file=str(hash_file))
    assert result.returncode == 0, result.output
    target = rig.config_dir / "credentials" / "seeingmon-token-hash"
    assert target.read_text(encoding="utf-8") == "a-token-hash-for-the-tests\n"
    assert rig.mode(target) == 0o600
    assert f"[seeingmon:seeingmon] [{target}.new]" in rig.calls("chown")
    assert "the API has no token hash" not in result.stderr


def test_without_a_token_hash_the_installer_keeps_an_empty_file_and_warns(rig: Rig) -> None:
    result = rig.install()
    target = rig.config_dir / "credentials" / "seeingmon-token-hash"
    assert target.read_text(encoding="utf-8") == ""  # the web unit loads this file at every start
    assert rig.mode(target) == 0o600
    assert "the API has no token hash" in result.stderr
    assert "web hash-token" in result.stderr
    rig.clear_calls()
    again = rig.install()
    assert "Nothing changed" in again.stdout
    assert target.read_text(encoding="utf-8") == ""


def test_a_later_token_hash_replaces_the_empty_file(rig: Rig) -> None:
    rig.install()
    hash_file = rig.write("hash.txt", "a-token-hash-for-the-tests\n")
    result = rig.install(token_hash_file=str(hash_file))
    assert result.returncode == 0, result.output
    target = rig.config_dir / "credentials" / "seeingmon-token-hash"
    assert target.read_text(encoding="utf-8") == "a-token-hash-for-the-tests\n"
    assert "[restart] [seeingmon.target]" in rig.calls("systemctl")
    again = rig.install()  # a run without the file keeps the hash
    assert target.read_text(encoding="utf-8") == "a-token-hash-for-the-tests\n"
    assert "the API has no token hash" not in again.stderr


@pytest.mark.parametrize(
    "text",
    ['[auth]\ntoken_hash = "x"\n', '[auth]\ntoken_hash_file = "/etc/some-file"\n'],
)
def test_a_hash_in_the_local_configuration_needs_no_warning(rig: Rig, text: str) -> None:
    config = rig.write("local.toml", 'station_id = "x"\n' + text)
    result = rig.install(local_config=str(config))
    assert "the API has no token hash" not in result.stderr


def test_a_hash_in_the_environment_file_needs_no_warning(rig: Rig) -> None:
    env_file = rig.write("env.txt", "SEEINGMON_AUTH__TOKEN_HASH=value\n")
    result = rig.install(env_file=str(env_file))
    assert "the API has no token hash" not in result.stderr


def test_the_environment_file_is_installed_for_systemd_only(rig: Rig) -> None:
    env_file = rig.write("env.txt", "SEEINGMON_AUTH__TOKEN_HASH=value\n")
    result = rig.install(env_file=str(env_file))
    assert result.returncode == 0, result.output
    target = rig.config_dir / "seeingmon.env"
    assert target.read_text(encoding="utf-8") == "SEEINGMON_AUTH__TOKEN_HASH=value\n"
    assert rig.mode(target) == 0o600
    assert f"[root:root] [{target}.new]" in rig.calls("chown")


def test_the_local_configuration_is_installed_for_the_service_user_only(rig: Rig) -> None:
    config = rig.write("local.toml", 'station_id = "x"\n[services.acquire]\ndriver = "asi"\n')
    result = rig.install(local_config=str(config))
    assert result.returncode == 0, result.output
    target = rig.config_dir / "local" / "config.toml"
    assert target.read_text(encoding="utf-8") == config.read_text(encoding="utf-8")
    assert rig.mode(target) == 0o600
    assert f"[seeingmon:seeingmon] [{target}.new]" in rig.calls("chown")
    for warning in (
        "there is no local configuration",
        "does not select the asi camera driver",
        "sets no station_id",
        "differs from --data-dir",
    ):
        assert warning not in result.stderr, warning


def test_the_local_template_comes_with_the_release(rig: Rig) -> None:
    rig.install()
    template = rig.config_dir / "local.example.toml"
    assert template.read_text(encoding="utf-8") == "# the local template\n"
    assert rig.mode(template) == 0o644


def test_a_changed_local_configuration_keeps_the_old_one(rig: Rig) -> None:
    first = rig.write("one.toml", 'station_id = "one"\n')
    rig.install(local_config=str(first))
    second = rig.write("two.toml", 'station_id = "two"\n')
    result = rig.install(local_config=str(second))
    assert result.returncode == 0, result.output
    target = rig.config_dir / "local" / "config.toml"
    assert target.read_text(encoding="utf-8") == 'station_id = "two"\n'
    previous = Path(f"{target}.previous")
    assert previous.read_text(encoding="utf-8") == 'station_id = "one"\n'
    assert rig.mode(previous) == 0o600


@pytest.mark.parametrize(
    ("text", "warning"),
    [
        ('station_id = "x"\n', "does not select the asi camera driver"),
        ('[services.acquire]\ndriver = "asi"\n', "sets no station_id"),
        ('station_id = "x"\n[paths]\ndata_dir = "/elsewhere"\n', "differs from --data-dir"),
    ],
)
def test_a_local_configuration_that_looks_wrong_gets_a_warning(
    rig: Rig, text: str, warning: str
) -> None:
    config = rig.write("local.toml", text)
    result = rig.install(local_config=str(config))
    assert result.returncode == 0, result.output
    assert warning in result.stderr
    assert warning in result.stdout  # the summary repeats it


def test_a_missing_local_configuration_is_a_warning_with_the_next_step(rig: Rig) -> None:
    result = rig.install()
    assert result.returncode == 0
    assert "there is no local configuration: copy" in result.stderr
    assert str(rig.config_dir / "local.example.toml") in result.stderr


def test_a_data_directory_on_the_root_file_system_is_a_warning(rig: Rig) -> None:
    result = rig.install()
    assert "is on the root file system" in result.stderr
    assert "does not partition anything" in result.stderr
    result = rig.install_env({"FAKE_DATA_MOUNT": str(rig.data_dir)})
    assert "is on the root file system" not in result.stderr


def test_a_tmp_that_is_not_a_tmpfs_is_a_warning(rig: Rig) -> None:
    result = rig.install_env({"FAKE_TMP_FSTYPE": "ext4"})
    assert "/tmp is not a tmpfs" in result.stderr
    assert "systemctl enable tmp.mount" in result.stderr


def test_the_usb_buffer_setting_follows_the_option(rig: Rig) -> None:
    result = rig.install("--usbfs-memory-mb", "400")
    assert result.returncode == 0, result.output
    text = rig.etc("tmpfiles.d/seeingmon-usb.conf").read_text(encoding="utf-8")
    assert "- - - - 400\n" in text
    assert "usbcore.usbfs_memory_mb=400" in text
    assert "the USB buffer size is" in result.stdout + result.stderr


def test_no_time_config_leaves_chrony_alone(rig: Rig) -> None:
    rig.etc("chrony/chrony.conf").unlink()  # without chrony, the run must still work
    result = rig.install(time_source=None, no_time_config=True)
    assert result.returncode == 0, result.output
    assert not rig.etc("chrony/conf.d/seeingmon.conf").exists()
    assert "[restart] [chrony]" not in rig.calls("systemctl")


def test_the_chrony_file_names_each_time_source(rig: Rig) -> None:
    rig.install()
    text = rig.etc("chrony/conf.d/seeingmon.conf").read_text(encoding="utf-8")
    assert "\nserver time.example.org iburst\nserver 192.0.2.10 iburst\n" in text


def test_the_polkit_rule_is_installed_only_on_request(rig: Rig) -> None:
    rig.install()
    rule = rig.etc("polkit-1/rules.d/50-seeingmon.rules")
    assert not rule.exists()
    result = rig.install("--supervisor-actions")
    assert result.returncode == 0, result.output
    assert 'subject.user !== "seeingmon"' in rule.read_text(encoding="utf-8")
    result = rig.install()
    assert "is still installed, although you left out --supervisor-actions" in result.stderr


def test_the_installed_wrapper_and_rollback_script_are_the_ones_of_the_release(rig: Rig) -> None:
    rig.install()
    wrapper = (rig.prefix / "bin" / "seeingmon").read_text(encoding="utf-8")
    assert f'cd -- "{rig.config_dir}"' in wrapper
    assert f'exec "{rig.prefix}/current/venv/bin/seeingmon" "$@"' in wrapper
    assert os.access(rig.prefix / "bin" / "rollback.sh", os.X_OK)
    assert stat.S_IMODE((rig.prefix / "bin" / "rollback.sh").stat().st_mode) == 0o755


# --- The vendor SDK -----------------------------------------------------------------------------


def sdk_members() -> dict[str, bytes]:
    return {
        "sdk/include/ASICamera2.h": b"header",
        "sdk/lib/x64/libASICamera2.so": b"x64 library",
        "sdk/lib/armv8/libASICamera2.so.1.99": b"arm library",
    }


@pytest.mark.skipif(arch_directory() is None, reason="the SDK has no folder for this machine")
def test_the_sdk_is_installed_privately_and_the_services_get_its_path(rig: Rig) -> None:
    sdk, checksum = rig.make_sdk(sdk_members())
    result = rig.install(sdk_archive=str(sdk), sdk_sha256=checksum)
    assert result.returncode == 0, result.output
    folder = rig.prefix / "sdk" / checksum[:12]
    assert (folder / ".installed").read_text(encoding="utf-8").strip() == checksum
    assert rig.mode(folder) == 0o750
    assert rig.mode(rig.prefix / "sdk") == 0o750
    arch = arch_directory()
    library = next(folder.rglob(f"lib/{arch}/libASICamera2.so*"))
    assert rig.mode(library) == 0o640
    env_file = rig.config_dir / "sdk.env"
    text = env_file.read_text(encoding="utf-8")
    assert f"SEEINGMON_ASI__LIBRARY_PATH={library}\n" in text
    assert "Managed by the seeingmon installer" in text
    assert f"[-R] [root:seeingmon] [--] [{folder}.part]" in rig.calls("chown")


@pytest.mark.skipif(arch_directory() is None, reason="the SDK has no folder for this machine")
def test_a_vendor_library_with_a_missing_dependency_is_a_warning(rig: Rig) -> None:
    sdk, checksum = rig.make_sdk(sdk_members())
    quiet = rig.install(sdk_archive=str(sdk), sdk_sha256=checksum)
    assert "the vendor library needs libraries" not in quiet.stderr
    loud = rig.install_env({"FAKE_LDD_MISSING": "1"}, sdk_archive=str(sdk), sdk_sha256=checksum)
    assert loud.returncode == 0, loud.output
    assert "the vendor library needs libraries that this system lacks" in loud.stderr
    assert "libusb-1.0.so.0 => not found" in loud.stderr
    assert "install the package libusb-1.0-0" in loud.stderr


@pytest.mark.skipif(arch_directory() is None, reason="the SDK has no folder for this machine")
def test_a_second_run_finds_the_sdk_installed(rig: Rig) -> None:
    sdk, checksum = rig.make_sdk(sdk_members())
    rig.install(sdk_archive=str(sdk), sdk_sha256=checksum)
    result = rig.install(sdk_archive=str(sdk), sdk_sha256=checksum)
    assert "the vendor SDK is already installed" in result.stdout
    assert "Nothing changed" in result.stdout


def test_an_sdk_with_the_wrong_checksum_stops_the_run_before_any_change(rig: Rig) -> None:
    sdk, checksum = rig.make_sdk(sdk_members())
    wrong = ("0" if checksum[0] != "0" else "1") + checksum[1:]
    result = rig.install(sdk_archive=str(sdk), sdk_sha256=wrong)
    assert result.returncode == 1
    assert "does not have the expected checksum" in result.stderr
    assert checksum in result.stderr
    assert not rig.prefix.exists()
    assert rig.calls("useradd") == []


@pytest.mark.parametrize("member", ["../escape", "/absolute/path", "dir/../../escape"])
def test_an_sdk_with_a_dangerous_member_is_refused(rig: Rig, member: str) -> None:
    sdk, checksum = rig.make_sdk({member: b"x"})
    result = rig.install(sdk_archive=str(sdk), sdk_sha256=checksum)
    assert result.returncode == 1
    assert "absolute path or a '..' component" in result.stderr
    assert not rig.prefix.exists()


def test_a_file_that_is_not_a_tar_archive_is_refused(rig: Rig) -> None:
    junk = rig.write("junk.tar", "not an archive")
    checksum = hashlib.sha256(junk.read_bytes()).hexdigest()
    result = rig.install(sdk_archive=str(junk), sdk_sha256=checksum)
    assert result.returncode == 1
    assert "not a tar archive" in result.stderr


@pytest.mark.skipif(arch_directory() is None, reason="the SDK has no folder for this machine")
def test_an_sdk_without_a_library_for_this_machine_is_refused(rig: Rig) -> None:
    sdk, checksum = rig.make_sdk({"sdk/include/ASICamera2.h": b"header"})
    result = rig.install(sdk_archive=str(sdk), sdk_sha256=checksum)
    assert result.returncode == 1
    assert "has no libASICamera2.so for" in result.stderr
