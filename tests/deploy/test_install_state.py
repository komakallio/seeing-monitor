"""`deploy/install.sh` and `deploy/rollback.sh` over several runs: reruns, upgrades, refusals,
failures, the wrapper, the rollback, and one run with a real virtual environment.

The rig is in `tests/deploy/rig.py`. Linux only.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.deploy.rig import Rig, make_rig
from tests.deploy.scripts import DEPLOY, linux_only, run_script

pytestmark = linux_only


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return make_rig(tmp_path)


def release_id(rig: Rig) -> str:
    result = rig.install("--dry-run")
    for line in result.stdout.splitlines():
        if line.startswith("Release "):
            return line.split()[1]
    raise AssertionError(result.output)


def install_releases(rig: Rig, count: int, *extra: str) -> list[str]:
    """Install `count` releases in a row, and give each marker a distinct age. Returns the names."""
    names: list[str] = []
    for index in range(count):
        rig.new_release(f"release {index}")
        result = rig.install(*extra)
        assert result.returncode == 0, result.output
        name = release_id(rig)
        stamp = 1_700_000_000 + index * 1000
        os.utime(rig.prefix / "releases" / name / ".installed", (stamp, stamp))
        names.append(name)
    return names


# --- Runs after the first -----------------------------------------------------------------------


def test_a_second_run_with_the_same_files_changes_nothing(rig: Rig) -> None:
    rig.install()
    rig.clear_calls()
    result = rig.install()
    assert result.returncode == 0, result.output
    assert "Nothing changed: the installation already matched" in result.stdout
    assert "What changed" not in result.stdout
    assert rig.calls("useradd") == []
    assert rig.calls("usermod") == []
    assert not any("[venv]" in call or "[pip]" in call for call in rig.calls("fakepython"))
    systemctl = rig.calls("systemctl")
    assert "[daemon-reload]" not in systemctl
    assert not any(call.startswith("[restart]") for call in systemctl)
    assert "[start] [seeingmon.target]" in systemctl
    assert rig.calls("udevadm") == []


def test_a_second_run_does_not_touch_a_running_service(rig: Rig) -> None:
    rig.install()
    unit = rig.etc("systemd/system/seeingmon-core.service")
    before = unit.stat().st_mtime_ns
    rig.install()
    assert unit.stat().st_mtime_ns == before


def test_an_upgrade_installs_a_new_release_and_keeps_the_old_one(rig: Rig) -> None:
    rig.install()
    first = release_id(rig)
    rig.new_release("two")
    rig.clear_calls()
    result = rig.install()
    second = release_id(rig)
    assert result.returncode == 0, result.output
    assert rig.releases == sorted([first, second])
    assert rig.link("current") == f"releases/{second}"
    assert rig.link("previous") == f"releases/{first}"
    assert f"(the earlier release is releases/{first})" in result.stdout
    systemctl = rig.calls("systemctl")
    assert "[restart] [seeingmon.target]" in systemctl
    assert "[daemon-reload]" not in systemctl  # the units did not change
    assert rig.calls("useradd") == []


def test_the_installer_keeps_only_as_many_releases_as_you_ask(rig: Rig) -> None:
    names = install_releases(rig, 3)
    assert rig.releases == sorted(names[1:])
    assert rig.link("current") == f"releases/{names[2]}"
    assert rig.link("previous") == f"releases/{names[1]}"


def test_the_keep_option_keeps_more(rig: Rig) -> None:
    names = install_releases(rig, 3, "--keep", "3")
    assert rig.releases == sorted(names)
    names.append(install_releases(rig, 1, "--keep", "3")[0])
    assert rig.releases == sorted(names[1:])


def test_the_current_and_the_previous_release_are_never_pruned(rig: Rig) -> None:
    names = install_releases(rig, 2)
    # Make the previous release the oldest by far, and install a third release with --keep 2.
    rig.new_release("three")
    old = rig.prefix / "releases" / names[0] / ".installed"
    os.utime(old, (1_000_000, 1_000_000))
    rig.install()
    third = release_id(rig)
    assert rig.link("previous") == f"releases/{names[1]}"
    assert rig.releases == sorted([names[1], third])


def test_an_unfinished_release_of_an_earlier_run_is_rebuilt(rig: Rig) -> None:
    name = release_id(rig)
    leftover = rig.prefix / "releases" / name / "venv"
    leftover.mkdir(parents=True)
    (leftover / "junk").write_text("half a venv", encoding="utf-8")
    result = rig.install()
    assert result.returncode == 0, result.output
    assert "removing the unfinished release of an earlier run" in result.stdout
    assert not (rig.prefix / "releases" / name / "venv" / "junk").exists()
    assert (rig.prefix / "releases" / name / ".installed").is_file()


def test_an_unfinished_release_that_is_current_stops_the_run(rig: Rig) -> None:
    name = release_id(rig)
    (rig.prefix / "releases" / name).mkdir(parents=True)
    (rig.prefix / "current").symlink_to(f"releases/{name}")
    result = rig.install()
    assert result.returncode == 1
    assert "is unfinished, but" in result.stderr
    assert (rig.prefix / "releases" / name).is_dir()


# --- Refusals: the installer stops with a message and changes nothing ----------------------------


def assert_untouched(rig: Rig) -> None:
    assert not rig.prefix.exists()
    assert not rig.config_dir.exists()
    assert not rig.data_dir.exists()
    assert rig.calls("useradd") == []
    assert not list(rig.system_root.rglob("seeingmon*"))


def test_it_refuses_to_run_without_root(rig: Rig) -> None:
    result = rig.install_env({"FAKE_UID": "1000"})
    assert result.returncode == 1
    assert "run the installer as root" in result.stderr
    assert_untouched(rig)


@pytest.mark.parametrize(
    ("variable", "message"),
    [
        ("FAKE_VERSION_STATUS", "is older than Python 3.11"),
        ("FAKE_VENV_STATUS", "cannot make virtual environments"),
    ],
)
def test_it_refuses_a_python_that_cannot_do_the_job(rig: Rig, variable: str, message: str) -> None:
    result = rig.install_env({variable: "1"})
    assert result.returncode == 1
    assert message in result.stderr
    assert_untouched(rig)


def test_it_refuses_a_missing_interpreter(rig: Rig) -> None:
    result = rig.install(python=str(rig.root / "no-such-python"))
    assert result.returncode == 1
    assert "there is no Python interpreter named" in result.stderr


def test_it_needs_chrony_when_you_ask_for_time_sources(rig: Rig) -> None:
    rig.etc("chrony/chrony.conf").unlink()
    result = rig.install()
    assert result.returncode == 1
    assert "chrony is not installed" in result.stderr
    assert_untouched(rig)


def test_it_needs_a_chrony_configuration_that_reads_the_drop_in_folder(rig: Rig) -> None:
    rig.etc("chrony/chrony.conf").write_text("pool pool.example.org iburst\n", encoding="utf-8")
    result = rig.install()
    assert result.returncode == 1
    assert "does not read /etc/chrony/conf.d" in result.stderr
    assert "--no-time-config" in result.stderr
    assert_untouched(rig)


def test_it_needs_polkit_for_the_supervisor_rule(rig: Rig) -> None:
    shutil.rmtree(rig.etc("polkit-1"))
    result = rig.install("--supervisor-actions")
    assert result.returncode == 1
    assert "polkit is not installed" in result.stderr
    assert_untouched(rig)


@pytest.mark.parametrize(
    "relative",
    [
        "systemd/system/seeingmon-core.service",
        "udev/rules.d/99-seeingmon-asi.rules",
        "tmpfiles.d/seeingmon-usb.conf",
        "systemd/journald.conf.d/seeingmon.conf",
        "chrony/conf.d/seeingmon.conf",
    ],
)
def test_it_does_not_overwrite_a_system_file_that_it_did_not_write(rig: Rig, relative: str) -> None:
    target = rig.etc(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# written by hand\n", encoding="utf-8")
    result = rig.install()
    assert result.returncode == 1
    assert f"{target} exists and the installer did not write it" in result.stderr
    assert target.read_text(encoding="utf-8") == "# written by hand\n"
    assert not rig.prefix.exists()
    assert rig.calls("useradd") == []


def test_it_overwrites_a_system_file_that_it_wrote_itself(rig: Rig) -> None:
    rig.install()
    target = rig.etc("systemd/system/seeingmon-core.service")
    target.write_text(
        target.read_text(encoding="utf-8") + "# an edit that the installer undoes\n",
        encoding="utf-8",
    )
    result = rig.install()
    assert result.returncode == 0, result.output
    assert "an edit that the installer undoes" not in target.read_text(encoding="utf-8")
    assert "[daemon-reload]" in rig.calls("systemctl")


def test_it_refuses_an_existing_account_with_a_login_shell(rig: Rig) -> None:
    (rig.state / "user-seeingmon").write_text("", encoding="utf-8")
    (rig.state / "shell-seeingmon").write_text("/bin/bash", encoding="utf-8")
    result = rig.install()
    assert result.returncode == 1
    assert "has a login shell (/bin/bash)" in result.stderr
    assert rig.calls("useradd") == []


def test_it_refuses_an_existing_account_that_is_not_a_system_account(rig: Rig) -> None:
    (rig.state / "user-seeingmon").write_text("", encoding="utf-8")
    (rig.state / "uid-seeingmon").write_text("1001", encoding="utf-8")
    result = rig.install()
    assert result.returncode == 1
    assert "is not a system account (user ID 1001)" in result.stderr


def test_it_accepts_an_existing_system_account(rig: Rig) -> None:
    (rig.state / "user-seeingmon").write_text("", encoding="utf-8")
    result = rig.install()
    assert result.returncode == 0, result.output
    assert rig.calls("useradd") == []
    assert "the account seeingmon exists" in result.stdout


def test_it_refuses_a_current_that_is_a_real_folder(rig: Rig) -> None:
    (rig.prefix / "releases").mkdir(parents=True)
    (rig.prefix / "current").mkdir()
    result = rig.install()
    assert result.returncode == 1
    assert f"{rig.prefix}/current is not a symbolic link" in result.stderr


def test_it_refuses_a_prefix_that_is_not_a_prefix(rig: Rig) -> None:
    rig.prefix.mkdir(parents=True)
    (rig.prefix / "something").write_text("not ours", encoding="utf-8")
    result = rig.install()
    assert result.returncode == 1
    assert "is not empty and has no releases folder" in result.stderr
    assert (rig.prefix / "something").is_file()


@pytest.mark.parametrize("which", ["prefix", "data_dir", "config_dir"])
def test_it_refuses_a_path_that_is_a_file(rig: Rig, which: str) -> None:
    path = getattr(rig, which)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("a file", encoding="utf-8")
    result = rig.install()
    assert result.returncode == 1
    assert "exists and is not a directory" in result.stderr


def test_it_needs_a_whole_copy_of_the_deploy_folder(rig: Rig, tmp_path: Path) -> None:
    copy = tmp_path / "deploy-copy"
    shutil.copytree(DEPLOY, copy)
    shutil.rmtree(copy / "udev")
    result = run_script(copy / "install.sh", rig.arguments(), env=rig.env())
    assert result.returncode == 1
    assert "lacks udev/99-seeingmon-asi.rules" in result.stderr
    assert_untouched(rig)


# --- Failures: the installer cleans up what it started -------------------------------------------


@pytest.mark.parametrize(
    "variable", ["FAKE_PIP_REQUIREMENTS_STATUS", "FAKE_PIP_WHEEL_STATUS", "FAKE_PIP_CHECK_STATUS"]
)
def test_a_failing_pip_leaves_no_half_release_and_no_switch(rig: Rig, variable: str) -> None:
    result = rig.install_env({variable: "1"})
    assert result.returncode != 0
    assert rig.releases == []
    assert rig.link("current") is None
    assert not rig.etc("systemd/system/seeingmon-core.service").exists()


def test_a_failing_pip_check_names_the_cause(rig: Rig) -> None:
    result = rig.install_env({"FAKE_PIP_CHECK_STATUS": "1"})
    assert "do not satisfy each other" in result.stderr


def test_a_failing_pip_keeps_the_release_that_runs(rig: Rig) -> None:
    rig.install()
    first = release_id(rig)
    rig.new_release("broken")
    result = rig.install_env({"FAKE_PIP_WHEEL_STATUS": "1"})
    assert result.returncode != 0
    assert rig.releases == [first]
    assert rig.link("current") == f"releases/{first}"


def test_a_service_that_does_not_start_is_reported_with_the_way_back(rig: Rig) -> None:
    result = rig.install_env({"FAIL_UNITS": "seeingmon-core.service"})
    assert result.returncode == 1
    assert "these units are not active: seeingmon-core.service" in result.stderr
    assert "journalctl -u seeingmon-core.service -n 50" in result.stderr
    assert f"{rig.prefix}/bin/rollback.sh --prefix {rig.prefix}" in result.stderr
    assert "Summary" in result.stdout  # the summary still prints


# --- The wrapper --------------------------------------------------------------------------------


def test_the_wrapper_runs_seeingmon_in_the_config_folder_with_the_key(rig: Rig) -> None:
    rig.install()
    result = run_script(
        rig.prefix / "bin" / "seeingmon",
        ["burst", "--help"],
        env=rig.env(FAKE_USER_NAME="seeingmon"),
    )
    assert result.returncode == 0, result.output
    lines = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert Path(lines["cwd"]).resolve() == rig.config_dir.resolve()
    assert lines["credentials"] == f"{rig.config_dir}/credentials"
    assert lines["data"] == str(rig.data_dir)
    assert lines["arguments"] == "burst --help"


def test_the_wrapper_switches_to_the_service_user_with_sudo(rig: Rig) -> None:
    rig.install()
    rig.stubs.add("sudo")
    wrapper = rig.prefix / "bin" / "seeingmon"
    result = run_script(wrapper, ["--version"], env=rig.env(FAKE_USER_NAME="operator"))
    assert result.returncode == 0, result.output
    assert rig.calls("sudo") == [f"[-u] [seeingmon] [--] [{wrapper}] [--version]"]


# --- The rollback -------------------------------------------------------------------------------


def test_the_rollback_points_current_at_the_previous_release_and_restarts(rig: Rig) -> None:
    first, second = install_releases(rig, 2)
    rig.clear_calls()
    result = rig.rollback()
    assert result.returncode == 0, result.output
    assert rig.link("current") == f"releases/{first}"
    assert rig.link("previous") == f"releases/{second}"
    assert "[restart] [seeingmon.target]" in rig.calls("systemctl")
    assert f"the services run release releases/{first}" in result.stdout
    assert not list(rig.prefix.glob("*.new"))


def test_a_second_rollback_goes_forward_again(rig: Rig) -> None:
    first, second = install_releases(rig, 2)
    rig.rollback()
    rig.rollback()
    assert rig.link("current") == f"releases/{second}"
    assert rig.link("previous") == f"releases/{first}"


def test_the_rollback_can_leave_the_services_alone(rig: Rig) -> None:
    first, _ = install_releases(rig, 2)
    rig.clear_calls()
    result = rig.rollback("--no-restart")
    assert result.returncode == 0, result.output
    assert rig.link("current") == f"releases/{first}"
    assert rig.calls("systemctl") == []
    assert "restart them with: systemctl restart seeingmon.target" in result.stdout


def test_a_rollback_dry_run_changes_nothing(rig: Rig) -> None:
    first, second = install_releases(rig, 2)
    rig.clear_calls()
    result = rig.rollback_env({"PATH": "/usr/bin:/bin"}, "--dry-run")  # no stubs, so no root
    assert result.returncode == 0, result.output
    assert f"the current release is releases/{second}" in result.stdout
    assert f"the previous release is releases/{first}" in result.stdout
    assert "dry run" in result.stdout
    assert rig.link("current") == f"releases/{second}"
    assert rig.stubs.calls() == []


def test_the_rollback_needs_a_previous_release(rig: Rig) -> None:
    rig.install()
    result = rig.rollback()
    assert result.returncode == 1
    assert "there is no previous release" in result.stderr


def test_the_rollback_needs_an_installation(rig: Rig) -> None:
    result = rig.rollback()
    assert result.returncode == 1
    assert "is not a symbolic link" in result.stderr


def test_the_rollback_refuses_a_previous_release_that_is_unfinished(rig: Rig) -> None:
    first, _ = install_releases(rig, 2)
    (rig.prefix / "releases" / first / ".installed").unlink()
    result = rig.rollback()
    assert result.returncode == 1
    assert "is missing or unfinished" in result.stderr
    assert rig.link("current") != f"releases/{first}"


def test_the_rollback_needs_root(rig: Rig) -> None:
    install_releases(rig, 2)
    result = rig.rollback_env({"FAKE_UID": "1000"})
    assert result.returncode == 1
    assert "run the script as root" in result.stderr


def test_the_rollback_reports_a_service_that_does_not_come_back(rig: Rig) -> None:
    install_releases(rig, 2)
    result = rig.rollback_env({"FAIL_UNITS": "seeingmon-web.service"})
    assert result.returncode == 1
    assert "not active after the rollback: seeingmon-web.service" in result.stderr


def test_the_rollback_checks_its_command_line(rig: Rig) -> None:
    missing = run_script(DEPLOY / "rollback.sh", [], env=rig.env())
    assert missing.returncode == 2
    assert "missing required parameter: --prefix" in missing.stderr
    relative = run_script(DEPLOY / "rollback.sh", ["--prefix", "opt/x"], env=rig.env())
    assert relative.returncode == 2
    assert "--prefix needs an absolute path" in relative.stderr
    unknown = run_script(DEPLOY / "rollback.sh", ["--prefix", "/opt/x", "--bogus"], env=rig.env())
    assert unknown.returncode == 2
    assert "unknown option: --bogus" in unknown.stderr


# --- One run with a real virtual environment ----------------------------------------------------


def make_wheel(path: Path, name: str, version: str, files: dict[str, str], entry: str = "") -> None:
    """Write a minimal wheel that pip accepts."""
    dist_info = f"{name}-{version}.dist-info"
    members = dict(files)
    members[f"{dist_info}/METADATA"] = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    members[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    if entry:
        members[f"{dist_info}/entry_points.txt"] = f"[console_scripts]\n{entry}\n"
    members[f"{dist_info}/RECORD"] = "".join(
        f"{member},,\n" for member in [*members, f"{dist_info}/RECORD"]
    )
    with zipfile.ZipFile(path, "w") as archive:
        for member, text in members.items():
            archive.writestr(member, text)


def real_venv_works() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import ensurepip, venv"], capture_output=True, check=False
    )
    return result.returncode == 0


@pytest.mark.skipif(not real_venv_works(), reason="this Python cannot make a virtual environment")
def test_a_real_virtual_environment_gets_the_pinned_requirements_and_the_wheel(
    rig: Rig, tmp_path: Path
) -> None:
    houses = tmp_path / "wheelhouse"
    houses.mkdir()
    dependency = houses / "dummydep-1.0-py3-none-any.whl"
    make_wheel(dependency, "dummydep", "1.0", {"dummydep/__init__.py": "VALUE = 1\n"})
    wheel = rig.stage / "seeingmon-0.0.1-py3-none-any.whl"
    make_wheel(
        wheel,
        "seeingmon",
        "0.0.1",
        {
            "seeingmon/__init__.py": "",
            "seeingmon/cli.py": "def main():\n    print('seeingmon 0.0.1')\n    return 0\n",
            "seeingmon/_data/config/local.example.toml": "# a real template\n",
        },
        entry="seeingmon = seeingmon.cli:main",
    )
    digest = hashlib.sha256(dependency.read_bytes()).hexdigest()
    requirements = rig.stage / "real-requirements.txt"
    requirements.write_text(f"dummydep==1.0 \\\n    --hash=sha256:{digest}\n", encoding="utf-8")
    result = rig.install(
        python=sys.executable,
        wheel=str(wheel),
        requirements=str(requirements),
        wheelhouse=str(houses),
    )
    assert result.returncode == 0, result.output
    current = rig.prefix / "current" / "venv"
    run = subprocess.run(
        [str(current / "bin" / "seeingmon")], capture_output=True, encoding="utf-8", check=False
    )
    assert run.stdout.strip() == "seeingmon 0.0.1"
    assert list(current.glob("lib/python3*/site-packages/dummydep/__init__.py"))
    assert (rig.config_dir / "local.example.toml").read_text(
        encoding="utf-8"
    ) == "# a real template\n"
    assert rig.link("current") == f"releases/{release_id_for(rig, wheel, requirements)}"


def release_id_for(rig: Rig, wheel: Path, requirements: Path) -> str:
    result = rig.install("--dry-run", wheel=str(wheel), requirements=str(requirements))
    for line in result.stdout.splitlines():
        if line.startswith("Release "):
            return line.split()[1]
    raise AssertionError(result.output)
