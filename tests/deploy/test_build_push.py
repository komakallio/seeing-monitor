"""`deploy/build.sh` and `deploy/push.sh`, run with stubs for `uv`, `ssh`, and `tar`.

The scripts run on the development machine. The tests run them under bash with a stub `uv` that
writes a wheel and a requirements file, and a stub `ssh` that runs the remote commands in a local
folder and never starts the installer. Linux only.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.deploy.scripts import DEPLOY, Result, Stubs, linux_only, run_script

pytestmark = linux_only

BUILD = DEPLOY / "build.sh"
PUSH = DEPLOY / "push.sh"
WHEEL = "seeingmon-0.1.0.dev0-py3-none-any.whl"

UV_BODY = r"""
case "$1" in
  build)
    out=''
    while [ $# -gt 0 ]; do
      if [ "$1" = --out-dir ]; then out=$2; fi
      shift
    done
    : > "$out/seeingmon-0.1.0.dev0-py3-none-any.whl"
    ;;
  export)
    out=''
    while [ $# -gt 0 ]; do
      if [ "$1" = --output-file ]; then out=$2; fi
      shift
    done
    printf 'numpy==2.0 \\\n    --hash=sha256:00\n' > "$out"
    ;;
esac
"""


@pytest.fixture
def stubs(tmp_path: Path) -> Stubs:
    return Stubs(tmp_path)


# --- build.sh ---------------------------------------------------------------------------------


def test_build_without_parameters_names_the_missing_one(stubs: Stubs) -> None:
    result = run_script(BUILD, [], env=stubs.env())
    assert result.returncode == 2
    assert "missing required parameter: --out-dir" in result.stderr


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--bogus"], "unknown option: --bogus"),
        (["--out-dir"], "the option --out-dir needs a value"),
        (["--out-dir", "--dry-run"], "needs a value, not the option --dry-run"),
        (["--out-dir", "x", "--python"], "the option --python needs a value"),
    ],
)
def test_build_reports_a_bad_command_line(stubs: Stubs, arguments: list[str], message: str) -> None:
    result = run_script(BUILD, arguments, env=stubs.env())
    assert result.returncode == 2
    assert message in result.stderr
    assert 'Run "build.sh --help"' in result.stderr


def test_build_help_lists_every_option(stubs: Stubs) -> None:
    result = run_script(BUILD, ["--help"], env=stubs.env())
    assert result.returncode == 0
    for option in ("--out-dir", "--python", "--dry-run"):
        assert option in result.stdout


def test_build_dry_run_prints_the_commands_and_runs_none(stubs: Stubs, tmp_path: Path) -> None:
    stubs.add("uv", "exit 99\n")
    out = tmp_path / "out"
    result = run_script(BUILD, ["--out-dir", str(out), "--dry-run"], env=stubs.env())
    assert result.returncode == 0, result.output
    assert not out.exists()
    assert stubs.calls() == []
    assert f"+ uv build --wheel --out-dir {out} " in result.stdout
    export = next(line for line in result.stdout.splitlines() if line.startswith("+ uv export"))
    for flag in ("--locked", "--no-dev", "--no-emit-project", "--all-extras", "--no-header"):
        assert flag in export
    assert f"--output-file {out}/requirements.txt" in export


def test_build_dry_run_needs_no_uv(stubs: Stubs, tmp_path: Path) -> None:
    result = run_script(BUILD, ["--out-dir", str(tmp_path / "out"), "--dry-run"], env=stubs.env())
    assert result.returncode == 0, result.output


def test_build_runs_uv_and_reports_the_files(stubs: Stubs, tmp_path: Path) -> None:
    stubs.add("uv", UV_BODY)
    out = tmp_path / "dist"
    arguments = ["--out-dir", str(out), "--python", "/usr/bin/python3"]
    result = run_script(BUILD, arguments, env=stubs.env())
    assert result.returncode == 0, result.output
    assert (out / WHEEL).is_file()
    assert "--hash=sha256:" in (out / "requirements.txt").read_text(encoding="utf-8")
    assert f"Wheel:         {out / WHEEL}" in result.stdout
    build, export = stubs.called("uv")
    assert build.startswith(
        f"[build] [--wheel] [--out-dir] [{out}] [--python] [/usr/bin/python3] ["
    )
    assert "[--locked] [--no-dev] [--no-emit-project] [--all-extras]" in export
    assert "[--format] [requirements-txt]" in export
    assert "[--python] [/usr/bin/python3]" in export


def test_build_never_lets_uv_download_a_python(stubs: Stubs, tmp_path: Path) -> None:
    marker = tmp_path / "downloads.txt"
    stubs.add("uv", UV_BODY + f'echo "$UV_PYTHON_DOWNLOADS" > "{marker}"\n')
    result = run_script(BUILD, ["--out-dir", str(tmp_path / "o")], env=stubs.env())
    assert result.returncode == 0, result.output
    assert marker.read_text(encoding="utf-8").strip() == "never"


def test_build_keeps_a_setting_of_the_caller(stubs: Stubs, tmp_path: Path) -> None:
    marker = tmp_path / "downloads.txt"
    stubs.add("uv", UV_BODY + f'echo "$UV_PYTHON_DOWNLOADS" > "{marker}"\n')
    env = stubs.env(UV_PYTHON_DOWNLOADS="automatic")
    result = run_script(BUILD, ["--out-dir", str(tmp_path / "o")], env=env)
    assert result.returncode == 0, result.output
    assert marker.read_text(encoding="utf-8").strip() == "automatic"


def test_build_replaces_the_files_of_an_earlier_build(stubs: Stubs, tmp_path: Path) -> None:
    stubs.add("uv", UV_BODY)
    out = tmp_path / "dist"
    out.mkdir()
    (out / "seeingmon-0.0.9-py3-none-any.whl").write_text("old", encoding="utf-8")
    result = run_script(BUILD, ["--out-dir", str(out)], env=stubs.env())
    assert result.returncode == 0, result.output
    assert sorted(path.name for path in out.glob("*.whl")) == [WHEEL]


def test_build_stops_when_uv_fails(stubs: Stubs, tmp_path: Path) -> None:
    stubs.add("uv", "exit 3\n")
    result = run_script(BUILD, ["--out-dir", str(tmp_path / "o")], env=stubs.env())
    assert result.returncode == 3


def test_build_refuses_requirements_without_hashes(stubs: Stubs, tmp_path: Path) -> None:
    stubs.add("uv", UV_BODY.replace("--hash=sha256:00", ""))
    result = run_script(BUILD, ["--out-dir", str(tmp_path / "o")], env=stubs.env())
    assert result.returncode == 1
    assert "has no hashes" in result.stderr


def test_build_asks_for_uv_when_it_is_missing(tmp_path: Path) -> None:
    if shutil.which("uv", path="/usr/bin:/bin") is not None:
        pytest.skip("uv is installed in a system folder")
    env = {"PATH": "/usr/bin:/bin"}
    result = run_script(BUILD, ["--out-dir", str(tmp_path / "o")], env=env)
    assert result.returncode == 1
    assert "uv is not installed" in result.stderr


# --- push.sh: the command line ----------------------------------------------------------------

REQUIRED = {
    "--host": "pi.example.org",
    "--user": "operator",
    "--prefix": "/opt/seeingmon",
    "--service-user": "seeingmon",
    "--data-dir": "/srv/seeing-data",
    "--config-dir": "/etc/seeingmon",
    "--time-source": "time.example.org",
}


def push_arguments(**changes: str | None) -> list[str]:
    """The required arguments, with some replaced (`None` removes one). Keys use underscores."""
    values = dict(REQUIRED)
    for key, value in changes.items():
        option = "--" + key.replace("_", "-")
        if value is None:
            del values[option]
        else:
            values[option] = value
    return [item for option, value in values.items() for item in (option, value)]


@pytest.mark.parametrize("option", sorted(REQUIRED))
def test_push_names_the_missing_required_parameter(stubs: Stubs, option: str) -> None:
    arguments = push_arguments(**{option[2:].replace("-", "_"): None})
    result = run_script(PUSH, arguments, env=stubs.env())
    assert result.returncode == 2
    assert f"missing required parameter: {option}" in result.stderr


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"host": "-oProxyCommand=x"}, "--host needs a host name or address"),
        ({"host": "pi;reboot"}, "--host needs a host name or address"),
        ({"user": "Operator"}, "--user needs a plain account name"),
        ({"service_user": "seeing mon"}, "--service-user needs a plain account name"),
        ({"prefix": "opt/seeingmon"}, "--prefix needs an absolute path"),
        ({"prefix": "/opt/see ingmon"}, "--prefix needs an absolute path"),
        ({"prefix": "/opt/../etc"}, "--prefix must not contain '..'"),
        ({"data_dir": "/srv/data/"}, "--data-dir must not end with a slash"),
        ({"time_source": "bad host"}, "--time-source needs a host name or address"),
    ],
)
def test_push_checks_the_values(stubs: Stubs, changes: dict[str, str], message: str) -> None:
    result = run_script(PUSH, push_arguments(**changes), env=stubs.env())
    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--local-config", "MISSING"], "--local-config names a file that does not exist"),
        (["--env-file", "MISSING"], "--env-file names a file that does not exist"),
        (["--connection-key-file", "MISSING"], "--connection-key-file names a file"),
        (["--identity", "MISSING"], "--identity names a file that does not exist"),
        (["--sdk-archive", "THIS"], "--sdk-archive needs --sdk-sha256"),
        (["--sdk-sha256", "0" * 64], "--sdk-sha256 needs --sdk-archive"),
        (["--sdk-archive", "THIS", "--sdk-sha256", "XYZ"], "needs 64 lowercase hex digits"),
        (["--port", "ssh"], "--port needs a number"),
        (["--ssh-option", "oops"], "--ssh-option needs NAME=VALUE"),
        (["--no-time-config"], "exclude each other"),
    ],
)
def test_push_checks_the_files_and_the_combinations(
    stubs: Stubs, tmp_path: Path, extra: list[str], message: str
) -> None:
    words = [str(tmp_path / "missing") if w == "MISSING" else w for w in extra]
    words = [__file__ if w == "THIS" else w for w in words]
    result = run_script(PUSH, [*push_arguments(), *words], env=stubs.env())
    assert result.returncode == 2
    assert message in result.stderr


def test_push_needs_a_time_source_or_no_time_config(stubs: Stubs) -> None:
    result = run_script(PUSH, push_arguments(time_source=None), env=stubs.env())
    assert result.returncode == 2
    assert "missing required parameter: --time-source (or --no-time-config)" in result.stderr
    arguments = [*push_arguments(time_source=None), "--no-time-config", "--dry-run"]
    result = run_script(PUSH, arguments, env=stubs.env())
    assert result.returncode == 0, result.output
    assert "--no-time-config" in result.stdout


def test_push_rejects_a_value_that_looks_like_an_option(stubs: Stubs) -> None:
    result = run_script(PUSH, push_arguments(host="--dry-run"), env=stubs.env())
    assert result.returncode == 2
    assert "needs a value, not the option --dry-run" in result.stderr


def test_push_passes_an_installer_argument_that_starts_with_dashes(stubs: Stubs) -> None:
    arguments = [
        *push_arguments(),
        "--installer-arg",
        "--keep",
        "--installer-arg",
        "3",
        "--dry-run",
    ]
    result = run_script(PUSH, arguments, env=stubs.env())
    assert result.returncode == 0, result.output
    assert "--keep 3" in result.stdout


# --- push.sh: a dry run -----------------------------------------------------------------------


def test_push_dry_run_prints_every_command_and_runs_none(stubs: Stubs, tmp_path: Path) -> None:
    for name in ("uv", "ssh", "tar", "sudo"):
        stubs.add(name, "exit 99\n")
    local_config = tmp_path / "config.toml"
    local_config.write_text("station_id = 'x'\n", encoding="utf-8")
    arguments = [
        *push_arguments(),
        "--local-config",
        str(local_config),
        "--time-source",
        "192.0.2.10",
        "--port",
        "2222",
        "--ssh-option",
        "StrictHostKeyChecking=accept-new",
        "--installer-arg",
        "--dry-run",
        "--dry-run",
    ]
    result = run_script(PUSH, arguments, env=stubs.env(TMPDIR=str(tmp_path)))
    assert result.returncode == 0, result.output
    assert stubs.calls() == []
    lines = result.stdout.splitlines()
    assert any(line.startswith("+ uv build --wheel") for line in lines)
    assert any(line.startswith("+ uv export") for line in lines)
    assert any(line.endswith("-- operator@pi.example.org 'mktemp -d'") for line in lines)
    assert any(line.startswith("+ tar -C ") and "-cf - . | ssh " in line for line in lines)
    install = next(line for line in lines if "install.sh" in line)
    expected = (
        "+ ssh -o ConnectTimeout=15 -p 2222 -o StrictHostKeyChecking=accept-new -- "
        "operator@pi.example.org 'sudo bash /tmp/seeingmon-install.XXXXXX/deploy/install.sh "
        "--prefix /opt/seeingmon --user seeingmon --data-dir /srv/seeing-data "
        "--config-dir /etc/seeingmon --wheel /tmp/seeingmon-install.XXXXXX/seeingmon-VERSION"
    )
    assert install.startswith(expected), install
    for word in (
        "--requirements /tmp/seeingmon-install.XXXXXX/requirements.txt",
        "--local-config /tmp/seeingmon-install.XXXXXX/local-config.toml",
        "--time-source time.example.org --time-source 192.0.2.10",
        "--dry-run'",
    ):
        assert word in install, word
    assert list(tmp_path.glob("tmp.*")) == []  # a dry run makes no temporary directory


def test_push_does_not_use_sudo_for_root(stubs: Stubs) -> None:
    result = run_script(PUSH, [*push_arguments(user="root"), "--dry-run"], env=stubs.env())
    assert result.returncode == 0, result.output
    install = next(line for line in result.stdout.splitlines() if "install.sh" in line)
    assert "'sudo" not in install
    assert "root@pi.example.org 'bash /tmp/" in install


def test_push_dry_run_uses_a_finished_build(stubs: Stubs, tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / WHEEL).write_text("wheel", encoding="utf-8")
    (dist / "requirements.txt").write_text("x==1 --hash=sha256:00\n", encoding="utf-8")
    arguments = [*push_arguments(), "--dist-dir", str(dist), "--dry-run"]
    result = run_script(PUSH, arguments, env=stubs.env())
    assert result.returncode == 0, result.output
    assert f"using the build in {dist}" in result.stdout
    assert f"/{WHEEL}" in result.stdout
    assert "uv build" not in result.stdout


def test_push_needs_the_files_of_a_finished_build(stubs: Stubs, tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    arguments = [*push_arguments(), "--dist-dir", str(dist), "--dry-run"]
    result = run_script(PUSH, arguments, env=stubs.env())
    assert result.returncode == 1
    assert "expected one seeingmon wheel" in result.stderr


# --- push.sh: a run against a stub ssh --------------------------------------------------------

SSH_BODY = r"""
# A stub ssh: run the remote commands in this folder, and never start the installer.
while [ "$1" != "--" ]; do shift; done
shift
shift
command=$*
case "$command" in
  "mktemp -d") mktemp -d "$REMOTE_ROOT/stage.XXXXXX" ;;
  tar\ *) sh -c "$command" ;;
  rm\ *) sh -c "$command" ;;
  *install.sh*)
    stage=$(printf '%s' "$command" | sed -n 's|.*bash \(/[^ ]*\)/deploy/install.sh.*|\1|p')
    printf 'INSTALL %s\n' "$command" >> "$CALLS"
    (cd "$stage" && find . -type f | sort | sed 's|^|STAGED |' >> "$CALLS")
    exit "${INSTALL_STATUS:-0}"
    ;;
esac
"""


def real_push(
    stubs: Stubs, tmp_path: Path, *extra: str, install_status: int = 0
) -> tuple[Result, Path, Path]:
    """Run push.sh with stubs. Returns the result, the fake remote folder, and the local scratch."""
    stubs.add("uv", UV_BODY)
    stubs.add("ssh", SSH_BODY)
    remote = tmp_path / "remote"
    remote.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = stubs.env(
        REMOTE_ROOT=str(remote),
        CALLS=str(stubs.log),
        TMPDIR=str(scratch),
        INSTALL_STATUS=str(install_status),
    )
    return run_script(PUSH, [*push_arguments(), *extra], env=env), remote, scratch


def test_push_copies_the_release_and_runs_the_installer(stubs: Stubs, tmp_path: Path) -> None:
    key = tmp_path / "key.txt"
    key.write_text("a-connection-key-for-the-tests\n", encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text("station_id = 'x'\n", encoding="utf-8")
    sdk = tmp_path / "sdk.tar.bz2"
    sdk.write_bytes(b"not really an archive")
    result, remote, _ = real_push(
        stubs,
        tmp_path,
        "--local-config",
        str(config),
        "--connection-key-file",
        str(key),
        "--sdk-archive",
        str(sdk),
        "--sdk-sha256",
        "a" * 64,
    )
    assert result.returncode == 0, result.output
    calls = stubs.calls()
    staged = {line[len("STAGED ./") :] for line in calls if line.startswith("STAGED ")}
    assert {
        "deploy/install.sh",
        "deploy/build.sh",
        "deploy/systemd/seeingmon-core.service",
        "deploy/udev/99-seeingmon-asi.rules",
        WHEEL,
        "requirements.txt",
        "local-config.toml",
        "connection-key",
        "sdk-sdk.tar.bz2",
    } <= staged
    install = next(line for line in calls if line.startswith("INSTALL "))
    stage = install.split("bash ", 1)[1].split("/deploy/install.sh")[0]
    assert stage.startswith(str(remote))
    for word in (
        f"--wheel {stage}/{WHEEL}",
        f"--requirements {stage}/requirements.txt",
        f"--local-config {stage}/local-config.toml",
        f"--connection-key-file {stage}/connection-key",
        f"--sdk-archive {stage}/sdk-sdk.tar.bz2 --sdk-sha256 {'a' * 64}",
    ):
        assert word in install, word


def test_push_removes_the_staging_directories(stubs: Stubs, tmp_path: Path) -> None:
    result, remote, scratch = real_push(stubs, tmp_path)
    assert result.returncode == 0, result.output
    assert list(remote.iterdir()) == []
    assert list(scratch.iterdir()) == []


def test_push_passes_the_exit_status_of_the_installer_and_still_cleans_up(
    stubs: Stubs, tmp_path: Path
) -> None:
    result, remote, scratch = real_push(stubs, tmp_path, install_status=7)
    assert result.returncode == 7
    assert list(remote.iterdir()) == []
    assert list(scratch.iterdir()) == []


def test_push_keeps_the_staging_directory_on_request(stubs: Stubs, tmp_path: Path) -> None:
    result, remote, _ = real_push(stubs, tmp_path, "--keep-stage")
    assert result.returncode == 0, result.output
    kept = list(remote.iterdir())
    assert len(kept) == 1
    assert "left the staging directory" in result.stderr
    assert (kept[0] / "deploy" / "install.sh").is_file()
