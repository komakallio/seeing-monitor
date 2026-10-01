"""The whole-repository lint: the files that ship, a copy with one fault, and the command line.

`shellcheck` is an external tool. The `shellcheck-py` package has wheels for Windows x64 and Linux
x64, and none for Linux arm64, so the tests that need the binary skip where it is missing. On the
platforms with a wheel, CI requires it, so the pass cannot disappear unnoticed.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from tests.deploy.helpers import copy_deploy, mutate, rules
from tools.lint_deploy import (
    LintError,
    executable_bit_findings,
    find_shellcheck,
    lint_repo,
    main,
    run_bash_syntax,
    run_shellcheck,
)

ROOT = Path(__file__).resolve().parents[2]
CLEAN_SCRIPT = '#!/usr/bin/env bash\nset -euo pipefail\necho "hello"\n'
# SC2086 (an unquoted variable) and SC2034 (an unused variable) are shellcheck findings that the
# structural checks do not look for.
SHELLCHECK_BAD = "#!/usr/bin/env bash\nset -euo pipefail\nunused=1\nname=$1\necho $name\n"


def has_wheel() -> bool:
    """Whether the `shellcheck-py` package ships a wheel for this platform."""
    machine = platform.machine().lower()
    system = platform.system()
    if system == "Windows":
        return machine in {"amd64", "x86_64"}
    return system == "Linux" and machine == "x86_64"


needs_shellcheck = pytest.mark.skipif(
    find_shellcheck() is None,
    reason="shellcheck is not installed (the shellcheck-py package has no Linux arm64 wheel)",
)
needs_bash = pytest.mark.skipif(
    os.name != "posix", reason="the bash -n fallback runs on POSIX systems only"
)


def fake_repo(tmp_path: Path) -> Path:
    return copy_deploy(ROOT, tmp_path / "repo")


# --- The files that ship ----------------------------------------------------------------------


def test_the_shipped_files_pass_every_check() -> None:
    report = lint_repo(ROOT)
    assert report.findings == [], "\n".join(str(finding) for finding in report.findings)
    assert report.clean


def test_the_report_says_which_shell_tool_ran() -> None:
    notes = " ".join(lint_repo(ROOT).notes)
    assert "shellcheck ran" in notes or "shellcheck is not available" in notes


@pytest.mark.skipif(not os.environ.get("CI"), reason="CI sets CI, and CI must run shellcheck")
def test_ci_runs_shellcheck_wherever_a_wheel_exists() -> None:
    if not has_wheel():
        pytest.skip("shellcheck-py has no wheel for this platform")
    assert find_shellcheck() is not None, "uv sync should install shellcheck-py (dev group)"


# --- A copy with one fault --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "old", "new", "rule"),
    [
        (
            "systemd/seeingmon-web.service",
            "PrivateDevices=yes",
            "PrivateDevices=no",
            "unit-set-devices",
        ),
        ("systemd/seeingmon-core.service", "RestrictAddressFamilies=", "#", "unit-hardening"),
        ("udev/99-seeingmon-asi.rules", 'MODE="0660"', 'MODE="0666"', "udev-policy"),
        ("chrony/seeingmon.conf", "@TIME_SOURCES@", "makestep 1 3", "chrony-directive"),
        ("chrony/seeingmon.conf", "@TIME_SOURCES@", "@TIME@", "template-placeholder"),
        ("tmpfiles/seeingmon-usb.conf", "@USBFS_MEMORY_MB@", "lots", "tmpfiles-value"),
        ("journald/seeingmon.conf", "volatile", "persistent", "journald-policy"),
        ("polkit/50-seeingmon.rules", "});", "}", "polkit-syntax"),
        ("wrapper/seeingmon.sh", "set -euo pipefail", "set -e", "shell-strict-mode"),
        (
            "wrapper/seeingmon.sh",
            'cd -- "@CONFIG_DIR@"',
            'cd -- "@CONFIG_DIR@"  # /' + "home" + "/someone",
            "shell-private-value",
        ),
        (
            "wrapper/seeingmon.sh",
            "Managed by the seeingmon installer",
            "Managed by hand",
            "template-marker",
        ),
    ],
)
def test_each_kind_of_file_is_checked(
    tmp_path: Path, path: str, old: str, new: str, rule: str
) -> None:
    repo = fake_repo(tmp_path)
    target = repo / "deploy" / path
    text = target.read_text(encoding="utf-8")
    assert old in text, f"{path} no longer contains {old!r}"
    target.write_text(text.replace(old, new), encoding="utf-8", newline="\n")  # every occurrence
    findings = lint_repo(repo, shell_tools=False).findings
    assert rule in rules(findings), [str(finding) for finding in findings]


def test_a_finding_names_the_file_and_the_line(tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    target = repo / "deploy" / "systemd" / "seeingmon-web.service"
    target.write_text(
        mutate(target.read_text(encoding="utf-8"), "ProtectSystem=strict", "ProtectSytem=strict"),
        encoding="utf-8",
        newline="\n",
    )
    findings = [
        f for f in lint_repo(repo, shell_tools=False).findings if f.rule == "unit-unknown-key"
    ]
    assert len(findings) == 1
    assert findings[0].path == "deploy/systemd/seeingmon-web.service"
    line = target.read_text(encoding="utf-8").splitlines()[findings[0].line - 1]
    assert line == "ProtectSytem=strict"


def test_crlf_line_endings_are_a_finding_in_any_file(tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    target = repo / "deploy" / "systemd" / "seeingmon.target"
    target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    assert "file-line-endings" in rules(lint_repo(repo, shell_tools=False).findings)


def test_a_file_that_is_not_utf8_is_a_finding(tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    (repo / "deploy" / "systemd" / "seeingmon.target").write_bytes(b"\xff\xfe not text")
    assert "file-encoding" in rules(lint_repo(repo, shell_tools=False).findings)


def test_a_missing_unit_is_a_finding(tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    (repo / "deploy" / "systemd" / "seeingmon-core.service").unlink()
    assert "unit-set-missing" in rules(lint_repo(repo, shell_tools=False).findings)


def test_a_repository_without_a_deploy_directory_cannot_be_linted(tmp_path: Path) -> None:
    with pytest.raises(LintError, match="deploy/"):
        lint_repo(tmp_path)


# --- The external tools -----------------------------------------------------------------------


@needs_shellcheck
def test_shellcheck_accepts_a_clean_script(tmp_path: Path) -> None:
    (tmp_path / "ok.sh").write_text(CLEAN_SCRIPT, encoding="utf-8", newline="\n")
    binary = find_shellcheck()
    assert binary is not None
    assert run_shellcheck(binary, tmp_path, ["ok.sh"]) == []


@needs_shellcheck
def test_shellcheck_findings_carry_the_file_the_line_and_the_code(tmp_path: Path) -> None:
    (tmp_path / "bad.sh").write_text(SHELLCHECK_BAD, encoding="utf-8", newline="\n")
    binary = find_shellcheck()
    assert binary is not None
    findings = run_shellcheck(binary, tmp_path, ["bad.sh"])
    assert rules(findings) == {"shellcheck"}
    assert {finding.path for finding in findings} == {"bad.sh"}
    assert any("SC2086" in finding.message for finding in findings)
    assert any(finding.line == 5 for finding in findings)


@needs_shellcheck
def test_a_shellcheck_failure_to_run_is_a_finding(tmp_path: Path) -> None:
    binary = find_shellcheck()
    assert binary is not None
    findings = run_shellcheck(binary, tmp_path, ["does-not-exist.sh"])
    assert findings
    assert rules(findings) == {"shellcheck"}


@needs_bash
def test_the_bash_fallback_finds_a_syntax_error(tmp_path: Path) -> None:
    (tmp_path / "bad.sh").write_text(
        "#!/usr/bin/env bash\nif then fi\n", encoding="utf-8", newline="\n"
    )
    (tmp_path / "ok.sh").write_text(CLEAN_SCRIPT, encoding="utf-8", newline="\n")
    findings = run_bash_syntax(tmp_path, ["bad.sh", "ok.sh"])
    assert rules(findings) == {"shell-syntax"}
    assert [finding.path for finding in findings] == ["bad.sh"]


def test_the_lint_says_when_shellcheck_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tools.lint_deploy.find_shellcheck", lambda: None)
    repo = fake_repo(tmp_path)
    report = lint_repo(repo)
    notes = " ".join(report.notes)
    assert "shellcheck is not available on this platform" in notes
    assert "Linux arm64" in notes
    if os.name == "posix":
        assert "ran bash -n" in notes


@needs_bash
def test_the_fallback_still_catches_a_syntax_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tools.lint_deploy.find_shellcheck", lambda: None)
    repo = fake_repo(tmp_path)
    (repo / "deploy" / "broken.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nif then fi\n", encoding="utf-8", newline="\n"
    )
    assert "shell-syntax" in rules(lint_repo(repo).findings)


def test_the_executable_bit_is_checked_in_the_git_index(tmp_path: Path) -> None:
    def git(*arguments: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *arguments], check=True, capture_output=True)

    try:
        git("init", "-q")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available")
    (tmp_path / "deploy").mkdir()
    (tmp_path / "deploy" / "run.sh").write_text(CLEAN_SCRIPT, encoding="utf-8", newline="\n")
    git("add", "deploy/run.sh")
    findings = executable_bit_findings(tmp_path, ["deploy/run.sh"])
    assert findings is not None
    assert rules(findings) == {"shell-not-executable"}
    git("add", "--chmod=+x", "deploy/run.sh")
    assert executable_bit_findings(tmp_path, ["deploy/run.sh"]) == []


def test_the_executable_bit_check_skips_outside_a_git_repository(tmp_path: Path) -> None:
    assert executable_bit_findings(tmp_path / "missing", ["deploy/run.sh"]) is None


# --- The command line -------------------------------------------------------------------------


def test_main_returns_zero_for_the_shipped_files(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--repo", str(ROOT), "--no-shellcheck"]) == 0
    assert "lint_deploy: clean" in capsys.readouterr().out


def test_main_returns_one_and_prints_the_findings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = fake_repo(tmp_path)
    (repo / "deploy" / "systemd" / "seeingmon-core.service").unlink()
    assert main(["--repo", str(repo), "--no-shellcheck"]) == 1
    output = capsys.readouterr().out
    assert "[unit-set-missing]" in output
    assert "finding(s)" in output


def test_main_returns_two_when_it_cannot_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--repo", str(tmp_path)]) == 2
    assert "no deploy/ directory" in capsys.readouterr().err


def test_the_command_runs_as_a_script_from_any_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "lint_deploy.py"), "--no-shellcheck"],
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "lint_deploy: clean" in result.stdout


def test_the_command_fails_on_a_fault_when_run_as_a_script(tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    (repo / "deploy" / "systemd" / "seeingmon-core.service").unlink()
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "lint_deploy.py"),
            "--repo",
            str(repo),
            "--no-shellcheck",
        ],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 1
    assert "unit-set-missing" in result.stdout
