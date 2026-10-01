"""Tests for `tools/check_repo.py`.

The test cases assemble their trigger strings from pieces, so this file never holds a
literal that the check itself would report.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.check_repo import DenyList, check_line, scan_commits, scan_files, scan_text

NO_DENY = DenyList()


def rules(line: str) -> set[str]:
    return {rule for rule, _ in check_line(line)}


@pytest.mark.parametrize(
    ("rule", "line"),
    [
        ("windows-path", "path = " + "C" + ":" + "\\" + "Users" + "\\" + "someone"),
        ("windows-path", "path = " + "D" + ":" + "/" + "astro" + "/" + "frames"),
        ("windows-path", "path = " + "E" + ":" + "\\\\" + "data" + "\\\\" + "x"),
        ("posix-path", "cd " + "/" + "home" + "/someone/project"),
        ("posix-path", "cd " + "/" + "Users" + "/someone/project"),
        ("posix-path", "ls " + "/" + "mnt" + "/c/"),
        ("ip-address", "host " + ".".join(["10", "1", "2", "3"]) + " is up"),
        ("ip-address", "gateway " + ".".join(["192", "168", "0", "1"]) + "."),
        ("ip-address", "the address " + ".".join(["8", "8", "4", "4"]) + " answers"),
        ("ip-address", "dns " + ".".join(["8", "8", "4", "4"]) + ":53"),
        ("ip-address", "peer " + "fe80" + "::" + "1ff:fe23:4567:890a"),
        ("mac-address", "mac " + ":".join(["aa", "bb", "cc", "dd", "ee", "ff"])),
        ("hostname", "ssh target raspberrypi" + ".local"),
        ("hostname", "name = 'seeing-pi" + ".lan'"),
        ("hostname", "allowed = seeing-pi" + ".tailnet" + ".ts" + ".net"),
        ("hostname", "tailnet is tail1234" + ".ts" + ".net."),
        ("ip-address", "peer " + ".".join(["100", "101", "102", "103"])),
        ("ip-address", "peer " + "fd7a" + ":115c:a1e0::1234:5678"),
        ("hostname", "url = http://" + "pi" + ":8080/"),
        ("url-credentials", "url = https://" + "user:hunter2@example.org/path"),
        ("hostname", "run: ssh " + "pi@" + "seeingpi" + " uptime"),
        ("serial-number", "camera serial number: " + "AB12CD34"),
        ("serial-number", "S/N " + "1234567"),
        ("serial-number", "id = " + "0123456789ab" + "cdef"),
    ],
)
def test_rule_fires(rule: str, line: str) -> None:
    assert rule in rules(line)


@pytest.mark.parametrize(
    "line",
    [
        "bind to 127.0.0.1 or 0.0.0.0, or use 192.0.2.10 in examples",
        "documentation range 198.51.100.7 and 203.0.113.9",
        "loopback ::1 and documentation 2001:db8::5",
        "url = http://localhost:8000/api and https://example.org/a",
        "ssh user@example.org or ssh user@host or ssh user@$TARGET",
        "settings.local.json and config.local.toml and *.local.*",
        "x = values[::2]",
        "label:\\ntext and a time 12:34:56 on 2026-10-01T12:34:56Z",
        "serial: str and serial_number: str and the serial number is stored locally",
        "version 3.13.5 and 1.41 and 0.97.1",
        "upstream is at 2.2.4.2 (August 2026) and the SDK reports 1.41.2.0",
        "see https://sources.debian.org/src/libasi/1.27%2B20221218230335-2/debian/copyright/",
        "a git hash 3d3c42e5aac5ba805825da76410c181273ba90b1 and 0123456789abcdef0123456789abcdef",
        "an octet above 255, as in 999.1.1.1, is not an address",
        "The camera sits behind a plastic dome. Use `local/config.toml` for site values.",
    ],
)
def test_clean_lines_pass(line: str) -> None:
    assert rules(line) == set()


def test_allow_marker_suppresses_a_finding() -> None:
    line = "gateway " + ".".join(["192", "168", "0", "1"]) + "  # repo-check: allow"
    assert list(scan_text("a.txt", line, NO_DENY)) == []
    assert list(scan_text("a.py", "x = '" + line + "'", NO_DENY, python=True)) == []


def test_findings_never_contain_the_matched_text() -> None:
    secret = ".".join(["10", "9", "8", "7"])
    findings = list(scan_text("a.txt", f"host {secret}", NO_DENY))
    assert findings
    assert all(secret not in str(finding) for finding in findings)


def test_python_mode_reads_strings_and_comments_only() -> None:
    host = "raspberrypi" + ".local"
    source = "import threading\nlocal = threading" + ".local()\n" + f"HOST = '{host}'  # note\n"
    findings = list(scan_text("a.py", source, NO_DENY, python=True))
    assert [(f.line, f.rule) for f in findings] == [(3, "hostname")]
    # Plain-text mode also reads the identifier on line 2.
    assert {f.line for f in scan_text("a.txt", source, NO_DENY)} == {2, 3}


def test_python_mode_reports_the_line_inside_a_docstring() -> None:
    source = (
        'def f():\n    """First line.\n\n    gateway '
        + ".".join(["10", "0", "0", "9"])
        + '\n    """\n'
    )
    findings = list(scan_text("a.py", source, NO_DENY, python=True))
    assert [f.line for f in findings] == [4]


def test_python_mode_falls_back_to_lines_on_a_tokenizer_error() -> None:
    source = "x = (\n# " + ".".join(["10", "0", "0", "9"]) + "\n"
    assert [f.rule for f in scan_text("a.py", source, NO_DENY, python=True)] == ["ip-address"]


def test_deny_list_matches_substrings_and_patterns(tmp_path: Path) -> None:
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "repo-check-deny.txt").write_text(
        "# comment\n\nNeedleName\nre:site-\\d{3}\n", encoding="utf-8"
    )
    deny = DenyList.load(tmp_path)
    text = "first needlename line\nsecond site-042 line\nthird line is fine\n"
    findings = list(scan_text("a.txt", text, deny))
    assert [(f.line, f.rule) for f in findings] == [(1, "deny-list"), (2, "deny-list")]


def test_missing_deny_file_means_no_deny_list(tmp_path: Path) -> None:
    assert DenyList.load(tmp_path) == DenyList()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "--quiet")
    return tmp_path


def test_scan_files_reads_tracked_files_only(repo: Path) -> None:
    leak = "gateway " + ".".join(["10", "1", "1", "1"])
    (repo / "tracked.txt").write_text(leak + "\n", encoding="utf-8")
    (repo / "untracked.txt").write_text(leak + "\n", encoding="utf-8")
    (repo / "image.png").write_bytes(b"\x89PNG\0\0" + leak.encode())
    git(repo, "add", "tracked.txt", "image.png")
    findings = list(scan_files(repo, staged=False))
    assert [f.where for f in findings] == ["tracked.txt"]


def test_scan_files_staged_reads_the_index(repo: Path) -> None:
    leak = "gateway " + ".".join(["10", "1", "1", "1"])
    (repo / "a.txt").write_text("clean\n", encoding="utf-8")
    git(repo, "add", "a.txt")
    git(repo, "commit", "--quiet", "-m", "Add a file")
    (repo / "a.txt").write_text(leak + "\n", encoding="utf-8")
    assert list(scan_files(repo, staged=True)) == []
    git(repo, "add", "a.txt")
    assert [f.rule for f in scan_files(repo, staged=True)] == ["ip-address"]


def test_scan_commits_flags_co_authors_and_leaks(repo: Path) -> None:
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    git(repo, "add", "a.txt")
    git(repo, "commit", "--quiet", "-m", "Add a file")
    first = git(repo, "rev-parse", "HEAD")
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    git(repo, "add", "b.txt")
    message = "Add b\n\nCo-" + "Authored-By: Someone <someone@example.org>\n"
    git(repo, "commit", "--quiet", "-m", message)
    (repo / "c.txt").write_text("c\n", encoding="utf-8")
    git(repo, "add", "c.txt")
    git(repo, "commit", "--quiet", "-m", "Copy from " + "/" + "home" + "/someone/data")
    rules_found = sorted(f.rule for f in scan_commits(repo, f"{first}..HEAD"))
    assert rules_found == ["co-author", "posix-path"]
    assert list(scan_commits(repo, f"{first}..{first}")) == []


def test_scan_commits_with_an_unknown_start_scans_recent_commits(repo: Path) -> None:
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    git(repo, "add", "a.txt")
    git(repo, "commit", "--quiet", "-m", "Add a " + "/" + "home" + "/someone/x")
    head = git(repo, "rev-parse", "HEAD")
    findings = list(scan_commits(repo, f"{'0' * 40}..{head}"))
    assert [f.rule for f in findings] == ["posix-path"]


def test_the_repository_itself_is_clean(repo_root: Path) -> None:
    if not (repo_root / ".git").exists():
        pytest.skip("not a git checkout")
    findings = [str(f) for f in scan_files(repo_root, staged=False)]
    assert findings == []
