#!/usr/bin/env python3
"""Fail when tracked files or commit messages carry values that must stay private.

The repository is public (see `CLAUDE.md`). This check looks for absolute machine paths,
IP and MAC addresses, private host names, ssh targets, serial numbers, URL credentials,
and co-author trailers. It prints the rule and the location of each finding, and it never
prints the matched text, because the CI logs of a public repository are public too.

Usage:
    python tools/check_repo.py                  every tracked file
    python tools/check_repo.py --staged         the files in the index, before a commit
    python tools/check_repo.py --commits A..B   also scan the commit messages in a range
    python tools/check_repo.py --repo PATH      act on a clone other than the working directory

Suppress a deliberate finding by putting `repo-check: allow` on the same line.

Private literals that no general rule can describe (a user name, a camera serial number,
a site name) go into `local/repo-check-deny.txt`, one per line. That file is untracked. A
plain line is a case-insensitive substring, and a line that starts with `re:` is a regular
expression.

In Python files, the rules read only strings and comments, so attribute names such as a
thread-local object never match. The deny list reads every line.
"""

from __future__ import annotations

import argparse
import io
import ipaddress
import re
import subprocess
import sys
import tokenize
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

ALLOW_MARKER = "repo-check: allow"
DENY_FILE = Path("local") / "repo-check-deny.txt"
MAX_FILE_BYTES = 2_000_000
SKIP_NAMES = frozenset({"uv.lock", ".secrets.baseline"})
SKIP_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".whl", ".ser", ".fits"}
)

_WINDOWS_PATH = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\{1,2}|/)"
    r"(?:Users\b|[^\s\\/:*?\"<>|]{2,}(?:\\{1,2}|/)[^\s\\/:*?\"<>|])"
)
_POSIX_PATH = re.compile(r"(?<![\w.:/~-])(?:/(?:home|Users)/[A-Za-z0-9_.-]+|/mnt/[a-z]/)")
_IPV4 = re.compile(r"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?!\w|\.\d)")
_IPV6_CANDIDATE = re.compile(r"(?<![\w:.])[0-9A-Fa-f:]{2,}:[0-9A-Fa-f:]*(?![\w:.])")
_MAC = re.compile(r"(?<![0-9A-Fa-f:-])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:-])")
_PRIVATE_HOST = re.compile(
    r"(?i)(?<![\w.-])[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*"
    r"\.(?:local|lan|internal|localdomain|intranet|home\.arpa)(?![\w-]|\.[a-z0-9])"
)
_URL_HOST = re.compile(
    r"(?i)\b[a-z][a-z0-9+.-]*://(?:[^/\s@]+@)?(?P<host>[A-Za-z0-9][A-Za-z0-9.-]*)(?![A-Za-z0-9.-])"
)
_URL_CREDENTIALS = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@<{$]+:[^/\s@<{$]+@")
_SSH_COMMAND = re.compile(r"(?i)\b(?:ssh|scp|sftp|rsync)\b(?P<rest>[^\n]*)")
_SSH_TARGET = re.compile(r"\b[A-Za-z0-9._-]+@(?P<host>[A-Za-z0-9][A-Za-z0-9.-]*)")
_SERIAL_LABEL = re.compile(
    r"(?i)\b(?:serial(?:\s+(?:number|no\.?|num))?|s/n)\b\s*[:=#]?\s*[\"']?"
    r"(?=[A-Za-z-]*\d)[A-Za-z0-9-]{6,}"
)
_HEX_ID = re.compile(
    r"(?<![0-9A-Fa-f])(?=[0-9A-Fa-f]*[0-9])(?=[0-9A-Fa-f]*[A-Fa-f])[0-9A-Fa-f]{16}(?![0-9A-Fa-f])"
)
_PLACEHOLDER_HOST = re.compile(
    r"(?i)^(?:(?:[a-z0-9-]+\.)*example(?:\.(?:com|org|net))?|host(?:name)?|server|remote|target)$"
)
_CO_AUTHOR = re.compile(r"(?im)^\s*co-authored-by:")
_URL_SPAN = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://\S+")
_ADDRESS_SUFFIX = re.compile(r"[:/]\d")
_ADDRESS_CONTEXT = re.compile(r"(?i)(?:\bip|addr(?:ess)?|host|server|://|@)\W{0,3}$")

_DOCUMENTATION_NETS = tuple(
    ipaddress.ip_network(net)
    for net in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "2001:db8::/32")
)
_PRIVATE_NETS = tuple(
    ipaddress.ip_network(net)
    for net in (
        "10.0.0.0/8",  # repo-check: allow
        "172.16.0.0/12",  # repo-check: allow
        "192.168.0.0/16",  # repo-check: allow
        "169.254.0.0/16",  # repo-check: allow
        "100.64.0.0/10",  # repo-check: allow
    )
)


@dataclass(frozen=True)
class Finding:
    """One problem. `where` is a path or a commit. The matched text stays out of the report."""

    where: str
    line: int
    rule: str
    hint: str

    def __str__(self) -> str:
        return f"{self.where}:{self.line}: [{self.rule}] {self.hint}"


@dataclass(frozen=True)
class DenyList:
    substrings: tuple[str, ...] = ()
    patterns: tuple[re.Pattern[str], ...] = ()

    @classmethod
    def load(cls, repo: Path) -> DenyList:
        path = repo / DENY_FILE
        if not path.is_file():
            return cls()
        substrings: list[str] = []
        patterns: list[re.Pattern[str]] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            entry = raw.strip()
            if not entry or entry.startswith("#"):
                continue
            if entry.startswith("re:"):
                patterns.append(re.compile(entry[3:], re.IGNORECASE))
            else:
                substrings.append(entry.lower())
        return cls(tuple(substrings), tuple(patterns))

    def matches(self, line: str) -> bool:
        lowered = line.lower()
        return any(s in lowered for s in self.substrings) or any(
            p.search(line) for p in self.patterns
        )


def _reportable_ipv4(match: re.Match[str], line: str) -> bool:
    """Decide whether a dotted quad is a private address worth reporting.

    Addresses in the private ranges always count. A public-looking quad can be a four-part
    version number, so it counts only when the text around it says it is an address: a port
    or prefix length after it, or a word such as "address" or "host" before it.
    """
    octets = [int(group) for group in match.groups()]
    if any(octet > 255 for octet in octets):
        return False
    address = ipaddress.IPv4Address(".".join(str(octet) for octet in octets))
    if address.is_loopback or address.is_unspecified or address == ipaddress.IPv4Address(2**32 - 1):
        return False
    if any(address in net for net in _DOCUMENTATION_NETS):
        return False
    if any(address in net for net in _PRIVATE_NETS):
        return True
    if _ADDRESS_SUFFIX.match(line, match.end()):
        return True
    return bool(_ADDRESS_CONTEXT.search(line[max(0, match.start() - 16) : match.start()]))


def _reportable_ipv6(text: str) -> bool:
    try:
        address = ipaddress.IPv6Address(text)
    except ValueError:
        return False
    if address.is_loopback or address.is_unspecified:
        return False
    return not any(address in net for net in _DOCUMENTATION_NETS)


def check_line(line: str) -> list[tuple[str, str]]:
    """Apply the general rules to one line. Returns (rule, hint) pairs."""
    found: list[tuple[str, str]] = []
    if _WINDOWS_PATH.search(line):
        found.append(("windows-path", "absolute Windows path; use a repository-relative path"))
    if _POSIX_PATH.search(line):
        found.append(("posix-path", "absolute home or mount path; use a repository-relative path"))
    if any(_reportable_ipv4(m, line) for m in _IPV4.finditer(line)):
        found.append(("ip-address", "IPv4 address; use a documentation address or a parameter"))
    if any(_reportable_ipv6(m.group()) for m in _IPV6_CANDIDATE.finditer(line)):
        found.append(("ip-address", "IPv6 address; use a documentation address or a parameter"))
    if _MAC.search(line):
        found.append(("mac-address", "MAC address; keep network details in local configuration"))
    if _PRIVATE_HOST.search(line):
        found.append(("hostname", "private host name; take host names as parameters"))
    for url in _URL_HOST.finditer(line):
        host = url.group("host")
        if "." not in host and host.lower() != "localhost" and not _PLACEHOLDER_HOST.match(host):
            found.append(("hostname", "single-label host in a URL; take host names as parameters"))
            break
    if _URL_CREDENTIALS.search(line):
        found.append(
            ("url-credentials", "credentials inside a URL; read them from the environment")
        )
    ssh = _SSH_COMMAND.search(line)
    if ssh and any(
        not _PLACEHOLDER_HOST.match(t.group("host"))
        for t in _SSH_TARGET.finditer(ssh.group("rest"))
    ):
        found.append(("hostname", "ssh target; take user and host as parameters"))
    plain = _URL_SPAN.sub(" ", line)  # URLs carry hashes and encoded characters, not serials
    if _SERIAL_LABEL.search(plain) or _HEX_ID.search(plain):
        found.append(("serial-number", "serial number or hardware ID; keep it in local files"))
    return found


def _python_fragments(text: str) -> dict[int, list[str]]:
    """Map each line number to the string and comment fragments on it."""
    wanted = {tokenize.STRING, tokenize.COMMENT}
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    if fstring_middle is not None:
        wanted.add(fstring_middle)
    fragments: dict[int, list[str]] = {}
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError):
        return {number: [line] for number, line in enumerate(text.splitlines(), start=1)}
    for token in tokens:
        if token.type not in wanted:
            continue
        for offset, part in enumerate(token.string.splitlines() or [""]):
            fragments.setdefault(token.start[0] + offset, []).append(part)
    return fragments


def scan_text(where: str, text: str, deny: DenyList, *, python: bool = False) -> Iterator[Finding]:
    """Scan the text of one file or commit message."""
    lines = text.splitlines()
    if python:
        fragments = _python_fragments(text)
    else:
        fragments = {number: [line] for number, line in enumerate(lines, start=1)}
    for number, line in enumerate(lines, start=1):
        parts = fragments.get(number, [])
        if ALLOW_MARKER in line or any(ALLOW_MARKER in part for part in parts):
            continue
        if deny.matches(line):
            yield Finding(where, number, "deny-list", "matches an entry in the local deny list")
        seen: set[str] = set()
        for part in parts:
            for rule, hint in check_line(part):
                if rule + hint not in seen:
                    seen.add(rule + hint)
                    yield Finding(where, number, rule, hint)


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return result.stdout


def _is_text(path: str, data: bytes) -> bool:
    name = Path(path)
    if name.name in SKIP_NAMES or name.suffix.lower() in SKIP_SUFFIXES:
        return False
    return len(data) <= MAX_FILE_BYTES and b"\0" not in data[:4096]


def scan_files(repo: Path, *, staged: bool) -> Iterator[Finding]:
    """Scan the tracked files (or the staged files) of a repository."""
    deny = DenyList.load(repo)
    if staged:
        names = _git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    else:
        names = _git(repo, "ls-files", "-z")
    for raw_name in names.split(b"\0"):
        if not raw_name:
            continue
        name = raw_name.decode("utf-8", errors="replace")
        if staged:
            data = _git(repo, "show", f":{name}")
        elif (repo / name).is_file():
            data = (repo / name).read_bytes()
        else:
            continue
        if not _is_text(name, data):
            continue
        text = data.decode("utf-8", errors="replace")
        yield from scan_text(name, text, deny, python=name.endswith(".py"))


def _revision_exists(repo: Path, revision: str) -> bool:
    try:
        _git(repo, "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}")
    except RuntimeError:
        return False
    return True


def scan_commits(repo: Path, rev_range: str) -> Iterator[Finding]:
    """Scan commit messages. A range with an unknown start scans the last 20 commits."""
    deny = DenyList.load(repo)
    start, separator, end = rev_range.replace("...", "..").partition("..")
    command = ["log", "--format=%H%x1f%B%x1e"]
    if separator and start.strip("0") and _revision_exists(repo, start):
        command.append(f"{start}..{end or 'HEAD'}")
    else:
        command += ["-n", "20", end or start or "HEAD"]
    output = _git(repo, *command).decode("utf-8", errors="replace")
    for record in output.split("\x1e"):
        sha, _, message = record.strip().partition("\x1f")
        if not sha:
            continue
        where = f"commit {sha[:7]}"
        if _CO_AUTHOR.search(message):
            yield Finding(
                where, 1, "co-author", "co-author trailer; the repository rules forbid it"
            )
        yield from scan_text(where, message, deny)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository to check")
    parser.add_argument("--staged", action="store_true", help="scan the index, not the tree")
    parser.add_argument("--commits", metavar="RANGE", help="also scan commit messages in RANGE")
    args = parser.parse_args(argv)
    try:
        findings = list(scan_files(args.repo, staged=args.staged))
        if args.commits:
            findings += scan_commits(args.repo, args.commits)
    except RuntimeError as exc:
        print(f"check_repo: {exc}", file=sys.stderr)
        return 2
    for finding in findings:
        print(finding)
    if findings:
        print(f"check_repo: {len(findings)} finding(s). Fix them, or add the allow marker.")
        return 1
    print("check_repo: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
