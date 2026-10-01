#!/usr/bin/env python3
"""Lint everything that ships under `deploy/`: shell scripts, systemd units, the udev rule, and
the templates that the installer fills in.

The linter is pure Python. It reads files, so it runs on every platform that CI covers. It never
needs a Raspberry Pi, systemd, or a shell.

Usage:
    python tools/lint_deploy.py                  lint the repository that holds this file
    python tools/lint_deploy.py --repo PATH      lint another clone
    python tools/lint_deploy.py --no-shellcheck  skip the external shell pass

Exit status: 0 when every check passes, 1 when a check finds a problem, 2 when the linter cannot
run (for example, the repository has no `deploy/` directory).

What it checks:

- **Shell scripts** (`deploy/**/*.sh`): the shebang, `set -eu` (and `pipefail` for bash), line
  endings, unquoted expansions of the parameters that the script defines, defaults for the
  required parameters, absolute developer paths and addresses, `eval`, and an error message for
  a missing required parameter. `shellcheck` runs too when it is available. Where it is not
  (the `shellcheck-py` package has no wheel for Linux arm64), the linter runs `bash -n` and says
  so.
- **systemd units** (`deploy/systemd/`): a small parser (known sections and keys), the required
  keys, `ExecStart` that starts with the prefix placeholder, `WatchdogSec` with `Type=notify`, the
  sandbox options, no unit that runs as root, and the invariants of the architecture (only core
  writes the data directory, web has no devices, only acquire touches USB, core switches the
  heater off when it stops).
- **The udev rule**: its syntax, and that it gives the service group access to ZWO cameras
  without world access and keeps them out of USB autosuspend.
- **Templates** (every file in a subdirectory of `deploy/`): the header marker, the placeholders,
  the syntax of the fragments, and a render with sample values that leaves no `@NAME@` behind.

A unit can document an exception with a comment line of the form:

    # lint-deploy: allow <rule>: <reason>

The rules that a comment can allow are `root-user` and `system-exec`. A comment without a reason
does not count.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

DEPLOY_DIRNAME = "deploy"
MARKER = "Managed by the seeingmon installer"
EXEC_PREFIX = "@PREFIX@/"
ZWO_VENDOR_ID = "03c3"  # the USB vendor ID of ZWO, a public value
CREDENTIAL = "seeingmon-connection-key"
TOKEN_CREDENTIAL = "seeingmon-token-hash"

PLACEHOLDER = re.compile(r"@([A-Z][A-Z0-9_]*)@")

# Sample values for the render test. They are generic examples, never defaults of the installer.
SAMPLE_VALUES: Mapping[str, str] = {
    "PREFIX": "/opt/seeingmon",
    "USER": "seeingmon",
    "GROUP": "seeingmon",
    "DATA_DIR": "/srv/seeingmon-data",
    "CONFIG_DIR": "/etc/seeingmon",
    "USBFS_MEMORY_MB": "1000",
    "TIME_SOURCES": "server time.example.org iburst\nserver 192.0.2.10 iburst",
}

ACQUIRE = "seeingmon-acquire.service"
CORE = "seeingmon-core.service"
WEB = "seeingmon-web.service"
MAIN_SERVICES = (ACQUIRE, CORE, WEB)
FAILURE_UNIT = "seeingmon-failed@.service"
TARGET_UNIT = "seeingmon.target"
HEATER_OFF = "@PREFIX@/current/venv/bin/seeingmon heater-off"
DATA_DIR = "@DATA_DIR@"

Add: TypeAlias = Callable[[int, str, str], None]


class LintError(Exception):
    """The linter cannot run, as opposed to a check that finds a problem."""


@dataclass(frozen=True)
class Finding:
    """One problem. `line` is 1-based, and 0 means the finding concerns the whole file."""

    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: [{self.rule}] {self.message}"


@dataclass
class Report:
    """The findings of a run, and notes about checks that did not run."""

    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings


# --- Rendering -----------------------------------------------------------------------------------


def placeholders(text: str) -> set[str]:
    """The names of the `@NAME@` placeholders in a text."""
    return set(PLACEHOLDER.findall(text))


def render(text: str, values: Mapping[str, str]) -> str:
    """Fill in a template the way the installer does: replace each `@NAME@` with its value.

    A placeholder without a value stays in the text, so the caller can report it.
    """
    for name, value in values.items():
        text = text.replace(f"@{name}@", value)
    return text


def _has_marker(text: str) -> bool:
    return any(MARKER in line for line in text.splitlines()[:5])


def lint_template(
    label: str, text: str, values: Mapping[str, str] = SAMPLE_VALUES
) -> list[Finding]:
    """Check a template: it carries the marker, and a render leaves no placeholder behind."""
    findings: list[Finding] = []
    if not _has_marker(text):
        findings.append(
            Finding(label, 1, "template-marker", f"the first lines must say '{MARKER}'")
        )
    left = placeholders(render(text, values))
    if left:
        names = ", ".join(f"@{name}@" for name in sorted(left))
        findings.append(Finding(label, 0, "template-placeholder", f"no value for {names}"))
    return findings


# --- A parser for systemd unit files -------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    key: str
    value: str
    line: int


@dataclass
class Section:
    name: str
    line: int
    entries: list[Entry] = field(default_factory=list)


@dataclass
class Unit:
    """A parsed unit file. `comments` holds the comment lines as (line number, text)."""

    name: str
    label: str
    sections: list[Section] = field(default_factory=list)
    comments: list[tuple[int, str]] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return self.name.rsplit(".", 1)[-1]

    def entries(self, section: str, key: str) -> list[Entry]:
        return [
            entry
            for found in self.sections
            if found.name == section
            for entry in found.entries
            if entry.key == key
        ]

    def values(self, section: str, key: str) -> list[str]:
        """The values of a key. An empty assignment resets the list, as in systemd."""
        result: list[str] = []
        for entry in self.entries(section, key):
            if entry.value == "":
                result.clear()
            else:
                result.append(entry.value)
        return result

    def value(self, section: str, key: str) -> str | None:
        """The last value of a key, or `None` when the key is absent or empty."""
        values = self.values(section, key)
        return values[-1] if values else None

    def has(self, section: str, key: str) -> bool:
        """Whether the key appears at all, even with an empty value."""
        return bool(self.entries(section, key))

    def words(self, section: str, key: str) -> list[str]:
        """The space-separated words of every value of a key."""
        return [word for value in self.values(section, key) for word in value.split()]


def parse_unit(name: str, text: str, label: str | None = None) -> tuple[Unit, list[Finding]]:
    """Parse a unit file. Returns the unit and the syntax problems that the parse found."""
    path = label or name
    unit = Unit(name=name, label=path)
    findings: list[Finding] = []
    section: Section | None = None
    pending = ""
    pending_line = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if pending:
            line = pending + " " + line
            number = pending_line
            pending = ""
        if line.startswith(("#", ";")):
            unit.comments.append((number, line[1:].strip()))
            continue
        if not line:
            continue
        if line.endswith("\\"):
            pending = line[:-1].rstrip()
            pending_line = number
            continue
        if line.startswith("["):
            if not line.endswith("]") or len(line) < 3:
                findings.append(Finding(path, number, "unit-syntax", f"malformed section: {line}"))
                continue
            section = Section(line[1:-1], number)
            unit.sections.append(section)
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            findings.append(Finding(path, number, "unit-syntax", f"not a Key=Value line: {line}"))
            continue
        if section is None:
            findings.append(Finding(path, number, "unit-syntax", "a setting before any section"))
            continue
        section.entries.append(Entry(key.strip(), value.strip(), number))
    if pending:
        findings.append(Finding(path, pending_line, "unit-syntax", "a continuation never ends"))
    return unit, findings


def _words(text: str) -> frozenset[str]:
    """The set of words in a text. It keeps the long tables below readable."""
    return frozenset(text.split())


# The sections and keys that systemd knows for the unit types that we ship. A key outside this
# table is a typo, or a new setting that this table must learn.
_UNIT_KEYS = _words(
    """Description Documentation Wants Requires Requisite BindsTo PartOf Upholds Conflicts Before
    After OnFailure OnSuccess PropagatesStopTo PropagatesReloadTo ConditionPathExists
    ConditionPathIsDirectory ConditionFileNotEmpty ConditionVirtualization AssertPathExists
    StartLimitIntervalSec StartLimitBurst StartLimitAction FailureAction SuccessAction
    RefuseManualStart RefuseManualStop DefaultDependencies AllowIsolate JobTimeoutSec
    StopWhenUnneeded IgnoreOnIsolate RequiresMountsFor WantsMountsFor"""
)
_SERVICE_KEYS = _words(
    """Type ExecStart ExecStartPre ExecStartPost ExecStop ExecStopPost ExecReload Restart RestartSec
    RestartPreventExitStatus RestartForceExitStatus SuccessExitStatus TimeoutStartSec
    TimeoutStopSec TimeoutAbortSec TimeoutSec WatchdogSec NotifyAccess RemainAfterExit
    GuessMainPID PIDFile BusName KillMode KillSignal SendSIGKILL SendSIGHUP FinalKillSignal
    WatchdogSignal User Group DynamicUser SupplementaryGroups WorkingDirectory RootDirectory
    RuntimeDirectory RuntimeDirectoryMode RuntimeDirectoryPreserve StateDirectory
    StateDirectoryMode CacheDirectory CacheDirectoryMode LogsDirectory LogsDirectoryMode
    ConfigurationDirectory UMask Environment EnvironmentFile PassEnvironment UnsetEnvironment
    StandardInput StandardOutput StandardError SyslogIdentifier SyslogLevelPrefix SyslogFacility
    SyslogLevel LogLevelMax LogRateLimitIntervalSec LogRateLimitBurst LimitNOFILE LimitCORE
    LimitRTPRIO LimitMEMLOCK Nice OOMScoreAdjust OOMPolicy IOSchedulingClass IOSchedulingPriority
    CPUSchedulingPolicy CPUSchedulingPriority CPUSchedulingResetOnFork CPUAffinity CPUWeight
    CPUQuota MemoryMax MemoryHigh MemoryLow MemoryMin MemorySwapMax TasksMax IOWeight Slice
    Delegate MemoryAccounting CPUAccounting TasksAccounting IPAddressAllow IPAddressDeny
    DevicePolicy DeviceAllow ProtectSystem ProtectHome PrivateTmp PrivateDevices PrivateNetwork
    PrivateUsers PrivateMounts ProtectKernelTunables ProtectKernelModules ProtectKernelLogs
    ProtectControlGroups ProtectClock ProtectHostname ProtectProc ProcSubset NoNewPrivileges
    CapabilityBoundingSet AmbientCapabilities SecureBits ReadWritePaths ReadOnlyPaths
    InaccessiblePaths ExecPaths NoExecPaths TemporaryFileSystem BindPaths BindReadOnlyPaths
    MountFlags RestrictAddressFamilies RestrictNamespaces RestrictRealtime RestrictSUIDSGID
    RestrictFileSystems LockPersonality MemoryDenyWriteExecute SystemCallFilter
    SystemCallArchitectures SystemCallErrorNumber SystemCallLog LoadCredential
    LoadCredentialEncrypted SetCredential ImportCredential KeyringMode RemoveIPC Personality
    UtmpIdentifier"""
)
_INSTALL_KEYS = _words("Alias WantedBy RequiredBy UpheldBy Also DefaultInstance")

ALLOWED_SECTIONS: Mapping[str, Mapping[str, frozenset[str]]] = {
    "service": {"Unit": _UNIT_KEYS, "Service": _SERVICE_KEYS, "Install": _INSTALL_KEYS},
    "target": {"Unit": _UNIT_KEYS, "Install": _INSTALL_KEYS},
}

REQUIRED_KEYS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "service": {"Unit": ("Description",), "Service": ("Type", "ExecStart", "User")},
    "target": {"Unit": ("Description",), "Install": ("WantedBy",)},
}

# The sandbox that every service of the project carries, as (key, required value). A value of
# `None` accepts any non-empty value, and `""` accepts the key with any value, even an empty one.
HARDENING: Mapping[str, str | None] = {
    "NoNewPrivileges": "yes",
    "ProtectSystem": "strict",
    "ProtectHome": "yes",
    "PrivateTmp": "yes",
    "ProtectKernelTunables": "yes",
    "ProtectKernelModules": "yes",
    "ProtectKernelLogs": "yes",
    "ProtectControlGroups": "yes",
    "ProtectHostname": "yes",
    "RestrictNamespaces": "yes",
    "RestrictSUIDSGID": "yes",
    "LockPersonality": "yes",
    "SystemCallArchitectures": "native",
    "RestrictAddressFamilies": None,
    "SystemCallFilter": None,
    "CapabilityBoundingSet": "",
}
_EXEC_KEYS = (
    "ExecStart",
    "ExecStartPre",
    "ExecStartPost",
    "ExecStop",
    "ExecStopPost",
    "ExecReload",
)
_LONG_RUNNING = frozenset({"notify", "simple", "exec", "forking"})
_TRUE = frozenset({"yes", "true", "1", "on"})
_ALLOW = re.compile(r"lint-deploy:\s*allow\s+(?P<rule>[a-z-]+)\s*(?P<reason>.*)")


def _allowed(unit: Unit, rule: str) -> bool:
    """Whether a comment documents an exception to `rule`, with a reason."""
    for _, comment in unit.comments:
        match = _ALLOW.search(comment)
        if match and match.group("rule") == rule and match.group("reason").strip(" :"):
            return True
    return False


# The characters that can precede the command of an Exec= line. An `@` is a modifier unless it
# starts one of our `@NAME@` placeholders.
_EXEC_MODIFIERS = re.compile(r"(?:[-+!:]|@(?![A-Z][A-Z0-9_]*@))*")


def _exec_command(value: str) -> str:
    """The command of an `Exec*=` value, without the modifier characters."""
    match = _EXEC_MODIFIERS.match(value)
    return value[match.end() :] if match else value


def lint_unit(name: str, text: str, label: str | None = None) -> list[Finding]:
    """Check one unit file."""
    path = label or name
    unit, findings = parse_unit(name, text, path)
    kind = unit.kind
    if kind not in ALLOWED_SECTIONS:
        return [*findings, Finding(path, 0, "unit-type", f"no rules for a .{kind} unit")]

    def add(line: int, rule: str, message: str) -> None:
        findings.append(Finding(path, line, rule, message))

    allowed = ALLOWED_SECTIONS[kind]
    for section in unit.sections:
        if section.name not in allowed:
            add(section.line, "unit-unknown-section", f"[{section.name}] is not a {kind} section")
            continue
        for entry in section.entries:
            if entry.key not in allowed[section.name]:
                add(entry.line, "unit-unknown-key", f"{entry.key} is not a [{section.name}] key")
    for section_name, keys in REQUIRED_KEYS[kind].items():
        for key in keys:
            if not unit.has(section_name, key):
                add(0, "unit-missing-key", f"[{section_name}] needs {key}=")
    if kind == "service":
        _lint_service(unit, add)
    return findings


def _lint_service(unit: Unit, add: Add) -> None:
    unit_type = (unit.value("Service", "Type") or "").lower()
    for key in _EXEC_KEYS:
        for entry in unit.entries("Service", key):
            command = _exec_command(entry.value)
            if (
                command
                and not command.startswith(EXEC_PREFIX)
                and not _allowed(unit, "system-exec")
            ):
                add(
                    entry.line,
                    "unit-exec-prefix",
                    f"{key} must start with {EXEC_PREFIX}, so that the installer decides where "
                    "the release lives",
                )
    if unit.has("Service", "WatchdogSec") and unit_type != "notify":
        add(0, "unit-watchdog", "WatchdogSec needs Type=notify")
    if unit.name in MAIN_SERVICES:
        if unit_type != "notify":
            add(0, "unit-watchdog", "the main services need Type=notify")
        if not unit.has("Service", "WatchdogSec"):
            add(0, "unit-watchdog", "the main services need WatchdogSec=")
    user = unit.value("Service", "User")
    if user in {None, "root", "0"} and not _allowed(unit, "root-user"):
        add(0, "unit-root", "the unit runs as root: set User=, or document an exception")
    _lint_hardening(unit, add)
    if unit_type in _LONG_RUNNING:
        _lint_long_running(unit, add)


def _lint_hardening(unit: Unit, add: Add) -> None:
    for key, wanted in HARDENING.items():
        entries = unit.entries("Service", key)
        if not entries:
            add(0, "unit-hardening", f"the sandbox needs {key}=")
        elif wanted is None:
            if not unit.values("Service", key):
                add(entries[-1].line, "unit-hardening", f"{key}= needs a value")
        elif wanted != "":
            actual = (unit.value("Service", key) or "").lower()
            if actual != wanted and not (wanted == "yes" and actual in _TRUE):
                add(entries[-1].line, "unit-hardening", f"{key} must be {wanted}, not {actual}")
    realtime_allowed = "CAP_SYS_NICE" in unit.words("Service", "AmbientCapabilities")
    realtime_blocked = (unit.value("Service", "RestrictRealtime") or "").lower() in _TRUE
    if not realtime_allowed and not realtime_blocked:
        add(0, "unit-hardening", "the sandbox needs RestrictRealtime=yes, or CAP_SYS_NICE")


def _lint_long_running(unit: Unit, add: Add) -> None:
    if (unit.value("Service", "Restart") or "").lower() != "always":
        add(0, "unit-restart", "a long-running service needs Restart=always")
    if not unit.has("Service", "RestartSec"):
        add(0, "unit-restart", "a long-running service needs RestartSec=")
    for key in ("StartLimitIntervalSec", "StartLimitBurst"):
        if not unit.has("Unit", key):
            add(0, "unit-start-limit", f"[Unit] needs {key}=, so that a restart storm stops")
    if "seeingmon-failed@%n.service" not in unit.words("Unit", "OnFailure"):
        add(0, "unit-start-limit", "OnFailure= must start seeingmon-failed@%n.service")
    for key in ("OOMScoreAdjust", "MemoryMax", "SyslogIdentifier"):
        if not unit.has("Service", key):
            add(0, "unit-memory", f"a long-running service needs {key}=")
    if not _loads(unit, CREDENTIAL):
        add(0, "unit-credential", f"LoadCredential= must load {CREDENTIAL} from @CONFIG_DIR@")
    if unit.value("Service", "RuntimeDirectory") != "seeingmon":
        add(0, "unit-runtime", "RuntimeDirectory= must be seeingmon, the directory of the sockets")
    if (unit.value("Service", "RuntimeDirectoryPreserve") or "").lower() not in _TRUE:
        add(0, "unit-runtime", "RuntimeDirectoryPreserve=yes keeps the sockets of the other units")
    if unit.value("Service", "WorkingDirectory") != "@CONFIG_DIR@":
        add(0, "unit-runtime", "WorkingDirectory= must be @CONFIG_DIR@, where local/config.toml is")
    if TARGET_UNIT not in unit.words("Unit", "PartOf"):
        add(0, "unit-install", f"PartOf={TARGET_UNIT} ties the unit to the target")
    if TARGET_UNIT not in unit.words("Install", "WantedBy"):
        add(0, "unit-install", f"[Install] needs WantedBy={TARGET_UNIT}")


def _loads(unit: Unit, credential: str) -> bool:
    """Whether the unit loads the credential from a file under `@CONFIG_DIR@`."""
    prefix = f"{credential}:@CONFIG_DIR@/"
    return any(value.startswith(prefix) for value in unit.values("Service", "LoadCredential"))


def lint_unit_set(units: Mapping[str, Unit]) -> list[Finding]:
    """Check the invariants of the architecture across the units.

    Each check names a sentence of `docs/architecture.md`: core is the only writer of the data
    directory, web has no camera access and no write access to the data directory, only acquire
    touches USB, and the heater outputs are off whenever core is not running.
    """
    findings = [
        Finding(f"deploy/systemd/{name}", 0, "unit-set-missing", "no such unit")
        for name in (*MAIN_SERVICES, TARGET_UNIT, FAILURE_UNIT)
        if name not in units
    ]
    if findings:
        return findings

    def add(name: str, rule: str, message: str) -> None:
        findings.append(Finding(units[name].label, 0, rule, message))

    target = units[TARGET_UNIT]
    for name in MAIN_SERVICES:
        if name not in target.words("Unit", "Wants"):
            add(TARGET_UNIT, "unit-set-target", f"the target must want {name}")
    for name in MAIN_SERVICES:
        unit = units[name]
        writes = DATA_DIR in unit.words("Service", "ReadWritePaths")
        if name == CORE and not writes:
            add(
                name, "unit-set-data", "core is the only writer: it needs ReadWritePaths=@DATA_DIR@"
            )
        if name != CORE and writes:
            add(name, "unit-set-data", "only core writes the data directory")
        if name != CORE and unit.has("Service", "StateDirectory"):
            add(name, "unit-set-data", "only core keeps state under /var/lib")
        if name == WEB and DATA_DIR not in unit.words("Service", "ReadOnlyPaths"):
            add(name, "unit-set-data", "web needs ReadOnlyPaths=@DATA_DIR@")
        if name == ACQUIRE and DATA_DIR not in unit.words("Service", "InaccessiblePaths"):
            add(name, "unit-set-data", "acquire needs InaccessiblePaths=@DATA_DIR@")
    for name in MAIN_SERVICES:
        loads_token = _loads(units[name], TOKEN_CREDENTIAL)
        if name == WEB and not loads_token:
            add(name, "unit-set-credential", f"web checks the API token: load {TOKEN_CREDENTIAL}")
        if name != WEB and loads_token:
            add(name, "unit-set-credential", f"only web needs {TOKEN_CREDENTIAL}")
    web = units[WEB]
    if (web.value("Service", "PrivateDevices") or "").lower() not in _TRUE:
        add(WEB, "unit-set-devices", "web has no camera access, so it needs PrivateDevices=yes")
    for name in MAIN_SERVICES:
        unit = units[name]
        if name != ACQUIRE and any("usb" in word for word in unit.words("Service", "DeviceAllow")):
            add(name, "unit-set-devices", "only acquire touches USB")
        if name != ACQUIRE and "CAP_SYS_NICE" in unit.words("Service", "AmbientCapabilities"):
            add(name, "unit-set-devices", "only acquire raises the priority of its capture thread")
    families = {
        ACQUIRE: {"AF_UNIX", "AF_NETLINK"},
        CORE: {"AF_UNIX", "AF_INET", "AF_INET6"},
        WEB: {"AF_UNIX", "AF_INET", "AF_INET6"},
    }
    for name, allowed in families.items():
        used = set(units[name].words("Service", "RestrictAddressFamilies"))
        extra = used - allowed
        if extra:
            add(name, "unit-set-network", f"{name} must not allow {', '.join(sorted(extra))}")
    for name in MAIN_SERVICES:
        stop = [_exec_command(value) for value in units[name].values("Service", "ExecStopPost")]
        has_hook = any(command.strip() == HEATER_OFF for command in stop)
        if name == CORE and not has_hook:
            add(name, "unit-set-heater", f"core needs ExecStopPost=-{HEATER_OFF}")
        if name != CORE and any("heater" in command for command in stop):
            add(name, "unit-set-heater", "only core controls the heater")
    after_web = set(web.words("Unit", "After"))
    for name in (ACQUIRE, CORE):
        if name not in after_web:
            add(WEB, "unit-set-order", f"web must start after {name}")
    if ACQUIRE not in units[CORE].words("Unit", "After"):
        add(CORE, "unit-set-order", f"core must start after {ACQUIRE}")
    acquire_score, core_score, web_score = (_oom_score(units[name]) for name in MAIN_SERVICES)
    if (
        acquire_score is None
        or core_score is None
        or web_score is None
        or not acquire_score < core_score < web_score
    ):
        add(
            ACQUIRE,
            "unit-set-oom",
            "OOMScoreAdjust= must rise from acquire to core to web, so that the killer takes web "
            "before core and core before acquire",
        )
    return findings


def _oom_score(unit: Unit) -> int | None:
    value = unit.value("Service", "OOMScoreAdjust")
    try:
        return None if value is None else int(value)
    except ValueError:
        return None


# --- The udev rule -------------------------------------------------------------------------------

_UDEV_TOKEN = re.compile(
    r"(?P<key>[A-Za-z_]+)(?:\{(?P<attr>[^}]*)\})?\s*(?P<op>==|!=|\+=|-=|:=|=)\s*"
    r'"(?P<value>(?:[^"\\]|\\.)*)"'
)
_UDEV_MATCH = _words(
    """ACTION DEVPATH KERNEL NAME SYMLINK SUBSYSTEM DRIVER ATTR SYSCTL KERNELS SUBSYSTEMS DRIVERS
    ATTRS TAGS ENV CONST TAG TEST PROGRAM RESULT"""
)
_UDEV_ASSIGN = _words(
    "NAME SYMLINK OWNER GROUP MODE TAG ENV RUN LABEL GOTO IMPORT OPTIONS ATTR SYSCTL SECLABEL"
)
_UDEV_MATCH_OPS = frozenset({"==", "!="})


def _udev_tokens(line: str) -> tuple[list[str], bool]:
    """Split a rule at the commas outside quotes. Returns the tokens and whether quotes balance."""
    tokens: list[str] = []
    current: list[str] = []
    quoted = False
    escaped = False
    for char in line:
        if quoted and escaped:
            escaped = False
        elif quoted and char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == "," and not quoted:
            tokens.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tokens.append("".join(current).strip())
    return [token for token in tokens if token], not quoted


def _udev_rules(text: str) -> list[tuple[int, str]]:
    """The logical rule lines of a rules file as (first line number, text)."""
    rules: list[tuple[int, str]] = []
    pending = ""
    start = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        if not pending:
            start = number
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        rules.append((start, (pending + line).strip()))
        pending = ""
    if pending:
        rules.append((start, pending.strip()))
    return rules


def lint_udev(label: str, text: str) -> list[Finding]:
    """Check the syntax of a udev rules file, and the policy for the ZWO camera rule.

    Pass a rendered text: the group name must be a real name by then. The unrendered placeholder
    `@GROUP@` and `@USER@` also pass.
    """
    findings: list[Finding] = []
    seen_camera = False
    for number, rule in _udev_rules(text):
        tokens, balanced = _udev_tokens(rule)
        if not balanced:
            findings.append(Finding(label, number, "udev-syntax", "a quote never closes"))
            continue
        parsed: list[tuple[str, str, str, str]] = []
        for token in tokens:
            match = _UDEV_TOKEN.fullmatch(token)
            if match is None:
                findings.append(Finding(label, number, "udev-syntax", f"cannot parse: {token}"))
                continue
            parsed.append((match["key"], match["attr"] or "", match["op"], match["value"]))
        matches = [item for item in parsed if item[2] in _UDEV_MATCH_OPS]
        assigns = [item for item in parsed if item[2] not in _UDEV_MATCH_OPS]
        for key, _, op, value in parsed:
            if key not in _UDEV_MATCH and key not in _UDEV_ASSIGN:
                findings.append(Finding(label, number, "udev-unknown-key", f"unknown key {key}"))
            elif op in _UDEV_MATCH_OPS and key not in _UDEV_MATCH:
                findings.append(Finding(label, number, "udev-operator", f"{key} cannot match"))
            elif op not in _UDEV_MATCH_OPS and key not in _UDEV_ASSIGN:
                findings.append(Finding(label, number, "udev-operator", f"{key} cannot assign"))
            if key == "MODE" and not re.fullmatch(r"[0-7]{3,4}", value):
                findings.append(Finding(label, number, "udev-mode", f"MODE {value!r} is not octal"))
            if key in {"GROUP", "OWNER"} and not re.fullmatch(r"@[A-Z]+@|[a-z_][a-z0-9_-]*", value):
                findings.append(Finding(label, number, "udev-name", f"bad {key} name {value!r}"))
        if parsed and (not matches or not assigns):
            findings.append(
                Finding(label, number, "udev-incomplete", "a rule needs a match and an assignment")
            )
        vendor = [v for k, a, o, v in matches if k == "ATTR" and a == "idVendor"]
        if ZWO_VENDOR_ID in vendor:
            seen_camera = True
            findings.extend(_lint_camera_rule(label, number, assigns))
    if not seen_camera:
        findings.append(
            Finding(label, 0, "udev-policy", f"no rule matches the ZWO vendor ID {ZWO_VENDOR_ID}")
        )
    return findings


def _lint_camera_rule(
    label: str, number: int, assigns: Sequence[tuple[str, str, str, str]]
) -> list[Finding]:
    """The policy for the rule that matches the ZWO vendor ID."""
    findings: list[Finding] = []
    values = {(key, attr): value for key, attr, _, value in assigns}

    def add(message: str) -> None:
        findings.append(Finding(label, number, "udev-policy", message))

    if not values.get(("GROUP", "")):
        add("the camera rule must set GROUP, so that the service group opens the device")
    mode = values.get(("MODE", ""))
    if mode is None:
        add("the camera rule must set MODE")
    elif re.fullmatch(r"[0-7]{3,4}", mode) and int(mode, 8) & 0o007:
        add("the camera rule must not give world access")
    if ("OWNER", "") in values:
        add("set GROUP and not OWNER")
    if values.get(("ATTR", "power/control")) != "on":
        add('the camera rule must set ATTR{power/control}="on", which keeps the camera awake')
    return findings


# --- Fragments -----------------------------------------------------------------------------------

_TMPFILES_TYPES = _words("f F w d D e q Q p L c b a A x X r R z Z t T h H m M v C")
_JOURNALD_KEYS = _words(
    """Storage Compress Seal SplitMode RateLimitIntervalSec RateLimitBurst SystemMaxUse
    SystemKeepFree SystemMaxFileSize SystemMaxFiles RuntimeMaxUse RuntimeKeepFree
    RuntimeMaxFileSize RuntimeMaxFiles MaxRetentionSec MaxFileSec ForwardToSyslog ForwardToKMsg
    ForwardToConsole ForwardToWall MaxLevelStore MaxLevelSyslog MaxLevelKMsg MaxLevelConsole
    MaxLevelWall LineMax ReadKMsg Audit"""
)
_CHRONY_SOURCES = ("server", "pool", "peer")


def lint_tmpfiles(label: str, text: str) -> list[Finding]:
    """Check a rendered `tmpfiles.d` file: it must raise the USB buffer limit."""
    findings: list[Finding] = []
    limit_set = False
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(None, 6)
        if fields[0].rstrip("!+-=~^") not in _TMPFILES_TYPES:
            findings.append(Finding(label, number, "tmpfiles-syntax", f"unknown type {fields[0]}"))
            continue
        if len(fields) < 2 or not fields[1].startswith("/"):
            findings.append(Finding(label, number, "tmpfiles-syntax", "the path must be absolute"))
            continue
        if fields[0] == "w":
            if len(fields) < 7:
                findings.append(Finding(label, number, "tmpfiles-syntax", "w needs an argument"))
            elif fields[1].endswith("/usbcore/parameters/usbfs_memory_mb"):
                limit_set = fields[6].isdigit() and int(fields[6]) >= 16
                if not limit_set:
                    findings.append(
                        Finding(label, number, "tmpfiles-value", "the limit must be a number in MB")
                    )
    if not limit_set and not findings:
        findings.append(Finding(label, 0, "tmpfiles-policy", "the file must set usbfs_memory_mb"))
    return findings


def lint_journald(label: str, text: str) -> list[Finding]:
    """Check a journald drop-in: the journal must stay in RAM."""
    unit, findings = parse_unit("journald.conf", text, label)
    settings: dict[str, str] = {}
    for section in unit.sections:
        if section.name != "Journal":
            findings.append(Finding(label, section.line, "journald-section", f"[{section.name}]"))
            continue
        for entry in section.entries:
            if entry.key not in _JOURNALD_KEYS:
                findings.append(Finding(label, entry.line, "journald-key", f"unknown {entry.key}"))
            settings[entry.key] = entry.value
    if settings.get("Storage") != "volatile":
        findings.append(Finding(label, 0, "journald-policy", "Storage=volatile keeps logs in RAM"))
    return findings


def lint_chrony(label: str, text: str) -> list[Finding]:
    """Check a rendered chrony fragment: source lines only, and at least one source."""
    findings: list[Finding] = []
    sources = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(("#", "!", ";", "%")):
            continue
        words = line.split()
        if words[0] not in _CHRONY_SOURCES:
            findings.append(
                Finding(
                    label,
                    number,
                    "chrony-directive",
                    f"the fragment holds sources only: {words[0]}",
                )
            )
        elif len(words) < 2:
            findings.append(Finding(label, number, "chrony-syntax", f"{words[0]} needs a name"))
        else:
            sources += 1
    if not sources:
        findings.append(
            Finding(label, 0, "chrony-policy", "the fragment needs at least one source")
        )
    return findings


def lint_polkit(label: str, text: str) -> list[Finding]:
    """Check the polkit rule: balanced brackets, and a limit to the service user."""
    findings: list[Finding] = []
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
    if "polkit.addRule(" not in code:
        findings.append(Finding(label, 0, "polkit-syntax", "no polkit.addRule call"))
    for opening, closing in ("()", "{}", "[]"):
        if code.count(opening) != code.count(closing):
            findings.append(Finding(label, 0, "polkit-syntax", f"unbalanced {opening}{closing}"))
    if not re.search(r'subject\.user\s*!==?\s*"(@USER@|[a-z_][a-z0-9_-]*)"', code):
        findings.append(
            Finding(label, 0, "polkit-policy", "the rule must limit itself to the service user")
        )
    return findings


# --- Shell scripts -------------------------------------------------------------------------------

_SHEBANG = re.compile(r"#!\s*(?:/usr/bin/env\s+|/bin/|/usr/bin/)(?P<shell>bash|sh)\s*$")
_ARM = re.compile(r"^\s*(?P<options>-[A-Za-z0-9-]+(?:\|-[A-Za-z0-9-]+)*)\)(?P<body>.*)$")
_ASSIGNED = re.compile(r"(?:^|[;&{\s])(?P<name>[A-Za-z_][A-Za-z0-9_]*)\+?=")
_REQUIRE = re.compile(r"^\s*require\s+(?P<name>[A-Z][A-Z0-9_]*)\s+(?P<option>--[a-z0-9-]+)\s*$")
_NAME_START = re.compile(r"[A-Za-z_]")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ASSIGNMENT_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?\+?=")
_DECLARATIONS = frozenset({"export", "local", "readonly", "declare", "typeset"})
_BEFORE_COMMAND = frozenset(
    {"then", "do", "else", "elif", "if", "while", "until", "{", "!", "time"}
)
_SPECIAL_PARAMETERS = "@*#?$!-0123456789"


@dataclass(frozen=True)
class Expansion:
    """An unquoted expansion of a variable: its name and line."""

    name: str
    line: int


@dataclass
class _Word:
    text: str = ""  # a sketch of the word: `$` stands for an expansion and `"` for quoted text
    pending: list[Expansion] = field(default_factory=list)


@dataclass
class _Command:
    words: list[str] = field(default_factory=list)
    in_double_bracket: bool = False
    skip_words: int = 0


class _Scanner:
    """Find the unquoted expansions of some variables in a shell script.

    The scanner follows quotes, comments, command substitutions, parameter expansions, here
    documents, and the contexts where bash does not split words: the right side of an assignment,
    the subject of `case`, `[[ ... ]]`, and arithmetic. It does not parse the whole language. It
    reads the style of the scripts in `deploy/`, and it errs toward reporting.
    """

    def __init__(self, text: str, names: Collection[str]) -> None:
        self.text = text
        self.names = frozenset(names)
        self.pos = 0
        self.line = 1
        self.found: list[Expansion] = []
        self.heredocs: list[tuple[str, bool]] = []

    def run(self) -> list[Expansion]:
        self._commands(stop=None)
        return self.found

    # --- Characters ---------------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> str:
        index = self.pos + offset
        return self.text[index] if index < len(self.text) else ""

    def _take(self) -> str:
        char = self.text[self.pos]
        self.pos += 1
        if char == "\n":
            self.line += 1
        return char

    def _at_end(self) -> bool:
        return self.pos >= len(self.text)

    # --- Commands -----------------------------------------------------------------------------

    def _commands(self, stop: str | None) -> None:
        """Scan commands until `stop` closes them: `)` for `$(...)`, a backtick, or the end."""
        command = _Command()
        word: _Word | None = None
        depth = 0
        while not self._at_end():
            char = self._peek()
            if char in " \t\r":
                self._end_word(command, word)
                word = None
                self._take()
            elif char == "\n":
                self._end_word(command, word)
                word = None
                self._take()
                self._read_heredocs()
                command = _Command()
            elif char == "\\":
                self._take()
                if not self._at_end():
                    escaped = self._take()
                    if escaped != "\n":  # a backslash before a newline continues the line
                        word = word or _Word()
                        word.text += '"'
            elif char == "#" and word is None:
                while not self._at_end() and self._peek() != "\n":
                    self._take()
            elif (char == "`" and stop == "`") or (char == ")" and stop == ")" and depth == 0):
                self._end_word(command, word)  # the end of a command substitution
                self._take()
                return
            elif char in "'\"`":
                word = word or _Word()
                word.text += '"'
                self._take()
                if char == "'":
                    self._skip_single_quoted()
                elif char == '"':
                    self._double_quoted()
                else:
                    self._commands(stop="`")
            elif char == "$":
                word = word or _Word()
                word.text += "$"
                self._dollar(word, quoted=False)
            elif char in ";&|":
                self._end_word(command, word)
                word = None
                self._take()
                command = _Command()
            elif char == "(":
                if word is not None and _ASSIGNMENT_WORD.fullmatch(word.text):
                    self._take()  # an array literal: NAME=(a b c)
                    self._commands(stop=")")
                    word.text += "()"
                elif word is None and not command.words and self._peek(1) == "(":
                    self._skip_arithmetic_command()
                else:
                    self._end_word(command, word)
                    word = None
                    self._take()
                    depth += 1
                    command = _Command()
            elif char == ")":
                if depth > 0:  # the end of a subshell, or of a function header
                    self._end_word(command, word)
                # Without an open parenthesis, this ends a case pattern. A pattern is not split.
                word = None
                self._take()
                depth = max(0, depth - 1)
                command = _Command()
            elif char in "<>":
                self._end_word(command, word)
                word = None
                self._redirection()
            else:
                word = word or _Word()
                word.text += char
                self._take()
        self._end_word(command, word)

    def _end_word(self, command: _Command, word: _Word | None) -> None:
        """Finish a word: report its unquoted expansions unless bash does not split it."""
        if word is None:
            return
        text = word.text
        first = command.words[0] if command.words else ""
        exempt = False
        if command.in_double_bracket:
            exempt = True
            if text == "]]":
                command.in_double_bracket = False
        elif not command.words and text == "[[":
            command.in_double_bracket = True
            exempt = True
        elif command.skip_words:
            command.skip_words -= 1
            exempt = True
        elif _ASSIGNMENT_WORD.match(text) and (
            first in _DECLARATIONS
            or all(_ASSIGNMENT_WORD.match(earlier) for earlier in command.words)
        ):
            exempt = True  # NAME=value, alone or ahead of a command, or after `local`
        if not exempt:
            self.found.extend(word.pending)
        if not command.words and text == "case":
            command.skip_words = 1
        if text in _BEFORE_COMMAND and (not command.words or command.words[-1] in _BEFORE_COMMAND):
            command.words = []  # the next word starts a command
        else:
            command.words.append(text)

    def _redirection(self) -> None:
        """Skip a redirection operator. A here document records its delimiter."""
        if self.text.startswith("<<<", self.pos):
            self.pos += 3
            return
        if self.text.startswith("<<", self.pos):
            self.pos += 2
            strip_tabs = self._peek() == "-"
            if strip_tabs:
                self.pos += 1
            while self._peek() in (" ", "\t"):
                self.pos += 1
            self.heredocs.append((self._heredoc_delimiter(), strip_tabs))
            return
        self._take()

    def _heredoc_delimiter(self) -> str:
        quote = self._peek()
        if quote in ("'", '"'):
            self._take()
            start = self.pos
            while not self._at_end() and self._peek() != quote:
                self._take()
            delimiter = self.text[start : self.pos]
            if not self._at_end():
                self._take()
            return delimiter
        start = self.pos
        while not self._at_end() and self._peek() not in " \t\n;&|<>()":
            self._take()
        return self.text[start : self.pos].replace("\\", "")

    def _read_heredocs(self) -> None:
        """Skip the bodies of the here documents that the line just ended announced."""
        while self.heredocs:
            delimiter, strip_tabs = self.heredocs.pop(0)
            while not self._at_end():
                end = self.text.find("\n", self.pos)
                line = self.text[self.pos :] if end == -1 else self.text[self.pos : end]
                self.pos = len(self.text) if end == -1 else end + 1
                if end != -1:
                    self.line += 1
                candidate = line.rstrip("\r")
                if strip_tabs:
                    candidate = candidate.lstrip("\t")
                if candidate == delimiter:
                    break

    # --- Quotes and expansions ------------------------------------------------------------

    def _skip_single_quoted(self) -> None:
        """Skip to the closing quote. The opening quote is already taken."""
        while not self._at_end():
            if self._take() == "'":
                return

    def _skip_ansi_quoted(self) -> None:
        """Skip a `$'...'` string, which honors backslash escapes. The `$'` is already taken."""
        while not self._at_end():
            char = self._take()
            if char == "\\" and not self._at_end():
                self._take()
            elif char == "'":
                return

    def _double_quoted(self) -> None:
        """Scan a double-quoted string. The opening quote is already taken."""
        while not self._at_end():
            char = self._peek()
            if char == '"':
                self._take()
                return
            if char == "\\":
                self._take()
                if not self._at_end():
                    self._take()
            elif char == "$":
                self._dollar(None, quoted=True)
            elif char == "`":
                self._take()
                self._commands(stop="`")
            else:
                self._take()

    def _dollar(self, word: _Word | None, *, quoted: bool) -> None:
        following = self._peek(1)
        if following == "(":
            if self._peek(2) == "(":
                self._skip_arithmetic_expansion()
            else:
                self._take()
                self._take()
                self._commands(stop=")")
        elif following == "{":
            self._parameter(word, quoted=quoted)
        elif following == "'" and not quoted:
            self._take()
            self._take()
            self._skip_ansi_quoted()
        elif following == '"' and not quoted:
            self._take()
            self._take()
            self._double_quoted()
        else:
            match = _NAME.match(self.text, self.pos + 1)
            if match:
                self.pos += 1 + len(match.group())
                self._record(match.group(), word, quoted=quoted)
            else:
                self._take()
                if self._peek() and self._peek() in _SPECIAL_PARAMETERS:
                    self._take()

    def _parameter(self, word: _Word | None, *, quoted: bool) -> None:
        """Scan `${...}`: record the name, then read the operand up to the closing brace."""
        self._take()
        self._take()
        prefix = ""
        if self._peek() in ("!", "#") and _NAME_START.match(self._peek(1) or " "):
            prefix = self._take()
        match = _NAME.match(self.text, self.pos)
        if match:
            self.pos += len(match.group())
            if not prefix:
                self._record(match.group(), word, quoted=quoted)
        elif self._peek() and self._peek() in _SPECIAL_PARAMETERS:
            self._take()
        while not self._at_end():
            char = self._peek()
            if char == "}":
                self._take()
                return
            if char == "\\":
                self._take()
                if not self._at_end():
                    self._take()
            elif char == "'" and not quoted:
                self._take()
                self._skip_single_quoted()
            elif char == '"':
                self._take()
                self._double_quoted()
            elif char == "$":
                self._dollar(word, quoted=quoted)
            else:
                self._take()

    def _record(self, name: str, word: _Word | None, *, quoted: bool) -> None:
        if name in self.names and not quoted and word is not None:
            word.pending.append(Expansion(name, self.line))

    def _skip_arithmetic_expansion(self) -> None:
        """Skip `$((...))`. Word splitting never applies to its result in a way that matters."""
        self._take()
        depth = 0
        while not self._at_end():
            char = self._take()
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return

    def _skip_arithmetic_command(self) -> None:
        """Skip the command `((...))`."""
        depth = 0
        while not self._at_end():
            char = self._take()
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return


def find_unquoted_expansions(text: str, names: Collection[str]) -> list[Expansion]:
    """The unquoted expansions of the variables in `names` that would undergo word splitting."""
    return _Scanner(text, names).run()


@dataclass
class ShellFacts:
    """What the option parser of a script defines."""

    options: dict[str, int] = field(default_factory=dict)  # option -> line of its case arm
    variables: dict[str, list[str]] = field(default_factory=dict)  # variable -> its options
    required: dict[str, str] = field(default_factory=dict)  # variable -> option, from `require`


def shell_facts(text: str) -> ShellFacts:
    """Find the options of the `case` arms, the variables they set, and the `require` calls."""
    facts = ShellFacts()
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        arm = _ARM.match(lines[index])
        if arm is None:
            require = _REQUIRE.match(lines[index])
            if require:
                facts.required[require["name"]] = require["option"]
            index += 1
            continue
        options = [option for option in arm["options"].split("|") if option.startswith("--")]
        body = arm["body"]
        end = index
        while ";;" not in lines[end] and end + 1 < len(lines):  # an arm can span lines
            end += 1
            body += " " + lines[end]
        for option in options:
            facts.options.setdefault(option, index + 1)
        for assigned in _ASSIGNED.finditer(body):
            for option in options:
                facts.variables.setdefault(assigned["name"], []).append(option)
        index = end + 1
    return facts


def _code_lines(text: str) -> list[tuple[int, str]]:
    """The lines that are not comments, with their line numbers."""
    return [
        (number, line)
        for number, line in enumerate(text.splitlines(), start=1)
        if not line.lstrip().startswith("#")
    ]


def _strict_mode_findings(
    label: str, lines: Sequence[tuple[int, str]], shell: str
) -> list[Finding]:
    for number, line in lines:
        if not line.strip():
            continue
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            break
        if not words or words[0] != "set":
            break
        flags: set[str] = set()
        names: set[str] = set()
        index = 1
        while index < len(words):
            word = words[index]
            if word.startswith("-") and not word.startswith("--"):
                flags.update(word[1:])
                if word.endswith("o") and index + 1 < len(words):  # -o takes the next word
                    names.add(words[index + 1])
                    index += 1
            index += 1
        findings: list[Finding] = []
        if "e" not in flags and "errexit" not in names:
            findings.append(Finding(label, number, "shell-strict-mode", "set -e is missing"))
        if "u" not in flags and "nounset" not in names:
            findings.append(Finding(label, number, "shell-strict-mode", "set -u is missing"))
        if shell == "bash" and "pipefail" not in names:
            findings.append(
                Finding(label, number, "shell-strict-mode", "a bash script needs -o pipefail")
            )
        if shell == "sh" and "pipefail" in names:
            findings.append(
                Finding(label, number, "shell-strict-mode", "pipefail is not POSIX: use bash")
            )
        return findings
    return [Finding(label, 1, "shell-strict-mode", "the first command must be set -eu")]


def _check_line() -> Callable[[str], list[tuple[str, str]]]:
    """`tools.check_repo.check_line`, which reports private values and machine paths."""
    try:
        from tools.check_repo import check_line
    except ModuleNotFoundError:  # run as a script, so the repository root is not on sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tools.check_repo import check_line
    return check_line


def lint_shell(label: str, text: str) -> list[Finding]:
    """Check one shell script with the structural rules. `shellcheck` runs separately."""
    findings: list[Finding] = []
    lines = text.splitlines()
    if "\r" in text:
        number = next(n for n, line in enumerate(text.split("\n"), start=1) if "\r" in line)
        findings.append(Finding(label, number, "shell-line-endings", "use LF line endings"))
    match = _SHEBANG.fullmatch(lines[0].strip()) if lines else None
    if match is None:
        findings.append(
            Finding(label, 1, "shell-shebang", "start with #!/usr/bin/env bash or #!/bin/sh")
        )
        shell = "bash"
    else:
        shell = match["shell"]
    code = _code_lines(text)
    findings.extend(_strict_mode_findings(label, code, shell))
    for number, line in code:
        if re.search(r"(?<![\w-])eval(?![\w-])", line):
            findings.append(Finding(label, number, "shell-eval", "avoid eval: it runs its input"))
        if "`" in line:
            findings.append(
                Finding(
                    label,
                    number,
                    "shell-backtick",
                    "avoid backticks: they run a command, even in the text of an unquoted here "
                    "document. Use $(...) for a command, and plain words in text",
                )
            )
    check_line = _check_line()
    for number, line in enumerate(lines, start=1):
        if "repo-check: allow" in line:
            continue
        for rule, hint in check_line(line):
            findings.append(Finding(label, number, "shell-private-value", f"{rule}: {hint}"))
    facts = shell_facts(text)
    findings.extend(_parameter_findings(label, text, facts))
    return findings


def _parameter_findings(label: str, text: str, facts: ShellFacts) -> list[Finding]:
    lines = text.splitlines()
    findings = [
        Finding(
            label,
            expansion.line,
            "shell-unquoted-parameter",
            f'quote the expansion of the parameter {expansion.name}: "${expansion.name}"',
        )
        for expansion in find_unquoted_expansions(text, sorted(facts.variables))
    ]
    for name, option in facts.required.items():
        if name not in facts.variables:
            findings.append(
                Finding(
                    label, 0, "shell-require", f"require {name} {option}: no option sets {name}"
                )
            )
        for number, line in enumerate(lines, start=1):
            initial = re.fullmatch(rf"{name}=(?P<value>[^#]*?)\s*(?:#.*)?", line)
            if initial and initial["value"] not in {"", "''", '""'}:
                findings.append(
                    Finding(
                        label,
                        number,
                        "shell-required-default",
                        f"{name} is required, so it has no default: pass {option}",
                    )
                )
        for number, line in _code_lines(text):
            if re.search(rf"\$\{{{name}:?[-=][^}}]", line):
                findings.append(
                    Finding(
                        label,
                        number,
                        "shell-required-default",
                        f"{name} is required, so it has no default expansion",
                    )
                )
    if facts.required and "missing required parameter" not in text:
        findings.append(
            Finding(label, 0, "shell-require", "a missing required parameter must say which one")
        )
    for option, number in facts.options.items():
        if option == "--help":
            continue  # the usage text is the help
        mentions = len(re.findall(rf"(?<![A-Za-z0-9-]){re.escape(option)}(?![A-Za-z0-9-])", text))
        if mentions < 2:
            findings.append(
                Finding(label, number, "shell-undocumented-option", f"{option} is not in the usage")
            )
    return findings


# --- External shell tools ------------------------------------------------------------------------

_SHELLCHECK_LINE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):\d+: (?P<severity>\w+): (?P<text>.*)$")


def find_shellcheck() -> str | None:
    """The `shellcheck` binary: on the PATH, or next to the running interpreter.

    The `shellcheck-py` package installs the binary beside the interpreter of the environment,
    which is not on the PATH when you run the interpreter by its path.
    """
    found = shutil.which("shellcheck")
    if found:
        return found
    name = "shellcheck.exe" if os.name == "nt" else "shellcheck"
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.is_file() else None


def run_shellcheck(binary: str, repo: Path, scripts: Sequence[str]) -> list[Finding]:
    """Run `shellcheck` on the scripts (paths relative to `repo`) and parse its report."""
    result = subprocess.run(
        [binary, "--format=gcc", *scripts],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=repo,
        check=False,
    )
    findings: list[Finding] = []
    for raw in result.stdout.splitlines():
        match = _SHELLCHECK_LINE.match(raw)
        if match:
            findings.append(
                Finding(
                    match["path"].replace("\\", "/"),
                    int(match["line"]),
                    "shellcheck",
                    f"{match['severity']}: {match['text']}",
                )
            )
    if result.returncode != 0 and not findings:
        detail = (result.stderr or result.stdout).strip().splitlines()[:1]
        findings.append(Finding("shellcheck", 0, "shellcheck", f"did not run: {' '.join(detail)}"))
    return findings


def run_bash_syntax(repo: Path, scripts: Sequence[str]) -> list[Finding]:
    """Check the syntax with `bash -n`, the fallback where `shellcheck` is not available."""
    findings: list[Finding] = []
    bash = shutil.which("bash")
    if bash is None:
        return findings
    for script in scripts:
        result = subprocess.run(
            [bash, "-n", script],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            cwd=repo,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()[:1]
            findings.append(
                Finding(script, 0, "shell-syntax", " ".join(detail) or "bash -n failed")
            )
    return findings


def executable_bit_findings(repo: Path, scripts: Sequence[str]) -> list[Finding] | None:
    """Check that git records each script as executable. Returns `None` when git cannot say."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--stage", "--", DEPLOY_DIRNAME],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    modes: dict[str, str] = {}
    for line in result.stdout.splitlines():
        info, _, name = line.partition("\t")
        modes[name] = info.split()[0]
    return [
        Finding(script, 0, "shell-not-executable", "git records this script without the x bit")
        for script in scripts
        if script in modes and modes[script] != "100755"
    ]


# --- The whole repository ------------------------------------------------------------------------


def _read(path: Path, label: str, findings: list[Finding]) -> str | None:
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        findings.append(Finding(label, 0, "file-encoding", "the file is not UTF-8"))
        return None


def lint_repo(repo: Path, *, shell_tools: bool = True) -> Report:
    """Lint every file under `deploy/`. Raises `LintError` when there is no such directory."""
    deploy = repo / DEPLOY_DIRNAME
    if not deploy.is_dir():
        raise LintError(f"there is no {DEPLOY_DIRNAME}/ directory in {repo}")
    report = Report()
    findings = report.findings
    units: dict[str, Unit] = {}
    scripts: list[str] = []
    for path in sorted(item for item in deploy.rglob("*") if item.is_file()):
        relative = path.relative_to(repo).as_posix()
        parts = path.relative_to(deploy).parts
        folder = parts[0] if len(parts) > 1 else ""
        text = _read(path, relative, findings)
        if text is None:
            continue
        if "\r" in text and path.suffix != ".sh":
            findings.append(Finding(relative, 0, "file-line-endings", "use LF line endings"))
        if path.suffix == ".sh":
            scripts.append(relative)
            findings.extend(lint_shell(relative, text))
        if not folder:
            continue
        findings.extend(lint_template(relative, text))
        rendered = render(text, SAMPLE_VALUES)
        if folder == "systemd":
            units[path.name], _ = parse_unit(path.name, text, relative)
            findings.extend(lint_unit(path.name, text, relative))
        elif folder == "udev":
            findings.extend(lint_udev(relative, rendered))
        elif folder == "tmpfiles":
            findings.extend(lint_tmpfiles(relative, rendered))
        elif folder == "journald":
            findings.extend(lint_journald(relative, rendered))
        elif folder == "chrony":
            findings.extend(lint_chrony(relative, rendered))
        elif folder == "polkit":
            findings.extend(lint_polkit(relative, text))
    if units:
        findings.extend(lint_unit_set(units))
    if shell_tools and scripts:
        _run_shell_tools(repo, scripts, report)
    bits = executable_bit_findings(repo, scripts)
    if bits is None:
        report.notes.append("git cannot say whether the scripts are executable: skipped that check")
    else:
        findings.extend(bits)
    findings.sort(key=lambda finding: (finding.path, finding.line, finding.rule))
    return report


def _run_shell_tools(repo: Path, scripts: Sequence[str], report: Report) -> None:
    binary = find_shellcheck()
    if binary is not None:
        report.notes.append(f"shellcheck ran on {len(scripts)} scripts")
        report.findings.extend(run_shellcheck(binary, repo, scripts))
        return
    report.notes.append(
        "shellcheck is not available on this platform (the shellcheck-py package has no wheel "
        "for Linux arm64): skipped the shellcheck pass"
    )
    if os.name == "posix" and shutil.which("bash"):
        report.notes.append("ran bash -n on the scripts instead")
        report.findings.extend(run_bash_syntax(repo, scripts))
    else:
        report.notes.append("no bash either: only the structural checks ran on the scripts")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository to lint (default: the one that holds this file)",
    )
    parser.add_argument(
        "--no-shellcheck", action="store_true", help="skip shellcheck and the bash -n fallback"
    )
    args = parser.parse_args(argv)
    try:
        report = lint_repo(args.repo, shell_tools=not args.no_shellcheck)
    except LintError as error:
        print(f"lint_deploy: {error}", file=sys.stderr)
        return 2
    for note in report.notes:
        print(f"lint_deploy: note: {note}")
    for finding in report.findings:
        print(finding)
    if report.findings:
        print(f"lint_deploy: {len(report.findings)} finding(s)")
        return 1
    print("lint_deploy: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
