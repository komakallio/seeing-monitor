"""The unit parser and the unit checks in `tools/lint_deploy.py`.

The positive tests read the units that ship. The negative tests change one line of a shipped unit
and check that the matching rule fires, so a rule that never fails would show here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.deploy.helpers import mutate, read_deploy, rules
from tools.lint_deploy import (
    ACQUIRE,
    CORE,
    FAILURE_UNIT,
    HARDENING,
    MAIN_SERVICES,
    TARGET_UNIT,
    WEB,
    Finding,
    Unit,
    lint_unit,
    lint_unit_set,
    parse_unit,
)

UNIT_NAMES = (*MAIN_SERVICES, FAILURE_UNIT, TARGET_UNIT)


@pytest.fixture(scope="module")
def texts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name: read_deploy(root, f"systemd/{name}") for name in UNIT_NAMES}


@pytest.fixture(scope="module")
def units(texts: dict[str, str]) -> dict[str, Unit]:
    return {name: parse_unit(name, text)[0] for name, text in texts.items()}


def lint(name: str, text: str) -> list[Finding]:
    return lint_unit(name, text)


# --- The parser -------------------------------------------------------------------------------


def test_the_parser_reads_sections_keys_comments_and_continuations() -> None:
    unit, findings = parse_unit(
        "x.service",
        "# first comment\n[Unit]\nDescription = A sample\n; another comment\n\n"
        "[Service]\nExecStart=/bin/true \\\n  --flag\nEnvironment=A=1\nEnvironment=B=2\n",
    )
    assert findings == []
    assert [section.name for section in unit.sections] == ["Unit", "Service"]
    assert unit.value("Unit", "Description") == "A sample"
    assert unit.value("Service", "ExecStart") == "/bin/true --flag"
    assert unit.values("Service", "Environment") == ["A=1", "B=2"]
    assert unit.comments == [(1, "first comment"), (4, "another comment")]
    assert unit.kind == "service"


def test_an_empty_assignment_resets_a_list() -> None:
    unit, _ = parse_unit(
        "x.service", "[Service]\nReadWritePaths=/a /b\nReadWritePaths=\nReadWritePaths=/c\n"
    )
    assert unit.values("Service", "ReadWritePaths") == ["/c"]
    assert unit.words("Service", "ReadWritePaths") == ["/c"]
    assert unit.has("Service", "ReadWritePaths")


def test_an_empty_key_is_present_but_has_no_value() -> None:
    unit, _ = parse_unit("x.service", "[Service]\nCapabilityBoundingSet=\n")
    assert unit.has("Service", "CapabilityBoundingSet")
    assert unit.value("Service", "CapabilityBoundingSet") is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[Unit]\nDescription\n", "not a Key=Value"),
        ("Description=x\n", "before any section"),
        ("[Unit\nDescription=x\n", "malformed section"),
        ("[]\n", "malformed section"),
        ("[Unit]\nDescription=x \\\n", "never ends"),
    ],
)
def test_the_parser_reports_syntax_problems(text: str, message: str) -> None:
    _, findings = parse_unit("x.service", text)
    assert rules(findings) == {"unit-syntax"}
    assert message in findings[0].message


# --- The units that ship ---------------------------------------------------------------------


@pytest.mark.parametrize("name", UNIT_NAMES)
def test_a_shipped_unit_passes(name: str, texts: dict[str, str]) -> None:
    assert lint(name, texts[name]) == []


def test_the_shipped_units_satisfy_the_architecture(units: dict[str, Unit]) -> None:
    assert lint_unit_set(units) == []


def test_only_core_may_write_the_data_directory(units: dict[str, Unit]) -> None:
    writers = [
        name
        for name in MAIN_SERVICES
        if "@DATA_DIR@" in units[name].words("Service", "ReadWritePaths")
    ]
    assert writers == [CORE]


def test_the_services_that_run_for_months_restart_and_report(units: dict[str, Unit]) -> None:
    for name in MAIN_SERVICES:
        unit = units[name]
        assert unit.value("Service", "Type") == "notify"
        assert unit.value("Service", "Restart") == "always"
        assert unit.has("Service", "WatchdogSec")
        assert unit.has("Unit", "StartLimitBurst")
        assert unit.words("Unit", "OnFailure") == ["seeingmon-failed@%n.service"]


# --- Negative tests: the unit checks ---------------------------------------------------------


def test_an_unknown_section_is_a_finding(texts: dict[str, str]) -> None:
    text = texts[WEB] + "\n[Timer]\nOnCalendar=daily\n"
    assert rules(lint(WEB, text)) == {"unit-unknown-section"}


def test_a_service_section_in_a_target_is_a_finding(texts: dict[str, str]) -> None:
    text = texts[TARGET_UNIT] + "\n[Service]\nExecStart=/bin/true\n"
    assert rules(lint(TARGET_UNIT, text)) == {"unit-unknown-section"}


def test_a_misspelled_key_is_a_finding(texts: dict[str, str]) -> None:
    text = mutate(texts[WEB], "ProtectSystem=strict", "ProtectSytem=strict")
    assert rules(lint(WEB, text)) == {"unit-unknown-key", "unit-hardening"}


def test_a_key_in_the_wrong_section_is_a_finding(texts: dict[str, str]) -> None:
    text = mutate(texts[WEB], "StartLimitBurst=5\n", "")
    text = mutate(text, "[Service]\n", "[Service]\nStartLimitBurst=5\n")
    assert {"unit-unknown-key", "unit-start-limit"} <= rules(lint(WEB, text))


@pytest.mark.parametrize(
    ("name", "line", "missing"),
    [
        (WEB, "Description=Seeing monitor REST API and web UI (web)\n", "Description"),
        (WEB, "Type=notify\n", "Type"),
        (TARGET_UNIT, "WantedBy=multi-user.target\n", "WantedBy"),
    ],
)
def test_a_missing_required_key_is_a_finding(
    texts: dict[str, str], name: str, line: str, missing: str
) -> None:
    findings = lint(name, mutate(texts[name], line, ""))
    assert "unit-missing-key" in rules(findings)
    assert any(missing in finding.message for finding in findings)


@pytest.mark.parametrize(
    "exec_line",
    [
        "ExecStart=/usr/bin/seeingmon web",
        "ExecStart=seeingmon web",
        "ExecStart=/opt/x/seeingmon web",
    ],
)
def test_exec_start_must_start_with_the_prefix_placeholder(
    texts: dict[str, str], exec_line: str
) -> None:
    text = mutate(texts[WEB], "ExecStart=@PREFIX@/current/venv/bin/seeingmon web", exec_line)
    assert rules(lint(WEB, text)) == {"unit-exec-prefix"}


def test_every_exec_line_must_start_with_the_prefix_placeholder(texts: dict[str, str]) -> None:
    text = mutate(
        texts[CORE],
        "ExecStopPost=-@PREFIX@/current/venv/bin/seeingmon heater-off",
        "ExecStopPost=-/usr/bin/true",
    )
    assert "unit-exec-prefix" in rules(lint(CORE, text))


def test_modifier_characters_do_not_hide_the_prefix(texts: dict[str, str]) -> None:
    text = mutate(
        texts[WEB],
        "ExecStart=@PREFIX@/current/venv/bin/seeingmon web",
        "ExecStart=!!@PREFIX@/current/venv/bin/seeingmon web",
    )
    assert lint(WEB, text) == []


def test_a_documented_exception_allows_a_system_command(texts: dict[str, str]) -> None:
    assert lint(FAILURE_UNIT, texts[FAILURE_UNIT]) == []
    no_reason = mutate(texts[FAILURE_UNIT], "allow system-exec: it must", "allow system-exec:")
    no_reason = mutate(no_reason, "still write the message when the release is broken, so\n", "\n")
    assert "unit-exec-prefix" in rules(lint(FAILURE_UNIT, no_reason))
    no_comment = mutate(texts[FAILURE_UNIT], "lint-deploy: allow", "lint-deploy: permit")
    assert "unit-exec-prefix" in rules(lint(FAILURE_UNIT, no_comment))


@pytest.mark.parametrize("name", MAIN_SERVICES)
def test_watchdog_needs_type_notify(texts: dict[str, str], name: str) -> None:
    findings = lint(name, mutate(texts[name], "Type=notify", "Type=simple"))
    assert "unit-watchdog" in rules(findings)


def test_watchdog_without_notify_is_a_finding_in_any_unit(texts: dict[str, str]) -> None:
    text = mutate(texts[FAILURE_UNIT], "Type=oneshot", "Type=oneshot\nWatchdogSec=30s")
    assert rules(lint(FAILURE_UNIT, text)) == {"unit-watchdog"}


def test_a_main_service_needs_a_watchdog(texts: dict[str, str]) -> None:
    findings = lint(WEB, mutate(texts[WEB], "WatchdogSec=60s\n", ""))
    assert rules(findings) == {"unit-watchdog"}


@pytest.mark.parametrize("user_line", ["", "User=root", "User=0"])
def test_a_unit_that_runs_as_root_is_a_finding(texts: dict[str, str], user_line: str) -> None:
    text = mutate(texts[WEB], "User=@USER@\n", user_line + "\n" if user_line else "")
    findings = lint(WEB, text)
    assert "unit-root" in rules(findings)


def test_a_documented_exception_allows_root(texts: dict[str, str]) -> None:
    text = mutate(
        texts[WEB], "User=@USER@\n", "# lint-deploy: allow root-user: it needs a port\nUser=root\n"
    )
    assert lint(WEB, text) == []
    bare = mutate(texts[WEB], "User=@USER@\n", "# lint-deploy: allow root-user\nUser=root\n")
    assert "unit-root" in rules(lint(WEB, bare))


@pytest.mark.parametrize("key", sorted(HARDENING))
@pytest.mark.parametrize("name", MAIN_SERVICES)
def test_every_sandbox_option_is_required(texts: dict[str, str], name: str, key: str) -> None:
    lines = [
        line for line in texts[name].splitlines(keepends=True) if not line.startswith(f"{key}=")
    ]
    assert len(lines) < len(texts[name].splitlines()), f"{name} has no {key}"
    findings = lint(name, "".join(lines))
    assert "unit-hardening" in rules(findings)
    assert any(key in finding.message for finding in findings)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("ProtectSystem=strict", "ProtectSystem=full"),
        ("NoNewPrivileges=yes", "NoNewPrivileges=no"),
        ("PrivateTmp=yes", "PrivateTmp=false"),
        ("SystemCallArchitectures=native", "SystemCallArchitectures=x86"),
    ],
)
def test_a_weak_sandbox_value_is_a_finding(texts: dict[str, str], old: str, new: str) -> None:
    assert "unit-hardening" in rules(lint(WEB, mutate(texts[WEB], old, new)))


def test_a_boolean_has_several_spellings(texts: dict[str, str]) -> None:
    text = mutate(texts[WEB], "NoNewPrivileges=yes", "NoNewPrivileges=true")
    assert lint(WEB, text) == []


def test_realtime_is_restricted_unless_the_unit_may_raise_its_priority(
    texts: dict[str, str],
) -> None:
    assert "\nRestrictRealtime=" not in texts[ACQUIRE]
    assert "\nAmbientCapabilities=CAP_SYS_NICE\n" in texts[ACQUIRE]
    without_capability = mutate(
        texts[ACQUIRE], "AmbientCapabilities=CAP_SYS_NICE", "AmbientCapabilities="
    )
    assert "unit-hardening" in rules(lint(ACQUIRE, without_capability))
    assert "unit-hardening" in rules(lint(WEB, mutate(texts[WEB], "RestrictRealtime=yes\n", "")))


@pytest.mark.parametrize(
    ("old", "new", "rule"),
    [
        ("Restart=always", "Restart=on-failure", "unit-restart"),
        ("RestartSec=5s\n", "", "unit-restart"),
        ("StartLimitBurst=5\n", "", "unit-start-limit"),
        ("StartLimitIntervalSec=600\n", "", "unit-start-limit"),
        ("OnFailure=seeingmon-failed@%n.service\n", "", "unit-start-limit"),
        ("MemoryMax=15%\n", "", "unit-memory"),
        ("OOMScoreAdjust=100\n", "", "unit-memory"),
        ("SyslogIdentifier=seeingmon-web\n", "", "unit-memory"),
        (
            "LoadCredential=seeingmon-connection-key:",
            "LoadCredential=other-key:",
            "unit-credential",
        ),
        ("RuntimeDirectory=seeingmon\n", "RuntimeDirectory=other\n", "unit-runtime"),
        ("RuntimeDirectoryPreserve=yes\n", "", "unit-runtime"),
        ("WorkingDirectory=@CONFIG_DIR@\n", "", "unit-runtime"),
        ("PartOf=seeingmon.target\n", "", "unit-install"),
        ("WantedBy=seeingmon.target", "WantedBy=multi-user.target", "unit-install"),
    ],
)
def test_a_long_running_service_needs_its_supervision_settings(
    texts: dict[str, str], old: str, new: str, rule: str
) -> None:
    assert rule in rules(lint(WEB, mutate(texts[WEB], old, new)))


def test_a_unit_of_an_unknown_type_is_a_finding() -> None:
    assert rules(lint_unit("x.timer", "[Timer]\n")) == {"unit-type"}


# --- Negative tests: the architecture invariants -------------------------------------------------


def parsed(texts: dict[str, str], **changes: tuple[str, str]) -> dict[str, Unit]:
    """Parse the shipped units after replacing text in some of them (keyed by short name)."""
    names = {"acquire": ACQUIRE, "core": CORE, "web": WEB, "target": TARGET_UNIT}
    result = dict(texts)
    for short, (old, new) in changes.items():
        result[names[short]] = mutate(result[names[short]], old, new)
    return {name: parse_unit(name, text)[0] for name, text in result.items()}


def test_a_missing_unit_is_reported(texts: dict[str, str]) -> None:
    units = parsed(texts)
    del units[FAILURE_UNIT]
    assert rules(lint_unit_set(units)) == {"unit-set-missing"}


def test_the_target_must_want_every_service(texts: dict[str, str]) -> None:
    units = parsed(texts, target=("seeingmon-web.service\nAfter", "x.service\nAfter"))
    assert "unit-set-target" in rules(lint_unit_set(units))


def test_web_must_not_write_the_data_directory(texts: dict[str, str]) -> None:
    units = parsed(texts, web=("ReadOnlyPaths=@DATA_DIR@", "ReadWritePaths=@DATA_DIR@"))
    assert rules(lint_unit_set(units)) == {"unit-set-data"}


def test_core_must_write_the_data_directory(texts: dict[str, str]) -> None:
    units = parsed(texts, core=("ReadWritePaths=@DATA_DIR@", "ReadWritePaths=/var/lib/other"))
    assert rules(lint_unit_set(units)) == {"unit-set-data"}


def test_acquire_must_not_see_the_data_directory(texts: dict[str, str]) -> None:
    units = parsed(texts, acquire=("InaccessiblePaths=@DATA_DIR@", "InaccessiblePaths=/var/empty"))
    assert rules(lint_unit_set(units)) == {"unit-set-data"}


def test_only_core_keeps_state(texts: dict[str, str]) -> None:
    units = parsed(
        texts, web=("PrivateDevices=yes", "PrivateDevices=yes\nStateDirectory=seeingmon")
    )
    assert rules(lint_unit_set(units)) == {"unit-set-data"}


def test_web_has_no_camera_access(texts: dict[str, str]) -> None:
    units = parsed(texts, web=("PrivateDevices=yes", "PrivateDevices=no"))
    assert rules(lint_unit_set(units)) == {"unit-set-devices"}


def test_only_acquire_touches_usb(texts: dict[str, str]) -> None:
    units = parsed(
        texts, core=("OOMScoreAdjust=-200", "OOMScoreAdjust=-200\nDeviceAllow=char-usb_device rw")
    )
    assert rules(lint_unit_set(units)) == {"unit-set-devices"}


def test_web_loads_the_token_hash_and_the_other_units_do_not(texts: dict[str, str]) -> None:
    line = "LoadCredential=seeingmon-token-hash:@CONFIG_DIR@/credentials/seeingmon-token-hash\n"
    assert lint_unit_set(parsed(texts)) == []
    assert rules(lint_unit_set(parsed(texts, web=(line, "")))) == {"unit-set-credential"}
    core = ("OOMScoreAdjust=-200", "OOMScoreAdjust=-200\n" + line.rstrip())
    assert rules(lint_unit_set(parsed(texts, core=core))) == {"unit-set-credential"}


def test_a_unit_may_load_several_credentials(texts: dict[str, str]) -> None:
    assert texts[WEB].count("LoadCredential=") == 2
    assert lint(WEB, texts[WEB]) == []
    only_key = mutate(
        texts[WEB],
        "LoadCredential=seeingmon-connection-key:",
        "LoadCredential=another-key:",
    )
    assert "unit-credential" in rules(lint(WEB, only_key))


def test_only_acquire_raises_its_priority(texts: dict[str, str]) -> None:
    units = parsed(texts, core=("AmbientCapabilities=\n", "AmbientCapabilities=CAP_SYS_NICE\n"))
    assert "unit-set-devices" in rules(lint_unit_set(units))


@pytest.mark.parametrize(
    ("short", "old", "new"),
    [
        ("acquire", "AF_UNIX AF_NETLINK", "AF_UNIX AF_NETLINK AF_INET"),
        ("web", "AF_UNIX AF_INET AF_INET6", "AF_UNIX AF_INET AF_INET6 AF_PACKET"),
        ("core", "AF_UNIX AF_INET AF_INET6", "AF_UNIX AF_INET AF_INET6 AF_NETLINK"),
    ],
)
def test_the_address_families_stay_narrow(
    texts: dict[str, str], short: str, old: str, new: str
) -> None:
    units = parsed(texts, **{short: (old, new)})
    assert rules(lint_unit_set(units)) == {"unit-set-network"}


def test_core_switches_the_heater_off_when_it_stops(texts: dict[str, str]) -> None:
    units = parsed(
        texts, core=("ExecStopPost=-@PREFIX@/current/venv/bin/seeingmon heater-off\n", "")
    )
    assert rules(lint_unit_set(units)) == {"unit-set-heater"}


def test_only_core_controls_the_heater(texts: dict[str, str]) -> None:
    units = parsed(
        texts,
        web=(
            "ExecStart=@PREFIX@",
            "ExecStopPost=-@PREFIX@/current/venv/bin/seeingmon heater-off\nExecStart=@PREFIX@",
        ),
    )
    assert rules(lint_unit_set(units)) == {"unit-set-heater"}


def test_web_starts_after_acquire_and_core(texts: dict[str, str]) -> None:
    units = parsed(
        texts,
        web=(
            "After=network-online.target seeingmon-acquire.service seeingmon-core.service",
            "After=network-online.target",
        ),
    )
    assert rules(lint_unit_set(units)) == {"unit-set-order"}


def test_core_starts_after_acquire(texts: dict[str, str]) -> None:
    units = parsed(texts, core=("After=seeingmon-acquire.service\n", ""))
    assert rules(lint_unit_set(units)) == {"unit-set-order"}


@pytest.mark.parametrize(
    ("short", "old", "new"),
    [
        ("web", "OOMScoreAdjust=100", "OOMScoreAdjust=-300"),
        ("core", "OOMScoreAdjust=-200", "OOMScoreAdjust=-600"),
        ("acquire", "OOMScoreAdjust=-500", "OOMScoreAdjust=0"),
        ("acquire", "OOMScoreAdjust=-500", "OOMScoreAdjust=high"),
    ],
)
def test_the_killer_takes_web_then_core_then_acquire(
    texts: dict[str, str], short: str, old: str, new: str
) -> None:
    units = parsed(texts, **{short: (old, new)})
    assert rules(lint_unit_set(units)) == {"unit-set-oom"}
