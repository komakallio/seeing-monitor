"""The udev rule, the fragments, and the template render in `tools/lint_deploy.py`.

The placeholders of a template are filled in by the installer. These tests render every template
with sample values, as the installer does, and check that no `@NAME@` is left.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.deploy.helpers import mutate, read_deploy, rules
from tools.lint_deploy import (
    MARKER,
    SAMPLE_VALUES,
    lint_chrony,
    lint_journald,
    lint_polkit,
    lint_template,
    lint_tmpfiles,
    lint_udev,
    placeholders,
    render,
)

ROOT = Path(__file__).resolve().parents[2]
RULE = read_deploy(ROOT, "udev/99-seeingmon-asi.rules")
TMPFILES = read_deploy(ROOT, "tmpfiles/seeingmon-usb.conf")
JOURNALD = read_deploy(ROOT, "journald/seeingmon.conf")
CHRONY = read_deploy(ROOT, "chrony/seeingmon.conf")
POLKIT = read_deploy(ROOT, "polkit/50-seeingmon.rules")


def rendered(text: str) -> str:
    return render(text, SAMPLE_VALUES)


def templates() -> list[Path]:
    """Every file in a subdirectory of `deploy/`."""
    deploy = ROOT / "deploy"
    return sorted(path for path in deploy.rglob("*") if path.is_file() and path.parent != deploy)


# --- Rendering --------------------------------------------------------------------------------


def test_render_replaces_every_occurrence() -> None:
    assert render("@A@ and @A@ and @B@", {"A": "x", "B": "y"}) == "x and x and y"


def test_render_leaves_a_placeholder_without_a_value() -> None:
    assert placeholders(render("@A@ and @B@", {"A": "x"})) == {"B"}


def test_placeholders_finds_names_and_ignores_other_at_signs() -> None:
    text = "@PREFIX@/bin and user@host and @lower@ and @A_1@"
    assert placeholders(text) == {"PREFIX", "A_1"}


def test_there_are_templates_to_render() -> None:
    names = {path.name for path in templates()}
    assert {
        "seeingmon-core.service",
        "99-seeingmon-asi.rules",
        "seeingmon-usb.conf",
        "seeingmon.conf",
        "seeingmon.sh",
    } <= names


@pytest.mark.parametrize("path", templates(), ids=lambda path: path.name)
def test_every_template_renders_without_a_placeholder_left(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert placeholders(rendered(text)) == set()
    assert lint_template(path.name, text) == []


def test_every_template_uses_only_known_placeholders() -> None:
    used = set().union(*(placeholders(path.read_text(encoding="utf-8")) for path in templates()))
    assert used <= set(SAMPLE_VALUES)


def test_every_sample_value_is_used_by_some_template() -> None:
    used = set().union(*(placeholders(path.read_text(encoding="utf-8")) for path in templates()))
    assert set(SAMPLE_VALUES) <= used


def test_an_unknown_placeholder_is_a_finding() -> None:
    findings = lint_template("x.conf", f"# {MARKER}\nvalue=@NOT_DEFINED@\n")
    assert rules(findings) == {"template-placeholder"}
    assert "@NOT_DEFINED@" in findings[0].message


def test_a_template_without_the_marker_is_a_finding() -> None:
    assert rules(lint_template("x.conf", "value=1\n")) == {"template-marker"}


def test_the_marker_must_be_near_the_top() -> None:
    text = "\n" * 6 + f"# {MARKER}\n"
    assert rules(lint_template("x.conf", text)) == {"template-marker"}


def test_the_sample_values_hold_no_machine_specific_value() -> None:
    from tools.check_repo import check_line

    for value in SAMPLE_VALUES.values():
        for line in value.splitlines():
            assert check_line(line) == []


# --- The udev rule ----------------------------------------------------------------------------


def test_the_shipped_udev_rule_passes() -> None:
    assert lint_udev("rule", rendered(RULE)) == []
    assert lint_udev("rule", RULE) == []  # the placeholder is also a valid group name


def test_the_udev_rule_matches_the_zwo_vendor_and_keeps_the_camera_awake() -> None:
    assert 'ATTR{idVendor}=="03c3"' in RULE
    assert 'ATTR{power/control}="on"' in RULE
    assert 'MODE="0660"' in RULE


@pytest.mark.parametrize(
    ("old", "new", "rule"),
    [
        ('ACTION=="add|change", ', 'ACTION=="add|change" ', "udev-syntax"),
        ('ATTR{idVendor}=="03c3"', 'ATTR{idVendor}=="03c3', "udev-syntax"),
        ('SUBSYSTEM=="usb"', 'SUBSYSTEM="usb"', "udev-operator"),
        ('GROUP="@GROUP@"', 'GROUP=="@GROUP@"', "udev-operator"),
        ('SUBSYSTEM=="usb"', 'SUBSYSTEMS=="usb", FROBNICATE=="x"', "udev-unknown-key"),
        ('MODE="0660"', 'MODE="rw-rw"', "udev-mode"),
        ('GROUP="@GROUP@"', 'GROUP="Not A Name"', "udev-name"),
        ('MODE="0660"', 'MODE="0666"', "udev-policy"),
        ('MODE="0660"', 'MODE="0664"', "udev-policy"),
        (', MODE="0660"', "", "udev-policy"),
        ('GROUP="@GROUP@"', 'OWNER="seeingmon"', "udev-policy"),
        (', TEST=="power/control", ATTR{power/control}="on"', "", "udev-policy"),
        ('ATTR{power/control}="on"', 'ATTR{power/control}="auto"', "udev-policy"),
        ('ATTR{idVendor}=="03c3"', 'ATTR{idVendor}=="1234"', "udev-policy"),
    ],
)
def test_a_broken_udev_rule_is_a_finding(old: str, new: str, rule: str) -> None:
    findings = lint_udev("rule", rendered(mutate(RULE, old, new)))
    assert rule in rules(findings)


def test_a_rule_without_a_match_or_an_assignment_is_a_finding() -> None:
    assert "udev-incomplete" in rules(lint_udev("rule", rendered(RULE) + 'MODE="0660"\n'))
    assert "udev-incomplete" in rules(lint_udev("rule", rendered(RULE) + 'SUBSYSTEM=="usb"\n'))


def test_the_udev_parser_follows_continuation_lines_and_comments() -> None:
    text = (
        "# comment\n\n"
        'ACTION=="add", SUBSYSTEM=="usb", \\\n'
        '    ATTR{idVendor}=="03c3", GROUP="seeingmon", MODE="0660", \\\n'
        '    ATTR{power/control}="on"\n'
    )
    assert lint_udev("rule", text) == []


def test_a_comma_inside_quotes_does_not_split_a_token() -> None:
    text = rendered(RULE) + 'ACTION=="add", RUN+="/bin/sh -c \'echo a, b\'"\n'
    assert lint_udev("rule", text) == []


# --- Fragments --------------------------------------------------------------------------------


def test_the_shipped_fragments_pass() -> None:
    assert lint_tmpfiles("f", rendered(TMPFILES)) == []
    assert lint_journald("f", rendered(JOURNALD)) == []
    assert lint_chrony("f", rendered(CHRONY)) == []
    assert lint_polkit("f", POLKIT) == []


@pytest.mark.parametrize(
    ("old", "new", "rule"),
    [
        ("w /sys", "q /sys", "tmpfiles-policy"),
        ("w /sys", "Z /sys", "tmpfiles-policy"),
        ("w /sys", "j /sys", "tmpfiles-syntax"),
        ("w /sys", "w sys", "tmpfiles-syntax"),
        ("- - - - 1000", "- - - -", "tmpfiles-syntax"),
        ("- - - - 1000", "- - - - lots", "tmpfiles-value"),
        ("- - - - 1000", "- - - - 4", "tmpfiles-value"),
        ("parameters/usbfs_memory_mb", "parameters/usbfs_other", "tmpfiles-policy"),
    ],
)
def test_a_broken_tmpfiles_file_is_a_finding(old: str, new: str, rule: str) -> None:
    findings = lint_tmpfiles("f", mutate(rendered(TMPFILES), old, new))
    assert rule in rules(findings)


@pytest.mark.parametrize(
    ("old", "new", "rule"),
    [
        ("Storage=volatile", "Storage=persistent", "journald-policy"),
        ("Storage=volatile\n", "", "journald-policy"),
        ("RuntimeMaxUse=64M", "RuntimeMaxUsage=64M", "journald-key"),
        ("[Journal]", "[Jurnal]", "journald-section"),
        ("[Journal]", "Storage=volatile\n[Journal]", "unit-syntax"),
    ],
)
def test_a_broken_journald_file_is_a_finding(old: str, new: str, rule: str) -> None:
    findings = lint_journald("f", mutate(rendered(JOURNALD), old, new))
    assert rule in rules(findings)


def test_chrony_gets_one_source_line_for_each_source() -> None:
    text = rendered(CHRONY)
    assert "server time.example.org iburst" in text
    assert placeholders(text) == set()


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("makestep 1 3\nserver a.example.org\n", "chrony-directive"),
        ("# only a comment\n", "chrony-policy"),
        ("server\n", "chrony-syntax"),
        ("bindcmdaddress 192.0.2.1\nserver a.example.org\n", "chrony-directive"),
    ],
)
def test_a_broken_chrony_fragment_is_a_finding(text: str, rule: str) -> None:
    assert rule in rules(lint_chrony("f", text))


def test_chrony_accepts_pools_and_peers() -> None:
    assert lint_chrony("f", "pool pool.example.org iburst\npeer 192.0.2.7\n") == []


@pytest.mark.parametrize(
    ("old", "new", "rule"),
    [
        (
            "polkit.addRule(function (action, subject) {",
            "polkit.addRule(function (action, subject)",
            "polkit-syntax",
        ),
        ("polkit.addRule(", "polkit.add(", "polkit-syntax"),
        ('subject.user !== "@USER@"', 'subject.user === "@USER@"', "polkit-policy"),
        ("});\n", "}\n", "polkit-syntax"),
    ],
)
def test_a_broken_polkit_rule_is_a_finding(old: str, new: str, rule: str) -> None:
    assert rule in rules(lint_polkit("f", mutate(POLKIT, old, new)))


def test_the_polkit_rule_names_only_the_service_user_and_the_seeingmon_units() -> None:
    assert 'subject.user !== "@USER@"' in POLKIT
    assert "seeingmon-(acquire|core|web)" in POLKIT
    assert "NOT_HANDLED" in POLKIT
