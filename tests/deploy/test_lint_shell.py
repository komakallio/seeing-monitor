"""The structural checks for shell scripts in `tools/lint_deploy.py`.

Each check has a negative test: a script that breaks the rule, and the finding that proves the
check can fail. The sample scripts assemble paths and addresses from pieces, so this file never
holds a literal that `tools/check_repo.py` would report.
"""

from __future__ import annotations

import pytest

from tests.deploy.helpers import mutate, rules
from tools.lint_deploy import (
    Finding,
    find_unquoted_expansions,
    lint_shell,
    shell_facts,
)

GOOD_SCRIPT = """\
#!/usr/bin/env bash
# A sample script for the linter tests.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sample.sh --prefix DIR [--keep N]
  --prefix DIR   the installation prefix (required)
  --keep N       how many releases to keep
EOF
}

usage_error() {
  printf '%s\\n' "$*" >&2
  exit 2
}

require() {
  [ -n "${!1}" ] || usage_error "missing required parameter: $2"
}

PREFIX=''
KEEP=2

while [ "$#" -gt 0 ]; do
  case $1 in
    --prefix) PREFIX=$2; shift 2 ;;
    --keep) KEEP=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage_error "unknown option: $1" ;;
  esac
done

require PREFIX --prefix

mkdir -p "$PREFIX/releases"
echo "keep $KEEP releases under ${PREFIX}"
"""


def check(text: str) -> list[Finding]:
    return lint_shell("sample.sh", text)


def test_the_good_script_passes() -> None:
    assert check(GOOD_SCRIPT) == []


# --- The scanner ------------------------------------------------------------------------------

UNQUOTED_CASES = [
    # Quoted expansions are fine.
    ('echo "$PREFIX"', []),
    ('echo "${PREFIX}"', []),
    ('echo "${PREFIX%/}"', []),
    ('echo "a $PREFIX b"', []),
    ('echo "$PREFIX"/x', []),
    ('printf "%s\\n" "$PREFIX/x"', []),
    ('cmd --opt="$PREFIX"', []),
    ('cmd > "$PREFIX"', []),
    ('for x in "$PREFIX"; do echo "$x"; done', []),
    ('if [ -z "$PREFIX" ]; then echo; fi', []),
    ('ARR+=("$PREFIX")', []),
    # Unquoted expansions are findings, in every form.
    ("echo $PREFIX", ["PREFIX"]),
    ("echo ${PREFIX}", ["PREFIX"]),
    ("echo ${PREFIX%/}", ["PREFIX"]),
    ("echo $PREFIX/x", ["PREFIX"]),
    ("echo ${PREFIX}x", ["PREFIX"]),
    ("[ -n $PREFIX ]", ["PREFIX"]),
    ("if ! [ -d $PREFIX ]; then :; fi", ["PREFIX"]),
    ('for x in $PREFIX; do echo "$x"; done', ["PREFIX"]),
    ("cmd > $PREFIX", ["PREFIX"]),
    ("cmd X=$PREFIX", ["PREFIX"]),
    ("cmd --opt=$PREFIX", ["PREFIX"]),
    ("exit $PREFIX", ["PREFIX"]),
    ("ARR=($PREFIX)", ["PREFIX"]),
    ('echo "${ARR[@]}" $PREFIX', ["PREFIX"]),
    ("f() { echo $PREFIX; }", ["PREFIX"]),
    ("(echo $PREFIX)", ["PREFIX"]),
    ("while read -r l; do echo $PREFIX; done", ["PREFIX"]),
    ("cat <<< $PREFIX", ["PREFIX"]),
    ("echo ${PREFIX:-$OTHER}", ["PREFIX", "OTHER"]),
    ("echo a#b $PREFIX", ["PREFIX"]),
    ("(( x = 1 << 2 )); echo $PREFIX", ["PREFIX"]),
    ("a=1\nb=$PREFIX\necho $OTHER", ["OTHER"]),
    # Command substitutions start a new command, whatever surrounds them.
    ("echo $(cmd $PREFIX)", ["PREFIX"]),
    ('echo "$(cmd $PREFIX)"', ["PREFIX"]),
    ('echo "$(cmd "$PREFIX")"', []),
    ("echo `cmd $PREFIX`", ["PREFIX"]),
    ('echo "`cmd $PREFIX`"', ["PREFIX"]),
    # Nested quotes inside a parameter expansion.
    ('echo "${PREFIX:-$OTHER}"', []),
    ('echo "${PREFIX:-"$OTHER"}"', []),
    ('x=${y:-"a b"}; echo $PREFIX', ["PREFIX"]),
    # Where bash does not split words, an unquoted expansion is fine.
    ("X=$PREFIX", []),
    ("X=${PREFIX:-a}", []),
    ("local X=$PREFIX", []),
    ("export X=$PREFIX", []),
    ("X=1 Y=$PREFIX cmd", []),
    ("case $PREFIX in a) echo hi ;; esac", []),
    ('case "$1" in --x) PREFIX=$2; shift 2 ;; esac', []),
    ("[[ $PREFIX == a ]] && echo", []),
    ("echo $((PREFIX + 1))", []),
    ("echo ${#PREFIX}", []),
    ('x=$(cmd "$PREFIX") ; y=$PREFIX', []),
    # Text that is not code.
    ("echo '$PREFIX'", []),
    ("echo \\$PREFIX", []),
    ("# $PREFIX in a comment\necho ok", []),
    ('echo "a" # $PREFIX', []),
    ("trap 'rm -f \"$PREFIX\"' EXIT", []),
    ("echo $'a\\'b' $PREFIX", ["PREFIX"]),
    # Here documents.
    ("cat <<EOF\n$PREFIX\nEOF\necho $PREFIX", ["PREFIX"]),
    ("cat <<'EOF'\n$PREFIX\nEOF\necho ok", []),
    ("cat <<-EOF\n\t$PREFIX\n\tEOF\necho ok", []),
    # A whole option parser.
    (
        'while [ "$#" -gt 0 ]; do case $1 in --prefix) PREFIX=$2; shift 2 ;; '
        '*) echo "$1" ;; esac; done',
        [],
    ),
]


@pytest.mark.parametrize(("script", "expected"), UNQUOTED_CASES)
def test_the_scanner_finds_the_unquoted_expansions(script: str, expected: list[str]) -> None:
    found = find_unquoted_expansions(script, {"PREFIX", "OTHER"})
    assert [expansion.name for expansion in found] == expected


def test_the_scanner_ignores_variables_that_are_not_parameters() -> None:
    assert find_unquoted_expansions("echo $other $HOME", {"PREFIX"}) == []


def test_the_scanner_counts_lines() -> None:
    found = find_unquoted_expansions("a=1\n# c\ncat <<EOF\nx\nEOF\necho $PREFIX\n", {"PREFIX"})
    assert [(e.name, e.line) for e in found] == [("PREFIX", 6)]


# --- The parameters of a script ---------------------------------------------------------------


def test_facts_name_the_options_variables_and_required_parameters() -> None:
    facts = shell_facts(GOOD_SCRIPT)
    assert set(facts.options) == {"--prefix", "--keep", "--help"}
    assert facts.variables == {"PREFIX": ["--prefix"], "KEEP": ["--keep"]}
    assert facts.required == {"PREFIX": "--prefix"}


def test_facts_follow_an_arm_that_spans_lines() -> None:
    script = 'case $1 in\n  --time-source)\n    SOURCES+=("$2")\n    shift 2\n    ;;\nesac\n'
    facts = shell_facts(script)
    assert facts.variables == {"SOURCES": ["--time-source"]}


# --- Negative tests, one for each check ---------------------------------------------------------


def test_an_unquoted_parameter_is_a_finding() -> None:
    text = mutate(GOOD_SCRIPT, 'mkdir -p "$PREFIX/releases"', "mkdir -p $PREFIX/releases")
    findings = check(text)
    assert rules(findings) == {"shell-unquoted-parameter"}
    assert findings[0].line == text.splitlines().index("mkdir -p $PREFIX/releases") + 1


@pytest.mark.parametrize(
    "replacement",
    ["PREFIX=/opt/seeingmon", 'PREFIX="/opt/seeingmon"', "PREFIX=prefix  # a default"],
)
def test_a_default_for_a_required_parameter_is_a_finding(replacement: str) -> None:
    findings = check(mutate(GOOD_SCRIPT, "PREFIX=''", replacement))
    assert rules(findings) == {"shell-required-default"}


@pytest.mark.parametrize("empty", ["PREFIX=", 'PREFIX=""', "PREFIX=''  # no default"])
def test_an_empty_initial_value_is_not_a_default(empty: str) -> None:
    assert check(mutate(GOOD_SCRIPT, "PREFIX=''", empty)) == []


@pytest.mark.parametrize("expansion", ['"${PREFIX:-/opt/x}"', '"${PREFIX:=/opt/x}"'])
def test_a_default_expansion_of_a_required_parameter_is_a_finding(expansion: str) -> None:
    findings = check(mutate(GOOD_SCRIPT, '"$PREFIX/releases"', expansion))
    assert rules(findings) == {"shell-required-default"}


def test_an_optional_parameter_may_have_a_default() -> None:
    assert check(mutate(GOOD_SCRIPT, "KEEP=2", "KEEP=5")) == []


def test_a_required_parameter_without_an_option_is_a_finding() -> None:
    findings = check(mutate(GOOD_SCRIPT, "require PREFIX --prefix", "require MISSING --missing"))
    assert "shell-require" in rules(findings)


def test_a_missing_parameter_must_say_which_one() -> None:
    text = mutate(GOOD_SCRIPT, "missing required parameter", "bad parameter")
    assert rules(check(text)) == {"shell-require"}


def test_an_option_must_appear_in_the_usage() -> None:
    text = mutate(GOOD_SCRIPT, "  --keep N       how many releases to keep\n", "")
    text = mutate(text, "--prefix DIR [--keep N]", "--prefix DIR")
    findings = check(text)
    assert rules(findings) == {"shell-undocumented-option"}
    assert "--keep" in findings[0].message


@pytest.mark.parametrize(
    ("first_command", "message"),
    [
        ("set -e", "set -u"),
        ("set -u", "set -e"),
        ("set -eu", "pipefail"),
        ("set -eo pipefail", "set -u"),
        ("echo hello", "the first command"),
    ],
)
def test_a_bash_script_needs_the_strict_mode(first_command: str, message: str) -> None:
    findings = check(mutate(GOOD_SCRIPT, "set -euo pipefail", first_command))
    assert rules(findings) == {"shell-strict-mode"}
    assert message in findings[0].message


@pytest.mark.parametrize(
    "first_command",
    ["set -euo pipefail", "set -e -u -o pipefail", "set -o errexit -o nounset -o pipefail"],
)
def test_the_strict_mode_has_several_spellings(first_command: str) -> None:
    assert check(mutate(GOOD_SCRIPT, "set -euo pipefail", first_command)) == []


def test_a_posix_script_needs_set_eu_and_no_pipefail() -> None:
    posix = mutate(GOOD_SCRIPT, "#!/usr/bin/env bash", "#!/bin/sh")
    findings = check(posix)
    assert "pipefail is not POSIX" in findings[0].message
    assert check(mutate(posix, "set -euo pipefail", "set -eu")) == []


def test_a_script_with_the_strict_mode_after_other_commands_is_a_finding() -> None:
    text = mutate(GOOD_SCRIPT, "set -euo pipefail\n", "umask 022\nset -euo pipefail\n")
    assert rules(check(text)) == {"shell-strict-mode"}


@pytest.mark.parametrize(
    "shebang", ["#!/usr/bin/python3", "# no shebang", "#!/usr/bin/env zsh", ""]
)
def test_a_script_needs_a_known_shebang(shebang: str) -> None:
    findings = check(mutate(GOOD_SCRIPT, "#!/usr/bin/env bash", shebang))
    assert "shell-shebang" in rules(findings)


@pytest.mark.parametrize("shebang", ["#!/usr/bin/env bash", "#!/bin/bash", "#!/usr/bin/env sh"])
def test_the_known_shebangs_pass(shebang: str) -> None:
    text = mutate(GOOD_SCRIPT, "#!/usr/bin/env bash", shebang)
    if shebang.endswith("sh") and not shebang.endswith("bash"):
        text = mutate(text, "set -euo pipefail", "set -eu")
    assert check(text) == []


def test_crlf_line_endings_are_a_finding() -> None:
    findings = check(GOOD_SCRIPT.replace("\n", "\r\n"))
    assert "shell-line-endings" in rules(findings)


def test_eval_is_a_finding() -> None:
    findings = check(GOOD_SCRIPT + 'eval "$command"\n')
    assert rules(findings) == {"shell-eval"}


def test_the_word_eval_in_a_comment_or_a_name_is_not_a_finding() -> None:
    assert check(GOOD_SCRIPT + "# eval is not used here\nretrieval=1\n") == []


@pytest.mark.parametrize(
    "line",
    [
        "cd " + "/" + "home" + "/someone/project",
        "cd " + "C" + ":" + "\\" + "Users" + "\\" + "someone",
        "host=" + ".".join(["192", "168", "1", "20"]),
        "ssh " + "pi@" + "seeingpi" + " uptime",
        "name=seeing-pi" + ".lan",
    ],
)
def test_a_developer_path_or_an_address_is_a_finding(line: str) -> None:
    findings = check(GOOD_SCRIPT + line + "\n")
    assert rules(findings) == {"shell-private-value"}


def test_the_allow_marker_silences_a_private_value() -> None:
    line = "cd " + "/" + "home" + "/someone/project  # repo-check: allow"
    assert check(GOOD_SCRIPT + line + "\n") == []


def test_documentation_addresses_and_placeholder_hosts_pass() -> None:
    text = GOOD_SCRIPT + "# ssh user@host or 192.0.2.10 or time.example.org\n"
    assert check(text) == []


def test_a_script_without_parameters_needs_no_options() -> None:
    script = '#!/usr/bin/env bash\nset -euo pipefail\necho "hello"\n'
    assert check(script) == []


def test_a_finding_prints_its_place_and_rule() -> None:
    finding = Finding("deploy/x.sh", 7, "shell-eval", "avoid eval")
    assert str(finding) == "deploy/x.sh:7: [shell-eval] avoid eval"
    assert str(Finding("deploy/x.sh", 0, "shell-eval", "avoid eval")) == (
        "deploy/x.sh: [shell-eval] avoid eval"
    )


def test_a_backtick_is_a_finding_even_inside_a_here_document() -> None:
    inside = mutate(GOOD_SCRIPT, "  --keep N       how many", "  --keep N       `date` how many")
    assert rules(check(inside)) == {"shell-backtick"}
    command = GOOD_SCRIPT + "stamp=`date`\n"
    assert rules(check(command)) == {"shell-backtick"}


def test_a_backtick_in_a_comment_is_not_a_finding() -> None:
    assert check(GOOD_SCRIPT + "# the `date` command\n") == []
