#!/usr/bin/env bash
# Build the release artifacts of seeingmon on the development machine: a wheel, and a file of
# pinned requirements with hashes. Run it from any directory. It needs uv.
#
# The installer on the Raspberry Pi installs the pinned requirements first, with
# --require-hashes, and then the wheel with --no-deps. So the wheel and the requirements file
# must come from the same checkout. push.sh runs this script and copies both files to the Pi.
set -euo pipefail

PROGRAM=${0##*/}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

usage() {
  cat <<EOF
Usage: $PROGRAM --out-dir DIR [--python PATH] [--dry-run]

Build the wheel and the hashed requirements of seeingmon with uv.

Parameters:
  --out-dir DIR   the directory for the wheel and for requirements.txt (required). The script
                  creates it, and it replaces the files of an earlier build there.
  --python PATH   the Python interpreter that uv uses. Without it, uv looks for one.
  --dry-run       print the commands, and run none of them.
  -h, --help      print this text.

The script sets UV_PYTHON_DOWNLOADS=never unless you set it, so that uv never downloads an
interpreter.
EOF
}

usage_error() {
  printf '%s: %s\n' "$PROGRAM" "$*" >&2
  printf 'Run "%s --help" for the usage.\n' "$PROGRAM" >&2
  exit 2
}

die() {
  printf '%s: %s\n' "$PROGRAM" "$*" >&2
  exit 1
}

# Check that an option has a value. Call it as: need_value "$@"
need_value() {
  [ "$#" -ge 2 ] || usage_error "the option $1 needs a value"
  case $2 in
    --*) usage_error "the option $1 needs a value, not the option $2" ;;
  esac
}

# Check that a required parameter has a value. Call it as: require VARIABLE --option
require() {
  [ -n "${!1}" ] || usage_error "missing required parameter: $2"
}

# Run a command, or print it with --dry-run.
run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

OUT_DIR=''
PYTHON=''
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case $1 in
    --out-dir) need_value "$@"; OUT_DIR=$2; shift 2 ;;
    --python) need_value "$@"; PYTHON=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage_error "unknown option: $1" ;;
  esac
done

require OUT_DIR --out-dir

if [ "$DRY_RUN" -eq 0 ]; then
  command -v uv >/dev/null 2>&1 || die "uv is not installed: see https://docs.astral.sh/uv/"
  mkdir -p -- "$OUT_DIR"
fi

# The project needs these uv flags.
#
#   uv build --wheel --out-dir DIR REPO
#     --wheel      Build the wheel and no source distribution. The installer needs only the wheel.
#
#   uv export --project REPO --locked --no-dev --no-emit-project --all-extras
#             --format requirements-txt --no-header --output-file DIR/requirements.txt
#     --locked           Fail when uv.lock is out of date, so a release never carries stale pins.
#     --no-dev           Leave out the dev dependency group (tests and linters).
#     --no-emit-project  Leave out the project itself. The installer adds the wheel at the end.
#     --all-extras       Include the extras of the project (fast, survey, web, and timescale).
#     --format requirements-txt
#                        The pip format. uv writes the hashes of every file by default, and the
#                        universal lock covers Windows x64, Linux x64, and Linux arm64, so the
#                        same file installs on the Raspberry Pi.
UV_ARGS=()
if [ -n "$PYTHON" ]; then
  UV_ARGS+=(--python "$PYTHON")
fi
export UV_PYTHON_DOWNLOADS=${UV_PYTHON_DOWNLOADS:-never}

run rm -f -- "$OUT_DIR"/seeingmon-*.whl "$OUT_DIR/requirements.txt"
run uv build --wheel --out-dir "$OUT_DIR" "${UV_ARGS[@]}" "$REPO_ROOT"
run uv export --project "$REPO_ROOT" --locked --no-dev --no-emit-project --all-extras \
  --format requirements-txt --no-header --output-file "$OUT_DIR/requirements.txt" "${UV_ARGS[@]}"

if [ "$DRY_RUN" -eq 1 ]; then
  exit 0
fi

wheels=("$OUT_DIR"/seeingmon-*.whl)
if [ "${#wheels[@]}" -ne 1 ] || [ ! -f "${wheels[0]}" ]; then
  die "expected one seeingmon wheel in $OUT_DIR"
fi
grep -q -- '--hash=sha256:' "$OUT_DIR/requirements.txt" ||
  die "the requirements file has no hashes: $OUT_DIR/requirements.txt"

printf 'Wheel:         %s\n' "${wheels[0]}"
printf 'Requirements:  %s\n' "$OUT_DIR/requirements.txt"
