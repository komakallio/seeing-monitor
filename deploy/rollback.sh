#!/usr/bin/env bash
# Go back to the previous seeingmon release: point the current link at it, and restart the
# services. Run it as root, on the Raspberry Pi. The installer keeps the previous release for this,
# so the switch takes seconds. Run the script again to go forward to the release that you left.
#
# The script switches the code only. The systemd units and the configuration stay as they are.
set -euo pipefail

PROGRAM=${0##*/}
SERVICES=(seeingmon-acquire.service seeingmon-core.service seeingmon-web.service)

usage() {
  cat <<EOF
Usage: $PROGRAM --prefix DIR [--no-restart] [--dry-run]

Point DIR/current at the previous release, and restart the services.

Parameters:
  --prefix DIR    the installation prefix that you gave the installer (required).
  --no-restart    switch the link and leave the services alone.
  --dry-run       print what the script would do, and change nothing.
  -h, --help      print this text.
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

say() {
  printf '==> %s\n' "$*"
}

# Check that an option has a value that is not another option. Call it as: need_value "$@"
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

PREFIX=''
NO_RESTART=0
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case $1 in
    --prefix) need_value "$@"; PREFIX=$2; shift 2 ;;
    --no-restart) NO_RESTART=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage_error "unknown option: $1" ;;
  esac
done

require PREFIX --prefix
[[ $PREFIX =~ ^/[A-Za-z0-9._+/-]+$ ]] || usage_error "--prefix needs an absolute path of plain characters: $PREFIX"

CURRENT_LINK=$PREFIX/current
PREVIOUS_LINK=$PREFIX/previous

[ -L "$CURRENT_LINK" ] || die "$CURRENT_LINK is not a symbolic link: there is no installation to roll back"
[ -L "$PREVIOUS_LINK" ] || die "there is no previous release in $PREFIX to go back to"
CURRENT_TARGET=$(readlink -- "$CURRENT_LINK")
PREVIOUS_TARGET=$(readlink -- "$PREVIOUS_LINK")
[ "$CURRENT_TARGET" != "$PREVIOUS_TARGET" ] || die "the current and the previous release are the same"
[ -f "$PREFIX/$PREVIOUS_TARGET/.installed" ] ||
  die "the previous release ($PREVIOUS_TARGET) is missing or unfinished"

say "the current release is $CURRENT_TARGET, and the previous release is $PREVIOUS_TARGET"
if [ "$DRY_RUN" -eq 1 ]; then
  say "dry run: the script would point $CURRENT_LINK at $PREVIOUS_TARGET, and $PREVIOUS_LINK at $CURRENT_TARGET"
  if [ "$NO_RESTART" -eq 0 ]; then say "dry run: it would restart seeingmon.target"; fi
  exit 0
fi

[ "$(id -u)" -eq 0 ] || die "run the script as root: use sudo"

# Point a symbolic link at a target in one step. Call it as: replace_link TARGET LINK
replace_link() {
  ln -sfn -- "$1" "$2.new"
  mv -T -- "$2.new" "$2"
}

replace_link "$PREVIOUS_TARGET" "$CURRENT_LINK"
replace_link "$CURRENT_TARGET" "$PREVIOUS_LINK"
say "pointed $CURRENT_LINK at $PREVIOUS_TARGET"

if [ "$NO_RESTART" -eq 1 ]; then
  say "left the services alone: restart them with: systemctl restart seeingmon.target"
  exit 0
fi

systemctl reset-failed "${SERVICES[@]}" 2>/dev/null || true
say "restarting the services"
systemctl restart seeingmon.target || true
FAILED=()
for unit in "${SERVICES[@]}"; do
  systemctl is-active --quiet "$unit" || FAILED+=("$unit")
done
if [ "${#FAILED[@]}" -gt 0 ]; then
  die "these units are not active after the rollback: ${FAILED[*]}. Read why with: journalctl -u ${FAILED[0]} -n 50"
fi
say "the services run release $PREVIOUS_TARGET"
