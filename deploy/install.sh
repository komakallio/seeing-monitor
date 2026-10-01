#!/usr/bin/env bash
# Install or upgrade a seeingmon release on a Raspberry Pi. Run it as root, on the Pi.
#
# push.sh copies this directory and your files to the Pi, and it runs this script there. You can
# also run the script yourself from a copy of the deploy/ directory. It is safe to run again: it
# changes only what differs, it restarts the services only when something changed, and it stops
# with a message when it finds a state that it does not expect.
#
# The script has two parts. It checks the parameters and the system first, and it changes nothing
# until every check passes. Then it installs, step by step, and it prints what it did.
set -euo pipefail

PROGRAM=${0##*/}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MARKER='Managed by the seeingmon installer'
SERVICES=(seeingmon-acquire.service seeingmon-core.service seeingmon-web.service)
UNITS=(seeingmon.target "${SERVICES[@]}" 'seeingmon-failed@.service')
CHRONY_DROPIN_DIR=/etc/chrony/conf.d
HOME_DIR=/var/lib/seeingmon

usage() {
  cat <<EOF
Usage: $PROGRAM --prefix DIR --user NAME --data-dir DIR --config-dir DIR \\
          --wheel FILE --requirements FILE --time-source NAME [options]

Install a seeingmon release. Run it as root, on the Raspberry Pi. Every path and name below has
no default: you pass it.

Required parameters:
  --prefix DIR         the installation prefix, for example /opt/seeingmon. Each release gets its
                       own virtual environment in DIR/releases, and DIR/current points to one.
  --user NAME          the account that runs the services. The installer creates it as a system
                       account without a login, and with a group of the same name.
  --data-dir DIR       the data directory. Use a folder on its own partition. The installer does
                       not partition anything.
  --config-dir DIR     the configuration directory. It holds local/config.toml, the connection
                       key, and the environment file.
  --wheel FILE         the seeingmon wheel that build.sh made.
  --requirements FILE  the hashed requirements file that build.sh made.
  --time-source NAME   a time source for chrony. Repeat the option for more sources.

Optional parameters:
  --no-time-config     leave chrony alone. Use it instead of --time-source.
  --sdk-archive FILE   the vendor SDK archive (a tar file). The installer checks its checksum
                       and installs it privately, and it points the services at the library.
  --sdk-sha256 HEX     the checksum that the archive must have. Required with --sdk-archive.
  --local-config FILE  your local configuration. The installer copies it to
                       CONFIG_DIR/local/config.toml, readable by the service user only.
  --connection-key-file FILE
                       the key that the three services share. Without it, the installer makes a
                       key at the first install and keeps it.
  --token-hash-file FILE
                       the hash of the API token, which the command web hash-token makes. The web
                       service loads it as a systemd credential. Without it, the installer keeps
                       an empty file, and web has no hash until you give one.
  --env-file FILE      a file of NAME=value lines that the services read. Put secrets in it.
  --wheelhouse DIR     install from the wheels in DIR and not from the internet.
  --python PATH        the Python interpreter that makes the virtual environment (default:
                       python3). It must be 3.11 or later.
  --usbfs-memory-mb N  the USB buffer size in MB (default: 1000).
  --keep N             how many releases to keep, the current and the previous included (default:
                       2).
  --supervisor-actions let the service user restart the seeingmon units and reboot the Pi, for
                       the last steps of the camera recovery ladder. It adds a polkit rule.
  --system-root DIR    put every system file (units, udev rules, and so on) under DIR. For tests
                       and for an image that you mounted. The installer still runs its commands.
  --dry-run            print the plan, and change nothing.
  -h, --help           print this text.
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

CHANGES=()
WARNINGS=()

say() {
  printf '==> %s\n' "$*"
}

note_change() {
  CHANGES+=("$*")
  say "$*"
}

warn() {
  WARNINGS+=("$*")
  printf '%s: warning: %s\n' "$PROGRAM" "$*" >&2
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

# Check an absolute path of plain characters. Call it as: check_path --option "$VALUE"
check_path() {
  [[ $2 =~ ^/[A-Za-z0-9._+/-]*$ ]] || usage_error "$1 needs an absolute path of plain characters: $2"
  [[ $2 != *..* ]] || usage_error "$1 must not contain '..': $2"
  [[ $2 == / || $2 != */ ]] || usage_error "$1 must not end with a slash: $2"
}

check_file() {
  [ -f "$2" ] || usage_error "$1 names a file that does not exist: $2"
}

check_host() {
  [[ $2 =~ ^[A-Za-z0-9][A-Za-z0-9._:-]*$ ]] || usage_error "$1 needs a host name or address: $2"
}

# Check a whole number of at least $3. Call it as: check_number --option "$VALUE" MINIMUM
check_number() {
  [[ $2 =~ ^[0-9]{1,6}$ ]] || usage_error "$1 needs a whole number: $2"
  [ "$2" -ge "$3" ] || usage_error "$1 needs a number of at least $3: $2"
}

# --- Parameters -----------------------------------------------------------------------------------

PREFIX=''
SERVICE_USER=''
DATA_DIR=''
CONFIG_DIR=''
WHEEL=''
REQUIREMENTS=''
NO_TIME_CONFIG=0
TIME_SOURCES=()
SDK_ARCHIVE=''
SDK_SHA256=''
LOCAL_CONFIG=''
CONNECTION_KEY_FILE=''
TOKEN_HASH_FILE=''
ENV_FILE=''
WHEELHOUSE=''
PYTHON=''
USBFS_MEMORY_MB=1000
KEEP=2
SUPERVISOR_ACTIONS=0
SYSTEM_ROOT=''
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case $1 in
    --prefix) need_value "$@"; PREFIX=$2; shift 2 ;;
    --user) need_value "$@"; SERVICE_USER=$2; shift 2 ;;
    --data-dir) need_value "$@"; DATA_DIR=$2; shift 2 ;;
    --config-dir) need_value "$@"; CONFIG_DIR=$2; shift 2 ;;
    --wheel) need_value "$@"; WHEEL=$2; shift 2 ;;
    --requirements) need_value "$@"; REQUIREMENTS=$2; shift 2 ;;
    --time-source) need_value "$@"; TIME_SOURCES+=("$2"); shift 2 ;;
    --no-time-config) NO_TIME_CONFIG=1; shift ;;
    --sdk-archive) need_value "$@"; SDK_ARCHIVE=$2; shift 2 ;;
    --sdk-sha256) need_value "$@"; SDK_SHA256=$2; shift 2 ;;
    --local-config) need_value "$@"; LOCAL_CONFIG=$2; shift 2 ;;
    --connection-key-file) need_value "$@"; CONNECTION_KEY_FILE=$2; shift 2 ;;
    --token-hash-file) need_value "$@"; TOKEN_HASH_FILE=$2; shift 2 ;;
    --env-file) need_value "$@"; ENV_FILE=$2; shift 2 ;;
    --wheelhouse) need_value "$@"; WHEELHOUSE=$2; shift 2 ;;
    --python) need_value "$@"; PYTHON=$2; shift 2 ;;
    --usbfs-memory-mb) need_value "$@"; USBFS_MEMORY_MB=$2; shift 2 ;;
    --keep) need_value "$@"; KEEP=$2; shift 2 ;;
    --supervisor-actions) SUPERVISOR_ACTIONS=1; shift ;;
    --system-root) need_value "$@"; SYSTEM_ROOT=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage_error "unknown option: $1" ;;
  esac
done

require PREFIX --prefix
require SERVICE_USER --user
require DATA_DIR --data-dir
require CONFIG_DIR --config-dir
require WHEEL --wheel
require REQUIREMENTS --requirements

# --- Check the parameters --------------------------------------------------------------------------

check_path --prefix "$PREFIX"
check_path --data-dir "$DATA_DIR"
check_path --config-dir "$CONFIG_DIR"
[[ $SERVICE_USER =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] ||
  usage_error "--user needs a plain account name of lowercase letters, digits, '_' and '-': $SERVICE_USER"
check_file --wheel "$WHEEL"
check_file --requirements "$REQUIREMENTS"

# The prefix, the data directory, and the configuration directory must not be a system directory,
# and none may hold another.
for path in "$PREFIX" "$DATA_DIR" "$CONFIG_DIR"; do
  case $path in
    / | /bin | /boot | /dev | /etc | /home | /lib | /media | /mnt | /opt | /proc | /root | /run | \
      /sbin | /srv | /sys | /tmp | /usr | /var)
      usage_error "$path is a system directory: name a folder of your own inside it"
      ;;
    /home/* | /root/* | /run/user/*)
      usage_error "the units hide $path from the services (ProtectHome): choose another folder"
      ;;
  esac
done
nested() {
  [[ $1 == "$2" || $1 == "$2"/* || $2 == "$1"/* ]]
}
if nested "$PREFIX" "$DATA_DIR"; then usage_error "--prefix and --data-dir must not hold each other"; fi
if nested "$PREFIX" "$CONFIG_DIR"; then usage_error "--prefix and --config-dir must not hold each other"; fi
if nested "$DATA_DIR" "$CONFIG_DIR"; then usage_error "--data-dir and --config-dir must not hold each other"; fi

if [ "$NO_TIME_CONFIG" -eq 0 ]; then
  [ "${#TIME_SOURCES[@]}" -gt 0 ] ||
    usage_error "missing required parameter: --time-source (or --no-time-config)"
elif [ "${#TIME_SOURCES[@]}" -gt 0 ]; then
  usage_error "--time-source and --no-time-config exclude each other"
fi
for time_source in "${TIME_SOURCES[@]}"; do
  check_host --time-source "$time_source"
done

if [ -n "$SDK_ARCHIVE" ] || [ -n "$SDK_SHA256" ]; then
  [ -n "$SDK_ARCHIVE" ] || usage_error "--sdk-sha256 needs --sdk-archive"
  [ -n "$SDK_SHA256" ] || usage_error "--sdk-archive needs --sdk-sha256"
  check_file --sdk-archive "$SDK_ARCHIVE"
  [[ $SDK_SHA256 =~ ^[0-9a-f]{64}$ ]] || usage_error "--sdk-sha256 needs 64 lowercase hex digits"
fi
if [ -n "$LOCAL_CONFIG" ]; then check_file --local-config "$LOCAL_CONFIG"; fi
if [ -n "$CONNECTION_KEY_FILE" ]; then check_file --connection-key-file "$CONNECTION_KEY_FILE"; fi
if [ -n "$TOKEN_HASH_FILE" ]; then check_file --token-hash-file "$TOKEN_HASH_FILE"; fi
if [ -n "$ENV_FILE" ]; then check_file --env-file "$ENV_FILE"; fi
if [ -n "$WHEELHOUSE" ]; then
  check_path --wheelhouse "$WHEELHOUSE"
  [ -d "$WHEELHOUSE" ] || usage_error "--wheelhouse names a folder that does not exist: $WHEELHOUSE"
fi
if [ -n "$SYSTEM_ROOT" ]; then
  check_path --system-root "$SYSTEM_ROOT"
  [ -d "$SYSTEM_ROOT" ] || usage_error "--system-root names a folder that does not exist: $SYSTEM_ROOT"
fi
check_number --usbfs-memory-mb "$USBFS_MEMORY_MB" 16
check_number --keep "$KEEP" 2

WHEEL_NAME=${WHEEL##*/}
case $WHEEL_NAME in
  seeingmon-*-py3-none-any.whl) ;;
  *) usage_error "--wheel must be a seeingmon wheel (seeingmon-VERSION-py3-none-any.whl): $WHEEL_NAME" ;;
esac
VERSION=${WHEEL_NAME#seeingmon-}
VERSION=${VERSION%%-*}
[[ $VERSION =~ ^[0-9][0-9A-Za-z._+!]*$ ]] || usage_error "the wheel has an unexpected version: $VERSION"
grep -q -- '--hash=sha256:' "$REQUIREMENTS" ||
  usage_error "--requirements has no hashes: use the file that build.sh made"

# The release name holds the version and the first digits of a checksum over the wheel and the
# requirements. A build of the same version with other contents becomes another release, so an
# installed release never changes under a running service.
RELEASE_HASH=$(cat -- "$WHEEL" "$REQUIREMENTS" | sha256sum | cut -c1-12)
RELEASE_ID=$VERSION-$RELEASE_HASH

PYTHON_BIN=${PYTHON:-python3}
GROUP=$SERVICE_USER

# --- Where things go --------------------------------------------------------------------------------

RELEASES_DIR=$PREFIX/releases
RELEASE_DIR=$RELEASES_DIR/$RELEASE_ID
CURRENT_LINK=$PREFIX/current
PREVIOUS_LINK=$PREFIX/previous
BIN_DIR=$PREFIX/bin
SDK_ROOT=$PREFIX/sdk
CONFIG_LOCAL_DIR=$CONFIG_DIR/local
CONFIG_FILE=$CONFIG_LOCAL_DIR/config.toml
CREDENTIALS_DIR=$CONFIG_DIR/credentials
KEY_FILE=$CREDENTIALS_DIR/seeingmon-connection-key
TOKEN_FILE=$CREDENTIALS_DIR/seeingmon-token-hash
ENV_TARGET=$CONFIG_DIR/seeingmon.env
SDK_ENV_FILE=$CONFIG_DIR/sdk.env
UNIT_DIR=$SYSTEM_ROOT/etc/systemd/system
UDEV_FILE=$SYSTEM_ROOT/etc/udev/rules.d/99-seeingmon-asi.rules
TMPFILES_FILE=$SYSTEM_ROOT/etc/tmpfiles.d/seeingmon-usb.conf
JOURNALD_FILE=$SYSTEM_ROOT/etc/systemd/journald.conf.d/seeingmon.conf
CHRONY_CONF=$SYSTEM_ROOT/etc/chrony/chrony.conf
CHRONY_FILE=$SYSTEM_ROOT$CHRONY_DROPIN_DIR/seeingmon.conf
POLKIT_FILE=$SYSTEM_ROOT/etc/polkit-1/rules.d/50-seeingmon.rules

TIME_SOURCE_LINES=''
for time_source in "${TIME_SOURCES[@]}"; do
  TIME_SOURCE_LINES+="server $time_source iburst"$'\n'
done
TIME_SOURCE_LINES=${TIME_SOURCE_LINES%$'\n'}

# The files that the installer renders from templates, as triples: the template, the target, and
# the mode of the target.
MANAGED=()
add_managed() {
  MANAGED+=("$1" "$2" "$3")
}
for unit in "${UNITS[@]}"; do
  add_managed "$SCRIPT_DIR/systemd/$unit" "$UNIT_DIR/$unit" 0644
done
add_managed "$SCRIPT_DIR/udev/99-seeingmon-asi.rules" "$UDEV_FILE" 0644
add_managed "$SCRIPT_DIR/tmpfiles/seeingmon-usb.conf" "$TMPFILES_FILE" 0644
add_managed "$SCRIPT_DIR/journald/seeingmon.conf" "$JOURNALD_FILE" 0644
if [ "$NO_TIME_CONFIG" -eq 0 ]; then
  add_managed "$SCRIPT_DIR/chrony/seeingmon.conf" "$CHRONY_FILE" 0644
fi
if [ "$SUPERVISOR_ACTIONS" -eq 1 ]; then
  add_managed "$SCRIPT_DIR/polkit/50-seeingmon.rules" "$POLKIT_FILE" 0644
fi
add_managed "$SCRIPT_DIR/wrapper/seeingmon.sh" "$BIN_DIR/seeingmon" 0755

print_plan() {
  local time_plan='chrony is left alone' sdk_plan=none supervisor_plan=no
  if [ "${#TIME_SOURCES[@]}" -gt 0 ]; then time_plan=${TIME_SOURCES[*]}; fi
  if [ -n "$SDK_ARCHIVE" ]; then sdk_plan='from the archive, after the checksum matches'; fi
  if [ "$SUPERVISOR_ACTIONS" -eq 1 ]; then supervisor_plan='yes (a polkit rule)'; fi
  cat <<EOF
Release          $RELEASE_ID
Prefix           $PREFIX
Service user     $SERVICE_USER (a system account without a login)
Data directory   $DATA_DIR
Config directory $CONFIG_DIR
Python           $PYTHON_BIN
Time sources     $time_plan
Vendor SDK       $sdk_plan
Keep releases    $KEEP
Supervisor rule  $supervisor_plan
System root      ${SYSTEM_ROOT:-none (the real system)}

Steps:
   1. Check the system: root, systemd, Python 3.11 or later, and the programs that the steps use.
   2. Create the service user and the group $GROUP, and join the gpio group when the Pi has one.
   3. Create $PREFIX, $CONFIG_DIR, and $DATA_DIR. Partitioning is up to you.
   4. Install release $RELEASE_ID in $RELEASE_DIR/venv, from the hashed requirements
      (pip --require-hashes) and then the wheel (pip --no-deps).
   5. Install the vendor SDK in $SDK_ROOT, after it checks the checksum (only with --sdk-archive).
   6. Install the connection key, the environment file, and the local configuration, readable by
      the service user only.
   7. Install the systemd units, the camera udev rule, the USB buffer setting, the journald setting
      (the journal stays in RAM), and the chrony time sources.
   8. Install the wrapper $BIN_DIR/seeingmon and the rollback script $BIN_DIR/rollback.sh.
   9. Point $CURRENT_LINK at the new release, keep the old one for a rollback, and remove older
      releases beyond --keep.
  10. Reload systemd and udev, apply the settings, and enable and start the units.
EOF
}

if [ "$DRY_RUN" -eq 1 ]; then
  print_plan
  printf '\n%s: dry run: nothing changed.\n' "$PROGRAM"
  exit 0
fi

# --- From here on, the script changes the system ---------------------------------------------------

WORK_DIR=''
PARTIAL_RELEASE=''
LAST_CHANGED=0
UNITS_CHANGED=0
UDEV_CHANGED=0
JOURNALD_CHANGED=0
CHRONY_CHANGED=0
RESTART_NEEDED=0
STATUS=0

cleanup() {
  if [ -n "$PARTIAL_RELEASE" ]; then
    rm -rf -- "$PARTIAL_RELEASE"
  fi
  if [ -n "$WORK_DIR" ]; then
    rm -rf -- "$WORK_DIR"
  fi
}
trap cleanup EXIT

# Print a template with its @NAME@ placeholders filled in.
render() {
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line//@PREFIX@/"$PREFIX"}
    line=${line//@USER@/"$SERVICE_USER"}
    line=${line//@GROUP@/"$GROUP"}
    line=${line//@DATA_DIR@/"$DATA_DIR"}
    line=${line//@CONFIG_DIR@/"$CONFIG_DIR"}
    line=${line//@USBFS_MEMORY_MB@/"$USBFS_MEMORY_MB"}
    line=${line//@TIME_SOURCES@/"$TIME_SOURCE_LINES"}
    printf '%s\n' "$line"
  done <"$1"
}

# Install a file with the mode that you give. The file appears whole or not at all, and it never
# exists with a wider mode. Sets LAST_CHANGED. Call it as: install_file SOURCE TARGET MODE OWNER GROUP
install_file() {
  local source=$1 target=$2 mode=$3 owner=$4 group=$5
  LAST_CHANGED=0
  if [ -f "$target" ] && cmp -s -- "$source" "$target"; then
    chmod "$mode" "$target"
    chown "$owner:$group" "$target"
    say "$target is unchanged"
    return 0
  fi
  mkdir -p -- "${target%/*}"
  install -m "$mode" -- "$source" "$target.new"
  chown "$owner:$group" "$target.new"
  mv -f -- "$target.new" "$target"
  LAST_CHANGED=1
  note_change "installed $target"
}

# Render a template and install the result. A file that the installer did not write stays alone.
# Call it as: install_managed TEMPLATE TARGET MODE
install_managed() {
  local template=$1 target=$2 mode=$3 rendered
  rendered=$(mktemp "$WORK_DIR/render.XXXXXX")
  render "$template" >"$rendered"
  if grep -Eq '@[A-Z][A-Z0-9_]*@' "$rendered"; then
    die "a placeholder is left in ${template##*/} after the render"
  fi
  grep -q -- "$MARKER" "$rendered" || die "${template##*/} does not carry the line '$MARKER'"
  if [ -e "$target" ] && ! grep -q -- "$MARKER" "$target"; then
    die "$target exists and the installer did not write it: rename or remove it, and run again"
  fi
  install_file "$rendered" "$target" "$mode" root root
}

# --- Step 1: check the system ----------------------------------------------------------------------

check_system() {
  local tool
  local missing=()
  say "checking the system"
  if [ "$(id -u)" -ne 0 ]; then
    die "run the installer as root: use sudo, or use push.sh with a user that may use sudo"
  fi
  [ "$(uname -s)" = Linux ] || die "the installer runs on Linux (Raspberry Pi OS)"
  for tool in base64 chmod chown cmp cut find grep head install ln mktemp mv sha256sum sort tar tr \
    systemctl udevadm systemd-tmpfiles useradd usermod getent; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    die "these programs are missing: ${missing[*]}"
  fi
  command -v "$PYTHON_BIN" >/dev/null 2>&1 ||
    die "there is no Python interpreter named $PYTHON_BIN: install python3 and python3-venv"
  "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' ||
    die "$PYTHON_BIN is older than Python 3.11: use Raspberry Pi OS based on Debian 12 or later"
  "$PYTHON_BIN" -c 'import ensurepip, venv' >/dev/null 2>&1 ||
    die "$PYTHON_BIN cannot make virtual environments: install python3-venv"
  if [ "$NO_TIME_CONFIG" -eq 0 ]; then
    [ -f "$CHRONY_CONF" ] || die "chrony is not installed (no $CHRONY_CONF): install the chrony package"
    grep -Eq "^[[:space:]]*(confdir|include)[[:space:]]+$CHRONY_DROPIN_DIR" "$CHRONY_CONF" ||
      die "$CHRONY_CONF does not read $CHRONY_DROPIN_DIR: add the line 'confdir $CHRONY_DROPIN_DIR', or use --no-time-config"
  fi
  if [ "$SUPERVISOR_ACTIONS" -eq 1 ]; then
    [ -d "${POLKIT_FILE%/*}" ] || die "polkit is not installed (no ${POLKIT_FILE%/*}): install polkitd, or leave out --supervisor-actions"
  fi
  if [ -e "$PREFIX" ] && [ ! -d "$PREFIX" ]; then die "$PREFIX exists and is not a directory"; fi
  if [ -d "$PREFIX" ] && [ ! -d "$RELEASES_DIR" ] && [ -n "$(ls -A -- "$PREFIX")" ]; then
    die "$PREFIX is not empty and has no releases folder: it is not a seeingmon prefix"
  fi
  if { [ -e "$CURRENT_LINK" ] || [ -L "$CURRENT_LINK" ]; } && [ ! -L "$CURRENT_LINK" ]; then
    die "$CURRENT_LINK is not a symbolic link: move it away, and run again"
  fi
  if [ -e "$DATA_DIR" ] && [ ! -d "$DATA_DIR" ]; then die "$DATA_DIR exists and is not a directory"; fi
  if [ -e "$CONFIG_DIR" ] && [ ! -d "$CONFIG_DIR" ]; then die "$CONFIG_DIR exists and is not a directory"; fi
  check_managed_files
  check_connection_key
  check_sdk_archive
  WORK_DIR=$(mktemp -d)
}

# Check that every template is here, and that no file of the system that the installer would
# replace is the work of someone else.
check_managed_files() {
  local index=0 template target
  [ -f "$SCRIPT_DIR/rollback.sh" ] || die "the folder $SCRIPT_DIR has no rollback.sh: run the installer from a whole copy of deploy/"
  while [ "$index" -lt "${#MANAGED[@]}" ]; do
    template=${MANAGED[$index]}
    target=${MANAGED[$((index + 1))]}
    [ -f "$template" ] || die "the folder $SCRIPT_DIR lacks ${template#"$SCRIPT_DIR"/}: run the installer from a whole copy of deploy/"
    if [ -e "$target" ] && ! grep -q -- "$MARKER" "$target"; then
      die "$target exists and the installer did not write it: rename or remove it, and run again"
    fi
    index=$((index + 3))
  done
}

# Check the connection key that you gave: the services refuse a key of the wrong length.
check_connection_key() {
  local characters
  [ -n "$CONNECTION_KEY_FILE" ] || return 0
  characters=$(tr -d '[:space:]' <"$CONNECTION_KEY_FILE" | wc -c)
  if [ "$characters" -lt 16 ] || [ "$characters" -gt 1024 ]; then
    die "the connection key in $CONNECTION_KEY_FILE must have 16 to 1,024 characters, and it has $characters"
  fi
}

# Check the SDK archive before any change: its checksum, and the names of its members.
check_sdk_archive() {
  local actual members
  [ -n "$SDK_ARCHIVE" ] || return 0
  actual=$(sha256sum -- "$SDK_ARCHIVE" | cut -d' ' -f1)
  if [ "$actual" != "$SDK_SHA256" ]; then
    die "the SDK archive does not have the expected checksum: it has $actual, and you gave $SDK_SHA256"
  fi
  members=$(tar -tf "$SDK_ARCHIVE") || die "the SDK archive is not a tar archive that tar can read"
  if grep -Eq '(^|/)\.\.(/|$)|^/' <<<"$members"; then
    die "the SDK archive has a member with an absolute path or a '..' component"
  fi
}

# --- Step 2: the service user ----------------------------------------------------------------------

ensure_user() {
  local entry uid shell membership
  say "checking the service user $SERVICE_USER"
  if entry=$(getent passwd "$SERVICE_USER"); then
    IFS=: read -r _ _ uid _ _ _ shell <<<"$entry"
    case $shell in
      /usr/sbin/nologin | /sbin/nologin | /usr/bin/false | /bin/false) ;;
      *) die "the account $SERVICE_USER exists and has a login shell ($shell): choose another name for the service user" ;;
    esac
    if [ "$uid" -ge 1000 ]; then
      die "the account $SERVICE_USER exists and is not a system account (user ID $uid): choose another name"
    fi
    say "the account $SERVICE_USER exists"
  else
    useradd --system --user-group --no-create-home --home-dir "$HOME_DIR" \
      --shell /usr/sbin/nologin --comment 'seeingmon service' "$SERVICE_USER"
    note_change "created the system account $SERVICE_USER"
  fi
  GROUP=$(id -gn "$SERVICE_USER")
  if getent group gpio >/dev/null 2>&1; then
    membership=$(id -nG "$SERVICE_USER")
    if ! grep -Eqw gpio <<<"$membership"; then
      usermod -aG gpio "$SERVICE_USER"
      note_change "added $SERVICE_USER to the gpio group"
      RESTART_NEEDED=1
    fi
  else
    say "this system has no gpio group, so $SERVICE_USER has no GPIO access (the heater needs it)"
  fi
}

# --- Step 3: the directories -----------------------------------------------------------------------

make_directories() {
  local mount_point
  say "creating the directories"
  install -d -m 0755 -- "$PREFIX" "$RELEASES_DIR" "$BIN_DIR"
  install -d -m 0750 -- "$CONFIG_DIR" "$CONFIG_LOCAL_DIR" "$CREDENTIALS_DIR"
  chown "root:$GROUP" -- "$CONFIG_DIR" "$CONFIG_LOCAL_DIR" "$CREDENTIALS_DIR"
  install -d -m 0750 -- "$DATA_DIR"
  chown "$SERVICE_USER:$GROUP" -- "$DATA_DIR"
  command -v findmnt >/dev/null 2>&1 || return 0
  mount_point=$(findmnt -n -o TARGET --target "$DATA_DIR" 2>/dev/null || true)
  if [ "$mount_point" = / ]; then
    warn "$DATA_DIR is on the root file system. The design keeps data on its own partition. Create a partition, format it with ext4, mount it at $DATA_DIR through /etc/fstab, and run the installer again. The installer does not partition anything. The runbook has the steps."
  fi
  if [ "$(findmnt -n -o FSTYPE --target /tmp 2>/dev/null || true)" != tmpfs ]; then
    warn "/tmp is not a tmpfs, so the services write their temporary files to the SD card. Enable the tmpfs with: systemctl enable tmp.mount (and reboot)."
  fi
}

# --- Step 4: the release ---------------------------------------------------------------------------

install_release() {
  local venv_python=$RELEASE_DIR/venv/bin/python
  local link
  local source_args=()
  if [ -f "$RELEASE_DIR/.installed" ]; then
    say "release $RELEASE_ID is already installed"
    return 0
  fi
  if [ -e "$RELEASE_DIR" ] || [ -L "$RELEASE_DIR" ]; then
    for link in "$CURRENT_LINK" "$PREVIOUS_LINK"; do
      if [ -L "$link" ] && [ "$(readlink -- "$link")" = "releases/$RELEASE_ID" ]; then
        die "$RELEASE_DIR is unfinished, but $link points to it: look at it, and remove it by hand"
      fi
    done
    say "removing the unfinished release of an earlier run"
    rm -rf -- "$RELEASE_DIR"
  fi
  PARTIAL_RELEASE=$RELEASE_DIR
  install -d -m 0755 -- "$RELEASE_DIR"
  say "making the virtual environment for release $RELEASE_ID"
  "$PYTHON_BIN" -m venv "$RELEASE_DIR/venv"
  # Ignore the pip configuration of the system. Raspberry Pi OS adds a second package index there,
  # and the hashes in the requirements file fit the files of the standard index only.
  export PIP_CONFIG_FILE=/dev/null
  if [ -n "$WHEELHOUSE" ]; then
    source_args=(--no-index --find-links "$WHEELHOUSE")
  fi
  say "installing the pinned requirements, which pip checks against their hashes"
  "$venv_python" -m pip install --disable-pip-version-check --no-input --no-cache-dir \
    "${source_args[@]}" --require-hashes --only-binary=:all: --no-deps -r "$REQUIREMENTS"
  say "installing the wheel"
  "$venv_python" -m pip install --disable-pip-version-check --no-input --no-cache-dir \
    --no-index --no-deps -- "$WHEEL"
  "$venv_python" -m pip check --disable-pip-version-check ||
    die "the installed packages do not satisfy each other: make the release again with build.sh"
  PYTHONDONTWRITEBYTECODE=1 "$RELEASE_DIR/venv/bin/seeingmon" --version ||
    die "the new release does not start"
  printf '%s\n' "$RELEASE_ID" >"$RELEASE_DIR/.installed"
  PARTIAL_RELEASE=''
  note_change "installed release $RELEASE_ID"
}

# --- Step 5: the vendor SDK ------------------------------------------------------------------------

install_sdk() {
  local sdk_dir=$SDK_ROOT/${SDK_SHA256:0:12}
  local arch_dir library missing_libraries
  local content
  [ -n "$SDK_ARCHIVE" ] || return 0
  if [ -f "$sdk_dir/.installed" ]; then
    say "the vendor SDK is already installed in $sdk_dir"
  else
    install -d -m 0750 -- "$SDK_ROOT"
    chown "root:$GROUP" -- "$SDK_ROOT"
    rm -rf -- "$sdk_dir" "$sdk_dir.part"
    install -d -m 0750 -- "$sdk_dir.part"
    tar -xf "$SDK_ARCHIVE" -C "$sdk_dir.part" --no-same-owner --no-same-permissions
    chmod -R u=rwX,g=rX,o= -- "$sdk_dir.part"
    chown -R "root:$GROUP" -- "$sdk_dir.part"
    printf '%s\n' "$SDK_SHA256" >"$sdk_dir.part/.installed"
    mv -- "$sdk_dir.part" "$sdk_dir"
    note_change "installed the vendor SDK in $sdk_dir"
  fi
  case $(uname -m) in
    aarch64 | arm64) arch_dir=armv8 ;;
    armv7l | armv7) arch_dir=armv7 ;;
    armv6l) arch_dir=armv6 ;;
    x86_64 | amd64) arch_dir=x64 ;;
    *) die "the vendor SDK has no library for this machine: $(uname -m)" ;;
  esac
  library=$(find "$sdk_dir" -path "*/lib/$arch_dir/libASICamera2.so" -print -quit)
  if [ -z "$library" ]; then
    library=$(find "$sdk_dir" -path "*/lib/$arch_dir/libASICamera2.so.*" -print -quit)
  fi
  [ -n "$library" ] || die "the SDK archive has no libASICamera2.so for $arch_dir"
  missing_libraries=$(ldd "$library" 2>/dev/null | grep 'not found' || true)
  if [ -n "$missing_libraries" ]; then
    warn "the vendor library needs libraries that this system lacks (${missing_libraries//$'\n'/;}). The camera driver needs libusb: install the package libusb-1.0-0."
  fi
  content=$WORK_DIR/sdk.env
  printf '# %s. Changes are overwritten.\nSEEINGMON_ASI__LIBRARY_PATH=%s\n' "$MARKER" "$library" >"$content"
  install_file "$content" "$SDK_ENV_FILE" 0644 root root
  if [ "$LAST_CHANGED" -eq 1 ]; then RESTART_NEEDED=1; fi
}

# --- Step 6: secrets and the local configuration ---------------------------------------------------

install_credentials() {
  local generated
  if [ -n "$CONNECTION_KEY_FILE" ]; then
    install_file "$CONNECTION_KEY_FILE" "$KEY_FILE" 0600 "$SERVICE_USER" "$GROUP"
  elif [ -f "$KEY_FILE" ]; then
    chmod 0600 "$KEY_FILE"
    chown "$SERVICE_USER:$GROUP" "$KEY_FILE"
    say "the connection key exists, and the installer keeps it"
    LAST_CHANGED=0
  else
    generated=$WORK_DIR/connection-key
    (umask 077 && head -c 32 /dev/urandom | base64 -w0 | tr '+/' '-_' | tr -d '=' >"$generated")
    install_file "$generated" "$KEY_FILE" 0600 "$SERVICE_USER" "$GROUP"
    say "made a new connection key"
  fi
  if [ "$LAST_CHANGED" -eq 1 ]; then RESTART_NEEDED=1; fi
  if [ -n "$ENV_FILE" ]; then
    install_file "$ENV_FILE" "$ENV_TARGET" 0600 root root
    if [ "$LAST_CHANGED" -eq 1 ]; then RESTART_NEEDED=1; fi
  fi
  install_token_hash
}

# The web unit loads the token hash as a credential, so the file must exist. When you give no hash,
# the installer keeps an empty file, which web reads as no hash.
install_token_hash() {
  local empty
  if [ -n "$TOKEN_HASH_FILE" ]; then
    install_file "$TOKEN_HASH_FILE" "$TOKEN_FILE" 0600 "$SERVICE_USER" "$GROUP"
  elif [ -f "$TOKEN_FILE" ]; then
    chmod 0600 "$TOKEN_FILE"
    chown "$SERVICE_USER:$GROUP" "$TOKEN_FILE"
    LAST_CHANGED=0
  else
    empty=$WORK_DIR/token-hash
    : >"$empty"
    install_file "$empty" "$TOKEN_FILE" 0600 "$SERVICE_USER" "$GROUP"
  fi
  if [ "$LAST_CHANGED" -eq 1 ]; then RESTART_NEEDED=1; fi
  if [ ! -s "$TOKEN_FILE" ] &&
    ! { [ -f "$CONFIG_FILE" ] && grep -Eq '^[[:space:]]*token_hash(_file)?[[:space:]]*=' "$CONFIG_FILE"; } &&
    ! { [ -f "$ENV_TARGET" ] && grep -q '^SEEINGMON_AUTH__TOKEN_HASH=' "$ENV_TARGET"; }; then
    warn "the API has no token hash, so no client can send a command. Make a token and its hash with: $BIN_DIR/seeingmon web hash-token. Then run the installer with --token-hash-file, or set token_hash in [auth] of the local configuration."
  fi
}

install_local_config() {
  local example
  if [ -n "$LOCAL_CONFIG" ]; then
    if [ -f "$CONFIG_FILE" ] && ! cmp -s -- "$LOCAL_CONFIG" "$CONFIG_FILE"; then
      install -m 0600 -- "$CONFIG_FILE" "$CONFIG_FILE.previous"
      chown "$SERVICE_USER:$GROUP" "$CONFIG_FILE.previous"
      say "kept the earlier local configuration as $CONFIG_FILE.previous"
    fi
    install_file "$LOCAL_CONFIG" "$CONFIG_FILE" 0600 "$SERVICE_USER" "$GROUP"
    if [ "$LAST_CHANGED" -eq 1 ]; then RESTART_NEEDED=1; fi
  fi
  example=$(find "$RELEASE_DIR/venv" -path '*/seeingmon/_data/config/local.example.toml' -print -quit)
  if [ -n "$example" ]; then
    install_file "$example" "$CONFIG_DIR/local.example.toml" 0644 root root
  fi
  if [ ! -f "$CONFIG_FILE" ]; then
    warn "there is no local configuration: copy $CONFIG_DIR/local.example.toml to $CONFIG_FILE, edit it, and run the installer with --local-config"
    return 0
  fi
  if ! grep -Eq '^[[:space:]]*driver[[:space:]]*=[[:space:]]*"asi"' "$CONFIG_FILE"; then
    warn "the local configuration does not select the asi camera driver, so acquire runs the simulator: set driver = \"asi\" in [services.acquire]"
  fi
  if ! grep -Eq '^[[:space:]]*station_id[[:space:]]*=' "$CONFIG_FILE"; then
    warn "the local configuration sets no station_id"
  fi
  if grep -Eq '^[[:space:]]*data_dir[[:space:]]*=' "$CONFIG_FILE" &&
    ! grep -Eq "^[[:space:]]*data_dir[[:space:]]*=[[:space:]]*\"$DATA_DIR\"" "$CONFIG_FILE"; then
    warn "data_dir in the local configuration differs from --data-dir. The units pass --data-dir to the services, and it wins."
  fi
}

# --- Step 7: system files --------------------------------------------------------------------------

install_system_files() {
  local index=0 target
  say "installing the systemd units and the system settings"
  while [ "$index" -lt "${#MANAGED[@]}" ]; do
    target=${MANAGED[$((index + 1))]}
    install_managed "${MANAGED[$index]}" "$target" "${MANAGED[$((index + 2))]}"
    if [ "$LAST_CHANGED" -eq 1 ]; then
      case $target in
        "$UNIT_DIR"/*) UNITS_CHANGED=1; RESTART_NEEDED=1 ;;
        "$UDEV_FILE") UDEV_CHANGED=1 ;;
        "$JOURNALD_FILE") JOURNALD_CHANGED=1 ;;
        "$CHRONY_FILE") CHRONY_CHANGED=1 ;;
      esac
    fi
    index=$((index + 3))
  done
  if [ "$SUPERVISOR_ACTIONS" -eq 0 ] && [ -f "$POLKIT_FILE" ]; then
    warn "$POLKIT_FILE is still installed, although you left out --supervisor-actions: remove it by hand if you do not want it"
  fi
  install_file "$SCRIPT_DIR/rollback.sh" "$BIN_DIR/rollback.sh" 0755 root root
}

# --- Step 9: switch releases -----------------------------------------------------------------------

# Point a symbolic link at a target in one step. Call it as: replace_link TARGET LINK
replace_link() {
  ln -sfn -- "$1" "$2.new"
  mv -T -- "$2.new" "$2"
}

activate_release() {
  local old=''
  local wanted=releases/$RELEASE_ID
  if [ -L "$CURRENT_LINK" ]; then
    old=$(readlink -- "$CURRENT_LINK")
  fi
  if [ "$old" = "$wanted" ]; then
    say "release $RELEASE_ID is already the current release"
    return 0
  fi
  if [ -n "$old" ]; then
    replace_link "$old" "$PREVIOUS_LINK"
  fi
  replace_link "$wanted" "$CURRENT_LINK"
  RESTART_NEEDED=1
  if [ -n "$old" ]; then
    note_change "switched $CURRENT_LINK to release $RELEASE_ID (the earlier release is $old)"
  else
    note_change "switched $CURRENT_LINK to release $RELEASE_ID"
  fi
}

# Remove the releases beyond --keep, oldest first. The current and the previous release stay.
prune_releases() {
  local kept=0 name entry
  local protected=()
  local listing
  for entry in "$CURRENT_LINK" "$PREVIOUS_LINK"; do
    if [ -L "$entry" ]; then
      protected+=("$(basename -- "$(readlink -- "$entry")")")
    fi
  done
  listing=$(find "$RELEASES_DIR" -mindepth 2 -maxdepth 2 -name .installed -printf '%T@ %h\n' | sort -rn) || true
  while read -r _ entry; do
    [ -n "$entry" ] || continue
    name=${entry##*/}
    kept=$((kept + 1))
    if [ "$kept" -le "$KEEP" ] || [[ " ${protected[*]} " == *" $name "* ]]; then
      continue
    fi
    rm -rf -- "${RELEASES_DIR:?}/$name"
    note_change "removed the old release $name"
  done <<<"$listing"
}

# --- Step 10: apply the changes --------------------------------------------------------------------

apply_system_changes() {
  local value
  if [ "$UNITS_CHANGED" -eq 1 ]; then
    systemctl daemon-reload
  fi
  if [ "$UDEV_CHANGED" -eq 1 ]; then
    udevadm control --reload-rules
    # Apply the rule to a camera that is already plugged in.
    udevadm trigger --subsystem-match=usb --attr-match=idVendor=03c3 --action=change || true
  fi
  systemd-tmpfiles --create "$TMPFILES_FILE" ||
    warn "systemd-tmpfiles could not apply $TMPFILES_FILE"
  value=$(cat /sys/module/usbcore/parameters/usbfs_memory_mb 2>/dev/null || true)
  if [ "$value" = "$USBFS_MEMORY_MB" ]; then
    say "the USB buffer size is $value MB"
  else
    warn "the USB buffer size is ${value:-unknown} MB and not $USBFS_MEMORY_MB MB. A reboot may be needed. If it is still wrong after the reboot, add usbcore.usbfs_memory_mb=$USBFS_MEMORY_MB to /boot/firmware/cmdline.txt, as the file $TMPFILES_FILE says."
  fi
  if [ "$JOURNALD_CHANGED" -eq 1 ]; then
    systemctl restart systemd-journald
  fi
  if [ "$CHRONY_CHANGED" -eq 1 ]; then
    systemctl restart chrony
  fi
}

start_services() {
  local unit
  local failed=()
  systemctl reset-failed "${SERVICES[@]}" 2>/dev/null || true
  systemctl enable seeingmon.target "${SERVICES[@]}" >/dev/null
  if [ "$RESTART_NEEDED" -eq 1 ]; then
    say "restarting the services"
    systemctl restart seeingmon.target || true
  else
    say "starting the services (nothing changed, so a running service keeps running)"
    systemctl start seeingmon.target || true
  fi
  for unit in "${SERVICES[@]}"; do
    systemctl is-active --quiet "$unit" || failed+=("$unit")
  done
  if [ "${#failed[@]}" -gt 0 ]; then
    STATUS=1
    warn "these units are not active: ${failed[*]}. Read why with: journalctl -u ${failed[0]} -n 50. To go back to the earlier release, run: $BIN_DIR/rollback.sh --prefix $PREFIX"
  fi
}

print_summary() {
  local item
  printf '\n%s\n' "Summary"
  printf '  Release          %s\n' "$RELEASE_ID"
  printf '  Prefix           %s\n' "$PREFIX"
  printf '  Service user     %s\n' "$SERVICE_USER"
  printf '  Data directory   %s\n' "$DATA_DIR"
  printf '  Config directory %s\n' "$CONFIG_DIR"
  if [ "${#CHANGES[@]}" -eq 0 ]; then
    printf '\n%s\n' "Nothing changed: the installation already matched what you asked for."
  else
    printf '\n%s\n' "What changed:"
    for item in "${CHANGES[@]}"; do printf '  - %s\n' "$item"; done
  fi
  if [ "${#WARNINGS[@]}" -gt 0 ]; then
    printf '\n%s\n' "Warnings:"
    for item in "${WARNINGS[@]}"; do printf '  - %s\n' "$item"; done
  fi
  cat <<EOF

Next steps:
  systemctl status seeingmon.target
  journalctl -u seeingmon-core -f
  chronyc tracking
  $BIN_DIR/seeingmon --help
  $BIN_DIR/rollback.sh --prefix $PREFIX    # go back to the previous release
EOF
}

check_system
ensure_user
make_directories
install_release
install_sdk
install_local_config
install_credentials
install_system_files
activate_release
prune_releases
apply_system_changes
start_services
print_summary
if [ "$STATUS" -ne 0 ]; then
  exit "$STATUS"
fi
