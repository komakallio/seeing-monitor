#!/usr/bin/env bash
# Build a release on the development machine, copy it to a Raspberry Pi, and install it there.
# Run it from any directory. It needs uv (through build.sh), and ssh and tar on this machine.
#
# The script builds the wheel and the hashed requirements, copies them with the install scripts
# and your local files to a private directory on the Pi, runs install.sh there, and removes the
# directory afterwards. Nothing about the Pi is stored in the repository: you pass every value.
set -euo pipefail

PROGRAM=${0##*/}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<EOF
Usage: $PROGRAM --host HOST --user USER --prefix DIR --service-user NAME --data-dir DIR \\
          --config-dir DIR --time-source NAME [options]

Required parameters:
  --host HOST          the Raspberry Pi: a host name or address that ssh can reach.
  --user USER          the account that you log in with. The installer needs root, so the script
                       runs it through sudo unless USER is root.
  --prefix DIR         the installation prefix on the Pi. The releases live under it.
  --service-user NAME  the account that runs the services. The installer creates it.
  --data-dir DIR       the data directory on the Pi. Use a folder on its own partition.
  --config-dir DIR     the configuration directory on the Pi. It holds your local configuration
                       and the connection key.
  --time-source NAME   a time source for chrony. Repeat the option for more sources.

Optional parameters:
  --no-time-config     leave chrony alone. Use it instead of --time-source.
  --local-config FILE  your local configuration, made from config/local.example.toml.
  --sdk-archive FILE   the vendor SDK archive. The installer installs it privately.
  --sdk-sha256 HEX     the checksum that the archive must have. Required with --sdk-archive.
  --connection-key-file FILE
                       a file with the key that the three services share. Without it, the
                       installer makes a key on the Pi at the first install.
  --token-hash-file FILE
                       a file with the hash of the API token, which the command web hash-token
                       makes. Without it, web has no token hash.
  --env-file FILE      a file of NAME=value lines that the services read (sink tokens, for example).
  --python PATH        the Python interpreter that uv uses to build the release.
  --dist-dir DIR       use the wheel and requirements.txt in DIR, and build nothing.
  --port N             the ssh port of the Pi.
  --identity FILE      the ssh key file.
  --ssh-option NAME=VALUE
                       pass -o NAME=VALUE to ssh. Repeat it for more options.
  --installer-arg ARG  pass ARG to install.sh. Repeat it for more arguments. For example,
                       --installer-arg --dry-run shows the plan of the installer on the Pi.
  --keep-stage         leave the directory on the Pi, so that you can look at it.
  --dry-run            print every command, and run none of them.
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

# Check that an option has an argument, which can start with dashes. Call it as: need_argument "$@"
need_argument() {
  [ "$#" -ge 2 ] || usage_error "the option $1 needs a value"
}

# Check that a required parameter has a value. Call it as: require VARIABLE --option
require() {
  [ -n "${!1}" ] || usage_error "missing required parameter: $2"
}

# Check an absolute path that has no spaces or shell characters. Call it as: check_path --option "$VALUE"
check_path() {
  [[ $2 =~ ^/[A-Za-z0-9._+/-]*$ ]] || usage_error "$1 needs an absolute path of plain characters: $2"
  [[ $2 != *..* ]] || usage_error "$1 must not contain '..': $2"
  [[ $2 == / || $2 != */ ]] || usage_error "$1 must not end with a slash: $2"
}

# Check an account name. Call it as: check_name --option "$VALUE"
check_name() {
  [[ $2 =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] || usage_error "$1 needs a plain account name: $2"
}

# Check a host name or address. It must not start with a dash, so ssh never reads it as an option.
check_host() {
  [[ $2 =~ ^[A-Za-z0-9][A-Za-z0-9._:-]*$ ]] || usage_error "$1 needs a host name or address: $2"
}

check_file() {
  [ -f "$2" ] || usage_error "$1 names a file that does not exist: $2"
}

# Print an ssh command for a dry run: pass the words before the remote command, then the remote
# command as one argument. The remote command goes in single quotes, so that you can read it.
print_ssh() {
  local remote=${!#}
  set -- "${@:1:$#-1}"
  printf '+'
  printf ' %q' "${SSH[@]}" "$@"
  printf " '%s'\n" "$remote"
}

HOST=''
SSH_USER=''
PREFIX=''
SERVICE_USER=''
DATA_DIR=''
CONFIG_DIR=''
NO_TIME_CONFIG=0
TIME_SOURCES=()
LOCAL_CONFIG=''
SDK_ARCHIVE=''
SDK_SHA256=''
CONNECTION_KEY_FILE=''
TOKEN_HASH_FILE=''
ENV_FILE=''
PYTHON=''
DIST_DIR=''
PORT=''
IDENTITY=''
SSH_OPTIONS=()
INSTALLER_ARGS=()
KEEP_STAGE=0
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case $1 in
    --host) need_value "$@"; HOST=$2; shift 2 ;;
    --user) need_value "$@"; SSH_USER=$2; shift 2 ;;
    --prefix) need_value "$@"; PREFIX=$2; shift 2 ;;
    --service-user) need_value "$@"; SERVICE_USER=$2; shift 2 ;;
    --data-dir) need_value "$@"; DATA_DIR=$2; shift 2 ;;
    --config-dir) need_value "$@"; CONFIG_DIR=$2; shift 2 ;;
    --time-source) need_value "$@"; TIME_SOURCES+=("$2"); shift 2 ;;
    --no-time-config) NO_TIME_CONFIG=1; shift ;;
    --local-config) need_value "$@"; LOCAL_CONFIG=$2; shift 2 ;;
    --sdk-archive) need_value "$@"; SDK_ARCHIVE=$2; shift 2 ;;
    --sdk-sha256) need_value "$@"; SDK_SHA256=$2; shift 2 ;;
    --connection-key-file) need_value "$@"; CONNECTION_KEY_FILE=$2; shift 2 ;;
    --token-hash-file) need_value "$@"; TOKEN_HASH_FILE=$2; shift 2 ;;
    --env-file) need_value "$@"; ENV_FILE=$2; shift 2 ;;
    --python) need_value "$@"; PYTHON=$2; shift 2 ;;
    --dist-dir) need_value "$@"; DIST_DIR=$2; shift 2 ;;
    --port) need_value "$@"; PORT=$2; shift 2 ;;
    --identity) need_value "$@"; IDENTITY=$2; shift 2 ;;
    --ssh-option) need_value "$@"; SSH_OPTIONS+=("$2"); shift 2 ;;
    --installer-arg) need_argument "$@"; INSTALLER_ARGS+=("$2"); shift 2 ;;
    --keep-stage) KEEP_STAGE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage_error "unknown option: $1" ;;
  esac
done

require HOST --host
require SSH_USER --user
require PREFIX --prefix
require SERVICE_USER --service-user
require DATA_DIR --data-dir
require CONFIG_DIR --config-dir

check_host --host "$HOST"
check_name --user "$SSH_USER"
check_path --prefix "$PREFIX"
check_name --service-user "$SERVICE_USER"
check_path --data-dir "$DATA_DIR"
check_path --config-dir "$CONFIG_DIR"
if [ "$NO_TIME_CONFIG" -eq 0 ]; then
  [ "${#TIME_SOURCES[@]}" -gt 0 ] || usage_error "missing required parameter: --time-source (or --no-time-config)"
elif [ "${#TIME_SOURCES[@]}" -gt 0 ]; then
  usage_error "--time-source and --no-time-config exclude each other"
fi
for time_source in "${TIME_SOURCES[@]}"; do
  check_host --time-source "$time_source"
done
if [ -n "$LOCAL_CONFIG" ]; then check_file --local-config "$LOCAL_CONFIG"; fi
if [ -n "$CONNECTION_KEY_FILE" ]; then check_file --connection-key-file "$CONNECTION_KEY_FILE"; fi
if [ -n "$TOKEN_HASH_FILE" ]; then check_file --token-hash-file "$TOKEN_HASH_FILE"; fi
if [ -n "$ENV_FILE" ]; then check_file --env-file "$ENV_FILE"; fi
if [ -n "$SDK_ARCHIVE" ] || [ -n "$SDK_SHA256" ]; then
  [ -n "$SDK_ARCHIVE" ] || usage_error "--sdk-sha256 needs --sdk-archive"
  [ -n "$SDK_SHA256" ] || usage_error "--sdk-archive needs --sdk-sha256"
  check_file --sdk-archive "$SDK_ARCHIVE"
  [[ $SDK_SHA256 =~ ^[0-9a-f]{64}$ ]] || usage_error "--sdk-sha256 needs 64 lowercase hex digits"
fi
if [ -n "$PORT" ]; then
  [[ $PORT =~ ^[0-9]{1,5}$ ]] || usage_error "--port needs a number: $PORT"
fi
if [ -n "$IDENTITY" ]; then check_file --identity "$IDENTITY"; fi
for option in "${SSH_OPTIONS[@]}"; do
  [[ $option =~ ^[A-Za-z]+=[A-Za-z0-9._:/@+-]+$ ]] || usage_error "--ssh-option needs NAME=VALUE: $option"
done

# Pieces of the ssh command that the steps share.
TARGET="$SSH_USER@$HOST"
SSH=(ssh -o ConnectTimeout=15)
if [ -n "$PORT" ]; then SSH+=(-p "$PORT"); fi
if [ -n "$IDENTITY" ]; then SSH+=(-i "$IDENTITY"); fi
for option in "${SSH_OPTIONS[@]}"; do
  SSH+=(-o "$option")
done
SUDO=(sudo)
if [ "$SSH_USER" = root ]; then SUDO=(); fi

LOCAL_TMP=''
STAGE=''

cleanup() {
  local status=$?
  trap - EXIT
  if [ -n "$LOCAL_TMP" ]; then
    rm -rf -- "$LOCAL_TMP"
  fi
  if [ -n "$STAGE" ] && [ "$DRY_RUN" -eq 0 ]; then
    if [ "$KEEP_STAGE" -eq 1 ]; then
      printf '%s: left the staging directory on the Pi: %s\n' "$PROGRAM" "$STAGE" >&2
    elif ! "${SSH[@]}" -- "$TARGET" rm -rf -- "$STAGE"; then
      printf '%s: could not remove the staging directory on the Pi: %s\n' "$PROGRAM" "$STAGE" >&2
    fi
  fi
  exit "$status"
}
trap cleanup EXIT

# --- Step 1: build -----------------------------------------------------------------------------

if [ -n "$DIST_DIR" ]; then
  BUILD_DIR=$DIST_DIR
  say "using the build in $BUILD_DIR"
else
  if [ "$DRY_RUN" -eq 1 ]; then
    BUILD_DIR=/tmp/seeingmon-build.XXXXXX
  else
    LOCAL_TMP=$(mktemp -d)
    BUILD_DIR=$LOCAL_TMP/build
  fi
  BUILD_ARGS=(--out-dir "$BUILD_DIR")
  if [ -n "$PYTHON" ]; then BUILD_ARGS+=(--python "$PYTHON"); fi
  if [ "$DRY_RUN" -eq 1 ]; then BUILD_ARGS+=(--dry-run); fi
  say "building the release"
  bash "$SCRIPT_DIR/build.sh" "${BUILD_ARGS[@]}"
fi

if [ "$DRY_RUN" -eq 1 ] && [ -z "$DIST_DIR" ]; then
  WHEEL=$BUILD_DIR/seeingmon-VERSION-py3-none-any.whl
else
  wheels=("$BUILD_DIR"/seeingmon-*.whl)
  if [ "${#wheels[@]}" -ne 1 ] || [ ! -f "${wheels[0]}" ]; then
    die "expected one seeingmon wheel in $BUILD_DIR"
  fi
  [ -f "$BUILD_DIR/requirements.txt" ] || die "there is no requirements.txt in $BUILD_DIR"
  WHEEL=${wheels[0]}
fi
WHEEL_NAME=${WHEEL##*/}

# --- Step 2: stage the files locally -----------------------------------------------------------

# The installer reads these names in its staging directory on the Pi.
INSTALL_ARGS=(
  --prefix "$PREFIX" --user "$SERVICE_USER" --data-dir "$DATA_DIR" --config-dir "$CONFIG_DIR"
)
STAGE_FILES=()  # pairs of: local file, name in the staging directory

add_file() {
  STAGE_FILES+=("$1" "$2")
}

add_file "$WHEEL" "$WHEEL_NAME"
add_file "$BUILD_DIR/requirements.txt" requirements.txt
if [ -n "$LOCAL_CONFIG" ]; then add_file "$LOCAL_CONFIG" local-config.toml; fi
if [ -n "$SDK_ARCHIVE" ]; then add_file "$SDK_ARCHIVE" "sdk-${SDK_ARCHIVE##*/}"; fi
if [ -n "$CONNECTION_KEY_FILE" ]; then add_file "$CONNECTION_KEY_FILE" connection-key; fi
if [ -n "$TOKEN_HASH_FILE" ]; then add_file "$TOKEN_HASH_FILE" token-hash; fi
if [ -n "$ENV_FILE" ]; then add_file "$ENV_FILE" seeingmon.env; fi

if [ "$DRY_RUN" -eq 1 ]; then
  LOCAL_STAGE=/tmp/seeingmon-stage.XXXXXX
  STAGE=/tmp/seeingmon-install.XXXXXX
  say "dry run: the staging directory on the Pi comes from mktemp -d, and the name below is an example"
else
  LOCAL_TMP=${LOCAL_TMP:-$(mktemp -d)}
  LOCAL_STAGE=$LOCAL_TMP/stage
  mkdir -p -- "$LOCAL_STAGE/deploy"
  cp -R -- "$SCRIPT_DIR/." "$LOCAL_STAGE/deploy/"
  index=0
  while [ "$index" -lt "${#STAGE_FILES[@]}" ]; do
    cp -- "${STAGE_FILES[$index]}" "$LOCAL_STAGE/${STAGE_FILES[$((index + 1))]}"
    index=$((index + 2))
  done
fi

# --- Step 3: copy to the Pi --------------------------------------------------------------------

say "creating a private directory on $HOST"
if [ "$DRY_RUN" -eq 1 ]; then
  print_ssh -- "$TARGET" 'mktemp -d'
else
  STAGE=$("${SSH[@]}" -- "$TARGET" mktemp -d)
  [[ $STAGE =~ ^/[A-Za-z0-9._/-]+$ ]] || die "the Pi returned an unexpected directory name: $STAGE"
fi

say "copying the release and your files"
UNPACK=$(printf 'tar -C %q -xf -' "$STAGE")
if [ "$DRY_RUN" -eq 1 ]; then
  printf '+ tar -C %s -cf - . | ' "$LOCAL_STAGE"
  print_ssh -- "$TARGET" "$UNPACK" | cut -c3-
else
  tar -C "$LOCAL_STAGE" -cf - . | "${SSH[@]}" -- "$TARGET" "$UNPACK"
fi

# --- Step 4: install ---------------------------------------------------------------------------

INSTALL_ARGS+=(--wheel "$STAGE/$WHEEL_NAME" --requirements "$STAGE/requirements.txt")
if [ -n "$LOCAL_CONFIG" ]; then INSTALL_ARGS+=(--local-config "$STAGE/local-config.toml"); fi
if [ -n "$SDK_ARCHIVE" ]; then
  INSTALL_ARGS+=(--sdk-archive "$STAGE/sdk-${SDK_ARCHIVE##*/}" --sdk-sha256 "$SDK_SHA256")
fi
if [ -n "$CONNECTION_KEY_FILE" ]; then
  INSTALL_ARGS+=(--connection-key-file "$STAGE/connection-key")
fi
if [ -n "$TOKEN_HASH_FILE" ]; then INSTALL_ARGS+=(--token-hash-file "$STAGE/token-hash"); fi
if [ -n "$ENV_FILE" ]; then INSTALL_ARGS+=(--env-file "$STAGE/seeingmon.env"); fi
if [ "$NO_TIME_CONFIG" -eq 1 ]; then INSTALL_ARGS+=(--no-time-config); fi
for time_source in "${TIME_SOURCES[@]}"; do
  INSTALL_ARGS+=(--time-source "$time_source")
done
INSTALL_ARGS+=("${INSTALLER_ARGS[@]}")

REMOTE=$(printf '%q ' "${SUDO[@]}" bash "$STAGE/deploy/install.sh" "${INSTALL_ARGS[@]}")
REMOTE=${REMOTE% }
TTY=()
if [ -t 0 ]; then TTY=(-t); fi  # sudo can ask for a password only when it has a terminal

say "running the installer on $HOST"
status=0
if [ "$DRY_RUN" -eq 1 ]; then
  print_ssh "${TTY[@]}" -- "$TARGET" "$REMOTE"
  say "dry run: then the script removes the staging directory on the Pi"
else
  "${SSH[@]}" "${TTY[@]}" -- "$TARGET" "$REMOTE" || status=$?
fi
if [ "$status" -ne 0 ]; then
  exit "$status"
fi
