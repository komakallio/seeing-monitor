#!/usr/bin/env bash
# Managed by the seeingmon installer. Changes are overwritten.
#
# Run seeingmon as the service user, in the configuration directory, with the connection key. Use
# it for the commissioning commands, so that they read the same configuration, key, and data
# directory as the services:
#
#     sudo @PREFIX@/bin/seeingmon burst --help
set -euo pipefail

if [ "$(id -un)" != "@USER@" ]; then
  exec sudo -u "@USER@" -- "$0" "$@"
fi

cd -- "@CONFIG_DIR@"
CREDENTIALS_DIRECTORY="@CONFIG_DIR@/credentials"
export CREDENTIALS_DIRECTORY
SEEINGMON_PATHS__DATA_DIR="@DATA_DIR@"
export SEEINGMON_PATHS__DATA_DIR
exec "@PREFIX@/current/venv/bin/seeingmon" "$@"
