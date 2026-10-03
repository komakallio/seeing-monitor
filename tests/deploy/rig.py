"""A rig for the installer tests: stub system programs, a fake system root, and a fake interpreter.

The installer is a root program that creates users and talks to systemd. The rig runs it as the
test user. The `id` stub says that the user is root, the stubs for the system programs only log
their arguments, `--system-root` moves the system files into a temporary folder, and a fake
interpreter makes fake virtual environments. Everything else (the file operations, the symbolic
links, the rendering, the modes, the checksums) runs for real.
"""

from __future__ import annotations

import hashlib
import io
import platform
import stat
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tests.deploy.scripts import DEPLOY, Result, Stubs, run_script

INSTALL = DEPLOY / "install.sh"
ROLLBACK = DEPLOY / "rollback.sh"
VERSION = "0.1.0.dev0"
WHEEL_NAME = f"seeingmon-{VERSION}-py3-none-any.whl"

# The interpreter that the installer gets with --python. It logs its calls, and it makes a virtual
# environment whose `python` is a copy of itself. An environment variable makes one step fail.
FAKE_PYTHON = r"""
case "$1" in
  -c)
    case "$2" in
      *version_info*) exit "${FAKE_VERSION_STATUS:-0}" ;;
      *ensurepip*) exit "${FAKE_VENV_STATUS:-0}" ;;
    esac
    exit 0
    ;;
  -m)
    case "$2" in
      venv)
        mkdir -p "$3/bin" "$3/lib/python3.13/site-packages/seeingmon/_data/config"
        cp "$0" "$3/bin/python"
        cat > "$3/bin/seeingmon" <<'INNER'
#!/bin/sh
echo "seeingmon 0.1.0.dev0"
echo "cwd=$PWD"
echo "credentials=$CREDENTIALS_DIRECTORY"
echo "data=$SEEINGMON_PATHS__DATA_DIR"
echo "arguments=$*"
INNER
        chmod +x "$3/bin/seeingmon"
        data="$3/lib/python3.13/site-packages/seeingmon/_data/config"
        printf '# the local template\n' > "$data/local.example.toml"
        ;;
      pip)
        case "$3" in
          check) exit "${FAKE_PIP_CHECK_STATUS:-0}" ;;
          install)
            for argument in "$@"; do
              case "$argument" in
                *.whl) exit "${FAKE_PIP_WHEEL_STATUS:-0}" ;;
              esac
            done
            exit "${FAKE_PIP_REQUIREMENTS_STATUS:-0}"
            ;;
        esac
        ;;
    esac
    ;;
esac
"""

STUBS_FOR_THE_SYSTEM = ("chown", "udevadm", "systemd-tmpfiles", "usermod")


@dataclass
class Rig:
    """Paths and helpers for one installer run."""

    root: Path  # the folder of the test
    stubs: Stubs
    system_root: Path
    prefix: Path
    data_dir: Path
    config_dir: Path
    stage: Path
    wheel: Path
    requirements: Path
    state: Path
    python: Path

    # --- Running -------------------------------------------------------------------------------

    def arguments(self, **changes: str | list[str] | bool | None) -> list[str]:
        """The arguments of a normal run, with some changed.

        A key is an option name with underscores. A value is a string, a list for a repeated
        option, or True for a flag. None drops the option.
        """
        values: dict[str, str | list[str] | bool | None] = {
            "prefix": str(self.prefix),
            "user": "seeingmon",
            "data_dir": str(self.data_dir),
            "config_dir": str(self.config_dir),
            "wheel": str(self.wheel),
            "requirements": str(self.requirements),
            "time_source": ["time.example.org", "192.0.2.10"],
            "python": str(self.python),
            "system_root": str(self.system_root),
        }
        values.update(changes)
        words: list[str] = []
        for key, value in values.items():
            option = "--" + key.replace("_", "-")
            if value is None or value is False:
                continue
            if value is True:
                words.append(option)
                continue
            for item in [value] if isinstance(value, str) else value:
                words += [option, item]
        return words

    def env(self, **extra: str) -> dict[str, str]:
        return self.stubs.env(**extra)

    def install(self, *extra: str, **changes: str | list[str] | bool | None) -> Result:
        """Run the installer with the normal arguments, the extra arguments, and some changes."""
        return self.install_env({}, *extra, **changes)

    def install_env(
        self,
        variables: Mapping[str, str],
        /,
        *extra: str,
        **changes: str | list[str] | bool | None,
    ) -> Result:
        """Like `install`, with more environment variables (they win over the stubs' PATH)."""
        arguments = [*self.arguments(**changes), *extra]
        return run_script(INSTALL, arguments, env=self.env(**variables))

    def rollback(self, *extra: str, prefix: str | None = None) -> Result:
        return self.rollback_env({}, *extra, prefix=prefix)

    def rollback_env(
        self, variables: Mapping[str, str], /, *extra: str, prefix: str | None = None
    ) -> Result:
        arguments = ["--prefix", prefix if prefix is not None else str(self.prefix), *extra]
        return run_script(ROLLBACK, arguments, env=self.env(**variables))

    # --- Inputs --------------------------------------------------------------------------------

    def new_release(self, label: str) -> None:
        """Change the wheel, so that the next run installs another release."""
        self.wheel.write_text(f"fake wheel {label}\n", encoding="utf-8")

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    def make_sdk(self, members: Mapping[str, bytes], name: str = "sdk.tar.bz2") -> tuple[Path, str]:
        """Write an SDK archive. Returns its path and its SHA-256."""
        path = self.root / name
        with tarfile.open(path, "w:bz2") as archive:
            for member, data in members.items():
                info = tarfile.TarInfo(member)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    # --- What the run left behind --------------------------------------------------------------

    @property
    def releases(self) -> list[str]:
        folder = self.prefix / "releases"
        return sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []

    def link(self, name: str) -> str | None:
        path = self.prefix / name
        return str(path.readlink()) if path.is_symlink() else None

    def mode(self, path: Path) -> int:
        return stat.S_IMODE(path.stat().st_mode)

    def etc(self, relative: str) -> Path:
        return self.system_root / "etc" / relative

    def write_controllers(self, text: str) -> Path:
        """Write the list of cgroup controllers that the installer reads from the system root.

        A new rig has no such file, as on a system without cgroup v2, so the installer skips
        its check for the memory cgroup.
        """
        path = self.system_root / "sys" / "fs" / "cgroup" / "cgroup.controllers"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    def values(self) -> dict[str, str]:
        """The values that the installer fills into the templates, for the parameters above."""
        return {
            "PREFIX": str(self.prefix),
            "USER": "seeingmon",
            "GROUP": "seeingmon",
            "DATA_DIR": str(self.data_dir),
            "CONFIG_DIR": str(self.config_dir),
            "USBFS_MEMORY_MB": "1000",
            "TIME_SOURCES": "server time.example.org iburst\nserver 192.0.2.10 iburst",
        }

    def calls(self, name: str) -> list[str]:
        return self.stubs.called(name)

    def clear_calls(self) -> None:
        self.stubs.clear()


def make_rig(tmp_path: Path) -> Rig:
    """Build the rig in `tmp_path`."""
    stubs = Stubs(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "group-gpio").write_text("", encoding="utf-8")

    stubs.add(
        "id",
        f"""
case "$*" in
  "-u") echo "${{FAKE_UID:-0}}" ;;
  "-un") echo "${{FAKE_USER_NAME:-root}}" ;;
  "-gn "*) echo "$2" ;;
  "-nG "*) echo "$2 $(cat "{state}/groups-of-$2" 2>/dev/null)" ;;
  *) exec /usr/bin/id "$@" ;;
esac
""",
    )
    stubs.add(
        "getent",
        f"""
case "$1 $2" in
  "passwd "*)
    [ -f "{state}/user-$2" ] || exit 2
    shell=$(cat "{state}/shell-$2" 2>/dev/null || echo /usr/sbin/nologin)
    uid=$(cat "{state}/uid-$2" 2>/dev/null || echo 998)
    echo "$2:x:$uid:$uid::/var/lib/seeingmon:$shell"
    ;;
  "group gpio") [ -f "{state}/group-gpio" ] ;;
  *) exit 2 ;;
esac
""",
    )
    stubs.add("useradd", f'for last; do :; done\n: > "{state}/user-$last"\n')
    stubs.add("usermod", f'for last; do :; done\necho gpio > "{state}/groups-of-$last"\n')
    stubs.add(
        "ldd", 'if [ -n "$FAKE_LDD_MISSING" ]; then echo "libusb-1.0.so.0 => not found"; fi\n'
    )
    for name in ("chown", "udevadm", "systemd-tmpfiles"):
        stubs.add(name)
    stubs.add(
        "systemctl",
        """
if [ "$1" = is-active ]; then
  for unit in $FAIL_UNITS; do
    if [ "$unit" = "$3" ]; then exit 3; fi
  done
fi
""",
    )
    stubs.add(
        "findmnt",
        """
case "$*" in
  *TARGET*) echo "${FAKE_DATA_MOUNT:-/}" ;;
  *FSTYPE*) echo "${FAKE_TMP_FSTYPE:-tmpfs}" ;;
esac
""",
    )
    python = stubs.add("fakepython", FAKE_PYTHON)

    system_root = tmp_path / "system"
    (system_root / "etc" / "chrony" / "conf.d").mkdir(parents=True)
    (system_root / "etc" / "chrony" / "chrony.conf").write_text(
        "confdir /etc/chrony/conf.d\npool pool.example.org iburst\n", encoding="utf-8"
    )
    (system_root / "etc" / "polkit-1" / "rules.d").mkdir(parents=True)

    stage = tmp_path / "stage"
    stage.mkdir()
    wheel = stage / WHEEL_NAME
    wheel.write_text("fake wheel\n", encoding="utf-8")
    requirements = stage / "requirements.txt"
    requirements.write_text("numpy==2.0 \\\n    --hash=sha256:00\n", encoding="utf-8")

    return Rig(
        root=tmp_path,
        stubs=stubs,
        system_root=system_root,
        prefix=tmp_path / "opt" / "seeingmon",
        data_dir=tmp_path / "data",
        config_dir=tmp_path / "etc-seeingmon",
        stage=stage,
        wheel=wheel,
        requirements=requirements,
        state=state,
        python=python,
    )


def arch_directory() -> str | None:
    """The folder of the SDK that matches this machine, or `None` for a machine that it skips."""
    return {"x86_64": "x64", "amd64": "x64", "aarch64": "armv8", "arm64": "armv8"}.get(
        platform.machine().lower()
    )
