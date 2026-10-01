"""The facts about the machine and the software that a report records.

A report says what ran it, so that you can compare two reports honestly: the architecture, the
operating-system family, the Python and library versions, the processor model, and the commit of
the code. It records no host name, no user name, no address, no serial number, and no path. The
reader of `/proc/cpuinfo` picks the lines that it needs by name, so the `Serial` line of a
Raspberry Pi never enters a report.
"""

from __future__ import annotations

import os
import platform
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import seeingmon

# Distributions whose versions matter for a measurement.
PACKAGES = (
    "seeingmon",
    "numpy",
    "scipy",
    "pydantic",
    "pyerfa",
    "astropy",
    "sep",
    "fastapi",
    "uvicorn",
    "pillow",
)

# The core of each ARM implementer part, for the processors that a Raspberry Pi uses.
_ARM_PARTS = {
    "0xd03": "Cortex-A53",
    "0xd08": "Cortex-A72",
    "0xd0b": "Cortex-A76",
}
_GIT_TIMEOUT_S = 10.0


@dataclass(frozen=True, slots=True)
class Environment:
    """What ran the harness. Every field is a fact about the machine or the software."""

    machine: str  # `x86-64` or `arm64`
    os_family: str  # `Windows`, `Linux`, or `Darwin`
    python: str
    python_implementation: str
    cpu_count: int | None
    cpu_model: str | None = None
    cpu_governor: str | None = None
    sqlite: str = ""
    packages: dict[str, str] = field(default_factory=dict)
    git_commit: str | None = None
    git_dirty: bool | None = None
    throttled: str | None = None  # `vcgencmd get_throttled` on a Raspberry Pi, a hex mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine": self.machine,
            "os_family": self.os_family,
            "python": self.python,
            "python_implementation": self.python_implementation,
            "cpu_count": self.cpu_count,
            "cpu_model": self.cpu_model,
            "cpu_governor": self.cpu_governor,
            "sqlite": self.sqlite,
            "packages": dict(self.packages),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "throttled": self.throttled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Environment:
        """Read the environment from `to_dict`. Raises `ValueError` when a key is missing."""
        try:
            packages = {str(name): str(version) for name, version in dict(data["packages"]).items()}
            count = data["cpu_count"]
            return cls(
                machine=str(data["machine"]),
                os_family=str(data["os_family"]),
                python=str(data["python"]),
                python_implementation=str(data["python_implementation"]),
                cpu_count=None if count is None else int(count),
                cpu_model=_optional_str(data.get("cpu_model")),
                cpu_governor=_optional_str(data.get("cpu_governor")),
                sqlite=str(data.get("sqlite", "")),
                packages=packages,
                git_commit=_optional_str(data.get("git_commit")),
                git_dirty=None if data.get("git_dirty") is None else bool(data["git_dirty"]),
                throttled=_optional_str(data.get("throttled")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"the environment is incomplete or malformed ({error})") from None

    @property
    def is_arm64(self) -> bool:
        return self.machine == "arm64"


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def normalize_machine(raw: str) -> str:
    """`x86-64` or `arm64` for the usual spellings of `platform.machine()`, else the lowercase
    spelling."""
    name = raw.strip().lower()
    if name in ("x86_64", "amd64", "x64"):
        return "x86-64"
    if name in ("arm64", "aarch64"):
        return "arm64"
    return name or "unknown"


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def parse_cpuinfo(text: str) -> str | None:
    """A processor name from the text of `/proc/cpuinfo`, without any serial number.

    On x86-64, the name is the first `model name` line. On ARM, it is the board `Model` line
    with the core name from `CPU part`, or the `Hardware` line when there is no model.
    """
    wanted: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key in ("model name", "model", "hardware", "cpu part") and key not in wanted:
            wanted[key] = value.strip()
    if "model name" in wanted:
        return wanted["model name"]
    board = wanted.get("model") or wanted.get("hardware")
    core = _ARM_PARTS.get(wanted.get("cpu part", "").lower(), wanted.get("cpu part"))
    if board and core:
        return f"{board} ({core})"
    return board or core


if sys.platform == "linux":

    def _cpu_model() -> str | None:
        text = _read_text("/proc/cpuinfo")
        return None if text is None else parse_cpuinfo(text)

elif sys.platform == "win32":

    def _cpu_model() -> str | None:
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",  # pragma: allowlist secret
            ) as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        except OSError:
            return platform.processor() or None
        return " ".join(str(name).split()) or None

else:

    def _cpu_model() -> str | None:
        return platform.processor() or None


def _cpu_governor() -> str | None:
    text = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return None if text is None else text.strip() or None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _source_root() -> Path | None:
    """The root of the git checkout that holds this package, or `None` for an installed copy."""
    root = Path(seeingmon.__file__).resolve().parents[2]
    return root if (root / ".git").exists() else None


def _git(root: Path, *arguments: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        completed = subprocess.run(
            [git, "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def git_facts() -> tuple[str | None, bool | None]:
    """The commit hash of the checkout and whether it has uncommitted changes.

    Returns `(None, None)` when the package is not in a git checkout or git is not available.
    """
    root = _source_root()
    if root is None:
        return None, None
    commit = (_git(root, "rev-parse", "HEAD") or "").strip()
    if not commit:
        return None, None
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return commit, None if status is None else bool(status.strip())


def pi_throttled() -> str | None:
    """The `vcgencmd get_throttled` mask of a Raspberry Pi, or `None` where the tool is absent.

    A mask other than `0x0` means that the board lost power or heat headroom since it booted,
    so a run on it measured a slower processor than its clock says.
    """
    tool = shutil.which("vcgencmd")
    if tool is None:
        return None
    try:
        completed = subprocess.run(
            [tool, "get_throttled"], capture_output=True, text=True, timeout=5.0, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    _, _, value = completed.stdout.strip().partition("=")
    return value.strip() or None


def collect_environment() -> Environment:
    """Collect the facts about this machine and software. It never raises."""
    commit, dirty = git_facts()
    return Environment(
        machine=normalize_machine(platform.machine()),
        os_family=platform.system() or "unknown",
        python=platform.python_version(),
        python_implementation=platform.python_implementation(),
        cpu_count=os.cpu_count(),
        cpu_model=_cpu_model(),
        cpu_governor=_cpu_governor(),
        sqlite=sqlite3.sqlite_version,
        packages=_package_versions(),
        git_commit=commit,
        git_dirty=dirty,
        throttled=pi_throttled(),
    )
