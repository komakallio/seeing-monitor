"""Locate the data directories that ship with the package.

The hardware profiles (`profiles/`) and the default configuration (`config/`) live at the root
of the repository. A wheel carries them as package data in `seeingmon/_data/`. `data_dir`
finds whichever layout is present, in this order:

1. An installed wheel: `<package>/_data/<name>`.
2. A source checkout, including an editable install: `<root>/<name>`, where `<root>` holds
   `pyproject.toml` and the `src/` folder that contains the package.

Each function takes an optional `package` argument, the directory of the `seeingmon` package,
so that tests can build either layout in a temporary folder.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DATA_DIR = "_data"
PROFILES_DIRNAME = "profiles"
CONFIG_DIRNAME = "config"
LOCAL_CONFIG_PARTS = ("local", "config.toml")


class DataDirectoryError(FileNotFoundError):
    """A data directory exists in neither the wheel layout nor the checkout layout."""


def package_dir() -> Path:
    """The directory of the `seeingmon` package that is running."""
    return Path(__file__).resolve().parent


def source_root(package: Path | None = None) -> Path | None:
    """The root of the source checkout that holds the package, or `None` for a wheel install."""
    package = package_dir() if package is None else package
    src = package.parent
    root = src.parent
    if src.name == "src" and (root / "pyproject.toml").is_file():
        return root
    return None


def data_dir(name: str, *, package: Path | None = None) -> Path:
    """Find the data directory `name` for an installed wheel or a source checkout.

    Raises `DataDirectoryError` and names every place it looked when neither layout has it.
    """
    package = package_dir() if package is None else package
    candidates = [package / PACKAGE_DATA_DIR / name]
    root = source_root(package)
    if root is not None:
        candidates.append(root / name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise DataDirectoryError(f"cannot find the {name!r} directory; looked in {searched}")


def profiles_dir(*, package: Path | None = None) -> Path:
    """The directory with the `<id>.toml` profile files."""
    return data_dir(PROFILES_DIRNAME, package=package)


def config_dir(*, package: Path | None = None) -> Path:
    """The directory with `default.toml`, `default.d/`, and `local.example.toml`."""
    return data_dir(CONFIG_DIRNAME, package=package)


def local_config_file(*, package: Path | None = None) -> Path:
    """The default location of the untracked local configuration file.

    A source checkout keeps it at `<root>/local/config.toml`. An installed wheel has no
    repository, so the file is `local/config.toml` under the current working directory.
    The file is optional, and the path is a default that `load_config` can override.
    """
    root = source_root(package)
    base = Path.cwd() if root is None else root
    return base.joinpath(*LOCAL_CONFIG_PARTS)
