"""Load, validate, and list profile files.

A profile is a TOML file named `<id>.toml`. `load_profile` takes a profile name (the file
stem of a file in `profiles/`) or a path to a file, and it checks that the `id` inside the
file equals the file stem. Every failure raises `ProfileError` with a message that names the
file and each wrong field.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from seeingmon import paths
from seeingmon.profile.errors import ProfileError
from seeingmon.profile.models import PROFILE_ID_PATTERN, Profile

PROFILE_SUFFIX = ".toml"
_VALUE_ERROR_PREFIX = "Value error, "


def _location(loc: Sequence[int | str], data: object) -> str:
    """Format an error location, such as `readout_modes[bin1].gain_points[1].gain`.

    A list index shows the `name` of the item when it has one, so you can find the item.
    """
    text = ""
    node: object = data
    for item in loc:
        if isinstance(item, int):
            child: object = None
            if isinstance(node, Sequence) and not isinstance(node, str) and 0 <= item < len(node):
                child = node[item]
            label = str(item)
            if isinstance(child, Mapping) and isinstance(child.get("name"), str):
                label = child["name"] or label
            text += f"[{label}]"
            node = child
        else:
            text += f".{item}" if text else item
            node = node.get(item) if isinstance(node, Mapping) else None
    return text


def format_validation_error(error: ValidationError, data: object = None) -> list[str]:
    """One line per problem: the location, then what is wrong. Input values stay out."""
    lines = []
    for item in error.errors(include_url=False, include_input=False):
        message = item["msg"].removeprefix(_VALUE_ERROR_PREFIX)
        location = _location(item["loc"], data)
        lines.append(f"{location}: {message}" if location else message)
    return lines


def parse_profile(data: Mapping[str, Any], *, source: str = "profile") -> Profile:
    """Validate the contents of a profile file. Raises `ProfileError` that names `source`."""
    try:
        return Profile.model_validate(data)
    except ValidationError as error:
        problems = "\n".join(f"  {line}" for line in format_validation_error(error, data))
        raise ProfileError(f"invalid profile {source}:\n{problems}") from error


def list_profiles(*, profiles_dir: Path | None = None) -> list[str]:
    """The names of the available profiles, sorted. A name is the file stem in `profiles/`."""
    directory = paths.profiles_dir() if profiles_dir is None else profiles_dir
    return sorted(path.stem for path in directory.glob(f"*{PROFILE_SUFFIX}") if path.is_file())


def _is_path_like(value: str) -> bool:
    return "/" in value or "\\" in value or value.lower().endswith(PROFILE_SUFFIX)


def profile_path(name_or_path: str | os.PathLike[str], *, profiles_dir: Path | None = None) -> Path:
    """The file for a profile name or a path. Raises `ProfileError` when a name has no file."""
    if isinstance(name_or_path, os.PathLike):
        return Path(os.fspath(name_or_path))
    if _is_path_like(name_or_path):
        return Path(name_or_path)
    if not re.fullmatch(PROFILE_ID_PATTERN, name_or_path):
        raise ProfileError(
            f"{name_or_path!r} is not a profile name (letters, digits, '.', '_', '-') "
            f"or a path to a {PROFILE_SUFFIX} file"
        )
    directory = paths.profiles_dir() if profiles_dir is None else profiles_dir
    path = directory / f"{name_or_path}{PROFILE_SUFFIX}"
    if not path.is_file():
        available = ", ".join(list_profiles(profiles_dir=directory)) or "none"
        raise ProfileError(f"no profile named {name_or_path!r}; available profiles: {available}")
    return path


def load_profile(
    name_or_path: str | os.PathLike[str], *, profiles_dir: Path | None = None
) -> Profile:
    """Load a profile by name (a file stem in `profiles/`) or by path to a `.toml` file.

    Raises `ProfileError` when the file is missing, is not valid TOML, fails validation, or
    has an `id` that differs from its file name.
    """
    path = profile_path(name_or_path, profiles_dir=profiles_dir)
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        raise ProfileError(f"profile file not found: {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise ProfileError(f"invalid profile {path}: not valid TOML ({error})") from error
    except (OSError, UnicodeDecodeError) as error:
        raise ProfileError(f"cannot read profile {path}: {error}") from error
    profile = parse_profile(data, source=str(path))
    if profile.id != path.stem:
        raise ProfileError(
            f"invalid profile {path}: the id {profile.id!r} must equal the file name "
            f"without {PROFILE_SUFFIX!r} ({path.stem!r})"
        )
    return profile
