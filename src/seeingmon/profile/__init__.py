"""Hardware profiles and the values that derive from them.

A profile describes one camera and telescope pair (see `seeingmon.profile.models`), and
`seeingmon.profile.derived` holds the pure functions of it. The names below import on first
use, so `seeingmon --help` does not load pydantic and NumPy to list the profile commands.

    from seeingmon.profile import load_profile

    profile = load_profile("asi294mm-gs250")
    profile.plate_scale_arcsec_per_px("bin1")
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from seeingmon.profile.derived import FieldOfView, Saturation
    from seeingmon.profile.errors import ProfileError
    from seeingmon.profile.loader import list_profiles, load_profile, parse_profile
    from seeingmon.profile.models import (
        DarkCurrentPoint,
        GainPoint,
        Limits,
        ModeSelection,
        Optics,
        Photometry,
        Profile,
        ReadoutMode,
        Sensor,
    )
    from seeingmon.profile.summary import profile_summary

_EXPORTS = {
    "DarkCurrentPoint": "seeingmon.profile.models",
    "FieldOfView": "seeingmon.profile.derived",
    "GainPoint": "seeingmon.profile.models",
    "Limits": "seeingmon.profile.models",
    "ModeSelection": "seeingmon.profile.models",
    "Optics": "seeingmon.profile.models",
    "Photometry": "seeingmon.profile.models",
    "Profile": "seeingmon.profile.models",
    "ProfileError": "seeingmon.profile.errors",
    "ReadoutMode": "seeingmon.profile.models",
    "Saturation": "seeingmon.profile.derived",
    "Sensor": "seeingmon.profile.models",
    "list_profiles": "seeingmon.profile.loader",
    "load_profile": "seeingmon.profile.loader",
    "parse_profile": "seeingmon.profile.loader",
    "profile_summary": "seeingmon.profile.summary",
}

__all__ = [
    "DarkCurrentPoint",
    "FieldOfView",
    "GainPoint",
    "Limits",
    "ModeSelection",
    "Optics",
    "Photometry",
    "Profile",
    "ProfileError",
    "ReadoutMode",
    "Saturation",
    "Sensor",
    "list_profiles",
    "load_profile",
    "parse_profile",
    "profile_summary",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # later lookups skip this function
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
