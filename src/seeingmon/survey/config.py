"""The `[survey]` configuration section.

The defaults live in `config/default.d/survey.toml`, and your installation overrides them in
`local/config.toml`. Paths, which are specific to an installation, stay empty in the defaults.
"""

from __future__ import annotations

from seeingmon.config import SectionModel


class SurveyConfig(SectionModel):
    """Settings of the survey path. Every key has a default that suits the reference camera."""

    # Files and programs.
    catalog_path: str = ""  # the cap catalog file; empty means "not configured"
    index_dir: str = ""  # the folder with the astrometry.net cap index files
    solve_field_command: str = "solve-field"
    astap_command: str = "astap"
    astap_database_dir: str = ""  # the folder with the ASTAP star database; empty uses its default
    solvers: tuple[str, ...] = ("astrometry.net", "astap")  # the order in which to try them
