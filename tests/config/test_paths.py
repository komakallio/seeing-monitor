from __future__ import annotations

from pathlib import Path

import pytest

from seeingmon import paths
from seeingmon.paths import DataDirectoryError


def make_wheel_layout(base: Path) -> Path:
    """Build `site-packages/seeingmon/_data/{profiles,config}` and return the package folder."""
    package = base / "site-packages" / "seeingmon"
    (package / "_data" / "profiles").mkdir(parents=True)
    (package / "_data" / "config").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    return package


def make_checkout_layout(base: Path) -> Path:
    """Build `repo/{pyproject.toml,profiles,config,src/seeingmon}` and return the package folder."""
    root = base / "repo"
    package = root / "src" / "seeingmon"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (root / "pyproject.toml").write_text("")
    (root / "profiles").mkdir()
    (root / "config").mkdir()
    return package


def test_a_wheel_install_resolves_to_the_package_data(tmp_path: Path) -> None:
    package = make_wheel_layout(tmp_path)
    assert paths.profiles_dir(package=package) == package / "_data" / "profiles"
    assert paths.config_dir(package=package) == package / "_data" / "config"
    assert paths.source_root(package) is None


def test_a_source_checkout_resolves_to_the_repository_root(tmp_path: Path) -> None:
    package = make_checkout_layout(tmp_path)
    root = package.parents[1]
    assert paths.profiles_dir(package=package) == root / "profiles"
    assert paths.config_dir(package=package) == root / "config"
    assert paths.source_root(package) == root


def test_package_data_wins_when_both_layouts_exist(tmp_path: Path) -> None:
    package = make_checkout_layout(tmp_path)
    (package / "_data" / "profiles").mkdir(parents=True)
    assert paths.profiles_dir(package=package) == package / "_data" / "profiles"
    assert paths.config_dir(package=package) == package.parents[1] / "config"


def test_a_package_under_src_without_a_pyproject_is_not_a_checkout(tmp_path: Path) -> None:
    package = tmp_path / "project" / "src" / "seeingmon"
    package.mkdir(parents=True)
    (tmp_path / "project" / "profiles").mkdir()
    assert paths.source_root(package) is None
    with pytest.raises(DataDirectoryError):
        paths.profiles_dir(package=package)


def test_a_missing_directory_names_every_place_that_was_searched(tmp_path: Path) -> None:
    package = make_checkout_layout(tmp_path)
    (package.parents[1] / "profiles").rmdir()
    with pytest.raises(DataDirectoryError) as error:
        paths.profiles_dir(package=package)
    message = str(error.value)
    assert "'profiles'" in message
    assert str(package / "_data" / "profiles") in message
    assert str(package.parents[1] / "profiles") in message


def test_a_missing_directory_is_a_file_not_found_error(tmp_path: Path) -> None:
    package = make_wheel_layout(tmp_path)
    (package / "_data" / "config").rmdir()
    with pytest.raises(FileNotFoundError):
        paths.config_dir(package=package)


def test_a_file_in_place_of_the_directory_does_not_count(tmp_path: Path) -> None:
    package = make_wheel_layout(tmp_path)
    (package / "_data" / "profiles").rmdir()
    (package / "_data" / "profiles").write_text("not a directory")
    with pytest.raises(DataDirectoryError):
        paths.profiles_dir(package=package)


def test_the_local_config_file_sits_in_the_checkout_root(tmp_path: Path) -> None:
    package = make_checkout_layout(tmp_path)
    expected = package.parents[1] / "local" / "config.toml"
    assert paths.local_config_file(package=package) == expected


def test_the_local_config_file_follows_the_working_directory_for_a_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = make_wheel_layout(tmp_path)
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    assert paths.local_config_file(package=package) == Path.cwd() / "local" / "config.toml"


def test_the_running_package_is_the_checkout_in_a_development_environment(
    repo_root: Path,
) -> None:
    """An editable install runs from `src/`, so the checkout layout applies."""
    assert paths.package_dir() == repo_root / "src" / "seeingmon"
    assert paths.source_root() == repo_root
