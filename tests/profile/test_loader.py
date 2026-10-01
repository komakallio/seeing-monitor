"""Loading profile files by name or by path, and the failures of a bad file."""

from __future__ import annotations

from pathlib import Path

import pytest

from seeingmon.profile import ProfileError, list_profiles, load_profile
from tests.profile.builders import REFERENCE_FILE

REFERENCE_TEXT = REFERENCE_FILE.read_text(encoding="utf-8")


def write_profile(directory: Path, stem: str, *, profile_id: str | None = None) -> Path:
    """Copy the reference profile to `<stem>.toml`, with another id if `profile_id` is set."""
    text = REFERENCE_TEXT
    if profile_id is not None:
        text = text.replace('id = "asi294mm-gs250"', f'id = "{profile_id}"', 1)
    path = directory / f"{stem}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_repository_profiles_load_and_carry_their_file_names(profiles_dir: Path) -> None:
    names = list_profiles(profiles_dir=profiles_dir)
    assert "asi294mm-gs250" in names
    for name in names:
        assert load_profile(name, profiles_dir=profiles_dir).id == name


def test_the_default_directory_is_the_repository_directory(profiles_dir: Path) -> None:
    """In a development environment, the packaged default and the repository folder coincide."""
    assert list_profiles() == list_profiles(profiles_dir=profiles_dir)
    assert load_profile("asi294mm-gs250").id == "asi294mm-gs250"


def test_a_profile_loads_by_name(tmp_path: Path) -> None:
    write_profile(tmp_path, "asi294mm-gs250")
    assert load_profile("asi294mm-gs250", profiles_dir=tmp_path).id == "asi294mm-gs250"


def test_a_profile_loads_by_path_object(tmp_path: Path) -> None:
    path = write_profile(tmp_path, "my-camera", profile_id="my-camera")
    assert load_profile(path).id == "my-camera"


def test_a_profile_loads_by_path_string(tmp_path: Path) -> None:
    path = write_profile(tmp_path, "my-camera", profile_id="my-camera")
    assert load_profile(str(path)).id == "my-camera"


def test_a_string_that_ends_in_toml_is_a_path_not_a_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_profile(tmp_path, "my-camera", profile_id="my-camera")
    monkeypatch.chdir(tmp_path)
    assert load_profile("my-camera.toml").id == "my-camera"


def test_a_name_that_matches_no_file_lists_the_available_profiles(tmp_path: Path) -> None:
    write_profile(tmp_path, "asi294mm-gs250")
    write_profile(tmp_path, "my-camera", profile_id="my-camera")
    with pytest.raises(ProfileError) as error:
        load_profile("nope", profiles_dir=tmp_path)
    assert (
        str(error.value) == "no profile named 'nope'; available profiles: asi294mm-gs250, my-camera"
    )


def test_a_name_that_is_neither_a_name_nor_a_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProfileError, match="is not a profile name"):
        load_profile("two words", profiles_dir=tmp_path)


def test_a_missing_file_path_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ProfileError, match="profile file not found"):
        load_profile(tmp_path / "missing.toml")


def test_a_profile_id_that_differs_from_the_file_name_is_rejected(tmp_path: Path) -> None:
    path = write_profile(tmp_path, "other-name")  # the id inside is asi294mm-gs250
    with pytest.raises(ProfileError) as error:
        load_profile(path)
    message = str(error.value)
    assert "the id 'asi294mm-gs250' must equal the file name" in message
    assert "('other-name')" in message
    assert str(path) in message


def test_the_id_check_applies_to_a_name_lookup_too(tmp_path: Path) -> None:
    write_profile(tmp_path, "my-camera", profile_id="something-else")
    with pytest.raises(ProfileError, match="the id 'something-else' must equal the file name"):
        load_profile("my-camera", profiles_dir=tmp_path)


def test_a_file_that_is_not_toml_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text('id = "broken"\nthis is not toml\n', encoding="utf-8")
    with pytest.raises(ProfileError, match="not valid TOML"):
        load_profile(path)


def test_a_file_that_is_not_utf8_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "binary.toml"
    path.write_bytes(b'id = "\xff\xfe"\n')
    with pytest.raises(ProfileError, match="cannot read profile"):
        load_profile(path)


def test_a_validation_failure_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "asi294mm-gs250.toml"
    path.write_text(REFERENCE_TEXT.replace("focal_length_mm = 250.0", "focal_length_mm = -1"))
    with pytest.raises(ProfileError) as error:
        load_profile(path)
    message = str(error.value)
    assert str(path) in message
    assert "optics.focal_length_mm: Input should be greater than 0" in message


def test_the_listing_is_sorted_and_skips_other_files_and_folders(tmp_path: Path) -> None:
    write_profile(tmp_path, "b-profile", profile_id="b-profile")
    write_profile(tmp_path, "a-profile", profile_id="a-profile")
    (tmp_path / "notes.txt").write_text("not a profile")
    (tmp_path / "folder.toml").mkdir()
    assert list_profiles(profiles_dir=tmp_path) == ["a-profile", "b-profile"]
