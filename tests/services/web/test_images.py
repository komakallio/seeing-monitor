"""Read-only access to the preview and FITS files: the IDs, the listing, and the limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from seeingmon.clock import NS_PER_S
from seeingmon.services.web.config import ImageSettings
from seeingmon.services.web.data import InvalidQueryError
from seeingmon.services.web.images import ImageKey, ImageStore, parse_image_id
from seeingmon.store.layout import DataLayout
from tests.services.web.seed import write_fits, write_preview

STAMPS = [
    "20260929T210000.000Z",
    "20260930T020000.500Z",
    "20260930T020000.750Z",
    "20260930T230000.000Z",
    "20261001T020000.000Z",
]


@pytest.fixture
def store(layout: DataLayout) -> ImageStore:
    return ImageStore(layout, ImageSettings())


@pytest.fixture
def filled(layout: DataLayout, store: ImageStore) -> ImageStore:
    for stamp in STAMPS:
        write_preview(layout, stamp)
    write_fits(layout, STAMPS[1])
    return store


# --- The ID ----------------------------------------------------------------------------------


def test_an_id_is_the_name_of_the_preview_without_its_extension() -> None:
    key = parse_image_id("preview-20261001T201500.123Z")
    assert key.kind == "preview"
    assert key.stamp == "20261001T201500.123Z"
    assert (key.year, key.month, key.day) == ("2026", "10", "01")
    assert key.id == "preview-20261001T201500.123Z"
    assert key.date == "20261001"


def test_an_id_knows_its_time() -> None:
    key = parse_image_id("preview-20261001T020000.500Z")
    assert key.t_utc_ns == 1_790_820_000 * NS_PER_S + NS_PER_S // 2


def test_a_stamp_without_milliseconds_is_an_id_too() -> None:
    assert parse_image_id("preview-20261001T020000Z").t_utc_ns == 1_790_820_000 * NS_PER_S


@pytest.mark.parametrize(
    "text",
    [
        "",
        "preview",
        "preview-",
        "../preview-20261001T020000.000Z",
        "preview-20261001T020000.000Z/../x",
        "preview-20261001T020000.000Z.jpg",
        "..\\preview-20261001T020000.000Z",
        "/etc/passwd",
        "C:\\Windows\\win.ini",  # repo-check: allow
        "preview-20261001T020000.000Z\x00",
        "preview-20261001T020000.000Z\n",
        "Preview-20261001T020000.000Z",
        "1preview-20261001T020000.000Z",
        "preview-2026100T020000.000Z",
        "preview-20261301T020000.000Z",
        "preview-20261032T020000.000Z",
        "preview-20261001T250000.000Z",
        "preview-20261001T026000.000Z",
        "preview-00001001T020000.000Z",
        "preview-20261001T020000.0000Z",
        "a" * 40 + "-20261001T020000.000Z",
        "preview-20261001T020000.000Z%2f..%2f",
        "preview-\u0662\u0660\u0662\u0666\u0661\u0660\u0660\u0661T020000.000Z",  # Arabic digits
    ],
)
def test_text_that_is_not_an_id_is_refused(text: str) -> None:
    with pytest.raises(InvalidQueryError):
        parse_image_id(text)


def test_the_id_pattern_never_lets_a_separator_through() -> None:
    for text in ("a/b", "a\\b", "..", "."):
        with pytest.raises(InvalidQueryError):
            parse_image_id(f"preview-20261001T020000.000Z{text}")


# --- The listing -----------------------------------------------------------------------------


def test_the_listing_is_newest_first_across_the_day_folders(filled: ImageStore) -> None:
    images, more = filled.recent(10)
    assert [image.key.stamp for image in images] == STAMPS[::-1]
    assert more is False


def test_the_latest_image_is_the_newest(filled: ImageStore) -> None:
    latest = filled.latest()
    assert latest is not None
    assert latest.key.stamp == STAMPS[-1]


def test_an_empty_folder_has_no_images(store: ImageStore) -> None:
    assert store.latest() is None
    assert store.recent(10) == ([], False)


def test_a_missing_data_directory_has_no_images(tmp_path: Path) -> None:
    store = ImageStore(DataLayout(tmp_path / "nothing"), ImageSettings())
    assert store.latest() is None


def test_a_page_says_whether_more_images_follow(filled: ImageStore) -> None:
    images, more = filled.recent(2)
    assert [image.key.stamp for image in images] == [STAMPS[4], STAMPS[3]]
    assert more is True
    images, more = filled.recent(5)
    assert len(images) == 5
    assert more is False


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 5])
def test_pages_follow_one_another_through_the_before_key(filled: ImageStore, limit: int) -> None:
    seen: list[str] = []
    before: ImageKey | None = None
    for _ in range(10):
        images, more = filled.recent(limit, before)
        seen.extend(image.key.stamp for image in images)
        if not more:
            break
        before = images[-1].key
    assert seen == STAMPS[::-1]


def test_two_kinds_with_one_stamp_are_two_images(layout: DataLayout, store: ImageStore) -> None:
    write_preview(layout, STAMPS[0], kind="preview")
    write_preview(layout, STAMPS[0], kind="align")
    images, _ = store.recent(10)
    assert sorted(image.id for image in images) == [
        f"align-{STAMPS[0]}",
        f"preview-{STAMPS[0]}",
    ]


def test_a_stray_file_or_folder_is_not_an_image(layout: DataLayout, store: ImageStore) -> None:
    write_preview(layout, STAMPS[1])
    folder = layout.previews_dir / "2026" / "09" / "30"
    (folder / "notes.txt").write_text("x", encoding="utf-8")
    (folder / "preview-20260930T020000.500Z.jpg.part").write_bytes(b"x")
    (folder / "preview-20260930T999999.000Z.jpg").write_bytes(b"x")
    (folder / "preview-20260929T020000.000Z.jpg").write_bytes(b"x")  # the stamp is of another day
    (folder / "subfolder.jpg").mkdir()
    (layout.previews_dir / "2026" / "notes").mkdir()
    (layout.previews_dir / "latest").mkdir()
    images, _ = store.recent(10)
    assert [image.key.stamp for image in images] == [STAMPS[1]]


def test_the_listing_visits_only_the_newest_day_folders(layout: DataLayout) -> None:
    for stamp in STAMPS:
        write_preview(layout, stamp)
    store = ImageStore(layout, ImageSettings(scan_days=2))
    images, more = store.recent(10)
    assert [image.key.stamp for image in images] == [STAMPS[4], STAMPS[3], STAMPS[2], STAMPS[1]]
    assert more is False  # the two older days are not visited, so they do not show


# --- The files -------------------------------------------------------------------------------


def test_an_image_knows_the_size_of_its_preview_and_of_its_fits_frame(filled: ImageStore) -> None:
    with_fits = filled.get(f"preview-{STAMPS[1]}")
    without = filled.get(f"preview-{STAMPS[2]}")
    assert with_fits is not None
    assert without is not None
    assert with_fits.has_fits
    assert with_fits.fits_bytes == 5760
    assert with_fits.size_bytes > 0
    assert not without.has_fits
    assert without.fits_bytes is None


def test_the_paths_come_from_the_layout(filled: ImageStore, layout: DataLayout) -> None:
    info = filled.get(f"preview-{STAMPS[1]}")
    assert info is not None
    preview = filled.preview_path(info.key)
    fits = filled.fits_path(info.key)
    assert preview == layout.previews_dir / "2026" / "09" / "30" / f"preview-{STAMPS[1]}.jpg"
    assert fits == layout.survey_dir / "2026" / "09" / "30" / f"{STAMPS[1]}.fits"
    assert preview is not None
    assert preview.is_file()


def test_an_image_that_does_not_exist_is_none(filled: ImageStore) -> None:
    assert filled.get("preview-20250101T000000.000Z") is None
    assert filled.get("other-20260930T020000.500Z") is None


def test_a_preview_above_the_size_limit_does_not_count(layout: DataLayout) -> None:
    path = write_preview(layout, STAMPS[0])
    path.write_bytes(path.read_bytes().ljust(2048, b"\0"))
    limit = path.stat().st_size
    assert limit == 2048
    small = ImageStore(layout, ImageSettings(max_preview_bytes=limit - 1))
    assert small.get(f"preview-{STAMPS[0]}") is None
    assert small.recent(10) == ([], False)
    exact = ImageStore(layout, ImageSettings(max_preview_bytes=limit))
    assert exact.get(f"preview-{STAMPS[0]}") is not None


def test_a_fits_frame_above_the_size_limit_does_not_count(layout: DataLayout) -> None:
    write_preview(layout, STAMPS[0])
    write_fits(layout, STAMPS[0], size=2880 * 3)
    store = ImageStore(layout, ImageSettings(max_fits_bytes=2880 * 2))
    info = store.get(f"preview-{STAMPS[0]}")
    assert info is not None
    assert not info.has_fits


def test_a_symbolic_link_is_not_followed_out_of_its_folder(
    layout: DataLayout, store: ImageStore, tmp_path: Path
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("a secret", encoding="utf-8")
    folder = layout.previews_dir / "2026" / "10" / "01"
    folder.mkdir(parents=True)
    link = folder / f"preview-{STAMPS[4]}.jpg"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow symbolic links")
    assert store.get(f"preview-{STAMPS[4]}") is None
    assert store.recent(10) == ([], False)


def test_a_symbolic_link_to_a_sibling_file_is_not_followed_either(
    layout: DataLayout, store: ImageStore
) -> None:
    write_preview(layout, STAMPS[3])
    folder = layout.previews_dir / "2026" / "09" / "30"
    link = folder / f"preview-{STAMPS[1]}.jpg"
    try:
        link.symlink_to(folder / f"preview-{STAMPS[3]}.jpg")
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow symbolic links")
    assert store.get(f"preview-{STAMPS[1]}") is None
    assert [image.key.stamp for image in store.recent(10)[0]] == [STAMPS[3]]


# --- From a record to its image --------------------------------------------------------------


class TestFromARecordToItsImage:
    """`for_ref` turns the `image_ref` of a `survey_frame` record into the image."""

    def test_a_reference_to_a_preview_names_that_image(
        self, layout: DataLayout, store: ImageStore
    ) -> None:
        write_preview(layout, STAMPS[1], kind="survey")
        ref = layout.relative(layout.preview_path(_time_of(STAMPS[1]), kind="survey"))
        assert ref == "previews/2026/09/30/survey-20260930T020000.500Z.jpg"
        info = store.for_ref(ref)
        assert info is not None
        assert info.id == f"survey-{STAMPS[1]}"
        assert not info.has_fits

    def test_a_reference_to_a_fits_file_names_the_preview_with_its_stamp(
        self, layout: DataLayout, store: ImageStore
    ) -> None:
        write_preview(layout, STAMPS[1], kind="event")  # the kind is not in the FITS name
        write_preview(layout, STAMPS[2], kind="survey")
        write_fits(layout, STAMPS[1])
        ref = layout.relative(layout.survey_path(_time_of(STAMPS[1])))
        assert ref == "survey/2026/09/30/20260930T020000.500Z.fits"
        info = store.for_ref(ref)
        assert info is not None
        assert info.id == f"event-{STAMPS[1]}"
        assert info.has_fits

    def test_the_image_of_a_reference_is_the_image_that_the_list_shows(
        self, layout: DataLayout, store: ImageStore
    ) -> None:
        for stamp in STAMPS:
            write_preview(layout, stamp, kind="survey")
        write_fits(layout, STAMPS[3])
        listed = {image.id: image for image in store.recent(10)[0]}
        for stamp in STAMPS:
            path = layout.survey_path(_time_of(stamp)) if stamp == STAMPS[3] else None
            path = path or layout.preview_path(_time_of(stamp), kind="survey")
            info = store.for_ref(layout.relative(path))
            assert info is not None
            assert info == listed[info.id]

    def test_a_file_that_retention_deleted_gives_none(
        self, layout: DataLayout, store: ImageStore
    ) -> None:
        preview = write_preview(layout, STAMPS[1], kind="survey")
        write_fits(layout, STAMPS[1])
        preview.unlink()
        assert store.for_ref("previews/2026/09/30/survey-20260930T020000.500Z.jpg") is None
        assert store.for_ref("survey/2026/09/30/20260930T020000.500Z.fits") is None  # no preview

    def test_a_fits_file_without_a_fits_frame_still_finds_the_preview(
        self, layout: DataLayout, store: ImageStore
    ) -> None:
        write_preview(layout, STAMPS[1], kind="survey")  # the FITS file expired, the preview stayed
        info = store.for_ref("survey/2026/09/30/20260930T020000.500Z.fits")
        assert info is not None
        assert not info.has_fits

    @pytest.mark.parametrize(
        "ref",
        [
            "",
            "survey",
            "../x",
            "/previews/2026/09/30/survey-20260930T020000.500Z.jpg",
            "previews/2026/09/30/../30/survey-20260930T020000.500Z.jpg",
            r"previews\2026\09\30\survey-20260930T020000.500Z.jpg",
            "C:/previews/2026/09/30/survey-20260930T020000.500Z.jpg",  # repo-check: allow
            "previews/2026/09/30/survey-20260930T020000.500Z.png",
            "previews/2026/09/30/survey-20260930T020000.500Z",
            "previews/2026/09/30/other.jpg",
            "previews/2026/09/29/survey-20260930T020000.500Z.jpg",  # the folder says another day
            "survey/2026/09/29/20260930T020000.500Z.fits",
            "survey/2026/09/30/20260930T020000.500Z.jpg",
            "survey/2026/09/30/20260930T020000.500Z.fits.gz",
            "survey/2026/09/30/2026.fits",
            "survey/2026/09/30/%2e%2e.fits",
            "calibration/darks/dark-20260930T020000Z-bin2-g120.fits",
            "bursts/20260930T020000Z/burst.ser",
            "previews/2026/09/30/survey-20260930T020000.500Z.jpg\x00",
            "previews/2026/09/30/survey-20260930T020000.500Z.jpg\n",
            "previews/2026/09/30/" + "a" * 200 + ".jpg",
        ],
    )
    def test_a_reference_that_is_not_an_image_gives_none(
        self, layout: DataLayout, store: ImageStore, ref: str
    ) -> None:
        write_preview(layout, STAMPS[1], kind="survey")
        write_fits(layout, STAMPS[1])
        assert store.for_ref(ref) is None


def _time_of(stamp: str) -> int:
    """The nanoseconds of a stamp, with the parser of the web process."""
    return parse_image_id(f"preview-{stamp}").t_utc_ns
