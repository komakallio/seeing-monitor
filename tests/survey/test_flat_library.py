"""The flat library: the files of a flat, the pointer, the retention, the session folder, and the
flat that the survey uses.

The flats of the library tests are small images, and the report comes from a real run of
`make_flat` on a small synthetic panel (the lens of the owner on a sensor of 256 x 176 pixels).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, VirtualClock, iso_to_utc_ns
from seeingmon.profile import Profile
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.survey import flat_library as fl
from seeingmon.survey import flat_make as fm
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.flat_files import FlatFileError
from seeingmon.survey.pipeline import PipelineSpec, SurveyPipeline, build_pipeline
from seeingmon.survey.sky import ArrayFlat, SkyError, UnitFlat
from tests.survey import flatfx as fx
from tests.survey import synth

SHAPE = (176, 256)  # (height, width) of the sensor of the library tests
NOW = iso_to_utc_ns("2026-10-04T20:15:00Z")
BIAS = fm.LibraryBias(133.9, "from the dark library (test)")


def small_result(*, sets: int = 1, seed: int = 1) -> fm.MakeResult:
    profile = fx.scaled_profile(SHAPE[1], SHAPE[0])
    readout = profile.survey_readout
    geometry = fm.Geometry(SHAPE, profile.plate_scale_arcsec_per_px(readout), readout.adc_bits)
    truth = fx.lens_flat(
        fx.OWNER_LENS, SHAPE, scale_down=fx.REFERENCE_SHAPE[1] / SHAPE[1], edge_artifact_x=3.0
    )
    gradient = fx.OWNER_SOURCE_GRADIENT
    frames = [
        fx.panel_set(truth, gradient, frames=16, seed=seed, shift=0),
        fx.panel_set(truth, (-gradient[0], -gradient[1]), frames=16, seed=seed + 1, shift=0),
    ][:sets]
    options = fm.MakeOptions(
        bin_factor=2, high_pass_px=10.0, edge_margin_px=6.0, full_scale=16383.0
    )
    return fm.make_flat(
        frames,
        bias=fm.BiasInput(library=BIAS),
        geometry=geometry,
        options=options,
        source_turned=sets > 1,
        clock=VirtualClock(),
    )


@pytest.fixture(scope="module")
def one_set() -> fm.MakeResult:
    return small_result()


@pytest.fixture(scope="module")
def two_sets() -> fm.MakeResult:
    return small_result(sets=2)


def info(**changes: Any) -> fl.FlatInfo:
    base = fl.FlatInfo(
        t_utc_ns=NOW,
        mode="bin2",
        gain=120,
        target_fraction=0.5,
        exposures_s=(0.04, 0.041),
        level_fractions=(0.49, 0.5),
        temperature_c=12.34,
    )
    return replace(base, **changes)


def tiny_flat(seed: int) -> np.ndarray:
    """A small flat that differs from every other seed, so that each has its own version."""
    rng = np.random.default_rng(seed)
    return np.asarray(1.0 + 0.05 * rng.standard_normal((12, 16)), dtype=np.float32)


def report_for(seed: int, *, t_utc_ns: int | None = None) -> dict[str, Any]:
    return {"t_utc_ns": NOW + seed * NS_PER_S if t_utc_ns is None else t_utc_ns, "mode": "bin2"}


@pytest.fixture
def library(tmp_path: Path) -> fl.FlatLibrary:
    return fl.FlatLibrary(tmp_path / "calibration" / "flats")


# --- The report -------------------------------------------------------------------------------


class TestTheReport:
    def test_it_is_plain_json_with_the_numbers_that_the_page_needs(
        self, one_set: fm.MakeResult
    ) -> None:
        report = fl.build_report(one_set, info(exposures_s=(0.04,), level_fractions=(0.49,)))
        text = json.dumps(report, allow_nan=False)
        assert json.loads(text) == report
        assert report["schema"] == 1
        assert report["t_utc"] == "2026-10-04T20:15:00Z"
        assert (report["mode"], report["gain"]) == ("bin2", 120)
        assert (report["width_px"], report["height_px"]) == (256, 176)
        assert report["sensor_temperature_c"] == 12.34
        assert report["exposure_s"] == 0.04
        assert report["second_set"] is False
        assert report["split"] is None
        assert report["agreement"] is None
        assert report["bias"]["source"] == "library"
        assert "dark library" in report["bias"]["note"]

    def test_the_vignetting_has_five_radii_and_the_corners(self, one_set: fm.MakeResult) -> None:
        report = fl.build_report(one_set, info(exposures_s=(0.04,), level_fractions=(0.5,)))
        points = report["vignetting"]["points"]
        assert [p["corner"] for p in points] == [False] * 5 + [True]
        assert [p["radius_deg"] for p in points[:5]] == [0.5, 1.0, 1.5, 2.0, 2.5]
        corner = points[-1]["change_percent"]
        assert report["vignetting"]["corner_percent"] == pytest.approx(corner, abs=0.01)
        assert -12.0 < corner < -8.0  # the lens of the owner loses 10% in the corners

    def test_the_tilt_the_shadows_and_the_noise_are_there(self, one_set: fm.MakeResult) -> None:
        report = fl.build_report(one_set, info(exposures_s=(0.04,), level_fractions=(0.5,)))
        assert set(report["tilt"]) == {"width_percent", "height_percent"}
        shadows = report["shadows"]
        assert shadows["count"] == len(one_set.summary.shadows)
        assert shadows["min_depth_percent"] >= 1.0
        for item in shadows["items"]:
            assert set(item) == {"x_px", "y_px", "depth_percent", "width_px"}
        assert report["edge_artifacts"] == len(one_set.summary.edge_artifacts)
        assert report["noise_percent"] > 0

    def test_a_set_reports_its_frames_its_exposure_and_its_level(
        self, one_set: fm.MakeResult
    ) -> None:
        report = fl.build_report(one_set, info(exposures_s=(0.04,), level_fractions=(0.49,)))
        (first,) = report["sets"]
        assert first["number"] == 1
        assert (first["frames"], first["used"]) == (16, 16)
        assert first["dropped"] == {}
        assert (first["exposure_s"], first["level_fraction"]) == (0.04, 0.49)

    def test_two_sets_add_the_agreement_and_the_split_of_the_tilt(
        self, two_sets: fm.MakeResult
    ) -> None:
        report = fl.build_report(two_sets, info())
        assert report["second_set"] is True
        assert report["source_turned"] is True
        assert [s["number"] for s in report["sets"]] == [1, 2]
        assert [s["exposure_s"] for s in report["sets"]] == [0.04, 0.041]
        assert set(report["split"]) == {"optics", "source"}
        assert set(report["agreement"]) == {
            "smooth_rms_percent",
            "fine_rms_percent",
            "expected_fine_rms_percent",
            "plane",
        }
        # the gradient of the source of the owner is about 0.65% across the width
        assert abs(report["split"]["source"]["width_percent"]) == pytest.approx(0.64, abs=0.2)

    def test_the_warning_about_one_set_has_no_command_line_option_in_it(
        self, one_set: fm.MakeResult
    ) -> None:
        assert any("--source-turned" in text for text in one_set.warnings)  # the tool's words
        report = fl.build_report(one_set, info(exposures_s=(0.04,), level_fractions=(0.5,)))
        assert report["warnings"]
        assert not any("--" in text for text in report["warnings"])
        assert any(
            "second set with the source turned by 180 degrees" in t for t in report["warnings"]
        )

    def test_a_value_that_is_not_finite_becomes_null(self, one_set: fm.MakeResult) -> None:
        report = fl.build_report(
            one_set, info(temperature_c=float("nan"), exposures_s=(0.04,), level_fractions=(0.5,))
        )
        assert report["sensor_temperature_c"] is None
        json.dumps(report, allow_nan=False)

    def test_the_report_names_no_path(self, one_set: fm.MakeResult, tmp_path: Path) -> None:
        report = fl.build_report(one_set, info(exposures_s=(0.04,), level_fractions=(0.5,)))
        text = json.dumps(report)
        assert str(tmp_path) not in text
        assert "\\\\" not in text


# --- The preview ------------------------------------------------------------------------------


class TestThePreview:
    def test_it_is_a_jpeg_stretched_to_plus_and_minus_ten_percent(self) -> None:
        from io import BytesIO

        from PIL import Image

        ramp = np.tile(np.linspace(0.85, 1.15, 64, dtype=np.float32), (32, 1))
        jpeg = fl.render_preview(ramp)
        assert jpeg.startswith(b"\xff\xd8\xff")
        pixels = np.asarray(Image.open(BytesIO(jpeg)))
        assert pixels.shape == (32, 64)
        row = pixels[16].astype(int)
        assert row[0] <= 4  # 0.85 is below 0.9, so black
        assert row[-1] >= 251  # and 1.15 is above 1.1, so white
        middle = row[24:40].mean()
        assert 118 <= middle <= 138  # 1.0 is gray

    def test_a_wide_flat_shrinks_to_the_limit(self) -> None:
        from io import BytesIO

        from PIL import Image

        wide = np.ones((300, 2200), dtype=np.float32)
        image = Image.open(BytesIO(fl.render_preview(wide, max_width_px=1000)))
        assert image.width <= 1000
        assert image.width >= 550  # the block is the smallest whole number that fits


# --- The files and the list -------------------------------------------------------------------


class TestAddingAFlat:
    def test_a_flat_has_three_files_and_the_version_of_its_pixels(
        self, library: fl.FlatLibrary
    ) -> None:
        flat = tiny_flat(1)
        entry = library.add(flat, report_for(1), preview=b"\xff\xd8\xff-jpeg")
        assert entry.version == ArrayFlat(flat).version
        assert fl.is_version(entry.version)
        names = sorted(path.name for path in library.directory.iterdir())
        assert names == [f"{entry.version}.jpg", f"{entry.version}.json", f"{entry.version}.npy"]
        assert library.jpeg(entry.version) == b"\xff\xd8\xff-jpeg"

    def test_a_new_flat_is_pending_and_the_survey_does_not_use_it(
        self, library: fl.FlatLibrary
    ) -> None:
        entry = library.add(tiny_flat(1), report_for(1))
        assert entry.pending is True
        assert entry.state == "pending"
        assert library.active_version() is None
        assert library.pointer_stamp() is None

    def test_the_file_loads_to_the_same_version_that_the_provenance_will_name(
        self, library: fl.FlatLibrary
    ) -> None:
        entry = library.add(tiny_flat(2), report_for(2))
        model = library.load(entry.version)
        assert model.version == entry.version
        assert library.get(entry.version) == entry

    def test_the_report_keeps_what_the_caller_gave_and_adds_the_version_and_the_state(
        self, library: fl.FlatLibrary
    ) -> None:
        entry = library.add(tiny_flat(1), {**report_for(1), "note": "kept"})
        stored = json.loads((library.directory / f"{entry.version}.json").read_text("utf-8"))
        assert stored["note"] == "kept"
        assert stored["version"] == entry.version
        assert stored["state"] == "pending"
        assert stored["schema"] == 1

    def test_a_flat_that_holds_a_value_at_or_below_zero_leaves_no_file(
        self, library: fl.FlatLibrary
    ) -> None:
        flat = tiny_flat(1)
        flat[0, 0] = 0.0
        with pytest.raises(SkyError):
            library.add(flat, report_for(1))
        assert not library.directory.exists()

    def test_a_report_with_a_value_that_json_refuses_leaves_no_file(
        self, library: fl.FlatLibrary
    ) -> None:
        with pytest.raises(ValueError, match="Out of range float"):
            library.add(tiny_flat(1), {"t_utc_ns": NOW, "bad": float("nan")})
        assert not library.directory.exists() or not any(library.directory.iterdir())

    def test_a_write_error_removes_the_files_that_were_written(
        self, library: fl.FlatLibrary, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(path: Path, data: object) -> Path:
            raise PermissionError("denied")

        monkeypatch.setattr(fl, "write_atomic", broken)
        with pytest.raises(PermissionError):
            library.add(tiny_flat(1), report_for(1), preview=b"x")
        assert not library.directory.exists() or not any(library.directory.iterdir())

    def test_a_failed_npy_write_is_a_flat_file_error_without_leftovers(
        self, library: fl.FlatLibrary, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(path: Path, flat: object) -> None:
            raise FlatFileError("cannot write the flat: disk full")

        monkeypatch.setattr(fl, "write_flat", broken)
        with pytest.raises(FlatFileError):
            library.add(tiny_flat(1), report_for(1))
        assert not library.directory.exists() or not any(library.directory.iterdir())


class TestTheList:
    def test_the_newest_flat_comes_first(self, library: fl.FlatLibrary) -> None:
        for seed in (1, 3, 2):
            library.add(tiny_flat(seed), report_for(seed))
        assert [e.t_utc_ns for e in library.entries()] == [
            NOW + 3 * NS_PER_S,
            NOW + 2 * NS_PER_S,
            NOW + 1 * NS_PER_S,
        ]

    def test_an_empty_or_missing_folder_is_an_empty_list(self, library: fl.FlatLibrary) -> None:
        assert library.entries() == []
        library.directory.mkdir(parents=True)
        assert library.entries() == []

    def test_a_report_that_cannot_be_read_is_skipped_and_named_once(
        self, library: fl.FlatLibrary, caplog: pytest.LogCaptureFixture
    ) -> None:
        good = library.add(tiny_flat(1), report_for(1))
        (library.directory / "flat-deadbeef.json").write_text("{not json", encoding="utf-8")
        (library.directory / "flat-12345678.json").write_text('{"schema": 7}', encoding="utf-8")
        (library.directory / "notes.json").write_text("{}", encoding="utf-8")
        with caplog.at_level("WARNING", logger="seeingmon.survey"):
            assert [e.version for e in library.entries()] == [good.version]
            library.entries()
        assert len([r for r in caplog.records if "skips the report" in r.getMessage()]) == 2

    def test_a_report_of_another_version_than_its_name_is_skipped(
        self, library: fl.FlatLibrary
    ) -> None:
        entry = library.add(tiny_flat(1), report_for(1))
        report = json.loads((library.directory / f"{entry.version}.json").read_text("utf-8"))
        (library.directory / "flat-00000000.json").write_text(json.dumps(report), encoding="utf-8")
        assert [e.version for e in library.entries()] == [entry.version]

    def test_get_answers_none_for_a_name_that_is_no_version(self, library: fl.FlatLibrary) -> None:
        library.add(tiny_flat(1), report_for(1))
        for name in ("", "flat-1", "../flat-12345678", "flat-1234567g", "FLAT-12345678"):
            assert library.get(name) is None

    def test_the_jpeg_of_a_flat_without_a_preview_is_none(self, library: fl.FlatLibrary) -> None:
        entry = library.add(tiny_flat(1), report_for(1))
        assert library.jpeg(entry.version) is None
        assert library.jpeg("not a version") is None


# --- Activation -------------------------------------------------------------------------------


class TestActivation:
    def test_the_pointer_names_the_flat_and_the_report_says_approved(
        self, library: fl.FlatLibrary
    ) -> None:
        entry = library.add(tiny_flat(1), report_for(1))
        active = library.activate(entry.version, now_utc_ns=NOW)
        assert library.active_version() == entry.version
        assert active.state == "approved"
        assert active.pending is False
        assert active.report["activated_utc"] == "2026-10-04T20:15:00Z"
        pointer = json.loads((library.directory / "current.json").read_text(encoding="utf-8"))
        assert pointer == {"version": entry.version, "activated_utc": "2026-10-04T20:15:00Z"}

    def test_the_stamp_of_the_pointer_changes_with_each_activation(
        self, library: fl.FlatLibrary
    ) -> None:
        first = library.add(tiny_flat(1), report_for(1))
        second = library.add(tiny_flat(2), report_for(2))
        library.activate(first.version, now_utc_ns=NOW)
        before = library.pointer_stamp()
        assert before is not None
        library.activate(second.version, now_utc_ns=NOW + NS_PER_S)
        after = library.pointer_stamp()
        assert after is not None
        assert after != before
        assert library.active_version() == second.version

    def test_an_unknown_flat_cannot_be_activated(self, library: fl.FlatLibrary) -> None:
        library.add(tiny_flat(1), report_for(1))
        for name in ("flat-00000000", "nonsense", "../current"):
            with pytest.raises(fl.FlatLibraryError) as error:
                library.activate(name, now_utc_ns=NOW)
            assert error.value.reason == "unknown"
        assert library.active_version() is None

    def test_a_flat_that_does_not_fit_the_survey_frame_is_refused_in_words(
        self, library: fl.FlatLibrary
    ) -> None:
        entry = library.add(tiny_flat(1), report_for(1))
        with pytest.raises(fl.FlatLibraryError) as error:
            library.activate(entry.version, now_utc_ns=NOW, expect_shape=(800, 1200))
        assert error.value.reason == "invalid"
        assert (
            error.value.message
            == "The flat has 16 x 12 pixels, and the survey frame has 1200 x 800."
        )
        assert library.active_version() is None
        library.activate(entry.version, now_utc_ns=NOW, expect_shape=(12, 16))
        assert library.active_version() == entry.version

    def test_a_flat_whose_pixels_are_gone_is_refused(self, library: fl.FlatLibrary) -> None:
        entry = library.add(tiny_flat(1), report_for(1))
        (library.directory / f"{entry.version}.npy").unlink()
        with pytest.raises(fl.FlatLibraryError) as error:
            library.activate(entry.version, now_utc_ns=NOW)
        assert error.value.reason == "unknown"

    def test_a_file_that_is_not_a_flat_is_refused_without_a_path(
        self, library: fl.FlatLibrary, tmp_path: Path
    ) -> None:
        entry = library.add(tiny_flat(1), report_for(1))
        (library.directory / f"{entry.version}.npy").write_bytes(b"not a numpy file")
        with pytest.raises(fl.FlatLibraryError) as error:
            library.activate(entry.version, now_utc_ns=NOW)
        assert error.value.reason == "invalid"
        assert str(tmp_path) not in error.value.message

    def test_a_pointer_that_names_nonsense_means_no_active_flat(
        self, library: fl.FlatLibrary
    ) -> None:
        library.directory.mkdir(parents=True)
        for text in ('{"version": "../../etc/passwd"}', "[]", "{", '{"version": 7}'):
            (library.directory / "current.json").write_text(text, encoding="utf-8")
            assert library.active_version() is None


class TestDeleting:
    def test_a_pending_flat_goes_with_its_three_files(self, library: fl.FlatLibrary) -> None:
        entry = library.add(tiny_flat(1), report_for(1), preview=b"x")
        library.delete(entry.version)
        assert library.entries() == []
        assert list(library.directory.iterdir()) == []

    def test_an_inactive_flat_goes_and_the_active_one_stays(self, library: fl.FlatLibrary) -> None:
        old = library.add(tiny_flat(1), report_for(1))
        new = library.add(tiny_flat(2), report_for(2))
        library.activate(old.version, now_utc_ns=NOW)
        library.activate(new.version, now_utc_ns=NOW + NS_PER_S)
        library.delete(old.version)  # approved once, inactive now
        assert [e.version for e in library.entries()] == [new.version]

    def test_the_active_flat_cannot_be_deleted(self, library: fl.FlatLibrary) -> None:
        entry = library.add(tiny_flat(1), report_for(1))
        library.activate(entry.version, now_utc_ns=NOW)
        with pytest.raises(fl.FlatLibraryError) as error:
            library.delete(entry.version)
        assert error.value.reason == "active"
        assert library.get(entry.version) is not None
        assert library.load(entry.version).version == entry.version

    def test_an_unknown_flat_cannot_be_deleted(self, library: fl.FlatLibrary) -> None:
        with pytest.raises(fl.FlatLibraryError) as error:
            library.delete("flat-00000000")
        assert error.value.reason == "unknown"


class TestRetention:
    def test_the_newest_ten_stay(self, library: fl.FlatLibrary) -> None:
        for seed in range(1, 13):
            library.add(tiny_flat(seed), report_for(seed))
        kept = library.entries()
        assert len(kept) == fl.KEEP_FLATS == 10
        assert [e.t_utc_ns for e in kept] == [NOW + s * NS_PER_S for s in range(12, 2, -1)]
        npy_files = sorted(library.directory.glob("*.npy"))
        assert len(npy_files) == 10  # the files of the two oldest are gone

    def test_the_active_flat_stays_however_old_it_is(self, library: fl.FlatLibrary) -> None:
        oldest = library.add(tiny_flat(1), report_for(1))
        library.activate(oldest.version, now_utc_ns=NOW)
        for seed in range(2, 15):
            library.add(tiny_flat(seed), report_for(seed))
        versions = [e.version for e in library.entries()]
        assert oldest.version in versions
        assert len(versions) == 11  # the newest ten and the active one
        assert library.load(oldest.version).version == oldest.version

    def test_the_number_to_keep_is_a_parameter(self, library: fl.FlatLibrary) -> None:
        for seed in range(1, 6):
            library.add(tiny_flat(seed), report_for(seed))
        gone = library.prune(keep=2)
        assert len(gone) == 3
        assert len(library.entries()) == 2


# --- The session folder -----------------------------------------------------------------------


def a_session(library: fl.FlatLibrary, version: str, *, t_utc_ns: int = NOW) -> fl.FlatSession:
    folder = library.session.directory
    folder.mkdir(parents=True, exist_ok=True)
    library.session.ser_path(1).write_bytes(b"frames")
    return fl.FlatSession(
        t_utc_ns=t_utc_ns,
        version=version,
        ser="set1.ser",
        frames=32,
        exposure_s=0.04,
        temperature_c=12.3,
        level_fraction=0.5,
        mode="bin2",
        gain=120,
        width_px=256,
        height_px=176,
    )


class TestTheSessionFolder:
    def test_a_saved_session_loads_while_it_is_young(self, library: fl.FlatLibrary) -> None:
        session = a_session(library, "flat-1a2b3c4d")
        library.session.save(session)
        assert library.session.load(NOW + 3600 * NS_PER_S) == session
        assert session.expires_ns() == NOW + 24 * 3600 * NS_PER_S

    def test_a_session_expires_after_24_hours(self, library: fl.FlatLibrary) -> None:
        session = a_session(library, "flat-1a2b3c4d")
        library.session.save(session)
        assert library.session.load(NOW + 24 * 3600 * NS_PER_S) == session
        assert library.session.load(NOW + 24 * 3600 * NS_PER_S + 1) is None

    def test_loading_deletes_nothing_and_sweeping_deletes_what_expired(
        self, library: fl.FlatLibrary
    ) -> None:
        library.session.save(a_session(library, "flat-1a2b3c4d"))
        late = NOW + 25 * 3600 * NS_PER_S
        assert library.session.load(late) is None
        assert library.session.ser_path(1).exists()  # a read keeps the frames
        assert library.session.sweep(late) is True
        assert not library.session.directory.exists()

    def test_sweeping_a_live_session_keeps_it(self, library: fl.FlatLibrary) -> None:
        library.session.save(a_session(library, "flat-1a2b3c4d"))
        assert library.session.sweep(NOW + 60 * NS_PER_S) is False
        assert library.session.ser_path(1).exists()

    def test_sweeping_removes_a_second_set_that_a_crash_left(self, library: fl.FlatLibrary) -> None:
        library.session.save(a_session(library, "flat-1a2b3c4d"))
        library.session.ser_path(2).write_bytes(b"half a set")
        assert library.session.sweep(NOW + 60 * NS_PER_S) is True
        assert not library.session.ser_path(2).exists()
        assert library.session.load(NOW + 60 * NS_PER_S) is not None

    def test_a_session_without_its_frames_is_no_session(self, library: fl.FlatLibrary) -> None:
        library.session.save(a_session(library, "flat-1a2b3c4d"))
        library.session.ser_path(1).unlink()
        assert library.session.load(NOW) is None
        library.session.sweep(NOW)
        assert not library.session.directory.exists()

    def test_the_idle_sweep_deletes_what_expired_when_no_session_captures(
        self, library: fl.FlatLibrary
    ) -> None:
        library.session.save(a_session(library, "flat-1a2b3c4d"))
        assert library.session.sweep_idle(NOW + 60 * NS_PER_S) is False  # a young session stays
        assert library.session.sweep_idle(NOW + 25 * 3600 * NS_PER_S) is True
        assert not library.session.directory.exists()

    def test_the_idle_sweep_leaves_the_folder_to_a_session_that_captures(
        self, library: fl.FlatLibrary
    ) -> None:
        library.session.directory.mkdir(parents=True)
        library.session.ser_path(1).write_bytes(b"frames that arrive")  # no session file yet
        with library.session.capturing():
            assert library.session.sweep_idle(NOW) is False
            assert library.session.ser_path(1).exists()
        assert library.session.sweep_idle(NOW) is True  # nobody captures now, and nobody owns it
        assert not library.session.directory.exists()

    @pytest.mark.parametrize(
        "text",
        ["", "{", "[]", '{"version": "x"}', '{"t_utc_ns": "soon"}'],
    )
    def test_a_file_that_is_not_a_session_is_ignored(
        self, library: fl.FlatLibrary, text: str
    ) -> None:
        library.session.directory.mkdir(parents=True)
        (library.session.directory / "session.json").write_text(text, encoding="utf-8")
        assert library.session.load(NOW) is None

    def test_a_session_may_not_name_a_file_outside_its_folder(
        self, library: fl.FlatLibrary
    ) -> None:
        session = a_session(library, "flat-1a2b3c4d")
        data = session.to_json()
        data["ser"] = "../current.json"
        assert fl.FlatSession.from_json(data) is None
        data = session.to_json()
        data["version"] = "../../x"
        assert fl.FlatSession.from_json(data) is None

    def test_clear_removes_everything_and_may_run_on_a_missing_folder(
        self, library: fl.FlatLibrary
    ) -> None:
        library.session.clear()
        library.session.save(a_session(library, "flat-1a2b3c4d"))
        library.session.clear()
        assert not library.session.directory.exists()

    def test_only_the_sets_one_and_two_have_a_file(self, library: fl.FlatLibrary) -> None:
        assert library.session.ser_path(2).name == "set2.ser"
        with pytest.raises(ValueError, match="sets 1 and 2"):
            library.session.ser_path(3)

    def test_the_session_does_not_show_in_the_list_of_flats(self, library: fl.FlatLibrary) -> None:
        library.session.save(a_session(library, "flat-1a2b3c4d"))
        assert library.entries() == []


# --- The flat that the survey uses ----------------------------------------------------------------


def survey_config(tmp_path: Path, *, flat_file: str = "") -> SurveyConfig:
    return SurveyConfig(calibration_dir=str(tmp_path / "calibration"), flat_file=flat_file)


def write_npy(path: Path, flat: np.ndarray) -> Path:
    np.save(path, flat)
    return path


class TestTheRuleOfTheSurvey:
    def test_without_a_flat_the_survey_uses_a_unit_flat(self, tmp_path: Path) -> None:
        model = fl.ActiveFlat.from_config(survey_config(tmp_path)).current()
        assert isinstance(model, UnitFlat)
        assert model.version == "unit"

    def test_without_a_calibration_folder_there_is_no_library(self) -> None:
        assert isinstance(fl.ActiveFlat.from_config(SurveyConfig()).current(), UnitFlat)

    def test_flat_file_is_used_when_the_library_has_no_active_flat(self, tmp_path: Path) -> None:
        pinned = write_npy(tmp_path / "pinned.npy", tiny_flat(5))
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        library.add(tiny_flat(1), report_for(1))  # pending: the survey ignores it
        model = fl.ActiveFlat.from_config(survey_config(tmp_path, flat_file=str(pinned))).current()
        assert model.version == ArrayFlat(tiny_flat(5)).version

    def test_the_active_flat_of_the_library_wins_over_flat_file(self, tmp_path: Path) -> None:
        pinned = write_npy(tmp_path / "pinned.npy", tiny_flat(5))
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        entry = library.add(tiny_flat(1), report_for(1))
        library.activate(entry.version, now_utc_ns=NOW)
        config = survey_config(tmp_path, flat_file=str(pinned))
        assert fl.ActiveFlat.from_config(config).current().version == entry.version
        pins = fl.flat_pins(config)
        assert pins == fl.FlatPins(entry.version, True)
        assert pins.library_overrides is True

    def test_the_pins_say_that_the_library_does_not_override_without_an_active_flat(
        self, tmp_path: Path
    ) -> None:
        pinned = write_npy(tmp_path / "pinned.npy", tiny_flat(5))
        pins = fl.flat_pins(survey_config(tmp_path, flat_file=str(pinned)))
        assert (pins.active_version, pins.flat_file_pinned, pins.library_overrides) == (
            None,
            True,
            False,
        )
        assert fl.flat_pins(SurveyConfig()) == fl.FlatPins(None, False)

    def test_a_pointer_that_moves_loads_the_new_flat_and_nothing_else_does(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        first = library.add(tiny_flat(1), report_for(1))
        second = library.add(tiny_flat(2), report_for(2))
        library.activate(first.version, now_utc_ns=NOW)
        source = fl.ActiveFlat.from_config(survey_config(tmp_path))
        loads: list[str] = []
        original = fl.FlatLibrary.load

        def counting(self: fl.FlatLibrary, version: str) -> ArrayFlat:
            loads.append(version)
            return original(self, version)

        monkeypatch.setattr(fl.FlatLibrary, "load", counting)
        assert source.current().version == first.version
        assert source.current().version == first.version
        assert source.current().version == first.version
        assert loads == [first.version]  # one read, however many frames
        library.activate(second.version, now_utc_ns=NOW + NS_PER_S)  # which reads the flat once too
        loads.clear()
        assert source.current().version == second.version
        assert source.current().version == second.version
        assert loads == [second.version]  # one read for the change, and none for the next frame

    def test_a_check_costs_one_stat_of_the_pointer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        entry = library.add(tiny_flat(1), report_for(1))
        library.activate(entry.version, now_utc_ns=NOW)
        source = fl.ActiveFlat.from_config(survey_config(tmp_path))
        source.current()
        calls: list[str] = []
        real = os.stat

        def counting(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
            calls.append(os.fspath(path))
            return real(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", counting)
        for _ in range(5):
            source.current()
        assert len(calls) == 5
        assert all(call.endswith("current.json") for call in calls)

    def test_the_library_flat_is_the_active_flat_alone(self, tmp_path: Path) -> None:
        pinned = write_npy(tmp_path / "pinned.npy", tiny_flat(5))
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        entry = library.add(tiny_flat(1), report_for(1))
        source = fl.ActiveFlat.from_config(survey_config(tmp_path, flat_file=str(pinned)))
        assert source.library_flat() is None  # a pending flat is not in use
        assert source.current().version == ArrayFlat(tiny_flat(5)).version  # flat_file serves
        library.activate(entry.version, now_utc_ns=NOW)
        found = source.library_flat()
        assert found is not None
        assert found.version == entry.version
        assert source.current().version == entry.version
        (library.directory / "current.json").unlink()
        assert source.library_flat() is None
        assert source.current().version == ArrayFlat(tiny_flat(5)).version

    def test_the_library_flat_needs_no_flat_file_and_no_calibration_folder(self) -> None:
        assert fl.ActiveFlat.from_config(SurveyConfig()).library_flat() is None

    def test_a_library_flat_that_cannot_be_read_leaves_the_flat_in_place(
        self, tmp_path: Path
    ) -> None:
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        first = library.add(tiny_flat(1), report_for(1))
        second = library.add(tiny_flat(2), report_for(2))
        library.activate(first.version, now_utc_ns=NOW)
        source = fl.ActiveFlat.from_config(survey_config(tmp_path))
        assert source.library_flat() is not None
        library.activate(second.version, now_utc_ns=NOW + NS_PER_S)
        (library.directory / f"{second.version}.npy").write_bytes(b"broken")
        kept = source.library_flat()
        assert kept is not None
        assert kept.version == first.version  # the flat in use stays

    def test_a_library_flat_that_cannot_be_read_at_the_start_is_none(self, tmp_path: Path) -> None:
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        entry = library.add(tiny_flat(1), report_for(1))
        library.activate(entry.version, now_utc_ns=NOW)
        (library.directory / f"{entry.version}.npy").unlink()
        assert fl.ActiveFlat.from_config(survey_config(tmp_path)).library_flat() is None

    def test_the_function_for_the_library_flat_shares_the_cache_of_active_flat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        entry = library.add(tiny_flat(1), report_for(1))
        library.activate(entry.version, now_utc_ns=NOW)
        config = survey_config(tmp_path)
        loads: list[str] = []
        original = fl.FlatLibrary.load

        def counting(self: fl.FlatLibrary, version: str) -> ArrayFlat:
            loads.append(version)
            return original(self, version)

        monkeypatch.setattr(fl.FlatLibrary, "load", counting)
        found = fl.library_flat(config)
        assert found is not None
        assert found.version == entry.version
        assert fl.active_flat(config) is found  # the same model, and no second read
        assert fl.library_flat(config) is found
        assert loads == [entry.version]

    def test_a_check_of_the_library_flat_costs_one_stat_of_the_pointer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        entry = library.add(tiny_flat(1), report_for(1))
        library.activate(entry.version, now_utc_ns=NOW)
        config = survey_config(tmp_path)
        assert fl.library_flat(config) is not None
        calls: list[str] = []
        real = os.stat

        def counting(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
            calls.append(os.fspath(path))
            return real(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", counting)
        for _ in range(5):
            assert fl.library_flat(config) is not None
        assert len(calls) == 5
        assert all(call.endswith("current.json") for call in calls)

    def test_taking_the_pointer_away_returns_to_flat_file(self, tmp_path: Path) -> None:
        pinned = write_npy(tmp_path / "pinned.npy", tiny_flat(5))
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        entry = library.add(tiny_flat(1), report_for(1))
        library.activate(entry.version, now_utc_ns=NOW)
        source = fl.ActiveFlat.from_config(survey_config(tmp_path, flat_file=str(pinned)))
        assert source.current().version == entry.version
        (library.directory / "current.json").unlink()
        assert source.current().version == ArrayFlat(tiny_flat(5)).version

    def test_an_active_flat_that_cannot_be_read_leaves_the_flat_in_place(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        first = library.add(tiny_flat(1), report_for(1))
        second = library.add(tiny_flat(2), report_for(2))
        library.activate(first.version, now_utc_ns=NOW)
        source = fl.ActiveFlat.from_config(survey_config(tmp_path))
        assert source.current().version == first.version
        library.activate(second.version, now_utc_ns=NOW + NS_PER_S)
        (library.directory / f"{second.version}.npy").write_bytes(b"broken")
        with caplog.at_level("WARNING", logger="seeingmon.survey"):
            assert source.current().version == first.version
            assert source.current().version == first.version
        assert len([r for r in caplog.records if "cannot be used" in r.getMessage()]) == 1

    def test_a_broken_active_flat_at_the_start_falls_back_to_flat_file(
        self, tmp_path: Path
    ) -> None:
        pinned = write_npy(tmp_path / "pinned.npy", tiny_flat(5))
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        entry = library.add(tiny_flat(1), report_for(1))
        library.activate(entry.version, now_utc_ns=NOW)
        (library.directory / f"{entry.version}.npy").unlink()
        config = survey_config(tmp_path, flat_file=str(pinned))
        assert (
            fl.ActiveFlat.from_config(config).current().version == ArrayFlat(tiny_flat(5)).version
        )

    def test_a_flat_file_that_cannot_be_read_stops_the_start_as_it_always_did(
        self, tmp_path: Path
    ) -> None:
        config = survey_config(tmp_path, flat_file=str(tmp_path / "missing.npy"))
        with pytest.raises(SkyError):
            fl.ActiveFlat.from_config(config).current()

    def test_the_function_is_the_same_rule_and_it_follows_the_pointer(self, tmp_path: Path) -> None:
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        first = library.add(tiny_flat(1), report_for(1))
        second = library.add(tiny_flat(2), report_for(2))
        config = survey_config(tmp_path)
        assert fl.active_flat(config).version == "unit"
        library.activate(first.version, now_utc_ns=NOW)
        assert fl.active_flat(config).version == first.version
        library.activate(second.version, now_utc_ns=NOW + NS_PER_S)
        assert fl.active_flat(config).version == second.version
        assert fl.active_flat(config) is fl.active_flat(config)  # a cached model, not a new read


# --- The pipeline ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_profile() -> Profile:
    return synth.cropped_profile(1200, 800)


@pytest.fixture(scope="module")
def pipeline_catalog() -> CapCatalog:
    return synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=3.0, seed=1)


def vignette(shape: tuple[int, int], depth: float) -> np.ndarray:
    height, width = shape
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = np.hypot(x - (width - 1) / 2, y - (height - 1) / 2) / np.hypot(width / 2, height / 2)
    return np.asarray(1.0 - depth * radius**2, dtype=np.float32)


class TestThePipelineFollowsThePointer:
    def sky_flat_versions(
        self, profile: Profile, catalog: CapCatalog, tmp_path: Path, activate_between: Any
    ) -> list[str]:
        frame, _ = synth.render_frame(
            catalog,
            profile,
            rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
            exposure_s=30.0,
            seed=3,
        )
        pipeline = pipeline_with_the_rule(profile, catalog, survey_config(tmp_path))
        versions = []
        for index in range(3):
            if index:
                activate_between(index)
            analysis = pipeline.analyze(replace(frame, seq=frame.seq + index))
            record = next(r for r in analysis.records if isinstance(r, SkyQualityRecord))
            versions.append(record.provenance["flat"])
        return versions

    def test_the_provenance_of_the_next_frame_names_the_new_flat(
        self, pipeline_profile: Profile, pipeline_catalog: CapCatalog, tmp_path: Path
    ) -> None:
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        first = library.add(vignette((800, 1200), 0.2), report_for(1))
        second = library.add(vignette((800, 1200), 0.1), report_for(2))

        def activate(index: int) -> None:
            chosen = (first, second)[index - 1]
            library.activate(chosen.version, now_utc_ns=NOW + index * NS_PER_S)

        versions = self.sky_flat_versions(pipeline_profile, pipeline_catalog, tmp_path, activate)
        # frame 0: no flat; frame 1: the first flat; frame 2: the second, with no restart
        assert versions == ["unit", first.version, second.version]

    def test_a_pipeline_that_was_given_its_own_flat_ignores_the_library(
        self, pipeline_profile: Profile, pipeline_catalog: CapCatalog, tmp_path: Path
    ) -> None:
        library = fl.FlatLibrary(tmp_path / "calibration" / "flats")
        entry = library.add(vignette((800, 1200), 0.2), report_for(1))
        library.activate(entry.version, now_utc_ns=NOW)
        own = ArrayFlat(vignette((800, 1200), 0.05))
        frame, _ = synth.render_frame(
            pipeline_catalog,
            pipeline_profile,
            rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
            exposure_s=30.0,
            seed=3,
        )
        pipeline = SurveyPipeline(
            station_id="test",
            profile=pipeline_profile,
            catalog=pipeline_catalog,
            solvers=[],
            flat=own,
        )
        analysis = pipeline.analyze(frame)
        record = next(r for r in analysis.records if isinstance(r, SkyQualityRecord))
        assert record.provenance["flat"] == own.version


def pipeline_with_the_rule(
    profile: Profile, catalog: CapCatalog, config: SurveyConfig
) -> SurveyPipeline:
    """The pipeline that `build_pipeline` makes, on the catalog and the profile of the test."""
    active = fl.ActiveFlat.from_config(config)
    return SurveyPipeline(
        station_id="test",
        profile=profile,
        catalog=catalog,
        solvers=[],
        config=config,
        flat=active.current(),
        flat_source=active.current,
    )


def test_build_pipeline_wires_the_rule_of_the_survey_into_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`build_pipeline` makes the pipeline with the flat source, so that the worker follows."""
    from seeingmon.survey import pipeline as module

    seen: dict[str, Any] = {}

    class Spy(SurveyPipeline):
        def __init__(self, **options: Any) -> None:
            seen.update(options)
            super().__init__(**options)

    monkeypatch.setattr(module, "SurveyPipeline", Spy)
    catalog = synth.synthetic_catalog(cap_radius_deg=3.0, density_scale=1.0, seed=1)
    monkeypatch.setattr(module, "load_catalog", lambda path: catalog)
    spec = PipelineSpec(
        station_id="test",
        profile=synth.cropped_profile(1200, 800).model_dump(mode="python"),
        config=survey_config(tmp_path).model_dump(mode="python"),
        catalog_path="unused",
    )
    pipeline = build_pipeline(spec)
    assert callable(seen["flat_source"])
    assert seen["flat"].version == "unit"
    assert isinstance(pipeline, SurveyPipeline)
