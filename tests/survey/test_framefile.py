"""The FITS file of a survey frame: pixels, header, Rice compression, and the fallback."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from seeingmon.frames import Frame, FrameFlag  # noqa: E402
from seeingmon.profile import Profile, load_profile  # noqa: E402
from seeingmon.solvers import fitsio  # noqa: E402
from seeingmon.survey import framefile  # noqa: E402
from tests.scheduler.helpers import make_frame  # noqa: E402

T_MID_NS = 1_790_000_000_123_456_789  # 2026-09-21T14:13:20.123456789Z


@pytest.fixture(scope="module")
def profile() -> Profile:
    return load_profile("asi294mm-gs250")


RICE_AVAILABLE = framefile.rice_available  # the cached function, before a test patches the name


@pytest.fixture(autouse=True)
def _fresh_probe() -> Iterator[None]:
    """`rice_available` caches its answer, and a test that changes it must not leak."""
    RICE_AVAILABLE.cache_clear()
    yield
    RICE_AVAILABLE.cache_clear()


def sky(shift: int = 2, shape: tuple[int, int] = (48, 64), seed: int = 5) -> np.ndarray:
    """A frame of 14-bit counts in the high bits of a 16-bit container, as the SDK delivers it."""
    rng = np.random.default_rng(seed)
    native = rng.normal(900.0, 12.0, shape).clip(0, 2 ** (16 - shift) - 1).astype(np.uint16)
    native[20, 30] = 16000
    return (native << shift).astype(np.uint16)


def frame_of(data: np.ndarray, **options: object) -> Frame:
    values: dict[str, object] = {
        "mode": "bin2",
        "gain": 120,
        "exposure_us": 30_000_000,
        "adc_bits": 14,
        "t_utc_ns": T_MID_NS,
        "seq": 7,
    }
    values.update(options)
    return make_frame(data, **values)  # type: ignore[arg-type]


def write(path: Path, frame: Frame, profile: Profile, *, compress: bool, **cards: object) -> bool:
    with path.open("wb") as handle:
        return framefile.write_frame_fits(
            handle,
            framefile.native_pixels(frame),
            framefile.frame_cards(frame, profile=profile, station_id="st", **cards),  # type: ignore[arg-type]
            framefile.frame_comments(frame),
            compress=compress,
        )


class TestThePixels:
    def test_a_16_bit_frame_is_shifted_down_to_native_counts(self) -> None:
        data = sky(shift=2)
        frame = frame_of(data, adc_bits=14)
        pixels = framefile.native_pixels(frame)
        assert pixels.dtype == np.uint16
        np.testing.assert_array_equal(pixels, data >> 2)
        assert int(pixels.max()) <= 2**14 - 1

    def test_the_shift_leaves_the_frame_alone(self) -> None:
        data = sky()
        data.flags.writeable = False  # a decoded frame is read-only
        before = data.copy()
        pixels = framefile.native_pixels(frame_of(data))
        np.testing.assert_array_equal(data, before)
        assert not np.shares_memory(pixels, data)

    def test_a_frame_that_fills_the_container_comes_back_without_a_copy(self) -> None:
        data = sky(shift=0)
        frame = frame_of(data, adc_bits=16)
        assert framefile.native_pixels(frame) is frame.data

    def test_an_8_bit_frame_stays_as_it_is(self) -> None:
        data = (sky(shift=0) >> 6).astype(np.uint8)
        frame = frame_of(data, adc_bits=14)
        assert framefile.native_pixels(frame) is frame.data


class TestTheHeader:
    def cards(
        self, profile: Profile, *, temperature_c: float | None = 19.5, reasons: tuple[str, ...] = ()
    ) -> dict[str, tuple[object, str]]:
        frame = dataclasses.replace(frame_of(sky()), temperature_c=temperature_c)
        found = framefile.frame_cards(frame, profile=profile, station_id="st", reasons=reasons)
        return {keyword: (value, comment) for keyword, value, comment in found}

    def test_the_time_the_exposure_and_the_settings_are_there(self, profile: Profile) -> None:
        cards = self.cards(profile)
        assert cards["DATE-AVG"][0] == "2026-09-21T14:13:20.123"  # the frame time, mid-exposure
        assert cards["DATE-OBS"][0] == "2026-09-21T14:13:05.123"  # 30 s exposure: 15 s earlier
        assert cards["TIMESYS"][0] == "UTC"
        assert cards["TIMEQUAL"][0] == "EXACT"
        assert cards["EXPTIME"][0] == 30.0
        assert cards["GAIN"][0] == 120
        assert cards["READMODE"][0] == "bin2"
        assert cards["ADCBITS"][0] == 14
        assert (cards["XORGSUBF"][0], cards["YORGSUBF"][0]) == (0, 0)
        assert cards["CCD-TEMP"][0] == 19.5
        assert cards["FRAMESEQ"][0] == 7
        assert cards["FRMFLAGS"][0] == int(FrameFlag.SIMULATED)

    def test_the_plate_scale_and_the_optics_come_from_the_profile(self, profile: Profile) -> None:
        cards = self.cards(profile)
        assert cards["PIXSCALE"][0] == pytest.approx(3.82, abs=0.01)  # bin2 of the reference
        assert cards["XBINNING"][0] == 2
        assert cards["XPIXSZ"][0] == pytest.approx(4.63)
        assert cards["FOCALLEN"][0] == 250.0
        assert cards["INSTRUME"][0] == "ZWO ASI294MM"
        assert cards["PROFILE"][0] == "asi294mm-gs250"
        assert cards["STATION"][0] == "st"
        assert cards["EGAIN"][0] == pytest.approx(profile.e_per_adu("bin2", 120))

    def test_a_missing_temperature_leaves_the_card_out(self, profile: Profile) -> None:
        assert "CCD-TEMP" not in self.cards(profile, temperature_c=None)

    def test_the_reasons_say_why_the_frame_was_kept(self, profile: Profile) -> None:
        cards = self.cards(profile, reasons=("every_10", "event:cloud"))
        assert cards["KEPT"][0] == "every_10,event:cloud"
        assert "KEPT" not in self.cards(profile)

    def test_the_header_holds_no_serial_number_and_no_site(self, profile: Profile) -> None:
        cards = self.cards(profile, reasons=("event:cloud",))
        assert set(cards) == {
            "DATE-OBS",
            "DATE-AVG",
            "TIMESYS",
            "TIMEQUAL",
            "TIMEERR",
            "IMAGETYP",
            "EXPTIME",
            "GAIN",
            "READMODE",
            "ADCBITS",
            "BUNIT",
            "XORGSUBF",
            "YORGSUBF",
            "FRAMESEQ",
            "STREAMID",
            "FRMFLAGS",
            "CCD-TEMP",
            "INSTRUME",
            "STATION",
            "PROFILE",
            "FOCALLEN",
            "APTDIA",
            "XBINNING",
            "YBINNING",
            "XPIXSZ",
            "YPIXSZ",
            "PIXSCALE",
            "EGAIN",
            "RDNOISE",
            "KEPT",
            "CREATOR",
        }
        text = " ".join(f"{value} {comment}" for value, comment in cards.values()).lower()
        for word in ("serial", "latitude", "longitude", "site", "host", "address"):
            assert word not in text

    def test_a_readout_mode_that_the_profile_lacks_leaves_the_derived_cards_out(
        self, profile: Profile
    ) -> None:
        frame = frame_of(sky(), mode="odd")
        keywords = {k for k, _, _ in framefile.frame_cards(frame, profile=profile)}
        assert {"INSTRUME", "FOCALLEN", "EXPTIME"} <= keywords
        assert not keywords & {"PIXSCALE", "XBINNING", "EGAIN"}

    def test_a_frame_without_a_profile_still_has_its_own_settings(self) -> None:
        keywords = {k for k, _, _ in framefile.frame_cards(frame_of(sky()), profile=None)}
        assert {"DATE-AVG", "EXPTIME", "GAIN", "READMODE"} <= keywords
        assert not keywords & {"INSTRUME", "PIXSCALE", "STATION"}


class TestTheFile:
    def test_a_compressed_file_reads_back_to_the_same_pixels_and_cards(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        frame = frame_of(sky(shape=(120, 160)))
        path = tmp_path / "frame.fits"
        assert write(path, frame, profile, compress=True, reasons=("every_10",)) is True
        back = framefile.read_frame_fits(path)
        assert back.compressed is True
        np.testing.assert_array_equal(back.pixels, framefile.native_pixels(frame))
        assert back.pixels.dtype == np.uint16
        assert back.header["EXPTIME"] == 30.0
        assert back.header["KEPT"] == "every_10"
        assert back.header["DATE-AVG"] == "2026-09-21T14:13:20.123"
        with fits.open(path) as hdus:  # astropy sees a primary HDU and a compressed image
            assert [type(h).__name__ for h in hdus] == ["PrimaryHDU", "CompImageHDU"]
            assert hdus[0].header["GAIN"] == 120  # the cards are in the primary header too
            assert hdus[1].compression_type == "RICE_1"
            assert hdus[1].header["BZERO"] == 32768
            comments = [str(c) for c in hdus[0].header["COMMENT"]]
            assert any("native ADC counts" in c for c in comments)

    def test_a_compressed_file_is_smaller_than_the_frame(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        frame = frame_of(sky(shape=(400, 600)))
        compressed, whole = tmp_path / "rice.fits", tmp_path / "whole.fits"
        write(compressed, frame, profile, compress=True)
        write(whole, frame, profile, compress=False)
        assert compressed.stat().st_size < 0.6 * whole.stat().st_size

    def test_an_uncompressed_file_reads_back_too(self, tmp_path: Path, profile: Profile) -> None:
        frame = frame_of(sky(shape=(60, 80)))
        path = tmp_path / "whole.fits"
        assert write(path, frame, profile, compress=False) is False
        back = framefile.read_frame_fits(path)
        assert back.compressed is False
        np.testing.assert_array_equal(back.pixels, framefile.native_pixels(frame))
        assert back.header["GAIN"] == 120
        header, _ = fitsio.read_image(path)
        assert header["BZERO"] == 32768
        with fits.open(path) as hdus:  # astropy reads it as a plain image
            np.testing.assert_array_equal(hdus[0].data, framefile.native_pixels(frame))

    def test_a_read_only_frame_is_written(self, tmp_path: Path, profile: Profile) -> None:
        data = sky(shift=0)
        data.flags.writeable = False
        frame = frame_of(data, adc_bits=16)
        path = tmp_path / "frame.fits"
        write(path, frame, profile, compress=True)
        np.testing.assert_array_equal(framefile.read_frame_fits(path).pixels, data)

    def test_an_8_bit_frame_round_trips_with_its_comment(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        data = (sky(shift=0) >> 6).astype(np.uint8)
        frame = frame_of(data)
        path = tmp_path / "frame.fits"
        write(path, frame, profile, compress=True)
        back = framefile.read_frame_fits(path)
        assert back.pixels.dtype == np.uint8
        np.testing.assert_array_equal(back.pixels, data)
        with fits.open(path) as hdus:
            assert any("top 8 bits" in str(c) for c in hdus[0].header["COMMENT"])

    def test_no_header_card_is_cut_short(self, tmp_path: Path, profile: Profile) -> None:
        """astropy warns about a card that does not fit, and the tests turn warnings into errors."""
        frame = dataclasses.replace(frame_of(sky()), temperature_c=-12.345)
        path = tmp_path / "frame.fits"
        write(path, frame, profile, compress=True, reasons=("every_10", "event:bright_sky+cloud"))
        with fits.open(path) as hdus:
            for card in hdus[0].header.cards:
                assert len(str(card)) == 80
                assert "..." not in str(card.comment)


class TestTheFallback:
    def test_without_the_compressor_the_file_is_whole_and_the_answer_says_so(
        self, tmp_path: Path, profile: Profile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(framefile, "rice_available", lambda: False)
        frame = frame_of(sky())
        path = tmp_path / "frame.fits"
        assert write(path, frame, profile, compress=True) is False
        back = framefile.read_frame_fits(path)
        assert back.compressed is False
        np.testing.assert_array_equal(back.pixels, framefile.native_pixels(frame))

    def test_a_missing_compressor_is_found_and_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def no_compressor(*args: object, **kwargs: object) -> None:
            raise ImportError("the compressed image extension is not built")

        monkeypatch.setattr(fits, "CompImageHDU", no_compressor)
        with caplog.at_level(logging.WARNING, logger="seeingmon.survey"):
            assert RICE_AVAILABLE() is False
        assert "cannot write Rice-compressed FITS" in caplog.text

    def test_the_probe_passes_here(self) -> None:
        assert framefile.rice_available() is True


class TestTheSkyCards:
    def cards(
        self,
        profile: Profile,
        *,
        cloud_fraction: float | None = None,
        transparency: float | None = None,
    ) -> dict[str, tuple[object, str]]:
        found = framefile.frame_cards(
            frame_of(sky()),
            profile=profile,
            station_id="st",
            cloud_fraction=cloud_fraction,
            transparency=transparency,
        )
        return {keyword: (value, comment) for keyword, value, comment in found}

    def test_the_cloud_fraction_and_the_transparency_have_a_card_each(
        self, profile: Profile
    ) -> None:
        cards = self.cards(profile, cloud_fraction=0.02345678, transparency=0.987654321)
        assert cards["CLOUDFRC"][0] == 0.0235
        assert cards["TRANSP"][0] == 0.9877
        assert "stars" in str(cards["CLOUDFRC"][1])

    def test_a_value_that_the_analysis_could_not_give_leaves_its_card_out(
        self, profile: Profile
    ) -> None:
        only_cloud = self.cards(profile, cloud_fraction=0.0)
        assert only_cloud["CLOUDFRC"][0] == 0.0
        assert "TRANSP" not in only_cloud
        neither = self.cards(profile)
        assert not {"CLOUDFRC", "TRANSP"} & set(neither)

    def test_the_cards_say_nothing_about_the_sun_the_moon_or_the_site(
        self, profile: Profile
    ) -> None:
        cards = self.cards(profile, cloud_fraction=0.1, transparency=0.9)
        text = " ".join(f"{key} {value} {comment}" for key, (value, comment) in cards.items())
        for word in ("sun", "moon", "twilight", "latitude", "longitude", "elevation"):
            assert word not in text.lower()

    @pytest.mark.parametrize("compress", [True, False])
    def test_the_values_survive_both_kinds_of_file(
        self, tmp_path: Path, profile: Profile, compress: bool
    ) -> None:
        path = tmp_path / "frame.fits"
        write(
            path,
            frame_of(sky()),
            profile,
            compress=compress,
            reasons=("every_10",),
            cloud_fraction=0.05,
            transparency=0.97,
        )
        header = framefile.read_frame_header(path)
        assert header["CLOUDFRC"] == 0.05
        assert header["TRANSP"] == 0.97
        meta = framefile.parse_frame_header(header)
        assert meta.cloud_fraction == 0.05
        assert meta.transparency == 0.97
        assert framefile.read_frame_fits(path).header["TRANSP"] == 0.97


class TestTheMeta:
    def header(self, profile: Profile, **values: object) -> dict[str, object]:
        frame = dataclasses.replace(frame_of(sky()), temperature_c=27.25)
        cards = framefile.frame_cards(frame, profile=profile, station_id="st", **values)  # type: ignore[arg-type]
        return {keyword: value for keyword, value, _ in cards}

    def test_the_meta_reads_the_cards_that_the_writer_makes(self, profile: Profile) -> None:
        meta = framefile.parse_frame_header(
            self.header(
                profile,
                reasons=("every_10", "event:cloud+bright_sky"),
                cloud_fraction=0.3,
                transparency=0.8,
            )
        )
        assert meta.t_utc_ns == T_MID_NS - T_MID_NS % 1_000_000  # the card keeps milliseconds
        assert meta.exposure_s == 30.0
        assert meta.gain == 120
        assert meta.mode == "bin2"
        assert meta.temperature_c == 27.25
        assert meta.adc_bits == 14
        assert meta.time_quality == "EXACT"
        assert meta.kept == ("every_10", "event:cloud+bright_sky")
        assert meta.events == ("cloud", "bright_sky")
        assert (meta.cloud_fraction, meta.transparency) == (0.3, 0.8)
        assert meta.origin == (0, 0)
        assert meta.flags == int(FrameFlag.SIMULATED)
        assert not meta.time_invalid

    def test_an_old_file_without_the_new_cards_reads_with_none_in_their_place(
        self, profile: Profile
    ) -> None:
        meta = framefile.parse_frame_header(self.header(profile, reasons=("every_10",)))
        assert meta.cloud_fraction is None
        assert meta.transparency is None
        assert meta.events == ()
        assert meta.exposure_s == 30.0  # the rest of the header still reads

    def test_an_empty_header_and_malformed_cards_give_a_meta_and_no_error(self) -> None:
        empty = framefile.parse_frame_header({})
        assert empty == framefile.FrameMeta()
        bad = framefile.parse_frame_header(
            {
                "DATE-AVG": "not a time",
                "EXPTIME": "thirty",
                "GAIN": True,
                "CLOUDFRC": float("nan"),
                "TRANSP": None,
                "KEPT": 7,
                "FRMFLAGS": "x",
            }
        )
        assert bad == framefile.FrameMeta()

    def test_a_time_without_a_zone_and_a_time_with_one_both_read(self) -> None:
        plain = framefile.parse_frame_header({"DATE-AVG": "2026-09-21T14:13:20.123"})
        zoned = framefile.parse_frame_header({"DATE-AVG": "2026-09-21T14:13:20.123Z"})
        assert plain.t_utc_ns == zoned.t_utc_ns == 1_790_000_000_123_000_000

    def test_a_clock_that_was_not_synchronized_marks_the_time_invalid(self) -> None:
        assert framefile.parse_frame_header({"FRMFLAGS": 1}).time_invalid
        assert framefile.parse_frame_header({"TIMEQUAL": "INVALID"}).time_invalid
        assert not framefile.parse_frame_header({"TIMEQUAL": "FITTED", "FRMFLAGS": 8}).time_invalid

    def test_the_header_of_a_file_reads_without_its_pixels(
        self, tmp_path: Path, profile: Profile
    ) -> None:
        path = tmp_path / "frame.fits"
        write(path, frame_of(sky(shape=(120, 160))), profile, compress=True)
        header = framefile.read_frame_header(path)
        assert header["READMODE"] == "bin2"
        assert "image" not in header
