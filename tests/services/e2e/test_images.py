"""The images of a simulated night: `core` writes them, and the web API lists and serves them.

`build_night` runs `core` on the simulated sky in virtual time, with the production survey pipeline.
After two survey steps, the data directory holds what `core` wrote, and the test opens a web
application on the same directory, with a read-only store, as the `web` process does. It then
follows the images from the records (`image_ref`) to the files and through the REST API.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("scipy", reason="the fast path needs the fast extra")
pytest.importorskip("fastapi", reason="web needs the web extra")
pytest.importorskip("PIL", reason="the previews need Pillow")
pytest.importorskip("astropy", reason="the FITS files need astropy")

from PIL import Image

from seeingmon.records import SurveyFrameRecord
from seeingmon.services.web.app import create_app
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.core_client import FakeCoreClient
from seeingmon.services.web.images import ImageStore
from seeingmon.store.db import StoreReader
from seeingmon.store.layout import DataLayout
from seeingmon.survey import framefile
from tests.services.web.client import TestClient

from .night import Night, build_night

API = "/api/v1"
# A frame of at least `[survey.sky] min_exposure_s` (1 s) is long: it carries the sky quality and a
# preview. The first long frame after a start takes `[survey.twilight] min_exposure_s` (1 s), and
# the long exposure grows from there by 4 times a step in the dark sky of the night.
LONG_MIN_EXPOSURE_S = 1.0
FIRST_LONG_EXPOSURE_S = 1.0


def long_frames(night: Night) -> list[SurveyFrameRecord]:
    records: list[SurveyFrameRecord] = [
        r for r in night.records("survey_frame") if r.exposure_s >= LONG_MIN_EXPOSURE_S
    ]
    return records


def jpeg_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "JPEG"
        return image.size


class TestAShortNight:
    def test_core_writes_the_images_and_the_web_api_serves_them(self, tmp_path: Path) -> None:
        night = build_night(tmp_path, window_s=3.0)
        try:
            night.run_until(lambda: len(night.records("sky_quality")) >= 2)
            night.app.tick()  # the writer has no thread in a stepped run: this call drains it
            layout: DataLayout = night.app.storage.layout  # type: ignore[union-attr]
            frames = night.records("survey_frame")
            longs = long_frames(night)
            assert len(longs) >= 2
            assert len(frames) > len(longs)  # the short frame of each step is there too

            # The records point at the files that exist.
            first, second = longs[0], longs[1]
            assert first.image_ref is not None
            assert second.image_ref is not None
            assert first.image_ref == layout.relative(layout.survey_path(first.t_utc_ns))
            assert layout.resolve(first.image_ref).is_file()  # the first long frame: a FITS file
            assert second.image_ref.startswith("previews/")  # the next ones: a preview only
            assert layout.resolve(second.image_ref).is_file()
            assert all(r.image_ref is None for r in frames if r.exposure_s < LONG_MIN_EXPOSURE_S)
            assert not list(layout.root.rglob("*.tmp"))

            settings = WebSettings()
            images = ImageStore(layout, settings.images)
            with (
                StoreReader.open(layout.db_path) as reader,
                TestClient(
                    create_app(settings, reader, images, FakeCoreClient(), clock=night.clock),
                    headers={"host": "localhost"},
                ) as client,
            ):
                self.check_the_api(client, images, layout, longs)
        finally:
            night.app.stop()

    def check_the_api(
        self,
        client: TestClient,
        images: ImageStore,
        layout: DataLayout,
        longs: list[SurveyFrameRecord],
    ) -> None:
        listing = client.get(f"{API}/images").json()
        items: list[dict[str, Any]] = listing["items"]
        assert [i["t_utc_ns"] for i in items] == sorted(
            (i["t_utc_ns"] for i in items), reverse=True
        )
        assert {i["t_utc_ns"] // 1_000_000 for i in items} == {
            r.t_utc_ns // 1_000_000 for r in longs
        }
        assert all(i["kind"] == "survey" for i in items)
        newest, oldest = items[0], items[-1]
        assert oldest["has_fits"] is True
        assert newest["has_fits"] is False

        # The newest image is the latest one, as a JPEG of at most a megapixel.
        latest = client.get(f"{API}/images/latest")
        assert latest.status_code == 200
        assert latest.headers["content-type"] == "image/jpeg"
        assert latest.headers["cache-control"] == "no-cache"
        width, height = jpeg_size(latest.content)
        assert 1000 < width * height <= 1_000_000
        assert latest.content == client.get(f"{API}/images/{newest['id']}").content
        assert client.get(f"{API}/images/latest", params={"format": "json"}).json() == newest

        # The JSON description, and the FITS file of the frame that `core` kept.
        assert (
            client.get(f"{API}/images/{oldest['id']}", params={"format": "json"}).json() == oldest
        )
        fits = client.get(f"{API}/images/{oldest['id']}", params={"format": "fits"})
        assert fits.status_code == 200
        assert fits.headers["content-type"] == "application/fits"
        assert len(fits.content) == oldest["fits_bytes"]
        path = layout.survey_path(oldest["t_utc_ns"])
        assert path.read_bytes() == fits.content
        back = framefile.read_frame_fits(path)
        assert back.compressed is True
        assert back.header["EXPTIME"] == FIRST_LONG_EXPOSURE_S
        assert back.header["KEPT"] == "every_10"
        missing = client.get(f"{API}/images/{newest['id']}", params={"format": "fits"})
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "no_fits"

        # From a record to its preview, through its `image_ref`.
        listed = {i["id"] for i in items}
        for record in longs:
            assert record.image_ref is not None
            info = images.for_ref(record.image_ref)
            assert info is not None
            assert info.id in listed
            assert info.has_fits is record.image_ref.startswith("survey/")
