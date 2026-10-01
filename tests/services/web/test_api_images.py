"""`/images`: the list, the latest image, one image as a JPEG, a FITS frame, or JSON."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI

from seeingmon.services.web.config import WebSettings
from seeingmon.store.layout import DataLayout
from tests.services.web.client import TestClient
from tests.services.web.helpers import tiny_jpeg
from tests.services.web.seed import write_fits, write_preview

API = "/api/v1"
STAMPS = [
    "20260929T210000.000Z",
    "20260930T020000.500Z",
    "20260930T230000.000Z",
    "20261001T020000.000Z",
]
IDS = [f"preview-{stamp}" for stamp in STAMPS]


@pytest.fixture
def pictures(layout: DataLayout) -> list[Path]:
    paths = [
        write_preview(layout, stamp, shade=40 * (index + 1)) for index, stamp in enumerate(STAMPS)
    ]
    write_fits(layout, STAMPS[1])
    return paths


def tree(root: Path) -> dict[str, tuple[int, int]]:
    """The names, sizes, and times of every file under a folder."""
    found = {}
    for path in sorted(root.rglob("*")):
        info = path.stat()
        found[str(path.relative_to(root))] = (info.st_size, info.st_mtime_ns)
    return found


# --- The list --------------------------------------------------------------------------------


def test_the_list_has_the_documented_shape_and_comes_newest_first(
    client: TestClient, pictures: list[Path]
) -> None:
    response = client.get(f"{API}/images")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"now", "items", "next_cursor"}
    assert [item["id"] for item in body["items"]] == IDS[::-1]
    assert body["next_cursor"] is None
    newest = body["items"][0]
    assert newest == {
        "id": IDS[3],
        "kind": "preview",
        "t_utc": "2026-10-01T02:00:00.000000Z",
        "t_utc_ns": 1_790_820_000_000_000_000,
        "size_bytes": len(tiny_jpeg(160)),
        "has_fits": False,
        "fits_bytes": None,
        "preview_url": f"/api/v1/images/{IDS[3]}",
        "fits_url": None,
    }


def test_an_image_with_a_fits_frame_says_so_and_links_to_it(
    client: TestClient, pictures: list[Path]
) -> None:
    item = next(i for i in client.get(f"{API}/images").json()["items"] if i["id"] == IDS[1])
    assert item["has_fits"] is True
    assert item["fits_bytes"] == 5760
    assert item["fits_url"] == f"/api/v1/images/{IDS[1]}?format=fits"


def test_the_list_pages_through_the_cursor(client: TestClient, pictures: list[Path]) -> None:
    seen: list[str] = []
    params: dict[str, Any] = {"limit": 3}
    for _ in range(5):
        body = client.get(f"{API}/images", params=params).json()
        seen.extend(item["id"] for item in body["items"])
        if body["next_cursor"] is None:
            break
        params = {"limit": 3, "cursor": body["next_cursor"]}
    assert seen == IDS[::-1]


def test_an_empty_folder_gives_an_empty_list(client: TestClient) -> None:
    assert client.get(f"{API}/images").json()["items"] == []


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 1001},
        {"limit": 201},  # the maximum of this server
        {"limit": "x"},
        {"cursor": "!"},
        {"cursor": "e30"},
        {"cursor": "eyJrIjoiLi4vLi4veCJ9"},  # {"k": "../../x"}
    ],
)
def test_a_bad_list_parameter_is_a_422(client: TestClient, params: dict[str, Any]) -> None:
    response = client.get(f"{API}/images", params=params)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_the_default_page_size_is_24(client: TestClient, layout: DataLayout) -> None:
    for minute in range(30):
        write_preview(layout, f"20261001T02{minute:02d}00.000Z")
    body = client.get(f"{API}/images").json()
    assert len(body["items"]) == 24
    assert body["next_cursor"] is not None


# --- One image -------------------------------------------------------------------------------


def test_an_image_is_served_as_a_jpeg_that_never_changes(
    client: TestClient, pictures: list[Path]
) -> None:
    response = client.get(f"{API}/images/{IDS[3]}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == tiny_jpeg(160)
    assert response.headers["cache-control"] == "public, max-age=86400, immutable"


def test_the_latest_image_is_the_newest_and_the_browser_must_ask_again(
    client: TestClient, pictures: list[Path]
) -> None:
    response = client.get(f"{API}/images/latest")
    assert response.status_code == 200
    assert response.content == tiny_jpeg(160)
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["content-type"] == "image/jpeg"


def test_the_format_json_describes_the_image(client: TestClient, pictures: list[Path]) -> None:
    listed = client.get(f"{API}/images").json()["items"][0]
    assert client.get(f"{API}/images/{IDS[3]}", params={"format": "json"}).json() == listed
    assert client.get(f"{API}/images/latest", params={"format": "json"}).json() == listed


def test_the_fits_frame_downloads_as_a_file(client: TestClient, pictures: list[Path]) -> None:
    response = client.get(f"{API}/images/{IDS[1]}", params={"format": "fits"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/fits"
    assert response.headers["content-disposition"] == f'attachment; filename="{STAMPS[1]}.fits"'
    assert response.content.startswith(b"SIMPLE  =")
    assert len(response.content) == 5760


def test_an_image_without_a_fits_frame_has_no_fits(
    client: TestClient, pictures: list[Path]
) -> None:
    response = client.get(f"{API}/images/{IDS[0]}", params={"format": "fits"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_fits"


def test_the_latest_image_can_come_as_fits_when_it_has_one(
    client: TestClient, layout: DataLayout, pictures: list[Path]
) -> None:
    write_fits(layout, STAMPS[3])
    response = client.get(f"{API}/images/latest", params={"format": "fits"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"


def test_an_image_that_does_not_exist_is_a_404(client: TestClient, pictures: list[Path]) -> None:
    response = client.get(f"{API}/images/preview-20200101T000000.000Z")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_the_latest_image_of_an_empty_folder_is_a_404(client: TestClient) -> None:
    response = client.get(f"{API}/images/latest")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_data"


@pytest.mark.parametrize("image_format", ["png", "JPEG", "", "fits;x"])
def test_an_unknown_format_is_a_422(
    client: TestClient, pictures: list[Path], image_format: str
) -> None:
    response = client.get(f"{API}/images/{IDS[3]}", params={"format": image_format})
    assert response.status_code == 422


SECRET = "a-secret-that-must-not-leave"


@pytest.mark.parametrize(
    "bad_id",
    [
        "..%2f..%2f..%2fetc%2fpasswd",
        "%2e%2e%2f%2e%2e%2fsecret.txt",
        "..%5c..%5csecret.txt",
        "....//....//secret.txt",
        "preview-20260930T020000.500Z%2f..%2f..%2fsecret.txt",
        "preview-20260930T020000.500Z/../../secret.txt",
        "preview-20260930T020000.500Z%00.jpg",
        "preview-20260930T020000.500Z.jpg",
        "%2fetc%2fpasswd",
        "C:%5cWindows%5cwin.ini",
        "preview-20260930T020000.500Z%5c..%5c..%5csecret.txt",
        "preview-" + "9" * 500,
        "preview-20260930T020000.500Z" + "a" * 100,
        "preview-\u2024\u2024T020000.500Z",
    ],
)
def test_no_path_from_a_client_can_reach_a_file(
    client: TestClient, layout: DataLayout, pictures: list[Path], tmp_path: Path, bad_id: str
) -> None:
    (tmp_path / "secret.txt").write_text(SECRET, encoding="utf-8")
    (layout.root / "secret.txt").write_text(SECRET, encoding="utf-8")
    for image_format in ("jpeg", "fits", "json"):
        response = client.get(f"{API}/images/{bad_id}", params={"format": image_format})
        assert response.status_code in {404, 422}, response.text
        assert SECRET not in response.text
        assert b"root:" not in response.content


def test_requests_for_images_never_change_the_data_directory(
    client: TestClient, layout: DataLayout, pictures: list[Path]
) -> None:
    before = tree(layout.root / "previews"), tree(layout.root / "survey")
    for path in (
        f"{API}/images",
        f"{API}/images/latest",
        f"{API}/images/{IDS[1]}",
        f"{API}/images/{IDS[1]}?format=fits",
        f"{API}/images/{IDS[0]}?format=json",
        f"{API}/images/preview-20200101T000000.000Z",
    ):
        client.get(path)
    assert (tree(layout.root / "previews"), tree(layout.root / "survey")) == before


def test_the_web_process_has_no_route_that_changes_an_image(
    client: TestClient, pictures: list[Path]
) -> None:
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(f"{API}/images/{IDS[3]}")
        assert response.status_code in {401, 403, 405}
    assert pictures[3].exists()


def test_a_preview_above_the_size_limit_is_not_served(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    layout: DataLayout,
    pictures: list[Path],
) -> None:
    small = WebSettings.model_validate({"images": {"max_preview_bytes": 1024}})
    path = pictures[3]
    path.write_bytes(path.read_bytes().ljust(2000, b"\0"))
    client = open_client(make_app(settings=small))
    assert client.get(f"{API}/images/{IDS[3]}").status_code == 404
    assert [i["id"] for i in client.get(f"{API}/images").json()["items"]] == IDS[2::-1]


def test_a_fits_frame_above_the_size_limit_is_not_served(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    layout: DataLayout,
    pictures: list[Path],
) -> None:
    small = WebSettings.model_validate({"images": {"max_fits_bytes": 1024}})
    client = open_client(make_app(settings=small))
    item = client.get(f"{API}/images/{IDS[1]}", params={"format": "json"}).json()
    assert item["has_fits"] is False
    assert client.get(f"{API}/images/{IDS[1]}", params={"format": "fits"}).status_code == 404


def test_a_jpeg_is_not_compressed_even_when_the_client_accepts_gzip(
    client: TestClient, pictures: list[Path]
) -> None:
    response = client.get(f"{API}/images/{IDS[3]}", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in response.headers
    assert response.content == tiny_jpeg(160)
