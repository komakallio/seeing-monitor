"""The static UI: the pages, the assets, the headers, and the rules that keep it self-contained.

The UI is plain HTML, CSS, and JavaScript with no build step. These tests check what a Python test
can check: every page and asset is served, every reference inside the files resolves, no file
loads anything from another host, no page needs an inline script or style (the security headers
forbid both), and the ids that a script looks up exist in its page. The behavior of the scripts
is checked in a browser (see `docs/architecture.md`), and CI also asks Node to parse each script
when Node is installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from seeingmon.services.web.app import STATIC_DIR
from tests.services.web.client import TestClient

PAGES = {
    "index.html": "now",
    "history.html": "history",
    "images.html": "images",
    "align.html": "align",
    "api.html": "api",
}
NAV_PAGES = ("now", "history", "images", "align")
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
MAX_FILE_BYTES = 120_000
MAX_TOTAL_BYTES = 450_000

# The ids that each script looks up with a computed name, such as $(prefix + "-value").
DYNAMIC_IDS = {
    "now.js": [
        f"{prefix}-{suffix}"
        for prefix in ("seeing", "sky", "pointing")
        for suffix in ("value", "note", "when", "facts", "chips", "plot")
    ],
    "history.js": [
        *(f"plot-{key}" for key in ("seeing", "r0", "sky", "cloud", "pointing")),
        *(f"{key}-count" for key in ("seeing", "r0", "sky", "cloud", "pointing")),
        "legend-seeing",
        "legend-cloud",
    ],
}


def static_files() -> list[Path]:
    return sorted(path for path in STATIC_DIR.rglob("*") if path.is_file())


def text_of(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def scripts() -> list[Path]:
    return sorted((STATIC_DIR / "js").glob("*.js"))


# --- Serving ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", [*PAGES, "favicon.svg", "css/app.css", "js/common.js"])
def test_every_file_is_served_with_its_type(client: TestClient, name: str) -> None:
    response = client.get(f"/{name}")
    assert response.status_code == 200
    # The app sets these types itself: the registry of the operating system must not decide them
    # (`.js` is `application/javascript` on some Windows machines and on Python 3.11).
    expected = {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".svg": "image/svg+xml",
    }[Path(name).suffix]
    assert response.headers["content-type"] == expected


def test_the_root_serves_the_now_page(client: TestClient) -> None:
    root = client.get("/")
    assert root.status_code == 200
    assert root.text == client.get("/index.html").text
    assert "<title>Seeing monitor: Now</title>" in root.text


def test_every_file_of_the_folder_is_served_byte_for_byte(client: TestClient) -> None:
    for path in static_files():
        relative = path.relative_to(STATIC_DIR).as_posix()
        response = client.get(f"/{relative}", headers={"accept-encoding": "identity"})
        assert response.status_code == 200, relative
        assert response.content == path.read_bytes(), relative


def test_static_files_must_be_revalidated_and_answer_a_conditional_request_with_304(
    client: TestClient,
) -> None:
    first = client.get("/css/app.css")
    assert first.headers["cache-control"] == "no-cache"
    etag = first.headers["etag"]
    assert etag
    second = client.get("/css/app.css", headers={"if-none-match": etag})
    assert second.status_code == 304
    assert second.content == b""
    assert second.headers["cache-control"] == "no-cache"


def test_text_files_are_compressed_when_the_client_accepts_it(client: TestClient) -> None:
    response = client.get("/js/common.js", headers={"accept-encoding": "gzip"})
    assert response.headers["content-encoding"] == "gzip"
    assert response.text == text_of("js/common.js")


@pytest.mark.parametrize(
    "path",
    [
        "/nothing.js",
        "/css/nothing.css",
        "/../pyproject.toml",
        "/css/../../app.py",
        "/css/..%2f..%2fapp.py",
        "/%2e%2e/app.py",
        "/..%5capp.py",
        "/css/%2e%2e%2f%2e%2e%2fapp.py",
        "/js/common.js/..%2f..%2fapp.py",
    ],
)
def test_a_path_outside_the_static_folder_is_never_served(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code in {400, 404}
    assert "create_app" not in response.text
    assert "StaticFiles" not in response.text


def test_the_pages_carry_the_security_headers_that_forbid_inline_code(client: TestClient) -> None:
    policy = client.get("/").headers["content-security-policy"]
    assert "script-src 'self'" in policy
    assert "style-src 'self'" in policy
    assert "unsafe-inline" not in policy
    assert "unsafe-eval" not in policy


# --- The pages -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", PAGES)
def test_a_page_has_the_basics_of_a_small_screen_page(name: str) -> None:
    html = text_of(name)
    assert html.startswith("<!doctype html>")
    assert '<html lang="en">' in html
    assert '<meta charset="utf-8">' in html
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert '<meta name="color-scheme" content="light dark">' in html
    assert re.search(r"<title>Seeing monitor: [A-Za-z ]+</title>", html)
    assert '<main id="main">' in html
    assert "<noscript>" in html


@pytest.mark.parametrize(("name", "page"), PAGES.items())
def test_a_page_loads_the_shared_script_first_and_names_itself_to_it(name: str, page: str) -> None:
    html = text_of(name)
    sources = re.findall(r'<script src="(js/[a-z]+\.js)" defer></script>', html)
    assert sources[0] == "js/common.js"
    assert sources[-1] == f"js/{page if page != 'now' else 'now'}.js"
    script = text_of(sources[-1])
    assert f'boot("{page}"' in script
    assert "DOMContentLoaded" in script
    assert html.count("<script") == len(sources)  # no inline script


def test_the_four_pages_of_the_brief_exist_and_are_linked_from_the_frame() -> None:
    frame = text_of("js/common.js")
    for page, href in (
        ("Now", "./"),
        ("History", "history.html"),
        ("Images", "images.html"),
        ("Align", "align.html"),
    ):
        assert f'href: "{href}", label: "{page}"' in frame
    for name in NAV_PAGES:
        assert any(page == name for page in PAGES.values())


def test_every_reference_inside_the_pages_resolves(client: TestClient) -> None:
    for name in PAGES:
        html = text_of(name)
        for value in re.findall(r'(?:href|src)="([^"#]+)"', html):
            assert not value.startswith(("http:", "https:", "//")), (name, value)
            response = client.get("/" + value)
            assert response.status_code == 200, (name, value)


def test_every_reference_inside_the_scripts_and_the_style_sheet_resolves(
    client: TestClient,
) -> None:
    frame = text_of("js/common.js")
    for value in re.findall(r'href: "([^"]+)"', frame):
        if value.startswith("api.html") or value.endswith(".html") or value == "./":
            assert client.get("/" + value.lstrip("./")).status_code == 200, value
    css = text_of("css/app.css")
    assert "url(" not in css  # no image and no font file
    assert "@import" not in css
    assert "@font-face" not in css


# --- Self-contained --------------------------------------------------------------------------

EXTERNAL = re.compile(r"""(?ix)
    (?:https?|wss?|ftp)://[a-z0-9\[]       # an absolute URL with a host
    | (?<![:\w])//[a-z0-9][a-z0-9.-]*\.[a-z]{2,}   # a protocol-relative URL
    | url\(\s*['"]?\s*(?:https?:)?//        # a CSS url() to another host
    | @import
    """)


def test_no_file_loads_anything_from_another_host() -> None:
    for path in static_files():
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".svg":
            text = text.replace(f'xmlns="{SVG_NAMESPACE}"', "")  # a namespace names no resource
        for match in EXTERNAL.finditer(text):
            raise AssertionError(f"{path.name} refers to another host: {match.group(0)!r}")


def test_no_page_needs_an_inline_script_style_or_handler() -> None:
    for name in PAGES:
        html = text_of(name)
        assert "<style" not in html, name
        assert not re.search(r"\sstyle=", html), name
        assert not re.search(r"\son[a-z]+=", html), name
        assert "javascript:" not in html, name
        assert not re.search(r"<script(?![^>]*\ssrc=)", html), name


def test_the_scripts_never_build_markup_from_text_or_run_text_as_code() -> None:
    forbidden = (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        'setAttribute("style"',
        "setAttribute('style'",
    )
    for path in scripts():
        text = path.read_text(encoding="utf-8")
        for word in forbidden:
            assert word not in text, f"{path.name} uses {word}"
        assert not re.search(r"""setTimeout\(\s*['"]""", text), path.name


def test_only_the_shared_script_touches_the_browser_storage() -> None:
    """Storage can be blocked or missing. The shared script reads it inside a try block."""
    for path in scripts():
        text = path.read_text(encoding="utf-8")
        if path.name == "common.js":
            assert "window[name].getItem" in text
            assert re.search(r"try \{\s*return window\[name\]\.getItem", text)
            assert re.search(r"try \{\s*if \(value === null\)", text)
        else:
            assert "localStorage" not in text, path.name
            assert "sessionStorage" not in text, path.name


def test_the_files_stay_small() -> None:
    sizes = {path.name: path.stat().st_size for path in static_files()}
    assert all(size <= MAX_FILE_BYTES for size in sizes.values()), sizes
    assert sum(sizes.values()) <= MAX_TOTAL_BYTES, sum(sizes.values())


# --- Ids -------------------------------------------------------------------------------------


def html_ids(name: str) -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', text_of(name)))


def created_ids(script: str) -> set[str]:
    return set(re.findall(r'\bid: "([^"]+)"', script))


def looked_up_ids(script: str) -> set[str]:
    found = set(re.findall(r'\$\("([^"]+)"\)', script))
    return found | set(re.findall(r'getElementById\("([^"]+)"\)', script))


@pytest.mark.parametrize(("name", "page"), PAGES.items())
def test_every_id_that_a_script_looks_up_exists_in_its_page(name: str, page: str) -> None:
    script_name = f"{page}.js"
    script = text_of(f"js/{script_name}")
    shared = text_of("js/common.js")
    available = html_ids(name) | created_ids(script) | created_ids(shared)
    missing = looked_up_ids(script) - available
    assert not missing, f"{script_name} looks up {sorted(missing)}, which no page element has"
    dynamic = set(DYNAMIC_IDS.get(script_name, []))
    assert dynamic <= available, f"missing for {script_name}: {sorted(dynamic - available)}"


def test_the_shared_script_looks_up_only_ids_that_it_creates() -> None:
    shared = text_of("js/common.js")
    missing = looked_up_ids(shared) - created_ids(shared)
    assert not missing, sorted(missing)


def test_a_page_has_no_duplicate_id() -> None:
    for name in PAGES:
        ids = re.findall(r'\bid="([^"]+)"', text_of(name))
        assert len(ids) == len(set(ids)), (name, sorted({i for i in ids if ids.count(i) > 1}))


# --- Themes ----------------------------------------------------------------------------------


def test_the_style_sheet_follows_the_dark_scheme_and_has_a_red_night_mode() -> None:
    css = text_of("css/app.css")
    assert "@media (prefers-color-scheme: dark)" in css
    assert ':root:not([data-theme="light"])' in css
    assert ':root[data-night="1"]' in css
    overlay = re.search(r"\.night-overlay\s*\{([^}]*)\}", css)
    assert overlay is not None
    rule = overlay.group(1)
    assert "position: fixed" in rule
    assert "mix-blend-mode: multiply" in rule
    assert "background: #ff0000" in rule
    assert "pointer-events: none" in rule
    assert "z-index: 1000" in rule


def test_no_dialog_element_sits_above_the_night_filter() -> None:
    """A <dialog> opened with showModal() goes to the top layer, above the red filter."""
    for path in static_files():
        text = path.read_text(encoding="utf-8")
        assert "<dialog" not in text, path.name
        assert "showModal" not in text, path.name
        assert "popover" not in text, path.name


def test_the_night_mode_is_a_toggle_with_a_stored_choice_and_a_filter_element() -> None:
    shared = text_of("js/common.js")
    assert '"seeingmon.night"' in shared
    assert 'id: "night-overlay"' in shared
    assert 'id: "night-button"' in shared
    assert '"aria-pressed"' in shared


def test_the_layout_gives_a_phone_one_column_and_no_horizontal_scroll() -> None:
    css = text_of("css/app.css")
    assert "overflow-x: hidden" in css
    assert "minmax(min(100%," in css  # a card never forces the grid wider than the screen
    assert "@media (max-width: 760px)" in css
    assert "@media (max-width: 420px)" in css
    assert re.search(r"\.align\s*\{[^}]*grid-template-columns: minmax\(0, 1\.7fr\)", css)
    narrow = r"@media \(max-width: 760px\)\s*\{\s*\.align\s*\{\s*grid-template-columns: "
    assert re.search(narrow + r"minmax\(0, 1fr\)", css)


# --- JavaScript syntax -----------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
@pytest.mark.parametrize("path", scripts(), ids=lambda path: path.name)
def test_node_parses_every_script(path: Path) -> None:
    result = subprocess.run(
        ["node", "--check", str(path)], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr
