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
    "dark.html": "dark",
    "flat.html": "flat",
    "api.html": "api",
}
NAV_PAGES = ("now", "history", "images", "align", "dark", "flat")
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


def test_the_six_pages_exist_and_are_linked_from_the_frame_in_order() -> None:
    frame = text_of("js/common.js")
    tabs = (
        ("now", "./", "Now"),
        ("history", "history.html", "History"),
        ("images", "images.html", "Images"),
        ("align", "align.html", "Align"),
        ("dark", "dark.html", "Dark"),
        ("flat", "flat.html", "Flat"),
    )
    for page, href, label in tabs:
        assert f'{{ id: "{page}", href: "{href}", label: "{label}" }}' in frame
    positions = [frame.index(f'label: "{label}"') for _, _, label in tabs]
    assert positions == sorted(positions)
    assert frame.count("href: ") >= 6
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


# --- The header ------------------------------------------------------------------------------


def media_block(css: str, query: str) -> str:
    """The text between the braces of the first `@media` block with this query."""
    start = css.index("{", css.index(query))
    depth = 0
    for position in range(start, len(css)):
        if css[position] == "{":
            depth += 1
        elif css[position] == "}":
            depth -= 1
            if depth == 0:
                return css[start + 1 : position]
    raise AssertionError(f"{query} has no closing brace")


def test_the_tools_sit_in_the_title_row_and_have_an_icon_and_a_label() -> None:
    frame = text_of("js/common.js")
    row = frame[frame.index('class: "top-row"') : frame.index('class: "tabs"')]
    assert 'class: "brand"' in row
    assert 'h("div", { class: "tools" }, tokenButton, nightButton)' in row
    assert 'class: "icon icon-" + icon' in frame
    assert '"aria-hidden": "true"' in frame
    assert 'class: "tool-label", text: label' in frame
    assert '"Token"' in frame
    assert '"Night mode"' in frame


def test_a_narrow_screen_hides_the_tool_label_from_the_eye_but_not_a_screen_reader() -> None:
    css = text_of("css/app.css")
    rule = rule_for(media_block(css, "@media (max-width: 559px)"), ".tool-label")
    assert "clip: rect(0 0 0 0)" in rule
    assert "display: none" not in rule
    assert "visibility: hidden" not in rule


def test_a_tall_screen_keeps_the_whole_header_at_the_top() -> None:
    css = text_of("css/app.css")
    assert "position: sticky" in rule_for(css, "\n.top {")


def test_a_short_screen_lets_the_title_row_scroll_away_and_keeps_the_tab_row_at_the_top() -> None:
    css = text_of("css/app.css")
    short = media_block(css, "@media (max-height: 639px)")
    assert "display: contents" in rule_for(short, ".top {")
    tabs = rule_for(short, ".tabs {")
    assert "position: sticky" in tabs
    assert "top: 0" in tabs


# --- The Dark page ---------------------------------------------------------------------------


def test_the_dark_page_loads_the_logic_before_the_page_script() -> None:
    html = text_of("dark.html")
    sources = re.findall(r'<script src="(js/[a-z]+\.js)" defer></script>', html)
    assert sources == ["js/common.js", "js/darktext.js", "js/dark.js"]


def test_every_field_of_the_dark_form_has_a_label() -> None:
    html = text_of("dark.html")
    fields = re.findall(r'<input type="(?:text|checkbox)" id="([^"]+)"', html)
    assert sorted(fields) == ["f-bias", "f-cover", "f-exposure", "f-frames", "f-pause"]
    for field in fields:
        assert f'for="{field}"' in html, field


def test_the_dark_page_announces_progress_and_mistakes_to_a_screen_reader() -> None:
    html = text_of("dark.html")
    assert re.search(r'id="task-message" role="status" aria-live="polite"', html)
    assert re.search(r'id="start-hint" aria-live="polite"', html)
    assert re.search(r'id="command-note" aria-live="polite"', html)
    assert re.search(r'id="dark-error" class="banner" hidden role="alert"', html)
    for field in ("exposure", "frames", "bias"):
        assert re.search(rf'id="e-{field}" hidden role="alert"', html), field


def test_the_table_of_dark_sets_keeps_its_roles_where_a_phone_turns_it_into_a_list() -> None:
    html = text_of("dark.html")
    for role in ('role="table"', 'role="rowgroup"', 'role="row"', 'role="columnheader"'):
        assert role in html, role
    script = text_of("js/dark.js")
    assert 'role: "row"' in script
    assert 'role: "cell"' in script
    phone = media_block(text_of("css/app.css"), "@media (max-width: 559px)")
    assert "display: grid" in rule_for(phone, ".sets tr {")
    assert "attr(data-label)" in rule_for(phone, ".sets td::before {")


def test_a_phase_of_a_session_shows_its_state_in_words_and_not_in_color_alone() -> None:
    script = text_of("js/dark.js")
    assert '"aria-current": phase.state === "active" ? "step" : null' in script
    assert 'class: "phase-name", text: phase.label' in script
    assert 'class: "phase-detail", text: phase.detail' in script


def test_the_dark_page_reads_the_library_with_the_poller_so_that_a_hidden_tab_waits() -> None:
    script = text_of("js/dark.js")
    assert "poller(read, () => DarkText.pollInterval(" in script
    assert 'api.get("dark")' in script
    assert 'command("commands/dark"' in script
    assert '"mode", { mode: "paused" }' in script  # Cancel is the pause command
    assert '"mode", { mode: "auto" }' in script  # Resume is the resume command


# --- The Flat page ---------------------------------------------------------------------------


def test_the_flat_page_loads_the_logic_before_the_page_script() -> None:
    html = text_of("flat.html")
    sources = re.findall(r'<script src="(js/[a-z]+\.js)" defer></script>', html)
    assert sources == ["js/common.js", "js/flattext.js", "js/flat.js"]


def test_every_field_of_the_flat_form_has_a_label() -> None:
    html = text_of("flat.html")
    fields = re.findall(r'<input type="(?:text|checkbox)" id="([^"]+)"', html)
    assert sorted(fields) == ["f-frames", "f-pause", "f-target"]
    for field in fields:
        assert f'for="{field}"' in html, field


def test_the_flat_page_announces_progress_and_mistakes_to_a_screen_reader() -> None:
    html = text_of("flat.html")
    assert re.search(r'id="task-message" role="status" aria-live="polite"', html)
    assert re.search(r'id="start-hint" aria-live="polite"', html)
    assert re.search(r'id="command-note" aria-live="polite"', html)
    assert re.search(r'id="review-note" aria-live="polite"', html)
    assert re.search(r'id="flat-error" class="banner" hidden role="alert"', html)
    for field in ("frames", "target"):
        assert re.search(rf'id="e-{field}" hidden role="alert"', html), field


def test_the_flat_page_has_the_four_steps_and_the_three_decisions_in_plain_words() -> None:
    html = text_of("flat.html")
    steps = re.search(r'<ol class="steps">(.*?)</ol>', html, re.S)
    assert steps is not None
    assert len(re.findall(r"<li>", steps.group(1))) == 4
    for label in ("Take flat", "Use this flat", "Discard", "Take a second set", "Stop", "Resume"):
        assert f">{label}</button>" in html, label


def test_the_table_of_flats_keeps_its_roles_where_a_phone_turns_it_into_a_list() -> None:
    html = text_of("flat.html")
    for role in ('role="table"', 'role="rowgroup"', 'role="row"', 'role="columnheader"'):
        assert role in html, role
    script = text_of("js/flat.js")
    assert 'role: "row"' in script
    assert 'role: "cell"' in script
    assert 'empty: "1"' in script  # the phone skips a cell with this mark
    css = text_of("css/app.css")
    assert "content: attr(data-label)" in css  # and shows the label in front of each value
    assert ".sets td[data-empty]" in css
    assert 'class="sets" id="flats-table"' in html  # the rules of the table of dark sets apply


def test_no_script_gives_a_dataset_a_null_because_the_dataset_turns_it_into_text() -> None:
    """A `null` in the dataset of an element becomes the text "null", and the attribute is there.

    The Flat page once marked every cell of its table as empty this way, and a phone hid the table.
    The helper `h` skips a `null` property, but it assigns the entries of a dataset as they are.
    """
    for path in scripts():
        text = path.read_text(encoding="utf-8")
        for literal in re.findall(r"dataset:\s*\{([^}]*)\}", text):
            assert not re.search(r"\bnull\b|\bundefined\b", literal), (path.name, literal)


def test_a_phase_of_a_flat_session_shows_its_state_in_words_and_not_in_color_alone() -> None:
    script = text_of("js/flat.js")
    assert '"aria-current": phase.state === "active" ? "step" : null' in script
    assert 'class: "phase-name", text: phase.label' in script
    assert 'class: "phase-detail", text: phase.detail' in script


def test_a_verdict_and_the_level_of_the_frames_have_a_word_and_not_a_color_alone() -> None:
    script = text_of("js/flat.js")
    assert 'h("span", { class: "chip", text: word, dataset: { level } })' in script
    assert "word.textContent = gauge.word" in script  # "On target", "Too dark", or "Too bright"
    assert "word.textContent = review.word" in script  # "Good", "Check", or "Problem"
    logic = text_of("js/flattext.js")
    assert 'const WORDS = { good: "Good", warn: "Check", bad: "Problem" }' in logic


def test_the_flat_page_reads_the_library_with_the_poller_so_that_a_hidden_tab_waits() -> None:
    script = text_of("js/flat.js")
    assert "poller(read, () => FlatText.pollInterval(" in script
    assert 'api.get("flat")' in script
    assert '"post", "flat/session"' in script
    assert '"post", "flat/session/stop"' in script
    assert '"flat/" + version + "/activate"' in script
    assert "api.delete(path)" in script
    assert '"post", "mode", { mode: "auto" }' in script  # Resume is the resume command


def test_the_flat_page_resumes_a_paused_scheduler_only_after_the_session_is_queued() -> None:
    script = text_of("js/flat.js")
    start = script[
        script.index("async function start(setNumber)") : script.index("async function startFirst")
    ]
    assert start.index('"flat/session"') < start.index('{ mode: "auto" }')
    assert "answer && answer.accepted && wasPaused" in start


def test_a_deletion_on_the_flat_page_asks_twice_and_a_use_asks_once() -> None:
    script = text_of("js/flat.js")
    assert 'const DESTRUCTIVE = ["discard", "delete"]' in script
    assert '"Press again to " + kind' in script  # the first press only changes the button
    assert "if (!DESTRUCTIVE.includes(kind))" in script
    ask = script[script.index("function ask(kind, version)") : script.index("function renderTable")]
    assert "setTimeout(() =>" in ask  # a deletion that nobody confirms lapses
    assert "decide(kind, version)" in ask


def test_the_flat_page_uses_only_the_tokens_of_the_style_sheet() -> None:
    css = text_of("css/app.css")
    tokens = set(re.findall(r"^\s+(--[a-z0-9-]+):", css, re.M))
    for name in ("flat.html", "js/flat.js", "js/flattext.js"):
        text = text_of(name)
        for used in re.findall(r"var\((--[a-z0-9-]+)", text):
            assert used in tokens, (name, used)
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", text), name  # no color outside the tokens
    start = css.index("/* --- Flat ")
    block = css[start : css.index("/* --- The red night mode")]
    for used in re.findall(r"var\((--[a-z0-9-]+)", block):
        assert used in tokens, used
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block)
    assert "rgb(" not in block


# --- Contrast --------------------------------------------------------------------------------

HEX_TOKEN = re.compile(r"(--[a-z0-9-]+):\s*(#[0-9a-fA-F]{6})")


def token_block(css: str, marker: str) -> dict[str, str]:
    """The color tokens of the first rule that follows `marker`."""
    start = css.index(marker)
    body = css[css.index("{", start) : css.index("}", start)]
    return dict(HEX_TOKEN.findall(body))


def palettes() -> dict[str, dict[str, str]]:
    css = text_of("css/app.css")
    light = token_block(css, "\n:root {")
    return {
        "light": light,
        "dark": {**light, **token_block(css, ':root:not([data-theme="light"])')},
        "night": {**light, **token_block(css, ':root[data-night="1"]')},
    }


def linear(channel: int) -> float:
    value = channel / 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def luminance(color: str, *, red_only: bool) -> float:
    """The relative luminance of a `#rrggbb` color (WCAG). The night filter keeps only the red
    channel, so with `red_only` the green and the blue count as zero."""
    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    if red_only:
        green = blue = 0
    return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)


def contrast(first: str, second: str, *, red_only: bool = False) -> float:
    lighter, darker = sorted(
        (luminance(first, red_only=red_only), luminance(second, red_only=red_only)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def seen(theme: str, foreground: str, background: str) -> float:
    """The contrast of two tokens as a person sees them: after the red filter in the night mode."""
    tokens = palettes()[theme]
    return contrast(tokens[foreground], tokens[background], red_only=theme == "night")


THEMES = ("light", "dark", "night")
SURFACES = ("--bg", "--surface", "--surface-2")
LEVELS = ("--good", "--warn", "--bad")
TEXT_PAIRS = [
    *((color, surface) for color in ("--text", "--muted") for surface in SURFACES),
    ("--accent-text", "--accent"),
    ("--accent", "--surface"),
    ("--accent", "--bg"),
    *((level, "--surface") for level in LEVELS),
    *((level, f"{level}-bg") for level in LEVELS),
    *(("--text", f"{level}-bg") for level in LEVELS),
]
# A red-only pair cannot pass 5.25 to 1, and the muted text of the night mode is secondary text,
# so it gets the 3 to 1 that WCAG asks for large text and interface parts.
NIGHT_MUTED_MINIMUM = 3.0


@pytest.mark.parametrize("theme", THEMES)
def test_every_text_pair_of_a_theme_has_the_contrast_that_wcag_asks_for(theme: str) -> None:
    for foreground, background in TEXT_PAIRS:
        minimum = NIGHT_MUTED_MINIMUM if theme == "night" and foreground == "--muted" else 4.5
        ratio = seen(theme, foreground, background)
        assert ratio >= minimum, f"{theme}: {foreground} on {background} is {ratio:.2f} to 1"


@pytest.mark.parametrize("theme", THEMES)
def test_the_series_and_the_marks_have_the_contrast_of_a_graphic(theme: str) -> None:
    for series in ("--series-1", "--series-2", "--series-3", "--series-4"):
        assert seen(theme, series, "--surface") >= 3.0, (theme, series)
    black = "#000000"  # the live view is a dark image in every theme
    tokens = palettes()[theme]
    for mark in ("--overlay-target", "--overlay-solved"):
        ratio = contrast(tokens[mark], black, red_only=theme == "night")
        assert ratio >= 3.0, f"{theme}: {mark} on the live view is {ratio:.2f} to 1"


def test_the_night_palette_sets_every_color_that_the_dark_palette_sets() -> None:
    """The night palette starts from the light tokens, so a token that it forgets stays light."""
    css = text_of("css/app.css")
    dark = set(token_block(css, ':root:not([data-theme="light"])'))
    night = set(token_block(css, ':root[data-night="1"]'))
    assert dark <= night, sorted(dark - night)
    assert {"--overlay-target", "--overlay-solved"} <= night


def test_the_contrast_helper_matches_known_values() -> None:
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#ffffff", "#ffffff") == pytest.approx(1.0)
    assert contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)
    assert contrast("#ffffff", "#000000", red_only=True) == pytest.approx(5.25, abs=0.01)
    assert contrast("#ffffff", "#0000ff", red_only=True) == pytest.approx(5.25, abs=0.01)


def rule_for(css: str, selector: str) -> str:
    """The declarations of the rule whose selector list holds `selector`."""
    position = css.index(selector)
    return css[css.index("{", position) + 1 : css.index("}", position)]


@pytest.mark.parametrize(
    "selector",
    [
        'button[aria-pressed="true"]:hover:not(:disabled)',
        'button[aria-current="true"]:hover:not(:disabled)',
        'button[aria-pressed="true"]:focus-visible',
        'button[aria-current="true"]:focus-visible',
    ],
)
def test_a_pressed_button_keeps_its_colors_under_the_pointer_and_the_keyboard_focus(
    selector: str,
) -> None:
    """The plain hover rule is more specific than the pressed rule. Without these rules, a pressed
    button under the pointer gets the quiet background and keeps its dark text, so it cannot be
    read (in every theme, and in the night mode as dark red on dark red)."""
    css = text_of("css/app.css")
    rule = rule_for(css, selector)
    assert "background: var(--accent)" in rule
    assert "color: var(--accent-text)" in rule
    assert "border-color: var(--accent)" in rule


def test_the_pressed_rules_come_after_the_quiet_hover_rule_that_they_override() -> None:
    css = text_of("css/app.css")
    hover = rule_for(css, "button:hover:not(:disabled)")
    assert "background: var(--surface-2)" in hover
    assert css.index('button[aria-pressed="true"]:hover:not(:disabled)') > css.index(
        "button:hover:not(:disabled)"
    )


# --- JavaScript syntax -----------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
@pytest.mark.parametrize("path", scripts(), ids=lambda path: path.name)
def test_node_parses_every_script(path: Path) -> None:
    result = subprocess.run(
        ["node", "--check", str(path)], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr
