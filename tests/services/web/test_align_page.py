"""The Align page: its structure, its scripts, and the colors of the sky overlay.

The behavior of the geometry and of the words is checked by the Node scenarios in
`tests/services/web/js/` (`skygrid_scenarios.js` and `aligntext_scenarios.js`). These tests check
what a Python test can: the order of the cards, the switches of the overlay and their size on a
phone, the scripts that the page loads, that the pure scripts stay pure, and the contrast of every
overlay color on the dark image, in the three themes and after the filter of the night mode.
"""

from __future__ import annotations

import re

import pytest

from seeingmon.services.web.app import STATIC_DIR

HEX_TOKEN = re.compile(r"(--[a-z0-9-]+):\s*(#[0-9a-fA-F]{6})")
NUMBER_TOKEN = re.compile(r"(--[a-z0-9-]+):\s*([0-9.]+)\s*;")

THEMES = ("light", "dark", "night")
OVERLAY_COLORS = (
    "--overlay-grid",
    "--overlay-orbit-good",
    "--overlay-orbit-warn",
    "--overlay-orbit-bad",
    "--overlay-pole",
    "--overlay-aim",
    "--overlay-target",
    "--overlay-solved",
)
GRAPHIC_MINIMUM = 3.0  # WCAG 1.4.11, for the parts of a graphic that carry meaning


def text_of(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


# --- The page --------------------------------------------------------------------------------


def test_the_page_loads_the_geometry_and_the_words_before_its_own_script() -> None:
    html = text_of("align.html")
    sources = re.findall(r'<script src="(js/[a-z]+\.js)" defer></script>', html)
    assert sources == [
        "js/common.js",
        "js/live.js",
        "js/skygrid.js",
        "js/aligntext.js",
        "js/focus.js",
        "js/align.js",
    ]


def test_the_overlays_are_three_switches_in_one_group() -> None:
    html = text_of("align.html")
    group = re.search(r'<fieldset class="overlays"[^>]*>(.*?)</fieldset>', html, re.S)
    assert group is not None
    body = group.group(1)
    assert "<legend>Overlays</legend>" in body
    labels = re.findall(
        r'<label class="check" for="([^"]+)">'
        r'<input type="checkbox" id="\1" checked> ([^<]+)</label>',
        body,
    )
    assert labels == [
        ("overlay-pole", "Pole and orbit"),
        ("overlay-grid", "Coordinate grid"),
        ("overlay-target", "Target and Polaris"),
    ]
    assert "Show the overlays" not in html


def test_the_switches_are_big_enough_for_a_thumb() -> None:
    css = text_of("css/app.css")
    rule = re.search(r"\.check\s*\{([^}]*)\}", css)
    assert rule is not None
    height = re.search(r"min-height:\s*(\d+)px", rule.group(1))
    assert height is not None
    assert int(height.group(1)) >= 44
    box = re.search(r"\.check input\s*\{([^}]*)\}", css)
    assert box is not None
    assert "width: 22px" in box.group(1)


def test_the_pole_card_comes_before_the_offset_card_and_has_its_three_lines() -> None:
    html = text_of("align.html")
    assert html.index('id="h-pole"') < html.index('id="h-offset"')
    assert "Pole alignment" in html
    for element in ('id="pole-sentence"', 'id="orbit-sentence"', 'id="coordinates-note"'):
        assert element in html
    assert "Coordinates are of date." in html


def test_the_offset_card_has_a_note_a_roll_block_and_the_target_settings() -> None:
    html = text_of("align.html")
    assert 'id="offset-note"' in html
    assert 'id="roll-block"' in html
    details = re.search(r'<details class="settings"[^>]*>(.*?)</details>', html, re.S)
    assert details is not None
    body = details.group(1)
    assert "<summary>Target settings for this frame</summary>" in body
    assert '<pre class="toml" id="target-toml"' in body
    assert 'id="copy-target"' in body
    assert body.index("target-toml") < body.index("copy-target")


def test_the_help_text_describes_a_first_alignment() -> None:
    html = text_of("align.html")
    assert "labeled pole" in html
    assert "labeled aim" in html
    assert "orbit of Polaris" in html


# --- The scripts -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["skygrid.js", "aligntext.js"])
def test_the_pure_scripts_touch_neither_the_page_nor_the_browser(name: str) -> None:
    """Node runs them in a context with an empty `window`, so they may use nothing else."""
    script = text_of(f"js/{name}")
    for word in (
        "document",
        "getElementById",
        "localStorage",
        "sessionStorage",
        "fetch(",
        "getContext",
    ):
        assert word not in script, f"{name} uses {word}"
    assert re.search(r"window\.Seeing\.(SkyGrid|AlignText) = ", script)


def test_the_page_script_keeps_its_choices_through_the_shared_helpers() -> None:
    script = text_of("js/align.js")
    assert 'recall("align." + key' in script
    assert 'remember("align." + key' in script
    assert "localStorage" not in script
    shared = text_of("js/common.js")
    assert 'readStorage("localStorage", PREFERENCE_PREFIX + name)' in shared
    assert re.search(r"try \{\s*return window\[name\]\.getItem", shared)  # inside the try block


def test_the_copy_button_uses_the_clipboard_where_it_exists_and_falls_back_to_the_selection() -> (
    None
):
    script = text_of("js/align.js")
    assert "navigator.clipboard && typeof navigator.clipboard.writeText" in script
    assert "selectText(node)" in script  # the text stays selectable on a page without a clipboard
    css = text_of("css/app.css")
    toml = re.search(r"\.toml\s*\{([^}]*)\}", css)
    assert toml is not None
    assert "user-select: all" in toml.group(1)


def test_the_canvas_draws_the_layers_in_the_documented_order() -> None:
    script = text_of("js/align.js")
    marks = [
        "// 1. The coordinate grid.",
        "// 2. The orbit of Polaris",
        "// 3. The pole",
        "// 4. The aim",
        "// 5. Polaris",
        "// 6. The target",
    ]
    positions = [script.index(mark) for mark in marks]
    assert positions == sorted(positions)
    assert script.index("ctx.drawImage(picture") < positions[0]  # the picture comes first


def test_the_polaris_marker_does_not_depend_on_the_target() -> None:
    """A new install has no target, and it still shows Polaris where the solver finds it."""
    script = text_of("js/align.js")
    assert "const solved = show.target && state.solved ?" in script
    assert "show.target && state.target ?" in script
    assert "!state.target" in script  # the aim cross appears when no target is set


# --- Colors ----------------------------------------------------------------------------------


def token_block(css: str, marker: str) -> str:
    start = css.index(marker)
    return css[css.index("{", start) + 1 : css.index("\n}", start)]


def palettes() -> dict[str, dict[str, str]]:
    css = text_of("css/app.css")
    light = dict(HEX_TOKEN.findall(token_block(css, "\n:root {")))
    dark = dict(HEX_TOKEN.findall(token_block(css, ':root:not([data-theme="light"])')))
    night = dict(HEX_TOKEN.findall(token_block(css, ':root[data-night="1"]')))
    return {"light": light, "dark": {**light, **dark}, "night": {**light, **night}}


def alphas() -> dict[str, float]:
    css = text_of("css/app.css")
    light = float(dict(NUMBER_TOKEN.findall(token_block(css, "\n:root {")))["--overlay-grid-alpha"])
    night = float(
        dict(NUMBER_TOKEN.findall(token_block(css, ':root[data-night="1"]')))[
            "--overlay-grid-alpha"
        ]
    )
    return {"light": light, "dark": light, "night": night}


def linear(channel: int) -> float:
    value = channel / 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def luminance(color: str, *, red_only: bool) -> float:
    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    if red_only:
        green = blue = 0
    return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)


def contrast_on_black(color: str, *, red_only: bool, alpha: float = 1.0) -> float:
    """The contrast of a color that is drawn with `alpha` over a black sky."""
    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    blended = "#" + "".join(f"{round(channel * alpha):02x}" for channel in (red, green, blue))
    return (luminance(blended, red_only=red_only) + 0.05) / 0.05


@pytest.mark.parametrize("theme", THEMES)
def test_every_color_of_the_overlay_reads_on_the_dark_image(theme: str) -> None:
    """The live view is a dark image in every theme, and the night filter keeps the red channel."""
    tokens = palettes()[theme]
    for name in OVERLAY_COLORS:
        ratio = contrast_on_black(tokens[name], red_only=theme == "night")
        assert ratio >= GRAPHIC_MINIMUM, f"{theme}: {name} is {ratio:.2f} to 1 on the image"


@pytest.mark.parametrize("theme", THEMES)
def test_the_grid_lines_read_on_the_dark_image_after_their_alpha(theme: str) -> None:
    tokens = palettes()[theme]
    alpha = alphas()[theme]
    assert 0.3 <= alpha <= 1.0
    ratio = contrast_on_black(tokens["--overlay-grid"], red_only=theme == "night", alpha=alpha)
    assert ratio >= GRAPHIC_MINIMUM, f"{theme}: the grid is {ratio:.2f} to 1 at alpha {alpha}"


def test_the_halo_under_the_labels_is_black_in_every_theme() -> None:
    for theme, tokens in palettes().items():
        assert tokens["--overlay-halo"] == "#000000", theme


def test_the_night_palette_sets_every_overlay_token_on_its_own() -> None:
    css = text_of("css/app.css")
    night = token_block(css, ':root[data-night="1"]')
    for name in (*OVERLAY_COLORS, "--overlay-grid-alpha", "--overlay-halo"):
        assert f"{name}:" in night, name


def test_the_orbit_states_differ_in_brightness_in_the_night_mode() -> None:
    """The red filter cannot tell green from amber, so the states differ in the red channel."""
    tokens = palettes()["night"]
    reds = [
        int(tokens[name][1:3], 16)
        for name in (
            "--overlay-orbit-good",
            "--overlay-orbit-warn",
            "--overlay-orbit-bad",
        )
    ]
    assert reds == sorted(reds, reverse=True)
    assert len(set(reds)) == 3


def test_the_script_reads_the_overlay_colors_from_the_tokens_and_sets_none_of_its_own() -> None:
    script = text_of("js/align.js")
    for name in OVERLAY_COLORS:
        assert name in script, name
    assert "--overlay-grid-alpha" in script
    assert "--overlay-halo" in script
