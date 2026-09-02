# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

"""Playwright fixtures for the anyplotlib backend browser tests.

Mirrors the harness used by anyplotlib's own test suite
(anyplotlib/tests/conftest.py + tests/test_interactive/):

* ``_pw_browser``     session-scoped headless Chromium
* ``interact_page``   open a hyperspy/anyplotlib figure in a page, return it
* ``take_screenshot`` figure -> H×W×C uint8 ndarray via the widget-root box
* ``apl_backend``     register the anyplotlib backend, restore mpl afterwards

Because the standalone HTML page has no live Python kernel, JS -> Python
event delivery is replayed explicitly: ``collect_events(page)`` records every
``event_json`` payload the browser emits, and ``replay_events(fig, events)``
feeds them through ``Figure._on_event`` exactly as anywidget would.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import tempfile

import numpy as np
import pytest

pytest.importorskip("anyplotlib")
pytest.importorskip("playwright.sync_api")

from hyperspy.tests.drawing.anyplotlib_playwright._png_utils import (  # noqa: E402
    compare_arrays,
    decode_png,
    encode_png,
)

# Layout constants for 1-D panels (match anyplotlib's figure_esm.js).
PAD_L, PAD_R, PAD_T, PAD_B = 58, 12, 12, 42

#: Golden PNGs are platform-specific — fonts and antialiasing differ between
#: operating systems — so each platform keeps its own set, and a platform
#: without one primes it instead of failing.
BASELINE_DIR = pathlib.Path(__file__).parent / "baselines" / platform.system().lower()

# Set HSPY_UPDATE_APL_BASELINES=1 to regenerate every golden PNG.
UPDATE_BASELINES = os.environ.get("HSPY_UPDATE_APL_BASELINES", "") not in ("", "0")


# ---------------------------------------------------------------------------
# Browser + backend fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _pw_browser():
    """Yield a headless Chromium browser for the whole test session."""
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True)
        except Error as exc:  # browser binary missing
            pytest.skip(f"Chromium not available ({exc}); run `playwright install chromium`")
        yield browser
        browser.close()


@pytest.fixture
def apl_backend():
    """Activate the anyplotlib backend; restore the matplotlib backend after."""
    from hyperspy.drawing import backends
    from hyperspy.drawing.backends._registry import load_backend
    from hyperspy.drawing.backends.anyplotlib import AnyplotlibBackend

    previous = backends.get_backend()
    backend = AnyplotlibBackend()
    backends.register_backend(backend)
    yield backend
    backends.register_backend(previous)
    try:
        load_backend  # keep import referenced for clarity
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Standalone-HTML helpers (sentinel + model injection, as in anyplotlib)
# ---------------------------------------------------------------------------


def real_figure(fig_or_proxy):
    """Return the underlying anyplotlib Figure from a figure or panel proxy."""
    return getattr(fig_or_proxy, "_real_fig", fig_or_proxy)


def _refresh_panel_traits(fig):
    """Force every panel to rewrite its ``panel_<id>_json`` trait.

    Widget ``set()`` calls push *targeted* updates that bypass the panel
    trait, so a standalone-HTML snapshot taken after widget mutations would
    otherwise embed stale widget geometry.
    """
    for ax in fig.get_axes():
        plot = getattr(ax, "_plot", None)
        if plot is not None:
            plot._push()


def _build_interact_html(fig):
    from anyplotlib._repr_utils import build_standalone_html

    fig = real_figure(fig)
    _refresh_panel_traits(fig)
    html = build_standalone_html(fig, resizable=False)
    html = html.replace(
        "renderFn({ model, el });",
        "renderFn({ model, el }); window._aplReady = true;",
    )
    html = html.replace(
        "const model   = makeModel(STATE);",
        "const model   = makeModel(STATE);\nwindow._aplModel = model;",
    )
    return html


@pytest.fixture
def interact_page(_pw_browser):
    """Return ``open(fig) -> Page`` rendering fig in headless Chromium."""
    pages, paths = [], []

    def _open(fig):
        html = _build_interact_html(fig)
        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="w", encoding="utf-8", delete=False
        ) as fh:
            fh.write(html)
            tmp = pathlib.Path(fh.name)
        paths.append(tmp)

        page = _pw_browser.new_page()
        pages.append(page)
        page.goto(tmp.as_uri())
        page.wait_for_function("() => window._aplReady === true", timeout=15_000)
        page.evaluate(
            "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
        )
        return page

    yield _open

    for page in pages:
        try:
            page.close()
        except Exception:
            pass
    for path in paths:
        path.unlink(missing_ok=True)


@pytest.fixture
def take_screenshot(interact_page):
    """Return ``shot(fig) -> H×W×C uint8 ndarray`` (fresh page per call)."""

    def _take(fig):
        page = interact_page(fig)
        png = page.locator("#widget-root").screenshot()
        page.close()
        return decode_png(png)

    return _take


def screenshot_page(page):
    """Screenshot the widget root of an already-open page."""
    page.evaluate(
        "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
    )
    return decode_png(page.locator("#widget-root").screenshot())


# ---------------------------------------------------------------------------
# JS event capture and Python replay (the "bridge" both ways)
# ---------------------------------------------------------------------------


def collect_events(page):
    """Record every event_json payload the JS engine writes to the model."""
    page.evaluate(
        """() => {
        window._aplAllEvents = [];
        const orig = window._aplModel.set.bind(window._aplModel);
        window._aplModel.set = (k, v) => {
            if (k === 'event_json') {
                try { window._aplAllEvents.push(JSON.parse(v)); } catch(_) {}
            }
            return orig(k, v);
        };
    }"""
    )


def get_events(page, event_type=None, widget_id=None):
    """Return collected events, optionally filtered."""
    events = page.evaluate("() => window._aplAllEvents") or []
    if event_type is not None:
        events = [e for e in events if e.get("event_type") == event_type]
    if widget_id is not None:
        events = [e for e in events if e.get("widget_id") == widget_id]
    return events


def replay_events(fig, events):
    """Feed browser-recorded event payloads into Python's dispatch pipeline.

    This is exactly what anywidget (or ``FigureBridge``) does when a kernel
    is attached; the standalone test page has none, so the test replays.
    """
    fig = real_figure(fig)
    for payload in events:
        fig._on_event({"new": json.dumps(payload)})


def push_panel_state(page, fig, plot):
    """Push a panel's current Python-side state into the open page.

    Mirrors ``FigureBridge`` -> ``handle.applyUpdate(key, value)``: after
    hyperspy mutates a plot, the refreshed ``panel_<id>_json`` trait is
    written into the JS model so the browser re-renders the panel.
    """
    fig = real_figure(fig)
    plot._push()  # fold any targeted widget updates back into the trait
    key = f"panel_{plot._id}_json"
    value = getattr(fig, key)
    page.evaluate(
        "([k, v]) => { window._aplModel.set(k, v); }",
        [key, value],
    )
    page.evaluate(
        "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
    )


# ---------------------------------------------------------------------------
# Panel geometry: data coords -> page coords
# ---------------------------------------------------------------------------


def interactive_boxes(page):
    """Bounding boxes of each panel's interactive canvas, sorted by page x.

    2-D panels: the interactive canvas covers exactly the plot area.
    1-D panels: it covers the whole panel; use ``PAD_*`` to reach the
    plot area inside it.
    """
    return page.evaluate(
        """() => {
        const boxes = [...document.querySelectorAll('canvas')]
            .filter(c => c.style.pointerEvents === 'all')
            .map(c => { const r = c.getBoundingClientRect();
                        return {x: r.x, y: r.y, w: r.width, h: r.height}; });
        boxes.sort((a, b) => a.x - b.x);
        return boxes;
    }"""
    )


def data_to_page_2d(box, xlim, ylim, x, y):
    """Map image data coords to page coords inside a 2-D panel box.

    2-D panels render aspect-equal: the image is letterboxed (centred)
    inside the interactive canvas, so the mapping must account for the
    fitted image rectangle, not the raw canvas.

    ``ylim`` must be given in *screen* order: ylim[0] renders at the top
    edge of the image (anyplotlib images default to origin='upper').
    """
    xrange_ = abs(xlim[1] - xlim[0])
    yrange_ = abs(ylim[1] - ylim[0])
    scale = min(box["w"] / xrange_, box["h"] / yrange_)
    img_w, img_h = xrange_ * scale, yrange_ * scale
    x0 = box["x"] + (box["w"] - img_w) / 2.0
    y0 = box["y"] + (box["h"] - img_h) / 2.0
    fx = (x - xlim[0]) / (xlim[1] - xlim[0])
    fy = (y - ylim[0]) / (ylim[1] - ylim[0])
    return x0 + fx * img_w, y0 + fy * img_h


def data_to_page_1d(box, xlim, ylim, x, y):
    """Map data coords to page coords inside a 1-D panel box."""
    plot_w = box["w"] - PAD_L - PAD_R
    plot_h = box["h"] - PAD_T - PAD_B
    fx = (x - xlim[0]) / (xlim[1] - xlim[0])
    fy = (ylim[1] - y) / (ylim[1] - ylim[0])
    return box["x"] + PAD_L + fx * plot_w, box["y"] + PAD_T + fy * plot_h


def drag(page, x0, y0, x1, y1, steps=12):
    """Mouse-drag from (x0, y0) to (x1, y1) in page coordinates."""
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x1, y1, steps=steps)
    page.mouse.up()
    page.wait_for_timeout(100)


# ---------------------------------------------------------------------------
# PNG baseline comparison
# ---------------------------------------------------------------------------


def assert_matches_baseline(arr, name, tol=8, max_diff_frac=0.02):
    """Compare *arr* against ``baselines/<platform>/<name>.png``.

    Missing baselines are created on first run (the test is then skipped so
    a fresh checkout, or a platform with no committed baselines, is primed
    rather than silently passing).  Set ``HSPY_UPDATE_APL_BASELINES=1`` to
    force-regenerate.
    """
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / f"{name}.png"

    if UPDATE_BASELINES or not path.exists():
        path.write_bytes(encode_png(arr))
        if UPDATE_BASELINES:
            return
        pytest.skip(f"baseline {name}.png created; re-run to compare")

    expected = decode_png(path.read_bytes())
    ok, message = compare_arrays(arr, expected, tol=tol, max_diff_frac=max_diff_frac)
    if not ok:
        failed = path.with_name(f"{name}.failed.png")
        failed.write_bytes(encode_png(arr))
        pytest.fail(f"{name}: {message} (actual saved to {failed.name})")


def assert_differs(arr_a, arr_b, min_diff_frac=0.001, tol=8):
    """Assert two screenshots differ in more than *min_diff_frac* of pixels."""
    assert arr_a.shape == arr_b.shape, (arr_a.shape, arr_b.shape)
    diff = np.abs(arr_a.astype(np.int32) - arr_b.astype(np.int32))
    bad = (diff > tol).any(axis=-1)
    frac = bad.sum() / bad.size
    assert frac >= min_diff_frac, (
        f"expected >= {min_diff_frac:.2%} of pixels to change, got {frac:.3%}"
    )
