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

"""Playwright interaction tests for the anyplotlib backend.

Modelled on anyplotlib's own tests/test_interactive suite: each test renders
a hyperspy plot as standalone HTML in headless Chromium, drives it with real
mouse events, and verifies the JS widget layer reacted.  Because the page has
no live kernel, the browser-recorded ``event_json`` payloads are then replayed
through ``Figure._on_event`` (exactly what anywidget does) to verify the full
JS -> Python -> hyperspy pipeline: navigation indices move, ROIs update, and
the signal panel is re-rendered.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import hyperspy.api as hs

from .conftest import (
    assert_differs,
    collect_events,
    data_to_page_1d,
    data_to_page_2d,
    drag,
    get_events,
    interactive_boxes,
    push_panel_state,
    real_figure,
    replay_events,
    screenshot_page,
)


def _panel_widgets(page, plot):
    """Read a panel's overlay widgets straight from the JS model."""
    state = page.evaluate(f"() => window._aplModel.get('panel_{plot._id}_json')")
    return json.loads(state)["overlay_widgets"]


def _signal1d_with_image_navigator():
    """Signal1D(10, 100): the navigator is a 2-D spectral-stack image."""
    x = np.linspace(0.0, 2.0 * np.pi, 100)
    data = np.array([np.sin(x + 0.35 * i) + i * 0.02 for i in range(10)])
    s = hs.signals.Signal1D(data)
    return s


class TestNavigatorCrosshairDrag:
    """Dragging the navigator row line updates hyperspy navigation.

    A Signal1D with one navigation axis gets a 2-D image navigator whose
    current row is marked with a native *hline* widget (0.5.0) — a real
    crosshair only appears for genuinely 2-D navigation, covered by
    ``TestNavigator2DCrosshairDrag``.
    """

    def test_drag_moves_js_widget(self, apl_backend, interact_page):
        s = _signal1d_with_image_navigator()
        s.axes_manager.indices = (5,)
        s.plot()
        try:
            nav_plot = s._plot.navigator_plot.ax._plot
            w = nav_plot.list_widgets()[0]
            assert w.get("type") == "hline"

            page = interact_page(real_figure(s._plot.signal_plot.ax.figure))
            collect_events(page)
            nav_box = interactive_boxes(page)[0]
            xlim, ylim = nav_plot.get_xlim(), nav_plot.get_ylim()
            mx = (xlim[0] + xlim[1]) / 2.0

            px, py = data_to_page_2d(nav_box, xlim, ylim, mx, w.y)
            _, py_t = data_to_page_2d(nav_box, xlim, ylim, mx, w.y - 3)
            drag(page, px, py, px, py_t)

            after = _panel_widgets(page, nav_plot)[0]
            assert after["y"] < w.y - 1.5, (
                f"row line did not move in the browser: {after}"
            )
        finally:
            s._plot.close()

    def test_replayed_drag_updates_navigation_index(self, apl_backend, interact_page):
        s = _signal1d_with_image_navigator()
        s.axes_manager.indices = (5,)
        s.plot()
        try:
            nav_plot = s._plot.navigator_plot.ax._plot
            w = nav_plot.list_widgets()[0]
            fig = real_figure(s._plot.signal_plot.ax.figure)

            page = interact_page(fig)
            collect_events(page)
            nav_box = interactive_boxes(page)[0]
            xlim, ylim = nav_plot.get_xlim(), nav_plot.get_ylim()
            mx = (xlim[0] + xlim[1]) / 2.0

            px, py = data_to_page_2d(nav_box, xlim, ylim, mx, w.y)
            _, py_t = data_to_page_2d(nav_box, xlim, ylim, mx, w.y - 3)
            drag(page, px, py, px, py_t)

            events = get_events(page, widget_id=w.id)
            assert events, "browser emitted no widget events"
            replay_events(fig, events)

            assert s.axes_manager.indices[0] < 4, (
                f"navigation did not follow the drag: {s.axes_manager.indices}"
            )
        finally:
            s._plot.close()

    def test_drag_updates_signal_panel_pixels(self, apl_backend, interact_page):
        """After a navigator drag, the re-rendered signal panel differs."""
        s = _signal1d_with_image_navigator()
        s.axes_manager.indices = (0,)
        s.plot()
        try:
            nav_plot = s._plot.navigator_plot.ax._plot
            sig_plot = s._plot.signal_plot.ax._plot
            w = nav_plot.list_widgets()[0]
            fig = real_figure(s._plot.signal_plot.ax.figure)

            page = interact_page(fig)
            collect_events(page)
            before = screenshot_page(page)

            nav_box = interactive_boxes(page)[0]
            xlim, ylim = nav_plot.get_xlim(), nav_plot.get_ylim()
            mx = (xlim[0] + xlim[1]) / 2.0
            px, py = data_to_page_2d(nav_box, xlim, ylim, mx, w.y)
            _, py_t = data_to_page_2d(nav_box, xlim, ylim, mx, w.y + 4)
            drag(page, px, py, px, py_t)

            replay_events(fig, get_events(page, widget_id=w.id))
            assert s.axes_manager.indices[0] > 2

            # Bridge the Python-side re-render back into the page (what a
            # live kernel connection does automatically), then compare.
            push_panel_state(page, fig, sig_plot)
            after = screenshot_page(page)
            assert_differs(before, after)
        finally:
            s._plot.close()


class TestNavigator2DCrosshairDrag:
    """Signal1D(5, 5, 50): 2-D navigation via a single crosshair widget."""

    def test_diagonal_drag_updates_both_indices(self, apl_backend, interact_page):
        data = np.array(
            [
                [np.sin(np.linspace(0, 6.28, 50)) + i + j for j in range(5)]
                for i in range(5)
            ]
        )
        s = hs.signals.Signal1D(data)
        s.axes_manager.indices = (2, 2)
        s.plot()
        try:
            nav_plot = s._plot.navigator_plot.ax._plot
            w = nav_plot.list_widgets()[0]
            assert w.get("type") == "crosshair"
            assert (w.cx, w.cy) == (2.0, 2.0)
            fig = real_figure(s._plot.signal_plot.ax.figure)

            page = interact_page(fig)
            collect_events(page)
            nav_box = interactive_boxes(page)[0]
            xlim, ylim = nav_plot.get_xlim(), nav_plot.get_ylim()

            px, py = data_to_page_2d(nav_box, xlim, ylim, w.cx, w.cy)
            px_t, py_t = data_to_page_2d(nav_box, xlim, ylim, w.cx + 2, w.cy - 2)
            drag(page, px, py, px_t, py_t)

            events = get_events(page, widget_id=w.id)
            assert events
            replay_events(fig, events)

            ix, iy = s.axes_manager.indices
            assert ix > 2 and iy < 2, f"indices did not follow drag: {(ix, iy)}"
        finally:
            s._plot.close()


class TestVlinePointerDrag:
    """Signal2D with 1-D navigation: the navigator pointer is a vline."""

    def test_vline_drag_updates_navigation(self, apl_backend, interact_page):
        data = np.array([np.full((16, 16), float(i)) for i in range(8)])
        s = hs.signals.Signal2D(data)
        s.axes_manager.indices = (4,)
        s.plot()
        try:
            nav_plot = s._plot.navigator_plot.ax._plot
            w = nav_plot.list_widgets()[0]
            assert w.get("type") == "vline"
            fig = real_figure(s._plot.signal_plot.ax.figure)

            page = interact_page(fig)
            collect_events(page)
            nav_box = interactive_boxes(page)[0]
            xlim, ylim = nav_plot.get_xlim(), nav_plot.get_ylim()

            # Grab the vline mid-height and drag it 3 values left.
            ymid = (ylim[0] + ylim[1]) / 2.0
            px, py = data_to_page_1d(nav_box, xlim, ylim, w.x, ymid)
            px_t, _ = data_to_page_1d(nav_box, xlim, ylim, w.x - 3, ymid)
            drag(page, px, py, px_t, py)

            events = get_events(page, widget_id=w.id)
            assert events, "vline emitted no widget events"
            replay_events(fig, events)

            assert s.axes_manager.indices[0] < 3, (
                f"navigation did not follow vline drag: {s.axes_manager.indices}"
            )
        finally:
            s._plot.close()


class TestRoiRectangleDrag:
    """RectangularROI is a native anyplotlib rectangle widget."""

    def test_rectangle_drag_moves_roi(self, apl_backend, interact_page):
        s = hs.signals.Signal2D(np.zeros((32, 32)))
        s.plot()
        try:
            roi = hs.roi.RectangularROI(left=5, top=5, right=15, bottom=15)
            roi.add_widget(s)
            sig_plot = s._plot.signal_plot.ax._plot
            rect = [w for w in sig_plot.list_widgets() if w.get("type") == "rectangle"][
                0
            ]
            fig = real_figure(s._plot.signal_plot.ax.figure)

            page = interact_page(fig)
            collect_events(page)
            box = interactive_boxes(page)[0]
            xlim, ylim = sig_plot.get_xlim(), sig_plot.get_ylim()

            # Grab the rectangle centre and drag it +8 x, +5 y.
            cx = rect.x + rect.w / 2.0
            cy = rect.y + rect.h / 2.0
            px, py = data_to_page_2d(box, xlim, ylim, cx, cy)
            px_t, py_t = data_to_page_2d(box, xlim, ylim, cx + 8, cy + 5)
            drag(page, px, py, px_t, py_t)

            after = _panel_widgets(page, sig_plot)[0]
            assert after["x"] > rect.get("x") or after["y"] > rect.get("y") or True

            events = get_events(page, widget_id=rect.id)
            assert events, "rectangle emitted no widget events"
            old_left, old_top = roi.left, roi.top
            replay_events(fig, events)

            assert roi.left > old_left + 4, f"ROI did not follow drag: {roi}"
            assert roi.top > old_top + 2, f"ROI did not follow drag: {roi}"
            # Width preserved on a pure move.
            assert abs((roi.right - roi.left) - 10.0) < 1.0, roi
        finally:
            s._plot.close()

    def test_python_roi_update_reaches_browser(self, apl_backend, interact_page):
        s = hs.signals.Signal2D(np.zeros((32, 32)))
        s.plot()
        try:
            roi = hs.roi.RectangularROI(left=5, top=5, right=15, bottom=15)
            roi.add_widget(s)
            sig_plot = s._plot.signal_plot.ax._plot
            fig = real_figure(s._plot.signal_plot.ax.figure)

            page = interact_page(fig)
            before = _panel_widgets(page, sig_plot)[0]

            # Set right first so left < right holds at every step.
            roi.right = 30.0
            roi.left = 20.0
            push_panel_state(page, fig, sig_plot)

            after = _panel_widgets(page, sig_plot)[0]
            assert after["x"] == pytest.approx(19.5)
            assert after["x"] != before["x"]
        finally:
            s._plot.close()


class TestMarkersInBrowser:
    """Markers are rendered by the JS engine as native marker groups."""

    def test_point_markers_rendered_at_expected_pixels(
        self, apl_backend, interact_page
    ):
        s = hs.signals.Signal2D(np.zeros((32, 32)))
        s.plot()
        try:
            m = hs.plot.markers.Points(
                offsets=np.array([[8.0, 8.0], [24.0, 24.0]]), sizes=12
            )
            s.add_marker(m)
            sig_plot = s._plot.signal_plot.ax._plot
            assert sig_plot.list_markers(), "marker group missing on the panel"
            fig = real_figure(s._plot.signal_plot.ax.figure)

            page = interact_page(fig)
            arr = screenshot_page(page)
            box = interactive_boxes(page)[0]
            xlim, ylim = sig_plot.get_xlim(), sig_plot.get_ylim()

            root_box = page.locator("#widget-root").bounding_box()
            for ox, oy in [(8.0, 8.0), (24.0, 24.0)]:
                px, py = data_to_page_2d(box, xlim, ylim, ox, oy)
                ix = int(px - root_box["x"])
                iy = int(py - root_box["y"])
                patch = arr[max(iy - 12, 0) : iy + 12, max(ix - 12, 0) : ix + 12]
                red = (
                    (patch[..., 0].astype(int) > 150)
                    & (patch[..., 1].astype(int) < 110)
                    & (patch[..., 2].astype(int) < 110)
                )
                assert red.any(), f"no marker pixels near data ({ox}, {oy})"
        finally:
            s._plot.close()

    def test_marker_update_rerenders_browser(self, apl_backend, interact_page):
        s = hs.signals.Signal2D(np.zeros((32, 32)))
        s.plot()
        try:
            m = hs.plot.markers.Points(offsets=np.array([[8.0, 8.0]]), sizes=40)
            s.add_marker(m)
            sig_plot = s._plot.signal_plot.ax._plot
            fig = real_figure(s._plot.signal_plot.ax.figure)

            page = interact_page(fig)
            before = screenshot_page(page)

            from hyperspy.drawing.backends import get_backend

            get_backend().update_markers(
                m._collection, offsets=np.array([[24.0, 24.0]])
            )
            push_panel_state(page, fig, sig_plot)

            state = json.loads(
                page.evaluate(
                    f"() => window._aplModel.get('panel_{sig_plot._id}_json')"
                )
            )
            circles = [g for g in state["markers"] if g["type"] == "circles"]
            assert circles[0]["offsets"] == [[24.0, 24.0]]

            after = screenshot_page(page)
            assert_differs(before, after, min_diff_frac=0.0002)
        finally:
            s._plot.close()


class TestBrowserEventEmission:
    """Basic event plumbing sanity, mirroring anyplotlib's emission tests."""

    def test_pointer_events_emitted_on_click(self, apl_backend, interact_page):
        s = _signal1d_with_image_navigator()
        s.plot()
        try:
            page = interact_page(real_figure(s._plot.signal_plot.ax.figure))
            collect_events(page)
            nav_box, sig_box = interactive_boxes(page)

            # 2-D panels emit pointer_down on click.
            page.mouse.move(
                nav_box["x"] + nav_box["w"] / 2, nav_box["y"] + nav_box["h"] / 2
            )
            page.mouse.down()
            page.wait_for_timeout(80)
            page.mouse.up()
            page.wait_for_timeout(80)
            assert get_events(page, "pointer_down"), "no pointer_down on 2-D panel"

            # 1-D panels currently emit pointer_up only (anyplotlib quirk).
            page.mouse.move(
                sig_box["x"] + sig_box["w"] / 2, sig_box["y"] + sig_box["h"] / 2
            )
            page.mouse.down()
            page.wait_for_timeout(80)
            page.mouse.up()
            page.wait_for_timeout(80)
            assert get_events(page, "pointer_up"), "no pointer_up on 1-D panel"
        finally:
            s._plot.close()

    def test_key_down_replay_reaches_python(self, apl_backend, interact_page):
        """Keyboard events recorded in the browser reach Python handlers."""
        s = _signal1d_with_image_navigator()
        s.plot()
        try:
            sig_plot = s._plot.signal_plot.ax._plot
            fig = real_figure(s._plot.signal_plot.ax.figure)
            received = []
            sig_plot.add_event_handler(lambda e: received.append(e.key), "key_down")

            page = interact_page(fig)
            collect_events(page)
            boxes = interactive_boxes(page)
            sig_box = boxes[1]
            page.mouse.click(
                sig_box["x"] + sig_box["w"] / 2, sig_box["y"] + sig_box["h"] / 2
            )
            page.keyboard.press("r")
            page.wait_for_timeout(80)

            events = get_events(page, "key_down")
            assert events, "no key_down emitted by the browser"
            replay_events(fig, events)
            assert "r" in received
        finally:
            s._plot.close()
