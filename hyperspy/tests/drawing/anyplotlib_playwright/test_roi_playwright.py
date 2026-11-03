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

"""Browser tests for the ROI widgets of the anyplotlib backend.

``test_interactive_playwright`` covers the navigator pointers and the
rectangle ROI; this module covers the rest of the ROI surface — the span,
circle, line and polygon widgets — plus the marker and colour-mapping paths
that only run once a figure is rendered by a browser.

Every drag follows the same round trip as a live kernel: the browser moves
the widget, the events it emits are replayed into ``Figure._on_event``, and
the assertion is made on the *hyperspy* ROI at the far end.
"""

from __future__ import annotations

import json

import numpy as np

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


def _panel_markers(page, plot):
    """Read a panel's marker groups straight from the JS model."""
    state = page.evaluate(f"() => window._aplModel.get('panel_{plot._id}_json')")
    return json.loads(state)["markers"]


def _widget_of_type(plot, wtype):
    widgets = [w for w in plot.list_widgets() if w.get("type") == wtype]
    assert widgets, f"no {wtype!r} widget on the panel"
    return widgets[0]


def _ramp_image(n=32):
    yy, xx = np.mgrid[0:n, 0:n]
    return (xx + yy).astype(float)


def _plot_pieces(s):
    """Return ``(signal panel plot, real figure)`` for a plotted signal."""
    plot = s._plot.signal_plot.ax._plot
    return plot, real_figure(s._plot.signal_plot.ax.figure)


class TestSpanROIDrag:
    """SpanROI is anyplotlib's native range widget (``_AplSpanSelector``)."""

    def test_drag_edge_widens_roi(self, apl_backend, interact_page):
        s = hs.signals.Signal1D(np.sin(np.linspace(0.0, 2.0 * np.pi, 100)))
        s.plot()
        try:
            roi = hs.roi.SpanROI(left=20.0, right=60.0)
            roi.add_widget(s)
            sig_plot, fig = _plot_pieces(s)
            span = _widget_of_type(sig_plot, "range")

            page = interact_page(fig)
            collect_events(page)
            box = interactive_boxes(page)[0]
            xlim, ylim = sig_plot.get_xlim(), sig_plot.get_ylim()
            ymid = (ylim[0] + ylim[1]) / 2.0

            # Grab the right-hand edge and pull it 15 units further right.
            px, py = data_to_page_1d(box, xlim, ylim, roi.right, ymid)
            px_t, _ = data_to_page_1d(box, xlim, ylim, roi.right + 15.0, ymid)
            drag(page, px, py, px_t, py)

            events = get_events(page, widget_id=span.id)
            assert events, "range widget emitted no events"
            replay_events(fig, events)

            assert roi.right > 65.0, f"span edge did not follow the drag: {roi}"
            assert abs(roi.left - 20.0) < 2.0, f"left edge moved too: {roi}"
        finally:
            s._plot.close()

    def test_remove_widget_takes_the_span_off_the_panel(
        self, apl_backend, interact_page
    ):
        s = hs.signals.Signal1D(np.sin(np.linspace(0.0, 2.0 * np.pi, 100)))
        s.plot()
        try:
            roi = hs.roi.SpanROI(left=20.0, right=60.0)
            roi.add_widget(s)
            sig_plot, fig = _plot_pieces(s)
            page = interact_page(fig)
            assert _panel_widgets(page, sig_plot)

            roi.remove_widget(s)

            assert not [
                w for w in sig_plot.list_widgets() if w.get("type") == "range"
            ], "range widget outlived the ROI"
            push_panel_state(page, fig, sig_plot)
            assert not _panel_widgets(page, sig_plot), (
                "the browser still draws the removed span"
            )
        finally:
            s._plot.close()


class TestCircleROIDrag:
    """CircleROI is a native circle widget; dragging it moves its centre."""

    def test_drag_centre_moves_roi(self, apl_backend, interact_page):
        s = hs.signals.Signal2D(_ramp_image())
        s.plot()
        try:
            roi = hs.roi.CircleROI(cx=16.0, cy=16.0, r=6.0)
            roi.add_widget(s)
            sig_plot, fig = _plot_pieces(s)
            circle = _widget_of_type(sig_plot, "circle")

            page = interact_page(fig)
            collect_events(page)
            box = interactive_boxes(page)[0]
            xlim, ylim = sig_plot.get_xlim(), sig_plot.get_ylim()

            px, py = data_to_page_2d(box, xlim, ylim, circle.cx, circle.cy)
            px_t, py_t = data_to_page_2d(
                box, xlim, ylim, circle.cx - 6.0, circle.cy + 4.0
            )
            drag(page, px, py, px_t, py_t)

            events = get_events(page, widget_id=circle.id)
            assert events, "circle widget emitted no events"
            replay_events(fig, events)

            assert roi.cx < 13.0, f"circle did not follow the drag: {roi}"
            assert roi.cy > 18.0, f"circle did not follow the drag: {roi}"
            assert abs(roi.r - 6.0) < 1.0, f"a move should not resize: {roi}"
        finally:
            s._plot.close()

    def test_python_move_reaches_the_browser(self, apl_backend, interact_page):
        s = hs.signals.Signal2D(_ramp_image())
        s.plot()
        try:
            roi = hs.roi.CircleROI(cx=16.0, cy=16.0, r=6.0)
            roi.add_widget(s)
            sig_plot, fig = _plot_pieces(s)
            page = interact_page(fig)
            before = _panel_widgets(page, sig_plot)[0]

            roi.cx, roi.cy = 24.0, 8.0
            push_panel_state(page, fig, sig_plot)

            after = _panel_widgets(page, sig_plot)[0]
            assert after["cx"] > before["cx"] + 4, (after, before)
            assert after["cy"] < before["cy"] - 4, (after, before)
        finally:
            s._plot.close()


class TestLine2DROIDrag:
    """Line2DROI materialises as a two-endpoint line widget."""

    def test_drag_endpoint_moves_roi(self, apl_backend, interact_page):
        s = hs.signals.Signal2D(_ramp_image())
        s.plot()
        try:
            roi = hs.roi.Line2DROI(x1=6.0, y1=6.0, x2=24.0, y2=24.0)
            roi.add_widget(s)
            sig_plot, fig = _plot_pieces(s)
            line = _widget_of_type(sig_plot, "line")

            page = interact_page(fig)
            collect_events(page)
            box = interactive_boxes(page)[0]
            xlim, ylim = sig_plot.get_xlim(), sig_plot.get_ylim()

            px, py = data_to_page_2d(box, xlim, ylim, line.x1, line.y1)
            px_t, py_t = data_to_page_2d(box, xlim, ylim, line.x1 + 8.0, line.y1)
            drag(page, px, py, px_t, py_t)

            events = get_events(page, widget_id=line.id)
            assert events, "line widget emitted no events"
            replay_events(fig, events)

            assert roi.x1 > 10.0, f"endpoint did not follow the drag: {roi}"
            assert abs(roi.x2 - 24.0) < 2.0, f"far endpoint moved too: {roi}"
        finally:
            s._plot.close()

    def test_widget_colour_change_reaches_the_browser(self, apl_backend, interact_page):
        s = hs.signals.Signal2D(_ramp_image())
        s.plot()
        try:
            roi = hs.roi.Line2DROI(x1=6.0, y1=6.0, x2=24.0, y2=24.0)
            roi.add_widget(s, color="red")
            sig_plot, fig = _plot_pieces(s)
            page = interact_page(fig)
            before = _panel_widgets(page, sig_plot)[0]

            list(roi.widgets)[0].color = "blue"
            push_panel_state(page, fig, sig_plot)

            after = _panel_widgets(page, sig_plot)[0]
            assert after["color"] != before["color"], (after, before)
            assert after["color"] == "blue", after
        finally:
            s._plot.close()


class TestPolygonROIDrag:
    """PolygonROI drives anyplotlib's polygon widget through the façade."""

    def test_drag_vertex_updates_roi(self, apl_backend, interact_page):
        s = hs.signals.Signal2D(_ramp_image())
        s.plot()
        try:
            roi = hs.roi.PolygonROI(vertices=[(6.0, 6.0), (24.0, 6.0), (24.0, 22.0)])
            roi.add_widget(s)
            sig_plot, fig = _plot_pieces(s)
            polygon = _widget_of_type(sig_plot, "polygon")

            page = interact_page(fig)
            collect_events(page)
            box = interactive_boxes(page)[0]
            xlim, ylim = sig_plot.get_xlim(), sig_plot.get_ylim()

            vx, vy = polygon.get("vertices")[0]
            px, py = data_to_page_2d(box, xlim, ylim, vx, vy)
            px_t, py_t = data_to_page_2d(box, xlim, ylim, vx + 5.0, vy + 5.0)
            drag(page, px, py, px_t, py_t)

            events = get_events(page, widget_id=polygon.id)
            assert events, "polygon widget emitted no events"
            replay_events(fig, events)

            moved = np.asarray(roi.vertices, dtype=float)
            assert moved.shape == (3, 2), moved
            assert moved[0][0] > 8.0 or moved[0][1] > 8.0, (
                f"no vertex followed the drag: {moved}"
            )
        finally:
            s._plot.close()


class TestCalibratedImageMarkers:
    """Markers on a calibrated image are rewritten into pixel indices."""

    def _calibrated(self):
        s = hs.signals.Signal2D(_ramp_image())
        for axis, scale, offset in (
            (s.axes_manager[0], 0.5, -8.0),
            (s.axes_manager[1], 0.25, 3.0),
        ):
            axis.scale, axis.offset = scale, offset
        return s

    def test_point_offsets_become_pixel_indices(self, apl_backend, interact_page):
        s = self._calibrated()
        s.plot()
        try:
            xaxis, yaxis = s.axes_manager.signal_axes
            offsets = np.array(
                [
                    [xaxis.axis[8], yaxis.axis[4]],
                    [xaxis.axis[16], yaxis.axis[20]],
                    [xaxis.axis[24], yaxis.axis[28]],
                ]
            )
            s.add_marker(hs.plot.markers.Points(offsets=offsets))

            sig_plot, fig = _plot_pieces(s)
            page = interact_page(fig)
            state = json.loads(
                page.evaluate(
                    f"() => window._aplModel.get('panel_{sig_plot._id}_json')"
                )
            )
            groups = state["markers"]
            assert groups, f"no marker group reached the browser: {sorted(state)}"
            drawn = np.asarray(groups[0]["offsets"], dtype=float)
            # An image panel addresses markers by pixel index, not by the
            # calibrated units hyperspy holds them in.
            assert np.allclose(drawn, [[8, 4], [16, 20], [24, 28]], atol=0.5), drawn
        finally:
            s._plot.close()

    def test_circle_markers_with_sizes_render(self, apl_backend, interact_page):
        s = self._calibrated()
        s.plot()
        try:
            sig_plot, fig = _plot_pieces(s)
            page = interact_page(fig)
            before = screenshot_page(page)

            s.add_marker(
                hs.plot.markers.Circles(
                    offsets=np.array([[-4.0, 5.0], [0.0, 7.0]]),
                    sizes=np.array([3.0]),
                    facecolor="none",
                    edgecolor="red",
                )
            )
            push_panel_state(page, fig, sig_plot)

            assert_differs(before, screenshot_page(page))
        finally:
            s._plot.close()


class TestImageNorms:
    """The contrast mapping is applied browser-side, so render it."""

    def test_symlog_differs_from_linear(self, apl_backend, take_screenshot):
        data = _ramp_image() - 16.0
        linear = hs.signals.Signal2D(data)
        linear.plot()
        try:
            shot_linear = take_screenshot(
                real_figure(linear._plot.signal_plot.ax.figure)
            )
        finally:
            linear._plot.close()

        symlog = hs.signals.Signal2D(data)
        symlog.plot(norm="symlog")
        try:
            shot_symlog = take_screenshot(
                real_figure(symlog._plot.signal_plot.ax.figure)
            )
        finally:
            symlog._plot.close()

        assert_differs(shot_linear, shot_symlog)


class TestIteratingMarkers:
    """Per-navigation-index markers have to be re-pushed as the index moves."""

    def test_circle_markers_follow_the_navigation_index(
        self, apl_backend, interact_page
    ):
        s = hs.signals.Signal2D(np.ones((3, 16, 16)))
        offsets = np.empty(3, dtype=object)
        sizes = np.empty(3, dtype=object)
        for i in range(3):
            offsets[i] = np.array([[2.0 + 4 * i, 3.0 + 2 * i]])
            sizes[i] = np.array([4.0 + 4 * i])
        s.plot()
        try:
            s.add_marker(hs.plot.markers.Points(offsets=offsets, sizes=sizes))
            sig_plot, fig = _plot_pieces(s)
            page = interact_page(fig)
            first = _panel_markers(page, sig_plot)[0]

            s.axes_manager.navigation_axes[0].index = 2
            push_panel_state(page, fig, sig_plot)
            last = _panel_markers(page, sig_plot)[0]

            assert last["offsets"] != first["offsets"], (last, first)
            # Point sizes are diameters in display pixels; the circles the
            # browser draws take a radius.
            assert last["sizes"] == [6.0] and first["sizes"] == [2.0], (last, first)
        finally:
            s._plot.close()


class TestLine2DROIWidth:
    """A Line2DROI with a width draws dotted indicators either side."""

    def test_width_indicators_reach_the_browser(self, apl_backend, interact_page):
        s = hs.signals.Signal2D(_ramp_image())
        s.plot()
        try:
            roi = hs.roi.Line2DROI(x1=6.0, y1=6.0, x2=24.0, y2=24.0, linewidth=4.0)
            roi.add_widget(s)
            sig_plot, fig = _plot_pieces(s)
            page = interact_page(fig)

            # The main segment is an interactive widget, the indicators are a
            # 'lines' marker group.
            assert _panel_widgets(page, sig_plot), "no line widget in the browser"
            groups = _panel_markers(page, sig_plot)
            assert [g for g in groups if g["type"] == "lines"], (
                f"width indicators missing: {groups}"
            )

            before = screenshot_page(page)
            list(roi.widgets)[0].color = "blue"
            push_panel_state(page, fig, sig_plot)
            assert_differs(before, screenshot_page(page))
        finally:
            s._plot.close()
