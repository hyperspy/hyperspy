# -*- coding: utf-8 -*-
"""Tests for the anyplotlib plotting backend.

These tests exercise AnyplotlibBackend directly (unit tests) and via
hs.Signal1D/Signal2D.plot() (integration tests).  They are skipped when
anyplotlib is not installed.
"""

import sys

import numpy as np
import pytest

anyplotlib = pytest.importorskip("anyplotlib")

import hyperspy.api as hs  # noqa: E402
from hyperspy.drawing.backends.anyplotlib import AnyplotlibBackend  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend():
    return AnyplotlibBackend()


@pytest.fixture()
def fig_ax(backend):
    fig = backend.create_figure()
    ax = backend.create_axes(fig)
    return fig, ax


@pytest.fixture(autouse=True)
def use_anyplotlib(monkeypatch):
    """Switch to anyplotlib backend for every test in this module."""
    from hyperspy.defaults_parser import preferences

    old = preferences.Plot.backend
    preferences.Plot.backend = "anyplotlib"
    yield
    preferences.Plot.backend = old


# ---------------------------------------------------------------------------
# Unit tests — AnyplotlibBackend directly
# ---------------------------------------------------------------------------


class TestFigureLifecycle:
    def test_create_figure_returns_object(self, backend):
        fig = backend.create_figure()
        assert fig is not None

    def test_create_figure_stores_hspy_ax(self, backend):
        fig = backend.create_figure()
        assert hasattr(fig, "_hspy_ax")
        assert fig._hspy_ax is not None

    def test_create_axes_returns_hspy_ax(self, backend):
        fig = backend.create_figure()
        ax = backend.create_axes(fig)
        assert ax is fig._hspy_ax

    def test_ax_has_figure_attribute(self, backend):
        """ax.figure must be set so widget.py code doesn't crash."""
        fig = backend.create_figure()
        ax = backend.create_axes(fig)
        assert hasattr(ax, "figure")
        assert ax.figure is fig

    def test_supports_blit_is_false(self, backend, fig_ax):
        fig, _ = fig_ax
        assert backend.supports_blit(fig) is False

    def test_draw_idle_is_noop(self, backend, fig_ax):
        fig, _ = fig_ax
        backend.draw_idle(fig)  # must not raise

    def test_close_figure(self, backend):
        closed = []
        fig = backend.create_figure(on_close=lambda: closed.append(True))
        backend.close_figure(fig)
        assert closed == [True]


class TestAxesSetup:
    def test_set_xlabel_after_plot(self, backend, fig_ax):
        _, ax = fig_ax
        h = backend.plot_line(ax, np.arange(10, dtype=float), np.zeros(10))
        backend.set_xlabel(ax, "Energy (eV)")
        # Plot1D stores xlabel in "units", ylabel in "y_units"
        assert h._state["units"] == "Energy (eV)"

    def test_set_ylabel_after_plot(self, backend, fig_ax):
        _, ax = fig_ax
        h = backend.plot_line(ax, np.arange(10, dtype=float), np.zeros(10))
        backend.set_ylabel(ax, "Counts")
        assert h._state["y_units"] == "Counts"

    def test_set_title_after_plot(self, backend, fig_ax):
        _, ax = fig_ax
        h = backend.plot_line(ax, np.arange(10, dtype=float), np.zeros(10))
        backend.set_title(ax, "My signal")
        assert h._state["title"] == "My signal"

    def test_labels_buffered_before_plot_applied_at_plot_time(self, backend, fig_ax):
        """Labels set before plot_line must be applied once the plot is created."""
        _, ax = fig_ax
        assert ax._plot is None
        backend.set_xlabel(ax, "eV")
        backend.set_ylabel(ax, "I")
        backend.set_title(ax, "Spec")
        h = backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        assert h._state["units"] == "eV"
        assert h._state["y_units"] == "I"
        assert h._state["title"] == "Spec"

    def test_set_ylim(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(10, dtype=float), np.zeros(10))
        backend.set_ylim(ax, -1.0, 1.0)  # must not raise

    def test_set_xlim(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(10, dtype=float), np.zeros(10))
        backend.set_xlim(ax, 2.0, 8.0)  # must not raise

    def test_get_ylim_returns_tuple(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(10, dtype=float), np.linspace(-1, 1, 10))
        ylim = backend.get_ylim(ax)
        assert len(ylim) == 2
        assert ylim[0] < ylim[1]

    def test_get_ylim_before_plot_returns_defaults(self, backend, fig_ax):
        _, ax = fig_ax
        ylim = backend.get_ylim(ax)
        assert ylim == (0.0, 1.0)

    def test_get_xbound_returns_correct_range(self, backend, fig_ax):
        _, ax = fig_ax
        x = np.linspace(2.0, 10.0, 50)
        backend.plot_line(ax, x, np.zeros(50))
        xb = backend.get_xbound(ax)
        assert xb[0] == pytest.approx(2.0)
        assert xb[1] == pytest.approx(10.0)

    def test_set_axis_off(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        backend.set_axis_off(ax)
        assert ax._plot._state["axis_visible"] is False


class TestLine1D:
    def test_plot_line_returns_plot1d(self, backend, fig_ax):
        from anyplotlib.plot1d import Plot1D

        _, ax = fig_ax
        h = backend.plot_line(ax, np.arange(10, dtype=float), np.zeros(10))
        assert isinstance(h, Plot1D)

    def test_plot_line_attaches_to_ax(self, backend, fig_ax):
        _, ax = fig_ax
        assert ax._plot is None
        backend.plot_line(ax, np.arange(10, dtype=float), np.zeros(10))
        assert ax._plot is not None

    def test_plot_line_with_none_x(self, backend, fig_ax):
        """x=None means default 0…N-1 x axis."""
        _, ax = fig_ax
        h = backend.plot_line(ax, None, np.zeros(20))
        assert h is not None

    def test_update_line(self, backend, fig_ax):
        _, ax = fig_ax
        x = np.arange(10, dtype=float)
        h = backend.plot_line(ax, x, np.zeros(10))
        new_y = np.ones(10) * 3.14
        backend.update_line(h, x, new_y)
        np.testing.assert_allclose(np.asarray(h._state["data"]), new_y)

    def test_remove_line_does_not_raise(self, backend, fig_ax):
        _, ax = fig_ax
        h = backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        backend.remove_line(ax, h)  # should be a no-op, not raise

    def test_line_get_xdata(self, backend, fig_ax):
        _, ax = fig_ax
        x = np.linspace(0.5, 4.5, 40)
        h = backend.plot_line(ax, x, np.zeros(40))
        xdata = backend.line_get_xdata(h)
        np.testing.assert_allclose(xdata, x)

    def test_line_get_color(self, backend, fig_ax):
        _, ax = fig_ax
        h = backend.plot_line(
            ax, np.arange(5, dtype=float), np.zeros(5), color="#aabbcc"
        )
        assert backend.line_get_color(h) == "#aabbcc"


class TestLinePropTranslation:
    """hyperspy speaks matplotlib's line vocabulary; anyplotlib does not.

    See ``AnyplotlibBackend._norm_line_props``.
    """

    def test_scatter_linestyle_none_is_passed_through(self, backend):
        """0.5.0 renders a real markers-only line; we used to fake it as solid."""
        out = backend._norm_line_props(
            {"linestyle": "None", "marker": "o", "markersize": 1}
        )
        assert out["linestyle"] == "none"
        assert out["marker"] == "o"
        assert out["markersize"] == 1.0

    def test_marker_none_is_normalised(self, backend):
        assert backend._norm_line_props({"marker": "None"})["marker"] == "none"
        assert backend._norm_line_props({"marker": None})["marker"] == "none"

    def test_unknown_marker_falls_back_to_circle(self, backend):
        assert backend._norm_line_props({"marker": "1"})["marker"] == "o"

    def test_drawstyle_steps_wins_over_linestyle(self, backend):
        out = backend._norm_line_props({"linestyle": "-", "drawstyle": "steps-mid"})
        assert out["linestyle"] == "step-mid"

    def test_markeredgecolor_stands_in_for_color(self, backend):
        # Signal1DLine.color moves the colour to markeredgecolor for scatter.
        out = backend._norm_line_props({"markeredgecolor": "#123456"})
        assert out["color"] == "#123456"

    def test_explicit_color_wins_over_markeredgecolor(self, backend):
        out = backend._norm_line_props(
            {"color": "#aaaaaa", "markeredgecolor": "#123456"}
        )
        assert out["color"] == "#aaaaaa"

    def test_absent_keys_are_not_invented(self, backend):
        assert backend._norm_line_props({}) == {}

    def test_scatter_line_plots_without_raising(self, backend, fig_ax):
        """The end-to-end case: a model's data line is drawn as a scatter."""
        _, ax = fig_ax
        h = backend.plot_line(
            ax,
            np.arange(5, dtype=float),
            np.zeros(5),
            linestyle="None",
            marker="o",
            markersize=1,
            markeredgecolor="#ff0000",
        )
        assert h._state["line_marker"] == "o"
        assert backend.line_get_color(h) == "#ff0000"

    def test_set_line_props_scatter_on_primary_plot(self, backend, fig_ax):
        _, ax = fig_ax
        h = backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        backend.set_line_props(h, linestyle="None", marker="o", markersize=2)
        assert h._state["line_linestyle"] == "none"
        assert h._state["line_marker"] == "o"
        assert h._state["line_markersize"] == pytest.approx(2.0)


class TestOverlayLine1D:
    """A second plot_line() call on the same ax overlays a Line1D."""

    def test_overlay_returns_line1d(self, backend, fig_ax):
        from anyplotlib.plot1d._plot1d import Line1D

        _, ax = fig_ax
        backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        overlay = backend.plot_line(
            ax, np.arange(5, dtype=float), np.ones(5), color="#123456"
        )
        assert isinstance(overlay, Line1D)

    def test_overlay_get_color_and_linewidth(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        overlay = backend.plot_line(
            ax,
            np.arange(5, dtype=float),
            np.ones(5),
            color="#123456",
            linewidth=3.0,
        )
        assert backend.line_get_color(overlay) == "#123456"
        assert backend.line_get_linewidth(overlay) == pytest.approx(3.0)

    def test_set_line_props_updates_overlay(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        overlay = backend.plot_line(ax, np.arange(5, dtype=float), np.ones(5))
        backend.set_line_props(
            overlay, color="#abcdef", linewidth=4.0, linestyle="dashed", alpha=0.5
        )
        assert backend.line_get_color(overlay) == "#abcdef"
        assert backend.line_get_linewidth(overlay) == pytest.approx(4.0)
        assert overlay.linestyle == "dashed"
        assert overlay.alpha == pytest.approx(0.5)

    def test_remove_overlay_line(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        overlay = backend.plot_line(ax, np.arange(5, dtype=float), np.ones(5))
        backend.remove_line(ax, overlay)  # must not raise
        with pytest.raises(KeyError):
            overlay.color  # entry no longer exists


class TestImage2D:
    def test_plot_image_returns_plot2d(self, backend, fig_ax):
        from anyplotlib.plot2d import Plot2D

        _, ax = fig_ax
        h = backend.plot_image(ax, np.random.rand(16, 16))
        assert isinstance(h, Plot2D)

    def test_plot_image_attaches_to_ax(self, backend, fig_ax):
        _, ax = fig_ax
        assert ax._plot is None
        backend.plot_image(ax, np.zeros((8, 8)))
        assert ax._plot is not None

    def test_image_labels_buffered_then_applied(self, backend, fig_ax):
        _, ax = fig_ax
        backend.set_xlabel(ax, "x-axis")
        backend.set_title(ax, "Image title")
        h = backend.plot_image(ax, np.zeros((8, 8)))
        assert h._state["x_label"] == "x-axis"
        assert h._state["title"] == "Image title"

    def test_image_set_data(self, backend, fig_ax):
        _, ax = fig_ax
        h = backend.plot_image(ax, np.zeros((8, 8)))
        new_data = np.ones((8, 8)) * 0.5
        backend.image_set_data(h, new_data)  # must not raise

    def test_image_set_clim(self, backend, fig_ax):
        _, ax = fig_ax
        h = backend.plot_image(ax, np.zeros((8, 8)))
        backend.image_set_clim(h, 0.1, 0.9)
        assert h._state["display_min"] == pytest.approx(0.1)
        assert h._state["display_max"] == pytest.approx(0.9)

    def test_get_image_handle(self, backend, fig_ax):
        _, ax = fig_ax
        h = backend.plot_image(ax, np.zeros((8, 8)))
        assert backend.get_image_handle(ax) is h

    def test_get_image_handle_before_plot_returns_none(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.get_image_handle(ax) is None


class TestColorbar:
    def test_add_colorbar_makes_visible(self, backend, fig_ax):
        fig, ax = fig_ax
        h = backend.plot_image(ax, np.zeros((8, 8)))
        assert h._state["show_colorbar"] is False
        backend.add_colorbar(fig, h, ax)
        assert h._state["show_colorbar"] is True

    def test_colorbar_set_label(self, backend, fig_ax):
        fig, ax = fig_ax
        h = backend.plot_image(ax, np.zeros((8, 8)))
        cb = backend.add_colorbar(fig, h, ax)
        backend.colorbar_set_label(cb, "Intensity (a.u.)")
        assert h._state["colorbar_label"] == "Intensity (a.u.)"

    def test_colorbar_remove(self, backend, fig_ax):
        fig, ax = fig_ax
        h = backend.plot_image(ax, np.zeros((8, 8)))
        cb = backend.add_colorbar(fig, h, ax)
        backend.colorbar_remove(cb)
        assert h._state["show_colorbar"] is False


class TestLinePointer:
    def test_create_line_pointer_x(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(20, dtype=float), np.zeros(20))
        w = backend.create_line_pointer(ax, "x", 5.0, color="red")
        assert w is not None

    def test_update_line_pointer_x(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(20, dtype=float), np.zeros(20))
        w = backend.create_line_pointer(ax, "x", 5.0)
        backend.update_line_pointer(w, "x", 10.0)
        assert w.x == pytest.approx(10.0)

    def test_create_line_pointer_before_plot_raises(self, backend, fig_ax):
        _, ax = fig_ax
        with pytest.raises(RuntimeError, match="no plot"):
            backend.create_line_pointer(ax, "x", 0.0)


class TestCirclePointer:
    """``create_circle_pointer`` — one native widget for circle *and* annulus."""

    def test_circle_when_no_inner_radius(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((10, 10)))
        (handle,) = backend.create_circle_pointer(ax, 5.0, 5.0, 2.0)
        assert handle.get("type") == "circle"
        assert handle.get("r") == pytest.approx(2.0)

    def test_annulus_when_inner_radius(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((10, 10)))
        (handle,) = backend.create_circle_pointer(ax, 5.0, 5.0, 4.0, 2.0)
        assert handle.get("type") == "annular"
        assert handle.get("r_inner") == pytest.approx(2.0)

    def test_update_moves_and_resizes(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((10, 10)))
        handles = backend.create_circle_pointer(ax, 5.0, 5.0, 2.0)
        handles = backend.update_circle_pointer(ax, handles, 6.0, 7.0, 3.0)
        assert handles[0].get("cx") == pytest.approx(6.0)
        assert handles[0].get("r") == pytest.approx(3.0)

    def test_update_swaps_circle_for_annulus(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((10, 10)))
        handles = backend.create_circle_pointer(ax, 5.0, 5.0, 4.0)
        assert handles[0].get("type") == "circle"
        swapped = backend.update_circle_pointer(ax, handles, 5.0, 5.0, 4.0, 2.0)
        assert swapped[0].get("type") == "annular"
        # ...and back again
        back = backend.update_circle_pointer(ax, swapped, 5.0, 5.0, 4.0, 0.0)
        assert back[0].get("type") == "circle"

    def test_raises_without_plot(self, backend, fig_ax):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        _, ax = fig_ax
        with pytest.raises(BackendCapabilityError):
            backend.create_circle_pointer(ax, 0.0, 0.0, 1.0)


class TestSpanSelector:
    """``create_span_selector`` — the matplotlib SpanSelector façade."""

    @pytest.fixture()
    def span(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(20, dtype=float), np.zeros(20))
        return backend.create_span_selector(ax, props={"color": "red"})

    def test_extents_roundtrip(self, span):
        span.extents = (2.0, 8.0)
        assert span.extents == pytest.approx((2.0, 8.0))

    def test_extents_are_ordered(self, span):
        span.extents = (8.0, 2.0)
        assert span.extents == pytest.approx((2.0, 8.0))

    def test_snap_values_reach_the_widget(self, span):
        """Snapping is the widget's job as of 0.5.0.

        It has to happen inside the JS drag: snapping in Python after the
        event moves an edge the user is still holding, which reads as the
        selection fighting back.  So the façade forwards the values rather
        than rounding ``extents`` itself.
        """
        span.snap_values = np.array([0.0, 5.0, 10.0])
        assert span._widget.get("snap_values") == [0.0, 5.0, 10.0]
        span.extents = (1.2, 8.9)
        assert span.extents == pytest.approx((1.2, 8.9))

    def test_artists_is_the_widget(self, span):
        assert len(span.artists) == 1

    def test_set_props_changes_colour(self, span):
        span.set_props(color="lime")
        assert span._widget.get("color") == "lime"

    def test_connect_and_disconnect_events(self, span):
        calls = []
        assert span.connect_event("motion_notify_event", calls.append) is not None
        # Unknown event names are ignored rather than raising.
        assert span.connect_event("button_press_event", calls.append) is None
        span.disconnect_events()
        assert span._handlers == []

    def test_clear_removes_widget(self, span):
        span.clear()
        assert span._widget.id not in span._plot._widgets


class TestPolygonSelector:
    """``create_polygon_selector`` — the matplotlib PolygonSelector façade."""

    @pytest.fixture()
    def selector(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((10, 10)))
        return backend.create_polygon_selector(ax, props={"color": "red"})

    def test_starts_empty(self, selector):
        assert selector.verts == []

    def test_too_few_vertices_stay_buffered(self, selector):
        selector.verts = [(1, 1), (2, 2)]
        assert selector._widget is None
        assert len(selector.verts) == 2

    def test_three_vertices_create_widget(self, selector):
        selector.verts = [(1, 1), (5, 1), (3, 4)]
        assert selector._widget is not None
        assert selector.verts == [(1, 1), (5, 1), (3, 4)]

    def test_updating_existing_widget(self, selector):
        selector.verts = [(1, 1), (5, 1), (3, 4)]
        widget = selector._widget
        selector.verts = [(0, 0), (6, 0), (3, 5)]
        assert selector._widget is widget  # reused, not recreated
        assert selector.verts[0] == (0, 0)

    def test_set_props_before_and_after_creation(self, selector):
        selector.set_props(color="lime")
        selector.verts = [(1, 1), (5, 1), (3, 4)]
        assert selector._widget.get("color") == "lime"
        selector.set_props(color="cyan")
        assert selector._widget.get("color") == "cyan"

    def test_clear_removes_widget(self, selector):
        selector.verts = [(1, 1), (5, 1), (3, 4)]
        selector.clear()
        assert selector._widget is None
        assert selector.verts == []


class TestCalibratedWidgetCoordinates:
    """Overlay widgets are positioned in image pixels, HyperSpy in data units.

    The two coincide only for an uncalibrated axis (scale 1, offset 0). On a
    calibrated image the backend must convert, or the widget is drawn at the
    wrong place and size — a 0.015 nm/px image put a 3.84 nm ROI at pixel 1.9
    with a width of 3.8 px, a smudge in the corner that could not be grabbed.
    """

    # 8 px spanning 0-16 in data units, i.e. 2.0 per pixel.
    EXTENT = (0.0, 16.0, 16.0, 0.0)

    @pytest.fixture()
    def image_ax(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((8, 8)), extent=self.EXTENT)
        return ax

    def test_pixel_centres_span_the_extent(self, backend, image_ax):
        """Axis arrays hold pixel centres, half a pixel inside each edge."""
        x_axis = image_ax._plot._state["x_axis"]
        assert x_axis[0] == pytest.approx(1.0)  # 0 + 2/2
        assert x_axis[-1] == pytest.approx(15.0)  # 16 - 2/2
        assert x_axis[1] - x_axis[0] == pytest.approx(2.0)

    def test_rect_pointer_converts_to_pixels(self, backend, image_ax):
        handle = backend.create_rect_pointer(image_ax, 4.0, 4.0, 8.0, 8.0)
        # data 4.0 is the centre of pixel 1.5; 8.0 wide is 4 pixels.
        assert handle.get("x") == pytest.approx(1.5)
        assert handle.get("w") == pytest.approx(4.0)

    def test_rect_pointer_update_converts_to_pixels(self, backend, image_ax):
        handle = backend.create_rect_pointer(image_ax, 4.0, 4.0, 8.0, 8.0)
        backend.update_rect_pointer(handle, 6.0, 6.0, 4.0, 4.0)
        assert handle.get("x") == pytest.approx(2.5)
        assert handle.get("w") == pytest.approx(2.0)

    def test_drag_reports_data_units(self, backend, image_ax):
        """A drag must arrive back in HyperSpy's units, not pixels."""
        handle = backend.create_rect_pointer(image_ax, 4.0, 4.0, 8.0, 8.0)
        seen = []
        backend.connect_widget_drag(handle, lambda *args: seen.append(args))
        handle.set(x=2.5, y=2.5, w=2.0, h=2.0)
        assert seen, "drag callback never fired"
        x, y, w, h = seen[-1]
        assert x == pytest.approx(6.0)
        assert w == pytest.approx(4.0)

    def test_circle_pointer_converts_to_pixels(self, backend, image_ax):
        (handle,) = backend.create_circle_pointer(image_ax, 8.0, 8.0, 4.0)
        assert handle.get("cx") == pytest.approx(3.5)
        assert handle.get("r") == pytest.approx(2.0)

    def test_polygon_selector_round_trips_data_units(self, backend, image_ax):
        selector = backend.create_polygon_selector(image_ax, props={"color": "red"})
        selector.verts = [(4.0, 4.0), (12.0, 4.0), (8.0, 12.0)]
        stored = selector._widget.get("vertices")
        assert stored[0] == pytest.approx([1.5, 1.5])
        # Reading back returns what was written, in data units.
        assert selector.verts[0] == pytest.approx((4.0, 4.0))

    def test_polygon_edit_reaches_onselect(self, backend, image_ax):
        """Dragging a polygon must reach the ROI, not just the JS widget."""
        seen = []
        selector = backend.create_polygon_selector(
            image_ax, props={"color": "red"}, onselect=seen.append
        )
        selector.verts = [(4.0, 4.0), (12.0, 4.0), (8.0, 12.0)]
        selector._widget.set(vertices=[[0.5, 0.5], [5.5, 0.5], [3.5, 5.5]])
        assert seen, "onselect never fired for a polygon edit"
        assert seen[-1][0] == pytest.approx((2.0, 2.0))

    def test_uncalibrated_axes_are_untouched(self, backend, fig_ax):
        """scale 1 / offset 0: pixel and data coordinates already agree."""
        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((10, 10)))
        handle = backend.create_rect_pointer(ax, 2.0, 3.0, 4.0, 5.0)
        assert handle.get("x") == pytest.approx(2.0)
        assert handle.get("w") == pytest.approx(4.0)


class TestEvents:
    def test_connect_disconnect_no_crash(self, backend, fig_ax):
        fig, ax = fig_ax
        backend.plot_line(ax, np.arange(10, dtype=float), np.zeros(10))
        cid = backend.connect_mouse_move(ax, lambda e: None)
        backend.disconnect_event(ax, cid)  # must not raise

    def test_connect_before_plot_returns_none(self, backend, fig_ax):
        _, ax = fig_ax
        cid = backend.connect_mouse_move(ax, lambda e: None)
        assert cid is None


# ---------------------------------------------------------------------------
# Integration tests — hs.Signal1D / Signal2D with anyplotlib backend
# ---------------------------------------------------------------------------


class TestSignal1DPlot:
    def test_plot_no_crash(self):
        s = hs.signals.Signal1D(np.random.rand(100))
        s.plot()
        s._plot.close()

    def test_plot_axes_created(self):
        s = hs.signals.Signal1D(np.random.rand(100))
        s.plot()
        ax = s._plot.signal_plot.ax
        assert ax is not None
        assert ax._plot is not None
        s._plot.close()

    def test_plot_labels_applied(self):
        s = hs.signals.Signal1D(np.random.rand(100))
        s.axes_manager[-1].name = "Energy"
        s.axes_manager[-1].units = "eV"
        s.plot()
        ax = s._plot.signal_plot.ax
        # Plot1D stores xlabel in "units" key
        x_label = ax._plot._state["units"]
        assert "Energy" in x_label or "eV" in x_label
        s._plot.close()

    def test_plot_multidim_navigate(self):
        """Navigating a multidimensional signal must not crash."""
        s = hs.signals.Signal1D(np.random.rand(5, 100))
        s.plot()
        s.axes_manager[0].index = 2
        s.axes_manager[0].index = 4
        s._plot.close()

    def test_plot_update_no_crash(self):
        s = hs.signals.Signal1D(np.random.rand(3, 50))
        s.plot()
        # Simulate data update
        s.axes_manager[0].index = 1
        s._plot.close()


class TestSignal2DPlot:
    def test_plot_no_crash(self):
        s = hs.signals.Signal2D(np.random.rand(32, 32))
        s.plot()
        s._plot.close()

    def test_plot_image_attached(self):
        s = hs.signals.Signal2D(np.random.rand(16, 16))
        s.plot()
        from anyplotlib.plot2d import Plot2D

        ax = s._plot.signal_plot.ax
        # Use backend.get_image_handle: ax._plot may be overwritten by overlay
        # artists (e.g. scalebar), so we rely on the explicit image reference.
        from hyperspy.drawing.backends import get_backend

        assert isinstance(get_backend().get_image_handle(ax), Plot2D)
        s._plot.close()

    def test_plot_multidim_navigate(self):
        s = hs.signals.Signal2D(np.random.rand(4, 16, 16))
        s.plot()
        s.axes_manager[0].index = 2
        s._plot.close()


# ---------------------------------------------------------------------------
# Combined figure layout tests
# ---------------------------------------------------------------------------


class TestCombinedFigurePanels:
    def test_returns_two_proxies(self, backend):
        from hyperspy.drawing.backends.anyplotlib import _AplFigureProxy

        nav_proxy, signal_proxy = backend.create_combined_figure_panels()
        assert isinstance(nav_proxy, _AplFigureProxy)
        assert isinstance(signal_proxy, _AplFigureProxy)

    def test_proxies_share_real_figure(self, backend):
        nav_proxy, signal_proxy = backend.create_combined_figure_panels()
        assert nav_proxy._real_fig is signal_proxy._real_fig

    def test_proxies_have_distinct_axes(self, backend):
        nav_proxy, signal_proxy = backend.create_combined_figure_panels()
        assert nav_proxy._hspy_ax is not signal_proxy._hspy_ax

    def test_create_figure_adopts_proxy(self, backend):
        nav_proxy, signal_proxy = backend.create_combined_figure_panels()
        result = backend.create_figure(fig=signal_proxy, on_close=None)
        assert result is signal_proxy

    def test_create_figure_adopts_proxy_stores_on_close(self, backend):
        nav_proxy, signal_proxy = backend.create_combined_figure_panels()
        sentinel = object()
        backend.create_figure(fig=signal_proxy, on_close=lambda: sentinel)
        assert signal_proxy._hspy_on_close is not None

    def test_create_axes_from_proxy(self, backend):
        nav_proxy, signal_proxy = backend.create_combined_figure_panels()
        backend.create_figure(fig=signal_proxy)
        ax = backend.create_axes(signal_proxy)
        assert ax is signal_proxy._hspy_ax

    def test_close_proxy_calls_callback(self, backend):
        nav_proxy, signal_proxy = backend.create_combined_figure_panels()
        called = []
        signal_proxy._hspy_on_close = lambda: called.append(1)
        backend.close_figure(signal_proxy)
        assert called == [1]
        # Second call is a no-op (callback cleared)
        backend.close_figure(signal_proxy)
        assert called == [1]

    def test_draw_idle_proxy_no_crash(self, backend):
        nav_proxy, signal_proxy = backend.create_combined_figure_panels()
        backend.draw_idle(signal_proxy)  # IPython not available → no-op
        backend.draw_idle(nav_proxy)

    def test_draw_idle_real_fig_no_crash(self, backend):
        nav_proxy, _ = backend.create_combined_figure_panels()
        backend.draw_idle(nav_proxy._real_fig)

    def test_panel_countdown_waits_for_both(self, backend):
        """First draw_idle call should NOT display; second one should."""
        nav_proxy, signal_proxy = backend.create_combined_figure_panels()
        # After nav draws, countdown goes 2→1 — still waiting.
        backend.draw_idle(nav_proxy)
        assert nav_proxy._real_fig._hspy_panels_remaining == 1
        assert not getattr(nav_proxy._real_fig, "_hspy_displayed", False)
        # After signal draws, countdown reaches 0 — (IPython absent, no real display).
        backend.draw_idle(signal_proxy)
        assert signal_proxy._real_fig._hspy_panels_remaining == 0

    def test_ensure_displayed_forces_display(self, backend):
        nav_proxy, signal_proxy = backend.create_combined_figure_panels()
        # Only nav draws (simulate slider navigator → signal draws, nav skips).
        backend.draw_idle(nav_proxy)
        assert nav_proxy._real_fig._hspy_panels_remaining == 1
        # ensure_displayed clears countdown.
        backend.ensure_displayed(signal_proxy)
        assert signal_proxy._real_fig._hspy_panels_remaining == 0

    def test_draw_idle_single_panel_immediate(self, backend):
        """Single-panel (no combined) figure should display on first draw_idle."""
        fig = backend.create_figure()
        # No _hspy_panels_remaining set → should not block.
        assert getattr(fig, "_hspy_panels_remaining", 0) == 0


class TestNativeVlineWidget:
    """Native anyplotlib vline widget for 1D-navigator navigation.

    Signal2D(n, h, w) sums the 2D signal axes to produce a 1D navigator
    (shape (n,)), which triggers VerticalLineWidget as the pointer.
    """

    def test_vline_created_on_1d_nav(self):
        """VerticalLineWidget attaches a native vline on anyplotlib 1D navigator."""
        from anyplotlib.widgets._widgets1d import VLineWidget as AplVLine

        from hyperspy.drawing._widgets.vertical_line import VerticalLineWidget

        s = hs.signals.Signal2D(np.random.rand(4, 8, 8))
        s.plot()
        pointer = s._plot.pointer
        assert isinstance(pointer, VerticalLineWidget), (
            f"expected VerticalLineWidget, got {type(pointer).__name__}"
        )
        assert pointer.patch, "VerticalLineWidget should have a patch handle"
        assert isinstance(pointer.patch[0], AplVLine)
        s._plot.close()

    def test_navigate_updates_vline_position(self):
        """Changing navigation index moves the native vline to the new position."""
        s = hs.signals.Signal2D(np.random.rand(5, 8, 8))
        s.plot()
        pointer = s._plot.pointer
        nav_ax = s.axes_manager.navigation_axes[0]
        nav_ax.index = 2
        expected_x = float(nav_ax.value)
        assert abs(pointer.patch[0].x - expected_x) < 1e-9
        s._plot.close()

    def test_position_set_from_drag_updates_nav(self):
        """Simulated drag updates the axes_manager navigation index."""
        s = hs.signals.Signal2D(np.random.rand(5, 8, 8))
        s.plot()
        pointer = s._plot.pointer
        nav_ax = s.axes_manager.navigation_axes[0]
        target_value = nav_ax.axis[3]  # fourth grid point
        pointer.position = (target_value,)
        assert nav_ax.index == 3
        s._plot.close()

    def test_native_vline_drag_updates_nav(self):
        """Dragging the anyplotlib VLineWidget (via set()) updates nav index."""
        s = hs.signals.Signal2D(np.random.rand(5, 8, 8))
        s.plot()
        pointer = s._plot.pointer
        nav_ax = s.axes_manager.navigation_axes[0]
        # Simulate JS drag by calling widget.set() with the new x coordinate
        pointer.patch[0].set(x=float(nav_ax.axis[3]))
        assert nav_ax.index == 3
        s._plot.close()


class TestNativeCrosshairWidget:
    """Native anyplotlib widgets for 2D image navigator navigation."""

    def test_hline_widget_uses_a_real_rule(self):
        """HorizontalLineWidget attaches a native hline on an anyplotlib navigator.

        Signal1D(n, length) has a 2D image navigator (full spectrum stack);
        the horizontal line marks the current navigation row (y-axis).

        This used to be faked with a crosshair pinned at cx=0, which left a
        spurious full-height rule down the left edge of the navigator.
        anyplotlib 0.5.0 has a real ``hline`` kind, so the pointer is one now.
        """
        from anyplotlib.widgets import HLineWidget as AplHLine

        from hyperspy.drawing._widgets.horizontal_line import HorizontalLineWidget

        s = hs.signals.Signal1D(np.random.rand(4, 50))
        s.plot()
        pointer = s._plot.pointer
        assert isinstance(pointer, HorizontalLineWidget), (
            f"expected HorizontalLineWidget, got {type(pointer).__name__}"
        )
        assert pointer.patch, "HorizontalLineWidget should have a patch handle"
        assert isinstance(pointer.patch[0], AplHLine), (
            f"expected a native hline, got {type(pointer.patch[0]).__name__}"
        )
        s._plot.close()

    def test_navigate_updates_hline_y(self):
        """Changing navigation index moves the rule to the new row value."""
        s = hs.signals.Signal1D(np.random.rand(5, 50))
        s.plot()
        pointer = s._plot.pointer
        nav_ax = s.axes_manager.navigation_axes[0]
        nav_ax.index = 3
        assert abs(pointer.patch[0].y - nav_ax.value) < 1e-9
        s._plot.close()

    def test_hline_drag_updates_nav(self):
        """Dragging the rule (simulated via set()) updates nav index."""
        s = hs.signals.Signal1D(np.random.rand(5, 50))
        s.plot()
        pointer = s._plot.pointer
        nav_ax = s.axes_manager.navigation_axes[0]
        pointer.patch[0].set(y=float(nav_ax.axis[2]))
        assert nav_ax.index == 2
        s._plot.close()

    def test_crosshair_created_for_square_widget(self):
        """SquareWidget attaches a native crosshair on anyplotlib 2D navigator.

        Signal1D(n, m, length) with 2D navigation uses SquareWidget.
        """
        from anyplotlib.widgets._widgets2d import CrosshairWidget as AplCrosshair

        from hyperspy.drawing._widgets.rectangles import SquareWidget

        s = hs.signals.Signal1D(np.random.rand(4, 4, 50))
        s.plot()
        pointer = s._plot.pointer
        assert isinstance(pointer, SquareWidget), (
            f"expected SquareWidget, got {type(pointer).__name__}"
        )
        assert pointer.patch, "SquareWidget should have a patch handle"
        assert isinstance(pointer.patch[0], AplCrosshair)
        s._plot.close()

    def test_navigate_updates_square_crosshair(self):
        """Changing both navigation axes moves the crosshair to the correct position."""
        s = hs.signals.Signal1D(np.random.rand(5, 5, 50))
        s.plot()
        pointer = s._plot.pointer
        nav0 = s.axes_manager.navigation_axes[0]
        nav1 = s.axes_manager.navigation_axes[1]
        nav0.index = 2
        nav1.index = 3
        assert abs(pointer.patch[0].cx - nav0.value) < 1e-9
        assert abs(pointer.patch[0].cy - nav1.value) < 1e-9
        s._plot.close()

    def test_square_drag_updates_nav(self):
        """Dragging the crosshair (simulated via set()) updates both nav indices."""
        s = hs.signals.Signal1D(np.random.rand(5, 5, 50))
        s.plot()
        pointer = s._plot.pointer
        nav0 = s.axes_manager.navigation_axes[0]
        nav1 = s.axes_manager.navigation_axes[1]
        pointer.patch[0].set(cx=float(nav0.axis[1]), cy=float(nav1.axis[3]))
        assert nav0.index == 1
        assert nav1.index == 3
        s._plot.close()


class TestCombinedFigureIntegration:
    """Integration: multidim signals use one shared anyplotlib Figure."""

    def test_1d_signal_with_nav_uses_combined(self):
        from hyperspy.drawing.backends.anyplotlib import _AplFigureProxy

        s = hs.signals.Signal1D(np.random.rand(5, 100))
        s.plot()
        # Both signal and navigator figures should be proxies sharing one fig
        sig_fig = s._plot.signal_plot.figure
        assert isinstance(sig_fig, _AplFigureProxy), (
            f"expected _AplFigureProxy, got {type(sig_fig)}"
        )
        nav_fig = s._plot.navigator_plot.figure
        assert isinstance(nav_fig, _AplFigureProxy)
        assert sig_fig._real_fig is nav_fig._real_fig
        s._plot.close()

    def test_navigate_combined_no_crash(self):
        s = hs.signals.Signal1D(np.random.rand(4, 3, 50))
        s.plot()
        s.axes_manager[0].index = 1
        s.axes_manager[1].index = 2
        s._plot.close()

    def test_2d_signal_with_nav_uses_combined(self):
        from hyperspy.drawing.backends.anyplotlib import _AplFigureProxy

        s = hs.signals.Signal2D(np.random.rand(4, 16, 16))
        s.plot()
        sig_fig = s._plot.signal_plot.figure
        assert isinstance(sig_fig, _AplFigureProxy)
        nav_fig = s._plot.navigator_plot.figure
        assert isinstance(nav_fig, _AplFigureProxy)
        assert sig_fig._real_fig is nav_fig._real_fig
        s._plot.close()

    def test_no_nav_single_figure(self):
        from hyperspy.drawing.backends.anyplotlib import _AplFigureProxy

        s = hs.signals.Signal1D(np.random.rand(100))
        s.plot()
        sig_fig = s._plot.signal_plot.figure
        # No navigator → single-panel figure, not a proxy
        assert not isinstance(sig_fig, _AplFigureProxy)
        s._plot.close()


# ---------------------------------------------------------------------------
# Native marker support
# ---------------------------------------------------------------------------


class TestNativeMarkers:
    """Test AnyplotlibBackend.create_markers / update_markers / remove_markers."""

    @pytest.fixture()
    def plot2d_ax(self, backend):
        import numpy as np

        fig = backend.create_figure()
        ax = backend.create_axes(fig)
        backend.plot_image(ax, np.zeros((10, 10)))
        return ax

    @pytest.fixture()
    def plot1d_ax(self, backend):
        import numpy as np

        fig = backend.create_figure()
        ax = backend.create_axes(fig)
        backend.plot_line(ax, np.arange(50), np.ones(50))
        return ax

    # ── create / update / remove round-trip ──────────────────────────────

    def test_create_circles_returns_marker_group(self, backend, plot2d_ax):
        from anyplotlib.markers import MarkerGroup

        handle = backend.create_markers(
            plot2d_ax,
            "circles",
            offset_space="data",
            transform_space="display",
            offsets=[[2, 3], [5, 7]],
            sizes=[0.5, 1.0],
        )
        assert isinstance(handle, MarkerGroup)
        assert handle._type == "circles"
        assert "radius" in handle._data

    def test_create_lines_returns_marker_group(self, backend, plot2d_ax):
        from anyplotlib.markers import MarkerGroup

        segs = np.array([[[1, 1], [3, 3]], [[5, 5], [8, 8]]])
        handle = backend.create_markers(
            plot2d_ax,
            "lines",
            offset_space="data",
            transform_space="display",
            segments=segs,
        )
        assert isinstance(handle, MarkerGroup)
        assert handle._type == "lines"

    def test_create_texts_returns_marker_group(self, backend, plot2d_ax):
        from anyplotlib.markers import MarkerGroup

        handle = backend.create_markers(
            plot2d_ax,
            "texts",
            offset_space="data",
            transform_space="display",
            offsets=[[2, 3], [5, 7]],
            texts=["A", "B"],
        )
        assert isinstance(handle, MarkerGroup)
        assert handle._type == "texts"

    def test_update_markers_changes_data(self, backend, plot2d_ax):
        handle = backend.create_markers(
            plot2d_ax,
            "circles",
            offset_space="data",
            transform_space="display",
            offsets=[[2, 3]],
            sizes=[0.5],
        )
        backend.update_markers(handle, offsets=[[9, 9]], sizes=[2.0])
        # [2.0] is a 1-element cycling sequence → _unwrap_cycling converts to scalar
        assert handle._data["radius"] == 2.0
        assert handle._data["offsets"] == [[9, 9]]

    def test_remove_markers_cleans_up(self, backend, plot2d_ax):
        handle = backend.create_markers(
            plot2d_ax,
            "circles",
            offset_space="data",
            transform_space="display",
            offsets=[[2, 3]],
            sizes=[1.0],
        )
        plot = backend._primary_plot(plot2d_ax)
        assert "circles" in plot.markers
        backend.remove_markers(plot2d_ax, handle)
        # After remove the group should be gone from the registry.
        assert "circles_1" not in plot.markers["circles"]

    # ── kwarg translation ─────────────────────────────────────────────────

    def test_translate_circles_sizes_to_radius(self, backend):
        out = backend._translate_marker_kwargs("circles", "data", {"sizes": [5, 10]})
        assert "radius" in out
        assert "sizes" not in out

    def test_translate_vlines_segments_to_offsets(self, backend):
        segs = np.array([[[3.0, 0.0], [3.0, 1.0]], [[7.0, 0.0], [7.0, 1.0]]])
        out = backend._translate_marker_kwargs("vlines", "data", {"segments": segs})
        assert "offsets" in out
        assert out["offsets"] == [[3.0], [7.0]]
        assert "segments" not in out

    def test_translate_hlines_segments_to_offsets(self, backend):
        segs = np.array([[[0.0, 0.4], [1.0, 0.4]], [[0.0, 0.8], [1.0, 0.8]]])
        out = backend._translate_marker_kwargs("hlines", "data", {"segments": segs})
        assert out["offsets"] == [[0.4], [0.8]]

    def test_translate_polygons_verts_to_vertices_list(self, backend):
        verts = [[[0, 0], [1, 0], [0.5, 1]], [[2, 2], [3, 2], [2.5, 3]]]
        out = backend._translate_marker_kwargs("polygons", "data", {"verts": verts})
        assert "vertices_list" in out
        assert "verts" not in out

    def test_translate_colors_plural_to_edgecolors(self, backend):
        out = backend._translate_marker_kwargs(
            "circles", "data", {"colors": ["red", "blue"], "offsets": [[1, 1]]}
        )
        assert "edgecolors" in out
        assert "colors" not in out

    def test_translate_strips_units(self, backend):
        out = backend._translate_marker_kwargs(
            "circles", "data", {"offsets": [[1, 1]], "units": "x"}
        )
        assert "units" not in out

    # ── 1-D marker types on Plot1D ────────────────────────────────────────

    def test_create_vlines_on_plot1d(self, backend, plot1d_ax):
        from anyplotlib.markers import MarkerGroup

        segs = np.array([[[10.0, 0.0], [10.0, 1.0]], [[30.0, 0.0], [30.0, 1.0]]])
        handle = backend.create_markers(
            plot1d_ax,
            "vlines",
            offset_space="xaxis",
            transform_space="display",
            segments=segs,
        )
        assert isinstance(handle, MarkerGroup)
        assert handle._type == "vlines"
        assert handle._data["offsets"] == [[10.0], [30.0]]

    def test_create_hlines_on_plot1d(self, backend, plot1d_ax):
        from anyplotlib.markers import MarkerGroup

        segs = np.array([[[0.0, 0.5], [1.0, 0.5]]])
        handle = backend.create_markers(
            plot1d_ax,
            "hlines",
            offset_space="yaxis",
            transform_space="display",
            segments=segs,
        )
        assert isinstance(handle, MarkerGroup)
        assert handle._type == "hlines"

    def test_create_points_on_plot1d(self, backend, plot1d_ax):
        from anyplotlib.markers import MarkerGroup

        handle = backend.create_markers(
            plot1d_ax,
            "points",
            offset_space="data",
            transform_space="display",
            offsets=[[10], [30]],
            sizes=[5],
        )
        assert isinstance(handle, MarkerGroup)
        assert handle._type == "points"

    def test_create_arrows_on_plot1d(self, backend, plot1d_ax):
        from anyplotlib.markers import MarkerGroup

        handle = backend.create_markers(
            plot1d_ax,
            "arrows",
            offset_space="data",
            transform_space="display",
            offsets=[[10.0, 1.0], [30.0, 1.2]],
            U=[2.0, -3.0],
            V=[0.5, 0.8],
            C=["red", "blue"],
        )
        assert isinstance(handle, MarkerGroup)
        assert handle._type == "arrows"
        assert handle._data["offsets"] == [[10.0, 1.0], [30.0, 1.2]]
        assert handle._data["U"] == [2.0, -3.0]
        assert handle._data["V"] == [0.5, 0.8]
        # ``C`` (per-arrow colours) is forwarded unchanged rather than stripped,
        # so once anyplotlib's arrows implementation consumes it, colouring works
        # with no further HyperSpy change. Do not drop this passthrough.
        assert handle._data["C"] == ["red", "blue"]

    # ── unsupported type raises BackendCapabilityError ────────────────────

    def test_unsupported_type_raises(self, backend, plot1d_ax):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        # An unknown marker type makes anyplotlib raise ValueError, which the
        # backend must surface as BackendCapabilityError. Real types such as
        # ``arrows`` are now supported on 1-D panels and no longer raise.
        with pytest.raises(BackendCapabilityError):
            backend.create_markers(
                plot1d_ax,
                "not_a_real_marker_type",
                offset_space="data",
                transform_space="display",
                offsets=[[1, 1]],
            )

    # ── Signal integration ────────────────────────────────────────────────

    def test_signal1d_vlines_no_crash(self):
        s = hs.signals.Signal1D(np.ones((4, 50)))
        m = hs.plot.markers.VerticalLines(offsets=np.array([10.0, 20.0, 30.0]))
        s.plot()
        s.add_marker(m)
        s._plot.close()

    def test_signal1d_hlines_no_crash(self):
        s = hs.signals.Signal1D(np.ones((4, 50)))
        m = hs.plot.markers.HorizontalLines(offsets=np.array([0.2, 0.5, 0.8]))
        s.plot()
        s.add_marker(m)
        s._plot.close()

    def test_signal2d_circles_no_crash(self):
        s = hs.signals.Signal2D(np.ones((3, 16, 16)))
        offsets = np.array([[4.0, 4.0], [8.0, 8.0], [12.0, 12.0]])
        m = hs.plot.markers.Circles(offsets=offsets, sizes=np.array([1.0, 1.5, 2.0]))
        s.plot()
        s.add_marker(m)
        s._plot.close()

    def test_signal2d_rectangles_no_crash(self):
        s = hs.signals.Signal2D(np.ones((3, 16, 16)))
        offsets = np.array([[4.0, 4.0], [8.0, 8.0]])
        m = hs.plot.markers.Rectangles(
            offsets=offsets,
            widths=np.array([2.0, 3.0]),
            heights=np.array([1.0, 2.0]),
        )
        s.plot()
        s.add_marker(m)
        s._plot.close()

    def test_signal1d_iterating_vlines_navigates(self):
        rng = np.random.default_rng(0)
        offsets = np.empty(4, dtype=object)
        for i in range(4):
            offsets[i] = rng.uniform(0, 50, size=3)
        m = hs.plot.markers.VerticalLines(offsets=offsets)
        s = hs.signals.Signal1D(np.ones((4, 50)))
        s.plot()
        s.add_marker(m)
        s.axes_manager[0].index = 2
        s._plot.close()


# ---------------------------------------------------------------------------
# Direct unit tests for gaps not exercised by the fixtures/integration tests
# above (exception-swallowing branches, capability-error raises, thin no-ops).
# ---------------------------------------------------------------------------


class TestFigureLifecycleGaps:
    def test_close_figure_swallows_exception(self, backend):
        class BadFig:
            _hspy_on_close = None

            def close(self):
                raise RuntimeError("boom")

        assert backend.close_figure(BadFig()) is None

    def test_create_combined_figure_panels_inches_like_figsize(self, backend):
        # max(figsize) < 50 is treated as matplotlib-style inches.
        nav_proxy, signal_proxy = backend.create_combined_figure_panels(
            figsize=(6.4, 4.8)
        )
        assert nav_proxy._real_fig is signal_proxy._real_fig

    def test_create_combined_figure_panels_pixel_like_figsize(self, backend):
        # max(figsize) >= 50 is treated as already being pixels.
        nav_proxy, signal_proxy = backend.create_combined_figure_panels(
            figsize=(640, 480)
        )
        assert nav_proxy._real_fig is signal_proxy._real_fig

    def test_ensure_displayed_none_fig_is_noop(self, backend):
        assert backend.ensure_displayed(None) is None

    def test_draw_idle_none_fig_is_noop(self, backend):
        assert backend.draw_idle(None) is None

    def test_ensure_displayed_swallows_importerror(self, backend, monkeypatch):
        fig = backend.create_figure()
        monkeypatch.setitem(sys.modules, "IPython.display", None)
        assert backend.ensure_displayed(fig) is None

    def test_draw_idle_swallows_importerror(self, backend, monkeypatch):
        fig = backend.create_figure()
        monkeypatch.setitem(sys.modules, "IPython.display", None)
        assert backend.draw_idle(fig) is None


class TestBlitMixinOverrides:
    """AnyplotlibBackend overrides BlitMixin defaults with its own no-ops;
    exercise them directly so the override lines (not the mixin's) are hit."""

    def test_copy_background_none(self, backend, fig_ax):
        fig, _ = fig_ax
        assert backend.copy_background(fig) is None

    def test_restore_background_noop(self, backend, fig_ax):
        fig, _ = fig_ax
        assert backend.restore_background(fig, None) is None

    def test_blit_noop(self, backend, fig_ax):
        fig, _ = fig_ax
        assert backend.blit(fig) is None

    def test_connect_draw_event_none(self, backend, fig_ax):
        fig, _ = fig_ax
        assert backend.connect_draw_event(fig, lambda *a: None) is None

    def test_disconnect_event_none_cid_is_noop(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.disconnect_event(ax, None) is None

    def test_draw_animated_artists_noop(self, backend, fig_ax):
        fig, _ = fig_ax
        assert backend.draw_animated_artists(fig) is None


class TestAxesSetupGaps:
    def test_get_xlim_before_plot_returns_defaults(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.get_xlim(ax) == (0.0, 1.0)

    def test_get_xbound_before_plot_returns_defaults(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.get_xbound(ax) == (0.0, 1.0)

    def test_set_aspect_on_image_plot(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((8, 8)))
        assert backend.set_aspect(ax, "equal") is None

    def test_add_right_axis_raises(self, backend, fig_ax):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        _, ax = fig_ax
        with pytest.raises(BackendCapabilityError):
            backend.add_right_axis(ax)

    def test_remove_right_axis_raises(self, backend, fig_ax):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        _, ax = fig_ax
        with pytest.raises(BackendCapabilityError):
            backend.remove_right_axis(ax, None)


class TestApplyPendingLabelsGaps:
    def test_delattr_failure_is_swallowed(self, backend):
        """When the pending-label attribute lives on the class (not the
        instance), delattr raises AttributeError, which must be swallowed."""
        from unittest.mock import MagicMock

        class FakeAxClassAttr:
            _hspy_pending_xlabel = "Energy"

        ax = FakeAxClassAttr()
        plot = MagicMock()
        backend._apply_pending_labels(ax, plot)
        plot.set_xlabel.assert_called_once_with("Energy")


class TestLineGaps:
    def test_line_get_linewidth_default(self, backend, fig_ax):
        _, ax = fig_ax
        h = backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        assert backend.line_get_linewidth(h) == 1.5


class TestTextAndArtistNoops:
    def test_remove_text_noop(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.remove_text(ax, None) is None

    def test_text_get_color_default(self, backend):
        assert backend.text_get_color(None) == "white"

    def test_artist_set_animated_noop(self, backend):
        assert backend.artist_set_animated(None, True) is None


class TestTextAnnotations:
    def test_add_text_before_plot_returns_none(self, backend, fig_ax):
        """No panel exists yet on ax, so there is nothing to attach text to."""
        _, ax = fig_ax
        assert backend.add_text(ax, 0.0, 0.0, "hi") is None

    def test_add_text_on_line_plot(self, backend, fig_ax):
        from anyplotlib._base_plot import _TextHandle

        _, ax = fig_ax
        backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        handle = backend.add_text(
            ax, 0.5, 0.9, "(0,)", transform="axes", fontsize=12, color="#ff0000"
        )
        assert isinstance(handle, _TextHandle)

    def test_add_text_on_image_plot(self, backend, fig_ax):
        from anyplotlib._base_plot import _TextHandle

        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((8, 8)))
        handle = backend.add_text(ax, 0.0, 0.0, "scale", transform="data")
        assert isinstance(handle, _TextHandle)

    def test_add_text_accepts_mpl_named_fontsize(self, backend, fig_ax):
        """scalebar.py passes size='medium' (an MPL fontsize name)."""
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        handle = backend.add_text(ax, 0.0, 0.0, "20 nm", size="medium")
        assert handle._group._data["fontsize"] == 12

    def test_add_text_drops_unsupported_mpl_kwargs(self, backend, fig_ax):
        """ha/bbox/picker/animated have no anyplotlib equivalent and must be
        silently ignored rather than raising."""
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        handle = backend.add_text(
            ax, 0.0, 0.0, "label", ha="center", bbox={}, picker=True, animated=False
        )
        assert handle is not None

    def test_update_text_changes_string(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        handle = backend.add_text(ax, 0.0, 0.0, "old")
        backend.update_text(handle, "new")
        assert handle._group._data["texts"] == ["new"]

    def test_text_set_and_get_color(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        handle = backend.add_text(ax, 0.0, 0.0, "x", color="#ff0000")
        assert backend.text_get_color(handle) == "#ff0000"
        backend.text_set_color(handle, "#00ff00")
        assert backend.text_get_color(handle) == "#00ff00"

    def test_remove_text_removes_handle(self, backend, fig_ax):
        _, ax = fig_ax
        h = backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        handle = backend.add_text(ax, 0.0, 0.0, "x")
        assert h.list_markers() != []
        backend.remove_text(ax, handle)
        assert h.list_markers() == []


class TestImageGaps:
    def test_plot_mesh(self, backend, fig_ax):
        _, ax = fig_ax
        x = np.arange(4, dtype=float)
        y = np.arange(5, dtype=float)
        data = np.random.rand(4, 3)
        h = backend.plot_mesh(ax, x, y, data)
        assert h is not None

    def test_image_set_extent(self, backend, fig_ax):
        _, ax = fig_ax
        h = backend.plot_image(ax, np.zeros((8, 8)))
        backend.image_set_extent(h, (0.0, 10.0, 20.0, 0.0))  # must not raise

    def test_image_set_norm_applies_clim(self, backend, fig_ax):
        from hyperspy.drawing.norm import LinearNorm

        _, ax = fig_ax
        h = backend.plot_image(ax, np.zeros((8, 8)))
        backend.image_set_norm(h, LinearNorm(vmin=0.1, vmax=0.9))
        assert h._state["display_min"] == pytest.approx(0.1)
        assert h._state["display_max"] == pytest.approx(0.9)


class TestEventConnectGaps:
    def test_connect_key_press_before_plot_returns_none(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.connect_key_press(ax, lambda e: None) is None

    def test_connect_mouse_press_before_plot_returns_none(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.connect_mouse_press(ax, lambda e: None) is None

    def test_connect_mouse_release_before_plot_returns_none(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.connect_mouse_release(ax, lambda e: None) is None

    def test_get_plot_fallback_returns_none(self, backend):
        class Dummy:
            pass

        assert backend._get_plot(Dummy()) is None

    def test_connect_widget_drag_swallows_importerror(self, backend, monkeypatch):
        monkeypatch.setitem(sys.modules, "anyplotlib.widgets._widgets1d", None)
        assert backend.connect_widget_drag(object(), lambda *a: None) is None

    def test_connect_widget_drag_unrelated_handle_type(self, backend):
        class Unrelated:
            pass

        assert backend.connect_widget_drag(Unrelated(), lambda *a: None) is None


class TestPointerGaps:
    def test_create_line_pointer_y_axis_uses_hline_on_plot1d(self, backend, fig_ax):
        """Plot1D supports a y-axis pointer natively via add_hline_widget."""
        from anyplotlib.widgets._widgets1d import HLineWidget as AplHLine

        _, ax = fig_ax
        backend.plot_line(ax, np.arange(20, dtype=float), np.zeros(20))
        handle = backend.create_line_pointer(ax, "y", 1.0)
        assert isinstance(handle, AplHLine)
        assert handle.y == 1.0
        backend.update_line_pointer(handle, "y", 3.0)
        assert handle.y == 3.0

    def test_create_rect_pointer_raises_when_plot_lacks_add_widget(
        self, backend, fig_ax
    ):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        _, ax = fig_ax
        backend.plot_line(ax, np.arange(20, dtype=float), np.zeros(20))
        with pytest.raises(BackendCapabilityError):
            backend.create_rect_pointer(ax, 0.0, 0.0, 1.0, 1.0)

    def test_create_rect_pointer_raises_when_no_plot(self, backend, fig_ax):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        _, ax = fig_ax
        with pytest.raises(BackendCapabilityError):
            backend.create_rect_pointer(ax, 0.0, 0.0, 1.0, 1.0)

    def test_remove_pointer_success(self, backend, fig_ax):
        """remove_pointer detaches the widget from its owning plot."""
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(20, dtype=float), np.zeros(20))
        handle = backend.create_line_pointer(ax, "x", 5.0)
        assert handle.id in [w.id for w in ax._plot.list_widgets()]
        backend.remove_pointer(ax, handle)
        assert handle.id not in [w.id for w in ax._plot.list_widgets()]

    def test_remove_pointer_twice_is_silent(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(20, dtype=float), np.zeros(20))
        handle = backend.create_line_pointer(ax, "x", 5.0)
        backend.remove_pointer(ax, handle)
        assert backend.remove_pointer(ax, handle) is None

    def test_remove_pointer_no_plot_is_silent(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.remove_pointer(ax, object()) is None

    def test_set_pointer_style_color(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(20, dtype=float), np.zeros(20))
        handle = backend.create_line_pointer(ax, "x", 5.0)
        backend.set_pointer_style(handle, color="blue")
        assert handle.color == "blue"

    def test_set_pointer_style_alpha_sets(self, backend, fig_ax):
        """alpha is stored on the widget instead of raising."""
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(20, dtype=float), np.zeros(20))
        handle = backend.create_line_pointer(ax, "x", 5.0)
        backend.set_pointer_style(handle, color="red", alpha=0.5)
        assert handle.alpha == 0.5
        assert handle.color == "red"


class TestUnsupportedCapabilityRaises:
    """Methods that are simple BackendCapabilityError raises / no-ops for
    features anyplotlib does not (yet) implement."""

    def test_add_artist_noop(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.add_artist(ax, None) is None

    def test_create_rect_patch_raises(self, backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            backend.create_rect_patch((0, 0), 1.0, 1.0)

    def test_get_data_transform_inverse_raises(self, backend, fig_ax):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        _, ax = fig_ax
        with pytest.raises(BackendCapabilityError):
            backend.get_data_transform_inverse(ax)

    def test_transform_point_raises(self, backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            backend.transform_point(None, (0.0, 0.0))

    def test_add_collection_raises(self, backend, fig_ax):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        _, ax = fig_ax
        with pytest.raises(BackendCapabilityError):
            backend.add_collection(ax, None)

    def test_collection_update_raises(self, backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            backend.collection_update(None)

    def test_collection_remove_raises(self, backend, fig_ax):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        _, ax = fig_ax
        with pytest.raises(BackendCapabilityError):
            backend.collection_remove(ax, None)

    def test_invalidate_blit_background_noop(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.invalidate_blit_background(ax) is None

    def test_create_span_selector_raises_without_plot(self, backend, fig_ax):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        _, ax = fig_ax
        with pytest.raises(BackendCapabilityError):
            backend.create_span_selector(ax)

    def test_create_polygon_selector_raises_without_plot(self, backend, fig_ax):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        _, ax = fig_ax
        with pytest.raises(BackendCapabilityError):
            backend.create_polygon_selector(ax)

    def test_create_span_selector_vertical_is_supported(self, backend, fig_ax):
        """0.5.0 added a vertical range orientation; this used to raise."""
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        sel = backend.create_span_selector(ax, direction="vertical")
        assert sel._widget.get("orientation") == "vertical"

    def test_get_ax_transform_returns_space_token(self, backend, fig_ax):
        _, ax = fig_ax
        for kind in ("data", "axes", "display", "xaxis", "yaxis", "relative"):
            assert backend.get_ax_transform(ax, kind) == kind

    def test_get_ax_transform_unknown_kind_raises(self, backend, fig_ax):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        _, ax = fig_ax
        with pytest.raises(BackendCapabilityError):
            backend.get_ax_transform(ax, "polar")

    def test_convert_coords_data_axes_roundtrip(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_line(ax, np.arange(20, dtype=float), np.arange(20, dtype=float))
        x0, x1 = backend.get_xlim(ax)
        y0, y1 = backend.get_ylim(ax)
        mid = [((x0 + x1) / 2, (y0 + y1) / 2)]
        frac = backend.convert_coords(ax, mid, "data", "axes")
        np.testing.assert_allclose(frac, [[0.5, 0.5]])
        back = backend.convert_coords(ax, frac, "axes", "data")
        np.testing.assert_allclose(back, mid)

    def test_convert_coords_unknown_space_raises(self, backend, fig_ax):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        _, ax = fig_ax
        backend.plot_line(ax, np.arange(20, dtype=float), np.zeros(20))
        with pytest.raises(BackendCapabilityError):
            backend.convert_coords(ax, [(0, 0)], "data", "polar")

    def test_convert_coords_requires_plot(self, backend, fig_ax):
        _, ax = fig_ax
        with pytest.raises(RuntimeError):
            backend.convert_coords(ax, [(0, 0)], "data", "axes")


class TestMarkerTranslationGaps:
    def test_create_markers_raises_runtimeerror_without_plot(self, backend, fig_ax):
        _, ax = fig_ax
        with pytest.raises(RuntimeError, match="no plot"):
            backend.create_markers(ax, "points", offsets=[[1, 1]])

    def test_update_markers_empty_kwargs_is_noop(self, backend):
        assert backend.update_markers(None) is None

    def test_remove_markers_swallows_exception(self, backend, fig_ax):
        _, ax = fig_ax

        class NoRemove:
            pass

        assert backend.remove_markers(ax, NoRemove()) is None

    def test_translate_single_element_colors_flattened_to_scalar(self, backend):
        out = backend._translate_marker_kwargs(
            "circles", "data", {"colors": ["red"], "offsets": [[1, 1]], "sizes": [1.0]}
        )
        assert out["edgecolors"] == "red"

    def test_translate_linewidth_singular_to_linewidths(self, backend):
        out = backend._translate_marker_kwargs(
            "points", "data", {"linewidth": 2.0, "offsets": [[1, 1]]}
        )
        assert out["linewidths"] == 2.0
        assert "linewidth" not in out

    @pytest.mark.parametrize(
        ("marker_type", "key"),
        [
            ("squares", "widths"),
            ("squares", "angles"),
            ("rectangles", "heights"),
            ("ellipses", "widths"),
            ("arrows", "U"),
        ],
    )
    def test_singleton_cycling_values_flattened(self, backend, marker_type, key):
        """hyperspy wraps a single value as ``(v,)`` so matplotlib cycles it.

        anyplotlib broadcasts against the marker count instead, so a 1-element
        sequence must be flattened or ``_broadcast`` rejects it whenever there
        is more than one marker.
        """
        out = backend._translate_marker_kwargs(
            marker_type, "data", {key: (5,), "offsets": [[1, 1], [2, 2]]}
        )
        assert out[key] == 5.0

    def test_per_marker_values_are_preserved(self, backend):
        out = backend._translate_marker_kwargs(
            "squares", "data", {"widths": (1, 2), "offsets": [[1, 1], [2, 2]]}
        )
        assert list(out["widths"]) == [1, 2]

    @pytest.mark.parametrize("marker_type", ["squares", "rectangles", "ellipses"])
    def test_singleton_geometry_reaches_the_wire(self, backend, fig_ax, marker_type):
        """End-to-end: a 2-marker group with one shared size must serialise."""
        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((50, 50)))
        kwargs = {"offsets": [[10, 10], [20, 20]], "widths": (5,), "angles": (0,)}
        if marker_type in ("rectangles", "ellipses"):
            kwargs["heights"] = (5,)
        group = backend.create_markers(ax, marker_type, **kwargs)
        assert group.to_wire("gid")["widths"] == [5.0, 5.0]


class TestScalebar:
    """create_scalebar/remove_scalebar use anyplotlib's native floating
    scale bar (driven by Plot2D units) instead of a marker-based artist."""

    def test_calibrated_image_uses_native_scalebar(self, backend, fig_ax):
        from hyperspy.drawing.backends.anyplotlib import _AplNativeScalebar

        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((8, 8)), extent=(0.0, 16.0, 16.0, 0.0))
        handle = backend.create_scalebar(ax, "nm")
        assert isinstance(handle, _AplNativeScalebar)
        assert ax._plot._state["units"] == "nm"

    def test_remove_native_scalebar_resets_units_to_px(self, backend, fig_ax):
        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((8, 8)), extent=(0.0, 16.0, 16.0, 0.0))
        handle = backend.create_scalebar(ax, "nm")
        backend.remove_scalebar(ax, handle)
        assert ax._plot._state["units"] == "px"

    def test_no_plot_falls_back_to_generic_scalebar(self, backend, fig_ax):
        from hyperspy.drawing._widgets.scalebar import ScaleBar

        _, ax = fig_ax
        # No plot_image() call yet, so there is no Plot2D to set units on.
        # ScaleBar.__init__ needs a real plot for get_xlim/get_ylim though,
        # so give it a line plot to attach the fallback markers to.
        backend.plot_line(ax, np.arange(5, dtype=float), np.zeros(5))
        handle = backend.create_scalebar(ax, "nm")
        assert isinstance(handle, ScaleBar)

    def test_explicit_pixel_size_falls_back_to_generic_scalebar(self, backend, fig_ax):
        from hyperspy.drawing._widgets.scalebar import ScaleBar

        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((8, 8)), extent=(0.0, 16.0, 16.0, 0.0))
        handle = backend.create_scalebar(ax, "nm", pixel_size=2.0)
        assert isinstance(handle, ScaleBar)

    def test_undefined_units_draws_no_scalebar(self, backend, fig_ax):
        """An uncalibrated axis reports `.units` as traits.Undefined.

        That sentinel must never reach Plot2D._state (it isn't
        JSON-serialisable), and no bar is drawn: anyplotlib labels the panel
        in pixels, while the generic fallback would render the units sentinel
        into the label as `10 <undefined>`.
        """
        from traits.api import Undefined

        from hyperspy.drawing.backends.anyplotlib import _AplNoScalebar

        _, ax = fig_ax
        backend.plot_image(ax, np.zeros((8, 8)), extent=(0.0, 16.0, 16.0, 0.0))
        handle = backend.create_scalebar(ax, Undefined)
        assert isinstance(handle, _AplNoScalebar)
        assert ax._plot._state["units"] != Undefined
        # Removing it must be a no-op rather than an error.
        backend.remove_scalebar(ax, handle)


class TestMiscBackendGaps:
    def test_get_figure_from_ax_raises_without_figure_attr(self, backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        class NoFig:
            pass

        with pytest.raises(BackendCapabilityError):
            backend.get_figure_from_ax(NoFig())

    def test_get_explorer_dim0_returns_hyperexplorer(self, backend):
        from hyperspy.drawing.he import HyperExplorer

        assert backend.get_explorer(0) is HyperExplorer

    def test_get_explorer_unsupported_dim_raises(self, backend):
        with pytest.raises(ValueError):
            backend.get_explorer(99)

    def test_remove_scalebar_noop(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.remove_scalebar(ax, None) is None

    def test_get_image_cmap_name_from_colormap_name(self, backend):
        class HasColormapName:
            colormap_name = "viridis"

        assert backend.get_image_cmap_name(HasColormapName()) == "viridis"

    def test_get_image_cmap_name_real_plot2d(self, backend, fig_ax):
        _, ax = fig_ax
        handle = backend.plot_image(ax, np.random.rand(4, 4), cmap="viridis")
        assert backend.get_image_cmap_name(handle) == "viridis"

    def test_get_image_cmap_name_default_gray(self, backend):
        class NoCmap:
            pass

        assert backend.get_image_cmap_name(NoCmap()) == "gray"

    def test_plot_step_falls_back_to_plot_line(self, backend, fig_ax):
        _, ax = fig_ax
        h = backend.plot_step(
            ax, np.arange(5, dtype=float), np.zeros(5), drawstyle="steps-mid"
        )
        assert h is not None

    def test_create_line2d_patch_is_detached_until_added(self, backend, fig_ax):
        _, ax = fig_ax
        patch = backend.create_line2d_patch([0, 1], [0, 1], c="red", lw=2)
        # No axes yet, so nothing has been drawn — but set_data still works.
        patch.set_data([0, 2], [0, 2])
        backend.plot_image(ax, np.zeros((4, 4)))
        backend.add_artist(ax, patch)
        assert patch._group is not None
        patch.set_data([0, 3], [0, 3])
        patch.remove()
        assert patch._group is None

    def test_create_circle_patch_raises(self, backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            backend.create_circle_patch((0, 0), 1.0)

    def test_set_autoscale_noop(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.set_autoscale(ax, True) is None

    def test_set_xticklabels_noop(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.set_xticklabels(ax, []) is None

    def test_set_yticklabels_noop(self, backend, fig_ax):
        _, ax = fig_ax
        assert backend.set_yticklabels(ax, []) is None
