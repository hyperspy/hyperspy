# -*- coding: utf-8 -*-
"""Tests for the anyplotlib plotting backend.

These tests exercise AnyplotlibBackend directly (unit tests) and via
hs.Signal1D/Signal2D.plot() (integration tests).  They are skipped when
anyplotlib is not installed.
"""

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
        backend.update_line_pointer(w, 10.0)
        assert w.x == pytest.approx(10.0)

    def test_create_line_pointer_before_plot_raises(self, backend, fig_ax):
        _, ax = fig_ax
        with pytest.raises(RuntimeError, match="no plot"):
            backend.create_line_pointer(ax, "x", 0.0)


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
    """Native anyplotlib CrosshairWidget for 2D image navigator navigation."""

    def test_crosshair_created_for_hline_widget(self):
        """HorizontalLineWidget attaches a native crosshair on anyplotlib 2D navigator.

        Signal1D(n, length) has a 2D image navigator (full spectrum stack);
        the horizontal line marks the current navigation row (y-axis).
        """
        from anyplotlib.widgets._widgets2d import CrosshairWidget as AplCrosshair

        from hyperspy.drawing._widgets.horizontal_line import HorizontalLineWidget

        s = hs.signals.Signal1D(np.random.rand(4, 50))
        s.plot()
        pointer = s._plot.pointer
        assert isinstance(pointer, HorizontalLineWidget), (
            f"expected HorizontalLineWidget, got {type(pointer).__name__}"
        )
        assert pointer.patch, "HorizontalLineWidget should have a patch handle"
        assert isinstance(pointer.patch[0], AplCrosshair)
        s._plot.close()

    def test_navigate_updates_hline_crosshair_cy(self):
        """Changing navigation index moves the crosshair cy to the new row value."""
        s = hs.signals.Signal1D(np.random.rand(5, 50))
        s.plot()
        pointer = s._plot.pointer
        nav_ax = s.axes_manager.navigation_axes[0]
        nav_ax.index = 3
        assert abs(pointer.patch[0].cy - nav_ax.value) < 1e-9
        s._plot.close()

    def test_hline_drag_updates_nav(self):
        """Dragging the crosshair (simulated via set()) updates nav index."""
        s = hs.signals.Signal1D(np.random.rand(5, 50))
        s.plot()
        pointer = s._plot.pointer
        nav_ax = s.axes_manager.navigation_axes[0]
        pointer.patch[0].set(cy=float(nav_ax.axis[2]))
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

    # ── unsupported type raises BackendCapabilityError ────────────────────

    def test_unsupported_type_raises(self, backend, plot1d_ax):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            backend.create_markers(
                plot1d_ax,
                "arrows",  # arrows not supported on Plot1D
                offset_space="data",
                transform_space="display",
                offsets=[[1, 1]],
                U=[1],
                V=[1],
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
