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

"""Behavioural conformance suite for plotting backends.

Every backend, built in or third party, runs the same tests by subclassing
:class:`BackendConformance` and providing two fixtures::

    import pytest
    from hyperspy.tests.drawing.backend_conformance import (
        BackendConformance, BackendDriver)

    class MyDriver(BackendDriver):
        def fire_key(self, backend, ax, key, modifiers=()):
            ...  # emit a native key press on ax
        def fire_pointer(self, backend, ax, kind, xdata, ydata,
                         button=1, modifiers=()):
            ...  # emit a native pointer event at a data position

    class TestMyBackendConformance(BackendConformance):
        backend_name = "mybackend"   # the entry-point name

        @pytest.fixture
        def backend(self):
            return MyBackend()

        @pytest.fixture
        def driver(self):
            return MyDriver()

The driver is the only backend-specific part: it emits the backend's own
*native* events, so the tests check that ``connect_*`` hands handlers the
neutral events of :mod:`hyperspy.drawing.backends._events`.  Features a
backend declares unsupported by raising
:class:`~hyperspy.drawing.backends._protocol.BackendCapabilityError` are
skipped, not failed.
"""

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.drawing.backends._events import KeyEvent, PointerEvent
from hyperspy.drawing.backends._protocol import (
    ArtistBackend,
    BackendCapabilityError,
    CoordSpace,
    MarkerType,
    PlottingBackend,
)


class BackendDriver:
    """Emits a backend's native input events for the conformance tests."""

    def fire_key(self, backend, ax, key, modifiers=()):
        """Press *key* (a canonical base name such as ``"left"`` or ``"a"``)
        with *modifiers* held, over *ax*."""
        raise NotImplementedError

    def fire_pointer(self, backend, ax, kind, xdata, ydata, button=1, modifiers=()):
        """Emit a pointer event of *kind* (``"move"``, ``"press"`` or
        ``"release"``) at data position (*xdata*, *ydata*) of *ax*.
        *button* uses the canonical numbering (1 = left)."""
        raise NotImplementedError


def _capability(fn, *args, **kwargs):
    """Call *fn*; skip the test if the backend declares it unsupported."""
    try:
        return fn(*args, **kwargs)
    except BackendCapabilityError as e:
        pytest.skip(f"unsupported by this backend: {e}")


class BackendConformance:
    """Shared behavioural tests; subclass it with ``backend`` and ``driver``
    fixtures (see the module docstring)."""

    #: Entry-point name of the backend, used by the integration tests.
    backend_name = None

    # ── fixtures ──────────────────────────────────────────────────────────

    @pytest.fixture
    def line_ax(self, backend):
        fig = backend.create_figure()
        ax = backend.create_axes(fig)
        backend.plot_line(ax, np.arange(48.0), np.linspace(0.0, 1.0, 48))
        backend.set_xlim(ax, 0.0, 47.0)
        backend.set_ylim(ax, 0.0, 1.0)
        backend.ensure_displayed(fig)
        yield ax
        backend.close_figure(fig)

    @pytest.fixture
    def image_ax(self, backend):
        fig = backend.create_figure()
        ax = backend.create_axes(fig)
        # All-different dimensions so a transposition would show.
        data = np.arange(25 * 12, dtype=float).reshape(25, 12)
        backend.plot_image(ax, data, extent=(-0.5, 11.5, 24.5, -0.5), cmap="gray")
        backend.ensure_displayed(fig)
        yield ax
        backend.close_figure(fig)

    @pytest.fixture
    def active(self, backend):
        """Select the backend for code that plots through ``hs``."""
        if self.backend_name is None:
            pytest.skip("backend_name not set")
        from hyperspy.defaults_parser import preferences

        old = preferences.Plot.backend
        preferences.Plot.backend = self.backend_name
        yield
        preferences.Plot.backend = old

    # ── protocol shape ────────────────────────────────────────────────────

    def test_satisfies_protocol(self, backend):
        assert isinstance(backend, PlottingBackend)

    def test_artist_extension_is_all_or_nothing(self, backend):
        """A backend implementing only part of ArtistBackend fails the
        isinstance check, so the generic layer would silently ignore the
        methods it does implement."""
        names = [
            n
            for n in dir(ArtistBackend)
            if not n.startswith("_") and callable(getattr(ArtistBackend, n))
        ]
        present = [n for n in names if callable(getattr(backend, n, None))]
        assert present in ([], names), (
            f"partial ArtistBackend: has {present}, missing "
            f"{sorted(set(names) - set(present))}"
        )

    @pytest.mark.parametrize("signal_dim", [0, 1, 2])
    def test_explorer_is_concrete(self, backend, signal_dim):
        from hyperspy.drawing.he import HyperExplorer
        from hyperspy.drawing.hie import HyperImage_Explorer
        from hyperspy.drawing.hse import HyperSignal1D_Explorer

        explorer = backend.get_explorer(signal_dim)
        assert issubclass(explorer, HyperExplorer)
        base = {1: HyperSignal1D_Explorer, 2: HyperImage_Explorer}.get(signal_dim)
        if base is not None:
            assert issubclass(explorer, base)

    # ── drawing primitives ────────────────────────────────────────────────

    def test_figure_of_axes(self, backend, line_ax):
        assert backend.get_figure_from_ax(line_ax) is not None

    def test_limits_round_trip(self, backend, line_ax):
        backend.set_xlim(line_ax, 3.0, 40.0)
        backend.set_ylim(line_ax, 0.25, 0.75)
        assert backend.get_xlim(line_ax) == pytest.approx((3.0, 40.0))
        assert backend.get_ylim(line_ax) == pytest.approx((0.25, 0.75))

    def test_line_update_and_remove(self, backend, line_ax):
        handle = backend.plot_line(line_ax, np.arange(5.0), np.zeros(5))
        backend.update_line(handle, np.arange(7.0), np.ones(7))
        np.testing.assert_allclose(backend.line_get_xdata(handle), np.arange(7.0))
        backend.remove_line(line_ax, handle)

    def test_text_lifecycle(self, backend, line_ax):
        handle = backend.add_text(line_ax, 0.5, 0.5, "label", transform="axes")
        backend.update_text(handle, "changed")
        backend.text_set_color(handle, "red")
        assert backend.text_get_color(handle) is not None
        backend.remove_text(line_ax, handle)

    def test_image_updates(self, backend, image_ax):
        handle = backend.get_image_handle(image_ax)
        assert handle is not None
        backend.image_set_data(handle, np.ones((25, 12)))
        backend.image_set_clim(handle, 0.0, 2.0)
        assert backend.get_image_cmap_name(handle) == "gray"

    def test_convert_coords_round_trip(self, backend, line_ax):
        points = np.array([[10.0, 0.25], [30.0, 0.75]])
        display = _capability(
            backend.convert_coords, line_ax, points, CoordSpace.DATA, CoordSpace.DISPLAY
        )
        back = backend.convert_coords(
            line_ax, display, CoordSpace.DISPLAY, CoordSpace.DATA
        )
        np.testing.assert_allclose(back, points, rtol=1e-6, atol=1e-9)

    def test_markers_lifecycle(self, backend, image_ax):
        handle = _capability(
            backend.create_markers,
            image_ax,
            MarkerType.POINTS,
            offsets=np.array([[2.0, 3.0], [5.0, 20.0]]),
            sizes=(5,),
        )
        backend.update_markers(handle, offsets=np.array([[1.0, 1.0]]))
        backend.remove_markers(image_ax, handle)

    # ── widget primitives ─────────────────────────────────────────────────

    def test_line_pointer(self, backend, line_ax):
        handle = _capability(backend.create_line_pointer, line_ax, "x", 10.0)
        backend.update_line_pointer(handle, "x", 20.0)
        backend.set_pointer_style(handle, color="blue", alpha=0.5)
        backend.connect_widget_drag(handle, lambda *args: None)
        backend.remove_pointer(line_ax, handle)

    def test_rect_pointer(self, backend, image_ax):
        handle = _capability(backend.create_rect_pointer, image_ax, 2.0, 3.0, 4.0, 5.0)
        backend.update_rect_pointer(handle, 3.0, 4.0, 2.0, 2.0)
        backend.connect_widget_drag(handle, lambda *args: None)
        backend.remove_pointer(image_ax, handle)

    def test_circle_pointer_crosses_annulus_boundary(self, backend, image_ax):
        handles = _capability(backend.create_circle_pointer, image_ax, 5.0, 10.0, 3.0)
        handles = backend.update_circle_pointer(image_ax, handles, 5.0, 10.0, 3.0, 1.0)
        handles = backend.update_circle_pointer(image_ax, handles, 5.0, 10.0, 3.0, 0.0)
        assert len(handles) >= 1
        for h in handles:
            backend.remove_pointer(image_ax, h)

    def test_line2d_pointer(self, backend, image_ax):
        handle = _capability(
            backend.create_line2d_pointer, image_ax, [1.0, 8.0], [2.0, 20.0]
        )
        backend.update_line2d_pointer(handle, [2.0, 9.0], [3.0, 21.0])
        passive = backend.create_line2d_pointer(
            image_ax, [1.0, 8.0], [2.0, 20.0], linestyle=":", interactive=False
        )
        backend.remove_pointer(image_ax, passive)
        backend.remove_pointer(image_ax, handle)

    # ── input events ──────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "key, modifiers, expected",
        [
            ("left", (), "left"),
            ("right", ("ctrl",), "ctrl+right"),
            ("pageup", (), "pageup"),
            ("a", (), "a"),
            ("up", ("alt", "shift"), "alt+shift+up"),
        ],
    )
    def test_key_event_is_canonical(
        self, backend, driver, line_ax, key, modifiers, expected
    ):
        received = []
        cid = backend.connect_key_press(line_ax, received.append)
        driver.fire_key(backend, line_ax, key, modifiers)
        backend.disconnect_event(line_ax, cid)
        assert len(received) == 1
        assert isinstance(received[0], KeyEvent)
        assert received[0].key == expected

    @pytest.mark.parametrize(
        "connect, kind",
        [
            ("connect_mouse_move", "move"),
            ("connect_mouse_press", "press"),
            ("connect_mouse_release", "release"),
        ],
    )
    def test_pointer_event_is_neutral(self, backend, driver, line_ax, connect, kind):
        received = []
        cid = getattr(backend, connect)(line_ax, received.append)
        driver.fire_pointer(
            backend, line_ax, kind, 12.0, 0.5, button=1, modifiers=("shift",)
        )
        backend.disconnect_event(line_ax, cid)
        assert len(received) == 1
        event = received[0]
        assert isinstance(event, PointerEvent)
        assert event.kind == kind
        assert event.inaxes is line_ax
        assert event.xdata == pytest.approx(12.0, abs=0.5)
        assert event.ydata == pytest.approx(0.5, abs=0.05)
        if kind != "move":
            assert event.button == 1
        assert "shift" in event.modifiers

    def test_disconnect_stops_delivery(self, backend, driver, line_ax):
        received = []
        cid = backend.connect_key_press(line_ax, received.append)
        backend.disconnect_event(line_ax, cid)
        driver.fire_key(backend, line_ax, "left")
        assert received == []

    # ── end to end through HyperSpy ───────────────────────────────────────

    def test_signal1d_navigation(self, active):
        s = hs.signals.Signal1D(np.random.random((12, 25, 48)))
        s.plot()
        try:
            s.axes_manager.indices = (3, 7)
            s.axes_manager.indices = (24, 11)
        finally:
            s._plot.close()

    def test_signal2d_navigation(self, active):
        s = hs.signals.Signal2D(np.random.random((6, 25, 12)))
        s.plot()
        try:
            s.axes_manager.indices = (4,)
        finally:
            s._plot.close()

    def test_keyboard_navigation(self, backend, driver, active):
        from hyperspy.defaults_parser import preferences

        s = hs.signals.Signal1D(np.zeros((12, 25, 48)))
        s.plot()
        try:
            nav_ax = s._plot.navigator_plot.ax
            mods = tuple(preferences.Plot.modifier_dims_01.split("+"))
            driver.fire_key(backend, nav_ax, preferences.Plot.dims_024_increase, mods)
            assert s.axes_manager.indices == (1, 0)
        finally:
            s._plot.close()

    def test_roi_widget_follows_roi(self, active):
        s = hs.signals.Signal2D(np.random.random((25, 12)))
        s.plot()
        try:
            roi = hs.roi.RectangularROI(left=1, top=2, right=5, bottom=9)
            widget = roi.add_widget(s)
            roi.left, roi.right = 2, 6
            assert widget.position[0] == pytest.approx(2.0)
            roi.remove_widget(s)
        finally:
            s._plot.close()
