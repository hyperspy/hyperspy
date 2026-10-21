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

"""The backend-neutral input event contract (``backends._events``)."""

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.drawing.backends._events import (
    KeyEvent,
    PickEvent,
    PointerEvent,
    canonical_key,
    canonical_modifiers,
)


class TestCanonicalKey:
    @pytest.mark.parametrize(
        "name, modifiers, expected",
        [
            ("ArrowLeft", (), "left"),
            ("ArrowRight", ("ctrl",), "ctrl+right"),
            ("PageUp", (), "pageup"),
            ("PageDown", ("shift",), "shift+pagedown"),
            ("left", ("alt", "ctrl"), "ctrl+alt+left"),
            ("ArrowUp", ("shift", "alt", "ctrl"), "ctrl+alt+shift+up"),
            ("a", (), "a"),
            # A shifted printable key already reports the shifted character.
            ("A", ("shift",), "A"),
            ("+", ("shift",), "+"),
            ("a", ("ctrl",), "ctrl+a"),
            ("F5", (), "f5"),
            ("Escape", (), "escape"),
            ("Enter", ("meta",), "super+enter"),
            # A modifier on its own is reported bare.
            ("Shift", ("shift",), "shift"),
            ("Control", ("ctrl",), "control"),
            (None, (), None),
            ("", (), None),
        ],
    )
    def test_canonical_key(self, name, modifiers, expected):
        assert canonical_key(name, modifiers) == expected

    def test_canonical_modifiers(self):
        assert canonical_modifiers(["control", "Shift", "meta", "bogus"]) == {
            "ctrl",
            "shift",
            "super",
        }
        assert canonical_modifiers(None) == frozenset()

    def test_pointer_event_key_compat(self):
        event = PointerEvent("press", modifiers=frozenset({"shift", "ctrl"}))
        assert event.key == "ctrl+shift"
        assert PointerEvent("move").key is None


class TestMplAdapter:
    """matplotlib events are converted before reaching a handler."""

    @pytest.fixture()
    def fig_ax(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        fig.canvas.draw()
        yield fig, ax
        plt.close(fig)

    def _mouse(self, fig, ax, name, button=None, key=None, data=(5.0, 5.0)):
        from matplotlib.backend_bases import MouseEvent

        x, y = ax.transData.transform(data)
        return MouseEvent(name, fig.canvas, x, y, button=button, key=key)

    def test_key_press(self, fig_ax):
        from matplotlib.backend_bases import KeyEvent as MplKeyEvent

        from hyperspy.drawing.backends.mpl import MplBackend

        fig, ax = fig_ax
        received = []
        MplBackend().connect_key_press(fig, received.append)
        native = MplKeyEvent("key_press_event", fig.canvas, "ctrl+left")
        fig.canvas.callbacks.process("key_press_event", native)
        (event,) = received
        assert isinstance(event, KeyEvent)
        assert event.key == "ctrl+left"
        assert event.native is native

    @pytest.mark.parametrize(
        "connect, name, kind",
        [
            ("connect_mouse_move", "motion_notify_event", "move"),
            ("connect_mouse_press", "button_press_event", "press"),
            ("connect_mouse_release", "button_release_event", "release"),
        ],
    )
    def test_pointer(self, fig_ax, connect, name, kind):
        from hyperspy.drawing.backends.mpl import MplBackend

        fig, ax = fig_ax
        received = []
        getattr(MplBackend(), connect)(ax, received.append)
        native = self._mouse(fig, ax, name, button=1, key="shift")
        fig.canvas.callbacks.process(name, native)
        (event,) = received
        assert isinstance(event, PointerEvent)
        assert event.kind == kind
        assert event.inaxes is ax
        assert event.xdata == pytest.approx(5.0)
        assert event.ydata == pytest.approx(5.0)
        assert event.button == 1
        assert "shift" in event.modifiers
        assert event.native is native

    def test_pick(self, fig_ax):
        import matplotlib.patches as mpatches
        from matplotlib.backend_bases import PickEvent as MplPickEvent

        from hyperspy.drawing.backends.mpl import MplBackend

        fig, ax = fig_ax
        patch = ax.add_patch(mpatches.Rectangle((4, 4), 2, 2, picker=True))
        received = []
        MplBackend().connect_pick(ax, received.append)
        mouse = self._mouse(fig, ax, "button_press_event", button=1)
        fig.canvas.callbacks.process(
            "pick_event", MplPickEvent("pick_event", fig.canvas, mouse, patch)
        )
        (event,) = received
        assert isinstance(event, PickEvent)
        assert event.artist is patch
        assert event.xdata == pytest.approx(5.0)
        assert event.button == 1

    def test_disconnect(self, fig_ax):
        from hyperspy.drawing.backends.mpl import MplBackend

        fig, ax = fig_ax
        backend = MplBackend()
        received = []
        cid = backend.connect_mouse_move(ax, received.append)
        backend.disconnect_event(ax, cid)
        native = self._mouse(fig, ax, "motion_notify_event")
        fig.canvas.callbacks.process("motion_notify_event", native)
        assert received == []


class TestAnyplotlibAdapter:
    """anyplotlib reports browser key names and 0-based buttons."""

    @pytest.fixture()
    def setup(self):
        pytest.importorskip("anyplotlib")
        from hyperspy.drawing.backends.anyplotlib import AnyplotlibBackend

        backend = AnyplotlibBackend()
        fig = backend.create_figure()
        ax = backend.create_axes(fig)
        backend.plot_line(ax, np.arange(10.0), np.arange(10.0))
        return backend, ax, backend._get_plot(ax)

    def _fire(self, plot, event_type, **fields):
        from anyplotlib.callbacks import Event

        plot.callbacks.fire(Event(event_type, **fields))

    def test_key_names_are_canonical(self, setup):
        backend, ax, plot = setup
        received = []
        backend.connect_key_press(ax, received.append)
        self._fire(plot, "key_down", key="ArrowLeft", modifiers=["ctrl"])
        self._fire(plot, "key_down", key="PageUp")
        assert [e.key for e in received] == ["ctrl+left", "pageup"]
        assert all(isinstance(e, KeyEvent) for e in received)
        assert received[0].inaxes is ax

    @pytest.mark.parametrize(
        "connect, event_type, kind",
        [
            ("connect_mouse_move", "pointer_move", "move"),
            ("connect_mouse_press", "pointer_down", "press"),
            ("connect_mouse_release", "pointer_up", "release"),
        ],
    )
    def test_pointer(self, setup, connect, event_type, kind):
        backend, ax, plot = setup
        received = []
        getattr(backend, connect)(ax, received.append)
        self._fire(
            plot,
            event_type,
            xdata=2.5,
            ydata=3.5,
            x=10.0,
            y=20.0,
            button=0,
            modifiers=["shift"],
        )
        (event,) = received
        assert isinstance(event, PointerEvent)
        assert event.kind == kind
        assert event.button == 1  # anyplotlib's 0 is the left button
        assert event.modifiers == {"shift"}
        assert event.inaxes is ax
        assert (event.xdata, event.ydata) == (2.5, 3.5)

    def test_pointer_outside_axes(self, setup):
        backend, ax, plot = setup
        received = []
        backend.connect_mouse_move(ax, received.append)
        self._fire(plot, "pointer_move")
        (event,) = received
        assert event.inaxes is None
        assert event.button is None

    def test_disconnect(self, setup):
        backend, ax, plot = setup
        received = []
        cid = backend.connect_key_press(ax, received.append)
        backend.disconnect_event(ax, cid)
        self._fire(plot, "key_down", key="ArrowLeft")
        assert received == []

    def test_keyboard_navigation(self, setup):
        """Arrow keys from the browser step the navigation indices."""
        from hyperspy.defaults_parser import preferences

        old = preferences.Plot.backend
        preferences.Plot.backend = "anyplotlib"
        try:
            s = hs.signals.Signal1D(np.zeros((12, 25, 48)))
            s.plot()
            backend = setup[0]
            nav_plot = backend._get_plot(s._plot.navigator_plot.ax)
            mod = preferences.Plot.modifier_dims_01.split("+")
            assert s.axes_manager.indices == (0, 0)
            self._fire(nav_plot, "key_down", key="ArrowRight", modifiers=mod)
            assert s.axes_manager.indices == (1, 0)
            self._fire(nav_plot, "key_down", key="ArrowDown", modifiers=mod)
            assert s.axes_manager.indices == (1, 1)
            s._plot.close()
        finally:
            preferences.Plot.backend = old
