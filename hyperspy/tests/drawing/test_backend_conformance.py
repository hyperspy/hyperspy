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

"""Run the backend conformance suite against the built-in backends."""

import pytest

from hyperspy.tests.drawing.backend_conformance import (
    BackendConformance,
    BackendDriver,
)

# ── matplotlib ────────────────────────────────────────────────────────────


class MplDriver(BackendDriver):
    """Emits matplotlib events through the canvas callback registry."""

    @staticmethod
    def _canvas(ax):
        return ax.figure.canvas

    def fire_key(self, backend, ax, key, modifiers=()):
        from matplotlib.backend_bases import KeyEvent

        # matplotlib's own key strings are the canonical names.
        prefix = [m for m in ("ctrl", "alt", "super", "shift") if m in modifiers]
        if len(key) == 1:
            prefix = [m for m in prefix if m != "shift"]
        name = "+".join(prefix + [key])
        x, y = ax.transAxes.transform((0.5, 0.5))
        self._canvas(ax).callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", self._canvas(ax), name, x, y),
        )

    def fire_pointer(self, backend, ax, kind, xdata, ydata, button=1, modifiers=()):
        from matplotlib.backend_bases import MouseEvent

        name = {
            "move": "motion_notify_event",
            "press": "button_press_event",
            "release": "button_release_event",
        }[kind]
        ax.figure.canvas.draw()
        x, y = ax.transData.transform((xdata, ydata))
        key = "+".join(modifiers) if modifiers else None
        event = MouseEvent(
            name,
            self._canvas(ax),
            x,
            y,
            button=None if kind == "move" else button,
            key=key,
        )
        self._canvas(ax).callbacks.process(name, event)


class TestMplConformance(BackendConformance):
    backend_name = "matplotlib"

    @pytest.fixture
    def backend(self):
        from hyperspy.drawing.backends.mpl import MplBackend

        return MplBackend()

    @pytest.fixture
    def driver(self):
        return MplDriver()


# ── anyplotlib ────────────────────────────────────────────────────────────

# Canonical base names -> browser ``KeyboardEvent.key`` values.
_BROWSER_KEYS = {
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "pageup": "PageUp",
    "pagedown": "PageDown",
}
# Canonical modifiers -> anyplotlib's modifier names.
_BROWSER_MODIFIERS = {"ctrl": "ctrl", "alt": "alt", "shift": "shift", "super": "meta"}


class AnyplotlibDriver(BackendDriver):
    """Fires anyplotlib ``Event`` objects as the browser bridge would."""

    def _fire(self, backend, ax, event_type, **fields):
        from anyplotlib.callbacks import Event

        backend._get_plot(ax).callbacks.fire(Event(event_type, **fields))

    def fire_key(self, backend, ax, key, modifiers=()):
        self._fire(
            backend,
            ax,
            "key_down",
            key=_BROWSER_KEYS.get(key, key),
            modifiers=[_BROWSER_MODIFIERS[m] for m in modifiers],
        )

    def fire_pointer(self, backend, ax, kind, xdata, ydata, button=1, modifiers=()):
        event_type = {
            "move": "pointer_move",
            "press": "pointer_down",
            "release": "pointer_up",
        }[kind]
        self._fire(
            backend,
            ax,
            event_type,
            xdata=xdata,
            ydata=ydata,
            # anyplotlib numbers buttons from 0 and reports none on a move.
            button=None if kind == "move" else button - 1,
            modifiers=[_BROWSER_MODIFIERS[m] for m in modifiers],
        )


class TestAnyplotlibConformance(BackendConformance):
    backend_name = "anyplotlib"

    @pytest.fixture
    def backend(self):
        pytest.importorskip("anyplotlib")
        from hyperspy.drawing.backends.anyplotlib import AnyplotlibBackend

        return AnyplotlibBackend()

    @pytest.fixture
    def driver(self):
        return AnyplotlibDriver()
