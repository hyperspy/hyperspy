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

"""Unit tests for hyperspy.drawing.hse.HyperSignal1D_Explorer."""

from unittest import mock

import numpy as np
import pytest

from hyperspy.signals import Signal1D


def test_auto_update_plot_noop_when_same_value():
    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    explorer = s._plot
    assert explorer.auto_update_plot is True
    with mock.patch.object(explorer.pointer, "disconnect") as mock_disconnect:
        explorer.auto_update_plot = True
    mock_disconnect.assert_not_called()
    s._plot.close()


def test_auto_update_plot_false_disconnects_pointer():
    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    explorer = s._plot
    with mock.patch.object(explorer.pointer, "disconnect") as mock_disconnect:
        explorer.auto_update_plot = False
    mock_disconnect.assert_called_once()
    s._plot.close()


def test_auto_update_plot_true_reconnects_pointer():
    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    explorer = s._plot
    # The setter's guard compares against ``_auto_update_plot``, which is
    # only ever initialised to True and never reassigned by the setter
    # itself; flip it directly so the True branch (reconnect) is reachable.
    explorer._auto_update_plot = False
    with mock.patch.object(explorer, "_connect_pointer") as mock_connect:
        explorer.auto_update_plot = True
    mock_connect.assert_called_once_with(explorer.pointer, explorer.navigator_plot)
    s._plot.close()


def test_auto_update_plot_false_without_pointer_skips_pointer_branch():
    """A signal with no navigation axes has no pointer; the setter must
    still update the lines' auto_update flag without touching self.pointer."""
    s = Signal1D(np.random.random(100))
    s.plot()
    explorer = s._plot
    assert explorer.pointer is None
    explorer.auto_update_plot = False
    assert all(not line.auto_update for line in explorer.signal_plot.ax_lines)
    s._plot.close()


def test_right_pointer_on_getter_default_false():
    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    assert s._plot.right_pointer_on is False
    s._plot.close()


def test_right_pointer_on_setter_false_removes_pointer():
    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    explorer = s._plot
    explorer.right_pointer_on = True
    assert explorer.right_pointer_on is True
    with mock.patch.object(explorer, "remove_right_pointer") as mock_remove:
        explorer.right_pointer_on = False
    mock_remove.assert_called_once()
    assert explorer.right_pointer_on is False
    s._plot.close()


def test_key2switch_right_pointer_toggles_on_e_key():
    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    explorer = s._plot
    assert explorer.right_pointer_on is False

    class _Event:
        key = "e"

    with mock.patch.object(explorer, "add_right_pointer") as mock_add:
        explorer.key2switch_right_pointer(_Event())
    mock_add.assert_called_once()
    assert explorer.right_pointer_on is True
    s._plot.close()


def test_key2switch_right_pointer_ignores_other_keys():
    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    explorer = s._plot

    class _Event:
        key = "x"

    with mock.patch.object(explorer, "add_right_pointer") as mock_add:
        explorer.key2switch_right_pointer(_Event())
    mock_add.assert_not_called()
    assert explorer.right_pointer_on is False
    s._plot.close()


def test_redraw_signal_figure_calls_draw_idle():
    from hyperspy.drawing.backends import get_backend

    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    explorer = s._plot
    with mock.patch.object(get_backend(), "draw_idle") as mock_draw_idle:
        explorer._redraw_signal_figure()
    mock_draw_idle.assert_called_once_with(explorer.signal_plot.figure)
    s._plot.close()


def test_redraw_signal_figure_noop_when_no_figure():
    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    explorer = s._plot
    explorer.signal_plot.figure = None
    # Should not raise even though there is no figure to redraw.
    explorer._redraw_signal_figure()
    s._plot.close()


def test_make_signal_figure_creates_axis_when_missing():
    """Defensive path: a backend whose create_signal1d_figure() leaves
    ``sf.ax`` unset must still get an axis via ``sf.create_axis()``."""
    from hyperspy.drawing.backends import get_backend

    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    explorer = s._plot

    real_sf = get_backend().create_signal1d_figure(
        title="stub", on_close=explorer.close
    )
    real_sf.ax = None
    with mock.patch.object(
        get_backend(), "create_signal1d_figure", return_value=real_sf
    ):
        with mock.patch.object(
            real_sf, "create_axis", wraps=real_sf.create_axis
        ) as mock_create_axis:
            explorer._make_signal_figure()
    mock_create_axis.assert_called_once()
    s._plot.close()


def test_do_add_right_pointer_without_pointer_widget():
    """assign_pointer() returning None (e.g. slider navigator) skips the
    right-axis pointer widget and goes straight to the line + redraw."""
    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    explorer = s._plot
    # Pre-set so the setter's ``right_pointer_on = True`` inside
    # _do_add_right_pointer() is a no-op and doesn't recurse back into
    # add_right_pointer() a second time.
    explorer._right_pointer_on = True

    with (
        mock.patch.object(explorer, "assign_pointer", return_value=None),
        mock.patch.object(explorer, "_add_right_line") as mock_add_line,
        mock.patch.object(explorer, "_redraw_signal_figure") as mock_redraw,
    ):
        explorer._do_add_right_pointer()

    mock_add_line.assert_called_once()
    mock_redraw.assert_called_once()
    assert explorer.right_pointer is None
    assert explorer.right_pointer_on is True
    s._plot.close()


def test_base_add_right_line_raises_backend_capability_error():
    """The undecorated HyperSignal1D_Explorer._add_right_line() is the
    fallback used by backends without twin-y-axis support."""
    from hyperspy.drawing.backends._protocol import BackendCapabilityError
    from hyperspy.drawing.hse import HyperSignal1D_Explorer

    explorer = HyperSignal1D_Explorer()
    with pytest.raises(BackendCapabilityError, match="twin-y axes"):
        explorer._add_right_line()


def test_base_redraw_signal_figure_calls_draw_idle():
    """MPL_HyperSignal1D_Explorer overrides _redraw_signal_figure(); test
    the generic base implementation (hse.py) directly."""
    from hyperspy.drawing.backends import get_backend
    from hyperspy.drawing.hse import HyperSignal1D_Explorer

    class _FigureStub:
        figure = "the-figure"

    explorer = HyperSignal1D_Explorer()
    explorer.signal_plot = _FigureStub()
    with mock.patch.object(get_backend(), "draw_idle") as mock_draw_idle:
        explorer._redraw_signal_figure()
    mock_draw_idle.assert_called_once_with("the-figure")


def test_base_redraw_signal_figure_noop_when_no_signal_plot():
    from hyperspy.drawing.hse import HyperSignal1D_Explorer

    explorer = HyperSignal1D_Explorer()
    assert explorer.signal_plot is None
    # Should not raise even though there is no signal plot yet.
    explorer._redraw_signal_figure()


def test_base_redraw_signal_figure_noop_when_no_figure():
    from hyperspy.drawing.hse import HyperSignal1D_Explorer

    class _FigureStub:
        figure = None

    explorer = HyperSignal1D_Explorer()
    explorer.signal_plot = _FigureStub()
    # Should not raise even though the figure hasn't been created yet.
    explorer._redraw_signal_figure()


def test_xlabel_appends_units_when_defined():
    s = Signal1D(np.random.random((10, 20, 100)))
    s.axes_manager.signal_axes[0].units = "eV"
    s.plot()
    assert s._plot.xlabel.endswith("(eV)")
    s._plot.close()


def test_make_signal_figure_complex_adds_imaginary_line():
    data = np.random.random((10, 20, 100)) + 1j * np.random.random((10, 20, 100))
    s = Signal1D(data)
    s.plot()
    # A second (imaginary-part) line is added on top of the real-part line.
    assert len(s._plot.signal_plot.ax_lines) == 2
    assert any(
        getattr(line, "_plot_imag", False) for line in s._plot.signal_plot.ax_lines
    )
    s._plot.close()


def test_connect_key_handler_skips_when_no_figure():
    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    explorer = s._plot

    class _FigureStub:
        figure = None

    with mock.patch("hyperspy.drawing.hse.get_backend") as mock_get_backend:
        explorer._connect_key_handler(_FigureStub(), lambda event: None)
    mock_get_backend.assert_not_called()
    s._plot.close()


def test_connect_key_handler_skips_appending_cid_when_none():
    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    explorer = s._plot
    n_cids = len(explorer._key_nav_cids)

    from hyperspy.drawing.backends import get_backend

    with mock.patch.object(get_backend(), "connect_key_press", return_value=None):
        explorer._connect_key_handler(explorer.signal_plot, lambda event: None)
    assert len(explorer._key_nav_cids) == n_cids
    s._plot.close()


def test_do_add_right_pointer_pointer_without_size_attribute():
    """1-D navigation uses a line pointer, which has no ``size`` attribute;
    the hasattr guard must skip copying it onto the right pointer."""
    s = Signal1D(np.random.random((10, 100)))
    s.plot()
    explorer = s._plot
    assert not hasattr(explorer.pointer, "size")
    explorer.add_right_pointer()
    assert not hasattr(explorer.right_pointer, "size")
    s._plot.close()


def test_do_add_right_pointer_second_call_reuses_right_pointer():
    """Calling add_right_pointer() again when a right_pointer already
    exists updates its navigation axes instead of recreating it.

    Uses 3 navigation axes so the 2-D pointer's ``_pointer_nav_dim`` leaves
    a trailing navigation axis, exercising the axes-reassignment loop.
    """
    s = Signal1D(np.random.random((5, 10, 20, 100)))
    s.plot()
    explorer = s._plot
    explorer.add_right_pointer()
    right_pointer = explorer.right_pointer
    assert explorer.axes_manager.navigation_axes[explorer._pointer_nav_dim :]
    explorer.add_right_pointer()
    assert explorer.right_pointer is right_pointer
    s._plot.close()
