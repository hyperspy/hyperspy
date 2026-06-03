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

from unittest import mock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import CloseEvent
from packaging.version import Version

import hyperspy.api as hs
from hyperspy._components.polynomial import Polynomial
from hyperspy.drawing._markers.points import Points
from hyperspy.drawing.figure import BlittedFigure
from hyperspy.drawing.tiles import HistogramTilePlot
from hyperspy.events import Event, Events
from hyperspy.misc.test_utils import check_closing_plot
from hyperspy.signals import Signal1D, Signal2D


def _close_figure_matplotlib_event(figure):
    try:
        # Introduced in matplotlib 3.6 and `clost_event` deprecated
        event = CloseEvent("close_event", figure)
        figure.canvas.callbacks.process("close_event", event)
    except Exception:  # Deprecated in matplotlib 3.6
        figure.canvas.close_event()


def test_figure_title_length():
    f = BlittedFigure()
    f.title = "Test" * 50
    assert max([len(line) for line in f.title.split("\n")]) < 61


def _assert_figure_state_after_close(fig):
    assert len(fig.events.closed.connected) == 0
    assert fig._draw_event_cid is None
    assert fig.figure is None
    assert fig._background is None
    assert fig.ax is None


def test_close_figure_using_close_method():
    fig = BlittedFigure()
    fig.create_figure()
    assert fig.figure is not None
    fig.close()
    _assert_figure_state_after_close(fig)


def test_close_figure_using_matplotlib():
    # check that matplotlib callback to `_on_close` is working fine
    fig = BlittedFigure()
    fig.create_figure()
    assert fig.figure is not None
    # Close using matplotlib, similar to using gui
    _close_figure_matplotlib_event(fig.figure)
    _assert_figure_state_after_close(fig)


def test_close_figure_with_plotted_marker():
    s = Signal1D(np.arange(10))
    m = Points(
        offsets=[
            [0, 0],
        ],
        color="red",
        sizes=100,
    )
    s.add_marker(m)
    s.plot(True)
    s._plot.close()
    check_closing_plot(s)


@pytest.mark.parametrize("navigator", ["auto", "slider", "spectrum"])
@pytest.mark.parametrize("nav_dim", [1, 2])
@pytest.mark.parametrize("sig_dim", [1, 2])
def test_close_figure(navigator, nav_dim, sig_dim):
    total_dim = nav_dim * sig_dim
    if sig_dim == 1:
        Signal = Signal1D
    elif sig_dim == 2:
        Signal = Signal2D
    s = Signal(np.arange(pow(10, total_dim)).reshape([10] * total_dim))
    s.plot(navigator=navigator)
    s._plot.close()
    check_closing_plot(s, check_data_changed_close=False)

    if sig_dim == 1:
        m = s.create_model()
        m.plot()
        # Close using matplotlib, similar to using gui
        _close_figure_matplotlib_event(m._plot.signal_plot.figure)
        m.extend([Polynomial(1)])


def test_remove_markers():
    s = Signal2D(np.arange(pow(10, 3)).reshape([10] * 3))
    s.plot()
    m = Points(
        offsets=[
            [0, 0],
        ],
        color="red",
        sizes=100,
    )
    s.add_marker(m)
    s._plot.signal_plot.remove_markers()
    assert len(s._plot.signal_plot.ax_markers) == 0
    assert m._collection is None  # Check that the collection is set to None


def test_remove_markers_renders_resets_blit_background():
    """Verify remove_markers(render_figure=True) invalidates blit cache
    and repaints."""
    s = Signal2D(np.arange(pow(10, 3)).reshape([10] * 3))
    s.plot()
    m = Points(
        offsets=[
            [0, 0],
        ],
        color="red",
        sizes=100,
    )
    s.add_marker(m)
    s._plot.signal_plot._background = "stale"
    s._plot.signal_plot.remove_markers(render_figure=True)
    assert s._plot.signal_plot._background is not None


@pytest.mark.skipif(
    Version(matplotlib.__version__) < Version("3.9.0"),
    reason="Subfigures plotting requires matplotlib >= 3.9.0",
)
def test_close_figure_with_subfigure():
    rng = np.random.default_rng()
    s = Signal1D(rng.random((10, 10, 10)))

    fig = plt.figure()
    subfig_nav, subfig_sig = fig.subfigures(1, 2)

    s.plot(navigator_kwds=dict(fig=subfig_nav), fig=subfig_sig)
    s._plot.close()

    # This shows an empty axis...
    # s.plot(
    #     navigator_kwds=dict(fig=subfig_nav),
    #     fig=subfig_sig
    #     )


@pytest.mark.skipif(
    Version(matplotlib.__version__) >= Version("3.9.0"),
    reason="Error raised for matplotlib < 3.9.0",
)
def test_subfigure_preferences_setting():
    rng = np.random.default_rng()
    s = Signal1D(rng.random((10, 10, 10)))

    hs.preferences.Plot.use_subfigure = True
    if Version(matplotlib.__version__) < Version("3.9.0"):
        with pytest.raises(ValueError):
            s.plot()
    else:
        s.plot()
    s._plot.close()
    # Set default setting back
    hs.preferences.Plot.use_subfigure = False


@pytest.mark.skipif(
    Version(matplotlib.__version__) < Version("3.9.0"),
    reason="Subfigures plotting requires matplotlib >= 3.9.0",
)
def test_close_figure_with_subfigure_matplotlib_event():
    rng = np.random.default_rng()
    s = Signal1D(rng.random((10, 10, 10)))

    fig = plt.figure()
    subfig_nav, subfig_sig = fig.subfigures(1, 2)

    s.plot(navigator_kwds=dict(fig=subfig_nav), fig=subfig_sig)
    plt.close(fig)


@pytest.mark.skipif(
    Version(matplotlib.__version__) < Version("3.9.0"),
    reason="Subfigures plotting requires matplotlib >= 3.9.0",
)
def test_subfigure_get_mpl_figure():
    rng = np.random.default_rng()
    s = Signal1D(rng.random((10, 10, 10)))

    hs.preferences.Plot.use_subfigure = True
    s.plot()
    assert isinstance(s._plot.signal_plot.get_mpl_figure(), matplotlib.figure.Figure)
    assert isinstance(s._plot.signal_plot.figure, matplotlib.figure.SubFigure)
    s._plot.signal_plot.close()
    # Set default setting back
    hs.preferences.Plot.use_subfigure = False


def test_separate_figure_get_mpl_figure():
    rng = np.random.default_rng()
    s = Signal1D(rng.random((10, 10, 10)))

    s.plot()
    assert isinstance(s._plot.signal_plot.get_mpl_figure(), matplotlib.figure.Figure)
    assert isinstance(s._plot.signal_plot.figure, matplotlib.figure.Figure)
    s._plot.signal_plot.close()


def test_remove_markers_iterates_copy():
    """Verify remove_markers uses list() copy to avoid mutation during iteration."""
    f = BlittedFigure()
    f.figure = mock.MagicMock()
    f.ax = mock.MagicMock()

    marker1 = mock.MagicMock()
    marker2 = mock.MagicMock()
    f.ax_markers = [marker1, marker2]

    f.remove_markers()

    marker1.close.assert_called_once_with(render_figure=False)
    marker2.close.assert_called_once_with(render_figure=False)


def test_draw_animated_skips_removed_artist():
    """Verify _draw_animated skips artist when artist.axes is None (guard)."""
    f = BlittedFigure()
    mock_ax = mock.MagicMock()
    mock_fig = mock.MagicMock()
    mock_fig.axes = [mock_ax]
    f.figure = mock_fig

    removed_artist = mock.MagicMock()
    removed_artist.get_animated.return_value = True
    removed_artist.axes = None
    removed_artist.zorder = 1

    mock_ax.get_children.return_value = [removed_artist]

    f._draw_animated()

    mock_ax.draw_artist.assert_not_called()


def test_draw_animated_draws_valid_artist():
    """Verify _draw_animated calls draw_artist when artist is animated and has axes."""
    f = BlittedFigure()
    mock_ax = mock.MagicMock()
    mock_fig = mock.MagicMock()
    mock_fig.axes = [mock_ax]
    f.figure = mock_fig

    valid_artist = mock.MagicMock()
    valid_artist.get_animated.return_value = True
    valid_artist.axes = mock_ax  # not None — still attached
    valid_artist.zorder = 1

    mock_ax.get_children.return_value = [valid_artist]

    f._draw_animated()

    mock_ax.draw_artist.assert_called_once_with(valid_artist)


def test_remove_right_pointer_resets_blit_background():
    """Verify remove_right_pointer() invalidates blit cache and repaints."""
    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    s._plot.add_right_pointer()
    s._plot.signal_plot._background = "stale"
    s._plot.remove_right_pointer()
    # After the fix, render_figure() captures a fresh background.
    assert s._plot.signal_plot._background is not None


def test_close_right_axis_resets_blit_background():
    """Verify close_right_axis() invalidates blit cache and repaints."""
    s = Signal1D(np.random.random((10, 20, 100)))
    s.plot()
    s._plot.signal_plot.create_right_axis()
    s._plot.signal_plot._background = "stale"
    s._plot.signal_plot.close_right_axis()
    # After the fix, render_figure() captures a fresh background.
    assert s._plot.signal_plot._background is not None


def test_on_close_iterates_marker_copy():
    """Verify _on_close uses list() copy to avoid mutation during iteration."""
    f = BlittedFigure()
    f.figure = mock.MagicMock()
    f.ax = mock.MagicMock()
    f.events = Events()
    f.events.closed = Event("", arguments=["obj"])

    marker1 = mock.MagicMock()
    marker2 = mock.MagicMock()
    f.ax_markers = [marker1, marker2]

    f.close()

    marker1.close.assert_called_once_with(render_figure=False)
    marker2.close.assert_called_once_with(render_figure=False)


def test_histogram_tile_plot_close_calls_super():
    """Verify HistogramTilePlot.close() delegates to BlittedFigure.close()."""
    htp = HistogramTilePlot()
    # HistogramTilePlot.__init__ bypasses super().__init__(),
    # so initialise inherited attributes manually.
    htp.ax_markers = []
    htp.events = Events()
    htp.events.closed = Event("", arguments=["obj"])
    htp._background = None
    htp.create_figure()
    htp.close()
    # _draw_event_cid is disconnected only through BlittedFigure._on_close(),
    # confirming super().close() was called.
    assert htp._draw_event_cid is None
    assert htp._background is None
    assert htp.figure is None
