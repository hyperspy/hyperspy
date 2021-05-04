# Copyright 2007-2021 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pytest

from hyperspy._components.polynomial import Polynomial
from hyperspy.datasets.example_signals import EDS_TEM_Spectrum
from hyperspy.drawing.figure import BlittedFigure
from hyperspy.misc.test_utils import check_closing_plot
from hyperspy.signals import Signal1D, Signal2D


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
    fig.figure.canvas.close_event()
    _assert_figure_state_after_close(fig)


def test_close_figure_with_plotted_marker():
    s = EDS_TEM_Spectrum()
    s.plot(True)
    s._plot.close()
    check_closing_plot(s)


@pytest.mark.filterwarnings("ignore:The API of the `Polynomial`")
@pytest.mark.parametrize('navigator', ["auto", "slider", "spectrum"])
@pytest.mark.parametrize('nav_dim', [1, 2])
@pytest.mark.parametrize('sig_dim', [1, 2])
def test_close_figure(navigator, nav_dim, sig_dim):
    total_dim = nav_dim*sig_dim
    if sig_dim == 1:
        Signal = Signal1D
    elif sig_dim == 2:
        Signal = Signal2D
    s = Signal(np.arange(pow(10, total_dim)).reshape([10]*total_dim))
    s.plot(navigator=navigator)
    s._plot.close()
    check_closing_plot(s, check_data_changed_close=False)

    if sig_dim == 1:
        m = s.create_model()
        m.plot()
        # Close with matplotlib event
        m._plot.signal_plot.figure.canvas.close_event()
        m.extend([Polynomial(1)])
