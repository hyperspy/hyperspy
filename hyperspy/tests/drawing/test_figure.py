# Copyright 2007-2016 The HyperSpy developers
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

from hyperspy.drawing.figure import BlittedFigure
from hyperspy.datasets.example_signals import EDS_TEM_Spectrum
from hyperspy.signals import Signal1D, Signal2D
from hyperspy._components.polynomial import Polynomial


def test_figure_title_length():
    f = BlittedFigure()
    f.title = "Test" * 50
    assert max([len(line) for line in f.title.split("\n")]) < 61


class TestCloseFigure():

    def _assert_figure_state_after_close(self, fig):
        assert len(fig.events.closed.connected) == 0
        assert fig._draw_event_cid is None
        assert fig.figure is None
        assert fig._background is None
        assert fig.ax is None
        # calling s._plot.close() should not raise an error after closing the plot
        fig.close()

    def test_close_figure_using_close_method(self):
        fig = BlittedFigure()
        fig.create_figure()
        assert fig.figure is not None
        fig.close()
        self._assert_figure_state_after_close(fig)

    def test_close_figure_using_matplotlib(self):
        # check that matplotlib callback to `_on_close` is working fine
        fig = BlittedFigure()
        fig.create_figure()
        assert fig.figure is not None
        # Close using matplotlib, similar to using gui
        fig.figure.canvas.close_event()
        self._assert_figure_state_after_close(fig)

    def test_close_figure_with_plotted_marker(self):
        s = EDS_TEM_Spectrum()
        s.plot(True)
        s._plot.close()
        self._assert_figure_state_after_close(s._plot.signal_plot)

    @pytest.mark.filterwarnings("ignore:The API of the `Polynomial`")
    @pytest.mark.parametrize('navigator', ["auto", "slider", "spectrum"])
    @pytest.mark.parametrize('nav_dim', [1, 2])
    @pytest.mark.parametrize('sig_dim', [1, 2])
    def test_close_figure(self, navigator, nav_dim, sig_dim):
        total_dim = nav_dim*sig_dim
        if sig_dim == 1:
            Signal = Signal1D
        elif sig_dim == 2:
            Signal = Signal2D            
        s = Signal(np.arange(pow(10, total_dim)).reshape([10]*total_dim))
        s.plot(navigator=navigator)
        s._plot.close()
        self._assert_figure_state_after_close(s._plot.signal_plot)
        assert s._plot.navigator_plot is None

        if sig_dim == 1:
            m = s.create_model()
            m.plot()
            # Close with matplotlib event
            m._plot.signal_plot.figure.canvas.close_event()
            m.extend([Polynomial(1)])
