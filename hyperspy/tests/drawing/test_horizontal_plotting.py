# Copyright 2007-2022 The HyperSpy developers
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
import matplotlib
import numpy as np
import pytest

import hyperspy.api as hs

ipympl = pytest.importorskip("ipympl")
ipywidgets = pytest.importorskip("ipywidgets")


# ipympl issue: https://github.com/matplotlib/ipympl/issues/236
# DeprecationWarning: Passing unrecognized arguments to
# super(Toolbar).__init__(). NavigationToolbar2WebAgg.__init__(
# missing 1 required positional argument: 'canvas'
@pytest.mark.filterwarnings("ignore:Passing unrecognized arguments to")
class TestIPYMPL:
    def test_horizontal(self, capsys):
        matplotlib.use("module://ipympl.backend_nbagg")
        s = hs.signals.Signal2D(np.random.random((4, 4, 2, 2)))
        s.plot(plot_style="horizontal")
        captured = capsys.readouterr()
        assert "HBox(children=(Canvas(" in captured.out

    def test_vertical(self, capsys):
        matplotlib.use("module://ipympl.backend_nbagg")
        s = hs.signals.Signal2D(np.random.random((4, 4, 2, 2)))
        s.plot(plot_style="vertical")
        captured = capsys.readouterr()

        assert "VBox(children=(Canvas(" in captured.out

    def test_only_signal(self, capsys):
        matplotlib.use("module://ipympl.backend_nbagg")
        s = hs.signals.Signal2D(np.random.random((2, 2)))
        s.plot()
        captured = capsys.readouterr()
        assert "Canvas(toolbar=Toolbar(" in captured.out

    def test_only_navigation(self, capsys):
        matplotlib.use("module://ipympl.backend_nbagg")
        s = hs.signals.Signal2D(np.random.random((2, 2))).T
        s.plot()
        captured = capsys.readouterr()
        assert "Canvas(toolbar=Toolbar(" in captured.out

    def test_warnings(
        self,
    ):
        with pytest.warns(UserWarning):
            s = hs.signals.Signal2D(np.random.random((4, 2, 2)))
            s.plot(plot_style="vertical")

    def test_misspelling(
        self,
    ):
        matplotlib.use("module://ipympl.backend_nbagg")
        with pytest.raises(ValueError):
            s = hs.signals.Signal2D(np.random.random((4, 2, 2)))
            s.plot(plot_style="Vert")
