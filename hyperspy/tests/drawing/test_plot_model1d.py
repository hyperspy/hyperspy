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

from hyperspy.signals import Signal1D
from hyperspy.components1d import Expression

DEFAULT_TOL = 2.0
BASELINE_DIR = 'plot_model1d'
STYLE_PYTEST_MPL = 'default'


class TestModelPlot:
    def setup_method(self, method):
        s = Signal1D(np.arange(1000).reshape((10, 100)))
        np.random.seed(0)
        s.add_poissonian_noise()
        m = s.create_model()
        line = Expression("a * x", name="line", a=1)
        m.append(line)
        self.m = m

    @pytest.mark.mpl_image_compare(
        baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_default_signal_plot(self):
        self.m.plot()
        return self.m._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_components(self):
        self.m.plot(plot_components=True)
        return self.m._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_disable_plot_components(self):
        self.m.plot(plot_components=True)
        self.m.disable_plot_components()
        return self.m._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_default_navigator_plot(self):
        self.m.plot()
        return self.m._plot.navigator_plot.figure

    def test_no_navigator(self):
        self.m.plot(navigator=None)
        assert self.m.signal._plot.navigator_plot is None
