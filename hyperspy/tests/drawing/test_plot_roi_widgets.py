# -*- coding: utf-8 -*-
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
import numpy.testing as nt
import pytest
import matplotlib

from hyperspy.signals import Signal2D, Signal1D
from hyperspy.utils import roi


BASELINE_DIR = 'plot_roi'
DEFAULT_TOL = 2.0
STYLE_PYTEST_MPL = 'default'


class TestPlotROI():

    def setup_method(self, method):
        # Create test image 100x100 pixels:
        self.im = Signal2D(np.arange(50000).reshape([10, 50, 100]))
        self.im.axes_manager[0].scale = 1e-1
        self.im.axes_manager[1].scale = 1e-2
        self.im.axes_manager[2].scale = 1e-3

    @pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_point1D_axis_0(self, mpl_cleanup):
        self.im.plot()
        p = roi.Point1DROI(0.5)
        p.add_widget(signal=self.im, axes=[0,], color="cyan")
        return self.im._plot.navigator_plot.figure


    @pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_point1D_axis_1(self, mpl_cleanup):
        self.im.plot()
        p = roi.Point1DROI(0.05)
        p.add_widget(signal=self.im, axes=[1,], color="cyan")
        return self.im._plot.signal_plot.figure


    @pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_point1D_axis_2(self, mpl_cleanup):
        self.im.plot()
        p = roi.Point1DROI(0.005)
        p.add_widget(signal=self.im, axes=[2,], color="cyan")
        return self.im._plot.signal_plot.figure


    @pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_spanroi_axis_0(self, mpl_cleanup):
        self.im.plot()
        p = roi.SpanROI(0.5, 0.7)
        p.add_widget(signal=self.im, axes=[0,], color="cyan")
        return self.im._plot.navigator_plot.figure


    @pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_spanroi_axis_1(self, mpl_cleanup):
        self.im.plot()
        p = roi.SpanROI(0.05, 0.07)
        p.add_widget(signal=self.im, axes=[1,], color="cyan")
        return self.im._plot.signal_plot.figure


    @pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_spanroi_axis_2(self, mpl_cleanup):
        self.im.plot()
        p = roi.SpanROI(0.005, 0.007)
        p.add_widget(signal=self.im, axes=[2,], color="cyan")
        return self.im._plot.signal_plot.figure

