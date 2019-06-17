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
import pytest

from hyperspy.signals import Signal2D, Signal1D
from hyperspy.utils import roi


BASELINE_DIR = 'plot_roi'
DEFAULT_TOL = 2.0
STYLE_PYTEST_MPL = 'default'


def _transpose_space(space, im):
    if space == "signal":
        im = im
        axes = im.axes_manager.signal_axes
        im.plot()
        figure = im._plot.signal_plot.figure
    else:
        im = im.T
        axes = im.axes_manager.navigation_axes
        im.plot()
        figure = im._plot.navigator_plot.figure
    return {
        "im": im,
        "figure": figure,
        "axes": axes,
    }


class TestPlotROI():

    def setup_method(self, method):
        # Create test image 100x100 pixels:
        self.im = Signal2D(np.arange(50000).reshape([10, 50, 100]))
        self.im.axes_manager[0].scale = 1e-1
        self.im.axes_manager[1].scale = 1e-2
        self.im.axes_manager[2].scale = 1e-3

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_point1D_axis_0(self):
        self.im.plot()
        p = roi.Point1DROI(0.5)
        p.add_widget(signal=self.im, axes=[0, ], color="cyan")
        return self.im._plot.navigator_plot.figure

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_point1D_axis_1(self):
        self.im.plot()
        p = roi.Point1DROI(0.05)
        p.add_widget(signal=self.im, axes=[1, ], color="cyan")
        return self.im._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_point1D_axis_2(self):
        self.im.plot()
        p = roi.Point1DROI(0.005)
        p.add_widget(signal=self.im, axes=[2, ], color="cyan")
        return self.im._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_spanroi_axis_0(self):
        self.im.plot()
        p = roi.SpanROI(0.5, 0.7)
        p.add_widget(signal=self.im, axes=[0, ], color="cyan")
        return self.im._plot.navigator_plot.figure

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_spanroi_axis_1(self):
        self.im.plot()
        p = roi.SpanROI(0.05, 0.07)
        p.add_widget(signal=self.im, axes=[1, ], color="cyan")
        return self.im._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_spanroi_axis_2(self):
        self.im.plot()
        p = roi.SpanROI(0.005, 0.007)
        p.add_widget(signal=self.im, axes=[2, ], color="cyan")
        return self.im._plot.signal_plot.figure

    @pytest.mark.parametrize("space", ("signal", "navigation"))
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_point2D(self, space):
        objs = _transpose_space(im=self.im, space=space)
        p = roi.Point2DROI(0.05, 0.01)
        p.add_widget(signal=objs["im"], axes=objs["axes"], color="cyan")
        return objs["figure"]

    @pytest.mark.parametrize("space", ("signal", "navigation"))
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_circle_roi(self, space):
        self.im.axes_manager[2].scale = 0.01
        objs = _transpose_space(im=self.im, space=space)
        p = roi.CircleROI(cx=0.1, cy=0.1, r=0.1)
        p.add_widget(signal=objs["im"], axes=objs["axes"], color="cyan")
        return objs["figure"]

    @pytest.mark.parametrize("space", ("signal", "navigation"))
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_rectangular_roi(self, space):
        objs = _transpose_space(im=self.im, space=space)
        p = roi.RectangularROI(left=0.01, top=0.01, right=0.1, bottom=0.03)
        p.add_widget(signal=objs["im"], axes=objs["axes"], color="cyan")
        return objs["figure"]

    @pytest.mark.parametrize("space", ("signal", "navigation"))
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_line2d_roi(self, space):
        objs = _transpose_space(im=self.im, space=space)
        p = roi.Line2DROI(x1=0.01, y1=0.01, x2=0.1, y2=0.03)
        p.add_widget(signal=objs["im"], axes=objs["axes"], color="cyan")
        return objs["figure"]


class TestROIsManager:

    def setup_method(self):
        s = Signal1D(np.arange(100))
        self.s = s

    def test_add_widget(self):
        s = self.s
        p = roi.SpanROI(25.0, 45.0)
        s.plot()
        p.add_widget(s)
        assert len(s.rois_manager) == 1

    def test_add_widgets_list(self):
        s = self.s
        assert len(s.rois_manager) == 0
        p0 = roi.SpanROI(25.0, 45.0)
        p1 = roi.SpanROI(50.0, 60.0)
        s.add_ROIs([p0, p1])
        assert len(s.rois_manager) == 2
        assert s.rois_manager[0] == p0
        assert s.rois_manager[1] == p1

    def test_add_widgets_list_to_dictionary(self):
        s = self.s
        assert len(s.rois_manager) == 0
        p = roi.SpanROI(25.0, 45.0)
        roi_dict = p.to_dictionary()
        assert roi_dict == {'roi_type': 'SpanROI',
                            'name': '',
                            'parameters': {'left': 25.0, 'right': 45.0},
                            'properties':
                                {'color': 'green', 
                                 'alpha': 0.5, 
                                 'snap_position': True},
                            'on_signal': True}

    @pytest.mark.parametrize("plot", (True, False))
    def test_add_widgets_list_from_dictionary(self, plot):
        s = self.s
        if plot:
            s.plot()
        assert len(s.rois_manager) == 0
        p0 = roi.SpanROI(25.0, 45.0)
        roi_dict0 = p0.to_dictionary()
        roi_dict1 = p0.to_dictionary()
        roi_dict1['parameters'] = {'left': 75.0, 'right': 85.0}
        roi_dict1['properties']['color'] = 'blue'
        roi_dict1['properties']['alpha'] = 0.25
        s.add_ROIs([roi_dict0, roi_dict1])
        assert len(s.rois_manager) == 2
        roi0 = s.rois_manager[0]
        roi1 = s.rois_manager[1]
        assert roi0.color == 'green'
        assert roi0.alpha == 0.5
        assert roi1.color == 'blue'
        assert roi1.alpha == 0.25

    @pytest.mark.parametrize("plot", (True, False))
    def test_add_widgets_list_from_dictionary_properties(self, plot):
        s = self.s
        if plot:
            s.plot()
        p0 = roi.SpanROI(25.0, 45.0)
        roi_dict1 = p0.to_dictionary()
        roi_dict1['parameters'] = {'left': 75.0, 'right': 85.0}
        roi_dict1['properties']['color'] = 'blue'
        roi_dict1['properties']['alpha'] = 0.25

    @pytest.mark.parametrize("plot", (True, False))
    def test_indexation(self, plot):
        s = self.s
        if plot:
            s.plot()
        p0 = roi.SpanROI(25.0, 45.0)
        p1 = roi.SpanROI(60.0, 70.0)
        s.add_ROIs([p0, p1])
        assert s.rois_manager[0] == p0
        assert s.rois_manager[1] == p1
        s.rois_manager[0:1]

    def test_Point1DROI(self):
        s = self.s
        roi0 = roi.Point1DROI(25.0)
        roi_dict1 = roi0.to_dictionary()
        roi_dict1['parameters'] = {'value': 75.0}
        s.add_ROIs([roi0, roi_dict1])
        assert len(s.rois_manager) == 2
        s.rois_manager[0].value = 25.0
        s.rois_manager[1].value = 75.0

    def test_Point2DROI(self):
        im = Signal2D(np.arange(100).reshape(10, 10))
        roi0 = roi.Point2DROI(5.0, 4.0)
        roi_dict1 = roi0.to_dictionary()
        roi_dict1['parameters'] = {'x': 2.0, 'y':3.0}
        im.add_ROIs([roi0, roi_dict1])
        assert len(im.rois_manager) == 2
        im.rois_manager[0].x = 5
        im.rois_manager[0].y = 4
        im.rois_manager[1].x = 2
        im.rois_manager[1].y = 3

    # def test_RectangularROI(self):
    #     im = Signal2D(np.arange(100).reshape(10, 10))
    #     roi0 = roi.RectangularROI(1, 3, 3, 1)
    #     roi_dict1 = roi0.to_dictionary()
    #     roi_dict1['parameters'] = {
    #             'left': 2, 'top':5, 'right':6, 'bottom':2}
    #     im.add_ROIs([roi0, roi_dict1])
    #     assert len(im.rois_manager) == 2
    #     im.rois_manager[0].left = 1
    #     im.rois_manager[0].top = 2
    #     im.rois_manager[0].right = 1
    #     im.rois_manager[0].bottom = 3
    #     im.rois_manager[1].left = 2
    #     im.rois_manager[1].top = 5
    #     im.rois_manager[1].right = 6
    #     im.rois_manager[1].bottom = 2

    # def test_CircleROI(self):
    #     im = Signal2D(np.arange(100).reshape(10, 10))
    #     roi0 = roi.CircleROI(4, 5, 1)
    #     roi_dict1 = roi0.to_dictionary()
    #     roi_dict1['parameters'] = {
    #             'cx': 2, 'cy':2, 'r':2, 'r_inner':4}
    #     im.add_ROIs([roi0, roi_dict1])
    #     assert len(im.rois_manager) == 2
    #     im.rois_manager[0].left = 1
    #     im.rois_manager[0].top = 2
    #     im.rois_manager[0].right = 1
    #     im.rois_manager[0].bottom = 3
    #     im.rois_manager[1].left = 2
    #     im.rois_manager[1].top = 5
    #     im.rois_manager[1].right = 6
    #     im.rois_manager[1].bottom = 2
