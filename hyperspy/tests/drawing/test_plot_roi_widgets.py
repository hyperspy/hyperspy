# -*- coding: utf-8 -*-
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

import numpy as np
import pytest

from hyperspy.signals import Signal1D, Signal2D
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
        im = Signal2D(np.arange(50000).reshape([10, 50, 100]))
        im.axes_manager[0].scale = 1e-1
        im.axes_manager[1].scale = 1e-2
        im.axes_manager[2].scale = 1e-3
        self.im = im

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_point1D_axis_0(self):
        im = self.im
        im.plot()
        p = roi.Point1DROI(0.5)
        p.add_widget(signal=im, axes=[0, ], color="cyan")
        return im._plot.navigator_plot.figure

    def test_plot_point1D_axis_0_non_iterable(self):
        self.im.plot()
        p = roi.Point1DROI(0.5)
        p.add_widget(signal=self.im, axes=0, color="cyan")

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_point1D_axis_1(self):
        im = self.im
        im.plot()
        p = roi.Point1DROI(0.05)
        p.add_widget(signal=im, axes=[1, ], color="cyan")
        return im._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_point1D_axis_2(self):
        im = self.im
        im.plot()
        p = roi.Point1DROI(0.005)
        p.add_widget(signal=im, axes=[2, ], color="cyan")
        return im._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_spanroi_axis_0(self):
        im = self.im
        im.plot()
        p = roi.SpanROI(0.5, 0.7)
        p.add_widget(signal=im, axes=[0, ], color="cyan")
        return im._plot.navigator_plot.figure

    def test_plot_spanroi_close(self):
        im = self.im
        im.plot()
        p = roi.SpanROI(0.5, 0.7)
        p.add_widget(signal=im, axes=[0, ], color="cyan")
        for widget in p.widgets:
            widget.close()

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_spanroi_axis_1(self):
        im = self.im
        im.plot()
        p = roi.SpanROI(0.05, 0.07)
        p.add_widget(signal=im, axes=[1, ], color="cyan")
        return im._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_spanroi_axis_2(self):
        im = self.im
        im.plot()
        p = roi.SpanROI(0.005, 0.007)
        p.add_widget(signal=im, axes=[2, ], color="cyan")
        return im._plot.signal_plot.figure

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
        p2 = roi.CircleROI(cx=0.3, cy=0.3, r=0.15, r_inner=0.05)
        p2.add_widget(signal=objs["im"], axes=objs["axes"])
        return objs["figure"]

    @pytest.mark.parametrize("space", ("signal", "navigation"))
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_rectangular_roi(self, space):
        objs = _transpose_space(im=self.im, space=space)
        p = roi.RectangularROI(left=0.01, top=0.01, right=0.1, bottom=0.03)
        p.add_widget(signal=objs["im"], axes=objs["axes"], color="cyan")
        return objs["figure"]

    @pytest.mark.parametrize("render_figure", [True, False])
    def test_plot_rectangular_roi_remove(self, render_figure):
        im = self.im
        im.plot()
        p = roi.RectangularROI(left=0.01, top=0.01, right=0.1, bottom=0.03)
        p.add_widget(signal=im)
        p.remove_widget(im, render_figure=render_figure)

    @pytest.mark.parametrize("space", ("signal", "navigation"))
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_line2d_roi(self, space):
        im = self.im
        objs = _transpose_space(im=im, space=space)
        p = roi.Line2DROI(x1=0.01, y1=0.01, x2=0.1, y2=0.03)
        p.add_widget(signal=objs["im"], axes=objs["axes"], color="cyan")
        p2 = roi.Line2DROI(x1=0.03, y1=0.015, x2=0.3, y2=0.03, linewidth=0.2)
        with pytest.raises(ValueError):
            p2.add_widget(signal=objs["im"], axes=objs["axes"])
        return objs["figure"]

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
    def test_plot_line2d_roi_linewidth(self):
        im = self.im
        for axis in im.axes_manager.signal_axes:
            axis.scale = 0.1
        objs = _transpose_space(im=im, space='signal')
        p = roi.Line2DROI(x1=0.3, y1=0.5, x2=6.0, y2=3.0, linewidth=0.5)
        p.add_widget(signal=objs["im"], axes=objs["axes"])

        p2 = roi.Line2DROI(x1=2.0, y1=0.5, x2=8.0, y2=3.0, linewidth=0.1)
        p2.add_widget(signal=objs["im"], axes=objs["axes"])
        widget2 = list(p2.widgets)[0]
        widget2.decrease_size()
        assert widget2.size == (0.0, )
        widget2.increase_size()
        assert widget2.size == (0.1, )

        p3 = roi.Line2DROI(x1=3.5, y1=0.5, x2=9.5, y2=3.0, linewidth=0.1)
        p3.add_widget(signal=objs["im"], axes=objs["axes"])
        widget3 = list(p3.widgets)[0]
        widget3.decrease_size()
        assert widget3.size == (0.0, )

        return objs["figure"]


def test_error_message():
    im = Signal2D(np.arange(50000).reshape([10, 50, 100]))
    im.plot()
    im._plot.close()
    p = roi.Point1DROI(0.5)
    with pytest.raises(Exception, match='does not have an active plot.'):
        p.add_widget(signal=im, axes=[0, ], color="cyan")


def test_remove_rois():
    s = Signal1D(np.arange(10))
    s2 = s.deepcopy()
    r = roi.SpanROI(2, 4)
    s.plot()
    s2.plot()

    _ = r.interactive(s)
    _ = r.interactive(s2)

    r.remove_widget(s)


@pytest.mark.parametrize('snap', [True, False])
def test_snapping_axis_values(snap):
    s = Signal2D(np.arange(100).reshape(10, 10))
    s.axes_manager[0].offset = 5

    r = roi.Line2DROI(x1=6, y1=0, x2=12, y2=4, linewidth=0)
    s.plot()
    _ = r.interactive(s, snap=snap)


def test_plot_span_roi_changed_event():
    s = Signal1D(np.arange(100))
    s.plot()
    r = roi.SpanROI()
    s_span = r.interactive(s)
    np.testing.assert_allclose(s_span.data, np.arange(25, 74))

    w = list(r.widgets)[0]
    assert w._pos == (24.5, )
    assert w._size == (50., )
    w._set_span_extents(10, 20)
    np.testing.assert_allclose(s_span.data, np.arange(9, 19))
