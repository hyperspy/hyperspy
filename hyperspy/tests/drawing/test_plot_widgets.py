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
from hyperspy.drawing import widgets


baseline_dir = 'plot_widgets'
default_tol = 2.0
style_pytest_mpl = 'default'


class TestPlotLine2DWidget():

    def setup_method(self, method):
        # Create test image 100x100 pixels:
        self.im = Signal2D(np.arange(10000).reshape([100, 100]))
        self.im.axes_manager[0].scale = 1.2
        self.im.axes_manager[1].scale = 1.2
        self.line2d = widgets.Line2DWidget(self.im.axes_manager)

    def test_init(self):
        assert self.line2d.axes_manager == self.im.axes_manager
        assert self.line2d.linewidth == 1
        assert self.line2d.color == 'red'
        assert self.line2d._size == np.array([0])
        nt.assert_allclose(self.line2d._pos, np.array([[0, 0], [1.2, 0]]))

        assert self.line2d.position == ([0.0, 0.0], [1.2, 0.0])
        nt.assert_allclose(self.line2d.indices[0], np.array([0, 0]))
        nt.assert_allclose(self.line2d.indices[1], np.array([1, 0]))
        nt.assert_allclose(self.line2d.get_centre(), np.array([0.6, 0.]))

    def test_position(self):
        self.line2d.position = ([12.0, 60.0], [36.0, 96.0])
        assert self.line2d.position == ([12.0, 60.0], [36.0, 96.0])
        nt.assert_allclose(self.line2d.indices[0], np.array([10, 50]))
        nt.assert_allclose(self.line2d.indices[1], np.array([30, 80]))
        nt.assert_allclose(self.line2d.get_centre(), np.array([24., 78.]))

    def test_position_snap_position(self):
        self.line2d.snap_position = True
        self.line2d.position = ([12.5, 61.0], [36.0, 96.0])
        nt.assert_allclose(self.line2d.position, ([12.0, 61.2], [36.0, 96.0]))
        nt.assert_allclose(self.line2d.indices[0], np.array([10, 51]))
        nt.assert_allclose(self.line2d.indices[1], np.array([30, 80]))
        nt.assert_allclose(self.line2d.get_centre(), np.array([24., 78.6]))

    def test_indices(self):
        self.line2d.indices = ([10, 50], [30, 80])
        nt.assert_allclose(self.line2d.indices[0], np.array([10, 50]))
        nt.assert_allclose(self.line2d.indices[1], np.array([30, 80]))
        assert self.line2d.position == ([12.0, 60.0], [36.0, 96.0])
        nt.assert_allclose(self.line2d.get_centre(), np.array([24., 78.]))

    def test_length(self):
        x = 10
        self.line2d.position = ([10.0, 10.0], [10.0 + x, 10.0])
        assert self.line2d.get_line_length() == x

        y = 20
        self.line2d.position = ([20.0, 10.0], [20.0 + x, 10 + y])
        nt.assert_almost_equal(self.line2d.get_line_length(),
                               np.sqrt(x**2 + y**2))

    def test_change_size(self, mpl_cleanup):
        # Need to plot the signal to set the mpl axis to the widget
        self.im.plot()
        self.line2d.set_mpl_ax(self.im._plot.signal_plot.ax)

        self.line2d.position = ([0.0, 0.0], [50.0, 50.0])
        assert self.line2d.size == (0, )
        self.line2d.increase_size()
        assert self.line2d.size == (1.2, )
        self.line2d.increase_size()
        assert self.line2d.size == (2.4, )
        self.line2d.decrease_size()
        assert self.line2d.size == (1.2, )

        self.line2d.size = (4.0, )
        assert self.line2d.size == (4.0, )

    def test_change_size_snap_size(self, mpl_cleanup):
        # Need to plot the signal to set the mpl axis to the widget
        self.im.plot()
        self.line2d.set_mpl_ax(self.im._plot.signal_plot.ax)

        self.line2d.snap_size = True
        self.line2d.position = ([12.0, 60.0], [36.0, 96.0])
        assert self.line2d.position == ([12.0, 60.0], [36.0, 96.0])
        nt.assert_allclose(self.line2d.indices[0], np.array([10, 50]))
        nt.assert_allclose(self.line2d.indices[1], np.array([30, 80]))
        nt.assert_allclose(self.line2d.get_centre(), np.array([24., 78.]))
        assert self.line2d.size == np.array([0])

        self.line2d.size = [3]
        nt.assert_allclose(self.line2d.size, np.array([2.4]))
        self.line2d.size = (5, )
        nt.assert_allclose(self.line2d.size, np.array([4.8]))
        self.line2d.size = np.array([7.4])
        nt.assert_allclose(self.line2d.size, np.array([7.2]))
        self.line2d.increase_size()
        nt.assert_allclose(self.line2d.size, np.array([8.4]))

    def test_change_size_snap_size_different_scale(self):
        self.line2d.axes[0].scale = 0.8
        assert self.line2d.axes[0].scale == 0.8
        assert self.line2d.axes[1].scale == 1.2
        self.line2d.snap_size = True
        # snapping size with the different axes scale is not supported
        assert self.line2d.snap_size == False

    @pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                                   tolerance=default_tol, style=style_pytest_mpl)
    def test_plot_line2d(self, mpl_cleanup):
        self.im.plot()
        self.line2d.color = 'green'
        self.line2d.position = ([12.0, 60.0], [36.0, 96.0])
        self.line2d.set_mpl_ax(self.im._plot.signal_plot.ax)
        assert self.line2d.ax == self.im._plot.signal_plot.ax

        line2d = widgets.Line2DWidget(self.im.axes_manager)
        line2d.snap_position = True
        line2d.set_mpl_ax(self.im._plot.signal_plot.ax)
        line2d.position = ([40.0, 20.0], [96.0, 36.0])
        line2d.linewidth = 4
        line2d.size = (15.0, )
        assert line2d.size == (15.0, )

        line2d_snap_all = widgets.Line2DWidget(self.im.axes_manager)
        line2d_snap_all.snap_all = True
        line2d_snap_all.set_mpl_ax(self.im._plot.signal_plot.ax)
        line2d_snap_all.position = ([50.0, 60.0], [96.0, 54.0])
        nt.assert_allclose(line2d_snap_all.position[0], [50.4, 60.0])
        nt.assert_allclose(line2d_snap_all.position[1], [96.0, 54.0])

        line2d_snap_all.size = (15.0, )
        nt.assert_allclose(line2d_snap_all.size[0], 14.4)
        nt.assert_allclose(line2d_snap_all.size[0], 14.4)

        return self.im._plot.signal_plot.figure


class TestPlotRangeWidget():

    def setup_method(self, method):
        self.s = Signal1D(np.arange(50))
        self.s.axes_manager[0].scale = 1.2
        self.range = widgets.RangeWidget(self.s.axes_manager)

    @pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                                   tolerance=default_tol, style=style_pytest_mpl)
    def test_plot_range(self, mpl_cleanup):
        self.s.plot()
        self.range.set_mpl_ax(self.s._plot.signal_plot.ax)
        assert self.range.ax == self.s._plot.signal_plot.ax
        assert self.range.color == 'red'  # default color
        assert self.range.position == (0.0, )
        assert self.range.size == (1.2, )
        assert self.range.span.rect.get_alpha() == 0.5

        w = widgets.RangeWidget(self.s.axes_manager, color='blue')
        w.set_mpl_ax(self.s._plot.signal_plot.ax)
        w.set_ibounds(left=4, width=3)
        assert w.color == 'blue'
        color_rgba = matplotlib.colors.to_rgba('blue', alpha=0.5)
        assert w.span.rect.get_fc() == color_rgba
        assert w.span.rect.get_ec() == color_rgba
        nt.assert_allclose(w.position[0], 4.8)
        nt.assert_allclose(w.size[0], 3.6)

        w2 = widgets.RangeWidget(self.s.axes_manager)
        w2.set_mpl_ax(self.s._plot.signal_plot.ax)
        assert w2.ax == self.s._plot.signal_plot.ax

        w2.set_bounds(left=24.0, width=12.0)
        w2.color = 'green'
        assert w2.color == 'green'
        w2.alpha = 0.25
        assert w2.alpha == 0.25

        return self.s._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                                   tolerance=default_tol, style=style_pytest_mpl)
    def test_plot_range_Signal2D(self, mpl_cleanup):
        im = Signal2D(np.arange(10 * 10).reshape((10, 10)))
        im.axes_manager[0].scale = 0.1
        im.axes_manager[1].scale = 5
        im.plot()

        range_h = widgets.RangeWidget(im.axes_manager, direction='horizontal')
        range_h.set_mpl_ax(im._plot.signal_plot.ax)

        range_v = widgets.RangeWidget(im.axes_manager, direction='vertical',
                                      color='blue')
        range_v.axes = (im.axes_manager[1],)
        range_v.set_mpl_ax(im._plot.signal_plot.ax)
        assert range_v.position == (0.0, )
        assert range_v.size == (5.0, )

        range_v.set_bounds(left=20.0, width=15.0)
        assert range_v.position == (20.0, )
        assert range_v.size == (15.0, )

        return im._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                                   tolerance=default_tol, style=style_pytest_mpl)
    def test_plot_ModifiableSpanSelector(self, mpl_cleanup):
        self.s.plot()
        from hyperspy.drawing._widgets.range import ModifiableSpanSelector
        ax = self.s._plot.signal_plot.ax
        span_v = ModifiableSpanSelector(ax, direction='vertical')
        span_v.set_initial((15, 20))
        assert span_v.range == (15, 20)

        span_v.range = (25, 30)
        assert span_v.range == (25, 30)

        span_h = ModifiableSpanSelector(ax, direction='horizontal',
                                        rectprops={'color': 'g', 'alpha': 0.2})
        color_rgba = matplotlib.colors.to_rgba('g', alpha=0.2)
        assert span_h.rect.get_fc() == color_rgba
        assert span_h.rect.get_ec() == color_rgba
        span_h.set_initial((50.4, 55.2))
        nt.assert_allclose(span_h.range[0], 50.4)
        nt.assert_allclose(span_h.range[1], 55.2)

        span_h.range = (40, 45)
        assert span_h.range == (40, 45)
        ax.figure.canvas.draw_idle()

        return self.s._plot.signal_plot.figure
