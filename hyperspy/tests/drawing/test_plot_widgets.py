# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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
from hyperspy.drawing import widgets
from hyperspy.misc.test_utils import mock_event
from hyperspy.signals import Signal1D, Signal2D

baseline_dir = "plot_widgets"
default_tol = 2.0
style_pytest_mpl = "default"


class TestPlotLine2DWidget:
    def setup_method(self, method):
        # Create test image 100x100 pixels:
        self.im = Signal2D(np.arange(10000).reshape([100, 100]))
        self.im.axes_manager[0].scale = 1.2
        self.im.axes_manager[1].scale = 1.2
        self.line2d = widgets.Line2DWidget(self.im.axes_manager)

    def test_init(self):
        assert self.line2d.axes_manager == self.im.axes_manager
        assert self.line2d.linewidth == 1
        assert self.line2d.color == "red"
        assert self.line2d._size == np.array([0])
        np.testing.assert_allclose(self.line2d._pos, np.array([[0, 0], [1.2, 0]]))

        assert self.line2d.position == ([0.0, 0.0], [1.2, 0.0])
        np.testing.assert_allclose(self.line2d.indices[0], np.array([0, 0]))
        np.testing.assert_allclose(self.line2d.indices[1], np.array([1, 0]))
        np.testing.assert_allclose(self.line2d.get_centre(), np.array([0.6, 0.0]))

    def test_position(self):
        self.line2d.position = ([12.0, 60.0], [36.0, 96.0])
        assert self.line2d.position == ([12.0, 60.0], [36.0, 96.0])
        np.testing.assert_allclose(self.line2d.indices[0], np.array([10, 50]))
        np.testing.assert_allclose(self.line2d.indices[1], np.array([30, 80]))
        np.testing.assert_allclose(self.line2d.get_centre(), np.array([24.0, 78.0]))

    def test_position_snap_position(self):
        self.line2d.snap_position = True
        self.line2d.position = ([12.5, 61.0], [36.0, 96.0])
        np.testing.assert_allclose(self.line2d.position, ([12.0, 61.2], [36.0, 96.0]))
        np.testing.assert_allclose(self.line2d.indices[0], np.array([10, 51]))
        np.testing.assert_allclose(self.line2d.indices[1], np.array([30, 80]))
        np.testing.assert_allclose(self.line2d.get_centre(), np.array([24.0, 78.6]))

    def test_indices(self):
        self.line2d.indices = ([10, 50], [30, 80])
        np.testing.assert_allclose(self.line2d.indices[0], np.array([10, 50]))
        np.testing.assert_allclose(self.line2d.indices[1], np.array([30, 80]))
        assert self.line2d.position == ([12.0, 60.0], [36.0, 96.0])
        np.testing.assert_allclose(self.line2d.get_centre(), np.array([24.0, 78.0]))

    def test_length(self):
        x = 10
        self.line2d.position = ([10.0, 10.0], [10.0 + x, 10.0])
        assert self.line2d.get_line_length() == x

        y = 20
        self.line2d.position = ([20.0, 10.0], [20.0 + x, 10 + y])
        np.testing.assert_almost_equal(
            self.line2d.get_line_length(), np.sqrt(x**2 + y**2)
        )

    def test_change_size(self):
        # Need to plot the signal to set the mpl axis to the widget
        self.im.plot()
        self.line2d.set_mpl_ax(self.im._plot.signal_plot.ax)

        self.line2d.position = ([0.0, 0.0], [50.0, 50.0])
        assert self.line2d.size == (0,)
        self.line2d.increase_size()
        assert self.line2d.size == (1.2,)
        self.line2d.increase_size()
        assert self.line2d.size == (2.4,)
        self.line2d.decrease_size()
        assert self.line2d.size == (1.2,)

        self.line2d.size = (4.0,)
        assert self.line2d.size == (4.0,)

    def test_change_size_snap_size(self):
        # Need to plot the signal to set the mpl axis to the widget
        self.im.plot()
        self.line2d.set_mpl_ax(self.im._plot.signal_plot.ax)

        self.line2d.snap_size = True
        self.line2d.position = ([12.0, 60.0], [36.0, 96.0])
        assert self.line2d.position == ([12.0, 60.0], [36.0, 96.0])
        np.testing.assert_allclose(self.line2d.indices[0], np.array([10, 50]))
        np.testing.assert_allclose(self.line2d.indices[1], np.array([30, 80]))
        np.testing.assert_allclose(self.line2d.get_centre(), np.array([24.0, 78.0]))
        assert self.line2d.size == np.array([0])

        self.line2d.size = [3]
        np.testing.assert_allclose(self.line2d.size, np.array([2.4]))
        self.line2d.size = (5,)
        np.testing.assert_allclose(self.line2d.size, np.array([4.8]))
        self.line2d.size = np.array([7.4])
        np.testing.assert_allclose(self.line2d.size, np.array([7.2]))
        self.line2d.increase_size()
        np.testing.assert_allclose(self.line2d.size, np.array([8.4]))

    def test_change_size_snap_size_different_scale(self):
        self.line2d.axes[0].scale = 0.8
        assert self.line2d.axes[0].scale == 0.8
        assert self.line2d.axes[1].scale == 1.2
        self.line2d.snap_size = True
        # snapping size with the different axes scale is not supported
        assert self.line2d.snap_size is False

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl
    )
    def test_plot_line2d(self):
        self.im.plot()
        self.line2d.color = "green"
        self.line2d.position = ([12.0, 60.0], [36.0, 96.0])
        self.line2d.set_mpl_ax(self.im._plot.signal_plot.ax)
        assert self.line2d.ax == self.im._plot.signal_plot.ax

        line2d = widgets.Line2DWidget(self.im.axes_manager)
        line2d.snap_position = True
        line2d.set_mpl_ax(self.im._plot.signal_plot.ax)
        line2d.position = ([40.0, 20.0], [96.0, 36.0])
        line2d.linewidth = 4
        line2d.size = (15.0,)
        assert line2d.size == (15.0,)

        line2d_snap_all = widgets.Line2DWidget(self.im.axes_manager)
        line2d_snap_all.snap_all = True
        line2d_snap_all.set_mpl_ax(self.im._plot.signal_plot.ax)
        line2d_snap_all.position = ([50.0, 60.0], [96.0, 54.0])
        np.testing.assert_allclose(line2d_snap_all.position[0], [50.4, 60.0])
        np.testing.assert_allclose(line2d_snap_all.position[1], [96.0, 54.0])

        line2d_snap_all.size = (15.0,)
        np.testing.assert_allclose(line2d_snap_all.size[0], 14.4)
        np.testing.assert_allclose(line2d_snap_all.size[0], 14.4)

        return self.im._plot.signal_plot.figure


class TestPlotCircleWidget:
    def setup_method(self, method):
        # Create test image 100x100 pixels:
        N = 100
        im = Signal2D(np.arange(N**2).reshape([N] * 2))
        im.axes_manager[0].scale = 1.2
        im.axes_manager[1].scale = 1.2
        circle = widgets.CircleWidget(im.axes_manager)
        self.im = im
        self.circle = circle

    def test_change_size_snap_size(self):
        # Need to plot the signal to set the mpl axis to the widget
        im = self.im
        circle = self.circle
        im.plot()
        circle.set_mpl_ax(im._plot.signal_plot.ax)
        circle.snap_all = True

        circle.position = (10, 10)
        circle.size = (5, 1.0)
        assert circle.position == (9.6, 9.6)
        np.testing.assert_allclose(circle.size, (5.4, 0.6))

        circle.decrease_size()
        np.testing.assert_allclose(circle.size, (4.2, 0.0))
        circle.decrease_size()
        np.testing.assert_allclose(circle.size, (3.0, 0.0))

        circle.increase_size()
        np.testing.assert_allclose(circle.size, (4.2, 0.0))

        circle.size = (5, 1.0)
        circle.increase_size()
        np.testing.assert_allclose(circle.size, (6.6, 1.8))

    def test_change_size(self):
        im = self.im
        circle = self.circle
        im.plot()
        circle.set_mpl_ax(im._plot.signal_plot.ax)
        circle.snap_all = False

        position, size = (10, 10), (5, 2.5)
        circle.position = position
        circle.size = size
        assert circle.position == position
        assert circle.size == size


class TestPlotPolygonWidget:
    def setup_method(self, method):
        # Create test image 100x100 pixels:
        N = 100
        im = Signal2D(np.arange(N**2).reshape([N] * 2))
        im.axes_manager[0].scale = 1.2
        im.axes_manager[1].scale = 1.2
        polygon = widgets.PolygonWidget(im.axes_manager)
        self.im = im
        self.polygon = polygon

    def test_polygon_setup(self):
        # Test default matplotlib axes
        im_4d = Signal2D(np.arange(4**4).reshape([4] * 4))
        im_4d.plot()
        polygon = widgets.PolygonWidget(im_4d.axes_manager)
        polygon.set_mpl_ax(im_4d._plot.navigator_plot.ax)
        assert polygon.axes == im_4d.axes_manager.navigation_axes[0:2]
        assert polygon.ax is im_4d._plot.navigator_plot.ax

        im_3d = im_4d.inav[0]
        im_3d.plot()
        polygon = widgets.PolygonWidget(im_3d.axes_manager)
        polygon.set_mpl_ax(im_3d._plot.signal_plot.ax)
        assert polygon.axes == im_3d.axes_manager.signal_axes[0:2]
        assert polygon.ax is im_3d._plot.signal_plot.ax

        im_2d = im_3d.inav[0]
        im_2d.plot()
        polygon = widgets.PolygonWidget(im_2d.axes_manager)
        polygon.set_mpl_ax(im_2d._plot.signal_plot.ax)
        assert polygon.axes == im_2d.axes_manager.signal_axes[0:2]
        assert polygon.ax is im_2d._plot.signal_plot.ax

        # Needs at least two axes to work
        im_1d = im_2d.isig[0]
        im_2d.plot()
        with pytest.raises(ValueError):
            widgets.PolygonWidget(im_1d.axes_manager)

        im_2d_navsig = im_3d.isig[0]
        polygon = widgets.PolygonWidget(im_2d_navsig.axes_manager)
        assert (
            polygon.axes
            == im_2d_navsig.axes_manager.signal_axes
            + im_2d_navsig.axes_manager.navigation_axes
        )

    def test_set_vertices(self):
        polygon = self.polygon
        im = self.im

        im.plot()
        polygon.set_mpl_ax(im._plot.signal_plot.ax)

        # Defaults
        assert polygon.get_vertices() == []
        assert polygon.get_centre() == tuple()

        verts = [(31, 41), (15, 92), (65, 35)]
        polygon.set_vertices(verts)
        np.testing.assert_allclose(polygon.get_vertices(), verts)
        assert polygon.get_centre() == (40.0, 63.5)

        verts = np.arange(100).reshape((50, 2))
        verts[::2, 1] = 0
        polygon.set_vertices(verts)
        np.testing.assert_allclose(polygon.get_vertices(), verts)
        assert polygon.get_centre() == (49.0, 49.5)

    def test_set_on(self):
        polygon = self.polygon
        im = self.im

        assert polygon.ax is None
        assert polygon._is_on

        im.plot()
        polygon.set_mpl_ax(im._plot.signal_plot.ax)

        assert polygon.ax == im._plot.signal_plot.ax
        assert polygon._is_on

        polygon.set_on(False, render_figure=True)
        assert polygon.ax is None
        assert not polygon._is_on


class TestPlotRangeWidget:
    def setup_method(self, method):
        s = Signal1D(np.arange(50))
        s.axes_manager[0].scale = 1.2
        range_widget = widgets.RangeWidget(s.axes_manager)
        self.s = s
        self.range_widget = range_widget

    def test_snap_position_span_None(self):
        # When span is None, there shouldn't an error
        assert self.range_widget.span is None
        self.range_widget.snap_position = True
        assert self.range_widget.snap_position

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl
    )
    def test_plot_range(self):
        s = self.s
        range_widget = self.range_widget
        s.plot()
        range_widget.set_mpl_ax(s._plot.signal_plot.ax)
        assert range_widget.ax == s._plot.signal_plot.ax
        assert range_widget.color == "r"  # default color
        assert range_widget.position == (0.0,)
        assert range_widget.size == (1.2,)
        assert range_widget.span.artists[0].get_alpha() == 0.25

        w = widgets.RangeWidget(s.axes_manager, color="blue")
        w.set_mpl_ax(s._plot.signal_plot.ax)
        w.set_ibounds(left=4, width=3)
        assert w.color == "blue"
        color_rgba = matplotlib.colors.to_rgba("blue", alpha=0.25)
        assert w.span.artists[0].get_fc() == color_rgba
        assert w.span.artists[0].get_ec() == color_rgba
        np.testing.assert_allclose(w.position[0], 4.8)
        np.testing.assert_allclose(w.size[0], 3.6)

        w2 = widgets.RangeWidget(s.axes_manager)
        w2.set_mpl_ax(s._plot.signal_plot.ax)
        assert w2.ax == s._plot.signal_plot.ax

        w2.set_bounds(left=24.0, width=12.0)
        assert w2.position[0] == 24.0
        assert w2.size[0] == 12.0
        w2.color = "green"
        assert w2.color == "green"
        w2.alpha = 0.25
        assert w2.alpha == 0.25

        return s._plot.signal_plot.figure

    @pytest.mark.parametrize("render_figure", [True, False])
    def test_set_on(self, render_figure):
        s = self.s
        range_widget = self.range_widget
        s.plot()
        range_widget.ax = s._plot.signal_plot.ax
        range_widget._is_on = False

        range_widget.set_on(True, render_figure=render_figure)
        assert range_widget.span.get_visible()

        range_widget.set_on(False, render_figure=render_figure)
        assert range_widget.span is None
        assert range_widget.ax is None

    def test_update(self):
        s = self.s
        range_widget = self.range_widget
        s.plot()
        range_widget.set_mpl_ax(s._plot.signal_plot.ax)
        range_widget.span.update()

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl
    )
    def test_plot_range_Signal2D(self):
        im = Signal2D(np.arange(10 * 10).reshape((10, 10)))
        im.axes_manager[0].scale = 0.1
        im.axes_manager[1].scale = 5
        im.plot()

        range_h = widgets.RangeWidget(im.axes_manager, direction="horizontal")
        range_h.set_mpl_ax(im._plot.signal_plot.ax)

        range_v = widgets.RangeWidget(
            im.axes_manager, direction="vertical", color="blue"
        )
        range_v.axes = (im.axes_manager[1],)
        range_v.set_mpl_ax(im._plot.signal_plot.ax)
        assert range_v.position == (0.0,)
        assert range_v.size == (5.0,)

        range_v.set_bounds(left=20.0, width=15.0)
        assert range_v.position == (20.0,)
        assert range_v.size == (15.0,)

        return im._plot.signal_plot.figure


class TestSquareWidget:
    @pytest.mark.parametrize("button", ("right-click", "left-click"))
    def test_jump_click(self, button):
        im = Signal2D(np.arange(10 * 10 * 10 * 10).reshape((10, 10, 10, 10)))
        im.axes_manager[0].scale = 0.1
        im.axes_manager[1].scale = 5
        im.plot()

        jump = mock_event(
            im._plot.navigator_plot.figure,
            im._plot.navigator_plot.figure.canvas,
            key="shift",
            button=button,
            xdata=0.5,
            ydata=10,
        )
        im._plot.pointer._onjumpclick(event=jump)
        current_index = [el.index for el in im.axes_manager.navigation_axes]
        assert current_index == [5, 2]

    def test_jump_click_single_trigger(self):
        im = Signal2D(np.arange(10 * 10 * 10 * 10).reshape((10, 10, 10, 10)))
        im.axes_manager[0].scale = 0.1
        im.axes_manager[1].scale = 5
        im.plot()

        def count_calls(obj):
            count_calls.counter += 1

        count_calls.counter = 0

        im.axes_manager.events.indices_changed.connect(count_calls)

        jump = mock_event(
            im._plot.navigator_plot.figure,
            im._plot.navigator_plot.figure.canvas,
            key="shift",
            button="left-click",
            xdata=0.5,
            ydata=10,
        )
        im._plot.pointer._onjumpclick(event=jump)
        assert count_calls.counter == 1
        current_index = [el.index for el in im.axes_manager.navigation_axes]
        assert current_index == [5, 2]

    def test_jump_click_1d(self):
        im = Signal1D(np.arange(10 * 10).reshape((10, 10)))
        im.axes_manager[0].scale = 0.1
        im.plot()
        jump = mock_event(
            im._plot.navigator_plot.figure,
            im._plot.navigator_plot.figure.canvas,
            key="shift",
            button="left-click",
            xdata=1,
            ydata=0.2,
        )
        im._plot.pointer._onjumpclick(event=jump)
        current_index = [el.index for el in im.axes_manager.navigation_axes]
        assert current_index == [2]

    def test_jump_click_1d_vertical(self):
        im = Signal2D(np.arange(10 * 10 * 10).reshape((10, 10, 10)))
        im.axes_manager[0].scale = 0.1
        im.plot()
        jump = mock_event(
            im._plot.navigator_plot.figure,
            im._plot.navigator_plot.figure.canvas,
            key="shift",
            button="left-click",
            xdata=0.2,
        )
        im._plot.pointer._onjumpclick(event=jump)
        current_index = [el.index for el in im.axes_manager.navigation_axes]
        assert current_index == [2]

    def test_jump_click_out_of_bounds(self):
        im = Signal2D(np.arange(10 * 10 * 10 * 10).reshape((10, 10, 10, 10)))
        im.axes_manager[0].scale = 0.1
        im.axes_manager[1].scale = 5
        im.plot()

        jump = mock_event(
            im._plot.navigator_plot.figure,
            im._plot.navigator_plot.figure.canvas,
            key="shift",
            button="left-click",
            xdata=-5,
            ydata=100,
        )
        im._plot.pointer._onjumpclick(event=jump)
        current_index = [el.index for el in im.axes_manager.navigation_axes]
        assert current_index == [0, 9]  # maybe this should fail and return [0,0]

    def test_drag_continuous_update(self):
        im = Signal2D(np.arange(10 * 10 * 10 * 10).reshape((10, 10, 10, 10)))
        im.axes_manager[0].scale = 1
        im.axes_manager[1].scale = 1
        im.plot()

        def count_calls(obj):
            count_calls.counter += 1

        count_calls.counter = 0
        im.axes_manager.events.indices_changed.connect(count_calls)

        widget = im._plot.pointer
        pick = mock_event(
            im._plot.navigator_plot.figure,
            im._plot.navigator_plot.figure.canvas,
            key=None,
            button="left-click",
            xdata=0.5,
            ydata=0.5,
            artist=widget.patch[0],
        )
        widget.onpick(pick)
        assert widget.picked
        drag_events = []
        for i in np.linspace(0.5, 5, 40):
            drag = mock_event(
                im._plot.navigator_plot.figure,
                im._plot.navigator_plot.figure.canvas,
                key=None,
                button="left-click",
                xdata=i,
                ydata=0.5,
                artist=widget.patch[0],
            )
            drag_events.append(drag)
        for d in drag_events:
            widget._onmousemove(d)
        assert count_calls.counter == 5
        assert im.axes_manager.navigation_axes[0].index == 5
        assert im.axes_manager.navigation_axes[1].index == 0

    def test_drag_continuous_update1d(self):
        im = Signal2D(np.arange(10 * 10 * 10).reshape((10, 10, 10)))
        im.axes_manager[0].scale = 1
        im.plot()

        def count_calls(obj):
            count_calls.counter += 1

        count_calls.counter = 0
        im.axes_manager.events.indices_changed.connect(count_calls)

        widget = im._plot.pointer
        pick = mock_event(
            im._plot.navigator_plot.figure,
            im._plot.navigator_plot.figure.canvas,
            key=None,
            button="left-click",
            xdata=0.5,
            ydata=0.5,
            artist=widget.patch[0],
        )
        widget.onpick(pick)
        assert widget.picked
        drag_events = []
        for i in np.linspace(0.5, 5, 40):
            drag = mock_event(
                im._plot.navigator_plot.figure,
                im._plot.navigator_plot.figure.canvas,
                key=None,
                button="left-click",
                xdata=i,
                ydata=0.5,
                artist=widget.patch[0],
            )
            drag_events.append(drag)
        for d in drag_events:
            widget._onmousemove(d)
        assert count_calls.counter == 5
        assert im.axes_manager.navigation_axes[0].index == 5

    def test_drag_continuous_update1d_no_change(self):
        # drag down and check that it doesn't change the index
        im = Signal2D(np.arange(10 * 10 * 10).reshape((10, 10, 10)))
        im.axes_manager[0].scale = 1
        im.plot()

        def count_calls(obj):
            count_calls.counter += 1

        count_calls.counter = 0
        im.axes_manager.events.indices_changed.connect(count_calls)

        widget = im._plot.pointer
        pick = mock_event(
            im._plot.navigator_plot.figure,
            im._plot.navigator_plot.figure.canvas,
            key=None,
            button="left-click",
            xdata=0.5,
            ydata=0.5,
            artist=widget.patch[0],
        )
        widget.onpick(pick)
        assert widget.picked
        drag_events = []
        for i, j in zip(np.linspace(0.5, 5, 40), np.linspace(0.0, 0.49, 40)):
            drag = mock_event(
                im._plot.navigator_plot.figure,
                im._plot.navigator_plot.figure.canvas,
                key=None,
                button="left-click",
                xdata=j,
                ydata=i,
                artist=widget.patch[0],
            )
            drag_events.append(drag)
        for d in drag_events:
            widget._onmousemove(d)
        assert count_calls.counter == 0
        assert im.axes_manager.navigation_axes[0].index == 0
        # drag down and check that it doesn't change


def test_widgets_non_uniform_axis():
    s = hs.data.luminescence_signal(uniform=False).isig[10:20]
    roi = hs.roi.SpanROI()
    s.plot()
    roi.add_widget(s)
    np.testing.assert_allclose(roi.left, 1.61375935)
    np.testing.assert_allclose(roi.right, 1.61887959)

    roi2 = hs.roi.SpanROI()
    with pytest.raises(ValueError):
        roi2.add_widget(s, snap=True)

    w = list(roi.widgets)[0]
    with pytest.raises(ValueError):
        w.snap_size = True
