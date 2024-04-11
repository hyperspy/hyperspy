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

import numpy as np

from hyperspy import roi, signals
from hyperspy.drawing import widget, widgets
from hyperspy.misc.test_utils import mock_event


def test_get_step():
    s = signals.Signal1D(np.zeros((4, 4)))
    axis = s.axes_manager.navigation_axes[0]
    step = widget.ResizableDraggableWidgetBase._get_step(
        s, s.axes_manager.navigation_axes[0]
    )
    assert step == 1
    axis.index = 3
    step = widget.ResizableDraggableWidgetBase._get_step(
        s, s.axes_manager.navigation_axes[0]
    )
    assert step == 1


def test_scalebar_remove():
    im = signals.Signal2D(-np.arange(10000).reshape([100, 100]))
    for ax in im.axes_manager.signal_axes:
        ax.scale = 1.2
        ax.units = "nm"
    im.plot()
    assert im._plot.signal_plot.ax.scalebar is not None
    im._plot.signal_plot.ax.scalebar.remove()


def test_remove_widget_line():
    s = signals.Signal1D(np.arange(10 * 25).reshape(10, 25))
    s.plot()

    ax = s._plot.navigator_plot.ax
    assert len(ax.get_lines()) == 2
    assert isinstance(s._plot.pointer, widgets.HorizontalLineWidget)
    assert len(s._plot.pointer.patch) == 1

    # Remove pointer
    s._plot.pointer.close(render_figure=True)
    assert len(ax.lines) == 1
    assert len(s._plot.pointer.patch) == 1

    im = signals.Signal2D(np.arange(10 * 25 * 25).reshape(10, 25, 25))
    im.plot()

    ax = im._plot.navigator_plot.ax
    assert len(ax.get_lines()) == 2
    assert isinstance(im._plot.pointer, widgets.VerticalLineWidget)
    assert len(im._plot.pointer.patch) == 1

    # Remove pointer
    im._plot.pointer.close(render_figure=True)
    assert len(ax.lines) == 1
    assert len(im._plot.pointer.patch) == 1


def test_calculate_size():
    s = signals.Signal2D(np.arange(10000).reshape(10, 10, 10, 10))

    # Test that scalebar.calculate_size passes only positive value to closest_nice_number
    s.axes_manager[0].scale = -1
    s.plot()


def test_adding_removing_resizers_on_pick_event():
    """
    Test adding and removing resizers on pick events
    """
    s = signals.Signal2D(np.random.random((10, 10)))

    xx2, yy2, xx1, yy1 = 0, 0, 2, 2

    shiftx = 5
    shifty = 3

    rect_roi0 = roi.RectangularROI(xx2, yy2, xx1, yy1)
    rect_roi1 = roi.RectangularROI(
        xx2 + shiftx, yy2 + shifty, xx1 + shiftx, yy1 + shifty
    )
    s.plot()

    _ = rect_roi0.interactive(s)
    _ = rect_roi1.interactive(s)
    widget0 = list(rect_roi0.widgets)[0]
    widget1 = list(rect_roi1.widgets)[0]

    assert not widget0.picked
    assert not widget1.picked

    fig = s._plot.signal_plot.figure

    # PickEvent on widget0
    mouseevent0 = mock_event(fig, fig.canvas, xdata=1, ydata=1, artist=widget0.patch[0])
    pickevent0 = mock_event(
        fig, fig.canvas, artist=widget0.patch[0], mouseevent=mouseevent0
    )

    # PickEvent on widget1
    mouseevent1 = mock_event(fig, fig.canvas, xdata=6, ydata=4, artist=widget1.patch[0])
    pickevent1 = mock_event(
        fig, fig.canvas, artist=widget1.patch[0], mouseevent=mouseevent1
    )

    # PickEvent outside widget0 and widget1
    mouseevent2 = mock_event(fig, fig.canvas, xdata=8, ydata=8)
    pickevent2 = mock_event(fig, fig.canvas, mouseevent=mouseevent2)

    widget0.onpick(pickevent0)
    widget1.onpick(pickevent0)
    assert widget0.picked
    assert widget0._resizers_on
    assert not widget1.picked
    assert not widget1._resizers_on

    widget0.onpick(pickevent2)
    widget1.onpick(pickevent2)
    assert not widget0.picked
    assert not widget0._resizers_on
    assert not widget1.picked
    assert not widget1._resizers_on

    widget0.onpick(pickevent1)
    widget1.onpick(pickevent1)
    assert not widget0.picked
    assert not widget0._resizers_on
    assert widget1.picked
    assert widget1._resizers_on

    widget0.onpick(pickevent0)
    widget1.onpick(pickevent0)
    assert widget0.picked
    assert widget0._resizers_on
    assert not widget1.picked
    assert not widget1._resizers_on

    widget0.onpick(pickevent2)
    widget1.onpick(pickevent2)
    assert not widget0.picked
    assert not widget0._resizers_on
    assert not widget1.picked
    assert not widget1._resizers_on
