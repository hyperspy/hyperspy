# -*- coding: utf-8 -*-
# Copyright 2007-2025 The HyperSpy developers
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
import traits.api as t

from hyperspy.axes import AxesManager
from hyperspy.drawing.widgets import Line2DWidget, VerticalLineWidget
from hyperspy.exceptions import SignalDimensionError


class LineInSignal2D(t.HasTraits):
    """
    Adds a vertical draggable line to a spectrum that reports its
    position to the position attribute of the class.

    Attributes
    ----------
    x0, y0, x1, y1 : floats
        Position of the line in scaled units.
    length : float
        Length of the line in scaled units.
    on : bool
        Turns on and off the line
    """

    x0, y0, x1, y1 = t.Float(0.0), t.Float(0.0), t.Float(1.0), t.Float(1.0)
    length = t.Float(1.0)
    is_ok = t.Bool(False)
    on = t.Bool(False)

    def __init__(self, signal):
        if signal.axes_manager.signal_dimension != 2:
            raise SignalDimensionError(signal.axes_manager.signal_dimension, 2)

        self.signal = signal
        if (self.signal._plot is None) or (not self.signal._plot.is_active):
            self.signal.plot()
        axis_dict0 = signal.axes_manager.signal_axes[0].get_axis_dictionary()
        axis_dict1 = signal.axes_manager.signal_axes[1].get_axis_dictionary()
        am = AxesManager([axis_dict1, axis_dict0])
        am._axes[0].navigate = True
        am._axes[1].navigate = True
        self.axes_manager = am
        self.on_trait_change(self.switch_on_off, "on")

    def draw(self):
        self.signal._plot.signal_plot.figure.canvas.draw_idle()

    def _get_initial_position(self):
        am = self.axes_manager
        d0 = (am[0].high_value - am[0].low_value) / 10
        d1 = (am[1].high_value - am[1].low_value) / 10
        position = (
            (am[0].low_value + d0, am[1].low_value + d1),
            (am[0].high_value - d0, am[1].high_value - d1),
        )
        return position

    def switch_on_off(self, obj, trait_name, old, new):
        if not self.signal._plot.is_active:
            return

        if new is True and old is False:
            self._line = Line2DWidget(self.axes_manager)
            self._line.position = self._get_initial_position()
            self._line.set_mpl_ax(self.signal._plot.signal_plot.ax)
            self._line.linewidth = 1
            self.update_position()
            self._line.events.changed.connect(self.update_position)
            self.draw()

        elif new is False and old is True:
            self._line.close()
            self._line = None
            self.draw()

    def update_position(self, *args, **kwargs):
        if not self.signal._plot.is_active:
            return
        pos = self._line.position
        (self.x0, self.y0), (self.x1, self.y1) = pos
        self.length = np.linalg.norm(np.diff(pos, axis=0), axis=1)[0]


class LineInSignal1D(t.HasTraits):
    """Adds a vertical draggable line to a Signal1D that reports its
    position to the position attribute of the class.

    Parameters
    ----------
    signal : Signal1D
        The signal to which the line is added.
    color : str, optional
        The color of the line. Default is 'blue'.
    linewidth : float, optional
        The width of the line. Default is 2.
    snap : bool, optional
        If True, the line will snap to the nearest axis value. Default is False.

    Attributes
    ----------
    position : float
        The position of the vertical line in the one dimensional signal.
    on : bool
        Turns on and off the line
    """

    position = t.Float(0.0)
    on = t.Bool(False)

    def __init__(self, signal, color="blue", linewidth=2, snap=False):
        self._line = None
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(signal.axes_manager.signal_dimension, 1)

        self.signal = signal
        if self.signal._plot is None or not self.signal._plot.is_active:
            self.signal.plot()

        self._axis = self.signal.axes_manager.signal_axes[0]
        self._color = color
        self._linewidth = linewidth
        self._snap_position = snap
        self.on = True

        # disconnect the line when the plot is closed
        self.signal._plot.signal_plot.events.closed.connect(self.disconnect, [])

    def _get_initial_position(self):
        # Set the position of the line in the middle of the spectral
        # range by default
        return (self._axis.high_value - self._axis.low_value) / 2

    # "on" traits change handler
    def _on_changed(self, old, new):
        if not self.signal._plot.is_active:
            self.on = False
            return

        if new is True and old is False:
            self._line = VerticalLineWidget(self.signal.axes_manager, color=self._color)
            # self._line.snap_position = self._snap_position
            # The default axis is the navigation axis; specify the signal axis instead.
            self._line.axes = (self._axis,)
            self._line.events.changed.connect(self._update_position_from_line, [])
            self._line.position = (self._get_initial_position(),)
            if self._snap_position:
                # when snap is on, update position to the nearest axis value
                self._update_position_from_line
            self._line.set_mpl_ax(self.signal._plot.signal_plot.ax)
            self._line.patch[0].set_linewidth(self._linewidth)

        elif new is False and old is True:
            self._line.close()
            self._line = None

    # "position" traits change handler
    def _position_changed(self, old, new):
        if old != new and self._line is not None:
            self._line.position = (new,)

    def _update_position_from_line(self):
        if self._line is not None:
            self.position = self._line.position[0]

    def disconnect(self):
        if self._line is not None:
            self._line.events.changed.disconnect(self._update_position_from_line)
        self.on = False
