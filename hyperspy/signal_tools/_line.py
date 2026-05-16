# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

from hyperspy.drawing.widgets import Line2DWidget, VerticalLineWidget
from hyperspy.exceptions import SignalDimensionError


class LineInSignal2D(t.HasTraits):
    """
    Adds a draggable line to a Signal2D that reports its
    position to the position attribute of the class.

    Parameters
    ----------
    signal : Signal2D
        The signal to which the line is added.
    color : str, optional
        The color of the line. Default is 'blue'.
    linewidth : float, optional
        The width of the line. Default is 2.
    snap : bool, optional
        If True, the line will snap to the nearest axis value. Default is False.

    Attributes
    ----------
    x0, y0, x1, y1 : float
        Position of the line in scaled units.
    on : bool
        Turns on and off the line

    Properties
    ----------
    length : float
        Length of the line in scaled units.
    """

    x0, y0, x1, y1 = t.Float(0.0), t.Float(0.0), t.Float(1.0), t.Float(1.0)
    on = t.Bool(False)
    length = t.Property(observe="x0,y0,x1,y1")

    def __init__(self, signal, color="blue", linewidth=2, snap=False):
        super().__init__()
        if signal.axes_manager.signal_dimension != 2:
            raise SignalDimensionError(signal.axes_manager.signal_dimension, 2)

        self._line = None
        self.signal = signal
        if (self.signal._plot is None) or (not self.signal._plot.is_active):
            self.signal.plot()

        self._xaxis = self.signal.axes_manager.signal_axes[0]
        self._yaxis = self.signal.axes_manager.signal_axes[1]
        self._color = color
        self._linewidth = linewidth
        self._snap_position = snap
        self.on = True

        # close the tool when the plot is closed
        self.signal._plot.signal_plot.events.closed.connect(self.close, [])

    def _get_length(self):
        # length is a property that observes x0, y0, x1, y1
        # this function is called when x0, y0, x1, y1 are changed
        position = (self.x0, self.y0), (self.x1, self.y1)
        return np.linalg.norm(np.diff(position, axis=0), axis=1)[0]

    def _get_initial_position(self):
        d0 = (self._xaxis.high_value - self._xaxis.low_value) / 4
        d1 = (self._yaxis.high_value - self._yaxis.low_value) / 4
        return (
            (self._xaxis.low_value + d0, self._yaxis.low_value + d1),
            (self._xaxis.high_value - d0, self._yaxis.high_value - d1),
        )

    # "on" traits change handler
    def _on_changed(self, old, new):
        if not self.signal._plot.is_active:
            self.on = False
            return

        if new is True and old is False:
            self._line = Line2DWidget(self.signal.axes_manager, color=self._color)
            self._line.snap_position = self._snap_position
            # The default axis is the navigation axis; specify the signal axis instead.
            self._line.axes = (self._xaxis, self._yaxis)
            self._line.events.changed.connect(self._update_position_from_line, [])
            self._line.position = self._get_initial_position()
            self._line.linewidth = self._linewidth
            self._line.set_mpl_ax(self.signal._plot.signal_plot.ax)

        elif new is False and old is True:
            self._line.close()
            self._line = None

    # "position" traits change handler
    def _x0_changed(self, old, new):
        if old != new and self._line is not None:
            with self._line.events.changed.suppress_callback(
                self._update_position_from_line
            ):
                self._line.position = ((new, self.y0), (self.x1, self.y1))

    def _y0_changed(self, old, new):
        if old != new and self._line is not None:
            with self._line.events.changed.suppress_callback(
                self._update_position_from_line
            ):
                self._line.position = ((self.x0, new), (self.x1, self.y1))

    def _x1_changed(self, old, new):
        if old != new and self._line is not None:
            with self._line.events.changed.suppress_callback(
                self._update_position_from_line
            ):
                self._line.position = ((self.x0, self.y0), (new, self.y1))

    def _y1_changed(self, old, new):
        if old != new and self._line is not None:
            with self._line.events.changed.suppress_callback(
                self._update_position_from_line
            ):
                self._line.position = ((self.x0, self.y0), (self.x1, new))

    def _update_position_from_line(self, *args, **kwargs):
        (self.x0, self.y0), (self.x1, self.y1) = self._line.position

    def close(self):
        if self._line is not None:
            self._line.events.changed.disconnect(self._update_position_from_line)
        self.on = False


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
        super().__init__()
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

        # close the tool when the plot is closed
        self.signal._plot.signal_plot.events.closed.connect(self.close, [])

    def _get_initial_position(self, *args, **kwargs):
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
            # The default axis is the navigation axis; specify the signal axis instead.
            self._line.axes = (self._axis,)
            # connect callback to update position of the tool from the widget
            self._line.events.changed.connect(self._update_position_from_line, [])
            # to enable _onjumpclick
            self._line.is_pointer = True
            self._line.snap_position = self._snap_position
            self._line.position = (self._get_initial_position(),)
            self._line.set_mpl_ax(self.signal._plot.signal_plot.ax)
            self._line.patch[0].set_linewidth(self._linewidth)

        elif new is False and old is True:
            self._line.close()
            self._line = None

    # "position" traits change handler
    def _position_changed(self, old, new):
        if old != new and self._line is not None:
            with self._line.events.changed.suppress_callback(
                self._update_position_from_line
            ):
                self._line.position = (new,)

    def _update_position_from_line(self):
        self.position = self._line.position[0]

    def close(self):
        if self._line is not None:
            self._line.events.changed.disconnect(self._update_position_from_line)
        self.on = False
