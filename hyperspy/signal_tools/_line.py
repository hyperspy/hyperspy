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
    color : wx.Colour
        The color of the line. It automatically redraws the line.

    """

    x0, y0, x1, y1 = t.Float(0.0), t.Float(0.0), t.Float(1.0), t.Float(1.0)
    length = t.Float(1.0)
    is_ok = t.Bool(False)
    on = t.Bool(False)
    # The following is disabled because as of traits 4.6 the Color trait
    # imports traitsui (!)
    # try:
    #     color = t.Color("black")
    # except ModuleNotFoundError:  # traitsui is not installed
    #     pass
    color_str = t.Str("black")

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
            self._color_changed("black", "black")
            self.update_position()
            self._line.events.changed.connect(self.update_position)
            # There is not need to call draw because setting the
            # color calls it.

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

    def _color_changed(self, old, new):
        if self.on is False:
            return
        self.draw()


class LineInSignal1D(t.HasTraits):
    """Adds a vertical draggable line to a spectrum that reports its
    position to the position attribute of the class.

    Attributes
    ----------
    position : float
        The position of the vertical line in the one dimensional signal. Moving
        the line changes the position but the reverse is not true.
    on : bool
        Turns on and off the line
    color : wx.Colour
        The color of the line. It automatically redraws the line.

    """

    position = t.Float()
    is_ok = t.Bool(False)
    on = t.Bool(False)
    # The following is disabled because as of traits 4.6 the Color trait
    # imports traitsui (!)
    # try:
    #     color = t.Color("black")
    # except ModuleNotFoundError:  # traitsui is not installed
    #     pass
    color_str = t.Str("black")

    def __init__(self, signal):
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(signal.axes_manager.signal_dimension, 1)

        self.signal = signal
        self.signal.plot()
        axis_dict = signal.axes_manager.signal_axes[0].get_axis_dictionary()
        am = AxesManager(
            [
                axis_dict,
            ]
        )
        am._axes[0].navigate = True
        # Set the position of the line in the middle of the spectral
        # range by default
        am._axes[0].index = int(round(am._axes[0].size / 2))
        self.axes_manager = am
        self.axes_manager.events.indices_changed.connect(self.update_position, [])
        self.on_trait_change(self.switch_on_off, "on")

    def draw(self):
        self.signal._plot.signal_plot.figure.canvas.draw_idle()

    def switch_on_off(self, obj, trait_name, old, new):
        if not self.signal._plot.is_active:
            return

        if new is True and old is False:
            self._line = VerticalLineWidget(self.axes_manager)
            self._line.set_mpl_ax(self.signal._plot.signal_plot.ax)
            self._line.patch.set_linewidth(2)
            self._color_changed("black", "black")
            # There is not need to call draw because setting the
            # color calls it.

        elif new is False and old is True:
            self._line.close()
            self._line = None
            self.draw()

    def update_position(self, *args, **kwargs):
        if not self.signal._plot.is_active:
            return
        self.position = self.axes_manager.coordinates[0]

    def _color_changed(self, old, new):
        if self.on is False:
            return

        self._line.patch.set_color(
            (
                self.color.Red() / 255.0,
                self.color.Green() / 255.0,
                self.color.Blue() / 255.0,
            )
        )
        self.draw()
