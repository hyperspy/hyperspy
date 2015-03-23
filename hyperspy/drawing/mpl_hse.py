# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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

from __future__ import division
import copy

import numpy as np
from traits.api import Undefined

from hyperspy.drawing.mpl_he import MPL_HyperExplorer
from hyperspy.drawing import spectrum, utils


class MPL_HyperSpectrum_Explorer(MPL_HyperExplorer):

    """Plots the current spectrum to the screen and a map with a cursor
    to explore the SI.

    """

    def __init__(self):
        super(MPL_HyperSpectrum_Explorer, self).__init__()
        self.xlabel = ''
        self.ylabel = ''
        self.right_pointer = None
        self._right_pointer_on = False
        self._auto_update_plot = True

    @property
    def auto_update_plot(self):
        return self._auto_update_plot

    @auto_update_plot.setter
    def auto_update_plot(self, value):
        if self._auto_update_plot is value:
            return
        for line in self.signal_plot.ax_lines + \
                self.signal_plot.right_ax_lines:
            line.auto_update = value
        if self.pointer is not None:
            if value is True:
                self.pointer.set_mpl_ax(self.navigator_plot.ax)
            else:
                self.pointer.disconnect(self.navigator_plot.ax)

    @property
    def right_pointer_on(self):
        """I'm the 'x' property."""
        return self._right_pointer_on

    @right_pointer_on.setter
    def right_pointer_on(self, value):
        if value == self._right_pointer_on:
            return
        self._right_pointer_on = value
        if value is True:
            self.add_right_pointer()
        else:
            self.remove_right_pointer()

    def plot_signal(self):
        if self.signal_plot is not None:
            self.signal_plot.plot()
            return
        # Create the figure
        self.xlabel = '%s' % str(self.axes_manager.signal_axes[0])
        if self.axes_manager.signal_axes[0].units is not Undefined:
            self.xlabel += ' (%s)' % self.axes_manager.signal_axes[0].units
        self.ylabel = 'Intensity'
        self.axis = self.axes_manager.signal_axes[0].axis
        sf = spectrum.SpectrumFigure(title=self.signal_title +
                                     " Signal")
        sf.xlabel = self.xlabel
        sf.ylabel = self.ylabel
        sf.axis = self.axis
        sf.create_axis()
        sf.axes_manager = self.axes_manager
        self.signal_plot = sf
        # Create a line to the left axis with the default indices
        sl = spectrum.SpectrumLine()
        sl.autoscale = True
        sl.data_function = self.signal_data_function
        sl.plot_indices = True
        if self.pointer is not None:
            color = self.pointer.color
        else:
            color = 'red'
        sl.set_line_properties(color=color, type='step')
        # Add the line to the figure

        sf.add_line(sl)
        # If the data is complex create a line in the left axis with the
        # default coordinates
        sl = spectrum.SpectrumLine()
        sl.data_function = self.signal_data_function
        sl.plot_coordinates = True
        sl.get_complex = any(np.iscomplex(sl.data_function()))
        if sl.get_complex:
            sl.set_line_properties(color="blue", type='step')
            # Add extra line to the figure
            sf.add_line(sl)

        self.signal_plot = sf
        sf.plot()
        if self.navigator_plot is not None and sf.figure is not None:
            utils.on_figure_window_close(self.navigator_plot.figure,
                                         self._disconnect)
            utils.on_figure_window_close(sf.figure,
                                         self.close_navigator_plot)
            self._key_nav_cid = \
                self.signal_plot.figure.canvas.mpl_connect(
                    'key_press_event',
                    self.axes_manager.key_navigator)
            self._key_nav_cid = \
                self.navigator_plot.figure.canvas.mpl_connect(
                    'key_press_event',
                    self.axes_manager.key_navigator)
            self.signal_plot.figure.canvas.mpl_connect(
                'key_press_event', self.key2switch_right_pointer)
            self.navigator_plot.figure.canvas.mpl_connect(
                'key_press_event', self.key2switch_right_pointer)

    def key2switch_right_pointer(self, event):
        if event.key == "e":
            self.right_pointer_on = not self.right_pointer_on

    def add_right_pointer(self):
        if self.signal_plot.right_axes_manager is None:
            self.signal_plot.right_axes_manager = \
                copy.deepcopy(self.axes_manager)
        if self.right_pointer is None:
            pointer = self.assign_pointer()
            self.right_pointer = pointer(
                self.signal_plot.right_axes_manager)
            self.right_pointer.size = self.pointer.size
            self.right_pointer.color = 'blue'
            self.right_pointer.set_mpl_ax(self.navigator_plot.ax)

        if self.right_pointer is not None:
            for axis in self.axes_manager.navigation_axes[
                    self._pointer_nav_dim:]:
                self.signal_plot.right_axes_manager._axes[
                    axis.index_in_array] = axis
        rl = spectrum.SpectrumLine()
        rl.autoscale = True
        rl.data_function = self.signal_data_function
        rl.set_line_properties(color=self.right_pointer.color,
                               type='step')
        self.signal_plot.create_right_axis()
        self.signal_plot.add_line(rl, ax='right')
        rl.plot_indices = True
        rl.text_position = (1., 1.05,)
        rl.plot()
        self.right_pointer_on = True

    def remove_right_pointer(self):
        for line in self.signal_plot.right_ax_lines:
            self.signal_plot.right_ax_lines.remove(line)
            line.close()
        self.right_pointer.close()
        self.right_pointer = None
        self.navigator_plot.update()
