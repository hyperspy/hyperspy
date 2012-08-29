# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

import widgets
import spectrum, image, utils
import copy

class MPL_HyperSpectrum_Explorer(object):
    """Plots the current spectrum to the screen and a map with a cursor 
    to explore the SI.
    
    """
    
    def __init__(self):
        self.signal_data_function = None
        self.navigator_data_function = None
        self.axes_manager = None
        self.signal_title = ''
        self.xlabel = ''
        self.ylabel = ''
        self.navigator_title = ''
        self.navigator_plot = None
        self.signal_plot = None
        self.pixel_size = None
        self.pixel_units =  None
        self.axis = None
        self.pointer = None
        self.right_pointer = None
        self._key_nav_cid = None
        self._right_pointer_on = False
        self._auto_update_plot = True
        self.plot_navigator_scalebar = False
        self.plot_navigator_plot_ticks = True
        
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
                self.pointer.add_axes(self.navigator_plot.ax)
            else:
                self.pointer.disconnect(self.navigator_plot.ax)
        
    @property
    def right_pointer_on(self):
        """I'm the 'x' property."""
        return self._right_pointer_on

    @right_pointer_on.setter
    def right_pointer_on(self, value):
        if value == self._right_pointer_on: return
        self._right_pointer_on = value
        if value is True:
            self.add_right_pointer()
        else:
            self.remove_right_pointer()
    
    def is_active(self):
        return utils.does_figure_object_exists(self.signal_plot.figure)
    
    def assign_pointer(self):
        nav_dim = self.axes_manager.navigation_dimension
        if nav_dim == 2:
            Pointer = widgets.DraggableSquare
        elif nav_dim == 1:
            Pointer = widgets.DraggableHorizontalLine
        else:
            Pointer = None
        return Pointer
   
    def plot(self):
        self.generate_labels()
        if self.pointer is None:
            pointer = self.assign_pointer()  
            if pointer is not None:
                self.pointer = pointer(self.axes_manager)
                self.pointer.color = 'red'
        if self.pointer is not None:
            self.plot_navigator()
        self.plot_signal()
        
    def generate_labels(self):
        # Spectrum plot labels
        self.xlabel = '%s (%s)' % (
                self.axes_manager.signal_axes[0].name,
                self.axes_manager.signal_axes[0].units)
        self.ylabel = 'Intensity'
        self.axis = self.axes_manager.signal_axes[0].axis
        # Navigator labels
        if self.axes_manager.navigation_dimension == 1:
            scalebar_axis = self.axes_manager.signal_axes[0]
            self.navigator_xlabel = '%s (%s)' % (
                self.axes_manager.signal_axes[0].name,
                self.axes_manager.signal_axes[0].units)
            self.navigator_ylabel = '%s (%s)' % (
                self.axes_manager.navigation_axes[0].name,
                self.axes_manager.navigation_axes[0].units)
            self.plot_navigator_scalebar = False
            self.plot_navigator_ticks = True
            self.pixel_units = scalebar_axis.units
            self.pixel_size = scalebar_axis.scale


        elif self.axes_manager.navigation_dimension == 2:
            scalebar_axis = \
                self.axes_manager.navigation_axes[-1]
            self.navigator_ylabel = '%s (%s)' % (
                self.axes_manager.navigation_axes[0].name,
                self.axes_manager.navigation_axes[0].units)
            self.navigator_xlabel = '%s (%s)' % (
                self.axes_manager.navigation_axes[1].name,
                self.axes_manager.navigation_axes[1].units)
            if (self.axes_manager.navigation_axes[0].units == 
                self.axes_manager.navigation_axes[1].units):
                    self.plot_navigator_scalebar = True
                    self.plot_navigator_ticks = False
            else:
                self.plot_navigator_scalebar = False
                self.plot_navigator_ticks = True
            self.pixel_units = scalebar_axis.units
            self.pixel_size = scalebar_axis.scale
        
        
    def plot_navigator(self):
        if self.navigator_plot is not None:
            self.navigator_plot.plot()
            return
        imf = image.ImagePlot()
        imf.data_function = self.navigator_data_function
        imf.pixel_units = self.pixel_units
        imf.pixel_size = self.pixel_size
        imf.xlabel = self.navigator_xlabel
        imf.ylabel = self.navigator_ylabel
        imf.plot_scalebar = self.plot_navigator_scalebar
        imf.plot_ticks = self.plot_navigator_ticks
        imf.plot()
        self.pointer.add_axes(imf.ax)
        self.navigator_plot = imf
        
    def plot_signal(self):
        if self.signal_plot is not None:
            self.signal_plot.plot()
            return
        
        # Create the figure
        sf = spectrum.SpectrumFigure()
        sf.xlabel = self.xlabel
        sf.ylabel = self.ylabel
        sf.title = self.signal_title
        sf.axis = self.axis
        sf.create_axis()
        sf.axes_manager = self.axes_manager
        self.signal_plot = sf
        # Create a line to the left axis with the default coordinates
        sl = spectrum.SpectrumLine()
        sl.data_function = self.signal_data_function
        sl.plot_coordinates = True
        if self.pointer is not None:
            color = self.pointer.color
        else:
            color = 'red'
        sl.line_properties_helper(color, 'step')        
        # Add the line to the figure
          
        sf.add_line(sl)
        self.signal_plot = sf
        sf.plot()
        if self.navigator_plot is not None and sf.figure is not None:
            utils.on_figure_window_close(self.navigator_plot.figure, 
            self._close_pointer)
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
    
    def close_navigator_plot(self):
        self._close_pointer()
        self.navigator_plot.close()
                
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
            self.right_pointer.add_axes(self.navigator_plot.ax)
        rl = spectrum.SpectrumLine()
        rl.data_function = self.signal_data_function
        rl.line_properties_helper(self.right_pointer.color, 'step')
        self.signal_plot.create_right_axis()
        self.signal_plot.add_line(rl, ax = 'right')
        rl.plot_coordinates = True
        rl.text_position = (1., 1.05,)
        rl.plot()
        self.right_pointer_on = True
        
    def remove_right_pointer(self):
        for line in self.signal_plot.right_ax_lines:
            self.signal_plot.right_ax_lines.remove(line)
            line.close()
        self.right_pointer.close()
        self.right_pointer = None
        self.navigator_plot.update_image()
        
    def _close_pointer(self):
        if self.pointer is not None:
            self.pointer.disconnect(self.navigator_plot.ax)
    
    def close(self):
        self._close_pointer()
        self.signal_plot.close()
        self.navigator_plot.close()
