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
    """Plots the current spectrum to the screen and a map with a cursor to 
    explore the SI."""
    
    def __init__(self):
        self.spectrum_data_function = None
        self.image_data_function = None
        self.axes_manager = None
        self.spectrum_title = ''
        self.xlabel = ''
        self.ylabel = ''
        self.image_title = ''
        self.navigator_plot = None
        self.spectrum_plot = None
        self.pixel_size = None
        self.pixel_units =  None
        self.axis = None
        self.pointer = None
        self.right_pointer = None
        self._key_nav_cid = None
        self._right_pointer_on = False
        self._auto_update_plot = True
        
    @property
    def auto_update_plot(self):
        return self._auto_update_plot
        
    @auto_update_plot.setter
    def auto_update_plot(self, value):
        if self._auto_update_plot is value:
            return
        for line in self.spectrum_plot.ax_lines + \
        self.spectrum_plot.right_ax_lines:
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
        return utils.does_figure_object_exists(self.spectrum_plot.figure)
    
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
        if self.pointer is None:
            pointer = self.assign_pointer()  
            if pointer is not None:
                self.pointer = pointer(self.axes_manager)
                self.pointer.color = 'red'
        if self.pointer is not None:
            self.plot_navigator()
        self.plot_spectrum()
        
    def plot_navigator(self):
        if self.navigator_plot is not None:
            self.navigator_plot.plot()
            return
        imf = image.ImagePlot()
        imf.data_function = self.image_data_function
        imf.pixel_units = self.pixel_units
        imf.pixel_size = self.pixel_size
        imf.plot()
        self.pointer.add_axes(imf.ax)
        self.navigator_plot = imf
        
    def plot_spectrum(self):
        if self.spectrum_plot is not None:
            self.spectrum_plot.plot()
            return
        
        # Create the figure
        sf = spectrum.SpectrumFigure()
        sf.xlabel = self.xlabel
        sf.ylabel = self.ylabel
        sf.title = self.spectrum_title
        sf.axis = self.axis
        sf.create_axis()
        sf.axes_manager = self.axes_manager
        self.spectrum_plot = sf
        # Create a line to the left axis with the default coordinates
        sl = spectrum.SpectrumLine()
        sl.data_function = self.spectrum_data_function
        if self.pointer is not None:
            color = self.pointer.color
        else:
            color = 'red'
        sl.line_properties_helper(color, 'step')        
        # Add the line to the figure
          
        sf.add_line(sl)
        self.spectrum_plot = sf
        sf.plot()
        if self.navigator_plot is not None and sf.figure is not None:
            utils.on_figure_window_close(self.navigator_plot.figure, 
            self._close_pointer)
            utils.on_figure_window_close(sf.figure, self.close_navigator_plot)
            self._key_nav_cid = self.spectrum_plot.figure.canvas.mpl_connect(
            'key_press_event', self.axes_manager.key_navigator)
            self._key_nav_cid = self.navigator_plot.figure.canvas.mpl_connect(
            'key_press_event', self.axes_manager.key_navigator)
            self.spectrum_plot.figure.canvas.mpl_connect(
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
        if self.spectrum_plot.right_axes_manager is None:
            self.spectrum_plot.right_axes_manager = \
            copy.deepcopy(self.axes_manager)
        if self.right_pointer is None:
            pointer = self.assign_pointer()
            self.right_pointer = pointer(self.spectrum_plot.right_axes_manager)
            self.right_pointer.color = 'blue'
            self.right_pointer.add_axes(self.navigator_plot.ax)
        rl = spectrum.SpectrumLine()
        rl.data_function = self.spectrum_data_function
        rl.line_properties_helper(self.right_pointer.color, 'step')
        self.spectrum_plot.create_right_axis()
        self.spectrum_plot.add_line(rl, ax = 'right')
        rl.plot()
        self.right_pointer_on = True
        
    def remove_right_pointer(self):
        for line in self.spectrum_plot.right_ax_lines:
            self.spectrum_plot.right_ax_lines.remove(line)
            line.close()
        self.right_pointer.close()
        self.right_pointer = None
        self.navigator_plot.update_image()
        
    def _close_pointer(self):
        if self.pointer is not None:
            self.pointer.disconnect(self.navigator_plot.ax)
    
    def close(self):
        self._close_pointer()
        self.spectrum_plot.close()
        self.navigator_plot.close()
