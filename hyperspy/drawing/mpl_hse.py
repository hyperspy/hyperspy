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

from __future__ import division
import copy

import numpy as np

from hyperspy.drawing import widgets, spectrum, image, utils
from hyperspy.gui.axes import navigation_sliders


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
        if self.navigator_data_function is None:              
            nav_dim = self.axes_manager.navigation_dimension
        else:
            nav_dim = len(self.navigator_data_function().shape)
        if nav_dim >= 2:
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
        self.plot_signal()
        
    def plot_navigator(self):
        if self.navigator_data_function is None:            
            navigation_sliders(
                self.axes_manager.navigation_axes[::-1])
            return
        if self.navigator_plot is not None:
            self.navigator_plot.plot()
            return
        elif len(self.navigator_data_function().shape) == 1:
            # Create the figure
            sf = spectrum.SpectrumFigure()
            axis = self.axes_manager.navigation_axes[0]
            sf.xlabel = '%s (%s)' % (axis.name, axis.units)
            sf.ylabel = r'$\Sigma\mathrm{Image\,intensity}$'
            sf.title = self.signal_title + ' Navigator'
            sf.axis = axis.axis
            sf.axes_manager = self.axes_manager
            self.navigator_plot = sf
            # Create a line to the left axis with the default 
            # indices
            sl = spectrum.SpectrumLine()
            sl.data_function = self.navigator_data_function
            sl.line_properties_helper('blue', 'step')        
            # Add the line to the figure
            sf.add_line(sl)
            sf.plot()
            self.pointer.add_axes(sf.ax)
            if self.axes_manager.navigation_dimension > 1:
                navigation_sliders(
                    self.axes_manager.navigation_axes[::-1])
                for axis in self.axes_manager.navigation_axes[:-2]:
                    axis.connect(sf.update_image)
            self.navigator_plot = sf
        elif len(self.navigator_data_function().shape) >= 2:
            imf = image.ImagePlot()
            imf.data_function = self.navigator_data_function
            # Navigator labels
            if self.axes_manager.navigation_dimension == 1:
                imf.yaxis = self.axes_manager.navigation_axes[0]
                imf.xaxis = self.axes_manager.signal_axes[0]
            elif self.axes_manager.navigation_dimension >= 2:
                imf.yaxis = self.axes_manager.navigation_axes[1]
                imf.xaxis = self.axes_manager.navigation_axes[0]
                if self.axes_manager.navigation_dimension > 2:
                    navigation_sliders(
                        self.axes_manager.navigation_axes)
                    for axis in self.axes_manager.navigation_axes[2:]:
                        axis.connect(imf.update_image)
                
            imf.title = self.signal_title + ' Navigator'
            imf.plot()
            self.pointer.add_axes(imf.ax)
            self.navigator_plot = imf        

        
    def plot_signal(self):
        if self.signal_plot is not None:
            self.signal_plot.plot()
            return
        # Create the figure
        self.xlabel = '%s (%s)' % (
            self.axes_manager.signal_axes[0].name,
            self.axes_manager.signal_axes[0].units)
        self.ylabel = 'Intensity'
        self.axis = self.axes_manager.signal_axes[0].axis
        sf = spectrum.SpectrumFigure()
        sf.xlabel = self.xlabel
        sf.ylabel = self.ylabel
        sf.title = self.signal_title
        sf.axis = self.axis
        sf.create_axis()
        sf.axes_manager = self.axes_manager
        self.signal_plot = sf
        # Create a line to the left axis with the default indices
        sl = spectrum.SpectrumLine()
        sl.data_function = self.signal_data_function
        sl.plot_indices = True
        if self.pointer is not None:
            color = self.pointer.color
        else:
            color = 'red'
        sl.line_properties_helper(color, 'step')        
        # Add the line to the figure
        sf.add_line(sl)
        # If the data is complex create a line in the left axis with the
        # default coordinates
        sl = spectrum.SpectrumLine()
        sl.data_function = self.signal_data_function
        sl.plot_coordinates = True
        sl.get_complex = any(np.iscomplex(sl.data_function()))        
        if sl.get_complex:
            sl.line_properties_helper("blue", 'step')        
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
    
    def close_navigator_plot(self):
        self._disconnect()
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
        self.navigator_plot.update_image()
        
    def _disconnect(self):
        if (self.axes_manager.navigation_dimension > 2 and 
            self.navigator_plot is not None):
                for axis in self.axes_manager.navigation_axes[:-2]:
                    axis.disconnect(self.navigator_plot.update_image)
        
        if self.pointer is not None:
            self.pointer.disconnect(self.navigator_plot.ax)
    
    def close(self):
        self._disconnect()
        self.signal_plot.close()
        self.navigator_plot.close()
        
