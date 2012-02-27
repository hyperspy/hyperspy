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

import image
import spectrum
import widgets
import utils

class MPL_HyperImage_Explorer():
    def __init__(self):
        self.image_data_function = None
        self.navigator_data_function = None
        self.axes_manager = None
        self.image_title = ''
        self.navigator_title = ''
        self.image_plot = None
        self.navigator_plot = None
        self.pixel_size = None
        self.pixel_units =  None
        self.axis = None
        self.pointer = None
        self._key_nav_cid = None

        
    def plot_image(self):
        if self.image_plot is not None:
            self.image_plot.plot()
            return
        imf = image.ImagePlot()
        imf.data_function = self.image_data_function
        imf.pixel_units = self.pixel_units
        imf.pixel_size = self.pixel_size
        imf.title = self.image_title
        imf.plot()
        self.image_plot = imf
        
        if self.navigator_plot is not None and imf.figure is not None:
            utils.on_figure_window_close(self.navigator_plot.figure, 
            self._close_pointer)
            utils.on_figure_window_close(imf.figure, self.close_navigator_plot)
            self._key_nav_cid = self.image_plot.figure.canvas.mpl_connect(
            'key_press_event', self.axes_manager.key_navigator)
            self._key_nav_cid = self.navigator_plot.figure.canvas.mpl_connect(
            'key_press_event', self.axes_manager.key_navigator)
    
    def close_navigator_plot(self):
        self._close_pointer()
        self.navigator_plot.close()
    
    def is_active(self):
        return utils.does_figure_object_exists(self.image_plot.figure)        
    def plot(self):
        self.pointer = self.assign_pointer()
        if self.pointer is not None:
            self.pointer = self.pointer(self.axes_manager)
            self.pointer.color = 'red'
            self.plot_navigator()
        self.plot_image()
        self.axes_manager.connect(self.image_plot.update_image)
            
    def assign_pointer(self):
        nav_dim = self.axes_manager.navigation_dimension
        if nav_dim == 2:
            Pointer = widgets.DraggableSquare
        elif nav_dim == 1:
            Pointer = widgets.DraggableVerticalLine
        else:
            Pointer = None
        return Pointer
        
    def plot_navigator(self):
        if self.navigator_plot is not None:
            self.navigator_plot.plot()
            return
        if self.axes_manager.navigation_dimension == 2:
            imf = image.ImagePlot()
            imf.data_function = self.navigator_data_function
            imf.pixel_units = self.axes_manager._non_slicing_axes[0].units
            imf.pixel_size = self.axes_manager._non_slicing_axes[0].scale
            imf.plot()
            self.pointer.add_axes(imf.ax)
            self.navigator_plot = imf
            
        if self.axes_manager.navigation_dimension == 1:
            
            # Create the figure
            sf = spectrum.SpectrumFigure()
            axis = self.axes_manager._non_slicing_axes[0]
            sf.xlabel = axis.name
#            sf.ylabel = self.ylabel
            sf.title = '1D navigator'
            sf.axis = axis.axis
            sf.axes_manager = self.axes_manager
            self.navigator_plot = sf
            # Create a line to the left axis with the default coordinates
            sl = spectrum.SpectrumLine()
            sl.data_function = self.navigator_data_function

            sl.line_properties_helper('blue', 'step')        
            # Add the line to the figure
              
            sf.add_line(sl)
            self.navigator_plot = sf
            sf.plot()
            self.pointer.add_axes(sf.ax)
    def _close_pointer(self):
        if self.pointer is not None:
            self.pointer.disconnect(self.navigator_plot.ax)            
    def close(self):         
        self.image_plot.close()
        self.navigator_plot.close()        
           
