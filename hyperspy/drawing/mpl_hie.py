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
        self.signal_data_function = None
        self.navigator_data_function = None
        self.axes_manager = None
        self.signal_title = ''
        self.navigator_title = ''
        self.signal_plot = None
        self.navigator_plot = None
        self.signal_pixel_size = None
        self.signal_pixel_units =  None
        self.navigator_pixel_size = None
        self.navigator_pixel_units =  None
        self.axis = None
        self.pointer = None
        self._key_nav_cid = None

        
    def plot_signal(self):
        if self.signal_plot is not None:
            self.signal_plot.plot()
            return
        imf = image.ImagePlot()
        imf.axes_manager = self.axes_manager
        imf.data_function = self.signal_data_function
        imf.pixel_units = self.signal_pixel_units
        imf.pixel_size = self.signal_pixel_size
        imf.title = self.signal_title
        imf.xlabel = self.signal_xlabel
        imf.ylabel = self.signal_ylabel
        imf.plot_scalebar = self.plot_signal_scalebar
        imf.plot_ticks = self.plot_signal_ticks
        imf.plot_colorbar = True
        imf.plot()
        self.signal_plot = imf
        
        if self.navigator_plot is not None and imf.figure is not None:
            utils.on_figure_window_close(self.navigator_plot.figure, 
            self._close_pointer)
            utils.on_figure_window_close(
                imf.figure, self.close_navigator_plot)
            self._key_nav_cid = \
                self.signal_plot.figure.canvas.mpl_connect(
                    'key_press_event', self.axes_manager.key_navigator)
            self._key_nav_cid = \
                self.navigator_plot.figure.canvas.mpl_connect(
                    'key_press_event', self.axes_manager.key_navigator)
                    
    def plot_navigator(self):
        if self.navigator_plot is not None:
            self.navigator_plot.plot()
            return
        if self.axes_manager.navigation_dimension == 2:
            imf = image.ImagePlot()
            imf.data_function = self.navigator_data_function
            imf.pixel_units = self.navigator_pixel_units
            imf.pixel_size = self.navigator_pixel_size
            imf.title = self.signal_title + ' Navigator'
            imf.xlabel = self.navigator_xlabel
            imf.ylabel = self.navigator_ylabel
            imf.plot_scalebar = self.plot_navigator_scalebar
            imf.plot_ticks = self.plot_navigator_ticks
            imf.plot()
            self.pointer.add_axes(imf.ax)
            self.navigator_plot = imf
            
        if self.axes_manager.navigation_dimension == 1:
            
            # Create the figure
            sf = spectrum.SpectrumFigure()
            axis = self.axes_manager.navigation_axes[0]
            sf.xlabel = self.navigator_xlabel
            sf.ylabel = '$\Sigma\mathrm{Image\,intensity\,(%s)}$' % (
                self.axes_manager.signal_axes[0].units)
            sf.title = self.signal_title + ' Navigator'
            sf.axis = axis.axis
            sf.axes_manager = self.axes_manager
            self.navigator_plot = sf
            # Create a line to the left axis with the default 
            # coordinates
            sl = spectrum.SpectrumLine()
            sl.data_function = self.navigator_data_function
            sl.line_properties_helper('blue', 'step')        
            # Add the line to the figure
            sf.add_line(sl)
            self.navigator_plot = sf
            sf.plot()
            self.pointer.add_axes(sf.ax)
    
    def close_navigator_plot(self):
        self._close_pointer()
        self.navigator_plot.close()
    
    def is_active(self):
        return utils.does_figure_object_exists(self.signal_plot.figure)        
    
    def plot(self):
        self.generate_labels()
        self.pointer = self.assign_pointer()
        if self.pointer is not None:
            self.pointer = self.pointer(self.axes_manager)
            self.pointer.color = 'red'
            self.plot_navigator()
        self.plot_signal()
        self.axes_manager.connect(self.signal_plot._update_image)
            
    def assign_pointer(self):
        nav_dim = self.axes_manager.navigation_dimension
        if nav_dim == 2:
            Pointer = widgets.DraggableSquare
        elif nav_dim == 1:
            Pointer = widgets.DraggableVerticalLine
        else:
            Pointer = None
        return Pointer
        
    def generate_labels(self):
        # Image labels
        self.signal_xlabel = '%s (%s)' % (
                self.axes_manager.signal_axes[1].name,
                self.axes_manager.signal_axes[1].units)
        self.signal_ylabel = '%s (%s)' % (
            self.axes_manager.signal_axes[0].name,
            self.axes_manager.signal_axes[0].units)
            
        if (self.axes_manager.signal_axes[0].units == 
            self.axes_manager.signal_axes[1].units):
                self.plot_signal_scalebar = True
                self.plot_signal_ticks = False
        else:
            self.plot_signal_scalebar = False
            self.plot_signal_ticks = True
        signal_scalebar_axis = \
                    self.axes_manager.signal_axes[-1]
        self.signal_pixel_units = signal_scalebar_axis.units
        self.signal_pixel_size = signal_scalebar_axis.scale
                
        # Navigator labels
        if self.axes_manager.navigation_dimension == 1:
            navigator_scalebar_axis = self.axes_manager.signal_axes[0]
            self.navigator_ylabel = (
                '$\Sigma \mathrm{Image\,instensity}$')
                
            self.navigator_xlabel = '%s (%s)' % (
                self.axes_manager.navigation_axes[0].name,
                self.axes_manager.navigation_axes[0].units)
            self.plot_navigator_scalebar = False
            self.plot_navigator_ticks = True

        elif self.axes_manager.navigation_dimension == 2:
            navigator_scalebar_axis = \
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
            self.navigator_pixel_units = navigator_scalebar_axis.units
            self.navigator_pixel_size = navigator_scalebar_axis.scale

        
    def _close_pointer(self):
        if self.pointer is not None:
            self.pointer.disconnect(self.navigator_plot.ax)            
    def close(self):         
        self.signal_plot.close()
        self.navigator_plot.close()        
           
