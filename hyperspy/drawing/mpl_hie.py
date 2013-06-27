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

from hyperspy.drawing import widgets, spectrum, image, utils
from hyperspy.gui.axes import navigation_sliders

class MPL_HyperImage_Explorer():
    """
    
    """
    def __init__(self):
        self.signal_data_function = None
        self.navigator_data_function = None
        self.axes_manager = None
        self.signal_title = ''
        self.navigator_title = ''
        self.signal_plot = None
        self.navigator_plot = None
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
        imf.title = self.signal_title
        imf.xaxis, imf.yaxis = self.axes_manager.signal_axes
        imf.plot_colorbar = True
        imf.plot()
        self.signal_plot = imf
        
        if self.navigator_plot is not None and imf.figure is not None:
            utils.on_figure_window_close(self.navigator_plot.figure, 
            self._disconnect)
            utils.on_figure_window_close(
                imf.figure, self.close_navigator_plot)
            self._key_nav_cid = \
                self.signal_plot.figure.canvas.mpl_connect(
                    'key_press_event', self.axes_manager.key_navigator)
            self._key_nav_cid = \
                self.navigator_plot.figure.canvas.mpl_connect(
                    'key_press_event', self.axes_manager.key_navigator)
                    
    def plot_navigator(self):
        if self.navigator_data_function is None:            
            navigation_sliders(
                self.axes_manager.navigation_axes[::-1])
            return
        elif self.navigator_plot is not None:
            self.navigator_plot.plot()
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
            sl.set_line_properties(color='blue',
                                   type='step')        
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
            imf.title = self.signal_title + ' Navigator'
            imf.xaxis, imf.yaxis = self.axes_manager.navigation_axes[:2]
            imf.plot()
            self.pointer.add_axes(imf.ax)
            if self.axes_manager.navigation_dimension > 2:
                navigation_sliders(
                    self.axes_manager.navigation_axes[::-1])
                for axis in self.axes_manager.navigation_axes[:-1]:
                    axis.connect(imf.update_image)
            self.navigator_plot = imf
    
    def close_navigator_plot(self):
        self._disconnect()
        self.navigator_plot.close()
    
    def is_active(self):
        return utils.does_figure_object_exists(self.signal_plot.figure)
    
    def plot(self):
        self.pointer = self.assign_pointer()
        if self.pointer is not None:
            self.pointer = self.pointer(self.axes_manager)
            self.pointer.color = 'red'
            self.plot_navigator()
        self.plot_signal()
        self.axes_manager.connect(self.signal_plot._update_image)
            
    def assign_pointer(self):
        if self.navigator_data_function is None:              
            nav_dim = self.axes_manager.navigation_dimension
        else:
            nav_dim = len(self.navigator_data_function().shape)
        if nav_dim >= 2:
            Pointer = widgets.DraggableSquare
        elif nav_dim == 1:
            Pointer = widgets.DraggableVerticalLine
        else:
            Pointer = None
        return Pointer
        
    def _disconnect(self):
        if (self.axes_manager.navigation_dimension > 2 and 
            self.navigator_plot is not None):
                for axis in self.axes_manager.navigation_axes[:-2]:
                    axis.disconnect(self.navigator_plot.update_image)
        if self.pointer is not None:
            self.pointer.disconnect(self.navigator_plot.ax)            
    def close(self):         
        self.signal_plot.close()
        self.navigator_plot.close()        
           
