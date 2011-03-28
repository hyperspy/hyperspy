# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
# Set the matplotlib cmap to gray (the default is jet)
plt.rcParams['image.cmap'] = 'gray'

import utils
import image

class MPL_HyperImage_Explorer():
    def __init__(self):
        self.image_data_function = None
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


#    def is_active(self):
#        return utils.does_figure_object_exists(self.spectrum_plot.figure)
#    
#    def assign_pointer(self):
#        shape = len(self.axes_manager.axes) - 1
#        if shape >= 1:
#            if shape > 1:
#                Pointer = widgets.DraggableSquare
#            else:
#                Pointer = widgets.DraggableHorizontalLine
#            return Pointer
#        else:
#            return None    
        
        
#    def plot(self):
#        if self.pointer is None:
#            pointer = self.assign_pointer()  
#            if pointer is not None:
#                self.pointer = pointer(self.axes_manager)
#                self.pointer.color = 'red'
#        if self.pointer is not None:
#            self.plot_image()
#        self.plot_spectrum()
        
    def plot_image(self):
        if self.image_plot is not None:
            self.image_plot.plot()
            return
        imf = image.ImagePlot()
        imf.data_function = self.image_data_function
        imf.pixel_units = self.pixel_units
        imf.pixel_size = self.pixel_size
        imf.plot()
        self.image_plot = imf
        
    def plot(self, filename=None):
        self.plot_image()
        
#    def plot_navigator(self):
#        if self.spectrum_plot is not None:
#            self.spectrum_plot.plot()
#            return
#        
#        # Create the figure
#        sf = spectrum.SpectrumFigure()
#        sf.xlabel = self.xlabel
#        sf.ylabel = self.ylabel
#        sf.title = self.spectrum_title
#        sf.axis = self.axis
#        sf.left_axes_manager = self.axes_manager
#        self.spectrum_plot = sf
#        # Create a line to the left axis with the default coordinates
#        sl = spectrum.SpectrumLine()
#        sl.data_function = self.spectrum_data_function
#        if self.pointer is not None:
#            color = self.pointer.color
#        else:
#            color = 'red'
#        sl.line_properties_helper(color, 'step')        
#        # Add the line to the figure
#          
#        sf.add_line(sl)
#        self.spectrum_plot = sf
#        sf.plot()
#        if self.image_plot is not None and sf.figure is not None:
#            utils.on_window_close(sf.figure, self.image_plot.close)
#            self._key_nav_cid = self.spectrum_plot.figure.canvas.mpl_connect(
#            'key_press_event', self.axes_manager.key_navigator)
#            self._key_nav_cid = self.image_plot.figure.canvas.mpl_connect(
#            'key_press_event', self.axes_manager.key_navigator)
#            self.spectrum_plot.figure.canvas.mpl_connect(
#                'key_press_event', self.key2switch_right_pointer)
#            self.image_plot.figure.canvas.mpl_connect(
#                'key_press_event', self.key2switch_right_pointer)
            
    
    def close(self):         
        self.spectrum_plot.close()
        self.image_plot.close()        
           

plt.show()