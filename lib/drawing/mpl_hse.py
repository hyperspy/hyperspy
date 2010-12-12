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

import widgets
import spectrum, image, utils



class MPL_HyperSpectrum_Explorer():
    '''Plots the current spectrum to the screen and a map with a cursor to 
    explore the SI.'''
    
    def __init__(self):
        self.spectrum_data_function = None
        self.image_data_function = None
        self.coordinates = None
        self.spectrum_title = ''
        self.xlabel = ''
        self.ylabel = ''
        self.image_title = ''
        self.image_plot = None
        self.spectrum_plot = None
        self.pixel_size = None
        self.pixel_units =  None
        self.axis = None
        self.pointer = None
    
    def assign_pointer(self):
        shape = self.coordinates.shape
        if shape[0] > 1:
            if shape[1] > 1:
                Pointer = widgets.DraggableSquare
            else:
                Pointer = widgets.DraggableHorizontalLine
        else:
            return
        
        self.pointer = Pointer(self.coordinates)
        self.pointer.color = 'red'
    def plot(self):
        if self.pointer is None:
            self.assign_pointer()
        if self.pointer is not None:
            self.plot_image()
        self.plot_spectrum()
        
    def plot_image(self):
        if self.image_plot is not None:
            self.image_plot.plot()
            return
        imf = image.ImagePlot()
        imf.data_function = self.image_data_function
        imf.pixel_units = self.pixel_units
        imf.pixel_size = self.pixel_size
        imf.plot()
        self.pointer.add_axes(imf.ax)
        self.image_plot = imf
        
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
        sf.left_coordinates = self.coordinates
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
#        if self.image_plot is not None and sf.figure is not None:
#            utils.on_window_close(sf.figure, self.image_plot.close)
        
    def close(self):
        self.spectrum_plot.close()
        self.image_plot.close()

          

