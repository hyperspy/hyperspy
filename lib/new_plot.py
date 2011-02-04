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

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import numpy as np
import enthought.traits.api as t
import enthought.traits.ui.api as tui
import enthought.chaco.api as chaco
from enthought.enable.component_editor import ComponentEditor

import messages
import new_coordinates
#import signal
from drawing.utils import on_window_close


def _does_figure_object_exists(fig_obj):
        if fig_obj is None:
            return False
        else:
            # Test if the figure really exists. If not call the reset function 
            # and start again. This is necessary because with some backends 
            # EELSLab fails to connect the close event to the function.
            try:
                fig_obj.show()
                return True
            except:
                fig_obj = None
                return False
                
class Plot1D(t.HasTraits):
#    signal = t.Instance(signal.Signal)
    coordinates = t.Instance(new_coordinates.NewCoordinates)
    
    figure = t.Instance(plt.Figure)
    ax1 = t.Instance(plt.Axes)
    ax2 = t.Instance(plt.Axes)
    autoscale = t.Bool(True)
    
    def __init__(self, signal, coordinates):
        super(Plot1D, self).__init__()
        self.signal = signal
        self.coordinates = coordinates
                
    def _create_figure(self):
        if _does_figure_object_exists(self.figure) is True:
            return            
        else:
            self.figure = plt.figure()
            # TODO: consistent window title
            self.figure.canvas.set_window_title(self.signal.name + ' Spectrum')
            on_window_close(self.figure, self._on_figure_close)
            
    def reset_attributes(self):
        self.autoscale = True
        self.figure = None
        self.ax1 = None
        self.ax2 = None
            
    def _on_figure_close(self, *arg):
        if _does_figure_object_exists(self.figure) is True:
            plt.close(self.figure)
        self.reset_attributes()
        
    def plot(self, filename=None):
        '''Plot or save as an image the current spectrum
        
        Parameters
        ----------
        filename : {None, str}
            if None, it will plot to the screen the current spectrum. If string 
            it will save the current spectrum as a png image
        '''
        if self.coordinates.output_dim != 1:
            messages.warning_exit('The view is not 1D')
        
        coord = self.coordinates._slicing_coordinates[0]
        self._create_figure()
        self.ax1 = self.figure.add_subplot(111)
        self.ax1.step(coord.axis, self.signal(), color = 'blue')
        plt.xlabel('%s (%s)' % (coord.name, coord.units))
        plt.ylabel('%s (%s)' % (self.signal.name, self.signal.units))
        self.on_trait_change(self.update_plot, 'coordinates.coordinates.index')
        if filename is not None:
            plt.savefig(filename)

    def update_plot(self):
        '''Update the plot'''
        if _does_figure_object_exists(self.figure) is False:
            self._on_figure_close()
            return
        
        coord = self.coordinates._slicing_coordinates[0]
        ydata = self.signal()
        self.ax1.lines[0].set_ydata(ydata)
        if self.autoscale is True:
            self.ax1.relim()
            y1, y2 = np.searchsorted(coord.axis, 
            self.ax1.get_xbound())
            y2 += 2
            y1, y2 = np.clip((y1,y2),0,len(ydata-1))
            clipped_ydata = ydata[y1:y2]
            y_max, y_min = clipped_ydata.max(), clipped_ydata.min()
            self.ax1.set_ylim(y_min, y_max)
        self.figure.canvas.draw()
        
class Plot2D(t.HasTraits):
#    signal = t.Instance(signal.Signal)
    coordinates = t.Instance(new_coordinates.NewCoordinates)    
    figure = t.Instance(plt.Figure)
    ax1 = t.Instance(plt.Axes)
    autoscale = t.Bool(True)
    
    def __init__(self, signal, coordinates):
        super(Plot2D, self).__init__()
        self.signal = signal
        self.coordinates = coordinates
                
    def _create_figure(self):
        if _does_figure_object_exists(self.figure) is True:
            return            
        else:
            self.figure = plt.figure()
            # TODO: consistent window title
            self.figure.canvas.set_window_title(self.signal.name)
            on_window_close(self.figure, self._on_figure_close)
            
    def reset_attributes(self):
        self.figure = None
        self.ax1 = None
            
    def _on_figure_close(self, *arg):
        if _does_figure_object_exists(self.figure) is True:
            plt.close(self.figure)
        self.reset_attributes()
        
    def plot(self, filename=None):
        '''Plot or save as an image the current spectrum
        
        Parameters
        ----------
        filename : {None, str}
            if None, it will plot to the screen the current spectrum. If string 
            it will save the current spectrum as a png image
        '''
        if self.coordinates.output_dim != 2:
            messages.warning_exit('The view is not 2D')
        
        self._create_figure()
        self.ax1 = self.figure.add_subplot(111)
        self.ax1.imshow(self.signal(), interpolation = 'nearest')
#        plt.xlabel('%s (%s)' % (coord.name, coord.units))
#        plt.ylabel('%s (%s)' % (self.signal.name, self.signal.units))
        self.on_trait_change(self.update_plot, 'coordinates.coordinates.index')
        if filename is not None:
            plt.savefig(filename)

    def update_plot(self):
        '''Update the current spectrum figure'''
        if _does_figure_object_exists(self.figure) is False:
            print "The figure has been closed"
            self._on_figure_close()
            return
        
#        coord = self.coordinates._slicing_coordinates[0]
        ydata = self.signal()
        self.ax1.images[0].set_data(ydata)
        if self.autoscale is True:
            self.ax1.images[0].autoscale()
        self.figure.canvas.draw()

plt.show()
