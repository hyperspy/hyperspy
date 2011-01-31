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

import utils

class SpectrumFigure():
    ''''''
    def __init__(self):
        self.figure = None
        self.left_ax = None
        self.right_ax = None
        self.left_ax_lines = list()
        self.right_ax_lines = list()
        self.autoscale = True
        self.blit = False
        self.lines = list()
        self.left_coordinates = None
        self.right_coordinates = None
        
        
        # Labels
        self.xlabel = ''
        self.ylabel = ''
        self.title = ''
        self.create_figure()
        self.create_left_axis()
#        self.create_right_axis()

        
    def create_figure(self):
        self.figure = utils.create_figure()
        
    def create_left_axis(self):
        self.left_ax = self.figure.add_subplot(111)
        ax = self.left_ax
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.title)
        
    def create_right_axis(self):
        if self.left_ax is None:
            self.create_left_axis()
        self.right_ax = self.left_ax.twinx()
        
    def add_line(self, line, ax = 'left'):
        if ax == 'left':
            line.ax = self.left_ax
            if line.coordinates is None:
                line.coordinates = self.left_coordinates
            self.left_ax_lines.append(line)
        elif ax == 'right':
            line.ax = self.right_ax
            self.right_ax_lines.append(line)
            if line.coordinates is None:
                line.coordinates = self.right_coordinates
        line.axis = self.axis
        line.autoscale = self.autoscale
        line.blit = self.blit
        
    def plot(self):   
        for line in self.left_ax_lines:
            line.plot()
        
    def close(self):
        for line in self.left_ax_lines + self.right_ax_lines:
            line.close()
        if utils.does_figure_object_exists(self.figure):
            plt.close(self.figure)

        
class SpectrumLine():
    def __init__(self):
        ''''''
        # Data attributes
        self.data_function = None
        self.axis = None
        self.coordinates = None
        
        # Properties
        self.line = None
        self.line_properties = dict()
        self.autoscale = True
    

    def line_properties_helper(self, color, type):
        '''This function provides an easy way of defining some basic line 
        properties.
        
        Further customization is possible by adding keys to the line_properties 
        attribute
        
        Parameters
        ----------
        
        color : any valid matplotlib color definition, e.g. 'red'
        type : it can be one of 'scatter', 'step', 'line'
        '''
        lp = self.line_properties
        if type == 'scatter':
            lp['marker'] = 'o'
            lp['linestyle'] = 'None'
            lp['markersize'] = 1
            lp['markeredgecolor'] = color
        elif type == 'line':
            lp['color'] = color
        elif type == 'step':
            lp['color'] = color
            lp['drawstyle'] = 'steps'
    def set_properties(self):
        for key in self.line_properties:
            plt.setp(self.line, **self.line_properties)
        self.ax.figure.canvas.draw()
        
    def plot(self, data = 1):
        f = self.data_function
        self.line, = self.ax.plot(
        self.axis, f(coordinates = self.coordinates), **self.line_properties)
        self.coordinates.connect(self.update)
        self.ax.figure.canvas.draw()
                  
    def update(self):
        '''Update the current spectrum figure'''            
        
        ydata = self.data_function(coordinates = self.coordinates)
        self.line.set_ydata(ydata)
        
        if self.autoscale is True:
            self.ax.relim()
            y1, y2 = np.searchsorted(self.axis, 
            self.ax.get_xbound())
            y2 += 2
            y1, y2 = np.clip((y1,y2),0,len(ydata-1))
            clipped_ydata = ydata[y1:y2]
            y_max, y_min = np.nanmax(clipped_ydata), np.nanmin(clipped_ydata)
            self.ax.set_ylim(y_min, y_max)
            self.ax.figure.canvas.draw()
        
    def close(self):
        self.coordinates.disconnect(self.update)
        
        
        
plt.show()