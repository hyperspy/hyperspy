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

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hyperspy.drawing import widgets
from hyperspy.drawing import utils
from hyperspy.gui.tools import ImageContrastEditor



class ImagePlot:
    def __init__(self):
        self.data_function = None
        self.pixel_size = None
        self.pixel_units = None
        self.plot_scalebar = True
        self.plot_ticks = False
        self.plot_colorbar = False
        self._colorbar = None
        self.figure = None
        self.ax = None
        self.title = ''
        self.window_title = ''
        self.vmin = None
        self.vmax = None
        self.auto_contrast = True
        self.ylabel = ''
        self.xlabel = ''
        self.plot_coordinates = True
        self.text = None
        self.text_position = (-0.1, 1.05,)
        self.axes_manager = None
        
    def optimize_contrast(self, data, perc = 0.01):
        dc = data.copy().ravel()
        if 'complex' in dc.dtype.name:
            dc = np.log(np.abs(dc))
        dc.sort()
        i = int(round(len(dc)*perc))
        i = i if i > 0 else 1
        vmin = np.nanmin(dc[i:])
        vmax = np.nanmax(dc[:-i])
        self.vmin = vmin
        self.vmax = vmax
        
    def create_figure(self):
        self.figure = utils.create_figure()
        
    def create_axis(self):
        self.ax = self.figure.add_subplot(111)
        #self.ax.set_axis_off()
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if self.plot_ticks is False:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        #self.figure.subplots_adjust(0,0,1,1)
        
    def plot(self):
        if not utils.does_figure_object_exists(self.figure):
            self.create_figure()
            self.create_axis()   
        data = self.data_function()
        if self.auto_contrast is True:
            self.optimize_contrast(data)
        if not self.axes_manager or not self.axes_manager.coordinates:
            self.plot_coordinates = False
        if self.plot_coordinates is True:
            self.text = self.ax.text(*self.text_position,
                            s=str(self.axes_manager.coordinates[::-1]),
                            transform = self.ax.transAxes,
                            fontsize=12,
                            color='red')
        self.update_image()
        if self.plot_scalebar is True:
            if self.pixel_size is not None:
                self.ax.scalebar = widgets.Scale_Bar(
                 ax = self.ax, units = self.pixel_units, 
                 pixel_size = self.pixel_size)
                 
        if self.plot_colorbar is True:
            fig = self.ax.figure
            self._colorbar = plt.colorbar(self.ax.images[0],ax=self.ax)

        
        # Adjust the size of the window
        #size = [ 6,  6.* data.shape[0] / data.shape[1]]
        #self.figure.set_size_inches(size, forward = True)        
        self.figure.canvas.draw()
        #if hasattr(self.figure, 'tight_layout'):
            #self.figure.tight_layout()
        self.connect()
        
    def update_image(self, auto_contrast=None):
        ims = self.ax.images
        if ims:
            ims.remove(ims[0])
        data = self.data_function()
        numrows, numcols = data.shape
        def format_coord(x, y):
            col = int(x+0.5)
            row = int(y+0.5)
            if col>=0 and col<numcols and row>=0 and row<numrows:
                z = data[row,col]
                return 'x=%1.4f, y=%1.4f, intensity=%1.4f'%(x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f'%(x, y)
        self.ax.format_coord = format_coord
        if auto_contrast is True or auto_contrast is None and\
            self.auto_contrast is True:
            self.optimize_contrast(data)
        if 'complex' in data.dtype.name:
            data = np.log(np.abs(data))
            
        self.ax.imshow(data, interpolation='nearest', vmin=self.vmin, 
                       vmax=self.vmax)
        if self.plot_coordinates is True:
            self.text.set_text((self.axes_manager.coordinates[::-1]))
        self.figure.canvas.draw()
        
    def _update_image(self):
        # This "wrapper" because on_trait_change fiddles with the 
        # method arguments and auto_contrast does not work then
        self.update_image()
    def adjust_contrast(self):
        ceditor = ImageContrastEditor(self)
        ceditor.edit_traits()
        return ceditor
        
    def connect(self):
        self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.figure.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'h':
            self.adjust_contrast()
                    
    def set_contrast(self, vmin, vmax):
        self.vmin, self.vmax =  vmin, vmax
        self.update_image()
            
    # TODO The next function must be improved
    
    def optimize_colorbar(self, number_of_ticks = 5, tolerance = 5, step_prec_max = 1):
        vmin, vmax = self.vmin, self.vmax
        _range = vmax - vmin
        step = _range / (number_of_ticks - 1)
        step_oom = utils.order_of_magnitude(step)
        def optimize_for_oom(oom):
            self.colorbar_step = math.floor(step / 10**oom)*10**oom
            self.colorbar_vmin = math.floor(vmin / 10**oom)*10**oom
            self.colorbar_vmax = self.colorbar_vmin + \
            self.colorbar_step * (number_of_ticks - 1)
            self.colorbar_locs = np.arange(0,number_of_ticks)* self.colorbar_step \
            + self.colorbar_vmin
        def check_tolerance():
            if abs(self.colorbar_vmax - vmax) / vmax > (tolerance / 100.) or \
            abs(self.colorbar_vmin - vmin) >  (tolerance / 100.):
                return True
            else:
                return False
                    
        optimize_for_oom(step_oom)
        i = 1
        while check_tolerance() and i <= step_prec_max:
            optimize_for_oom(step_oom - i)
            i += 1
            
    def close(self):
        if utils.does_figure_object_exists(self.figure) is True:
            plt.close(self.figure)


