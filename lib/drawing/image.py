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
import math
import numpy as np
try:
    import matplotlib.pyplot as plt
    import matplotlib.widgets
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.widgets

import widgets
import utils

class ImagePlot:
    def __init__(self):
        self.data_function = None
        self.pixel_size = None
        self.pixel_units = None
        self.plot_scale_bar = True
        self.figure = None
        self.ax = None
        self.title = ''
        self.window_title = ''
        self.vmin = None
        self.vmax = None
        self.auto_contrast = True
        
    def optimize_contrast(self, data, perc = 0.01):
        dc = data[np.isnan(data) == False]
        try:
            # check if it's an RGB structured array
            dc = dc['R']
        except ValueError, msg:
            if 'field named R not found' in msg:
                pass
            else:
                raise
        if 'complex' in dc.dtype.name:
            dc = np.log(np.abs(dc))
        i = int(round(len(dc)*perc/100.))
        i = i if i > 0 else 1
        vmin = np.min(dc)
        vmax = np.max(dc)
        print "Automatically setting the constrast values"
        self.vmin = vmin
        self.vmax = vmax
        print "Min = ", vmin
        print "Max = ", vmax
        
    def create_figure(self):
        self.figure = utils.create_figure()
        
    def create_axis(self):
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()
        self.figure.subplots_adjust(0,0,1,1)
        
    def plot(self):
        if not utils.does_figure_object_exists(self.figure):
            self.create_figure()
            self.create_axis()     
        data = self.data_function()
        if self.auto_contrast is True:
            self.optimize_contrast(data)
        self.update_image()
        if self.plot_scale_bar is True:
            if self.pixel_size is not None:
                self.ax.scale_bar = widgets.Scale_Bar(
                 ax = self.ax, units = self.pixel_units, 
                 pixel_size = self.pixel_size)
        
        # Adjust the size of the window
        #size = [ 6,  6.* data.shape[0] / data.shape[1]]
        #self.figure.set_size_inches(size, forward = True)        
        self.figure.canvas.draw()
        self.connect()
        
    def update_image(self):
        ims = self.ax.images
        if ims:
            ims.remove(ims[0])
        data = self.data_function()
        if 'complex' in data.dtype.name:
            data = np.log(np.abs(data))
        try:
            # check if it's an RGB structured array
            data_r = data['R']
            data_g = data['G']
            data_b = data['B']
            # modify the data so that it can be read by matplotlib
            data = np.rollaxis(np.array((data_r, data_g, data_b)), 0, 3)
        except ValueError, msg:
            if 'field named R not found' in msg:
                pass
            else:
                raise
        self.ax.imshow(data, interpolation='nearest', vmin = self.vmin, 
                       vmax = self.vmax)
        self.figure.canvas.draw()
        
    def connect(self):
        self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.figure.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'h':
            self.plot_histogram()
        
            
    def histogram_key_press(self, event):
        if event.key == 'a':
            self.optimize_contrast(self.data_function())
            self.set_contrast(self.vmin, self.vmax)
            
    def set_contrast(self, vmin, vmax):
        self.vmin, self.vmax =  vmin, vmax
        del(self.histogram_span_selector)
        plt.close(self.histogram_figure)
        del(self.histogram_figure)
        self.plot_histogram()
        self.update_image()
        
    def plot_histogram(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        data = self.data_function().ravel()
        ax.hist(data,100, range = (self.vmin, self.vmax))
        self.histogram_span_selector = matplotlib.widgets.SpanSelector(
        ax, self.set_contrast, direction = 'horizontal')
        self.histogram_figure = f
        self.histogram_figure.canvas.mpl_connect(
        'key_press_event', self.histogram_key_press)
    
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
        
plt.show()
