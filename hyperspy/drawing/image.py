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

import math

import numpy as np
import matplotlib.pyplot as plt
from traits.api import Undefined

from hyperspy.drawing import widgets
from hyperspy.drawing import utils
from hyperspy.gui.tools import ImageContrastEditor
from hyperspy.misc import math_tools

class ImagePlot:
    """Class to plot an image with the necessary machinery to update
    the image when the coordinates of an AxesManager change.
    
    Attributes
    ----------
    data_fuction : function or method
        A function that returns a 2D array when called without any 
        arguments.
    pixel_units : {None, string}
        The pixel units for the scale bar. Normally 
    plot_scalebar, plot_ticks, plot_colorbar, plot_indices : bool
    title : str
        The title is printed at the top of the image.
    vmin, vmax : float
        Limit the range of the color map scale to the given values.
    auto_contrast : bool
        If True, vmin and vmax are calculated automatically.
    min_aspect : float
        Set the minimum aspect ratio of the image and the figure. To 
        keep the image in the aspect limit the pixels are made
        rectangular.
        
    """
    
    def __init__(self):
        self.data_function = None
        self.pixel_units = None
        self.plot_scalebar = True
        self.plot_ticks = False
        self.plot_colorbar = True
        self._colorbar = None
        self.figure = None
        self.ax = None
        self.title = ''
        self.vmin = None
        self.vmax = None
        self.auto_contrast = True
        self._ylabel = ''
        self._xlabel = ''
        self.plot_indices = True
        self._text = None
        self._text_position = (0, 1.05,)
        self.axes_manager = None
        self._aspect = 1
        self._extent = None
        self.xaxis = None
        self.yaxis = None
        self.min_aspect = 0.1
        
    def configure(self):
        xaxis = self.xaxis
        yaxis = self.yaxis
        # Image labels
        self._xlabel = '%s' % str(xaxis)
        if xaxis.units is not Undefined:
            self._xlabel += ' (%s)' % xaxis.units
        
        self._ylabel = '%s' % str(yaxis)
        if yaxis.units is not Undefined:
            self._ylabel += ' (%s)' % yaxis.units 
            
        if (xaxis.units == yaxis.units) and (
            xaxis.scale == yaxis.scale):
            self.plot_scalebar = True
            self.plot_ticks = False
            self.pixel_units = xaxis.units
        else:
            self.plot_scalebar = False
            self.plot_ticks = True
            
        # Calibrate the axes of the navigator image
        self._extent = (xaxis.axis[0] -  xaxis.scale / 2.,
                       xaxis.axis[-1] + xaxis.scale / 2.,
                       yaxis.axis[-1] + yaxis.scale / 2.,
                       yaxis.axis[0]  - yaxis.scale / 2.)
        # Apply aspect ratio constraint
        if self.min_aspect:
            min_asp = self.min_aspect
            if yaxis.size / xaxis.size < min_asp: 
                factor = min_asp * xaxis.size / yaxis.size
                self.plot_scalebar = False
                self.plot_ticks = True
            elif yaxis.size / xaxis.size > min_asp**-1:
                factor = min_asp**-1 * xaxis.size / yaxis.size
                self.plot_scalebar = False
                self.plot_ticks = True
            else:
                factor = 1
        self._aspect = np.abs(factor * xaxis.scale / yaxis.scale)

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
        
    def create_figure(self, max_size=8, min_size=2):
        if self.plot_scalebar is True:
        
            wfactor = 1.1
        else:
            wfactor = 1
        height = abs(self._extent[3] - self._extent[2]) * self._aspect
        width = abs(self._extent[1] - self._extent[0])
        figsize = np.array((width * wfactor, height)) * max_size / max(
                           (width * wfactor, height))
        self.figure = utils.create_figure(
                        window_title=("Figure " + self.title
                                        if self.title
                                        else None),
                        figsize=figsize.clip(min_size, max_size))
        self.figure.canvas.mpl_connect('draw_event', self._on_draw)

        
    def create_axis(self):
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self._xlabel)
        self.ax.set_ylabel(self._ylabel)
        if self.plot_ticks is False:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        self.ax.hspy_fig = self

        
    def plot(self):
        self.configure()
        if not utils.does_figure_object_exists(self.figure):
            self.create_figure()
            self.create_axis()   
        data = self.data_function()
        if self.auto_contrast is True:
            self.optimize_contrast(data)
        if (not self.axes_manager or 
            self.axes_manager.navigation_size==0):
            self.plot_indices = False
        if self.plot_indices is True:
            self._text = self.ax.text(*self._text_position,
                            s=str(self.axes_manager.indices),
                            transform = self.ax.transAxes,
                            fontsize=12,
                            color='red')
        self.update()
        if self.plot_scalebar is True:
            if self.pixel_units is not None:
                self.ax.scalebar = widgets.Scale_Bar(
                    ax=self.ax,
                    units=self.pixel_units,)
                 
        if self.plot_colorbar is True:
            self._colorbar = plt.colorbar(self.ax.images[0], ax=self.ax)
        
        self.figure.canvas.draw()
        if hasattr(self.figure, 'tight_layout'):
            self.figure.tight_layout()
        self.connect()
        
    def update(self, auto_contrast=None):
        ims = self.ax.images
        if ims:
            ims.remove(ims[0])
        data = self.data_function()
        numrows, numcols = data.shape
        def format_coord(x, y):
            col = self.xaxis.value2index(x)
            row = self.yaxis.value2index(y)
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
            
        self.ax.imshow(data,
                       interpolation='nearest',
                       vmin=self.vmin, 
                       vmax=self.vmax,
                       extent=self._extent,
                       aspect=self._aspect)
        if self.plot_indices is True:
            self._text.set_text((self.axes_manager.indices))
        self.figure.canvas.draw()
        
    def _update(self):
        # This "wrapper" because on_trait_change fiddles with the 
        # method arguments and auto_contrast does not work then
        self.update()
    def adjust_contrast(self):
        ceditor = ImageContrastEditor(self)
        ceditor.edit_traits()
        return ceditor
        
    def connect(self):
        self.figure.canvas.mpl_connect('key_press_event',
                                        self.on_key_press)
        self.figure.canvas.draw()
        if self.axes_manager:
            self.axes_manager.connect(self._update)

    def on_key_press(self, event):
        if event.key == 'h':
            self.adjust_contrast()
                    
    def set_contrast(self, vmin, vmax):
        self.vmin, self.vmax =  vmin, vmax
        self.update()
            
    def optimize_colorbar(self,
                          number_of_ticks=5,
                          tolerance=5,
                          step_prec_max=1):
        vmin, vmax = self.vmin, self.vmax
        _range = vmax - vmin
        step = _range / (number_of_ticks - 1)
        step_oom = math_tools.order_of_magnitude(step)
        def optimize_for_oom(oom):
            self.colorbar_step = math.floor(step / 10**oom)*10**oom
            self.colorbar_vmin = math.floor(vmin / 10**oom)*10**oom
            self.colorbar_vmax = self.colorbar_vmin + \
            self.colorbar_step * (number_of_ticks - 1)
            self.colorbar_locs = np.arange(0, number_of_ticks
                        )* self.colorbar_step + self.colorbar_vmin
        def check_tolerance():
            if abs(self.colorbar_vmax - vmax) / vmax > (
                tolerance / 100.) or abs(self.colorbar_vmin - vmin
                ) >  (tolerance / 100.):
                return True
            else:
                return False
                    
        optimize_for_oom(step_oom)
        i = 1
        while check_tolerance() and i <= step_prec_max:
            optimize_for_oom(step_oom - i)
            i += 1
            
    def disconnect(self):
        if self.axes_manager:
            self.axes_manager.disconnect(self._update)
            
    def close(self):
        self.disconnect()
        if utils.does_figure_object_exists(self.figure) is True:
            plt.close(self.figure)
            
    def _on_draw(self, *args):
        canvas = self.figure.canvas
        self._background = canvas.copy_from_bbox(self.figure.bbox)
        self._draw_animated()
        
    def _draw_animated(self):
        canvas = self.ax.figure.canvas
        canvas.restore_region(self._background)
        ax = self.ax
        artists = []
        artists.extend(ax.collections)
        artists.extend(ax.patches)
        artists.extend(ax.lines)
        artists.extend(ax.texts)
        artists.extend(ax.artists)
        [ax.draw_artist(a) for a in artists if a.get_animated()]
        canvas.blit()
