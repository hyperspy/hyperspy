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

from __future__ import division
import copy

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
import numpy as np

from utils import on_window_close

    
class MovingPatch(object):
    """
    """

    def __init__(self, coordinates):
        """
        Add a cursor to ax.
        """
        self.coordinates = coordinates
        self.axs = list()
        self.picked = False
        self.size = 1.
        self.color = 'red'
        self.patch = None # Must be provided by the subclass
        self.__is_on = True
    
    def is_on(self):
        return self.__is_on
        
    def set_is_on(self, value):
        self.__is_on = value
        
    def add_axes(self, ax):
        self.axs.append(ax)
        canvas = ax.figure.canvas
        if not hasattr(canvas, 'background'):
            canvas.background = None
        if self.is_on() is True:
            self.connect(ax)
            ax.add_patch(self.patch)
#            self.draw_patch(axs = [ax,])
            
    def connect(self, ax):
        if not hasattr(ax, 'cids'):
            ax.cids = list()
        canvas = ax.figure.canvas
        ax.cids.append(canvas.mpl_connect('motion_notify_event', self.onmove))
        ax.cids.append(canvas.mpl_connect('pick_event', self.onpick))
        ax.cids.append(canvas.mpl_connect('draw_event', self.clear))
        ax.cids.append(canvas.mpl_connect('button_release_event', 
        self.button_release))
        self.coordinates.connect(self.update_patch_position)
        on_window_close(ax.figure, self.remove_axes)

        
    def remove_axes(self,window):
        for ax in self.axs:
            if hasattr(ax.figure.canvas.manager, 'window'):
                ax_window = ax.figure.canvas.manager.window
            else:
                ax_window = False
            if ax_window is window or ax_window is False:
                if self.patch in ax.patches:
                    ax.patches.remove(self.patch)
                for cid in ax.cids:
                    ax.figure.canvas.mpl_disconnect(cid)
                self.axs.remove(ax)
                if not self.axs:
                    self.call_on_move = list()
    
    def clear(self, event = None):
        'clear the patch'
        canvas = event.canvas
        if canvas.figure.axes:
            ax = canvas.figure.axes[0]
            canvas.background = canvas.copy_from_bbox(ax.bbox)
            for patch in ax.patches:
                ax.draw_artist(patch)

    def onpick(self, event):
        if event.artist is self.patch:
            self.picked = True

    def onmove(self, event):
        'on mouse motion draw the cursor if picked'
        if self.picked is True and event.inaxes:
            new_coordinates = np.array((round(event.xdata), 
            round(event.ydata)))
            if not (new_coordinates == self.coordinates.coordinates).all():
                self.coordinates.ix, self.coordinates.iy = new_coordinates
            self.update_patch_position()
            
    def update_patch_position(self):
        '''This method must be provided by the subclass'''
        pass
    
    def button_release(self, event):
        'whenever a mouse button is released'
        if event.button != 1: return
        if self.picked is True:
            self.picked = False
    
    def draw_patch(self, axs = None):
        if self.is_on() is True:
            if axs is None:
                axs = self.axs
            for ax in self.axs:
                background = ax.figure.canvas.background
                if background is not None:
                    ax.figure.canvas.restore_region(background)
                ax.draw_artist(self.patch)
                ax.figure.canvas.blit(ax.bbox)

class ResizebleMovingPatch(MovingPatch):
    
    def __init__(self, coordinates):
        MovingPatch.__init__(self, coordinates)
        self.size = 1
    
    def set_size(self, size):
        self.size = size
        self.update_patch_size()
    
    def increase_size(self):
        self.size += 1
        self.update_patch_size()
        
    def decrease_size(self):
        self.size -= 1
        self.update_patch_size()
    def update_patch_size(self):
        '''This method must be provided by the subclass'''
        pass
    
    def on_key_press(self, event):
        if event.key == "+":
            self.increase_size()
        if event.key == "-":
            self.decrease_size()
    def connect(self, ax):
        MovingPatch.connect(self, ax)
        canvas = ax.figure.canvas
        ax.cids.append(canvas.mpl_connect('key_press_event', self.on_key_press))

class MovingSquare(ResizebleMovingPatch):
    
    def __init__(self, coordinates):
        MovingPatch.__init__(self, coordinates)
        self.patch = \
        plt.Rectangle(
        self.coordinates.coordinates - (self.size / 2, self.size / 2), 
        self.size, self.size, 
        fill= False, lw = 2,  ec = self.color,
        animated = True, picker = True,)
        
    def update_patch_size(self):
        self.patch.set_width(self.size)
        self.patch.set_height(self.size)
        self.draw_patch()
        
    def update_patch_position(self):
        if self.patch is not None:
            delta = self.size / 2
            self.patch.set_xy(self.coordinates.coordinates - (delta, delta))
            self.draw_patch()
        
class MovingHorizontalLine(MovingPatch):
            
    def update_patch_position(self):
        if self.patch is not None:
            self.patch.set_ydata(self.coordinates.ix)
            self.draw_patch()
        
    def add_axes(self, ax):
        self.axs.append(ax)
        canvas = ax.figure.canvas
        if not hasattr(canvas, 'background'):
            canvas.background = None
        if self.is_on() is True:
            if self.patch is None:
                self.patch = ax.axhline(self.coordinates.ix, color = self.color, 
                                    animated = True, picker = True)
            else:
                ax.add_line(self.patch)
            self.connect(ax)
            self.update_patch_position()
            
    def onmove(self, event):
        'on mouse motion draw the cursor if picked'
        if self.picked is True and event.inaxes:
            if not round(event.ydata) == self.coordinates.ix:
                self.coordinates.ix = round(event.ydata)
            
class Scale_Bar():
    def __init__(self, ax, units, pixel_size, color = 'white', position = None, 
    ratio = 0.25, lw = 1, lenght_in_units = None):
        self.axs = ax
        self.units = units
        self.pixel_size = pixel_size
        self.xmin, self.xmax = ax.get_xlim()
        self.ymin, self.ymax = ax.get_ylim()
        self.text = None
        self.line = None
        self.tex_bold = False
        self.lenght_in_units = lenght_in_units
        if position is None:
            self.position = self.calculate_line_position()
        else:
            self.position = position
        self.calculate_scale_size(ratio = ratio)
        self.calculate_text_position()
        self.plot_scale(line_width = lw)
        self.set_color(color)
        
    def get_units_string(self):
        if self.tex_bold is True:
            if (self.units[0] and self.units[-1]) == '$':
                return r'$\mathbf{%i\,%s}$' % \
            (self.lenght_in_units, self.units[1:-1])
            else:
                return r'$\mathbf{%i\,}$\textbf{%s}' % \
            (self.lenght_in_units, self.units)
        else:
            return r'$%i\,$%s' % (self.lenght_in_units, self.units)
    def calculate_line_position(self):
        return 0.95*self.xmin + 0.05*self.xmax, 0.95*self.ymin+0.05*self.ymax
    def calculate_text_position(self, pad = 1/180.):
        pad = float(pad)
        x1, y1 = self.position
        x2, y2 = x1 + self.lenght_in_pixels, y1
        self.text_position = (x1+x2)/2, y2+self.ymax * pad
    def calculate_scale_size(self, ratio = 0.25):
        if self.lenght_in_units is None:
            self.lenght_in_units = (self.xmax * self.pixel_size) // (1/ratio)
        self.lenght_in_pixels = self.lenght_in_units / self.pixel_size
    def delete_scale_if_exists(self):
        if self.line is not None:
            self.axs.lines.remove(self.line)
        if self.text is not None:
            self.axs.texts.remove(self.text)
    def plot_scale(self, line_width = 1):
        self.delete_scale_if_exists()
        x1, y1 = self.position
        x2, y2 = x1 + self.lenght_in_pixels, y1
        self.line, = self.axs.plot([x1,x2],[y1,y2], linestyle='-', 
        lw = line_width)
        self.text = self.axs.text(*self.text_position, s=self.get_units_string(), 
        ha = 'center', size = 'medium') 
        self.axs.set_xlim(self.xmin, self.xmax)
        self.axs.set_ylim(self.ymin, self.ymax)
        self.axs.figure.canvas.draw_idle()
    def set_position(self,x,y):
        self.position = x, y
        self.calculate_text_position()
        self.plot_scale(line_width = self.line.get_linewidth())

    def set_color(self, c):
        self.line.set_color(c)
        self.text.set_color(c)
        self.axs.figure.canvas.draw_idle()
    def set_lenght_in_units(self, lenght):
        color = self.line.get_color()
        self.lenght_in_units = lenght
        self.calculate_scale_size()
        self.calculate_text_position()
        self.plot_scale(line_width = self.line.get_linewidth())
        self.set_color(color)
    def set_tex_bold(self):
        self.tex_bold = True
        self.text.set_text(self.get_units_string())
        self.axs.figure.canvas.draw_idle()


def scale_bar(ax, units, pixel_size, color = 'white', position = None, 
    ratio = 0.25, lw = 1, lenght_in_units = None):
    ax.scale_bar = Scale_Bar(ax = ax, units = units, pixel_size = pixel_size, 
    color = color, position = position, ratio = ratio, lw = lw, 
    lenght_in_units = lenght_in_units)


