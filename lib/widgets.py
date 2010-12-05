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

from coordinates import cursor, cursor2
from utils import on_window_close


class SquarePointer(object):
    """
    """

    def __init__(self):
        """
        Add a cursor to ax.
        """
        self.ax = list()
        self.callback_on_move = list()
        self.picked = None
        self.cursors = list()
        self.cursor2s = list()
        self.cursor2_ON = False
        self.picked_list = None
        self.square_width = 1.
        self.cursor_color = 'red'
        self.cursor2_color = 'blue'
        self.__cursor = plt.Rectangle(cursor2.coordinates - 
        (self.square_width / 2, self.square_width / 2), 
        self.square_width,self.square_width, fc = 'r', fill= False,lw = 2, 
        animated = True, picker = True, ec = self.cursor_color)
        self.__cursor2 = plt.Rectangle(cursor2.coordinates - 
        (self.square_width / 2, self.square_width / 2), 
        self.square_width,self.square_width, fc = 'r', fill= False,lw = 2, 
        animated = True, picker = True, ec = self.cursor2_color)
        cursor.connect(self._update_squares)
        cursor2.connect(self._update_squares)
        
    def set_square_width(self, width):
        self.square_width = width
        self._update_squares()
        
    def _update_squares(self):
        for rect in self.cursors:
            rect.set_width(self.square_width)
            rect.set_height(self.square_width)
            delta = self.square_width / 2
            rect.set_xy(cursor.coordinates - (delta, delta))
        for rect in self.cursor2s:
            rect.set_width(self.square_width)
            rect.set_height(self.square_width)
            delta = self.square_width / 2
            rect.set_xy(cursor2.coordinates - (delta, delta))
        self._update()
    
    def increase_square_width(self):
        self.square_width += 1
        self._update_squares()
        
    def decrease_square_width(self):
        self.square_width -= 1
        self._update_squares()
        
    def add_axes(self, ax):
        self.ax.append(ax)
        canvas = ax.figure.canvas
        if not hasattr(canvas, 'background'):
            canvas.background = None
        self.connect(ax)
        self.add_cursor(ax)
        if self.cursor2_ON:
            self.add_cursor2(ax)
            
    def connect(self, ax):
        if not hasattr(ax, 'cids'):
            ax.cids = list()
        canvas = ax.figure.canvas
        ax.cids.append(canvas.mpl_connect('motion_notify_event', self.onmove))
        ax.cids.append(canvas.mpl_connect('pick_event', self.onpick))
        ax.cids.append(canvas.mpl_connect('draw_event', self.clear))
        ax.cids.append(canvas.mpl_connect('button_release_event', 
        self.button_release))
        ax.cids.append(canvas.mpl_connect('key_press_event', self.on_key_press))
        on_window_close(ax.figure, self.remove_axes)


    def on_key_press(self, event):
        if event.key == "e":
            self.cursor2_ON = not self.cursor2_ON
        if event.key == "+":
            self.increase_square_width()
        if event.key == "-":
            self.decrease_square_width()
        
    def add_cursor(self, ax):
        self.cursors.append(copy.copy(self.__cursor))
        ax.add_patch(self.cursors[-1])
        
    def add_cursor2(self, ax):
        self.cursor2s.append(copy.copy(self.__cursor2))
        ax.add_patch(self.cursor2s[-1])
    
    def remove_cursor2(self, ax):
        for cursor2 in self.cursor2s:
            if cursor2 in ax.patches:
                ax.patches.remove(cursor2)
                self.cursor2s.remove(cursor2)
            
    def remove_axes(self,window):
        for ax in self.ax:
            if hasattr(ax.figure.canvas.manager, 'window'):
                ax_window = ax.figure.canvas.manager.window
            else:
                ax_window = False
            if ax_window is window or not ax_window:
                for patch in ax.patches:
                    if patch in self.cursors:
                        self.cursors.remove(patch)
                    elif patch in self.cursor2s:
                        self.cursor2s.remove(patch)
                for cid in ax.cids:
                    ax.figure.canvas.mpl_disconnect(cid)
                self.ax.remove(ax)
                if not self.ax:
                    self.callback_on_move = list()
        
    def _set_cursor2ON(self, value):
        if value:
            if self.cursor2_ON: return
            self.__cursor2ON = True
            self._turn_on_cursor2()
            for ax in self.ax:
                ax.figure.canvas.draw_idle()
        elif not value:
            self.__cursor2ON = False
            self._turn_off_cursor2()
            for ax in self.ax:
                ax.figure.canvas.draw_idle()
    def _get_cursor2ON(self):
        return self.__cursor2ON
    def _turn_on_cursor2(self):
        for ax in self.ax:
            self.add_cursor2(ax)
        self._update_squares()
        cursor.coordinates_change_signal()
        cursor2.coordinates_change_signal()

    def _turn_off_cursor2(self):
        for ax in self.ax:
            self.remove_cursor2(ax)
        self._update_squares()
        cursor.coordinates_change_signal()
        cursor2.coordinates_change_signal()

    cursor2_ON = property(_get_cursor2ON, _set_cursor2ON)
    
    def clear(self, event = None):
        'clear the cursor'
        canvas = event.canvas
        ax = canvas.figure.axes[0]
        if ax:
            canvas.background = canvas.copy_from_bbox(ax.bbox)
            for patch in ax.patches:
                ax.draw_artist(patch)

    def onpick(self, event):
        if event.artist in self.cursors:
            self.picked_list = self.cursors
            self.picked = cursor
        elif self.cursor2_ON and event.artist in self.cursor2s:
            self.picked_list = self.cursor2s
            self.picked = cursor2
    def onmove(self, event, bridge = False):
        'on mouse motion draw the cursor if picked'
        if bridge is True:
            if self.picked == cursor:
                self.picked_list = self.cursors
            elif self.picked == cursor2:
                self.picked_list = self.cursor2s
        if self.picked_list is not None:
            if event.inaxes:
                new_coordinates = np.array((round(event.xdata), 
                round(event.ydata)))
                if not (new_coordinates == self.picked.coordinates).all():
                    self.picked.ix, self.picked.iy = new_coordinates
                    for function in self.callback_on_move:
                        function()
        if bridge:
            self.picked_list = None

    def button_release(self, event):
        'whenever a mouse button is released'
        if event.button != 1: return
        self.picked_list = None
    
    def _update(self,axs = None):
        if axs is None:
            axs = self.ax
        for ax in axs:
            background = ax.figure.canvas.background
            if background is not None:
                ax.figure.canvas.restore_region(background)
            for patch in ax.patches:
                ax.draw_artist(patch)
            ax.figure.canvas.blit(ax.bbox)
    def reset(self):
        for ax in self.ax:
            self.remove_cursor2(ax)
            self.remove_axes(ax.figure.canvas.manager.window)
        self.callback_on_move = list()
        

class LinePointer(object):
    """
    """

    def __init__(self):
        """
        Add a cursor to ax.
        """
        self.ax = list()
        self.callback_on_move = list()
        self.picked = None
        self.cursors = list()
        self.cursor2s = list()
        self.cursor2_ON = False
        self.picked_list = None
        cursor.connect(self._update_position)
        cursor2.connect(self._update_position)
        self.cursor_color = 'red'
        self.cursor2_color = 'blue'
        
    def add_axes(self, ax):
        self.ax.append(ax)
        canvas = ax.figure.canvas
        ax.figure.canvas.manager.window.ax = ax
        if not hasattr(canvas, 'background'):
            canvas.background = None
        self.connect(ax)
        self.add_cursor(ax)
        if self.cursor2_ON:
            self.add_cursor2(ax)
            
    def connect(self, ax):
        if not hasattr(ax, 'cids'):
            ax.cids = list()
        canvas = ax.figure.canvas
        ax.cids.append(canvas.mpl_connect('motion_notify_event', self.onmove))
        ax.cids.append(canvas.mpl_connect('pick_event', self.onpick))
        ax.cids.append(canvas.mpl_connect('draw_event', self.clear))
        ax.cids.append(canvas.mpl_connect('button_release_event', 
        self.button_release))
        ax.cids.append(canvas.mpl_connect('key_press_event', self.on_key_press))
        on_window_close(ax.figure, self.remove_axes)


    def on_key_press(self, event):
        if event.key == "e":
            self.cursor2_ON = not self.cursor2_ON
            
    def add_cursor(self, ax):
        self.cursors.append(
        ax.axhline(cursor.ix,color = self.cursor_color, animated = True, 
                   picker=True))
          
    def add_cursor2(self, ax):
        self.cursor2s.append(
        ax.axhline(cursor2.ix,color = self.cursor2_color, animated = True, 
                   picker=True))
      
    def remove_cursor2(self, ax):
        for cursor2 in self.cursor2s:
            if cursor2 in ax.lines:
                ax.lines.remove(cursor2)
                self.cursor2s.remove(cursor2)
            
    def remove_axes(self,window):
        for ax in self.ax:
            if window.ax is ax:
                for line in ax.lines:
                    if line in self.cursors:
                        self.cursors.remove(line)
                    elif line in self.cursor2s:
                        self.cursor2s.remove(line)
                for cid in ax.cids:
                    ax.figure.canvas.mpl_disconnect(cid)
                self.ax.remove(ax)
                if not self.ax:
                    self.callback_on_move = list()
        
    def _set_cursor2ON(self, value):
        if value:
            if self.cursor2_ON: return
            self.__cursor2ON = True
            self._turn_on_cursor2()
            for ax in self.ax:
                ax.figure.canvas.draw_idle()
        elif not value:
            self.__cursor2ON = False
            self._turn_off_cursor2()
            for ax in self.ax:
                ax.figure.canvas.draw_idle()
    def _get_cursor2ON(self):
        return self.__cursor2ON
    def _turn_on_cursor2(self):
        for ax in self.ax:
            self.add_cursor2(ax)
        cursor.coordinates_change_signal()
        cursor2.coordinates_change_signal()
    def _turn_off_cursor2(self):
        for ax in self.ax:
            self.remove_cursor2(ax)
        cursor.coordinates_change_signal()
        cursor2.coordinates_change_signal()
    cursor2_ON = property(_get_cursor2ON, _set_cursor2ON)
    
    def clear(self, event = None):
        'clear the cursor'
        canvas = event.canvas
        ax = canvas.figure.axes[0]
        if ax:
            canvas.background = canvas.copy_from_bbox(ax.bbox)
            for line in ax.lines:
                ax.draw_artist(line)

    def onpick(self, event):
        if event.artist in self.cursors:
            self.picked_list = self.cursors
            self.picked = cursor
        elif self.cursor2_ON and event.artist in self.cursor2s:
            self.picked_list = self.cursor2s
            self.picked = cursor2
            
    def _update_position(self):
        for line in self.cursor2s:
            line.set_ydata(cursor2.ix)
        for line in self.cursors:
            line.set_ydata(cursor.ix)
        self._update()
        
    def onmove(self, event, bridge = False):
        'on mouse motion draw the cursor if picked'
        if bridge:
            if self.picked == cursor:
                self.picked_list = self.cursors
            elif self.picked == cursor2:
                self.picked_list = self.cursor2s
                
        if self.picked_list:
            if event.inaxes:
                if not event.ydata == self.picked.ix:
                    self.picked.ix = event.ydata
                    self._update_position()
                    for function in self.callback_on_move:
                        function()
        if bridge:
            self.picked_list = None

    def button_release(self, event):
        'whenever a mouse button is released'
        if event.button != 1: return
        self.picked_list = None
    
    def _update(self, axs = None):
        if axs is None:
            axs = self.ax
        for ax in axs:
            background = ax.figure.canvas.background
            if background is not None:
                ax.figure.canvas.restore_region(background)
            for line in ax.lines:
                ax.draw_artist(line)
            ax.figure.canvas.blit(ax.bbox)
    def reset(self):
        for ax in self.ax:
            self.remove_cursor2(ax)
            self.remove_axes(ax.figure.canvas.manager.window)
        self.callback_on_move = list()

class Scale_Bar():
    def __init__(self, ax, units, pixel_size, color = 'white', position = None, 
    ratio = 0.25, lw = 1, lenght_in_units = None):
        self.ax = ax
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
            self.ax.lines.remove(self.line)
        if self.text is not None:
            self.ax.texts.remove(self.text)
    def plot_scale(self, line_width = 1):
        self.delete_scale_if_exists()
        x1, y1 = self.position
        x2, y2 = x1 + self.lenght_in_pixels, y1
        self.line, = self.ax.plot([x1,x2],[y1,y2], linestyle='-', 
        lw = line_width)
        self.text = self.ax.text(*self.text_position, s=self.get_units_string(), 
        ha = 'center', size = 'medium') 
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.figure.canvas.draw_idle()
    def set_position(self,x,y):
        self.position = x, y
        self.calculate_text_position()
        self.plot_scale(line_width = self.line.get_linewidth())

    def set_color(self, c):
        self.line.set_color(c)
        self.text.set_color(c)
        self.ax.figure.canvas.draw_idle()
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
        self.ax.figure.canvas.draw_idle()


def scale_bar(ax, units, pixel_size, color = 'white', position = None, 
    ratio = 0.25, lw = 1, lenght_in_units = None):
    ax.scale_bar = Scale_Bar(ax = ax, units = units, pixel_size = pixel_size, 
    color = color, position = position, ratio = ratio, lw = lw, 
    lenght_in_units = lenght_in_units)
    
cursors = SquarePointer()
lines = LinePointer()

