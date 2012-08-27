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
import copy

import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np
import traits

from utils import on_figure_window_close
#if self.blit is True:
#                self.patch.set_animated(True)
#                canvas = self.ax.figure.canvas
#                canvas.draw()
#                self.background = canvas.copy_from_bbox(self.ax.bbox)
#                self.ax.draw_artist(self.patch)
#                self.ax.figure.canvas.blit()
class DraggablePatch(object):
    """
    """

    def __init__(self, axes_manager = None):
        """
        Add a cursor to ax.
        """
        self.axes_manager = axes_manager
        self.ax = None
        self.picked = False
        self.size = 1.
        self.color = 'red'
        self.__is_on = True
        self._2D = True # Whether the cursor lives in the 2D dimension
        self.patch = None
        self.cids = list()
        # Blitting seems to be supported by all the backends but Qt4
        # however, ATM there is a bug in Hyperspy that precludes using 2 pointers
        # when blit is active, therefore it is disable by default for the moment
        self.blit = False #(plt.get_backend() !=  'Qt4Agg')

    def is_on(self):
        return self.__is_on

    def set_on(self, value):
        if value is not self.is_on():
            if value is True:
                self.add_patch_to(self.ax)
                self.connect(self.ax)
            elif value is False:
                if self._2D:
                    self.ax.patches.remove(self.patch)
                else:
                    self.ax.lines.remove(self.patch)
                self.we_are_animated.remove(self.patch)
                self.disconnect(self.ax)
            self.__is_on = value
            self.ax.figure.canvas.draw()

    def set_patch(self):
        pass
        # Must be provided by the subclass

    def add_patch_to(self, ax):
        self.set_patch()
        if self._2D is True:
            ax.add_patch(self.patch)
        else:
            ax.add_line(self.patch)
        canvas = ax.figure.canvas
        if not hasattr(canvas, 'we_are_animated'):
            canvas.we_are_animated = list()
        ax.figure.canvas.we_are_animated.append(self.patch)

    def add_axes(self, ax):
        self.ax = ax
        canvas = ax.figure.canvas

        if self.is_on() is True:
            self.add_patch_to(ax)
            self.connect(ax)
            canvas.draw()

    def connect(self, ax):
        canvas = ax.figure.canvas
        self.cids.append(canvas.mpl_connect('motion_notify_event', self.onmove))
        self.cids.append(canvas.mpl_connect('pick_event', self.onpick))
        if self.blit is True:
            self.cids.append(
            canvas.mpl_connect('draw_event', self.update_background))
        self.cids.append(canvas.mpl_connect('button_release_event',
        self.button_release))
        self.axes_manager.connect(self.update_patch_position)
        on_figure_window_close(ax.figure, self.close)

    def disconnect(self, ax):
        for cid in self.cids:
            try:
                ax.figure.canvas.mpl_disconnect(cid)
            except:
                pass
        self.axes_manager.disconnect(self.update_patch_position)

    def close(self, window = None):
        ax = self.ax
        if self._2D is True:
            if self.patch in ax.patches:
                ax.patches.remove(self.patch)
        else:
            if self.patch in ax.lines:
                ax.lines.remove(self.patch)
        self.disconnect(ax)

    def onpick(self, event):
        if event.artist is self.patch:
            self.picked = True

    def onmove(self, event):
        """This method must be provided by the subclass"""
        pass

    def update_patch_position(self):
        """This method must be provided by the subclass"""
        pass

    def button_release(self, event):
        'whenever a mouse button is released'
        if event.button != 1: return
        if self.picked is True:
            self.picked = False

    def update_background(self, *args):
        if self.blit is True:
            canvas = self.ax.figure.canvas
            self.background = canvas.copy_from_bbox(self.ax.bbox)
            for artist in canvas.we_are_animated:
                self.ax.draw_artist(artist)
            self.ax.figure.canvas.blit()

    def draw_patch(self, *args):
        canvas = self.ax.figure.canvas
        if self.blit is True:
            canvas.restore_region(self.background)
            # redraw just the current rectangle
            for artist in canvas.we_are_animated:
                self.ax.draw_artist(artist)
            # blit just the redrawn area
            canvas.blit()
        else:
            canvas.draw()

class ResizebleDraggablePatch(DraggablePatch):

    def __init__(self, axes_manager):
        DraggablePatch.__init__(self, axes_manager)
        self.size = 1.

    def set_size(self, size):
        self.size = size
        self.update_patch_size()

    def increase_size(self):
        self.set_size(self.size + 1)

    def decrease_size(self):
        if self.size > 1:
            self.set_size(self.size - 1)

    def update_patch_size(self):
        """This method must be provided by the subclass"""
        pass

    def on_key_press(self, event):
        if event.key == "+":
            self.increase_size()
        if event.key == "-":
            self.decrease_size()

    def connect(self, ax):
        DraggablePatch.connect(self, ax)
        canvas = ax.figure.canvas
        self.cids.append(canvas.mpl_connect('key_press_event', self.on_key_press))

class DraggableSquare(ResizebleDraggablePatch):

    def __init__(self, axes_manager):
        DraggablePatch.__init__(self, axes_manager)

    def set_patch(self):
        indexes = self.axes_manager._indexes[::-1]
        self.patch = \
        plt.Rectangle(indexes - (self.size / 2.,) * 2,
        self.size, self.size, animated = self.blit,
        fill = False, lw = 2,  ec = self.color, picker = True,)

    def update_patch_size(self):
        self.patch.set_width(self.size)
        self.patch.set_height(self.size)
        self.update_patch_position()

    def update_patch_position(self):
        indexes = self.axes_manager._indexes[::-1]
        self.patch.set_xy(indexes - (self.size / 2.,) * 2)
        self.draw_patch()

    def onmove(self, event):
        'on mouse motion draw the cursor if picked'

        if self.picked is True and event.inaxes:
            if self.axes_manager._indexes[0] != round(event.ydata):
                try:
                    self.axes_manager.navigation_axes[0].index = \
                    round(event.ydata)
                except traits.api.TraitError:
                    # Index out of range, we do nothing
                    pass
                    
            if  self.axes_manager._indexes[1] != round(event.xdata):
                try:
                    self.axes_manager.navigation_axes[1].index = \
                    round(event.xdata)
                except traits.api.TraitError:
                    # Index out of range, we do nothing
                    pass

class DraggableHorizontalLine(DraggablePatch):
    def __init__(self, axes_manager):
        DraggablePatch.__init__(self, axes_manager)
        self._2D = False
        # Despise the bug, we use blit for this one because otherwise the
        # it gets really slow
        self.blit = True

    def update_patch_position(self):
        if self.patch is not None:
            self.patch.set_ydata(self.axes_manager._indexes[0])
            self.draw_patch()

    def set_patch(self):
        ax = self.ax
        self.patch = ax.axhline(self.axes_manager._indexes[0],
                                color = self.color,
                               picker = 5, animated = self.blit)

    def onmove(self, event):
        'on mouse motion draw the cursor if picked'
        if self.picked is True and event.inaxes:
            try:
                self.axes_manager.navigation_axes[0].index = event.ydata
            except traits.api.TraitError:
                # Index out of range, we do nothing
                pass

class DraggableVerticalLine(DraggablePatch):
    def __init__(self, axes_manager):
        DraggablePatch.__init__(self, axes_manager)
        self._2D = False
        # Despise the bug, we use blit for this one because otherwise the
        # it gets really slow
        self.blit = True

    def update_patch_position(self):
        if self.patch is not None:
            self.patch.set_xdata(self.axes_manager._values[0])
            self.draw_patch()

    def set_patch(self):
        ax = self.ax
        self.patch = ax.axvline(self.axes_manager._values[0],
                                color = self.color,
                               picker = 5, animated = self.blit)

    def onmove(self, event):
        'on mouse motion draw the cursor if picked'
        if self.picked is True and event.inaxes:
            try:
                self.axes_manager.navigation_axes[0].value = event.xdata
            except traits.api.TraitError:
                # Index out of range, we do nothing
                pass

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

def in_interval(number, interval):
        if number >= interval[0] and number <= interval[1]:
            return True
        else:
            return False

class ModifiableSpanSelector(matplotlib.widgets.SpanSelector):

    def __init__(self, ax, **kwargs):
        matplotlib.widgets.SpanSelector.__init__(
        self, ax, direction = 'horizontal', useblit = False, **kwargs)
        self.tolerance = 1 # The tolerance in points to pick the rectangle sizes
        self.on_move_cid = None
        self.range = None

    def release(self, event):
        """When the button is realeased, the span stays in the screen and the
        iteractivity machinery passes to modify mode"""
        if self.pressv is None or (self.ignore(event) and not self.buttonDown):
            return
        self.buttonDown = False
        self.update_range()
        self.onselect()
        # We first disconnect the previous signals
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)

        # And connect to the new ones
        self.cids.append(
        self.canvas.mpl_connect('button_press_event', self.mm_on_press))
        self.cids.append(
        self.canvas.mpl_connect('button_release_event', self.mm_on_release))
        self.cids.append(
        self.canvas.mpl_connect('draw_event', self.update_background))

    def mm_on_press(self, event):
        if (self.ignore(event) and not self.buttonDown): return
        self.buttonDown = True

        # Calculate the point size in data units
        invtrans = self.ax.transData.inverted()
        x_pt = abs((invtrans.transform((1,0)) -
        invtrans.transform((0,0)))[0])

        # Determine the size of the regions for moving and stretching
        rect = self.rect
        self.range = rect.get_x(), rect.get_x() + rect.get_width()
        left_region = self.range[0] - x_pt, self.range[0] + x_pt
        right_region = self.range[1] - x_pt, self.range[1] + x_pt
        middle_region = self.range[0] + x_pt, self.range[1] - x_pt

        if in_interval(event.xdata, left_region) is True:
            self.on_move_cid = \
            self.canvas.mpl_connect('motion_notify_event', self.move_left)
        elif in_interval(event.xdata, right_region):
            self.on_move_cid = \
            self.canvas.mpl_connect('motion_notify_event', self.move_right)
        elif in_interval(event.xdata, middle_region):
            self.pressv = event.xdata
            self.on_move_cid = \
            self.canvas.mpl_connect('motion_notify_event', self.move_rect)
        else:
            return
    def update_range(self):
        self.range = (self.rect.get_x(),
            self.rect.get_x() + self.rect.get_width())
    def move_left(self, event):
        if self.buttonDown is False or self.ignore(event): return
        width_increment = self.range[0] - event.xdata
        self.rect.set_x(event.xdata)
        self.rect.set_width(self.rect.get_width() + width_increment)
        self.update_range()
        if self.onmove_callback is not None:
            self.onmove_callback(*self.range)
        self.update()

    def move_right(self, event):
        if self.buttonDown is False or self.ignore(event): return
        width_increment = \
        event.xdata - self.range[1]
        self.rect.set_width(self.rect.get_width() + width_increment)
        self.update_range()
        if self.onmove_callback is not None:
            self.onmove_callback(*self.range)
        self.update()

    def move_rect(self, event):
        if self.buttonDown is False or self.ignore(event): return
        x_increment = event.xdata - self.pressv
        self.rect.set_x(self.rect.get_x() + x_increment)
        self.update_range()
        self.pressv = event.xdata
        if self.onmove_callback is not None:
            self.onmove_callback(*self.range)
        self.update()

    def mm_on_release(self, event):
        if self.buttonDown is False or self.ignore(event): return
        self.buttonDown = False
        self.canvas.mpl_disconnect(self.on_move_cid)
        self.on_move_cid = None

    def turn_off(self):
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)
        if self.on_move_cid is not None:
            self.canvas.mpl_disconnect(cid)
        self.ax.patches.remove(self.rect)
        self.ax.figure.canvas.draw()
