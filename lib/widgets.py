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

import copy

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
import numpy as np

from coordinates import pointer, explorer
from utils import on_window_close

class foo:
    pass


class SquarePointer(object):
    """
    """

    def __init__(self):
        """
        Add a pointer to ax.
        """
        self.ax = list()
        self.callback_on_move = list()
        self.picked = None
        self.pointers = list()
        self.explorers = list()
        self.explorer_ON = False
        self.picked_list = None
        self.__pointer = plt.Rectangle(pointer.coordinates - (0.5,0.5), 
        1, 1, fc = 'r', fill= False,lw = 2, animated = True, picker = True, 
        ec = 'black')
        self.__explorer = plt.Rectangle(explorer.coordinates - (0.5, 0.5), 
        1,1, fc = 'r', fill= False,lw = 2, animated = True, picker = True, 
        ec = 'white')
        
    def add_axes(self, ax):
        self.ax.append(ax)
        canvas = ax.figure.canvas
        if not hasattr(canvas, 'background'):
            canvas.background = None
        self.connect(ax)
        self.add_pointer(ax)
        if self.explorer_ON:
            self.add_explorer(ax)
            
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
            self.explorer_ON = not self.explorer_ON
    def add_pointer(self, ax):
        self.pointers.append(copy.copy(self.__pointer))
        ax.add_patch(self.pointers[-1])
        
    def add_explorer(self, ax):
        self.explorers.append(copy.copy(self.__explorer))
        ax.add_patch(self.explorers[-1])
    
    def remove_explorer(self, ax):
        for explorer in self.explorers:
            if explorer in ax.patches:
                ax.patches.remove(explorer)
                self.explorers.remove(explorer)
            
    def remove_axes(self,window):
        for ax in self.ax:
            if hasattr(ax.figure.canvas.manager, 'window'):
                ax_window = ax.figure.canvas.manager.window
            else:
                ax_window = False
            if ax_window is window or not ax_window:
                for patch in ax.patches:
                    if patch in self.pointers:
                        self.pointers.remove(patch)
                    elif patch in self.explorers:
                        self.explorers.remove(patch)
                for cid in ax.cids:
                    ax.figure.canvas.mpl_disconnect(cid)
                self.ax.remove(ax)
                if not self.ax:
                    self.callback_on_move = list()
        
    def _set_explorerON(self, value):
        if value:
            if self.explorer_ON: return
            self.__explorerON = True
            self._turn_on_explorer()
            for ax in self.ax:
                ax.figure.canvas.draw_idle()
        elif not value:
            self.__explorerON = False
            self._turn_off_explorer()
            for ax in self.ax:
                ax.figure.canvas.draw_idle()
    def _get_explorerON(self):
        return self.__explorerON
    def _turn_on_explorer(self):
        for ax in self.ax:
            self.add_explorer(ax)
    def _turn_off_explorer(self):
        for ax in self.ax:
            self.remove_explorer(ax)
    explorer_ON = property(_get_explorerON, _set_explorerON)
    
    def clear(self, event = None):
        'clear the cursor'
        canvas = event.canvas
        ax = canvas.figure.axes[0]
        if ax:
            canvas.background = canvas.copy_from_bbox(ax.bbox)
            for patch in ax.patches:
                ax.draw_artist(patch)

    def onpick(self, event):
        if event.artist in self.pointers:
            self.picked_list = self.pointers
            self.picked = pointer
        elif self.explorer_ON and event.artist in self.explorers:
            self.picked_list = self.explorers
            self.picked = explorer
    def onmove(self, event, bridge = False):
        'on mouse motion draw the cursor if picked'
        if bridge is True:
            if self.picked == pointer:
                self.picked_list = self.pointers
            elif self.picked == explorer:
                self.picked_list = self.explorers
        if self.picked_list is not None:
            if event.inaxes:
                new_coordinates = np.array((round(event.xdata), 
                round(event.ydata)))
                if not (new_coordinates == self.picked.coordinates).all():
                    self.picked.ix, self.picked.iy = new_coordinates
                    for patch in self.picked_list:
                        patch.set_x(self.picked.ix-0.5)
                        patch.set_y(self.picked.iy-0.5)
                    for ax in self.ax:
                        self._update(ax)
                    for function in self.callback_on_move:
                        function()
        if bridge:
            self.picked_list = None

    def button_release(self, event):
        'whenever a mouse button is released'
        if event.button != 1: return
        self.picked_list = None
    
    def _update(self,ax):
        background = ax.figure.canvas.background
        if background is not None:
            ax.figure.canvas.restore_region(background)
        for patch in ax.patches:
            ax.draw_artist(patch)
        ax.figure.canvas.blit(ax.bbox)
    def reset(self):
        for ax in self.ax:
            self.remove_explorer(ax)
            self.remove_axes(ax.figure.canvas.manager.window)
        self.callback_on_move = list()
        

class LinePointer(object):
    """
    """

    def __init__(self):
        """
        Add a pointer to ax.
        """
        self.ax = list()
        self.callback_on_move = list()
        self.picked = None
        self.pointers = list()
        self.explorers = list()
        self.explorer_ON = False
        self.picked_list = None
        
    def add_axes(self, ax):
        self.ax.append(ax)
        canvas = ax.figure.canvas
        ax.figure.canvas.manager.window.ax = ax
        if not hasattr(canvas, 'background'):
            canvas.background = None
        self.connect(ax)
        self.add_pointer(ax)
        if self.explorer_ON:
            self.add_explorer(ax)
            
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
            self.explorer_ON = not self.explorer_ON
            
    def add_pointer(self, ax):
        ax.axhline(pointer.ix,color='red', animated = True, picker=True )
        self.pointers.append(ax.lines[-1])
        
    def add_explorer(self, ax):
        ax.axhline(pointer.ix,color='white', animated = True, picker=True )
        self.explorers.append(ax.lines[-1])
    
    def remove_explorer(self, ax):
        for explorer in self.explorers:
            if explorer in ax.lines:
                ax.lines.remove(explorer)
                self.explorers.remove(explorer)
            
    def remove_axes(self,window):
        for ax in self.ax:
            if window.ax is ax:
                for line in ax.lines:
                    if line in self.pointers:
                        self.pointers.remove(line)
                    elif line in self.explorers:
                        self.explorers.remove(line)
                for cid in ax.cids:
                    ax.figure.canvas.mpl_disconnect(cid)
                self.ax.remove(ax)
                if not self.ax:
                    self.callback_on_move = list()
        
    def _set_explorerON(self, value):
        if value:
            if self.explorer_ON: return
            self.__explorerON = True
            self._turn_on_explorer()
            for ax in self.ax:
                ax.figure.canvas.draw_idle()
        elif not value:
            self.__explorerON = False
            self._turn_off_explorer()
            for ax in self.ax:
                ax.figure.canvas.draw_idle()
    def _get_explorerON(self):
        return self.__explorerON
    def _turn_on_explorer(self):
        for ax in self.ax:
            self.add_explorer(ax)
    def _turn_off_explorer(self):
        for ax in self.ax:
            self.remove_explorer(ax)
    explorer_ON = property(_get_explorerON, _set_explorerON)
    
    def clear(self, event = None):
        'clear the cursor'
        canvas = event.canvas
        ax = canvas.figure.axes[0]
        if ax:
            canvas.background = canvas.copy_from_bbox(ax.bbox)
            for line in ax.lines:
                ax.draw_artist(line)

    def onpick(self, event):
        if event.artist in self.pointers:
            self.picked_list = self.pointers
            self.picked = pointer
        elif self.explorer_ON and event.artist in self.explorers:
            self.picked_list = self.explorers
            self.picked = explorer
    def onmove(self, event, bridge = False):
        'on mouse motion draw the cursor if picked'
        if bridge:
            if self.picked == pointer:
                self.picked_list = self.pointers
            elif self.picked == explorer:
                self.picked_list = self.explorers
                
        if self.picked_list:
            if event.inaxes:
                if not event.ydata == self.picked.ix:
                    self.picked.ix = event.ydata
                    for line in self.picked_list:
                        line.set_ydata(self.picked.ix)
                    for ax in self.ax:
                        self._update(ax)
                    for function in self.callback_on_move:
                        function()
        if bridge:
            self.picked_list = None

    def button_release(self, event):
        'whenever a mouse button is released'
        if event.button != 1: return
        self.picked_list = None
    
    def _update(self,ax):
        background = ax.figure.canvas.background
        if background is not None:
            ax.figure.canvas.restore_region(background)
        for line in ax.lines:
            ax.draw_artist(line)
        ax.figure.canvas.blit(ax.bbox)
    def reset(self):
        for ax in self.ax:
            self.remove_explorer(ax)
            self.remove_axes(ax.figure.canvas.manager.window)
        self.callback_on_move = list()

cursors = SquarePointer()
lines = LinePointer()

