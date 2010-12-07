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
    
class Coordinates(object):
    def __init__(self, shape):
        self.shape = shape
        self.coordinates = np.array((0,0))
        self.on_coordinates_change = []
        self.step = 1
        self.ix = 0
        self.iy = 0
        
    def _set_ix(self, ix):
        if abs(ix) >= self.shape[0]:
            ix = ix % self.shape[0]
        if ix < 0:
            ix = ix + self.shape[0]
        self.coordinates[0] = ix
        self.coordinates_change_signal()
    def _get_ix(self):
        return self.coordinates[0]
    def _set_iy(self, iy):
        if abs(iy) >= self.shape[1]:
            iy = iy % self.shape[1]
        if iy < 0:
            iy = iy + self.shape[1]
        self.coordinates[1] = iy
        self.coordinates_change_signal()
    def _get_iy(self):
        return self.coordinates[1]
    ix = property(_get_ix, _set_ix)
    iy = property(_get_iy, _set_iy)
        
    np.array((0,0))
    
    def coordinates_change_signal(self):
        for f in self.on_coordinates_change:
            f()
    def _eval_functions(self, coord, function_list, mask):
            if mask[coord]:
                self.ix = coord[0]
                self.iy = coord[1]
                for function in function_list:
                    try:
                        function()
                    except:
                        self.disconnect(function)
                    
    def _eval_ifunctions(self, coord, function_list, mask):
        if mask[coord]:
            for function in function_list:
                function(coord[0], coord[1])

    def eval4all(self, function_list, mask = None, mode = 'lines'):
        if self.shape:
            if mask is None:
                 mask = np.ones(self.shape).astype('Bool')
                 
            if mode == 'lines':
                for y in np.arange(self.shape[1]) :
                    for x in np.arange(self.shape[0]) :
                        self._eval_functions((x,y), function_list, mask)
            elif mode == 'zigzag':
                inverter = 1
                for y in range(self.shape[1]) :
                    if inverter == 1 :
                        for x in range(self.shape[0]):
                            self._eval_functions((x,y), function_list, mask)
                    if inverter == -1 :
                        for x in reversed(range(self.shape[0])):
                            self._eval_functions((x,y), function_list, mask)
                    inverter*=-1
                    
    def reset(self):
#        self.on_coordinates_change = []
        self.ix = 0
        self.iy = 0
        
    def connect(self, function):
        if function not in self.on_coordinates_change:
            self.on_coordinates_change.append(function)
            
    def disconnect(self, function):
        if function in self.on_coordinates_change:
            self.on_coordinates_change.remove(function)
        
    def key_navigator(self, event):
        if event.key == "up" or event.key == "8":
            self.iy -= self.step
        elif event.key == "down" or event.key == "2":
            self.iy += self.step
        elif event.key == "right" or event.key == "6":
            self.ix += self.step
        elif event.key == "left" or event.key == "4":
            self.ix -= self.step
        elif event.key == "pageup":
            self.step += 1
            print "Step = ", self.step
        elif event.key == "pagedown":
            if self.step > 1:
                self.step -= 1
            print "Step = ", self.step

class TwoCoordinates():
    def __init__(self, shape):
        self.shape = shape
        self.coordinates1 = Coordinates(self.shape)
        self.coordinates2 = Coordinates(self.shape)
        self.pointers = None

    
