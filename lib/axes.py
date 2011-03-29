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
import enthought.traits.api as t
import enthought.traits.ui.api as tui 

import messages

    
class BoundedIndex(t.Int):
    def validate(self, object, name, value):
        value = super(BoundedIndex, self).validate(object, name, value)
        if abs(value) >= object.size:
            value = value % object.size
        return value
        

def generate_axis(offset, scale, size, offset_index=0):
    '''Creates an axis given the offset, scale and number of channels
    
    Alternatively, the offset_index of the offset channel can be specified.
    
    Parameters
    ----------
    offset : float
    scale : float
    size : number of channels
    offset_index : int
        offset_index number of the offset
    
    Returns
    -------
    Numpy array
    '''
    return np.linspace(offset-offset_index*scale, 
                       offset+scale*(size-1-offset_index),
                       size)  

class DataAxis(t.HasTraits):
    name = t.Str()
    units = t.Str()
    scale = t.Float()
    offset = t.Float()
    size = t.Int()
    index_in_array = t.Int()
    low_value = t.Float()
    high_value = t.Float()
    value = t.Range('low_value', 'high_value')
    low_index = t.Int(0)
    high_index = t.Int()
    slice = t.Instance(slice)
    
    index = t.Range('low_index', 'high_index')
    axis = t.Array()
    
    def __init__(self, name, scale, offset, size, units, index_in_array):
        super(DataAxis, self).__init__()
        
        self.name = name
        self.units = units
        self.scale = scale
        self.offset = offset
        self.size = size
        self.high_index = self.size - 1
        self.low_index = 0
        self.index = 0
        self.index_in_array = index_in_array
        self.slice = None
        self.update_axis()
                
        self.on_trait_change(self.update_axis, ['scale', 'offset', 'size'])
        self.on_trait_change(self.update_value, 'index')
        self.on_trait_change(self.set_index_from_value, 'value')
        
    def __repr__(self):
        if self.name is not None:
            return self.name + ' index: ' + str(self.index_in_array)
          
    def update_index_bounds(self):
        self.high_index = self.size - 1
    
    def update_axis(self):
        self.axis = generate_axis(self.offset, self.scale, self.size)
        self.low_value, self.high_value = self.axis.min(), self.axis.max()
#        self.update_value()

    def get_axis_dictionary(self):
        adict = {
            'name' : self.name,
            'scale' : self.scale,
            'offset' : self.offset,
            'size' : self.size,
            'units' : self.units,
            'index_in_array' : self.index_in_array,
        }
        return adict
        
    def update_value(self):
        self.value = self.axis[self.index]
        
    def value2index(self, value):
        '''
        Return the closest index to the given value if between the limits,
        otherwise it will return either the upper or lower limits
        
        Parameters
        ----------
        value : float
        
        Returns
        -------
        int
        '''
        if value is None:
            return None
        else:
            index = int(round((value - self.offset) / \
            self.scale))
            if self.size > index >= 0: 
                return index
            elif index < 0:
                messages.warning("The given value is below the axis limits")
                return 0
            else:
                messages.warning("The given value is above the axis limits")
                return int(self.size - 1)
        
    def index2value(self, index):
        return self.axis[index]
    
    def set_index_from_value(self, value):
        self.index = self.value2index(value)
        # If the value is above the limits we must correct the value
        self.value = self.index2value(self.index)
        
    def calibrate(self, value_tuple, index_tuple):
        scale = (value_tuple[1] - value_tuple[0]) /\
        (index_tuple[1] - index_tuple[0])
        offset = value_tuple[0] - scale * index_tuple[0]
        print "Scale = ", scale
        print "Offset = ", offset
        self.offset = offset
        self.scale = scale
        
    traits_view = \
    tui.View(
        tui.Group(
            tui.Group(
                tui.Item(name = 'name'),
                tui.Item(name = 'size', style = 'readonly'),
                tui.Item(name = 'index_in_array', style = 'readonly'),
                tui.Item(name = 'index'),
                tui.Item(name = 'value', style = 'readonly'),
                tui.Item(name = 'units'),
                tui.Item(name = 'slice'),
            show_border = True,),
            tui.Group(
                tui.Item(name = 'scale'),
                tui.Item(name = 'offset'),
            label = 'Calibration',
            show_border = True,),
        label = "Data Axis properties",
        show_border = True,),
    )
    
class AxesManager(t.HasTraits):
    axes = t.List(DataAxis)
    _slicing_axes = t.List()
    _non_slicing_axes = t.List()
    _step = t.Int(1)
    def __init__(self, axes_list):
        super(AxesManager, self).__init__()
        ncoord = len(axes_list)
        self.axes = [None] * ncoord
        for axis_dict in axes_list:
            self.axes[axis_dict['index_in_array']] = DataAxis(**axis_dict)
#        self.set_coordinate_attribute()
        self.set_view()
        self.set_output_dim()
#        self.on_trait_change(self.set_axis_attribute, 'axes.name')
        # TODO: MCS 22/03/11 - on the following two lines, axis might 
        # need to be axes
        self.on_trait_change(self.set_output_dim, 'axes.slice')
        self.on_trait_change(self.set_output_dim, 'axes.index')

    def set_output_dim(self):
        getitem_tuple = []
        indexes = []
        self._slicing_axes = []
        self._non_slicing_axes = []
        i = 0
        for axis in self.axes:
            if axis.slice is None:
                getitem_tuple.append(axis.index)
                indexes.append(axis.index)
                self._non_slicing_axes.append(axis)
            else:
                getitem_tuple.append(axis.slice)
                self._slicing_axes.append(axis)
                i += 1
        self._getitem_tuple = getitem_tuple
        self._indexes = np.array(indexes)
        self.output_dim = i
        
    def set_view(self, view = 'hyperspectrum'):
        '''
        view : 'hyperspectrum' or 'image'
        '''
        if view == 'hyperspectrum':
            # We limit the output_dim to 1 to get a spectrum
            i = 0
            for axis in self.axes[::-1]:
                if i < 1:
                    axis.slice = slice(None)
                    i += 1
                else:
                    axis.slice = None
        elif view == 'image':
            # We limit the output_dim to 2 to get a spectrum
            i = 0
            for axis in self.axes:
                if i < 2:
                    axis.slice = slice(None)
                    i += 1
                else:
                    axis.slice = None
                    
    def connect(self, f):
        for axis in self.axes:
            if axis.slice is None:
                axis.on_trait_change(f, 'index')
                
    def disconnect(self, f):
        for axis in self.axes:
            if axis.slice is None:
                axis.on_trait_change(f, 'index', remove = True)
                
    def key_navigator(self, event):
        if len(self._non_slicing_axes) not in (1,2): return
        x = self._non_slicing_axes[-1]

        if event.key == "right" or event.key == "6":
            x.index += self._step
        elif event.key == "left" or event.key == "4":
            x.index -= self._step
        elif event.key == "pageup":
            self._step += 1
        elif event.key == "pagedown":
            if self._step > 1:
                self._step -= 1
        if len(self._non_slicing_axes) == 2:
            y = self._non_slicing_axes[-2]
            if event.key == "up" or event.key == "8":
                y.index -= self._step
            elif event.key == "down" or event.key == "2":
                y.index += self._step
                
    def edit_axes_traits(self):
        for axis in self.axes:
            axis.edit_traits()
            
    traits_view = tui.View(tui.Item('axes', style = 'custom'))
    
        

    
