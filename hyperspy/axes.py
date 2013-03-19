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

import copy

import numpy as np
import traits.api as t
import traitsui.api as tui
from traits.trait_errors import TraitError

from hyperspy import messages

def get_axis_group(n , label = ''):
    group = tui.Group(
            tui.Group(
                tui.Item('axis%i.name' % n),
                tui.Item('axis%i.size' % n, style = 'readonly'),
                tui.Item('axis%i.index_in_array' % n, style = 'readonly'),
                tui.Item('axis%i.low_index' % n, style = 'readonly'),
                tui.Item('axis%i.high_index' % n, style = 'readonly'),
                # The style of the index is chosen to be readonly because of 
                # a bug in Traits 4.0.0 when using context with a Range traits
                # where the limits are defined by another traits_view
                tui.Item('axis%i.index' % n, style = 'readonly'),
                tui.Item('axis%i.value' % n, style = 'readonly'),
                tui.Item('axis%i.units' % n),
                tui.Item('axis%i.navigate' % n, label = 'slice'),
            show_border = True,),
            tui.Group(
                tui.Item('axis%i.scale' % n),
                tui.Item('axis%i.offset' % n),
            label = 'Calibration',
            show_border = True,),
        label = label,
        show_border = True,)
    return group
    
def generate_axis(offset, scale, size, offset_index=0):
    """Creates an axis given the offset, scale and number of channels

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
    
    """
    return np.linspace(offset - offset_index * scale,
                       offset + scale * (size - 1 - offset_index),
                       size)

class DataAxis(t.HasTraits):
    name = t.Str()
    units = t.Str()
    scale = t.Float()
    offset = t.Float()
    size = t.CInt()
    index_in_array = t.Int()
    low_value = t.Float()
    high_value = t.Float()
    value = t.Range('low_value', 'high_value')
    low_index = t.Int(0)
    high_index = t.Int()
    slice = t.Instance(slice)
    navigate = t.Bool(True)
    index = t.Range('low_index', 'high_index')
    axis = t.Array()
    continuous_value = t.Bool(False)

    def __init__(self,
                 size,
                 index_in_array,
                 name='',
                 scale=1.,
                 offset=0.,
                 units='undefined',
                 navigate=True):
                     
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
        self.update_axis()
        self.navigate = navigate
        self.on_trait_change(self.update_axis,
                             ['scale', 'offset', 'size'])
        self.on_trait_change(self.update_value, 'index')
        self.on_trait_change(self.set_index_from_value, 'value')
        self.on_trait_change(self._update_slice, 'navigate')
        self.on_trait_change(self.update_index_bounds, 'size')
        # The slice must be updated even if the default value did not 
        # change to correctly set its value.
        self._update_slice(self.navigate)
        
    def _get_positive_index(self, index):
        if index < 0:
            index = self.size + index
            if index < 0:
                raise IndexError("index out of bounds")
        return index
        
    def _slice_me(self, slice_):
        """Returns a slice to slice the corresponding data axis and 
        change the offset and scale of the DataAxis acordingly.
        
        Parameters
        ----------
        slice_ : {float, int, slice}
        
        Returns
        -------
        my_slice : slice
        
        """
        i2v = self.index2value
        v2i = self.value2index

        if isinstance(slice_, slice):
            start = slice_.start
            stop = slice_.stop
            step = slice_.step
        else:
            start = self._get_positive_index(slice_)
            if isinstance(start, float):
                stop = i2v(start) + 1
            else:
                stop = start + 1
            step = None
            
        if isinstance(step, float):
            step = int(round(step / self.scale))
        if isinstance(start, float):
            start = v2i(start)
        if isinstance(stop, float):
            stop = v2i(stop) 
            
        if step == 0:
            raise ValueError("slice step cannot be zero")
            
        my_slice = slice(start, stop, step)
        
        if start is None:
            if step > 0 or step is None:
                start = 0
            else:
                start = self.size - 1
        self.offset = i2v(start)
        if step is not None:
            self.scale *= step
            
        return my_slice


    def __repr__(self):
        if self.name is not None:
            text = '<%s axis, index: %s>' % (self.name,
                                               self.index_in_array)
            return text
            
    def connect(self, f, trait='value'):
        self.on_trait_change(f, trait)

    def disconnect(self, f, trait='value'):
        self.on_trait_change(f, trait, remove=True)

    def update_index_bounds(self):
        self.high_index = self.size - 1

    def update_axis(self):
        self.axis = generate_axis(self.offset, self.scale, self.size)
        if len(self.axis) != 0:
            self.low_value, self.high_value = (
                self.axis.min(), self.axis.max())

    def _update_slice(self, value):
        if value is False:
            self.slice = slice(None)
        else:
            self.slice = None

    def get_axis_dictionary(self):
        adict = {
            'name' : self.name,
            'scale' : self.scale,
            'offset' : self.offset,
            'size' : self.size,
            'units' : self.units,
            'index_in_array' : self.index_in_array,
            'navigate' : self.navigate
        }
        return adict
        
    def copy(self):
        return DataAxis(**self.get_axis_dictionary())

    def update_value(self):
        self.value = self.axis[self.index]

    def value2index(self, value):
        """Return the closest index to the given value if between the limits,
        otherwise it will return either the upper or lower limits

        Parameters
        ----------
        value : float

        Returns
        -------
        int
        """
        if value is None:
            return None
        else:
            index = int(round((value - self.offset) / \
            self.scale))
            if self.size > index >= 0:
                return index
            elif index < 0:
                return 0
            else:
                return int(self.size - 1)

    def index2value(self, index):
        return self.axis[index]

    def set_index_from_value(self, value):
        self.index = self.value2index(value)
        # If the value is above the limits we must correct the value
        if self.continuous_value is False:
            self.value = self.index2value(self.index)

    def calibrate(self, value_tuple, index_tuple, modify_calibration = True):
        scale = (value_tuple[1] - value_tuple[0]) /\
        (index_tuple[1] - index_tuple[0])
        offset = value_tuple[0] - scale * index_tuple[0]
        if modify_calibration is True:
            self.offset = offset
            self.scale = scale
        else:
            return offset, scale

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
                tui.Item(name = 'navigate', label = 'navigate'),
            show_border = True,),
            tui.Group(
                tui.Item(name = 'scale'),
                tui.Item(name = 'offset'),
            label = 'Calibration',
            show_border = True,),
        label = "Data Axis properties",
        show_border = True,),
    title = 'Axis configuration',
    )

class AxesManager(t.HasTraits):
    """Contains and manages the data axes.
    
    It can iterate over the navigation coordiantes returning the 
    indices at the current iteration.
    
    
    Attributes
    ----------
    
    coordinates : tuple
        Get and set the current coordinates if the navigation dimension
        is not 0. If the navigation dimension is 0 it raises 
        AttributeError when attempting to set its value.

    
    indices : tuple
        Get and set the current indices if the navigation dimension
        is not 0. If the navigation dimension is 0 it raises 
        AttributeError when attempting to set its value.

    Examples
    --------
    
    >>> import numpy as np
    
    Create a spectrum with random data
    
    >>> s = signals.Spectrum({'data' : np.random.random((2,2,2,10))})
    >>> s.axes_manager
    <Axes manager, 4 axes, signal dimension: 1, navigation dimension: 3>
    
    >>> s.axes_manager[1]
    <undefined axis, index: 1>
    >>> for i in s.axes_manager:
    >>>     print i, s.axes_manager.indices
    (0, 0, 0) (0, 0, 0)
    (0, 0, 1) (0, 0, 1)
    (0, 1, 0) (0, 1, 0)
    (0, 1, 1) (0, 1, 1)
    (1, 0, 0) (1, 0, 0)
    (1, 0, 1) (1, 0, 1)
    (1, 1, 0) (1, 1, 0)
    (1, 1, 1) (1, 1, 1)
    
    """
    axes = t.List(DataAxis)
    signal_axes = t.List()
    navigation_axes = t.List()
    _step = t.Int(1)
    
    def _get_positive_index(self, axis):
        if axis < 0:
            axis = len(self.axes) + axis
            if axis < 0:
                raise IndexError("index out of bounds")
        return axis
    
    def __getitem__(self, y):
        """x.__getitem__(y) <==> x[y]
        
        """
        return self.axes[y]
        
    def __getslice__(self, i=None, j=None):
        """x.__getslice__(i, j) <==> x[i:j]
        
        """
        return self.axes[i:j]
        
    def remove(self, axis):
        """Remove the given Axis.
        
        Raises
        ------
        ValueError if the Axis is not present.
        
        """
        if axis not in self.axes:
            raise ValueError(
                "AxesManager.remove(x): x not in AxesManager")
        index = self.axes.index(axis)
        self.axes.remove(axis)
        for axis in self.axes[index:]:
            axis.index_in_array -= 1
            
    def __delitem__(self, i):
        self.remove(self[i])
        
    def _get_data_slice(self, fill=None):
        """Return a tuple of slice objects to slice the data.
        
        Parameters
        ----------
        fill: None or iterable of (int, slice)
            If not None, fill the tuple of index int with the given
            slice.
            
        """
        cslice = [slice,] * len(self.axes)
        if fill is not None:
            for index, slice_ in fill:
                cslice[index] = slice_
        return tuple(cslice)
        
    def __init__(self, axes_list):
        super(AxesManager, self).__init__()
        ncoord = len(axes_list)
        self.axes = [None] * ncoord
        for axis_dict in axes_list:
            self.axes[axis_dict['index_in_array']] = DataAxis(
                                                            **axis_dict)
        navigates = [i.navigate for i in self.axes if 
                                                hasattr(i, 'navigate')]
        # set_view is called only if there is no current view
        if not navigates or np.all(np.array(navigates) == True):
            self.set_view()
        self.update_attributes()
        self.on_trait_change(self.update_attributes, 'axes.slice')
        self.on_trait_change(self.update_attributes, 'axes.index')
        self.on_trait_change(self.update_attributes, 'axes.size')
        self._index = None # index for the iterator
        
    def _update_max_index(self):
        self._max_index = 1
        for i in self.navigation_shape:
            self._max_index *= i
        if self._max_index != 0:
            self._max_index -= 1

    def next(self):
        """
        Standard iterator method, updates the index and returns the 
        current coordiantes

        Returns
        -------
        val : tuple of ints
            Returns a tuple containing the coordiantes of the current 
            iteration.

        """
        if self._index is None:
            self._index = 0
            self._indices_backup = self.indices
            val = (0,) * self.navigation_dimension
            self.indices = val
        elif (self._index >= self._max_index):
            self._index = None
            self.indices = self._indices_backup
            del self._indices_backup
            raise StopIteration
        else:
            self._index += 1
            val = np.unravel_index(self._index, 
                                    tuple(self.navigation_shape))
            self.indices = val
        return val

    def __iter__(self):
        # Reset the _index that can have a value != None due to 
        # a previous iteration that did not hit a StopIteration
        self._index = None
        return self
        
    def append_axis(self, *args, **kwargs):
        axis = DataAxis(*args, **kwargs)
        self.axes.append(axis)

    def update_attributes(self):
        getitem_tuple = []
        values = []
        self.signal_axes = []
        self.navigation_axes = []
        for axis in self.axes:
            if axis.slice is None:
                getitem_tuple.append(axis.index)
                values.append(axis.value)
                self.navigation_axes.append(axis)
            else:
                getitem_tuple.append(axis.slice)
                self.signal_axes.append(axis)
                
        self._getitem_tuple = getitem_tuple
        self.signal_dimension = len(self.signal_axes)
        self.navigation_dimension = len(self.navigation_axes)
        if self.navigation_dimension != 0:
            self.navigation_shape = [
                axis.size for axis in self.navigation_axes]
        else:
            self.navigation_shape = [0,]
            
        if self.signal_dimension != 0:
            self.signal_shape = [
                axis.size for axis in self.signal_axes]
        else:
            self.signal_shape = [0,]
        self.navigation_size = \
            np.cumprod(self.navigation_shape)[-1]
        self.signal_size = \
            np.cumprod(self.signal_shape)[-1]
        self._update_max_index()

    def set_view(self, view = 'spectrum'):
        """Adjust the navigate attribute depending on the desired view-
        
        Attributes
        ----------
        view : {'spectrum', 'image'}
            If spectrum all but the last index will be set to "navigate". If 
            'image' the all but the last two indices will be set to navigate.            
        
        """
        tl = [True] * len(self.axes)
        if view == 'spectrum':
            # We limit the signal_dimension to 1 to get a spectrum
            tl[-1] = False
        elif view == 'image':
            tl[-2:] = False, False

        for axis in self.axes:
            axis.navigate = tl.pop(0)
            
    def set_signal_dimension(self, value):
        """Set the dimension of the signal.
                
        Attributes
        ----------
        value : int
        
        Raises
        ------
        ValueError if value if greater than the number of axes or 
        is negative         
        
        """
        if len(self.axes) == 0:
            return
        elif value > len(self.axes):
            raise ValueError("The signal dimension cannot be greater"
                " than the number of axes which is %i" % len(self.axes))
        elif value < 0:
            raise ValueError(
                "The signal dimension must be a positive integer")
                
        tl = [True] * len(self.axes)
        tl[-value:] = (False,) * value
            
        for axis in self.axes:
            axis.navigate = tl.pop(0)

    def setsignal_axes(self, signal_axes):
        '''Easily choose which axes are slicing

        Parameters
        ----------

        signal_axes: tuple of ints
            A list of the axis indices that we want to slice

        '''
        for axis in self.axes:
            if axis.index_in_array in signal_axes:
                axis.navigate = False
            else:
                axis.navigate = True

    def connect(self, f):
        for axis in self.axes:
            if axis.slice is None:
                axis.on_trait_change(f, 'index')

    def disconnect(self, f):
        for axis in self.axes:
            if axis.slice is None:
                axis.on_trait_change(f, 'index', remove=True)

    def key_navigator(self, event):
        if len(self.navigation_axes) not in (1,2): return
        x = self.navigation_axes[-1]
        try:
            if event.key == "right" or event.key == "6":
                x.index += self._step
            elif event.key == "left" or event.key == "4":
                x.index -= self._step
            elif event.key == "pageup":
                self._step += 1
            elif event.key == "pagedown":
                if self._step > 1:
                    self._step -= 1
            if len(self.navigation_axes) == 2:
                y = self.navigation_axes[-2]
                if event.key == "up" or event.key == "8":
                    y.index -= self._step
                elif event.key == "down" or event.key == "2":
                    y.index += self._step
        except TraitError:
            pass

    def gui(self):
        for axis in self.axes:
            axis.edit_traits()

    def copy(self):
        return(copy.copy(self))

    def deepcopy(self):
        return(copy.deepcopy(self))

    def __deepcopy__(self, *args):
        return AxesManager(self._get_axes_dicts())

    def _get_axes_dicts(self):
        axes_dicts = []
        for axis in self.axes:
            axes_dicts.append(axis.get_axis_dictionary())
        return axes_dicts
        
    def _get_signal_axes_dicts(self):
        axes_dicts = []
        i = 0
        for axis in self.signal_axes:
            axes_dicts.append(axis.get_axis_dictionary())
            axes_dicts[-1]['index_in_array'] = i
            i += 1
        return axes_dicts

    def _get_navigation_axes_dicts(self):
        axes_dicts = []
        i = 0
        for axis in self.navigation_axes:
            axes_dicts.append(axis.get_axis_dictionary())
            axes_dicts[-1]['index_in_array'] = i
            i += 1
        return axes_dicts
        
    def _set_axes_index_in_array_from_position(self):
        i = 0
        for axis in self.axes:
            axis.index_in_array = i
            i += 1
        
    def show(self):
        context = {}
        ag = []
        for n in range(0,len(self.axes)):
            ag.append(get_axis_group(n, self.axes[n].name))
            context['axis%i' % n] = self.axes[n]
        ag = tuple(ag)
        self.edit_traits(view = tui.View(*ag), context = context)
        
    def __repr__(self):
        text = '<Axes manager, %i axes, signal dimension: %i, navigation dimension: %i>' % (
            len(self.axes), self.signal_dimension,
            self.navigation_dimension)
        return text
    
    @property        
    def coordinates(self):
        """Get the coordinates of the navigation axes.
        
        Returns
        -------
        list
            
        """
        return tuple([axis.value for axis in self.navigation_axes])
        
    @coordinates.setter    
    def coordinates(self, coordinates):
        """Set the coordinates of the navigation axes.
        
        Parameters
        ----------
        coordinates : tuple
            The len of the the tuple must coincide with the navigation
            dimension
            
        """
        
        if len(coordinates) != self.navigation_dimension:
            raise AttributeError(
            "The number of coordinates must be equal to the "
            "navigation dimension that is %i" % 
                self.navigation_dimension)
        for value, axis in zip(coordinates, self.navigation_axes):
            axis.value = value
            
    @property        
    def indices(self):
        """Get the index of the navigation axes.
        
        Returns
        -------
        list
            
        """
        return tuple([axis.index for axis in self.navigation_axes])
        
    @indices.setter    
    def indices(self, indices):
        """Set the index of the navigation axes.
        
        Parameters
        ----------
        indices : tuple
            The len of the the tuple must coincide with the navigation
            dimension
            
        """
        
        if len(indices) != self.navigation_dimension:
            raise AttributeError(
            "The number of indices must be equal to the "
            "navigation dimension that is %i" % 
                self.navigation_dimension)
        for index, axis in zip(indices, self.navigation_axes):
            axis.index = index
