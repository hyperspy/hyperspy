# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import copy
import math

import numpy as np
import traits.api as t
from traits.trait_errors import TraitError

from hyperspy.events import Events, Event
from hyperspy.misc.utils import isiterable, ordinal
from hyperspy.misc.math_tools import isfloat

import warnings


class ndindex_nat(np.ndindex):

    def __next__(self):
        return super(ndindex_nat, self).next()[::-1]


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
    low_value = t.Float()
    high_value = t.Float()
    value = t.Range('low_value', 'high_value')
    low_index = t.Int(0)
    high_index = t.Int()
    slice = t.Instance(slice)
    navigate = t.Bool(t.Undefined)
    index = t.Range('low_index', 'high_index')
    axis = t.Array()
    continuous_value = t.Bool(False)

    def __init__(self,
                 size,
                 index_in_array=None,
                 name=t.Undefined,
                 scale=1.,
                 offset=0.,
                 units=t.Undefined,
                 navigate=t.Undefined):
        super(DataAxis, self).__init__()
        self.events = Events()
        self.events.index_changed = Event("""
            Event that triggers when the index of the `DataAxis` changes

            Triggers after the internal state of the `DataAxis` has been
            updated.

            Arguments:
            ---------
            obj : The DataAxis that the event belongs to.
            index : The new index
            """, arguments=["obj", 'index'])
        self.events.value_changed = Event("""
            Event that triggers when the value of the `DataAxis` changes

            Triggers after the internal state of the `DataAxis` has been
            updated.

            Arguments:
            ---------
            obj : The DataAxis that the event belongs to.
            value : The new value
            """, arguments=["obj", 'value'])
        self._suppress_value_changed_trigger = False
        self._suppress_update_value = False
        self.name = name
        self.units = units
        self.scale = scale
        self.offset = offset
        self.size = size
        self.high_index = self.size - 1
        self.low_index = 0
        self.index = 0
        self.update_axis()
        self.navigate = navigate
        self.axes_manager = None
        self.on_trait_change(self.update_axis,
                             ['scale', 'offset', 'size'])
        self.on_trait_change(self._update_slice, 'navigate')
        self.on_trait_change(self.update_index_bounds, 'size')
        # The slice must be updated even if the default value did not
        # change to correctly set its value.
        self._update_slice(self.navigate)

    def _index_changed(self, name, old, new):
        self.events.index_changed.trigger(obj=self, index=self.index)
        if not self._suppress_update_value:
            new_value = self.axis[self.index]
            if new_value != self.value:
                self.value = new_value

    def _value_changed(self, name, old, new):
        old_index = self.index
        new_index = self.value2index(new)
        if self.continuous_value is False:  # Only values in the grid alowed
            if old_index != new_index:
                self.index = new_index
                if new == self.axis[self.index]:
                    self.events.value_changed.trigger(obj=self, value=new)
            elif old_index == new_index:
                new_value = self.index2value(new_index)
                if new_value == old:
                    self._suppress_value_changed_trigger = True
                    try:
                        self.value = new_value
                    finally:
                        self._suppress_value_changed_trigger = False

                elif new_value == new and not\
                        self._suppress_value_changed_trigger:
                    self.events.value_changed.trigger(obj=self, value=new)
        else:  # Intergrid values are alowed. This feature is deprecated
            self.events.value_changed.trigger(obj=self, value=new)
            if old_index != new_index:
                self._suppress_update_value = True
                self.index = new_index
                self._suppress_update_value = False

    @property
    def index_in_array(self):
        if self.axes_manager is not None:
            return self.axes_manager._axes.index(self)
        else:
            raise AttributeError(
                "This DataAxis does not belong to an AxesManager"
                " and therefore its index_in_array attribute "
                " is not defined")

    @property
    def index_in_axes_manager(self):
        if self.axes_manager is not None:
            return self.axes_manager._get_axes_in_natural_order().\
                index(self)
        else:
            raise AttributeError(
                "This DataAxis does not belong to an AxesManager"
                " and therefore its index_in_array attribute "
                " is not defined")

    def _get_positive_index(self, index):
        if index < 0:
            index = self.size + index
            if index < 0:
                raise IndexError("index out of bounds")
        return index

    def _get_index(self, value):
        if isfloat(value):
            return self.value2index(value)
        else:
            return value

    def _get_array_slices(self, slice_):
        """Returns a slice to slice the corresponding data axis without
        changing the offset and scale of the DataAxis.

        Parameters
        ----------
        slice_ : {float, int, slice}

        Returns
        -------
        my_slice : slice

        """
        v2i = self.value2index

        if isinstance(slice_, slice):
            start = slice_.start
            stop = slice_.stop
            step = slice_.step
        else:
            if isfloat(slice_):
                start = v2i(slice_)
            else:
                start = self._get_positive_index(slice_)
            stop = start + 1
            step = None

        if isfloat(step):
            step = int(round(step / self.scale))
        if isfloat(start):
            try:
                start = v2i(start)
            except ValueError:
                if start > self.high_value:
                    # The start value is above the axis limit
                    raise IndexError(
                        "Start value above axis high bound for  axis %s."
                        "value: %f high_bound: %f" % (repr(self), start,
                                                      self.high_value))
                else:
                    # The start value is below the axis limit,
                    # we slice from the start.
                    start = None
        if isfloat(stop):
            try:
                stop = v2i(stop)
            except ValueError:
                if stop < self.low_value:
                    # The stop value is below the axis limits
                    raise IndexError(
                        "Stop value below axis low bound for  axis %s."
                        "value: %f low_bound: %f" % (repr(self), stop,
                                                     self.low_value))
                else:
                    # The stop value is below the axis limit,
                    # we slice until the end.
                    stop = None

        if step == 0:
            raise ValueError("slice step cannot be zero")

        return slice(start, stop, step)

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

        my_slice = self._get_array_slices(slice_)

        start, stop, step = my_slice.start, my_slice.stop, my_slice.step

        if start is None:
            if step is None or step > 0:
                start = 0
            else:
                start = self.size - 1
        self.offset = i2v(start)
        if step is not None:
            self.scale *= step

        return my_slice

    def _get_name(self):
        if self.name is t.Undefined:
            if self.axes_manager is None:
                name = "Unnamed"
            else:
                name = "Unnamed " + ordinal(self.index_in_axes_manager)
        else:
            name = self.name
        return name

    def __repr__(self):
        text = '<%s axis, size: %i' % (self._get_name(),
                                       self.size,)
        if self.navigate is True:
            text += ", index: %i" % self.index
        text += ">"
        return text

    def __str__(self):
        return self._get_name() + " axis"

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
            'name': self.name,
            'scale': self.scale,
            'offset': self.offset,
            'size': self.size,
            'units': self.units,
            'navigate': self.navigate
        }
        return adict

    def copy(self):
        return DataAxis(**self.get_axis_dictionary())

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        cp = self.copy()
        return cp

    def value2index(self, value, rounding=round):
        """Return the closest index to the given value if between the limit.

        Parameters
        ----------
        value : number or numpy array

        Returns
        -------
        index : integer or numpy array

        Raises
        ------
        ValueError if any value is out of the axis limits.

        """
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            if rounding is round:
                rounding = np.round
            elif rounding is math.ceil:
                rounding = np.ceil
            elif rounding is math.floor:
                rounding = np.floor

        index = rounding((value - self.offset) / self.scale)

        if isinstance(value, np.ndarray):
            index = index.astype(int)
            if np.all(self.size > index) and np.all(index >= 0):
                return index
            else:
                raise ValueError("A value is out of the axis limits")
        else:
            index = int(index)
            if self.size > index >= 0:
                return index
            else:
                raise ValueError("The value is out of the axis limits")

    def index2value(self, index):
        if isinstance(index, np.ndarray):
            return self.axis[index.ravel()].reshape(index.shape)
        else:
            return self.axis[index]

    def calibrate(self, value_tuple, index_tuple, modify_calibration=True):
        scale = (value_tuple[1] - value_tuple[0]) /\
            (index_tuple[1] - index_tuple[0])
        offset = value_tuple[0] - scale * index_tuple[0]
        if modify_calibration is True:
            self.offset = offset
            self.scale = scale
        else:
            return offset, scale

    def value_range_to_indices(self, v1, v2):
        """Convert the given range to index range.

        When an out of the axis limits, the endpoint is used instead.

        Parameters
        ----------
        v1, v2 : float
            The end points of the interval in the axis units. v2 must be
            greater than v1.

        """
        if v1 is not None and v2 is not None and v1 > v2:
            raise ValueError("v2 must be greater than v1.")

        if v1 is not None and self.low_value < v1 <= self.high_value:
            i1 = self.value2index(v1)
        else:
            i1 = 0
        if v2 is not None and self.high_value > v2 >= self.low_value:
            i2 = self.value2index(v2)
        else:
            i2 = self.size - 1
        return i1, i2

    def update_from(self, axis, attributes=["scale", "offset", "units"]):
        """Copy values of specified axes fields from the passed AxesManager.

        Parameters
        ----------
        axis : DataAxis
            The DataAxis instance to use as a source for values.
        attributes : iterable container of strings.
            The name of the attribute to update. If the attribute does not
            exist in either of the AxesManagers, an AttributeError will be
            raised.
        Returns
        -------
        A boolean indicating whether any changes were made.

        """
        any_changes = False
        changed = {}
        for f in attributes:
            if getattr(self, f) != getattr(axis, f):
                changed[f] = getattr(axis, f)
        if len(changed) > 0:
            self.trait_set(**changed)
            any_changes = True
        return any_changes


class AxesManager(t.HasTraits):

    """Contains and manages the data axes.

    It supports indexing, slicing, subscriptins and iteration. As an iterator,
    iterate over the navigation coordinates returning the current indices.
    It can only be indexed and sliced to access the DataAxis objects that it
    contains. Standard indexing and slicing follows the "natural order" as in
    Signal, i.e. [nX, nY, ...,sX, sY,...] where `n` indicates a navigation axis
    and `s` a signal axis. In addition AxesManager support indexing using
    complex numbers a + bj, where b can be one of 0, 1, 2 and 3 and a a valid
    index. If b is 3 AxesManager is indexed using the order of the axes in the
    array. If b is 1(2), indexes only the navigation(signal) axes in the
    natural order. In addition AxesManager supports subscription using
    axis name.

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

    signal_axes, navigation_axes : list
        Contain the corresponding DataAxis objects

    Examples
    --------

    Create a spectrum with random data

    >>> s = hs.signals.Signal1D(np.random.random((2,3,4,5)))
    >>> s.axes_manager
    <Axes manager, axes: (4, 3, 2|5)>
                Name |   size |  index |  offset |   scale |  units
    ================ | ====== | ====== | ======= | ======= | ======
         <undefined> |      4 |      0 |       0 |       1 | <undefined>
         <undefined> |      3 |      0 |       0 |       1 | <undefined>
         <undefined> |      2 |      0 |       0 |       1 | <undefined>
    ---------------- | ------ | ------ | ------- | ------- | ------
         <undefined> |      5 |        |       0 |       1 | <undefined>
    >>> s.axes_manager[0]
    <Unnamed 0th axis, size: 4, index: 0>
    >>> s.axes_manager[3j]
    <Unnamed 2nd axis, size: 2, index: 0>
    >>> s.axes_manager[1j]
    <Unnamed 0th axis, size: 4, index: 0>
    >>> s.axes_manager[2j]
    <Unnamed 3rd axis, size: 5>
    >>> s.axes_manager[1].name = "y"
    >>> s.axes_manager["y"]
    <y axis, size: 3, index: 0>
    >>> for i in s.axes_manager:
    ...     print(i, s.axes_manager.indices)
    ...
    (0, 0, 0) (0, 0, 0)
    (1, 0, 0) (1, 0, 0)
    (2, 0, 0) (2, 0, 0)
    (3, 0, 0) (3, 0, 0)
    (0, 1, 0) (0, 1, 0)
    (1, 1, 0) (1, 1, 0)
    (2, 1, 0) (2, 1, 0)
    (3, 1, 0) (3, 1, 0)
    (0, 2, 0) (0, 2, 0)
    (1, 2, 0) (1, 2, 0)
    (2, 2, 0) (2, 2, 0)
    (3, 2, 0) (3, 2, 0)
    (0, 0, 1) (0, 0, 1)
    (1, 0, 1) (1, 0, 1)
    (2, 0, 1) (2, 0, 1)
    (3, 0, 1) (3, 0, 1)
    (0, 1, 1) (0, 1, 1)
    (1, 1, 1) (1, 1, 1)
    (2, 1, 1) (2, 1, 1)
    (3, 1, 1) (3, 1, 1)
    (0, 2, 1) (0, 2, 1)
    (1, 2, 1) (1, 2, 1)
    (2, 2, 1) (2, 2, 1)
    (3, 2, 1) (3, 2, 1)

    """

    _axes = t.List(DataAxis)
    signal_axes = t.Tuple()
    navigation_axes = t.Tuple()
    _step = t.Int(1)

    def __init__(self, axes_list):
        super(AxesManager, self).__init__()
        self.events = Events()
        self.events.indices_changed = Event("""
            Event that triggers when the indices of the `AxesManager` changes

            Triggers after the internal state of the `AxesManager` has been
            updated.

            Arguments:
            ----------
            obj : The AxesManager that the event belongs to.
            """, arguments=['obj'])
        self.events.any_axis_changed = Event("""
            Event that trigger when the space defined by the axes transforms.

            Specifically, it triggers when one or more of the folloing
            attributes changes on one or more of the axes:
                `offset`, `size`, `scale`

            Arguments:
            ----------
            obj : The AxesManager that the event belongs to.
            """, arguments=['obj'])
        self.create_axes(axes_list)
        # set_signal_dimension is called only if there is no current
        # view. It defaults to spectrum
        navigates = [i.navigate for i in self._axes]
        if t.Undefined in navigates:
            # Default to Signal1D view if the view is not fully defined
            self.set_signal_dimension(len(axes_list))

        self._update_attributes()
        self._update_trait_handlers()
        self._index = None  # index for the iterator

    def _update_trait_handlers(self, remove=False):
        things = {self._on_index_changed: '_axes.index',
                  self._on_slice_changed: '_axes.slice',
                  self._on_size_changed: '_axes.size',
                  self._on_scale_changed: '_axes.scale',
                  self._on_offset_changed: '_axes.offset'}

        for k, v in things.items():
            self.on_trait_change(k, name=v, remove=remove)

    def _get_positive_index(self, axis):
        if axis < 0:
            axis += len(self._axes)
            if axis < 0:
                raise IndexError("index out of bounds")
        return axis

    def _array_indices_generator(self):
        shape = (self.navigation_shape[::-1] if self.navigation_size > 0 else
                 [1, ])
        return np.ndindex(*shape)

    def _am_indices_generator(self):
        shape = (self.navigation_shape if self.navigation_size > 0 else
                 [1, ])[::-1]
        return ndindex_nat(*shape)

    def __getitem__(self, y):
        """x.__getitem__(y) <==> x[y]

        """
        if isinstance(y, str) or not np.iterable(y):
            return self[(y,)][0]
        axes = [self._axes_getter(ax) for ax in y]
        _, indices = np.unique(
            [_id for _id in map(id, axes)], return_index=True)
        ans = tuple(axes[i] for i in sorted(indices))
        return ans

    def _axes_getter(self, y):
        if y in self._axes:
            return y
        if isinstance(y, str):
            axes = list(self._get_axes_in_natural_order())
            while axes:
                axis = axes.pop()
                if y == axis.name:
                    return axis
            raise ValueError("There is no DataAxis named %s" % y)
        elif (isfloat(y.real) and not y.real.is_integer() or
                isfloat(y.imag) and not y.imag.is_integer()):
            raise TypeError("axesmanager indices must be integers, "
                            "complex intergers or strings")
        if y.imag == 0:  # Natural order
            return self._get_axes_in_natural_order()[y]
        elif y.imag == 3:  # Array order
            # Array order
            return self._axes[int(y.real)]
        elif y.imag == 1:  # Navigation natural order
            #
            return self.navigation_axes[int(y.real)]
        elif y.imag == 2:  # Signal natural order
            return self.signal_axes[int(y.real)]
        else:
            raise IndexError("axesmanager imaginary part of complex indices "
                             "must be 0, 1, 2 or 3")

    def __getslice__(self, i=None, j=None):
        """x.__getslice__(i, j) <==> x[i:j]

        """
        return self._get_axes_in_natural_order()[i:j]

    def _get_axes_in_natural_order(self):
        return self.navigation_axes + self.signal_axes

    @property
    def _navigation_shape_in_array(self):
        return self.navigation_shape[::-1]

    @property
    def _signal_shape_in_array(self):
        return self.signal_shape[::-1]

    @property
    def shape(self):
        nav_shape = (self.navigation_shape
                     if self.navigation_shape != (0,)
                     else tuple())
        sig_shape = (self.signal_shape
                     if self.signal_shape != (0,)
                     else tuple())
        return nav_shape + sig_shape

    def remove(self, axes):
        """Remove one or more axes
        """
        axes = self[axes]
        if not np.iterable(axes):
            axes = (axes,)
        for ax in axes:
            self._remove_one_axis(ax)

    def _remove_one_axis(self, axis):
        """Remove the given Axis.

        Raises
        ------
        ValueError if the Axis is not present.

        """
        axis = self._axes_getter(axis)
        axis.axes_manager = None
        self._axes.remove(axis)

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
        cslice = [slice(None), ] * len(self._axes)
        if fill is not None:
            for index, slice_ in fill:
                cslice[index] = slice_
        return tuple(cslice)

    def create_axes(self, axes_list):
        """Given a list of dictionaries defining the axes properties
        create the DataAxis instances and add them to the AxesManager.

        The index of the axis in the array and in the `_axes` lists
        can be defined by the index_in_array keyword if given
        for all axes. Otherwise it is defined by their index in the
        list.

        See also
        --------
        _append_axis

        """
        # Reorder axes_list using index_in_array if it is defined
        # for all axes and the indices are not repeated.
        indices = set([axis['index_in_array'] for axis in axes_list if
                       hasattr(axis, 'index_in_array')])
        if len(indices) == len(axes_list):
            axes_list.sort(key=lambda x: x['index_in_array'])
        for axis_dict in axes_list:
            self._append_axis(**axis_dict)

    def _update_max_index(self):
        self._max_index = 1
        for i in self.navigation_shape:
            self._max_index *= i
        if self._max_index != 0:
            self._max_index -= 1

    def __next__(self):
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
            val = (0,) * self.navigation_dimension
            self.indices = val
        elif self._index >= self._max_index:
            raise StopIteration
        else:
            self._index += 1
            val = np.unravel_index(
                self._index,
                tuple(self._navigation_shape_in_array)
            )[::-1]
            self.indices = val
        return val

    def __iter__(self):
        # Reset the _index that can have a value != None due to
        # a previous iteration that did not hit a StopIteration
        self._index = None
        return self

    def _append_axis(self, *args, **kwargs):
        axis = DataAxis(*args, **kwargs)
        axis.axes_manager = self
        self._axes.append(axis)

    def _on_index_changed(self):
        self._update_attributes()
        self.events.indices_changed.trigger(obj=self)

    def _on_slice_changed(self):
        self._update_attributes()

    def _on_size_changed(self):
        self._update_attributes()
        self.events.any_axis_changed.trigger(obj=self)

    def _on_scale_changed(self):
        self.events.any_axis_changed.trigger(obj=self)

    def _on_offset_changed(self):
        self.events.any_axis_changed.trigger(obj=self)

    def update_axes_attributes_from(self, axes,
                                    attributes=["scale", "offset", "units"]):
        """Update the axes attributes to match those given.

        The axes are matched by their index in the array. The purpose of this
        method is to update multiple axes triggering `any_axis_changed` only
        once.

        Parameters
        ----------
        axes: iterable of `DataAxis` instances.
            The axes to copy the attributes from.
        attributes: iterable of strings.
            The attributes to copy.

        """

        # To only trigger once even with several changes, we suppress here
        # and trigger manually below if there were any changes.
        changes = False
        with self.events.any_axis_changed.suppress():
            for axis in axes:
                changed = self._axes[axis.index_in_array].update_from(
                    axis=axis, attributes=attributes)
                changes = changes or changed
        if changes:
            self.events.any_axis_changed.trigger(obj=self)

    def _update_attributes(self):
        getitem_tuple = []
        values = []
        self.signal_axes = ()
        self.navigation_axes = ()
        for axis in self._axes:
            # Until we find a better place, take property of the axes
            # here to avoid difficult to debug bugs.
            axis.axes_manager = self
            if axis.slice is None:
                getitem_tuple += axis.index,
                values.append(axis.value)
                self.navigation_axes += axis,
            else:
                getitem_tuple += axis.slice,
                self.signal_axes += axis,
        if not self.signal_axes and self.navigation_axes:
            getitem_tuple[-1] = slice(axis.index, axis.index + 1)

        self.signal_axes = self.signal_axes[::-1]
        self.navigation_axes = self.navigation_axes[::-1]
        self._getitem_tuple = tuple(getitem_tuple)
        self.signal_dimension = len(self.signal_axes)
        self.navigation_dimension = len(self.navigation_axes)
        if self.navigation_dimension != 0:
            self.navigation_shape = tuple([
                axis.size for axis in self.navigation_axes])
        else:
            self.navigation_shape = ()

        if self.signal_dimension != 0:
            self.signal_shape = tuple([
                axis.size for axis in self.signal_axes])
        else:
            self.signal_shape = ()
        self.navigation_size = (np.cumprod(self.navigation_shape)[-1]
                                if self.navigation_shape else 0)
        self.signal_size = (np.cumprod(self.signal_shape)[-1]
                            if self.signal_shape else 0)
        self._update_max_index()

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
        if len(self._axes) == 0:
            return
        elif value > len(self._axes):
            raise ValueError(
                "The signal dimension cannot be greater"
                " than the number of axes which is %i" % len(self._axes))
        elif value < 0:
            raise ValueError(
                "The signal dimension must be a positive integer")

        tl = [True] * len(self._axes)
        if value != 0:
            tl[-value:] = (False,) * value

        for axis in self._axes:
            axis.navigate = tl.pop(0)

    def key_navigator(self, event):
        if len(self.navigation_axes) not in (1, 2):
            return
        x = self.navigation_axes[0]
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
                y = self.navigation_axes[1]
                if event.key == "up" or event.key == "8":
                    y.index -= self._step
                elif event.key == "down" or event.key == "2":
                    y.index += self._step
        except TraitError:
            pass

    def gui(self):
        from hyperspy.gui.axes import data_axis_view
        for axis in self._axes:
            axis.edit_traits(view=data_axis_view)

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, *args):
        return AxesManager(self._get_axes_dicts())

    def _get_axes_dicts(self):
        axes_dicts = []
        for axis in self._axes:
            axes_dicts.append(axis.get_axis_dictionary())
        return axes_dicts

    def as_dictionary(self):
        am_dict = {}
        for i, axis in enumerate(self._axes):
            am_dict['axis-%i' % i] = axis.get_axis_dictionary()
        return am_dict

    def _get_signal_axes_dicts(self):
        return [axis.get_axis_dictionary() for axis in
                self.signal_axes[::-1]]

    def _get_navigation_axes_dicts(self):
        return [axis.get_axis_dictionary() for axis in
                self.navigation_axes[::-1]]

    def show(self):
        from hyperspy.gui.axes import get_axis_group
        import traitsui.api as tui
        context = {}
        ag = []
        for n, axis in enumerate(self._get_axes_in_natural_order()):
            ag.append(get_axis_group(n, str(axis)))
            context['axis%i' % n] = axis
        ag = tuple(ag)
        self.edit_traits(view=tui.View(*ag), context=context)

    def _get_dimension_str(self):
        string = "("
        for axis in self.navigation_axes:
            string += str(axis.size) + ", "
        string = string.rstrip(", ")
        string += "|"
        for axis in self.signal_axes:
            string += str(axis.size) + ", "
        string = string.rstrip(", ")
        string += ")"
        return string

    def __repr__(self):
        text = ('<Axes manager, axes: %s>\n' %
                self._get_dimension_str())
        ax_signature = "% 16s | %6g | %6s | %7.2g | %7.2g | %6s "
        signature = "% 16s | %6s | %6s | %7s | %7s | %6s "
        text += signature % ('Name', 'size', 'index', 'offset', 'scale',
                             'units')
        text += '\n'
        text += signature % ('=' * 16, '=' * 6, '=' * 6,
                             '=' * 7, '=' * 7, '=' * 6)
        for ax in self.navigation_axes:
            text += '\n'
            text += ax_signature % (str(ax.name)[:16], ax.size, str(ax.index),
                                    ax.offset, ax.scale, ax.units)
        text += '\n'
        text += signature % ('-' * 16, '-' * 6, '-' * 6,
                             '-' * 7, '-' * 7, '-' * 6)
        for ax in self.signal_axes:
            text += '\n'
            text += ax_signature % (str(ax.name)[:16], ax.size, ' ', ax.offset,
                                    ax.scale, ax.units)

        return text

    def _repr_html_(self):
        text = ("<style>\n"
                "table, th, td {\n\t"
                "border: 1px solid black;\n\t"
                "border-collapse: collapse;\n}"
                "\nth, td {\n\t"
                "padding: 5px;\n}"
                "\n</style>")
        text += ('\n<p><b>< Axes manager, axes: %s ></b></p>\n' %
                 self._get_dimension_str())

        def format_row(*args, tag='td', bold=False):
            if bold:
                signature = "\n<tr class='bolder_row'> "
            else:
                signature = "\n<tr> "
            signature += " ".join(("{}" for _ in args)) + " </tr>"
            return signature.format(*map(lambda x:
                                         '\n<' + tag +
                                         '>{}</'.format(x) + tag + '>',
                                         args))
        if self.navigation_axes:
            text += "<table style='width:100%'>\n"
            text += format_row('Navigation axis name', 'size', 'index', 'offset',
                               'scale', 'units', tag='th')
            for ax in self.navigation_axes:
                text += format_row(ax.name, ax.size, ax.index, ax.offset, ax.scale,
                                   ax.units)
            text += "</table>\n"
        if self.signal_axes:
            text += "<table style='width:100%'>\n"
            text += format_row('Signal axis name', 'size', 'offset', 'scale',
                               'units', tag='th')
            for ax in self.signal_axes:
                text += format_row(ax.name, ax.size, ax.offset, ax.scale,
                                   ax.units)
            text += "</table>\n"
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

    def _get_axis_attribute_values(self, attr):
        return [getattr(axis, attr) for axis in self._axes]

    def _set_axis_attribute_values(self, attr, values):
        """Set the given attribute of all the axes to the given
        value(s)

        Parameters
        ----------
        attr : string
            The DataAxis attribute to set.
        values: any
            If iterable, it must have the same number of items
            as axes are in this AxesManager instance. If not iterable,
            the attribute of all the axes are set to the given value.

        """
        if not isiterable(values):
            values = [values, ] * len(self._axes)
        elif len(values) != len(self._axes):
            raise ValueError("Values must have the same number"
                             "of items are axes are in this AxesManager")
        for axis, value in zip(self._axes, values):
            setattr(axis, attr, value)

    @property
    def navigation_indices_in_array(self):
        return tuple([axis.index_in_array for axis in self.navigation_axes])

    @property
    def signal_indices_in_array(self):
        return tuple([axis.index_in_array for axis in self.signal_axes])

    @property
    def axes_are_aligned_with_data(self):
        """Verify if the data axes are aligned with the signal axes.

        When the data are aligned with the axes the axes order in `self._axes`
        is [nav_n, nav_n-1, ..., nav_0, sig_m, sig_m-1 ..., sig_0].

        Returns
        -------
        aligned : bool

        """
        nav_iia_r = self.navigation_indices_in_array[::-1]
        sig_iia_r = self.signal_indices_in_array[::-1]
        iia_r = nav_iia_r + sig_iia_r
        aligned = iia_r == tuple(range(len(iia_r)))
        return aligned

    def _sort_axes(self):
        """Sort _axes to align them.

        When the data are aligned with the axes the axes order in `self._axes`
        is [nav_n, nav_n-1, ..., nav_0, sig_m, sig_m-1 ..., sig_0]. This method
        sort the axes in this way. Warning: this doesn't sort the `data` axes.

        """
        am = self
        new_axes = am.navigation_axes[::-1] + am.signal_axes[::-1]
        self._axes = list(new_axes)
