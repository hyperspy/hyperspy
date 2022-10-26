# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

from operator import attrgetter
import numpy as np
import dask.array as da

from hyperspy.misc.utils import attrsetter
from hyperspy.misc.export_dictionary import parse_flag_string
from hyperspy import roi


def _slice_target(target, dims, both_slices, slice_nav=None, issignal=False):
    """Slices the target if appropriate

    Parameters
    ----------
    target : object
        Target object
    dims : tuple
        (navigation_dimensions, signal_dimensions) of the original object that
        is sliced
    both_slices : tuple
        (original_slices, array_slices) of the operation that is performed
    slice_nav : {bool, None}
        if None, target is returned as-is. Otherwise navigation and signal
        dimensions are sliced for True and False values respectively.
    issignal : bool
        if the target is signal and should be sliced as one
    """
    if slice_nav is None:
        return target
    if target is None:
        return None
    nav_dims, sig_dims = dims
    slices, array_slices = both_slices
    if slice_nav is True:  # check explicitly for safety
        if issignal:
            return target.inav[slices]
        sl = tuple(array_slices[:nav_dims])
        if isinstance(target, np.ndarray):
            return np.atleast_1d(target[sl])
        if isinstance(target, da.Array):
            return target[sl]
        raise ValueError(
            'tried to slice with navigation dimensions, but was neither a '
            'signal nor an array')
    if slice_nav is False:  # check explicitly
        if issignal:
            return target.isig[slices]
        sl = tuple(array_slices[-sig_dims:])
        if isinstance(target, np.ndarray):
            return np.atleast_1d(target[sl])
        if isinstance(target, da.Array):
            return target[sl]
        raise ValueError(
            'tried to slice with navigation dimensions, but was neither a '
            'signal nor an array')


def copy_slice_from_whitelist(
        _from, _to, dims, both_slices, isNav, order=None):
    """Copies things from one object to another, according to whitelist, slicing
    where required.

    Parameters
    ----------
    _from : object
        Original object
    _to : object
        Target object
    dims : tuple
        (navigation_dimensions, signal_dimensions) of the original object that
        is sliced
    both_slices : tuple
        (original_slices, array_slices) of the operation that is performed
    isNav : bool
        if the slicing operation is performed on navigation dimensions of the
        object
    order : tuple, None
        if given, performs the copying in the order given. If not all attributes
        given, the rest is random (the order a whitelist.keys() returns them).
        If given in the object, _slicing_order is looked up.
    """

    def make_slice_navigation_decision(flags, isnav):
        if isnav:
            if 'inav' in flags:
                return True
            return None
        if 'isig' in flags:
            return False
        return None

    swl = None
    if hasattr(_from, '_slicing_whitelist'):
        swl = _from._slicing_whitelist

    if order is not None and not isinstance(order, tuple):
        raise ValueError('order argument has to be None or a tuple of strings')

    if order is None:
        order = ()
    if hasattr(_from, '_slicing_order'):
        order = order + \
            tuple(k for k in _from._slicing_order if k not in order)

    keys = order + tuple(k for k in _from._whitelist.keys() if k not in
                         order)

    for key in keys:
        val = _from._whitelist[key]
        if val is None:
            # attrsetter(_to, key, attrgetter(key)(_from))
            # continue
            flags = []
        else:
            flags_str = val[0]
            flags = parse_flag_string(flags_str)

        if swl is not None and key in swl:
            flags.extend(parse_flag_string(swl[key]))

        if 'init' in flags:
            continue
        if 'id' in flags:
            continue

        if key == 'self':
            target = None
        else:
            target = attrgetter(key)(_from)

        if 'inav' in flags or 'isig' in flags:
            slice_nav = make_slice_navigation_decision(flags, isNav)
            result = _slice_target(
                target,
                dims,
                both_slices,
                slice_nav,
                'sig' in flags)
            attrsetter(_to, key, result)
            continue
        else:
            # 'fn' in flag or no flags at all
            attrsetter(_to, key, target)
            continue

def to_tuple(slices):
    """Handles conversion into a tuple for further slicing etc.
    """
    if isinstance(slices, tuple):
        return slices
    elif isinstance(slices, list) or isinstance(slices, np.ndarray):
        slices = (slices,)
        return slices
    else:
        slices = (slices, )
        return slices


def get_dim(slic):
    if isinstance(slic, np.ndarray) and slic.ndim > 1 and slic.dtype == bool:
        return slic.ndim
    elif isinstance(slic, (roi.CircleROI, roi.Point2DROI, roi.RectangularROI)):
        return 2
    elif slic is Ellipsis:
        return 0
    else:
        return 1


def to_axes_order(axes, slices, dimensions):
    grouped_axes = []
    array_index = []
    i = 0
    for d, sl in zip(dimensions, slices):
        grouped_axes.append(axes[i:i + d])
        array_index.append(axes[i].index_in_array)
        i += d
    array_index = np.argsort(array_index)
    new_slices = tuple([slices[a] for a in array_index])
    new_axes = tuple([grouped_axes[a] for a in array_index])
    return new_slices, new_axes


class SpecialSlicers(object):

    def __init__(self, obj, isNavigation):
        """Create a slice of the signal. The indexing supports integer,
        decimal numbers or strings (containing a decimal number and an units).

        >>> s = hs.signals.Signal1D(np.arange(10))
        >>> s
        <Signal1D, title: , dimensions: (|10)>
        >>> s.data
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> s.axes_manager[0].scale = 0.5
        >>> s.axes_manager[0].axis
        array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5])
        >>> s.isig[0.5:4.].data
        array([1, 2, 3, 4, 5, 6, 7])
        >>> s.isig[0.5:4].data
        array([1, 2, 3])
        >>> s.isig[0.5:4:2].data
        array([1, 3])
        >>> s.axes_manager[0].units = 'µm'
        >>> s.isig[:'2000 nm'].data
        array([0, 1, 2, 3])
        """
        self.isNavigation = isNavigation
        self.obj = obj

    def __getitem__(self, slices, out=None):
        return self.obj._slicer(slices, self.isNavigation, out=out)

class FancySlicing(object):

    def roi2slice(self, sl, starting_index=None):
        if isinstance(sl, roi.SpanROI):
                return (slice(float(sl.left), float(sl.right), None),)
        elif isinstance(sl, roi.Point1DROI):
            return (float(sl.value),)
        elif isinstance(sl, roi.Point2DROI):
            return (float(sl.x), float(sl.y),)
        elif isinstance(sl, roi.RectangularROI):
            return (slice(float(sl.left), float(sl.right), None),
                    slice(float(sl.top), float(sl.bottom), None),
                    )
        elif isinstance(sl, roi.CircleROI):
            ax1 = self.axes_manager[starting_index]
            ax2 = self.axes_manager[starting_index+1]
            gx,gy = np.meshgrid(ax1.axis-sl.cx, ax2.axis-sl.cy)
            gr = gx**2+gy**2
            mask = gr > sl.r**2
            mask |= gr < self.r_inner ** 2
            return (mask,)
        else:
            raise ValueError(f"The roi of type {sl.__class__} is not supported." )


    def _get_array_slices(self,slices,isNavigation=None):
        slices = to_tuple(slices)
        if isNavigation is None:
            axes = self.axes_manager.navigation_axes+self.axes_manager.signal_axes
        elif isNavigation:
            axes = self.axes_manager.navigation_axes
        else:
            axes = self.axes_manager.signal_axes

        # Need to handle Ellipsis before ROI because of Circle ROI needs a specific axis
        if any([sl is Ellipsis for sl in slices]):
            total_dim = np.sum([get_dim(sl) for sl in slices])
            ellipse_dim = len(axes)-total_dim
            slices = (slices[:slices.index(Ellipsis)] +
                      (slice(None),) * ellipse_dim +
                      slices[slices.index(Ellipsis)+1:]
                      )
        # Convert ROI's to slices or boolean indexes
        _slices = ()
        for sl in slices:
            _slices += self.roi2slice(sl) if isinstance(sl, roi.BaseROI) else (sl,)
        slices = _slices

        dimensions = [get_dim(sl) for sl in slices]
        # Add missing dimensions. Needed for reverse array indexing
        if np.sum(dimensions) < len(axes):
            slices = slices + (slice(None),) *(len(axes)-np.sum(dimensions))
            dimensions = [get_dim(sl) for sl in slices]
        if isNavigation == False: # Slicing signal Dimensions need to add in empty slices
            slices = (slice(None),) *self.axes_manager.navigation_dimension + slices
            axes = self.axes_manager.navigation_axes + axes
            dimensions = [get_dim(sl) for sl in slices]

        # reordering slices to array order
        slices, grouped_axes = to_axes_order(axes, slices, dimensions)

        array_slices = []
        if len(self.axes_manager._axes) == 1 and len(slices)==1 and isinstance(slices[0], (float, int)):
            slices = ([slices[0], ], )
        for sl, ax in zip(slices, grouped_axes):
            if len(ax)==1:
                if isinstance(sl, slice):
                    array_slices.append(ax[0]._get_array_slices(sl))
                else:
                    array_slices.append(ax[0].value2index(sl))
            else:
                array_slices.append(sl)
        return tuple(array_slices)

    def _slicer(self, slices, isNavigation=None, out=None):
        if self.axes_manager._ragged and not isNavigation:
            raise RuntimeError("`isig` is not supported for ragged signal.")

        array_slices = self._get_array_slices(slices, isNavigation)
        new_data = self.data[array_slices]
        if (self.ragged and new_data.dtype != np.dtype(object) and
                isinstance(new_data, np.ndarray)):
            # Numpy will convert the array to non-ragged, for consistency,
            # we make a ragged array with only one item
            data = new_data.copy()
            new_data = np.empty((1, ), dtype=object)
            new_data[0] = data

        if out is None:
            _obj = self._deepcopy_with_new_data(new_data, copy_variance=True)
            _to_remove = []
            for slice_, axis in zip(array_slices, _obj.axes_manager._axes):
                if (isinstance(slice_, (slice, list, np.ndarray)) or
                        len(self.axes_manager._axes) < 2):
                    axis._slice_me(slice_)
                else:
                    _to_remove.append(axis.index_in_axes_manager)
            _obj._remove_axis(_to_remove)
        else:
            out.data = new_data
            _obj = out
            i = 0
            for slice_, axis_src in zip(array_slices, self.axes_manager._axes):
                axis_src = axis_src.copy()
                if (isinstance(slice_, slice) or
                        len(self.axes_manager._axes) < 2):
                    axis_src._slice_me(slice_)
                    axis_dst = out.axes_manager._axes[i]
                    i += 1
                    axis_dst.update_from(axis_src, attributes=(
                        "scale", "offset", "size"))

        if hasattr(self, "_additional_slicing_targets"):
            for ta in self._additional_slicing_targets:
                try:
                    t = attrgetter(ta)(self)
                    if out is None:
                        if hasattr(t, '_slicer'):
                            attrsetter(
                                _obj,
                                ta,
                                t._slicer(
                                    slices,
                                    isNavigation))
                    else:
                        target = attrgetter(ta)(_obj)
                        t._slicer(
                            slices,
                            isNavigation,
                            out=target)

                except AttributeError:
                    pass

        if out is None:
            return _obj
        else:
            out.events.data_changed.trigger(obj=out)
