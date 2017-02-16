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

from hyperspy.misc.array_tools import are_aligned
from itertools import zip_longest
import numpy as np
import dask.array as da


def are_signals_aligned(*args):
    if len(args) < 2:
        raise ValueError(
            "This function requires at least two signal instances")
    args = list(args)
    am = args.pop().axes_manager

    while args:
        amo = args.pop().axes_manager
        if not (are_aligned(am.signal_shape[::-1], amo.signal_shape[::-1]) and
                are_aligned(am.navigation_shape[::-1],
                            amo.navigation_shape[::-1])):
            return False
    return True

def broadcast_signals(*args, ignore_axis=None):
    """Broadcasts all passed signals according to the HyperSpy broadcasting
    rules: signal and navigation spaces are each separately broadcasted
    according to the numpy broadcasting rules. One axis can be ignored and
    left untouched (or set to be size 1) across all signals.

    Parameters
    ----------
    *args : BaseSignal
        Signals to broadcast together
    ignore_axis : {None, str, int, Axis}
        The axis to be ignored when broadcasting

    Returns
    -------
    list of signals
    """
    if len(args) < 2:
        raise ValueError(
            "This function requires at least two signal instances")
    args = list(args)
    if not are_signals_aligned(*args):
        raise ValueError("The signals cannot be broadcasted")
    else:
        if ignore_axis is not None:
            for s in args:
                try:
                    ignore_axis = s.axes_manager[ignore_axis]
                    break
                except ValueError:
                    pass
        new_nav_axes = []
        new_nav_shapes = []
        for axes in zip_longest(*[s.axes_manager.navigation_axes
                                 for s in args], fillvalue=None):
            only_left = filter(lambda x: x is not None, axes)
            longest = sorted(only_left, key=lambda x: x.size, reverse=True)[0]
            new_nav_axes.append(longest)
            new_nav_shapes.append(longest.size if (ignore_axis is None or
                                                   ignore_axis not in
                                                   axes)
                                  else None)
        new_sig_axes = []
        new_sig_shapes = []
        for axes in zip_longest(*[s.axes_manager.signal_axes
                                 for s in args], fillvalue=None):
            only_left = filter(lambda x: x is not None, axes)
            longest = sorted(only_left, key=lambda x: x.size, reverse=True)[0]
            new_sig_axes.append(longest)
            new_sig_shapes.append(longest.size if (ignore_axis is None or
                                                   ignore_axis not in
                                                   axes)
                                  else None)

        results = []
        new_axes = new_nav_axes[::-1] + new_sig_axes[::-1]
        new_data_shape = new_nav_shapes[::-1] + new_sig_shapes[::-1]
        for s in args:
            data = s._data_aligned_with_axes
            sam = s.axes_manager
            sdim_diff = len(new_sig_axes) - sam.signal_dimension
            while sdim_diff > 0:
                slices = (slice(None),) * sam.navigation_dimension
                slices += (None, Ellipsis)
                data = data[slices]
                sdim_diff -= 1
            thisshape = new_data_shape.copy()
            if ignore_axis is not None:
                _id = new_data_shape.index(None)
                newlen = data.shape[_id] if len(data.shape) > _id else 1
                thisshape[_id] = newlen
            thisshape = tuple(thisshape)
            if data.shape != thisshape:
                if isinstance(data, np.ndarray):
                    data = np.broadcast_to(data, thisshape)
                else:
                    data = da.broadcast_to(data, thisshape)

            ns = s._deepcopy_with_new_data(data)
            ns.axes_manager._axes = [ax.copy() for ax in new_axes]
            ns.get_dimensions_from_data() 
            results.append(ns.transpose(signal_axes=len(new_sig_axes)))
        return results
