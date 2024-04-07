# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

import logging
from itertools import zip_longest

import numpy as np

from hyperspy.misc.array_tools import are_aligned
from hyperspy.misc.axis_tools import check_axes_calibration

_logger = logging.getLogger(__name__)


def _get_shapes(am, ignore_axis):
    if ignore_axis is not None:
        try:
            ignore_axis = am[ignore_axis]
        except ValueError:
            pass
    sigsh = (
        tuple(
            axis.size if (ignore_axis is None or axis is not ignore_axis) else 1
            for axis in am.signal_axes
        )
        if am.signal_dimension != 0
        else ()
    )

    navsh = (
        tuple(
            axis.size if (ignore_axis is None or axis is not ignore_axis) else 1
            for axis in am.navigation_axes
        )
        if am.navigation_dimension != 0
        else ()
    )
    return sigsh, navsh


def are_signals_aligned(*args, ignore_axis=None):
    if len(args) < 2:
        raise ValueError("This function requires at least two signal instances")

    args = list(args)
    am = args.pop().axes_manager

    sigsh, navsh = _get_shapes(am, ignore_axis)
    while args:
        amo = args.pop().axes_manager
        sigsho, navsho = _get_shapes(amo, ignore_axis)

        if not (
            are_aligned(sigsh[::-1], sigsho[::-1])
            and are_aligned(navsh[::-1], navsho[::-1])
        ):
            return False
    return True


def _check_and_get_longest_axis(axes):
    """Return the longest axis from a list of axes.

    In the case of ties, the first element in the list
    will be returned. Logs a warning if the axes have
    different calibrations.
    """
    only_left = filter(lambda x: x is not None, axes)
    longest_idx, longest = sorted(
        enumerate(only_left),
        key=lambda x: x[1].size,
        reverse=True,
    )[0]

    # Exit early if not DataAxis objects - None is used
    # in broadcast_signals() below as a filler so we
    # skip it here.
    def _check_wrapper(ax1, ax2):
        if ax1 is None or ax2 is None:
            return True

        return check_axes_calibration(ax1, ax2)

    # Returns a list of bools, where False
    # indicates an axis with a different
    # calibration to the longest axis
    axes_check = [
        _check_wrapper(axes[longest_idx], axes[i])
        for i in range(len(axes))
        if i != longest_idx
    ]

    if not all(axes_check):
        _logger.warning(
            "Axis calibration mismatch detected along axis "
            f"{longest.index_in_axes_manager}. The "
            f"calibration of signal {longest_idx} along "
            "this axis will be applied to all signals "
            "after stacking."
        )

    return longest


def broadcast_signals(*args, ignore_axis=None):
    """Broadcasts signals according to the HyperSpy broadcasting rules.

    signal and navigation spaces are each separately broadcasted according to
    the numpy broadcasting rules. One axis can be ignored and left untouched
    (or set to be size 1) across all signals.

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
    from hyperspy.signal import BaseSignal

    if len(args) < 2:
        raise ValueError("This function requires at least two signal instances")
    if any([not isinstance(a, BaseSignal) for a in args]):
        raise ValueError("Arguments must be of signal type.")
    args = list(args)
    if not are_signals_aligned(*args, ignore_axis=ignore_axis):
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
        for axes in zip_longest(
            *[s.axes_manager.navigation_axes for s in args], fillvalue=None
        ):
            longest = _check_and_get_longest_axis(axes)
            new_nav_axes.append(longest)
            new_nav_shapes.append(
                longest.size
                if (ignore_axis is None or ignore_axis not in axes)
                else None
            )
        new_sig_axes = []
        new_sig_shapes = []
        for axes in zip_longest(
            *[s.axes_manager.signal_axes for s in args], fillvalue=None
        ):
            longest = _check_and_get_longest_axis(axes)
            new_sig_axes.append(longest)
            new_sig_shapes.append(
                longest.size
                if (ignore_axis is None or ignore_axis not in axes)
                else None
            )

        results = []
        new_axes = new_nav_axes[::-1] + new_sig_axes[::-1]
        new_data_shape = new_nav_shapes[::-1] + new_sig_shapes[::-1]
        for s in args:
            data = s._data_aligned_with_axes
            sam = s.axes_manager
            sdim_diff = len(new_sig_axes) - len(sam.signal_axes)
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
                data = np.broadcast_to(data, thisshape)

            ns = s._deepcopy_with_new_data(data)
            ns.axes_manager._axes = [ax.copy() for ax in new_axes]
            ns.get_dimensions_from_data()
            results.append(ns.transpose(signal_axes=len(new_sig_axes)))
        return results
