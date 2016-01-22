# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import traits.api as t
from hyperspy.events import Events, Event
from hyperspy.axes import DataAxis


class BaseROI(t.HasTraits):

    """Base class for all ROIs.

    Provides some basic functionality that is likely to be shared between all
    ROIs, and serve as a common type that can be checked for.
    """

    def __init__(self):
        """Sets up events.changed event, and inits HasTraits.
        """
        super(BaseROI, self).__init__()
        self.events = Events()
        self.events.changed = Event("""
            Event that triggers when the ROI has changed.

            What constitues a change varies from ROI to ROI, but in general it
            should correspond to the region selected by the ROI has changed.

            Arguments:
            ----------
                roi :
                    The ROI that was changed.
            """, arguments=['roi'])
        self.signal_map = dict()

    _ndim = 0
    ndim = property(lambda s: s._ndim)

    def is_valid(self):
        """
        Determine if the ROI is in a valid state.

        This is typically determined by all the coordinates being defined,
        and that the values makes sense relative to each other.
        """
        raise NotImplementedError()

    def update(self):
        """Function responsible for updating anything that depends on the ROI.
        It should be called by implementors whenever the ROI changes.
        The base implementation simply triggers the changed event.
        """
        if self.is_valid():
            self.events.changed.trigger(self)

    def _get_ranges(self):
        """
        Utility to get the value ranges that the ROI would select.

        If the ROI is point base or is rectangluar in nature, these can be used
        slice a signal. Extracted from `_make_slices` to ease implementation
        in inherited ROIs.
        """
        raise NotImplementedError()

    def _make_slices(self, axes_collecion, axes, ranges=None):
        """
        Utility function to make a slice structure that will slice all the axes
        in 'axes_collecion'. The axes in the `axes` argument will be sliced by
        the ROI, all other axes with 'slice(None)'. Alternatively, if 'ranges'
        is passed, `axes[i]` will be sliced with 'ranges[i]'.
        """
        if ranges is None:
            # Use ROI to slice
            ranges = self._get_ranges()
        slices = []
        for ax in axes_collecion:
            if ax in axes:
                i = axes.index(ax)
                try:
                    ilow = ax.value2index(ranges[i][0])
                except ValueError:
                    if ranges[i][0] < ax.low_value:
                        ilow = ax.low_index
                    else:
                        raise
                if len(ranges[i]) == 1:
                    slices.append(ilow)
                else:
                    try:
                        ihigh = 1 + ax.value2index(
                            ranges[i][1], rounding=lambda x: round(x - 1))
                    except ValueError:
                        if ranges[i][0] < ax.high_value:
                            ihigh = ax.high_index + 1
                        else:
                            raise
                    slices.append(slice(ilow, ihigh))
            else:
                slices.append(slice(None))
        return tuple(slices)

    def __call__(self, signal, axes=None):
        """Slice the signal according to the ROI, and return it.

        Arguments
        ---------
        signal : Signal
            The signal to slice with the ROI.
        axes : specification of axes to use, default = None
            The axes argument specifies which axes the ROI will be applied on.
            The items in the collection can be either of the following:
                * a tuple of:
                    - DataAxis. These will not be checked with
                      signal.axes_manager.
                    - anything that will index signal.axes_manager
                * For any other value, it will check whether the navigation
                  space can fit the right number of axis, and use that if it
                  fits. If not, it will try the signal space.
        """
        if axes is None and signal in self.signal_map:
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager)

        natax = signal.axes_manager._get_axes_in_natural_order()
        slices = self._make_slices(natax, axes)
        if axes[0].navigate:
            if len(axes) == 2 and not axes[1].navigate:
                # Special case, since we can no longer slice axes in different
                # spaces together.
                return signal.inav[slices[0]].isig[slices[1]]
            slicer = signal.inav.__getitem__
            slices = slices[0:signal.axes_manager.navigation_dimension]
        else:
            slicer = signal.isig.__getitem__
            slices = slices[signal.axes_manager.navigation_dimension:]
        roi = slicer(slices)
        return roi

    def _parse_axes(self, axes, axes_manager):
        """Utility function to parse the 'axes' argument to a tuple of
        DataAxis, and find the matplotlib Axes that contains it.

        Arguments
        ---------
        axes : specification of axes to use, default = None
            The axes argument specifies which axes the ROI will be applied on.
            The DataAxis in the collection can be either of the following:
                * a tuple of:
                    - DataAxis. These will not be checked with
                      signal.axes_manager.
                    - anything that will index signal.axes_manager
                * For any other value, it will check whether the navigation
                  space can fit the right number of axis, and use that if it
                  fits. If not, it will try the signal space.
        axes_manager : AxesManager
            The AxesManager to use for parsing axes, if axes is not already a
            tuple of DataAxis.

        Returns
        -------
        (tuple(<DataAxis>), matplotlib Axes)
        """
        nd = self.ndim
        axes_out = []
        if isinstance(axes, (tuple, list)):
            for i in xrange(nd):
                if isinstance(axes[i], DataAxis):
                    axes_out.append(axes[i])
                else:
                    axes_out.append(axes_manager[axes[i]])
        else:
            if axes_manager.navigation_dimension >= nd:
                axes_out = axes_manager.navigation_axes[:nd]
            elif axes_manager.signal_dimension >= nd:
                axes_out = axes_manager.signal_axes[:nd]
            elif nd == 2 and axes_manager.navigation_dimensions == 1 and \
                    axes_manager.signal_dimension == 1:
                # We probably have a navigator plot including both nav and sig
                # axes.
                axes_out = [axes_manager.signal_axes[0],
                            axes_manager.navigation_axes[0]]
            else:
                raise ValueError("Could not find valid axes configuration.")

        return axes_out
