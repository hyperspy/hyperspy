# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

import numpy as np
import matplotlib.pyplot as plt
from hyperspy.events import Event, Events
import logging
from matplotlib.collections import Collection

_logger = logging.getLogger(__name__)


class MarkerCollection(object):

    """This represents a collection of Markers.  For many markers this
    is __much much__ faster.  It also has a bit more of a universal
    approach to handling many patches at the slight cost of a little
    bit of flexibility.

    As a general rule each marker is defined by an offset and a size

    Attributes
    ----------
    marker_properties : dictionary
        Accepts a dictionary of valid (i.e. recognized by mpl.plot)
        containing valid line properties. In addition it understands
        the keyword `type` that can take the following values:
        {'line', 'text'}
    """

    def __init__(self):
        # Data attributes
        self.data = None
        self.axes_manager = None
        self.ax = None
        self.auto_update = True

        # Properties
        self.collection = None
        self._marker_properties = {}
        self.signal = None
        self._plot_on_signal = True
        self.name = ''
        self.plot_marker = True
        self._column_keys= {}

        # Events
        self.events = Events()
        self.events.closed = Event("""
            Event triggered when a marker is closed.

            Arguments
            ---------
            marker : Marker
                The marker that was closed.
            """, arguments=['obj'])
        self._closing = False

    @property
    def navigation_shape(self):
        if self.data.dtype == object:
            return self.data.shape
        else:
            return ()
    @property
    def is_static(self):
        return self.data.dtype == object

    def _to_dictionary(self):
        marker_dict = {
            'marker_type': self.__class__.__name__,
            'plot_on_signal': self._plot_on_signal,
            'data': self.data}
        return marker_dict

    def set_data(self,
                 data=None,
                 **kwargs):
        """
        Set data for some signal.
        There are two cases which are handled independently.
        Case1:
            Data is passed in one block.  In this case the data is parsed
            and every column is defined by the default column keys for each
            marker subclass.
        Case2:
            Data is passed as a set of **kwargs.  This must equal the _column_keys for
            some subclass.
        """
        if data is None: # Case 2 iterating marker from kwargs
            for k in kwargs:
                kwargs[k]= np.asanyarray(kwargs[k])
            if not all([k in self._column_keys.keys() for k in kwargs]):
                raise ValueError(f"The data for a marker needs to be either a "
                                 f"properly formatted array and passed using the"
                                 f"`data` kwarg otherwise the individual keys: "
                                 f"{self._column_keys.keys()} can be passed as keyword"
                                 f"arguements")
            shapes = np.array([np.array(kwargs[k]).shape for k in kwargs])
            if not np.all(shapes == shapes[0]):
                raise ValueError("All of the shapes for the "
                                 "data fields must be equal")
            if any([kwargs[k].dtype == object for k in kwargs]): # marker not static
                data = np.empty(shape=shapes[0],
                                    dtype=object)
                for ind in np.ndindex[shapes[0]]:
                    d = np.stack([kwargs[k][ind] for k in kwargs], axis=1)
                    data[ind] = d  # save dict at position
            else: # Case 3 static marker
                data = np.stack([kwargs[k] for k in kwargs], axis=1)
        self.data =data

    def get_data_position(self):
        if self.is_static:
            current_data = self.data
        else:
            indices = self.axes_manager.indices[::-1]
            current_data = self.data[indices]
        kwds = {}
        columns = current_data.shape[1] # number of columns
        for k, v in self._column_keys.items():
            if v < columns:
                kwds[k] = current_data[:, v]
            else:
                # If the column isn't in the array return None
                kwds[k] = None
        return kwds

    def update(self, offsets, sizes):
        """
        Generic Update to some collection.  This method is overwritten by the
        LineCollection,  EllipseCollection and RectangleCollection.
        Parameters
        ----------
        offsets
        sizes

        Returns
        -------

        """
        self.collection.set_offsets(offsets)
        self.collection.set_sizes(sizes)


    def plot(self, render_figure=True):
        """
        Plot a marker which has been added to a signal.

        Parameters
        ----------
        render_figure : bool, optional, default True
            If True, will render the figure after adding the marker.
            If False, the marker will be added to the plot, but will the figure
            will not be rendered. This is useful when plotting many markers,
            since rendering the figure after adding each marker will slow
            things down.
        """
        if self.ax is None:
            raise AttributeError(
                "To use this method the marker needs to be first add to a " +
                "figure using `s._plot.signal_plot.add_marker(m)` or " +
                "`s._plot.navigator_plot.add_marker(m)`")
        self._plot_marker()
        self.marker.set_animated(self.ax.figure.canvas.supports_blit)
        if render_figure:
            self._render_figure()

    def _render_figure(self):
        self.ax.hspy_fig.render_figure()

    def close(self, render_figure=True):
        """Remove and disconnect the marker.

        Parameters
        ----------
        render_figure : bool, optional, default True
            If True, the figure is rendered after removing the marker.
            If False, the figure is not rendered after removing the marker.
            This is useful when many markers are removed from a figure,
            since rendering the figure after removing each marker will slow
            things down.
        """
        if self._closing:
            return
        self._closing = True
        self.marker.remove()
        self.events.closed.trigger(obj=self)
        for f in self.events.closed.connected:
            self.events.closed.disconnect(f)
        if render_figure:
            self._render_figure()


class LineSegmentCollection(MarkerCollection):
    def __init__(self, **kwargs):
        MarkerCollection.__init__(self)
        self.set_data(**kwargs)
        self._column_keys={"segments":slice(0,3), "linewidth":3}

    def update(self):
        data = self.get_data_position()
        self.collection.update_segments(data)


