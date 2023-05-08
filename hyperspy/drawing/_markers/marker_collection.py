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
import dask.array as da
import matplotlib
from hyperspy.events import Event, Events
import logging
from packaging.version import Version
_logger = logging.getLogger(__name__)


class MarkerCollection(object):

    """This represents a collection of Markers defined a list of
    offsets and a size.

    """

    def __init__(self,
                 collection_class=None,
                 **kwargs):
        """
        Initialize a Marker Collection.
        Parameters
        ----------
        Collection:
            A Matplotlib collection to be initialized.
        offsets: [2,n]
            Positions of the markers
        size:
            Size of the markers
        args: tuple
            Arguments passed to the underlying marker collection. Any argument
            that is array-like and has `dtype=object` is assumed to be an iterating
            argument and is treated as such.
        kwargs: dict
            Keyword arguments passed to the underlying marker collection. Any argument
            that is array-like and has `dtype=object` is assumed to be an iterating
            argument and is treated as such.
        """
        # Data attributes
        self.kwargs = kwargs
        self.axes_manager = None
        self.ax = None
        self.auto_update = True

        # Properties
        self.collection = None
        self.collection_class = collection_class
        self.signal = None
        self.is_iterating = np.any([is_iterating(value) for key, value
                                 in self.kwargs.items()])
        self._plot_on_signal = True
        self.name = ''
        self.plot_marker = True

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

    def _get_data_shape(self):
        for key,item in self.kwargs.items():
            if is_iterating(item):
                return item.shape
        return ()

    def _to_dictionary(self):
        marker_dict = {
            'marker_type': self.collection_class,
            'plot_on_signal': self._plot_on_signal,
            'kwargs': self.kwargs}
        return marker_dict


    def get_data_position(self):
        """
        Return the current keyword arguments for updating the collection.
        """
        indices = self.axes_manager.indices[::-1]
        current_keys = {}
        for key, value in self.kwargs.items():
            if is_iterating(value):
                current_keys[key]=value[indices]
            else:
                current_keys[key]=value
        return current_keys

    def update(self):
        if self.auto_update is False:
            return
        kwds = self.get_data_position()
        self.collection.set(**kwds)

    def initialize_collection(self):
        if self.collection_class is None:
            self.collection = self.ax.scatter([],[],
                                              **self.get_data_position(),)
        else:
            self.collection = self.collection_class(**self.get_data_position(),)
        if Version(matplotlib.__version__) < Version("3.5.0"):
            self.collection._transOffset= self.ax.transData
        else:
            self.collection.set_offset_transform(self.ax.transData)

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
        self.initialize_collection()
        self.collection.set_animated(self.ax.figure.canvas.supports_blit)
        self.ax.add_collection(self.collection)
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
        self.collection.remove()
        self.events.closed.trigger(obj=self)
        for f in self.events.closed.connected:
            self.events.closed.disconnect(f)
        if render_figure:
            self._render_figure()



def is_iterating(arg):
    return isinstance(arg,(np.ndarray, da.Array)) and arg.dtype==object




