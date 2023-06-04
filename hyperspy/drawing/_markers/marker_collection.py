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
from hyperspy.drawing.marker import dict2marker
from matplotlib.transforms import Affine2D
import logging

_logger = logging.getLogger(__name__)


def convert_positions(peaks, signal_axes):
    if peaks.dtype == object and peaks.shape == (1,):
        peaks = peaks[0]
    new_data = np.empty(peaks.shape[:-1] + (len(signal_axes),))
    for i, ax in enumerate(signal_axes[::-1]):
        # indexes need to be reversed
        new_data[..., (-i - 1)] = ax.scale * peaks[:, i] + ax.offset
    return new_data


class MarkerCollection(object):
    """
    A Collection of Markers for faster plotting. A collection is a set of
    markers which have the same properties.

    In most cases each marker is defined by some keyword argument and
    a (n,2) array of offsets which define the position for each marker
    in the plot.

    For example if we wanted to define a set of ellipses with constant
    height, width, and size we can do the following.

    >>>from matplotlib.collections import EllipseCollection
    >>>from hyperspy.drawing.marker import MarkerCollection
    >>>import hyperspy.api as hs
    >>>import numpy as np
    >>>m = MarkerCollection(collection_class=EllipseCollection, widths=(2,),
    >>>                 heights=(1,), angles=(1,), units="xy", offsets=np.random.rand(10,2)*10)
    >>>s = hs.signals.Signal2D(np.ones((10,10,10,10)))
    >>>s.plot()
    >>>s.add_marker(m)

    To define a non-static marker any kwarg that can be set with the `matplotlib.collections.set`
    method can be passed as an array with `dtype=object` and the same size as the navigation axes
    for a signal.
    """

    def __init__(self,
                 collection_class=None,
                 **kwargs):
        """
        Initialize a Marker Collection.

        Parameters
        ----------
        Collection: None or matplotlib.collections
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
        self.is_iterating = np.any(
            [is_iterating(value) for key, value in self.kwargs.items()]
        )
        self._plot_on_signal = True
        self.name = "MarkerCollection"
        self.plot_marker = True

        # Events
        self.events = Events()
        self.events.closed = Event(
            """
            Event triggered when a marker is closed.

            Arguments
            ---------
            marker : Marker
                The marker that was closed.
            """,
            arguments=["obj"],
        )
        self._closing = False

    def __repr__(self):
        if self.collection_class is not None:
            return (f"<{self.name}| Class{self.collection_class} |" 
                    f"Iterating == {self.is_iterating}")
        else:
            return f"<{self.name}| Iterating == {self.is_iterating}"


    @classmethod
    def from_signal(
        cls,
        signal,
        key="offsets",
        collection_class=None,
        signal_axes="metadata",
        **kwargs
    ):
        """
        Initialize a marker collection from a hyperspy Signal.

        Parameters
        ----------
        signal: BaseSignal
            A value passed to the Collection as {key:signal.data}
        key: str
            The key used to create a key value pair to
            create the Collection. Passed as {key: signal.data}.
        collection_class: None or matplotlib.collections
            The collection which is initialized
        signal_axes: str, tuple of UniformAxes or None
            If "metadata" look for signal_axes saved in metadata under .metadata.Peaks.signal_axes
            and convert from pixel positions to real units before creating the collection. If a tuple
            of signal axes those Axes will be used otherwise no transformation will
            happen.
        """
        if signal_axes is None or (
            signal_axes == "metadata"
            and not signal.metadata.has_item("Peaks.signal_axes")
        ):
            new_signal = signal
        elif signal_axes == "metadata" and signal.metadata.has_item(
            "Peaks.signal_axes"
        ):
            new_signal = signal.map(
                convert_positions,
                inplace=False,
                ragged=True,
                signal_axes=signal.metadata.Peaks.signal_axes,
            )
        elif isinstance(signal_axes, (tuple, list)):
            new_signal = signal.map(
                convert_positions, inplace=False, ragged=True, signal_axes=signal_axes
            )
        else:
            raise ValueError(
                "The keyword argument `signal_axes` must be one of 'metadata' a"
                "tuple of `DataAxes` or None"
            )
        kwargs[key] = new_signal.data
        return cls(collection_class=collection_class, **kwargs)

    def _get_data_shape(self):
        for key, item in self.kwargs.items():
            if is_iterating(item):
                return item.shape
        return ()

    def __deepcopy__(self, memo):
        new_marker = dict2marker(self._to_dictionary(), self.name)
        return new_marker

    def _to_dictionary(self):
        marker_dict = {
            "marker_type": "MarkerCollection",
            "collection_class": self.collection_class,
            "plot_on_signal": self._plot_on_signal,
            "kwargs": self.kwargs,
        }
        return marker_dict

    def get_data_position(self,
                          get_static_kwargs=True):
        """
        Return the current keyword arguments for updating the collection.
        """
        current_keys = {}
        if self.is_iterating:
            indices = self.axes_manager.indices[::-1]
            for key, value in self.kwargs.items():
                if is_iterating(value):
                    val = value[indices]
                    # some keys values need to iterate
                    if key in ["sizes", "color"] and not hasattr(val, "__len__"):
                        val = (val,)
                    current_keys[key] = val
                elif get_static_kwargs:
                    val = value
                    if key in ["sizes", "color"] and not hasattr(val, "__len__"):
                        val = (val,)
                    current_keys[key] = val
                else:  # key already set in init
                    pass
        else:
            current_keys = self.kwargs
            for key, value in self.kwargs.items():
                if key in ["sizes", "color"] and not hasattr(value, "__len__"):
                    current_keys[key] = (value,)
        return current_keys

    def update(self):
        if self.is_iterating is False:
            return
        kwds = self.get_data_position(get_static_kwargs=False)
        self.collection.set(**kwds)

    def _initialize_collection(self):
        if self.collection_class is None:
            self.collection = self.ax.scatter([], [],)
            self.collection.set(**self.get_data_position())
        else:
            self.collection = self.collection_class(
                **self.get_data_position(),
                transOffset=self.ax.transData,
            )
        sc = self.ax.bbox.width / self.ax.viewLim.width
        trans = Affine2D().scale(sc)
        self.collection.set_transform(trans)

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
                "To use this method the marker needs to be first add to a "
                + "figure using `s._plot.signal_plot.add_marker(m)` or "
                + "`s._plot.navigator_plot.add_marker(m)`"
            )
        self._initialize_collection()
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
    return isinstance(arg, (np.ndarray, da.Array)) and arg.dtype == object
