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
import logging
import matplotlib

from matplotlib.transforms import Affine2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

from hyperspy.events import Event, Events
from hyperspy.misc.array_tools import _get_navigation_dimension_chunk_slice
from hyperspy.misc.utils import isiterable

_logger = logging.getLogger(__name__)


def convert_positions(peaks, signal_axes):
    new_data = np.empty(peaks.shape[:-1] + (len(signal_axes),))
    for i, ax in enumerate(signal_axes[::-1]):
        # indexes need to be reversed
        new_data[..., (-i - 1)] = ax.scale * peaks[..., i] + ax.offset
    return new_data


class Markers(object):
    """
    A set of markers for faster plotting.

    This is a generic class which is subclassed by other marker classes. Mostly
    by defining the collection_class.

    In most cases each marker is defined by some keyword argument and
    a (n,2) array of offsets which define the position for each marker
    in the plot.

    For example if we wanted to define a set of ellipses with constant
    height, width, and size we can do the following.

    >>>from matplotlib.collections import EllipseCollection
    >>>import hyperspy.api as hs
    >>>import hyperspy.api as hs
    >>>import numpy as np
    >>>m = hs.plot.marker.Markers(collection_class=EllipseCollection, widths=(2,),
    >>>                 heights=(1,), angles=(1,), units="xy", offsets=np.random.rand(10,2)*10)
    >>>s = hs.signals.Signal2D(np.ones((10,10,10,10)))
    >>>s.plot()
    >>>s.add_marker(m)

    To define a non-static marker any kwarg that can be set with the `matplotlib.collections.set`
    method can be passed as an array with `dtype=object` and the same size as the navigation axes
    for a signal.
    """

    def __init__(self, collection_class=None, **kwargs):
        """
        Initialize a Marker Collection.

        Parameters
        ----------
        collection_class: None or matplotlib.collections
            A Matplotlib collection to be initialized.
        offsets: [2,n]
            Positions of the markers
        size:
            Size of the markers
        *args: tuple
            Arguments passed to the underlying marker collection. Any argument
            that is array-like and has `dtype=object` is assumed to be an iterating
            argument and is treated as such.
        **kwargs:
            Keyword arguments passed to the underlying marker collection. Any argument
            that is array-like and has `dtype=object` is assumed to be an iterating
            argument and is treated as such.

        Examples
        --------

        Adding a marker using an EllipseCollection

        >>>from matplotlib.collections import EllipseCollection
        >>>import hyperspy.api as hs
        >>>import numpy as np
        >>>m = hs.plot.markers.Markers(collection_class=EllipseCollection, widths=(2,),
        ...                 heights=(1,), angles=(1,), units="xy", offsets=np.random.rand(10,2)*10)
        >>>s = hs.signals.Signal2D(np.ones((10,10,10,10)))
        >>>s.plot()
        >>>s.add_marker(m)

        Adding a marker using a PatchCollection (Making a circle)

        >>>from matplotlib.collections import PatchCollection
        >>>from matplotlib.patches import Circle
        >>>import hyperspy.api as hs
        >>>import numpy as np
        >>>m = hs.plot.markers.Markers(collection_class=PatchCollection,
        ...                                     patches=[Circle((0,0), 1)], offsets=np.random.rand(10,2)*10)
        >>>s = hs.signals.Signal2D(np.ones((10,10,10,10)))
        >>>s.plot()
        >>>s.add_marker(m)

        Adding a series of lines using a MarkerCollection

        >>>from matplotlib.collections import LineCollection
        >>>import hyperspy.api as hs
        >>>import numpy as np
        >>>m = hs.plot.markers.Markers(collection_class=LineCollection,
        ...                                     segments=np.random.rand(10,2,2)*10,)
        >>>s = hs.signals.Signal2D(np.ones((10,10,10,10)))
        >>>s.plot()
        >>>s.add_marker(m)

        Adding a marker using PolyCollection (Making a square)

        >>>from matplotlib.collections import PolyCollection
        >>>import hyperspy.api as hs
        >>>import numpy as np
        >>>m = hs.plot.markers.Markers(collection_class=PolyCollection,
        ...                                     offsets=np.random.rand(10,2)*10,
        ...                                     verts=np.array([[[0,0],[0,1],[1,1],[1,0]]]),color="red")
        >>>s = hs.signals.Signal2D(np.ones((10,10,10,10)))
        >>>s.plot()
        >>>s.add_marker(m)

        Adding a marker using scatter

        >>>import hyperspy.api as hs
        >>>import numpy as np
        >>>m = hs.plot.markers.Markers(offsets=np.random.rand(10,2)*10, sizes=1,)
        >>>s = hs.signals.Signal2D(np.ones((10,10,10,10)))
        >>>s.plot()

        """
        # Data attributes
        self.kwargs = kwargs  # all keyword arguments.
        self.axes_manager = None
        self.ax = None
        self.auto_update = True

        # Handling dask arrays
        self.dask_kwargs = {}
        for key, value in self.kwargs.items():
            if isinstance(value, da.Array) and value.dtype == object:
                self.dask_kwargs[key] = self.kwargs[key]
            elif isinstance(value, da.Array):  # and value.dtype != object:
                self.kwargs[key] = value.compute()
            # Patches shouldn't be cast to array
            elif (
                isinstance(value, list)
                and len(value) > 0
                and not isinstance(value[0], Patch)
                and not key == "verts"
            ):
                self.kwargs[key] = np.array(value)
            elif isinstance(value, list) and len(value) == 0:
                self.kwargs[key] = np.array(value)

            if key in ["sizes", "color"] and (not hasattr(value, "__len__") or isinstance(value, str)):
                self.kwargs[key] = (value,)
            if key in ["s"] and isinstance(value, str):
                self.kwargs[key] = (value,)

        self._cache_dask_chunk_kwargs = {}
        self._cache_dask_chunk_kwargs_slice = {}

        # Properties
        self.collection = None
        self.collection_class = collection_class
        self.signal = None
        self.temp_signal = None
        self._check_iterating_kwargs()
        self.is_iterating = np.any(
            [is_iterating(value) for key, value in self.kwargs.items()]
        )
        self._plot_on_signal = True
        self.name = "Markers"
        self.plot_marker = True
        self.offsets_structure = [["o", "get_xlim[0]"], ["o", "get_xlim[1]"]]

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

    def _check_iterating_kwargs(self):
        if self.collection_class is None:
            cc = matplotlib.collections.PathCollection
        else:
            cc = self.collection_class
        # for key, value in self.kwargs.items():
        #     if is_iterating(value) and key not in str(inspect.signature(cc.set)):
        #         raise ValueError(f"The keyword {key} is an dynamic argument and cannot be set using the"
        #                          f" {self.collection_class.__name__}.set function.  Another option is to"
        #                          f" use the IterPatchCollection which provides additional control at the"
        #                          f" cost of speed.")

    def delete_index(self, keys, index):
        """
        Delete the index from the kwargs.

        Parameters
        ----------
        keys: list
            List of keys to delete from.
        index: slice, int or array of ints
            Indicate indices of sub-arrays to remove along the specified axis.
        """
        if isinstance(keys, str):
            keys = [
                keys,
            ]
        for key in keys:
            if self.kwargs[key].dtype == object:
                for i in np.ndindex(self.kwargs[key].shape):
                    self.kwargs[key][i] = np.delete(self.kwargs[key][i], index, axis=0)
            else:
                self.kwargs[key] = np.delete(self.kwargs[key], index, axis=0)

    def append_kwarg(self, keys, value):
        """
        Add the index from the kwargs.

        Parameters
        ----------
        keys: list
            List of keys to append.
        value:
            The value to append to the kwarg.
        """
        if isinstance(keys, str):
            keys = [
                keys,
            ]
        # skip if value is empty
        if len(value) == 0:
            return
        for key in keys:
            if self.kwargs[key].dtype == object:
                for i in np.ndindex(self.kwargs[key].shape):
                    self.kwargs[key][i] = np.append(self.kwargs[key][i], value, axis=0)
            else:
                self.kwargs[key] = np.append(self.kwargs[key], value, axis=0)

    def _get_chunk_slice(self, key, index_slice):
        """
        Get the slice for a chunk of data.

        Parameters
        ----------
        key: str
            The key to get the slice for.
        index_slice: slice, int or array of ints
            Indicate indices of sub-arrays to remove along the specified axis.
        """
        if self.kwargs[key].dtype == object:
            return index_slice
        else:
            return index_slice

    def _get_cache_dask_kwargs_chunk(self, indices):
        """
        Get the kwargs at some index.  If the index is cached return the cached value
        otherwise compute the kwargs and cache them.
        """
        chunks = {key: value.chunks for key, value in self.dask_kwargs.items()}
        chunk_slices = {
            key: _get_navigation_dimension_chunk_slice(indices, chunk)
            for key, chunk in chunks.items()
        }
        to_compute = {}
        for key, value in self.dask_kwargs.items():
            index_slice = chunk_slices[key]
            current_slice = self._cache_dask_chunk_kwargs_slice.get(key, None)
            if current_slice is None or current_slice != index_slice:
                to_compute[key] = value[index_slice]
                self._cache_dask_chunk_kwargs_slice[key] = index_slice

        if len(to_compute) > 0:
            # values = da.compute([value for value in to_compute.values()])
            # self._cache_dask_chunk_kwargs.update({key: value for
            #                                      key, value in zip(to_compute.keys(), values)})
            for key in to_compute:  # this should be one da.compute() function...
                self._cache_dask_chunk_kwargs[key] = to_compute[key].compute()

        out_kwargs = {}
        for key, value in self._cache_dask_chunk_kwargs.items():
            temp_indices = list(indices)
            for i, temp_slice in enumerate(chunk_slices[key]):
                # add offset to the indices
                temp_indices[i] -= temp_slice.start
            temp_indices = tuple(temp_indices)
            out_kwargs[key] = self._cache_dask_chunk_kwargs[key][temp_indices]
        return out_kwargs

    def __repr__(self):
        if self.collection_class is not None:
            return (
                f"<{self.name}| Class{self.collection_class} |"
                f"Iterating == {self.is_iterating}"
            )
        else:
            return f"<{self.name}| Iterating == {self.is_iterating}"

    @classmethod
    def from_signal(
        cls,
        signal,
        key="offsets",
        signal_axes="metadata",
        **kwargs,
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
                output_dtype=object,
                signal_axes=signal.metadata.Peaks.signal_axes,
            )
        elif isinstance(signal_axes, (tuple, list)):
            new_signal = signal.map(
                convert_positions,
                inplace=False,
                ragged=True,
                output_dtype=object,
                signal_axes=signal_axes
            )
        else:
            raise ValueError(
                "The keyword argument `signal_axes` must be one of 'metadata' a"
                "tuple of `DataAxes` or None"
            )
        kwargs[key] = new_signal.data
        return cls(**kwargs)

    def _get_data_shape(self):
        for key, item in self.kwargs.items():
            if is_iterating(item):
                return item.shape
        return ()

    def __deepcopy__(self, memo):
        new_marker = markers2collection(self._to_dictionary())
        return new_marker

    def _to_dictionary(self):
        marker_dict = {
            "marker_type": "MarkerCollection",
            "collection_class": self.collection_class,
            "plot_on_signal": self._plot_on_signal,
            "kwargs": self.kwargs,
        }
        return marker_dict

    def get_data_position(self, get_static_kwargs=True):
        """
        Return the current keyword arguments for updating the collection.
        """
        current_keys = {}
        if self.is_iterating:
            indices = self.axes_manager.indices[::-1]
            for key, value in self.kwargs.items():
                if is_iterating(value):
                    if key not in self.dask_kwargs:
                        val = value[indices]
                        # some keys values need to iterate
                        if key in ["sizes", "color"] and not hasattr(val, "__len__"):
                            val = (val,)
                        current_keys[key] = val
                elif get_static_kwargs:
                    current_keys[key] = value
                else:  # key already set in init
                    pass
            if len(self.dask_kwargs) > 0:
                current_keys.update(self._get_cache_dask_kwargs_chunk(indices))
        else:
            current_keys = self.kwargs
        return current_keys

    def update(self):
        if not self.is_iterating:
            return
        else:
            kwds = self.get_data_position(get_static_kwargs=False)
            self.collection.set(**kwds)

    def _initialize_collection(self):
        if self.collection_class is None:
            self.collection = self.ax.scatter(
                [],
                [],
            )
            self.collection.set(**self.get_data_position())
        elif self.collection_class is LineCollection:
            self.collection = self.collection_class(
                **self.get_data_position(),
            )
        else:
            self.collection = self.collection_class(
                **self.get_data_position(),
                transOffset=self.ax.transData,
            )

        # Set the size of each marker based on the "x" axis scale (or the x scale for the original dpi)
        # In order to get each Marker to scale we would need to define a list of transforms
        # This is possible but difficult to implement cleanly. For now,
        # we will just scale the markers based on the origional x axis and they won't scale with the
        # figure size changing...
        if self.collection_class is not LineCollection:
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
        self.temp_signal = None
        for f in self.events.closed.connected:
            self.events.closed.disconnect(f)
        if render_figure:
            self._render_figure()


def is_iterating(arg):
    return isinstance(arg, (np.ndarray, da.Array)) and arg.dtype == object


def dict2vector(data, keys=None, return_size=True, dtype=float):
    """Take some dictionary of values and create offsets based on the input keys.
    For instances like creating a horizontal or vertical line then some key is duplicated.

    Multiple keys can be passed as well. For example to define a rectangle:

    >>> dict2offsets(data,keys= [['x1','y1'], ['x2','y1'], ['x2', 'y2'],['x1', 'y2']])

    In this example the keys will be unpacked to create a rectangle.
    """
    if keys is None:
        keys = [["x1", "x2"]]
    keys = np.array(keys)
    # check to see if the array should be ragged
    unique_keys = np.unique(keys)
    is_key_iter = [isiterable(data[key]) and not isinstance(data[key], str) for key in unique_keys]
    if not any(is_key_iter):# no iterable keys
        if dtype is str:
            dtype = object
        vector = np.empty(keys.shape, dtype=dtype)
        for i in np.ndindex(
            keys.shape
        ):  # iterate through keys and create resulting vector
            vector[i] = data[keys[i]]
        if dtype is object:
            vector = np.asarray(vector, dtype=str)
    else:
        iter_key = unique_keys[is_key_iter][0]
        nav_shape = data[iter_key].shape
        if not all(is_key_iter):  # only some values are iterating
            non_iterkeys = unique_keys[np.logical_not(is_key_iter)]
            for k in non_iterkeys:
                data[k] = np.full(shape=nav_shape, fill_value=data[k])
        vector = np.empty(nav_shape, dtype=object)  # Create ragged array
        for i in np.ndindex(nav_shape):
            if dtype is str:
                vect = []
                for j in np.ndindex(keys.shape):
                    vect.append(data[keys[j]][i])
                vect = np.array(vect)
            else:
                vect = np.empty(keys.shape, dtype=dtype)
                for j in np.ndindex(keys.shape):
                    vect[j] = data[keys[j]][i]
            vector[i] = vect
    if return_size:
        if not isiterable(data["size"]):
            size = data["size"]
        else:
            size = np.empty(data["size"].shape, dtype=object)
            for i in np.ndindex(data["size"].shape):
                size[i] = data["size"][i]
        return vector, size
    else:
        return vector


def markers2collection(marker_dict):
    """This function maps a marker dict to a MarkerCollection class.

    This provides continuity from markers saved with hyperspy 1.x.x and hyperspy 2.x.x.

    Warning: This function is not complete and will transfer all of the marker properties.
    """
    from hyperspy.utils.markers import (
        Markers,
        VerticalLines,
        HorizontalLines,
        Texts,
        Lines,
        Ellipses,
        Rectangles,
        Arrows,
    )
    from matplotlib.collections import PolyCollection

    if len(marker_dict) == 0:
        return {}
    marker_type = marker_dict.pop("marker_type")
    plot_on_signal = marker_dict.pop("plot_on_signal")

    if marker_type == "Point":
        offsets, size = dict2vector(
            marker_dict["data"], keys=None, return_size=True
        )
        marker = Markers(
            offsets=offsets, sizes=size, **marker_dict["marker_properties"]
        )
        marker
    elif marker_type == "HorizontalLine":
        y = dict2vector(marker_dict["data"], keys=["y1"], return_size=False)
        marker = HorizontalLines(y=y, **marker_dict["marker_properties"])

    elif marker_type == "HorizontalLineSegment":
        segments = dict2vector(
            marker_dict["data"], keys=[[["x1", "y1"], ["x2", "y1"]]], return_size=False
        )

        marker = Lines(segments=segments, **marker_dict["marker_properties"])
    elif marker_type == "LineSegment":
        segments = dict2vector(
            marker_dict["data"], keys=[[["x1", "y1"], ["x2", "y2"]]], return_size=False
        )
        marker = Lines(segments=segments, **marker_dict["marker_properties"])
    elif marker_type == "Arrow":
        offsets = dict2vector(
            marker_dict["data"], keys=[["x1", "y1"],], return_size=False
        )

        dx = dict2vector(
            marker_dict["data"], keys=[["x2"],], return_size=False
        )
        dy = dict2vector(
            marker_dict["data"], keys=[["y2"],], return_size=False
        )

        marker = Arrows(offsets, dx,
                        dy,
                        **marker_dict["marker_properties"])

    elif marker_type == "Rectangle":
        verts = dict2vector(
            marker_dict["data"],
            keys=[[["x1", "y1"], ["x2", "y1"], ["x2", "y2"], ["x1", "y2"]],],
            return_size=False,
        )
        marker = Markers(collection_class=PolyCollection,
                         verts=verts, **marker_dict["marker_properties"],
        )
    elif marker_type == "Ellipse":
        offsets = dict2vector(marker_dict["data"], keys=[["x1", "y1"], ],
                         return_size=False)

        width = dict2vector(marker_dict["data"], keys=["x2"], return_size=False)
        height = dict2vector(marker_dict["data"], keys=["y2"], return_size=False)
        marker = Ellipses(offsets=offsets,
                          widths=width,
                          heights=height,
                          **marker_dict["marker_properties"])
    elif marker_type == "Text":
        offsets = dict2vector(
            marker_dict["data"], keys=[["x1", "y1"]], return_size=False
        )
        text = dict2vector(marker_dict["data"],
                           keys=[["text"]],
                           return_size=False,
                           dtype=str)
        marker = Texts(
            offsets=offsets, s=text, **marker_dict["marker_properties"]
        )
    elif marker_type == "VerticalLine":
        x = dict2vector(marker_dict["data"], keys=["x1"], return_size=False)

        marker = VerticalLines(
            x=x, **marker_dict["marker_properties"]
        )
    elif marker_type == "VerticalLineSegment":
        segments = dict2vector(
            marker_dict["data"], keys=[[["x1", "y1"], ["x1", "y2"]]], return_size=False
        )

        marker = Lines(segments=segments,
                       **marker_dict["marker_properties"],
                       )
    elif marker_type == "MarkerCollection":
        marker = Markers(collection_class=marker_dict["collection_class"],
                         **marker_dict["kwargs"])
    else:
        raise ValueError(
            f"The marker_type: {marker_type} is not a hyperspy.marker class "
            f"and cannot be converted to a MarkerCollection"
        )
    marker.plot_on_signal = plot_on_signal
    return marker
