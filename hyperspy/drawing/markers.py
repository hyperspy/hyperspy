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
from copy import deepcopy

import matplotlib.collections as mpl_collections
from matplotlib.transforms import Affine2D, IdentityTransform
from matplotlib.patches import Patch

import hyperspy
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


class Markers:

    # The key defining the position, typically: `offsets`, `segments` or `verts`
    _position_key = "offsets"
    # For VerticalLines and HorizontalLines, the key to set is different from
    # `_position_key`
    _position_key_to_set = None

    def __init__(
        self,
        collection,
        offset_transform="data",
        transform="display",
        shift=None,
        plot_on_signal=True,
        name="",
        **kwargs,
    ):
        """
        Create a set of markers using Matplotlib collections.

        The markers are defined by a set of arugment required by the collections,
        typically, ``offsets``, ``verts`` or ``segments`` will define their
        positions.

        To define a non-static marker any argument that can be set with the
        :py:meth:`matplotlib.collections.Collection.set` method can be passed
        as an array with `dtype=object` of the constructor and the same size as
        the navigation axes of the a signal the markers will be added to.

        Parameters
        ----------
        collection : matplotlib.collections or str
            A Matplotlib collection to be initialized.
        offset_transform, transform : str
            Define the transformation used for the `offsets`. This only operates on the offset point so it won't
            scale the size of the ``Path``.  It can be one of the following:
            - ``"data"``: the offsets are defined in data coordinates and the ``ax.transData`` transformation is used.
            - ``"relative"``: The offsets are defined in data coordinates in x and coordinates in y relative to the
              data plotted. Only for 1D figure.
            - ``"axes"``: the offsets are defined in axes coordinates and the ``ax.transAxes`` transformation is used.
              (0, 0) is bottom left of the axes, and (1, 1) is top right of the axes.
            - ``"xaxis"``: The offsets are defined in data coordinates in x and axes coordinates in y direction; use
              :py:meth:`matplotlib.axes.Axes.get_xaxis_transform` transformation.
            - ``"yaxis"``: The offsets are defined in data coordinates in y and axes coordinates in x direction; use
              :py:meth:`matplotlib.axes.Axes.get_xaxis_transform` transformation..
            - ``"display"``: the offsets are not transformed, i.e. are defined in the display coordinate system.
              (0, 0) is the bottom left of the window, and (width, height) is top right of the output in "display units"
              :py:class:`matplotlib.transforms.IndentityTransform`.
        shift : None or float
            Shift in matplotlib ``"axes"`` coordinate system. Must be a value
            between 0 and 1.
        plot_on_signal : bool
            If True, plot on signal figure, otherwise on navigator.
        name : str
            The name of the markers.
        **kwargs : dict
            Keyword arguments passed to the underlying marker collection. Any argument
            that is array-like and has ``dtype=object`` is assumed to be an iterating
            argument and is treated as such.

        Examples
        --------
        Add markers using a :py:class:`matplotlib.collections.PatchCollection`
        which will display the specified subclass of :py:class:`matplotlib.patches.Patch`
        at the position defined by the argument ``offsets`` .

        >>> from matplotlib.collections import PatchCollection
        >>> from matplotlib.patches import Circle
        >>> import hyperspy.api as hs
        >>> import numpy as np
        >>> m = hs.plot.markers.Markers(
        ...    collection=PatchCollection,
        ...    patches=[Circle((0, 0), 1)],
        ...    offsets=np.random.rand(10,2)*10,
        ...    )
        >>> s = hs.signals.Signal2D(np.ones((10,10,10,10)))
        >>> s.plot()
        >>> s.add_marker(m)

        Adding star "iterating" markers using :py:meth:`matplotlib.collections.StarPolygonCollection`

        >>> import hyperspy.api as hs
        >>> from matplotlib.collections import StarPolygonCollection
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> data = np.ones((25, 25, 100, 100))
        >>> s = hs.signals.Signal2D(data)
        >>> offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
        >>> for ind in np.ndindex(offsets.shape):
        ...    offsets[ind] = rng.random((10, 2)) * 100
        >>> # every other star has a size of 50/100
        >>> m = hs.plot.markers.Markers(
        ...    collection=StarPolygonCollection,
        ...    offsets=offsets,
        ...    numsides=5,
        ...    color="orange",
        ...    sizes=(50, 100),
        ...    )
        >>> s.plot()
        >>> s.add_marker(m)

        Adding markers using PolyCollection using verts

        >>> from matplotlib.collections import PolyCollection
        >>> import hyperspy.api as hs
        >>> import numpy as np
        >>> m = hs.plot.markers.Markers(
        ...    collection=PolyCollection,
        ...    offsets=np.random.rand(10,2)*10,
        ...    verts=np.array([[0,0], [0,1], [1,1], [1,0]]),
        ...    color="red",
        ...    )
        >>>s = hs.signals.Signal2D(np.ones((10,10,10,10)))
        >>>s.plot()
        >>>s.add_marker(m)

        """
        if isinstance(collection, str):
            try:
                collection = getattr(mpl_collections, collection)
            except AttributeError:
                raise ValueError(
                    f"'{collection}' is not the name of a matplotlib collection class."
                 )

        if not issubclass(collection, mpl_collections.Collection):
            raise ValueError(
                f"{collection} is not a subclass of `matplotlib.collection.Collection`."
                )

        if ".".join(collection.__module__.split('.')[:2]) not in [
            'matplotlib.collections', 'hyperspy.external'
            ]:
            # To be able to load a custom markers, we need to be able to instantiate
            # the class and the safe way to do that is to import from
            # `matplotlib.collections` or `hyperspy.external` (patched matplotlib collection)
            raise ValueError(
                "To support loading file saved with custom markers, the collection must be "
                "implemented in matplotlib or hyperspy"
            )

        # Data attributes
        self.kwargs = kwargs  # all keyword arguments.
        self.axes_manager = None
        self.ax = None
        self.auto_update = True
        self._offset_transform = None
        self._transform = None
        if self._position_key_to_set is None:
            self._position_key_to_set = self._position_key

        # Handling dask arrays
        self.dask_kwargs = {}
        for key, value in self.kwargs.items():
            if isinstance(value, da.Array) and value.dtype == object:
                self.dask_kwargs[key] = self.kwargs[key]
            elif isinstance(value, da.Array):  # and value.dtype != object:
                self.kwargs[key] = value.compute()
            # Patches or verts shouldn't be cast to array
            elif (
                isinstance(value, list)
                and len(value) > 0
                and not isinstance(value[0], Patch)
                and not key == "verts"
            ):
                self.kwargs[key] = np.array(value)
            elif isinstance(value, list) and len(value) == 0:
                self.kwargs[key] = np.array(value)

            if key in ["sizes", "color"] and (
                not hasattr(value, "__len__") or isinstance(value, str)
            ):
                self.kwargs[key] = (value,)
            if key in ["s"] and isinstance(value, str):
                self.kwargs[key] = (value,)

        self._cache_dask_chunk_kwargs = {}
        self._cache_dask_chunk_kwargs_slice = {}

        self._class_name = self.__class__.__name__
        self.name = name
        # Properties
        self.collection = None
        # used in _initialize_collection
        self._collection_class = collection
        self._signal = None
        self._plot_on_signal = plot_on_signal
        self.shift = shift
        self.plot_marker = True
        self.offset_transform = offset_transform
        self.transform = transform

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

    @property
    def _is_iterating(self):
        if self._plot_on_signal:
            return np.any([is_iterating(value) for key, value in self.kwargs.items()])
        else:  # currently iterating navigation markers arenot supported
            return False

    def _get_transform(self, attr="_transform"):
        if self.ax is not None:  # return the transform
            transforms = {
                "data": self.ax.transData,
                "axes": self.ax.transAxes,
                "display": IdentityTransform(),
                "yaxis": self.ax.get_yaxis_transform(),
                "xaxis": self.ax.get_xaxis_transform(),
                "relative": self.ax.transData,
            }
            return transforms[getattr(self, attr)]
        else:  # return the string value
            return getattr(self, attr)

    def _set_transform(self, value, attr="_transform"):
        arg_list = ["data", "axes", "xaxis", "yaxis", "display", "relative"]
        if value not in arg_list:
            str_ = ", ".join([f"`{v}`" for v in arg_list])
            raise ValueError(
                f"The transform must be one of {str_}."
            )
        setattr(self, attr, value)
        if self.collection is not None and self.ax is not None:
            getattr(self.collection, f"set{attr}")(getattr(self, attr[1:]))
            # Update plot
            self.update()

    @property
    def offset_transform(self):
        return self._get_transform(attr="_offset_transform")

    @offset_transform.setter
    def offset_transform(self, value):
        self._set_transform(value, attr="_offset_transform")

    @property
    def transform(self):
        return self._get_transform(attr="_transform")

    @transform.setter
    def transform(self, value):
        self._set_transform(value, attr="_transform")

    def __len__(self):
        """
        Return the number of markers in the collection at the current
        navigation coordinate.
        """
        if self._is_iterating and self.axes_manager is None:
            # with variable length markers, the axes_manager is needed to
            # know the navigation coordinates of the signal
            raise RuntimeError(
                "Variable length markers must have been plotted to provide "
                "the numbers of markers at the current navigation coordinates."
                )

        return self.get_data_position()[self._position_key_to_set].shape[0]

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
        if self.name:
            text = '<%s (%s)' % (self.name, self.__class__.__name__)
        else:
            text = '<%s' % self.__class__.__name__

        text += ", length: "
        if self._is_iterating:
            lentgh = getattr(self, "__len__ ", "not plotted")
            text += "variable (current: %s)" % len(self)
        else:
            text += "%s"% len(self)
        text += ">"

        return text

    @classmethod
    def from_signal(
        cls,
        signal,
        key=None,
        signal_axes="metadata",
        collection=None,
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
            create the subclass of :py:class:`matplotlib.collections.Collection.
            Passed as {key: signal.data}.
        collection: None, str or subclass of :py:class:`matplotlib.collections.Collection`
            The collection which is initialized. If None, default to `Points` marker.
        signal_axes: str, tuple of UniformAxes or None
            If "metadata" look for signal_axes saved in metadata under .metadata.Peaks.signal_axes
            and convert from pixel positions to real units before creating the collection. If a tuple
            of signal axes those Axes will be used otherwise no transformation will
            happen.
        """
        if collection is None:
            # By default, use `Points` with "display" coordinate system to
            # avoid dependence on data coordinates
            from hyperspy.utils.markers import Points
            cls = Points
            kwargs.setdefault('sizes', 10)
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
                signal_axes=signal_axes,
            )
        else:
            raise ValueError(
                "The keyword argument `signal_axes` must be one of "
                "'metadata', a tuple of `DataAxes` or None."
            )

        if key is None:
            key = cls._position_key
        kwargs[key] = new_signal.data

        return cls(**kwargs)

    def _get_data_shape(self):
        for key, item in self.kwargs.items():
            if is_iterating(item):
                return item.shape
        return ()

    def __deepcopy__(self, memo):
        new_marker = markers_dict_to_markers(self._to_dictionary())
        return new_marker

    def _to_dictionary(self):
        class_name = self.__class__.__name__
        marker_dict = {
            "class": class_name,
            "name": self.name,
            "plot_on_signal": self._plot_on_signal,
            "offset_transform": self._offset_transform,
            "transform": self._transform,
            "kwargs": self.kwargs,
        }
        if class_name == "Markers":
            marker_dict['collection'] = self._collection_class.__name__

        return marker_dict

    def get_data_position(self, get_static_kwargs=True):
        """
        Return the current keyword arguments for updating the collection.
        """
        current_keys = {}
        if self._is_iterating:
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

        # Handling relative markers
        if "relative" in [self._offset_transform, self._transform]:
            # scale based on current data
            current_keys = self._scale_kwarg(current_keys)

        return current_keys

    def _scale_kwarg(self, kwds, key=None):
        """
        Scale the kwarg by the current data.  This is useful for scaling the
        marker position by the current index or data value.

        When self.reference is "data" the kwarg is scaled by the current data value of the
        "offset" or "segments" key

        When self.reference is "data_index" the kwarg is scaled by the current data value of the
        "offset" or "segments" key and the given value of the index.  This is useful when you want
        to scale things by some value in the data that is not the same value.
        """
        if key is None:
            key = self._position_key

        new_kwds = deepcopy(kwds)
        current_data = self._signal(as_numpy=True)
        x_positions = new_kwds[key][..., 0]
        ax = self.axes_manager.signal_axes[0]
        indexes = np.round((x_positions - ax.offset) / ax.scale).astype(int)
        y_positions = new_kwds[key][..., 1]
        new_y_positions = current_data[indexes] * y_positions

        if self.shift is not None:
            yrange = np.max(current_data) - np.min(current_data)
            new_y_positions = new_y_positions + self.shift * yrange

        new_kwds[key][..., 1] = new_y_positions

        return new_kwds

    def update(self):
        if self._is_iterating or "relative" in [self._offset_transform, self._transform]:
            kwds = self.get_data_position(get_static_kwargs=False)
            self.collection.set(**kwds)

    def _initialize_collection(self):
        if self.collection is None:
            self.collection = self._collection_class(
                **self.get_data_position(),
                offset_transform=self.offset_transform,
            )
            self.collection.set_transform(self.transform)

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
        self._signal = None
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
    is_key_iter = [
        isiterable(data[key]) and not isinstance(data[key], str) for key in unique_keys
    ]
    if not any(is_key_iter):  # no iterable keys
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


def markers_dict_to_markers(marker_dict):
    """
    This function maps a marker dict to a Markers object. It supports parsing
    old markers API, typically for file saved with hyperspy < 2.0.
    """
    # hyperspy 1.x markers uses `marker_type`, 2.x uses name
    markers_class = marker_dict.pop('class', marker_dict.pop('marker_type', None))
    if markers_class is None:
        raise ValueError("Not a valid marker dictionary.")

    kwargs = {
        # in hyperspy >= 2.0, all data and properties are in kwargs
        **marker_dict.pop("kwargs", {}),
        # in hyperspy < 2.0, "markers properties" are saved in `marker_properties`
        **marker_dict.pop("marker_properties", {})
        }
    # Parse old markers API: add relevant "data" to kwargs
    if "data" in marker_dict:
        if "Point" in markers_class:
            kwargs["offsets"], kwargs["sizes"] = dict2vector(
                marker_dict["data"], keys=None, return_size=True
                )
            markers_class = "Points"

        elif "HorizontalLine" in markers_class:
            kwargs["offsets"] = dict2vector(
                marker_dict["data"], keys=["y1"], return_size=False
                )
            markers_class = "HorizontalLines"

        elif "HorizontalLineSegment" in markers_class:
            kwargs["offsets"] = dict2vector(
                marker_dict["data"], keys=[[["x1", "y1"], ["x2", "y1"]]], return_size=False
                )
            markers_class = "Lines"

        elif "VerticalLine" in markers_class:
            kwargs["offsets"] = dict2vector(
                marker_dict["data"], keys=["x1"], return_size=False
                )
            markers_class = "VerticalLines"

        elif "VerticalLineSegment" in markers_class:
            kwargs["segments"] = dict2vector(
                marker_dict["data"], keys=[[["x1", "y1"], ["x1", "y2"]]], return_size=False
                )
            markers_class = "Lines"

        elif "Line" in markers_class:
            kwargs["segments"] = dict2vector(
                marker_dict["data"], keys=[[["x1", "y1"], ["x2", "y2"]]], return_size=False
                )
            markers_class = "Lines"

        elif "Arrow" in markers_class:
            # check if dx == x2 or dx == x2 - x1, etc.
            kwargs["offsets"] = dict2vector(
                marker_dict["data"], keys=[["x1", "y1"],], return_size=False
                )
            kwargs["U"] = dict2vector(
                marker_dict["data"], keys=[["x2"],], return_size=False
                )
            kwargs["V"] = dict2vector(
                marker_dict["data"], keys=[["y2"],], return_size=False
                )
            markers_class = "Arrows"

        elif "Rectangle" in markers_class:
            # check if dx == x2 or dx == x2 - x1, etc.
            kwargs["offsets"] = dict2vector(
                marker_dict["data"], keys=[["x1", "y1"], ], return_size=False
                )
            kwargs["widths"] = dict2vector(
                marker_dict["data"], keys=["x2"], return_size=False
                )
            kwargs["heights"] = dict2vector(
                marker_dict["data"], keys=["y2"], return_size=False
                )
            markers_class = "Rectangles"

        elif "Ellipse" in markers_class:
            kwargs["offsets"] = dict2vector(
                marker_dict["data"], keys=[["x1", "y1"], ], return_size=False
                )
            kwargs["widths"] = dict2vector(
                marker_dict["data"], keys=["x2"], return_size=False
                )
            kwargs["heights"] = dict2vector(
                marker_dict["data"], keys=["y2"], return_size=False
                )
            markers_class = "Ellipses"

        elif "Text" in markers_class:
            kwargs["offsets"] = dict2vector(
                marker_dict["data"], keys=[["x1", "y1"]], return_size=False
            )
            kwargs["texts"] = dict2vector(
                marker_dict["data"], keys=["text"], return_size=False, dtype=str
                )
            markers_class = "Texts"

        # remove "data" key:value
        del marker_dict["data"]

    return getattr(hyperspy.utils.markers, markers_class)(
        **marker_dict, **kwargs
        )
