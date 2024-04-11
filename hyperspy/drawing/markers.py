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
from copy import deepcopy

import dask.array as da
import matplotlib.collections as mpl_collections
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import IdentityTransform

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
    """A set of markers using Matplotlib collections."""

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
        ScalarMappable_array=None,
        **kwargs,
    ):
        """
        The markers are defined by a set of arugment required by the collections,
        typically, ``offsets``, ``verts`` or ``segments`` will define their
        positions.

        To define a non-static marker any argument that can be set with the
        :meth:`matplotlib.collections.Collection.set` method can be passed
        as an array with `dtype=object` of the constructor and the same size as
        the navigation axes of the a signal the markers will be added to.

        Parameters
        ----------
        collection : matplotlib.collections.Collection or str
            A Matplotlib collection to be initialized.
        offset_transform, transform : str
            ``offset_transform`` define the transformation used for the
            `offsets`` value  and ``tranform`` define the transformation for
            other arguments, typically to scale the size of the ``Path``.
            It can be one of the following:

            - ``"data"``: the offsets are defined in data coordinates and the ``ax.transData`` transformation is used.
            - ``"relative"``: The offsets are defined in data coordinates in x and coordinates in y relative to the
              data plotted. Only for 1D figure.
            - ``"axes"``: the offsets are defined in axes coordinates and the ``ax.transAxes`` transformation is used.
              (0, 0) is bottom left of the axes, and (1, 1) is top right of the axes.
            - ``"xaxis"``: The offsets are defined in data coordinates in x and axes coordinates in y direction; use
              :meth:`matplotlib.axes.Axes.get_xaxis_transform` transformation.
            - ``"yaxis"``: The offsets are defined in data coordinates in y and axes coordinates in x direction; use
              :meth:`matplotlib.axes.Axes.get_xaxis_transform` transformation..
            - ``"display"``: the offsets are not transformed, i.e. are defined in the display coordinate system.
              (0, 0) is the bottom left of the window, and (width, height) is top right of the output in "display units"
              :class:`matplotlib.transforms.IdentityTransform`.

        shift : None or float
            Only for ``offset_transform="relative"``. This applied a systematic
            shift in the y component of the ``offsets`` values. The shift is
            defined in the matplotlib ``"axes"`` coordinate system.
            This provides a constant shift from the data for labeling
            :class:`~.api.signals.Signal1D`.
        plot_on_signal : bool
            If True, plot on signal figure, otherwise on navigator.
        name : str
            The name of the markers.
        ScalarMappable_array : Array-like
            Set the array of the :class:`matplotlib.cm.ScalarMappable` of the
            matplotlib collection.
            The ``ScalarMappable`` array will overwrite ``facecolor`` and
            ``edgecolor``. Default is None.
        **kwargs : dict
            Keyword arguments passed to the underlying marker collection. Any argument
            that is array-like and has ``dtype=object`` is assumed to be an iterating
            argument and is treated as such.

        Examples
        --------
        Add markers using a :class:`matplotlib.collections.PatchCollection`
        which will display the specified subclass of :class:`matplotlib.patches.Patch`
        at the position defined by the argument ``offsets`` .

        >>> from matplotlib.collections import PatchCollection
        >>> from matplotlib.patches import Circle

        >>> m = hs.plot.markers.Markers(
        ...    collection=PatchCollection,
        ...    patches=[Circle((0, 0), 1)],
        ...    offsets=np.random.rand(10,2)*10,
        ...    )
        >>> s = hs.signals.Signal2D(np.ones((10, 10, 10, 10)))
        >>> s.plot()
        >>> s.add_marker(m)

        Adding star "iterating" markers using :class:`matplotlib.collections.StarPolygonCollection`

        >>> from matplotlib.collections import StarPolygonCollection
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> data = np.ones((25, 25, 100, 100))
        >>> s = hs.signals.Signal2D(data)
        >>> offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
        >>> for ind in np.ndindex(offsets.shape):
        ...    offsets[ind] = rng.random((10, 2)) * 100

        Every other star has a size of 50/100

        >>> m = hs.plot.markers.Markers(
        ...    collection=StarPolygonCollection,
        ...    offsets=offsets,
        ...    numsides=5,
        ...    color="orange",
        ...    sizes=(50, 100),
        ...    )
        >>> s.plot()
        >>> s.add_marker(m)

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

        if ".".join(collection.__module__.split(".")[:2]) not in [
            "matplotlib.collections",
            "hyperspy.external",
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
        self.ax = None
        self._offset_transform = None
        self._transform = None
        if self._position_key_to_set is None:
            self._position_key_to_set = self._position_key
        # The list of keys of iterable argument other than the "_position_key"
        self._iterable_argument_keys = []

        self.dask_kwargs = {}
        for key, value in self.kwargs.items():
            # Populate `_iterable_argument_keys`
            if (
                isiterable(value)
                and not isinstance(value, str)
                and key != self._position_key
            ):
                self._iterable_argument_keys.append(key)

            # Handling dask arrays
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

        self._cache_dask_chunk_kwargs = {}
        self._cache_dask_chunk_kwargs_slice = {}

        self._class_name = self.__class__.__name__
        self.name = name
        # Properties
        self._collection = None
        # used in _initialize_collection
        self._collection_class = collection
        self._signal = None
        self._plot_on_signal = plot_on_signal
        self.shift = shift
        self.offset_transform = offset_transform
        self.transform = transform
        self._ScalarMappable_array = ScalarMappable_array

        # Events
        self.events = Events()
        self.events.closed = Event(
            """
            Event triggered when a marker is closed.

            Parameters
            ----------
            marker : Marker
                The marker that was closed.
            """,
            arguments=["obj"],
        )
        self._closing = False

    @property
    def _axes_manager(self):
        if self._signal is not None:
            return self._signal.axes_manager
        else:
            return None

    @property
    def _is_iterating(self):
        if self._plot_on_signal:
            return np.any([is_iterating(value) for key, value in self.kwargs.items()])
        else:  # currently iterating navigation markers are not supported
            return False

    @property
    def _signal(self):
        return self.__signal

    @_signal.setter
    def _signal(self, signal):
        if signal is not None:
            for key, value in self.kwargs.items():
                nav_shape = value.shape if is_iterating(value) else ()
                if (
                    len(nav_shape) != 0
                    and nav_shape != signal.axes_manager.navigation_shape
                ):
                    raise ValueError(
                        "The shape of the variable length argument must match "
                        "the navigation shape of the signal."
                    )
        self.__signal = signal

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
            raise ValueError(f"The transform must be one of {str_}.")
        setattr(self, attr, value)
        if self._collection is not None and self.ax is not None:
            getattr(self._collection, f"set{attr}")(getattr(self, attr[1:]))
            # Update plot
            self.update()

    @property
    def offset_transform(self):
        """The tranform being used for the ``offsets`` values."""
        return self._get_transform(attr="_offset_transform")

    @offset_transform.setter
    def offset_transform(self, value):
        self._set_transform(value, attr="_offset_transform")

    @property
    def transform(self):
        """
        The tranform being used for the values other than ``offsets``,
        typically ``sizes``, etc.
        """
        return self._get_transform(attr="_transform")

    @transform.setter
    def transform(self, value):
        self._set_transform(value, attr="_transform")

    def __len__(self):
        """
        Return the number of markers in the collection at the current
        navigation coordinate.
        """
        if self._is_iterating and self._axes_manager is None:
            # with variable length markers, the axes_manager is needed to
            # know the navigation coordinates of the signal
            raise RuntimeError(
                "Variable length markers must be added to a signal to provide "
                "the numbers of markers at the current navigation coordinates."
            )

        return self.get_current_kwargs()[self._position_key_to_set].shape[0]

    def remove_items(self, indices, keys=None, navigation_indices=None):
        """
        Remove items from the markers.

        Parameters
        ----------
        indices : slice, int or numpy.ndarray
            Indicate indices of sub-arrays to remove along the specified axis.
        keys : str, list of str or None
            Specify the key of the ``Markers.kwargs`` to remove. If ``None``,
            use all compatible keys. Default is ``None``.
        navigation_indices : tuple
            Only for variable-lenght markers. If ``None``, remove for all
            navigation coordinates.

        Examples
        --------
        Remove a single item:

        >>> offsets = np.array([[1, 1], [2, 2]])
        >>> m = hs.plot.markers.Points(offsets=offsets)
        >>> print(m)
        <Points, length: 2>
        >>> m.remove_items(indices=(1, ))
        >>> print(len(m))
        1

        Remove a single item at specific navigation position for variable
        length markers:

        >>> offsets = np.empty(4, dtype=object)
        >>> texts = np.empty(4, dtype=object)
        >>> for i in range(len(offsets)):
        ...    offsets[i] = np.array([[1, 1], [2, 2]])
        ...    texts[i] = ['a' * (i+1)] * 2
        >>> m = hs.plot.markers.Texts(offsets=offsets, texts=texts)
        >>> m.remove_items(1, navigation_indices=(1, ))

        Remove several items:

        >>> offsets = np.stack([np.arange(0, 100, 10)]*2).T + np.array([5,]*2)
        >>> texts = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'f', 'h', 'i'])
        >>> m = hs.plot.markers.Texts(offsets=offsets, texts=texts)
        >>> print(m)
        <Texts, length: 10>
        >>> m.remove_items(indices=[1, 2])
        >>> print(m)
        <Texts, length: 8>

        """
        if keys is None:
            keys = self._iterable_argument_keys + [self._position_key]
            # keeps value actually in kwargs
        elif isinstance(keys, str):
            keys = [
                keys,
            ]

        if navigation_indices and not self._is_iterating:
            raise ValueError(
                "`navigation_indices` is only for variable length markers."
            )

        for key in keys:
            value = self.kwargs[key]
            # Don't remove when it doesn't have the same length as the
            # position kwargs because it is a "cycling" argument
            if (
                isiterable(value)
                and not isinstance(value, str)
                and len(value) == len(self.kwargs[self._position_key])
            ):
                if isinstance(value, np.ndarray) and value.dtype == object:
                    # when navigation_indices is not provided
                    nav_iterator = navigation_indices or np.ndindex(
                        self.kwargs[self._position_key].shape
                    )
                    for nav_indices in nav_iterator:
                        self.kwargs[key][nav_indices] = np.delete(
                            value[nav_indices], indices, axis=0
                        )
                else:
                    self.kwargs[key] = np.delete(value, indices, axis=0)
        self._update()

    def add_items(self, navigation_indices=None, **kwargs):
        """
        Add items to the markers.

        Parameters
        ----------
        navigation_indices : tuple or None
            Only for variable-lenght markers. If ``None``, all for all
            navigation coordinates.

        **kwargs : dict
            Mapping of keys:values to add to the markers

        Examples
        --------
        Add a single item:

        >>> offsets = np.array([[1, 1], [2, 2]])
        >>> texts = np.array(["a", "b"])
        >>> m = hs.plot.markers.Texts(offsets=offsets, texts=texts)
        >>> print(m)
        <Texts, length: 2>
        >>> m.add_items(offsets=np.array([[0, 1]]), texts=["c"])
        >>> print(m)
        <Texts, length: 3>

        Add a single item at a defined navigation position of variable
        length markers:

        >>> offsets = np.empty(4, dtype=object)
        >>> texts = np.empty(4, dtype=object)
        >>> for i in range(len(offsets)):
        ...    offsets[i] = np.array([[1, 1], [2, 2]])
        ...    texts[i] = ['a' * (i+1)] * 2
        >>> m = hs.plot.markers.Texts(offsets=offsets, texts=texts)
        >>> m.add_items(
        ...    offsets=np.array([[0, 1]]), texts=["new_text"],
        ...    navigation_indices=(1, )
        ...    )
        """

        if navigation_indices and not self._is_iterating:
            raise ValueError(
                "`navigation_indices` is only for variable length markers."
            )

        for key, value in kwargs.items():
            if self.kwargs[key].dtype == object:
                nav_iterator = navigation_indices or np.ndindex(
                    self.kwargs[self._position_key].shape
                )
                for nav_indices in nav_iterator:
                    self.kwargs[key][nav_indices] = np.append(
                        self.kwargs[key][nav_indices], value, axis=0
                    )
            else:
                self.kwargs[key] = np.append(self.kwargs[key], value, axis=0)
        self._update()

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
            text = "<%s (%s)" % (self.name, self.__class__.__name__)
        else:
            text = "<%s" % self.__class__.__name__

        text += ", length: "
        if self._is_iterating:
            try:
                current = len(self)
            except RuntimeError:
                current = "not plotted"
            text += "variable (current: %s)" % current
        else:
            text += "%s" % len(self)
        text += ">"

        return text

    @classmethod
    def from_signal(
        cls,
        signal,
        key=None,
        signal_axes="metadata",
        **kwargs,
    ):
        """
        Initialize a marker collection from a hyperspy Signal.

        Parameters
        ----------
        signal: :class:`~.api.signals.BaseSignal`
            A value passed to the Collection as ``{key:signal.data}``
        key: str or None
            The key used to create a key value pair to create the subclass of
            :class:`matplotlib.collections.Collection`. If ``None`` (default)
            the key is set to ``"offsets"``.
        signal_axes: str, tuple of :class:`~.axes.UniformDataAxis` or None
            If ``"metadata"`` look for signal_axes saved in metadata under
            ``s.metadata.Peaks.signal_axes`` and convert from pixel positions
            to real units before creating the collection. If a ``tuple`` of
            signal axes, those axes will be used otherwise (``None``)
            no transformation will happen.
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
                signal_axes=signal_axes,
            )
        else:
            raise ValueError(
                "The keyword argument `signal_axes` must be one of "
                "'metadata', a tuple of `DataAxes` or None."
            )

        if key is None:
            key = cls._position_key

        # in case ragged array, we need to take the transpose to match the
        # navigation shape of the signal, for static marker, there is no
        # array dimention match the signal dimension and there is no
        # navigation dimension, therefore it shouldn't be transposed
        kwargs[key] = new_signal.data.T if new_signal.ragged else new_signal.data

        return cls(**kwargs)

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
            "ScalarMappable_array": self._ScalarMappable_array,
        }
        if class_name == "Markers":
            marker_dict["collection"] = self._collection_class.__name__

        return marker_dict

    def get_current_kwargs(self, only_variable_length=False):
        """
        Return the current keyword arguments for updating the collection.

        Parameters
        ----------
        only_variable_length : bool
            If ``True``, only returns the variable length kwargs. Default is
            ``False``.

        Returns
        -------
        kwargs : dict
            The argument at the current navigation position.
        """
        current_keys = {}
        if self._is_iterating:
            indices = self._axes_manager.indices
            for key, value in self.kwargs.items():
                if is_iterating(value):
                    if key not in self.dask_kwargs:
                        val = value[indices]
                        # some keys values need to iterate
                        if key in ["sizes", "color"] and not hasattr(val, "__len__"):
                            val = (val,)
                        current_keys[key] = val
                elif not only_variable_length:
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

        x_positions = kwds[key][..., 0]
        if len(x_positions) == 0:
            # can't scale as there is no marker at this coordinate
            return kwds

        new_kwds = deepcopy(kwds)
        current_data = self._signal._get_current_data(as_numpy=True)
        axis = self._axes_manager.signal_axes[0]
        indexes = np.round((x_positions - axis.offset) / axis.scale).astype(int)
        y_positions = new_kwds[key][..., 1]
        new_y_positions = current_data[indexes] * y_positions

        if self.shift is not None:
            yrange = np.max(current_data) - np.min(current_data)
            new_y_positions = new_y_positions + self.shift * yrange

        new_kwds[key][..., 1] = new_y_positions

        return new_kwds

    def update(self):
        """Update the markers on the plot."""
        if self._is_iterating or "relative" in [
            self._offset_transform,
            self._transform,
        ]:
            self._update()

    def _update(self):
        if self._signal:
            kwds = self.get_current_kwargs(only_variable_length=True)
            self._collection.set(**kwds)

    def _initialize_collection(self):
        self._collection = self._collection_class(
            **self.get_current_kwargs(),
            offset_transform=self.offset_transform,
        )
        self._collection.set_transform(self.transform)

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
        self._collection.set_animated(self.ax.figure.canvas.supports_blit)
        self.ax.add_collection(self._collection)
        if render_figure:
            self._render_figure()

    def _render_figure(self):
        self.ax.hspy_fig.render_figure()

    def close(self, render_figure=True):
        """
        Remove and disconnect the marker.

        Parameters
        ----------
        render_figure : bool, optional, default True
            If True, the figure is rendered after removing the marker.
            If False, the figure is not rendered after removing the marker.
            This is useful when many markers are removed from a figure,
            since rendering the figure after removing each marker will slow
            things down.
        """
        if self._closing:  # pragma: no cover
            return
        self._closing = True
        self._collection.remove()
        self._collection = None
        self.events.closed.trigger(obj=self)
        self._signal = None
        for f in self.events.closed.connected:
            self.events.closed.disconnect(f)
        if render_figure:
            self._render_figure()
        self._closing = False

    def set_ScalarMappable_array(self, array):
        """
        Set the array of the :class:`matplotlib.cm.ScalarMappable` of the
        matplotlib collection.
        The ``ScalarMappable`` array will overwrite ``facecolor`` and
        ``edgecolor``.

        Parameters
        ----------
        array : array-like
            The value that are mapped to the colors.

        See Also
        --------
        plot_colorbar
        """
        self._ScalarMappable_array = array
        if self._collection is not None:
            self._collection.set_array(array)

    def plot_colorbar(self):
        """
        Add a colorbar for the collection.

        Returns
        -------
        matplotlib.colorbar.Colorbar
            The colorbar of the collection.

        See Also
        --------
        set_ScalarMappable_array

        Examples
        --------
        >>> rng = np.random.default_rng(0)
        >>> s = hs.signals.Signal2D(np.ones((100, 100)))
        >>> # Define the size of the circles
        >>> sizes = rng.random((10, )) * 10 + 20
        >>> # Define the position of the circles
        >>> offsets = rng.random((10, 2)) * 100
        >>> m = hs.plot.markers.Circles(
        ...    sizes=sizes,
        ...    offsets=offsets,
        ...    linewidth=2,
        ...    )
        >>> s.plot()
        >>> s.add_marker(m)
        >>> m.set_ScalarMappable_array(sizes.ravel() / 2)
        >>> cbar = m.plot_colorbar()
        >>> cbar.set_label('Circle radius')
        """
        if self.ax is None:
            raise RuntimeError("The markers needs to be plotted.")
        self.set_ScalarMappable_array(self._ScalarMappable_array)
        cbar = self.ax.figure.colorbar(self._collection)

        return cbar


def is_iterating(arg):
    return isinstance(arg, (np.ndarray, da.Array)) and arg.dtype == object


def dict2vector(data, keys, return_size=True, dtype=float):
    """Take some dictionary of values and create offsets based on the input keys.
    For instances like creating a horizontal or vertical line then some key is duplicated.

    Multiple keys can be passed as well. For example to define a rectangle:

    >>> dict2vector(data, keys= [[["x1", "y1"], ["x2", "y2"]]]) # doctest: +SKIP

    In this example the keys will be unpacked to create a line segment
    """
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
    markers_class = marker_dict.pop("class", marker_dict.pop("marker_type", None))
    if markers_class is None:
        raise ValueError("Not a valid marker dictionary.")

    kwargs = {
        # in hyperspy >= 2.0, all data and properties are in kwargs
        **marker_dict.pop("kwargs", {}),
        # in hyperspy < 2.0, "markers properties" are saved in `marker_properties`
        **marker_dict.pop("marker_properties", {}),
    }
    # Parse old markers API: add relevant "data" to kwargs
    if "data" in marker_dict:
        if "Point" in markers_class:
            kwargs["offsets"], kwargs["sizes"] = dict2vector(
                marker_dict["data"], keys=["x1", "y1"], return_size=True
            )
            kwargs["facecolors"] = kwargs["color"]
            kwargs["units"] = "dots"
            if "size" not in kwargs:
                kwargs["size"] = 20
            kwargs["size"] = kwargs["size"] / np.pi
            markers_class = "Points"
        elif "HorizontalLineSegment" in markers_class:
            kwargs["segments"] = dict2vector(
                marker_dict["data"],
                keys=[[["x1", "y1"], ["x2", "y1"]]],
                return_size=False,
            )
            markers_class = "Lines"

        elif "HorizontalLine" in markers_class:
            kwargs["offsets"] = dict2vector(
                marker_dict["data"], keys=["y1"], return_size=False
            )
            markers_class = "HorizontalLines"

        elif "VerticalLineSegment" in markers_class:
            kwargs["segments"] = dict2vector(
                marker_dict["data"],
                keys=[[["x1", "y1"], ["x1", "y2"]]],
                return_size=False,
            )
            markers_class = "Lines"
        elif "VerticalLine" in markers_class:
            kwargs["offsets"] = dict2vector(
                marker_dict["data"], keys=["x1"], return_size=False
            )
            markers_class = "VerticalLines"
        elif "Line" in markers_class:
            kwargs["segments"] = dict2vector(
                marker_dict["data"],
                keys=[[["x1", "y1"], ["x2", "y2"]]],
                return_size=False,
            )
            markers_class = "Lines"

        elif "Arrow" in markers_class:
            # check if dx == x2 or dx == x2 - x1, etc.
            vectors = dict2vector(
                marker_dict["data"], keys=["x1", "y1", "x2", "y2"], return_size=False
            )
            if vectors.dtype == object:
                offsets = np.empty(vectors.shape, dtype=object)
                U = np.empty(vectors.shape, dtype=object)
                V = np.empty(vectors.shape, dtype=object)
                for i in np.ndindex(vectors.shape):
                    offsets[i] = np.array(
                        [
                            [vectors[i][0], vectors[i][1]],
                        ]
                    )
                    U[i] = np.array([vectors[i][0] - vectors[i][2]])
                    V[i] = np.array([vectors[i][1] - vectors[i][3]])
            else:
                offsets = np.array(
                    [
                        [vectors[0], vectors[1]],
                    ]
                )
                U = np.array([vectors[2] - vectors[0]])
                V = np.array([vectors[3] - vectors[1]])

            kwargs["offsets"] = offsets
            kwargs["U"] = U
            kwargs["V"] = V
            markers_class = "Arrows"

        elif "Rectangle" in markers_class:
            # check if dx == x2 or dx == x2 - x1, etc.
            vectors = dict2vector(
                marker_dict["data"], keys=["x1", "y1", "x2", "y2"], return_size=False
            )
            if vectors.dtype == object:
                offsets = np.empty(vectors.shape, dtype=object)
                widths = np.empty(vectors.shape, dtype=object)
                heights = np.empty(vectors.shape, dtype=object)
                for i in np.ndindex(vectors.shape):
                    offsets[i] = [
                        [
                            (vectors[i][0] + vectors[i][2]) / 2,
                            (vectors[i][1] + vectors[i][3]) / 2,
                        ],
                    ]
                    widths[i] = [
                        np.abs(vectors[i][0] - vectors[i][2]),
                    ]
                    heights[i] = [
                        np.abs(vectors[i][1] - vectors[i][3]),
                    ]
            else:
                offsets = [
                    [((vectors[0] + vectors[2]) / 2), ((vectors[1] + vectors[3]) / 2)],
                ]
                widths = [
                    np.abs(vectors[0] - vectors[2]),
                ]
                heights = [
                    np.abs(vectors[1] - vectors[3]),
                ]
            kwargs["offsets"] = offsets
            kwargs["widths"] = widths
            kwargs["heights"] = heights
            fill = kwargs.pop("fill")
            if not fill:
                kwargs["facecolor"] = "none"
            markers_class = "Rectangles"

        elif "Ellipse" in markers_class:
            kwargs["offsets"] = dict2vector(
                marker_dict["data"],
                keys=[
                    ["x1", "y1"],
                ],
                return_size=False,
            )
            kwargs["widths"] = dict2vector(
                marker_dict["data"], keys=["x2"], return_size=False
            )
            kwargs["heights"] = dict2vector(
                marker_dict["data"], keys=["y2"], return_size=False
            )
            fill = kwargs.pop("fill")
            if not fill:
                kwargs["facecolor"] = "none"
            markers_class = "Ellipses"

        elif "Text" in markers_class:
            kwargs["offsets"] = dict2vector(
                marker_dict["data"], keys=[["x1", "y1"]], return_size=False
            )
            kwargs["texts"] = dict2vector(
                marker_dict["data"], keys=["text"], return_size=False, dtype=str
            )
            kwargs["verticalalignment"] = "bottom"
            kwargs["horizontalalignment"] = "left"
            markers_class = "Texts"

        # remove "data" key:value
        del marker_dict["data"]
    if "size" in kwargs:
        kwargs["sizes"] = kwargs.pop("size")

    return getattr(hyperspy.utils.markers, markers_class)(**marker_dict, **kwargs)
