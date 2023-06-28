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
import hyperspy.drawing._markers as markers
from hyperspy.misc.utils import isiterable

import logging

_logger = logging.getLogger(__name__)


class MarkerBase(object):

    """Marker that can be added to the signal figure

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
        self.marker = None
        self._marker_properties = {}
        self.signal = None
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

    def __deepcopy__(self, memo):
        new_marker = dict2marker(
            self._to_dictionary(),
            self.name)
        return new_marker

    @property
    def marker_properties(self):
        return self._marker_properties

    @marker_properties.setter
    def marker_properties(self, kwargs):

        for key, item in kwargs.items():
            if item is None and key in self._marker_properties:
                del self._marker_properties[key]
            else:
                self._marker_properties[key] = item
        if self.marker is not None:
            plt.setp(self.marker, **self.marker_properties)
            self._render_figure()

    def _to_dictionary(self):
        marker_dict = {
            'marker_properties': self.marker_properties,
            'marker_type': self.__class__.__name__,
            'plot_on_signal': self._plot_on_signal,
            'data': {k: self.data[k][()].tolist() for k in (
                'x1', 'x2', 'y1', 'y2', 'text', 'size')}
        }
        return marker_dict

    def _get_data_shape(self):
        data_shape = None
        for key in ('x1', 'x2', 'y1', 'y2'):
            ar = self.data[key][()]
            if next(ar.flat) is not None:
                data_shape = ar.shape
                break
        if data_shape is None:
            raise ValueError("None of the coordinates have value")
        else:
            return data_shape

    def set_marker_properties(self, **kwargs):
        """
        Set the line_properties attribute using keyword
        arguments.
        """
        self.marker_properties = kwargs

    def set_data(self, x1=None, y1=None,
                 x2=None, y2=None, text=None, size=None):
        """
        Set data to the structured array. Each field of data should have
        the same dimensions than the navigation axes. The other fields are
        overwritten.
        """
        self.data = np.array((np.array(x1), np.array(y1),
                              np.array(x2), np.array(y2),
                              np.array(text), np.array(size)),
                             dtype=[('x1', object), ('y1', object),
                                    ('x2', object), ('y2', object),
                                    ('text', object), ('size', object)])
        self._is_marker_static()

    def add_data(self, **kwargs):
        """
        Add data to the structured array. Each field of data should have
        the same dimensions than the navigation axes. The other fields are
        not changed.
        """
        if self.data is None:
            self.set_data(**kwargs)
        else:
            for key in kwargs.keys():
                self.data[key][()] = np.array(kwargs[key])
        self._is_marker_static()

    def isiterable(self, obj):
        return not isinstance(obj, (str, bytes)) and hasattr(obj, '__iter__')

    def _is_marker_static(self):

        test = [self.isiterable(self.data[key].item()[()]) is False
                for key in self.data.dtype.names]
        if np.alltrue(test):
            self.auto_update = False
        else:
            self.auto_update = True

    def get_data_position(self, ind):
        data = self.data
        if data[ind].item()[()] is None:
            return None
        elif self.isiterable(data[ind].item()[()]) and self.auto_update:
            if self.axes_manager is None:
                return self.data['x1'].item().flatten()[0]
            indices = self.axes_manager.indices[::-1]
            return data[ind].item()[indices]
        else:
            return data[ind].item()[()]

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

# markers are imported in hyperspy.utils.markers
def dict2marker(marker_dict, marker_name):
    marker_type = marker_dict['marker_type']
    if marker_type == 'Point':
        marker = markers.point.Point(0, 0)
    elif marker_type == 'HorizontalLine':
        marker = markers.horizontal_line.HorizontalLine(0)
    elif marker_type == 'HorizontalLineSegment':
        marker = markers.horizontal_line_segment.HorizontalLineSegment(0, 0, 0)
    elif marker_type == 'LineSegment':
        marker = markers.line_segment.LineSegment(0, 0, 0, 0)
    elif marker_type == 'Arrow':
        marker = markers.arrow.Arrow(0, 0, 0, 0)
    elif marker_type == 'Rectangle':
        marker = markers.rectangle.Rectangle(0, 0, 0, 0)
    elif marker_type == 'Ellipse':
        marker = markers.ellipse.Ellipse(0, 0, 0, 0)
    elif marker_type == 'Text':
        marker = markers.text.Text(0, 0, "")
    elif marker_type == 'VerticalLine':
        marker = markers.vertical_line.VerticalLine(0)
    elif marker_type == 'VerticalLineSegment':
        marker = markers.vertical_line_segment.VerticalLineSegment(0, 0, 0)
    elif marker_type == 'MarkerCollection':
        marker = markers.marker_collection.MarkerCollection(collection_class=marker_dict["collection_class"],
                                                            **marker_dict["kwargs"])
        marker._plot_on_signal = marker_dict['plot_on_signal']
        return(marker)
    else:
        _log = logging.getLogger(__name__)
        _log.warning(
            "Marker {} with marker type {} "
            "not recognized".format(marker_name, marker_type))
        return(False)
    marker.set_data(**marker_dict['data'])
    marker.set_marker_properties(**marker_dict['marker_properties'])
    marker._plot_on_signal = marker_dict['plot_on_signal']
    marker.name = marker_name
    return(marker)


def dict2vector(data,
                keys=None,
                return_size=True):
    """Take some dictionary of values and create offsets based on the input keys.
    For instances like creating a horizontal or vertical line then some key is duplicated.

    Multiple keys can be passed as well. For example to define a rectangle:

    >>> dict2offsets(data,keys= [['x1','y1'], ['x2','y1'], ['x2', 'y2'],['x1', 'y2']])

    In this example the keys will be unpacked to create a rectangle.
    """
    if keys is None:
        keys = [["x1, x2"]]
    keys = np.array(keys)
    # check to see if the array should be ragged
    unique_keys = np.unique(keys)
    is_key_iter = [isiterable(data[key]) for key in unique_keys]
    if not any(is_key_iter):  # no iterable keys
        vector = np.empty(keys.shape)
        for i in np.ndindex(keys.shape): # iterate through keys and create resulting vector
            vector[i] = data[keys[i]]
    else:
        iter_key = unique_keys[is_key_iter][0]
        nav_shape = data[iter_key].shape
        if not all(is_key_iter):  # only some values are iterating
            non_iterkeys = unique_keys[np.logical_not(is_key_iter)]
            for k in non_iterkeys:
                data[k] = np.full(shape=nav_shape, fill_value=data[k])
        vector = np.empty(nav_shape, dtype=object)  # Create ragged array
        for i in np.ndindex(nav_shape):
            vect = np.empty(keys.shape)
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
    """This function maps a maker dict to a MarkerCollection class
    """
    from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
    from matplotlib.patches import FancyArrowPatch, Ellipse
    from hyperspy.drawing._markers.line_collection import VerticalLineCollection, HorizontalLineCollection

    marker_type = marker_dict["marker_type"]
    if marker_type == 'Point':
        offsets, size = dict2vector(marker_dict["data"],
                                    keys=[["x1", "y1"]],
                                    return_size=True)
        marker = markers.marker_collection.MarkerCollection(offsets=offsets,
                                                            sizes=size,
                                                            **marker_dict['marker_properties'])
    elif marker_type == 'HorizontalLine':
        segments = dict2vector(marker_dict["data"],
                               keys=["y1"], return_size=False)

        marker = HorizontalLineCollection(segments=segments,
                                          **marker_dict['marker_properties'])

    elif marker_type == 'HorizontalLineSegment':
        segments = dict2vector(marker_dict["data"],
                               keys=[[["x1", "y1"], ["x2", "y1"]]], return_size=False)

        marker = markers.marker_collection.MarkerCollection(segments=segments,
                                                            collection_class=LineCollection,
                                                            **marker_dict['marker_properties'])
    elif marker_type == 'LineSegment':
        segments = dict2vector(marker_dict["data"],
                               keys=[[["x1", "y1"], ["x2", "y2"]]], return_size=False)

        marker = markers.marker_collection.MarkerCollection(segments=segments,
                                                            collection_class=LineCollection,
                                                            **marker_dict['marker_properties'])
    elif marker_type == 'Arrow':
        segments = dict2vector(marker_dict["data"],
                               keys=[["x1", "y1"], ["x2", "y2"]], return_size=False)
        if segments.dtype == object:
            arrows = np.empty_like(segments, dtype=object)
            for i in np.ndindex(segments.shape):
                arrows[i] = [FancyArrowPatch(posA=segments[i][0], posB=segments[i][1],
                                            **marker_dict['marker_properties']),]
        else:
            arrows = [FancyArrowPatch(posA=segments[0], posB=segments[1],
                                      **marker_dict['marker_properties']),
                      ]
        marker = markers.marker_collection.MarkerCollection(patches=arrows,
                                                            collection_class=PatchCollection)

    elif marker_type == 'Rectangle':
        verts = dict2vector(marker_dict["data"],
                               keys=[[["x1", "y1"], ["x1", "y2"],
                                     ["x2", "y1"], ["x2", "y2"]]],
                               return_size=False,
                               )

        marker = markers.marker_collection.MarkerCollection(verts=verts,
                                                            collection_class=PolyCollection,
                                                            **marker_dict['marker_properties'])
    elif marker_type == 'Ellipse':
        segments = dict2vector(marker_dict["data"],
                               keys=[["x1", "y1"], ["x2", "y2"]], return_size=False)
        if segments.dtype == object:
            ellipses = np.empty_like(segments, dtype=object)
            for i in np.ndindex(segments.shape):
                ellipses[i] = [Ellipse(xy=segments[i][0], width=segments[i][1][0],
                                       height=segments[i][1][1],
                                       **marker_dict['marker_properties']), ]
        else:
            ellipses = [Ellipse(xy=segments[0], width=segments[1][0],
                                height=segments[1][1],
                                **marker_dict['marker_properties']), ]
        marker = markers.marker_collection.MarkerCollection(patches=ellipses,
                                                            collection_class=PatchCollection)
    elif marker_type == 'Text':
        raise ValueError("Converting from Text to a Marker Collection is not supported"
                         "as there is no MarkerCollection which can render text")
    elif marker_type == 'VerticalLine':
        segments = dict2vector(marker_dict["data"],
                               keys=["x1"], return_size=False)

        marker = VerticalLineCollection(segments=segments,
                                        **marker_dict['marker_properties'])
    elif marker_type == 'VerticalLineSegment':
        segments = dict2vector(marker_dict["data"],
                               keys=[[["x1", "y1"], ["x1", "y2"]]],
                               return_size=False)

        marker = markers.marker_collection.MarkerCollection(segments=segments,
                                                            collection_class=LineCollection,
                                                            **marker_dict['marker_properties'])
    else:
        raise ValueError(f"The marker_type: {marker_type} is not a hyperspy.marker class "
                         f"and cannot be converted to a MarkerCollection")

    return marker

def markers_metadata_dict_to_markers(metadata_markers_dict, axes_manager):
    markers_dict = {}
    for marker_name, m_dict in metadata_markers_dict.items():
        try:
            marker = dict2marker(m_dict, marker_name)
            if marker is not False:
                marker.axes_manager = axes_manager
                markers_dict[marker_name] = marker
        except Exception as expt:
            _logger.warning(
                "Marker {} could not be loaded, skipping it. "
                "Error: {}".format(marker_name, expt))
    return(markers_dict)
