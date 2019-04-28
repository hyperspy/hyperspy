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

import numpy as np
import matplotlib.pyplot as plt
from hyperspy.events import Event, Events
import hyperspy.drawing._markers as markers
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
        if self.ax.figure.canvas.supports_blit:
            self.ax.hspy_fig._update_animated()
        else:
            self.ax.figure.canvas.draw_idle()

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
    elif marker_type == 'Rectangle':
        marker = markers.rectangle.Rectangle(0, 0, 0, 0)
    elif marker_type == 'Text':
        marker = markers.text.Text(0, 0, "")
    elif marker_type == 'VerticalLine':
        marker = markers.vertical_line.VerticalLine(0)
    elif marker_type == 'VerticalLineSegment':
        marker = markers.vertical_line_segment.VerticalLineSegment(0, 0, 0)
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
