# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

import matplotlib.pyplot as plt

from hyperspy.drawing.marker import MarkerBase


class VerticalLineSegment(MarkerBase):

    """Vertical line segment marker that can be added to the signal figure

    Parameters
    ----------
    x : array or float
        The position of line segment in x.
        If float, the marker is fixed.
        If array, the marker will be updated when navigating. The array should
        have the same dimensions in the navigation axes.
    y1 : array or float
        The position of the start of the line segment in x.
        see x1 arguments
    y2 : array or float
        The position of the start of the line segment in y.
        see x1 arguments
    kwargs :
        Keyword arguments are passed to
        :py:meth:`matplotlib.axes.Axes.axvline`.

    Example
    -------
    >>> im = hs.signals.Signal2D(np.zeros((100, 100)))
    >>> m = hs.plot.markers.vertical_line_segment(
    >>>     x=20, y1=30, y2=70, linewidth=4, color='red', linestyle='dotted')
    >>> im.add_marker(m)

    Add a marker permanently to a marker

    >>> im = hs.signals.Signal2D(np.zeros((60, 60)))
    >>> m = hs.plot.markers.vertical_line_segment(x=10, y1=20, y2=50)
    >>> im.add_marker(m, permanent=True)
    """

    def __init__(self, x, y1, y2, **kwargs):
        MarkerBase.__init__(self)
        lp = {'color': 'black', 'linewidth': 1}
        self.marker_properties = lp
        self.set_data(x1=x, y1=y1, y2=y2)
        self.set_marker_properties(**kwargs)
        self.name = 'vertical_line_segment'

    def __repr__(self):
        string = "<marker.{}, {} (x={},y1={},y2={},color={})>".format(
            self.__class__.__name__,
            self.name,
            self.get_data_position('x1'),
            self.get_data_position('y1'),
            self.get_data_position('y2'),
            self.marker_properties['color'],
        )
        return(string)

    def update(self):
        if self.auto_update is False:
            return
        self._update_segment()

    def _plot_marker(self):
        self.marker = self.ax.vlines(0, 0, 1, **self.marker_properties)
        self._update_segment()

    def _update_segment(self):
        segments = self.marker.get_segments()
        segments[0][0, 0] = self.get_data_position('x1')
        segments[0][1, 0] = segments[0][0, 0]
        if self.get_data_position('y1') is None:
            segments[0][0, 1] = plt.getp(self.marker.axes, 'ylim')[0]
        else:
            segments[0][0, 1] = self.get_data_position('y1')
        if self.get_data_position('y2') is None:
            segments[0][1, 1] = plt.getp(self.marker.axes, 'ylim')[1]
        else:
            segments[0][1, 1] = self.get_data_position('y2')
        self.marker.set_segments(segments)
