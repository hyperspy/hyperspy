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


class HorizontalLineSegment(MarkerBase):

    """Horizontal line segment marker that can be added to the signal figure

    Parameters
    ----------
    x1 : array or float
        The position of the start of the line segment in x.
        If float, the marker is fixed.
        If array, the marker will be updated when navigating. The array should
        have the same dimensions in the navigation axes.
    x2 : array or float
        The position of the end of the line segment in x.
        see x1 arguments
    y : array or float
        The position of line segment in y.
        see x1 arguments
    kwargs :
        Keywords arguments are passed to
        :py:meth:`matplotlib.axes.Axes.hlines`.

    Example
    -------
    >>> im = hs.signals.Signal2D(np.zeros((100, 100)))
    >>> m = hs.plot.markers.horizontal_line_segment(
    >>>     x1=20, x2=70, y=70, linewidth=4, color='red', linestyle='dotted')
    >>> im.add_marker(m)

    Adding a marker permanently to a signal

    >>> im = hs.signals.Signal2D(np.zeros((100, 100)))
    >>> m = hs.plot.markers.horizontal_line_segment(
    >>>     x1=10, x2=30, y=42, linewidth=4, color='red', linestyle='dotted')
    >>> im.add_marker(m, permanent=True)

    """

    def __init__(self, x1, x2, y, **kwargs):
        MarkerBase.__init__(self)
        lp = {'color': 'black', 'linewidth': 1}
        self.marker_properties = lp
        self.set_data(x1=x1, x2=x2, y1=y)
        self.set_marker_properties(**kwargs)
        self.name = 'horizontal_line_segment'

    def __repr__(self):
        string = "<marker.{}, {} (x1={},x2={},y={},color={})>".format(
            self.__class__.__name__,
            self.name,
            self.get_data_position('x1'),
            self.get_data_position('x2'),
            self.get_data_position('y1'),
            self.marker_properties['color'],
        )
        return(string)

    def update(self):
        if self.auto_update is False:
            return
        self._update_segment()

    def _plot_marker(self):
        self.marker = self.ax.hlines(0, 0, 1, **self.marker_properties)
        self._update_segment()

    def _update_segment(self):
        segments = self.marker.get_segments()
        segments[0][0, 1] = self.get_data_position('y1')
        segments[0][1, 1] = segments[0][0, 1]
        if self.get_data_position('x1') is None:
            segments[0][0, 0] = plt.getp(self.marker.axes, 'xlim')[0]
        else:
            segments[0][0, 0] = self.get_data_position('x1')
        if self.get_data_position('x2') is None:
            segments[0][1, 0] = plt.getp(self.marker.axes, 'xlim')[1]
        else:
            segments[0][1, 0] = self.get_data_position('x2')
        self.marker.set_segments(segments)
