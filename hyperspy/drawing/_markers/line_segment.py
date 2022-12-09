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

from hyperspy.drawing.marker import MarkerBase
from hyperspy.misc.utils import isiterable


class LineSegment(MarkerBase):

    """Line segment marker that can be added to the signal figure

    Parameters
    ----------
    x1 : array or float
        The position of the start of the line segment in x.
        If float, the marker is fixed.
        If array, the marker will be updated when navigating. The array should
        have the same dimensions in the navigation axes.
    y1 : array or float
        The position of the start of the line segment in y.
        see x1 arguments
    x2 : array or float
        The position of the end of the line segment in x.
        see x1 arguments
    y2 : array or float
        The position of the end of the line segment in y.
        see x1 arguments
    kwargs :
        Keyword arguments are passed to
        :py:meth:`matplotlib.axes.Axes.plot`.

    Example
    -------
    >>> im = hs.signals.Signal2D(np.zeros((100, 100)))
    >>> m = hs.plot.markers.line_segment(
    >>>     x1=20, x2=70, y1=20, y2=70,
    >>>     linewidth=4, color='red', linestyle='dotted')
    >>> im.add_marker(m)

    Permanently adding a marker to a signal

    >>> im = hs.signals.Signal2D(np.zeros((100, 100)))
    >>> m = hs.plot.markers.line_segment(
    >>>     x1=10, x2=30, y1=50, y2=70,
    >>>     linewidth=4, color='red', linestyle='dotted')
    >>> im.add_marker(m, permanent=True)

    """

    def __init__(self, x1, y1, x2, y2, **kwargs):
        MarkerBase.__init__(self)
        lp = {'color': 'black', 'linewidth': 1}
        self.marker_properties = lp
        self.set_data(x1=x1, y1=y1, x2=x2, y2=y2)
        self.set_marker_properties(**kwargs)
        self.name = 'line_segment'

    def __repr__(self):
        string = "<marker.{}, {} (x1={},x2={},y1={},y2={},color={})>".format(
            self.__class__.__name__,
            self.name,
            self.get_data_position('x1'),
            self.get_data_position('x2'),
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
        from matplotlib.collections import LineCollection
        x1 = self.get_data_position('x1')
        x2 = self.get_data_position('x2')
        y1 = self.get_data_position('y1')
        y2 = self.get_data_position('y2')

        if not isiterable(x1):
            x1 = [x1]
        if not isiterable(x2):
            x2 = [x2]
        if not isiterable(y1):
            y1 = [y1]
        if not isiterable(y2):
            y2 = [y2]

        line_seg = np.stack([np.stack([x1, x2], axis=-1),
                             np.stack([y1, y2], axis=-1)],
                            axis=-1)
        self.marker = LineCollection(line_seg, **self.marker_properties)
        self.ax.add_collection(self.marker)

    def _update_segment(self):
        x1 = self.get_data_position('x1')
        x2 = self.get_data_position('x2')
        y1 = self.get_data_position('y1')
        y2 = self.get_data_position('y2')

        if not isiterable(x1):
            x1 = [x1]
        if not isiterable(x2):
            x2 = [x2]
        if not isiterable(y1):
            y1 = [y1]
        if not isiterable(y2):
            y2 = [y2]

        line_seg = np.stack([np.stack([x1, x2], axis=-1),
                             np.stack([y1, y2], axis=-1)],
                            axis=-1)
        self.marker.set_segments(line_seg)
