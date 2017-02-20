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

from hyperspy.drawing.marker import MarkerBase


class LineSegment(MarkerBase):

    """Line segment marker that can be added to the signal figure

    Parameters
    ---------
    x1: array or float
        The position of the start of the line segment in x.
        If float, the marker is fixed.
        If array, the marker will be updated when navigating. The array should
        have the same dimensions in the nagivation axes.
    y1: array or float
        The position of the start of the line segment in y.
        see x1 arguments
    x2: array or float
        The position of the end of the line segment in x.
        see x1 arguments
    y2: array or float
        The position of the end of the line segment in y.
        see x1 arguments
    kwargs:
        Kewywords argument of axvline valid properties (i.e. recognized by
        mpl.plot).

    Example
    -------
    >>> im = hs.signals.Signal2D(np.zeros((100, 100)))
    >>> m = hs.plot.markers.line_segment(
    >>>     x1=20, x2=70, y1=20, y2=70,
    >>>     linewidth=4, color='red', linestyle='dotted')
    >>> im.add_marker(m)

    """

    def __init__(self, x1, y1, x2, y2, **kwargs):
        MarkerBase.__init__(self)
        lp = {'color': 'black', 'linewidth': 1}
        self.marker_properties = lp
        self.set_data(x1=x1, y1=y1, x2=x2, y2=y2)
        self.set_marker_properties(**kwargs)

    def update(self):
        if self.auto_update is False:
            return
        self._update_segment()

    def plot(self):
        if self.ax is None:
            raise AttributeError(
                "To use this method the marker needs to be first add to a " +
                "figure using `s._plot.signal_plot.add_marker(m)` or " +
                "`s._plot.navigator_plot.add_marker(m)`")
        x1 = self.get_data_position('x1')
        x2 = self.get_data_position('x2')
        y1 = self.get_data_position('y1')
        y2 = self.get_data_position('y2')
        self.marker = self.ax.plot((x1, x2), (y1, y2),
                                   **self.marker_properties)[0]
        self.marker.set_animated(True)
        try:
            self.ax.hspy_fig._draw_animated()
        except:
            pass

    def _update_segment(self):
        x1 = self.get_data_position('x1')
        x2 = self.get_data_position('x2')
        y1 = self.get_data_position('y1')
        y2 = self.get_data_position('y2')
        self.marker.set_data((x1, x2), (y1, y2))
