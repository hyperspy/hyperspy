# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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


class VerticalLine(MarkerBase):

    """Vertical line marker that can be added to the signal figure

    Parameters
    ----------
    x : array or float
        The position of the line. If float, the marker is fixed.
        If array, the marker will be updated when navigating. The array should
        have the same dimensions in the navigation axes.
    kwargs :
        Keywords argument of axvline valid properties (i.e. recognized by
        mpl.plot).

    Example
    -------
    >>> s = hs.signals.Signal1D(np.random.random([10, 100]))
    >>> m = hs.plot.markers.vertical_line(x=range(10), color='green')
    >>> s.add_marker(m)

    Adding a marker permanently to a signal

    >>> s = hs.signals.Signal1D(np.random.random((100, 100)))
    >>> m = hs.plot.markers.vertical_line(x=30)
    >>> s.add_marker(m, permanent=True)
    """

    def __init__(self, x, **kwargs):
        MarkerBase.__init__(self)
        lp = {'linewidth': 1, 'color': 'black'}
        self.marker_properties = lp
        self.set_data(x1=x)
        self.set_marker_properties(**kwargs)
        self.name = 'vertical_line'

    def __repr__(self):
        string = "<marker.{}, {} (x={},color={})>".format(
            self.__class__.__name__,
            self.name,
            self.get_data_position('x1'),
            self.marker_properties['color'],
        )
        return(string)

    def update(self):
        if self.auto_update is False:
            return
        self.marker.set_xdata(self.get_data_position('x1'))

    def _plot_marker(self):
        self.marker = self.ax.axvline(self.get_data_position('x1'),
                                      **self.marker_properties)
