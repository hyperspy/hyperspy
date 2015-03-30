# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

from hyperspy.drawing.marker import MarkerBase


class HorizontalLine(MarkerBase):
    """Horizontal line marker that can be added to the signal figure

    Parameters
    ---------
    y: array or float
        The position of the line. If float, the marker is fixed.
        If array, the marker will be updated when navigating. The array should
        have the same dimensions than the nagivation axes.
    kwargs:
        Kewywords argument of axvline valid properties (i.e. recognized by
        mpl.plot).

    Example
    -------
    >>> s = signals.Spectrum(random.random([10, 100])) * 10
    >>> m = utils.plot.markers.horizontal_line(y=range(10), color='green')
    >>> s.plot()
    >>> s._plot.signal_plot.add_marker(m)
    >>> m.plot()

    """

    def __init__(self, y, **kwargs):
        MarkerBase.__init__(self)
        lp = {}
        lp['linewidth'] = 1
        lp['color'] = 'black'
        self.marker_properties = lp
        self.set_data(y1=y)
        self.set_marker_properties(**kwargs)

    def update(self):
        if self.auto_update is False:
            return
        self.marker.set_ydata(self.get_data_position('y1'))

    def plot(self):
        if self.ax is None:
            raise AttributeError(
                "To use this method the marker needs to be first add to a " +
                "figure using `s._plot.signal_plot.add_marker(m)` or " +
                "`s._plot.navigator_plot.add_marker(m)`")
        self.marker = self.ax.axhline(self.get_data_position('y1'),
                                      **self.marker_properties)
        self.marker.set_animated(True)
        try:
            self.ax.hspy_fig._draw_animated()
        except:
            pass
