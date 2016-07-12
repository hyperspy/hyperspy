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

import matplotlib.pyplot as plt

from hyperspy.drawing.marker import MarkerBase


class Rectangle(MarkerBase):

    """Rectangle marker that can be added to the signal figure

    Parameters
    ---------
    x1: array or float
        The position of the up left corner of the rectangle in x.
        If float, the marker is fixed.
        If array, the marker will be updated when navigating. The array should
        have the same dimensions in the nagivation axes.
    y1: array or float
        The position of the up left corner of the rectangle in y.
        see x1 arguments
    x2: array or float
        The position of the down right corner of the rectangle in x.
        see x1 arguments
    y2: array or float
        The position of the down right of the rectangle in y.
        see x1 arguments
    kwargs:
        Kewywords argument of axvline valid properties (i.e. recognized by
        mpl.plot).

    Example
    -------
    >>> import scipy.misc
    >>> im = hs.signals.Signal2D(scipy.misc.ascent())
    >>> m = hs.plot.markers.rectangle(x1=150, y1=100, x2=400, y2=400,
    >>>                                  color='red')
    >>> im.add_marker(m)

    """

    def __init__(self, x1, y1, x2, y2, **kwargs):
        MarkerBase.__init__(self)
        lp = {'color': 'black', 'fill': None, 'linewidth': 1}
        self.marker_properties = lp
        self.set_data(x1=x1, y1=y1, x2=x2, y2=y2)
        self.set_marker_properties(**kwargs)

    def update(self):
        if self.auto_update is False:
            return
        self.marker.set_xdata([self.get_data_position('x1'),
                               self.get_data_position('x2')])
        self.marker.set_ydata([self.get_data_position('y1'),
                               self.get_data_position('y2')])

    def plot(self):
        if self.ax is None:
            raise AttributeError(
                "To use this method the marker needs to be first add to a " +
                "figure using `s._plot.signal_plot.add_marker(m)` or " +
                "`s._plot.navigator_plot.add_marker(m)`")
        width = abs(self.get_data_position('x1') -
                    self.get_data_position('x2'))
        height = abs(self.get_data_position('y1') -
                     self.get_data_position('y2'))
        self.marker = self.ax.add_patch(plt.Rectangle(
            (self.get_data_position('x1'), self.get_data_position('y1')),
            width, height, **self.marker_properties))
        self.marker.set_animated(True)
        try:
            self.ax.hspy_fig._draw_animated()
        except:
            pass
