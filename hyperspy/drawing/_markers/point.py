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


class Point(MarkerBase):

    """Point marker that can be added to the signal figure.

    If the signal has one or several navigation axes, the point marker
    can change as a function of the navigation position. This done by
    using an array for the x and y parameters. This array must have
    the same shape as the navigation axes of the signal.

    Parameters
    ----------
    x : array or float
        The position of the point in x. If float, the marker is fixed.
        If array, the marker will be updated when navigating. The array should
        have the same dimensions in the navigation axes.
    y : array or float
        The position of the point in y. see x arguments
    size : array or float, optional, default 20
        The size of the point. see x arguments
    kwargs :
        Keywords argument of axvline valid properties (i.e. recognized by
        mpl.plot).

    Example
    -------
    >>> im = hs.signals.Signal2D(np.random.random([10, 50, 50]))
    >>> m = hs.plot.markers.point(x=range(10), y=range(10)[::-1],
                                     color='red')
    >>> im.add_marker(m)

    Adding a marker permanently to a signal

    >>> im = hs.signals.Signal2D(np.random.random([10, 50, 50]))
    >>> m = hs.plot.markers.point(10, 30, color='blue', size=50)
    >>> im.add_marker(m, permanent=True)

    Markers on local maxima

    >>> from skimage.feature import peak_local_max
    >>> import scipy.misc
    >>> im = hs.signals.Signal2D(scipy.misc.ascent()).as_signal2D([2,0])
    >>> index = array([peak_local_max(i.data, min_distance=100, num_peaks=4)
    >>>                for i in im])
    >>> for i in range(4):
    >>>     m = hs.plot.markers.point(x=index[:, i, 1],
    >>>                                  y=index[:, i, 0], color='red')
    >>>     im.add_marker(m)
    """

    def __init__(self, x, y, size=20, **kwargs):
        MarkerBase.__init__(self)
        lp = {'color': 'black', 'linewidth': None}
        self.marker_properties = lp
        self.set_data(x1=x, y1=y, size=size)
        self.set_marker_properties(**kwargs)
        self.name = 'point'

    def __repr__(self):
        string = "<marker.{}, {} (x={},y={},color={},size={})>".format(
            self.__class__.__name__,
            self.name,
            self.get_data_position('x1'),
            self.get_data_position('y1'),
            self.marker_properties['color'],
            self.get_data_position('size'),
        )
        return(string)

    def update(self):
        if self.auto_update is False:
            return
        self.marker.set_offsets([self.get_data_position('x1'),
                                 self.get_data_position('y1')])
        self.marker._sizes = [self.get_data_position('size')]

    def _plot_marker(self):
        self.marker = self.ax.scatter(self.get_data_position('x1'),
                                      self.get_data_position('y1'),
                                      **self.marker_properties)
        self.marker._sizes = [self.get_data_position('size')]
