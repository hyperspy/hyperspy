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

import matplotlib.patches as patches
from hyperspy.drawing.marker import MarkerBase


class Arrow(MarkerBase):

    """Rectangle marker that can be added to the signal figure

    Parameters
    ----------
    x1 : array or float
        The position of the up left corner of the rectangle in x.
        If float, the marker is fixed.
        If array, the marker will be updated when navigating. The array should
        have the same dimensions in the navigation axes.
    y1 : array or float
        The position of the up left corner of the rectangle in y.
        see x1 arguments
    x2 : array or float
        The position of the down right corner of the rectangle in x.
        see x1 arguments
    y2 : array or float
        The position of the down right of the rectangle in y.
        see x1 arguments
    kwargs :
        Keywords argument of axvline valid properties (i.e. recognized by
        mpl.plot).

    Example
    -------
    >>> import scipy.misc
    >>> im = hs.signals.Signal2D(scipy.misc.ascent())
    >>> m = hs.plot.markers.rectangle(x1=150, y1=100, x2=400, y2=400,
    >>>                                  color='red')
    >>> im.add_marker(m)

    Adding a marker permanently to a signal

    >>> im = hs.signals.Signal2D(np.random.random((50, 50))
    >>> m = hs.plot.markers.rectangle(x1=20, y1=30, x2=40, y2=49)
    >>> im.add_marker(m, permanent=True)
    """

    def __init__(self, x1, y1, x2, y2, **kwargs):
        MarkerBase.__init__(self)
        lp = {'edgecolor': 'black', 'facecolor': None, 'fill': None,
              'linewidth': 1, 'arrow_style': '->', 'zorder' : 0.1}
        self.marker_properties = lp
        self.set_data(x1=x1, y1=y1, x2=x2, y2=y2)
        self.set_marker_properties(**kwargs)
        mp = self.marker_properties
        if 'color' in mp:
            mp['edgecolor'] = mp['color']
            del mp['color']
        self.name = 'arrow'

    def __repr__(self):
        string = "<marker.{}, {} (x1={},x2={},y1={},y2={},edgecolor={},facecolor={})arrow_style={},fill={},linewidth={},zorder={}>".format(
            self.__class__.__name__,
            self.name,
            self.get_data_position('x1'),
            self.get_data_position('x2'),
            self.get_data_position('y1'),
            self.get_data_position('y2'),
            self.marker_properties['edgecolor'],
            self.marker_properties['facecolor'],
            self.marker_properties['arrow_style'],
            self.marker_properties['fill'],
            self.marker_properties['linewidth'],
            self.marker_properties['zorder'],
        )
        return(string)

    def update(self):
        x1 = self.get_data_position('x1')
        x2 = self.get_data_position('x2')
        y1 = self.get_data_position('y1')
        y2 = self.get_data_position('y2')
        self.marker.set_data(x1=x1, y1=y1, x2=x2, y2=y2)

    def _plot_marker(self):
        xy1, xy2, _, _ = self.get_xywh()
        mp = self.marker_properties
        self.marker = self.ax.annotate('',xy1,xy2, arrowprops={
            'facecolor' : mp['facecolor'],
            'edgecolor' : mp['edgecolor'],
            'arrowstyle' : mp['arrow_style'],
        })
        if 'zorder' in self.marker_properties:
            self.marker.set_zorder(self.marker_properties['zorder'])
        


