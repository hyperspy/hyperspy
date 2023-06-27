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

import matplotlib.patches as patches
from hyperspy.drawing.marker import MarkerBase


class Ellipse(MarkerBase):

    """Ellipse marker that can be added to the signal figure

    Parameters
    ----------
    x : array or float
        The position of the center of ellipse in x.
        If float, the marker is fixed.
        If array, the marker will be updated when navigating. The array should
        have the same dimensions in the navigation axes.
    y : array or float
        The position of the center of ellipse in y.
        see x arguments
    width : array or float
        The width (diameter) of the ellipse in x.
        see x arguments
    height : array or float
        The height (diameter) of the ellipse in y.
        see x arguments
    kwargs :
        Keyword arguments are passed to
        :py:class:`matplotlib.patches.Ellipse`.

    Example
    -------
    >>> import skimage
    >>> im = hs.signals.Signal2D(skimage.data.camera())
    >>> m = hs.plot.markers.ellipse(x=150, y=100, width=400, height=400,
    >>>                             edgecolor='red', facecolor='white', fill=True)
    >>> im.add_marker(m)

    Adding a marker permanently to a signal

    >>> im = hs.signals.Signal2D(np.random.random((50, 50))
    >>> m = hs.plot.markers.ellipse(x=20, y=30, width=40, height=49)
    >>> im.add_marker(m, permanent=True)
    """

    def __init__(self, x, y, width, height, **kwargs):
        MarkerBase.__init__(self)
        lp = {'edgecolor': 'black', 'facecolor': None, 'fill': None,
              'linewidth': 1}
        self.marker_properties = lp
        self.set_data(x1=x, y1=y, x2=width, y2=height)
        self.set_marker_properties(**kwargs)
        self.name = 'ellipse'

    def __repr__(self):
        string = "<marker.{}, {} (x={},y={},width={},height={},edgecolor={})>".format(
            self.__class__.__name__,
            self.name,
            self.get_data_position('x1'),
            self.get_data_position('y1'),
            self.get_data_position('x2'),
            self.get_data_position('y2'),
            self.marker_properties['edgecolor'],
        )
        return(string)

    def update(self):
        if self.auto_update is False:
            return
        x1 = self.get_data_position('x1')
        y1 = self.get_data_position('y1')
        width = self.get_data_position('x2')
        height = self.get_data_position('y2')
        self.marker.set_center([x1,y1])
        try:
            self.marker.set_width(width)
            self.marker.set_height(height)
        except AttributeError: # pragma: no cover
            # set_width() and set_height() appear on matplotlib==3.3.0
            # after 3.3.0, marker.width, marker.height are renamed.
            self.marker.width = width
            self.marker.height = height
            self.marker.stale = True

    def _plot_marker(self):
        x1 = self.get_data_position('x1')
        y1 = self.get_data_position('y1')
        width = self.get_data_position('x2')
        height = self.get_data_position('y2')
        self.marker = self.ax.add_patch(patches.Ellipse(
            [x1,y1], width, height, **self.marker_properties))
