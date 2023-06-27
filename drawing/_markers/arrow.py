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


class Arrow(MarkerBase):

    """Arrow marker that can be added to the signal figure

    Parameters
    ----------
    x1 : array or float
        The position of the tail of the arrow in x.
        If float, the marker is fixed.
        If array, the marker will be updated when navigating. The array should
        have the same dimensions in the navigation axes.
    y1 : array or float
        The position of the tail of the arrow in y.
        see x1 arguments
    x2 : array or float
        The position of the head of the arrow in x.
        see x1 arguments
    y2 : array or float
        The position of the head of the arrow in y.
        see x1 arguments
    kwargs :
        Keyword arguments are passed to
        :py:class:`matplotlib.patches.FancyArrowPatch`.

    Example
    -------
    >>> import skimage
    >>> im = hs.signals.Signal2D(skimage.data.camera())
    >>> m = hs.plot.markers.arrow(x1=150, y1=100, x2=400, y2=400,
    >>>                           arrowprop={'arrowstyle':'<->', edgecolor='red'})
    >>> im.add_marker(m)

    Adding a marker permanently to a signal

    >>> im.add_marker(m, permanent=True)
    """

    def __init__(self, x1, y1, x2, y2, **kwargs):
        MarkerBase.__init__(self)
        lp = {'edgecolor': 'black', 'facecolor': None,
              'linewidth': None, 'arrowstyle': '->',
              # size of arrow head
              'mutation_scale': 12,
              # both ends of arrow on the exact pos.
              # default of matplotlib is 2
              'shrinkA': 0, 'shrinkB': 0,
        }
        self.marker_properties = lp
        self.set_data(x1=x1, y1=y1, x2=x2, y2=y2)
        self.set_marker_properties(**kwargs)
        self.name = 'arrow'

    def __repr__(self):
        string = "<marker.{}, {} (x1={},x2={},y1={},y2={},edgecolor={},arrowstyle={})>".format(
            self.__class__.__name__,
            self.name,
            self.get_data_position('x1'),
            self.get_data_position('x2'),
            self.get_data_position('y1'),
            self.get_data_position('y2'),
            self.marker_properties['edgecolor'],
            self.marker_properties['arrowstyle'],
        )
        return(string)

    def update(self):
        if self.auto_update is False:
            return
        x1 = self.get_data_position('x1')
        x2 = self.get_data_position('x2')
        y1 = self.get_data_position('y1')
        y2 = self.get_data_position('y2')
        self.marker.set_positions((x1, y1), (x2, y2))

    def _plot_marker(self):
        x1 = self.get_data_position('x1')
        x2 = self.get_data_position('x2')
        y1 = self.get_data_position('y1')
        y2 = self.get_data_position('y2')
        self.marker = self.ax.add_patch(patches.FancyArrowPatch(
            (x1,y1), (x2,y2), **self.marker_properties))
