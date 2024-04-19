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

from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector

from hyperspy.drawing.widgets import MPLWidgetBase


class PolygonWidget(MPLWidgetBase):

    """PolygonWidget is a widget for drawing one or more arbitrary
    polygons, which can then be used as a region-of-interest.
    The active polygon can be moved by shift-clicking.
    A polygon vertex can be moved by clicking its handle. If incomplete,
    it is also necessary to press control.
    The active polygon can be deleted by pressing escape.
    To delete other polygons, click inside it until it has a red outline,
    then press `Delete`.
    """

    def __init__(self, axes_manager, mpl_ax=None, polygons=None, **kwargs):
        """
        Parameters
        ----------
        axes_manager : hyperspy.axes.AxesManager
            The axes over which the `PolygonWidget` will interact.
        mpl_ax: matplotlib.axes._subplots.AxesSubplot
            The `matplotlib` axis that the `PolygonWidget` will attach to. This
            can be added later with the member function `set_mpl_ax`.
        polygons : list of lists of tuples
            List of lists, where each inner list contains (x, y) values
            of the vertices of a polygon. If a single polygon is desired,
            a list of the (x, y) values can be given directly.
            These will be added to the initial widget state.
        """

        super().__init__(axes_manager, **kwargs)

        self._widget = None

        self._vertices = []

    def set_mpl_ax(self, ax):
        """
        Parameters
        ----------
        mpl_ax: matplotlib.axes._subplots.AxesSubplot
            The `matplotlib` axis that the `PolygonWidget` will attach to.
        """
        if ax is self.ax or ax is None:
            return  # Do nothing
        # Disconnect from previous axes if set
        if self.ax is not None and self.is_on:
            self.disconnect()
        self.ax = ax

        if self.is_on is True:

            handle_props = dict(color="blue")
            props = dict(color="blue")

            self._widget = PolygonSelector(
                ax,
                onselect=self._onselect,
                useblit=self.blit,
                handle_props=handle_props,
                props=props,
            )

        self.ax.figure.canvas.draw_idle()


    def connect(self, ax):
        super().connect(ax)

    def get_polygons(self):
        """Returns a list of lists, where each inner list contains `(x, y)` tuples
        of the vertices of a single polygon. Only completed polygons are included.
        The polygons are not closed."""
        if self._finished:
            return self._vertices
        else:
            return self._vertices

    def _onselect(self, vertices):
        if len(self._widget.verts) > 0:

            self.position = (0, 0)

    def get_centre(self):
        """Returns the xy coordinates of the patch centre. In this implementation, the
        centre of the widget is the centre of the polygon's bounding box.
        """
        return self.position
