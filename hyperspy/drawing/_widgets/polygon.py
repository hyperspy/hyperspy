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

from matplotlib.widgets import PolygonSelector

from hyperspy.drawing.widgets import MPLWidgetBase


class PolygonWidget(MPLWidgetBase):
    """PolygonWidget is a widget for drawing an arbitrary
    polygon, which can then be used as a region-of-interest.
    The polygon can be moved by pressing 'shift' and clicking.
    A polygon vertex can be moved by clicking its handle. If incomplete,
    it is also necessary to press 'ctrl'.
    The polygon can be deleted by pressing 'esc'.
    """

    def __init__(self, axes_manager, **kwargs):
        """
        Parameters
        ----------
        axes_manager : hyperspy.axes.AxesManager
            The axes over which the `PolygonWidget` will interact.
        """

        super().__init__(axes_manager, **kwargs)

        self.set_on(False)
        self._widget = None
        self.position = None

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

        self.set_on(True)

        # Colors of widget. Usually set from constructor.
        handle_props = dict(color=self._color)
        line_props = dict(color=self._color)

        self._widget = PolygonSelector(
            ax,
            onselect=self._onselect,
            useblit=self.blit,
            handle_props=handle_props,
            props=line_props,
        )

        self.ax.figure.canvas.draw_idle()

    def set_vertices(self, vertices):
        """Function for deleting the currently saved polygon and setting a new one.

        Parameters
        ----------
        vertices : list of tuples
            List of  `(x, y)` tuples of the vertices of the polygon to add.
        """

        if self.ax is not None and self.is_on:
            if len(vertices) > 2:
                self._widget.verts = list(vertices)
            self.ax.figure.canvas.draw_idle()

    def connect(self, ax):
        super().connect(ax)

    def get_vertices(self):
        """Returns a list where each entry contains a `(x, y)` tuple
        of the vertices of the polygon. The polygon is not closed."""

        return self._widget.verts.copy()

    def _onselect(self, vertices):

        self.events.changed.trigger(self)

        xmax = max(x for x, y in self._widget.verts)
        ymax = max(y for x, y in self._widget.verts)
        xmin = min(x for x, y in self._widget.verts)
        ymin = min(y for x, y in self._widget.verts)

        self.position = ((xmax + xmin) / 2, (ymax + ymin) / 2)

    def get_centre(self):
        """Returns the xy coordinates of the patch centre. In this implementation, the
        centre of the widget is the centre of the polygon's bounding box.
        """
        return self.position
