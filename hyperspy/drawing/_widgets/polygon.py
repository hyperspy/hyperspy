# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

import matplotlib
from matplotlib.widgets import PolygonSelector

from hyperspy.drawing.widget import WidgetBase

from packaging.version import Version


class PolygonWidget(WidgetBase):
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

        if self.axes_manager is not None:
            if self.axes_manager.navigation_dimension > 1:
                self.axes = self.axes_manager.navigation_axes[0:2]
            elif self.axes_manager.signal_dimension > 1:
                self.axes = self.axes_manager.signal_axes[0:2]
            elif len(self.axes_manager.shape) > 1:
                self.axes = (
                    self.axes_manager.signal_axes + self.axes_manager.navigation_axes
                )
            else:
                raise ValueError("PolygonWidget needs at least two axes!")

        self._widget = None
        self.position = tuple()

    def set_on(self, value, render_figure=True):
        """Change the on state of the widget. If turning off, the widget will
        disconnect from the events. If turning on, the widget will connect to
        the default events.
        """
        if value is not self.is_on and self.ax is not None:
            if value is True:
                self.blit = (
                    hasattr(self.ax, "hspy_fig") and self.ax.figure.canvas.supports_blit
                )
                self.connect(self.ax)
                if render_figure:
                    self.ax.figure.canvas.draw_idle()
            elif value is False:
                self.disconnect()
                if render_figure:
                    self.ax.figure.canvas.draw_idle()
                self.ax = None
        self._is_on = value

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

        self.set_on(True, render_figure=False)

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

                if Version(matplotlib.__version__) >= Version("3.6"):
                    self._widget.verts = list(vertices)
                else:
                    # Accessing some private members to be able to change the `matplotlib`
                    # widgets with older `matplotlib` versions.
                    # Can hopefully be removed soon in the future
                    self._widget._xs, self._widget._ys = [
                        list(dimension) for dimension in zip(*vertices)
                    ]

                    # Close the polygon
                    self._widget._xs.append(self._widget._xs[0])
                    self._widget._ys.append(self._widget._ys[0])

                    self._widget._selection_completed = True
                    self._widget.set_visible(True)
                    self._widget._draw_polygon()
                self._onselect(vertices)
            self.ax.figure.canvas.draw_idle()

    def get_vertices(self):
        """Returns a list where each entry contains a `(x, y)` tuple
        of the vertices of the polygon. The polygon is not closed.
        Returns an empty list if no polygon is set."""

        if self._widget is None:
            return []

        return self._widget.verts.copy()

    def _onselect(self, vertices):

        self.events.changed.trigger(self)

        xmax = max(x for x, y in vertices)
        ymax = max(y for x, y in vertices)
        xmin = min(x for x, y in vertices)
        ymin = min(y for x, y in vertices)

        self.position = ((xmax + xmin) / 2, (ymax + ymin) / 2)

    def get_centre(self):
        """Returns the xy coordinates of the patch centre. In this implementation, the
        centre of the widget is the centre of the polygon's bounding box.
        """
        return self.position
