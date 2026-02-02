# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

from hyperspy.drawing.widget import WidgetBase


class PolygonWidget(WidgetBase):
    """
    PolygonWidget is a widget for drawing an arbitrary
    polygon, which can then be used as a region-of-interest.
    To construct the polygon, click at the positions to put the
    vertices. Click the first vertex again to complete construction.
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

        # Set default axes
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
        self._finished_building = False
        self._cached_vertices = []

    def _complete_building(self, verts):
        self._finished_building = True
        self._cached_vertices = list(verts)
        self.events.changed.trigger(self)

    def set_on(self, value, render_figure=True):
        """
        Change the on state of the widget. If turning off, the widget will
        disconnect from the events. If turning on, the widget will connect to
        the default events.
        """
        if value is not self.is_on and self.ax is not None:
            existing_ax = self.ax
            if value is True:
                self.connect(self.ax)
            elif value is False:
                self.disconnect()
                self.ax = None
            if render_figure:
                existing_ax.figure.canvas.draw_idle()
        self._is_on = value

    def set_mpl_ax(self, ax):
        """
        Parameters
        ----------
        mpl_ax: matplotlib.axes._subplots.AxesSubplot
            The `matplotlib` axis that the `PolygonWidget` will attach to.
        """
        if ax is self.ax:
            return  # Do nothing
        # Disconnect from previous axes if set
        self.set_on(False, render_figure=False)

        # Connect to new axes
        self.ax = ax
        self.set_on(True, render_figure=False)

        # Colors of widget. Usually set from constructor.
        handle_props = dict(color=self._color)
        line_props = dict(color=self._color)

        useblit = hasattr(self.ax, "hspy_fig") and self.ax.figure.canvas.supports_blit

        self._widget = PolygonSelector(
            ax,
            onselect=self._complete_building,
            useblit=useblit,
            handle_props=handle_props,
            props=line_props,
        )
        self._widget.connect_event("motion_notify_event", self._onmove)

        self.ax.figure.canvas.draw_idle()

    def set_vertices(self, vertices):
        """
        Function for deleting the currently saved polygon and setting a new one.

        Parameters
        ----------
        vertices : list of tuples
            List of  `(x, y)` tuples of the vertices of the polygon to add.
        """

        if self.ax is not None and self.is_on:
            if len(vertices) > 2:
                vertices = [(x, y) for x, y in vertices]
                self._widget.verts = vertices
                self._finished_building = True
                self._cached_vertices = vertices.copy()
            self.ax.figure.canvas.draw_idle()

    def get_vertices(self):
        """
        Returns a list where each entry contains a `(x, y)` tuple
        of the vertices of the polygon. The polygon is not closed.
        Returns an empty list if no polygon is set.
        """

        if self._widget is None:
            return []

        return self._widget.verts.copy()

    def _onmove(self, event):
        # Filter event, see matplotlib event API for more details
        if not self._widget.ignore(event) and event.button is not None:
            self.events.changed.trigger(self)

    @property
    def position(self):
        if self._widget is not None and self._widget.verts:
            vertices = self._widget.verts
            xmax = max(x for x, y in vertices)
            ymax = max(y for x, y in vertices)
            xmin = min(x for x, y in vertices)
            ymin = min(y for x, y in vertices)
            position = ((xmax + xmin) / 2, (ymax + ymin) / 2)
        else:
            position = tuple()

        return position

    def get_centre(self):
        """
        Returns the xy coordinates of the patch centre. In this implementation, the
        centre of the widget is the centre of the polygon's bounding box.
        """
        return self.position

    def finished_building(self):
        """
        Function for checking if polygon is finished being built. Useful for finding
        out if the vertices are complete.
        """

        if self._widget is None:
            return False
        return self._finished_building and self._cached_vertices == self._widget.verts
