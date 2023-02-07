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
from hyperspy.roi import PolygonROI


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

        self._vertices_lists = []
        self._vertices_plots = []

        self._selected_polygon = None
        self._selected_plot = None

        self._finished = True
        self._is_on = False

        if polygons is not None:
            self._vertices_lists = polygons.copy()

        if mpl_ax is not None:
            self.set_mpl_ax(mpl_ax)

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

        if self._vertices_lists:
            for plot in self._vertices_plots:
                plot.remove()
            self._vertices_plots = []
            for polygon in self._vertices_lists:
                self._plot_polygon(polygon)

        handle_props = dict(color="blue")
        props = dict(color="blue")

        self._widget = PolygonSelector(
            ax,
            onselect=self._onselect,
            useblit=self.blit,
            handle_props=handle_props,
            props=props,
        )
        self._finished = False

        self.ax.figure.canvas.draw_idle()

    def set_polygons(self, polygons):
        """Function for deleting currently saved polygons and setting new ones."""

        if self._vertices_plots:
            for plot in self._vertices_plots:
                plot.remove()
        self._vertices_plots = []

        self._vertices_lists = polygons.copy()

        if self.ax is not None and self.is_on:
            for polygon in polygons:
                self._plot_polygon(polygon)
            self.ax.figure.canvas.draw_idle()

    def connect(self, ax):
        super().connect(ax)
        self._set_mpl_callbacks()

    def _set_mpl_callbacks(self):
        self.cids.append(
            self.ax.figure.canvas.mpl_connect("button_press_event", self._click_event)
        )
        self.cids.append(
            self.ax.figure.canvas.mpl_connect("key_press_event", self._key_event)
        )

    def get_polygons(self):
        """Returns a list of lists, where each inner list contains `(x, y)` tuples
        of the vertices of a single polygon. Only completed polygons are included.
        The polygons are not closed."""
        if self._finished:
            return self._vertices_lists + [self._widget.verts.copy()]
        else:
            return self._vertices_lists

    def _plot_polygon(self, polygon, selected_plot=False):
        closed_polygon = zip(*(polygon + [polygon[0]]))
        if not selected_plot:
            plot = self.ax.plot(*closed_polygon, animated=self.blit)
            self._vertices_plots.append(plot)
        else:
            self._selected_plot = self.ax.plot(
                *closed_polygon, animated=self.blit, linewidth=3, color="red"
            )

    def _onselect(self, vertices):
        if len(self._widget.verts) > 0:
            self._finished = True
            self.events.changed.trigger(self)

            all_polygons = self.get_polygons()
            all_x_coordinates = [x for polygon in all_polygons for x, y in polygon]
            all_y_coordinates = [y for polygon in all_polygons for x, y in polygon]

            bounding_box = (
                min(all_x_coordinates),
                max(all_x_coordinates),
                min(all_y_coordinates),
                max(all_y_coordinates),
            )
            tup = (
                (bounding_box[0] + bounding_box[1]) / 2,
                (bounding_box[2] + bounding_box[3]) / 2,
            )

            self.position = tup

    def _click_event(self, event):

        if (
            event.inaxes is None
            or event.inaxes.axes is None
            or event.inaxes.axes != self.ax.axes
        ):
            return

        # Shift and escape are used to interact with `self._widget`
        if (
            hasattr(event, "key")
            and event.key
            and ("shift" in event.key or "esc" in event.key)
        ):
            return

        x_coord, y_coord = event.x, event.y

        # Do nothing if within grab range of widget vertices
        if self._selected_polygon is None:

            grab_range_sq = self._widget.grab_range**2
            xy_vals = self._widget.verts

            # Transform from value space to pixel space
            xy_pixels = self.ax.transData.transform(xy_vals)

            if any(
                (x_pix - x_coord) ** 2 + (y_pix - y_coord) ** 2 < grab_range_sq
                for x_pix, y_pix in xy_pixels
            ):
                return

        if self._finished or len(self._widget.verts) == 0:

            start_index = 0

            if self._selected_polygon is not None:

                self._selected_plot[0].remove()
                del self._selected_plot
                self._selected_plot = None
                start_index = self._selected_polygon + 1

            # If clicked within another polygon, set that polygon to active
            click_x, click_y = event.xdata, event.ydata
            self._selected_polygon = self._find_clicked_polygon(
                click_x, click_y, start_index=start_index
            )

            if self._selected_polygon is not None:
                vertices = self._vertices_lists[self._selected_polygon]
                self._plot_polygon(vertices, selected_plot=True)
                self._widget.set_active(False)
                self.ax.figure.canvas.draw_idle()

                return
            elif start_index > 0:
                self.ax.figure.canvas.draw_idle()
                self._widget.set_active(True)
                return

        if self._is_on and self._finished and len(self._widget.verts) > 2:

            self._vertices_lists.append(self._widget.verts)
            self._plot_polygon(self._vertices_lists[-1])
            self.ax.figure.canvas.draw_idle()

            self._widget.clear()
            self._widget.set_active(False)
            del self._widget

            self._widget = PolygonSelector(
                self.ax,
                onselect=self._onselect,
                useblit=self.blit,
                handle_props=dict(color="blue"),
                props=dict(color="blue"),
            )
            self._widget.set_active(True)

            self._finished = False
            self.ax.figure.canvas.draw_idle()
            self.events.changed.trigger(self)

    def _key_event(self, event):
        if (
            event.inaxes is None
            or event.inaxes.axes is None
            or event.inaxes.axes != self.ax.axes
        ):
            return

        if "escape" == event.key.lower() and self._widget.active:
            self._finished = False
            self.events.changed.trigger(self)
            return

        if self._selected_polygon is not None and "delete" == event.key.lower():

            self._selected_plot[0].remove()
            del self._selected_plot
            self._selected_plot = None

            del self._vertices_lists[self._selected_polygon]
            self._vertices_plots[self._selected_polygon][0].remove()
            del self._vertices_plots[self._selected_polygon]
            self._selected_polygon = None
            self.ax.figure.canvas.draw_idle()
            self.events.changed.trigger(self)
            self._widget.set_active(True)
            return

    def _find_clicked_polygon(self, x, y, start_index=0):
        for i, vertices in enumerate(
            self._vertices_lists[start_index:], start=start_index
        ):
            if Path(vertices).contains_point((x, y)):
                return i
        return None

    def get_centre(self):
        """Returns the xy coordinates of the patch centre. In this implementation, the
        centre of the widget is the centre of the bounding box encompassing all vertices
        of all polygons.
        """
        return self.position

    def get_roi(self):
        """Returns the selected polygons as a `hyperspy.widgets.PolygonROI`, which
        can be used to extract or mask out the selected areas from a dataset.
        """
        return PolygonROI(self.get_polygons())

    def get_mask(self):
        """Returns the selected polygons as a boolean numpy array, where
        the insides of the polygons are `True`. Self-overlapping polygons
        might have overlapping areas registered as False.
        """

        return self.get_roi().boolean_mask(axes=self.axes)
