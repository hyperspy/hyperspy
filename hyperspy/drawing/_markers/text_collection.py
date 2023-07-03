# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

from hyperspy.drawing.marker_collection import MarkerCollection

class TextCollection(MarkerCollection):
    """
    A Collection of Text for faster plotting. A collection is a set of
    markers which have the same properties.

    This uses the same syntax and the MarkerCollection, where offsets are
    used to define the position of text on some plot.
    """

    def update(self):
        """
        Update the collection of text markers.  This will add new text artists if needed,
        and update the properties of existing text artists.
        """
        if self.is_iterating is False:
            return
        for c in self.collection:
            c.remove()
        self.collection = []
        current_kwargs = self.get_data_position(get_static_kwargs=True)  # need all parameters
        other_kwargs = {k: current_kwargs[k]
                        for k in current_kwargs if k != "offsets"}
        for i, o in enumerate(current_kwargs["offsets"]):
            temp_kwargs = {k: other_kwargs[k][i % len(other_kwargs[k])]
                           for k in other_kwargs}
            t = self.ax.text(o[0], o[1], **temp_kwargs)
            t.animated = self.ax.figure.canvas.supports_blit  # for blitting
            self.collection.append(t)  # Add the text to the collection

    def _initialize_collection(self):
        current_kwargs = self.get_data_position(get_static_kwargs=True)
        other_kwargs = {k: current_kwargs[k]
                        for k in current_kwargs if k != "offsets"}
        self.collection = []
        for i, o in enumerate(current_kwargs["offsets"]):
            temp_kwargs = {k: other_kwargs[k][i % len(other_kwargs[k])]
                           for k in other_kwargs}
            t = self.ax.text(o[0], o[1], **temp_kwargs)
            self.collection.append(t)  # Add the text to the collection

    def plot(self, render_figure=True):
        """
        Plot a marker which has been added to a signal.

        Parameters
        ----------
        render_figure : bool, optional, default True
            If True, will render the figure after adding the marker.
            If False, the marker will be added to the plot, but will the figure
            will not be rendered. This is useful when plotting many markers,
            since rendering the figure after adding each marker will slow
            things down.
        """
        if self.ax is None:
            raise AttributeError(
                "To use this method the marker needs to be first add to a "
                + "figure using `s._plot.signal_plot.add_marker(m)` or "
                + "`s._plot.navigator_plot.add_marker(m)`"
            )
        self._initialize_collection()
        if render_figure:
            self._render_figure()

    def close(self, render_figure=True):
        """Remove and disconnect the marker.

        Parameters
        ----------
        render_figure : bool, optional, default True
            If True, the figure is rendered after removing the marker.
            If False, the figure is not rendered after removing the marker.
            This is useful when many markers are removed from a figure,
            since rendering the figure after removing each marker will slow
            things down.
        """
        if self._closing:
            return
        self._closing = True
        for c in self.collection:
            c.remove()
        self.collection = []
        self.events.closed.trigger(obj=self)
        for f in self.events.closed.connected:
            self.events.closed.disconnect(f)
        if render_figure:
            self._render_figure()
