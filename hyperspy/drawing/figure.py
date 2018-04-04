# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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

import textwrap
import matplotlib.pyplot as plt
import logging

from hyperspy.events import Event, Events


_logger = logging.getLogger(__name__)


class BlittedFigure(object):

    def __init__(self):
        self._draw_event_cid = None
        self._background = None
        self.events = Events()
        self.events.closed = Event("""
            Event that triggers when the figure window is closed.

            Arguments:
                obj:  SpectrumFigure instances
                    The instance that triggered the event.
            """, arguments=["obj"])

    def _update_background(self, *args):
        fig = self.figure
        self._background = fig.canvas.copy_from_bbox(fig.bbox)
        self._draw_animated()

    def _draw_animated(self):
        for ax in self.figure.axes:
            artists = []
            artists.extend(ax.images)
            artists.extend(ax.collections)
            artists.extend(ax.patches)
            artists.extend(ax.lines)
            artists.extend(ax.texts)
            artists.extend(ax.artists)
            artists.append(ax.get_yaxis())
            artists.append(ax.get_xaxis())
            [ax.draw_artist(a) for a in artists if a.get_animated()]

    def _update_animated(self):
        canvas = self.ax.figure.canvas
        canvas.restore_region(self._background)
        self._draw_animated()
        canvas.blit(self.figure.bbox)

    def add_marker(self, marker):
        marker.ax = self.ax
        if marker.axes_manager is None:
            marker.axes_manager = self.axes_manager
        self.ax_markers.append(marker)
        marker.events.closed.connect(lambda obj: self.ax_markers.remove(obj))

    def _on_close(self):
        if self.figure is None:
            return  # Already closed
        for marker in self.ax_markers:
            marker.close(update_plot=False)
        self.events.closed.trigger(obj=self)
        for f in self.events.closed.connected:
            self.events.closed.disconnect(f)
        if self._draw_event_cid:
            self.figure.canvas.mpl_disconnect(self._draw_event_cid)
        self.figure = None

    def close(self):
        figure = self.figure
        self._on_close()   # Needs to trigger serially for a well defined state
        plt.close(figure)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        # Wrap the title so that each line is not longer than 60 characters.
        self._title = textwrap.fill(value, 60)
