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
from hyperspy.drawing import utils


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

    def create_figure(self, **kwargs):
        """Create matplotlib figure

        Parameters
        ----------
        **kwargs
            All keyword arguments are passed to ``plt.figure``.

        """
        self.figure = utils.create_figure(
            window_title="Figure " + self.title if self.title
            else None, **kwargs)
        utils.on_figure_window_close(self.figure, self._on_close)
        if self.figure.canvas.supports_blit:
            self._draw_event_cid = self.figure.canvas.mpl_connect(
                'draw_event', self._on_blit_draw)

    def _on_blit_draw(self, *args):
        fig = self.figure
        # As draw doesn't draw animated elements, in its current state the
        # canvas only contains the backgroud. The following line simply stores
        # it for the consumption of _update_animated.
        self._background = fig.canvas.copy_from_bbox(fig.bbox)
        # draw does not draw animated elements, so we must draw them
        # manually
        self._draw_animated()

    def _draw_animated(self):
        """Draw animated plot elements

        """
        for ax in self.figure.axes:
            # Create a list of animated artists and draw them.
            artists = sorted(ax.get_children(), key=lambda x: x.zorder)
            for artist in artists:
                if artist.get_animated():
                    ax.draw_artist(artist)

    def _update_animated(self):
        _logger.debug('Updating animated.')
        canvas = self.ax.figure.canvas
        # As the background haven't changed, we can simply restore it.
        canvas.restore_region(self._background)
        # Now it is when we draw the animated elements using the blit method
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
            marker.close(render_figure=False)
        self.events.closed.trigger(obj=self)
        for f in self.events.closed.connected:
            self.events.closed.disconnect(f)
        if self._draw_event_cid:
            self.figure.canvas.mpl_disconnect(self._draw_event_cid)
            self._draw_event_cid = None

    def close(self):
        figure = self.figure
        self._on_close()   # Needs to trigger serially for a well defined state
        plt.close(figure)
        self.figure = None
        self._background = None

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        # Wrap the title so that each line is not longer than 60 characters.
        self._title = textwrap.fill(value, 60)
