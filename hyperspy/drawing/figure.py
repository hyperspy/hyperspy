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

import logging
import textwrap

import matplotlib
import matplotlib.pyplot as plt

from hyperspy.drawing import utils
from hyperspy.events import Event, Events

_logger = logging.getLogger(__name__)


class BlittedFigure:
    def __init__(self):
        self._draw_event_cid = None
        self._background = None
        self.events = Events()
        self.events.closed = Event(
            """
            Event that triggers when the figure window is closed.

            Parameters
            ----------
            obj:  SpectrumFigure instances
                The instance that triggered the event.
            """,
            arguments=["obj"],
        )
        # The matplotlib Figure or SubFigure
        # To access the matplotlib figure, use `get_mpl_figure`
        self.figure = None
        # The matplotlib Axis
        self.ax = None
        self.title = ""
        self.ax_markers = list()

    def create_figure(self, **kwargs):
        """
        Create matplotlib figure.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments are passed to
            :func:`hyperspy.drawing.utils.create_figure`.

        """
        kwargs.setdefault("_on_figure_window_close", self.close)
        self.figure = utils.create_figure(
            window_title="Figure " + self.title if self.title else None,
            **kwargs,
        )
        if self.figure.canvas.supports_blit:
            self._draw_event_cid = self.figure.canvas.mpl_connect(
                "draw_event", self._on_blit_draw
            )

    def _on_blit_draw(self, *args):
        fig = self.figure
        # As draw doesn't draw animated elements, in its current state the
        # canvas only contains the background. The following line simply stores
        # it for the consumption of _update_animated.
        self._background = fig.canvas.copy_from_bbox(fig.bbox)
        # draw does not draw animated elements, so we must draw them
        # manually
        self._draw_animated()

    def _draw_animated(self):
        """Draw animated plot elements"""
        for ax in self.figure.axes:
            # Create a list of animated artists and draw them.
            artists = sorted(ax.get_children(), key=lambda x: x.zorder)
            for artist in artists:
                if artist.get_animated():
                    ax.draw_artist(artist)

    def _update_animated(self):
        _logger.debug("Updating animated.")
        canvas = self.ax.figure.canvas
        # As the background haven't changed, we can simply restore it.
        canvas.restore_region(self._background)
        # Now it is when we draw the animated elements using the blit method
        self._draw_animated()
        canvas.blit(self.figure.bbox)

    def get_mpl_figure(self):
        """Retuns the matplotlib figure"""
        if self.figure is None:
            return None
        else:
            # See https://github.com/matplotlib/matplotlib/pull/28177
            figure = self.figure
            # matplotlib SubFigure can be nested and we don't support it
            if isinstance(figure, matplotlib.figure.SubFigure):
                return figure.figure
            else:
                return figure

    def add_marker(self, marker):
        marker.ax = self.ax
        self.ax_markers.append(marker)
        marker.events.closed.connect(lambda obj: self.ax_markers.remove(obj))

    def remove_markers(self, render_figure=False):
        """Remove all markers"""
        for marker in self.ax_markers:
            marker.close(render_figure=False)
        if render_figure:
            self.render_figure()

    def _on_close(self):
        _logger.debug("Closing `BlittedFigure`.")
        self.ax = None
        self._background = None
        for marker in self.ax_markers:
            marker.close(render_figure=False)
        self.events.closed.trigger(obj=self)
        for f in self.events.closed.connected:
            self.events.closed.disconnect(f)
        if self._draw_event_cid:
            self.figure.canvas.mpl_disconnect(self._draw_event_cid)
            self._draw_event_cid = None
        self.figure = None
        _logger.debug("`BlittedFigure` closed.")

    def close(self):
        _logger.debug("`close` `BlittedFigure` called.")
        self._on_close()  # Needs to trigger serially for a well defined state
        plt.close(self.get_mpl_figure())

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        # Wrap the title so that each line is not longer than 60 characters.
        self._title = textwrap.fill(value, 60)

    def render_figure(self):
        if self.figure.canvas.supports_blit and self._background is not None:
            self._update_animated()
        else:
            self.figure.canvas.draw_idle()
