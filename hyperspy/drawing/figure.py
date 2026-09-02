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

import logging
import textwrap
from abc import ABC, abstractmethod

from hyperspy.drawing.backends import get_backend
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
        # The figure object (backend-dependent).
        # To access the matplotlib figure, use `get_mpl_figure`.
        self.figure = None
        # The axes object (backend-dependent).
        self.ax = None
        self.title = ""
        self.ax_markers = list()

    def create_figure(self, **kwargs):
        """
        Create a figure via the active plotting backend.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments forwarded to the backend's ``create_figure``.
        """
        kwargs.setdefault("on_close", self.close)
        backend = get_backend()
        self.figure = backend.create_figure(title=self.title, **kwargs)
        if backend.supports_blit(self.figure):
            self._draw_event_cid = backend.connect_draw_event(
                self.figure, self._on_blit_draw
            )

    def _on_blit_draw(self, *args):
        fig = self.figure
        # As draw doesn't draw animated elements, in its current state the
        # canvas only contains the background. The following line simply stores
        # it for the consumption of _update_animated.
        self._background = get_backend().copy_background(fig)
        # draw does not draw animated elements, so we must draw them manually
        self._draw_animated()

    def _draw_animated(self):
        """Draw animated plot elements."""
        get_backend().draw_animated_artists(self.figure)

    def _update_animated(self):
        _logger.debug("Updating animated.")
        backend = get_backend()
        # As the background hasn't changed, we can simply restore it.
        backend.restore_background(self.figure, self._background)
        # Now draw the animated elements using the blit method
        self._draw_animated()
        backend.blit(self.figure)

    def get_mpl_figure(self):
        """Return the matplotlib Figure (name kept for backward compatibility).

        When ``self.figure`` is a matplotlib ``SubFigure``, the parent
        ``Figure`` is returned instead.
        """
        if self.figure is None:
            return None
        figure = self.figure
        # Only a SubFigure has a .figure attribute (its parent Figure).
        parent = getattr(figure, "figure", None)
        if parent is not None and parent is not figure:
            return parent
        return figure

    def add_marker(self, marker):
        marker.ax = self.ax
        self.ax_markers.append(marker)
        # marker.close() → events.closed → this lambda → mutates ax_markers
        marker.events.closed.connect(lambda obj: self.ax_markers.remove(obj))

    def remove_markers(self, render_figure=False):
        """Remove all markers."""
        # Iterate a snapshot copy: marker.close() triggers events.closed,
        # which calls self.ax_markers.remove(obj) via the lambda registered
        # in add_marker().  Mutating the list during iteration causes
        # every other marker to be skipped.
        for marker in list(self.ax_markers):
            marker.close(render_figure=False)
        if render_figure:
            # Markers closed above removed their collections from the axes
            # but did not touch the blit cache — invalidate it before
            # rendering so the canvas repaints without the old pixels.
            self._background = None
            self.render_figure()

    def _on_close(self):
        _logger.debug("Closing `BlittedFigure`.")
        self.ax = None
        self._background = None
        # Same snapshot-copy rationale as remove_markers (see above).
        for marker in list(self.ax_markers):
            marker.close(render_figure=False)
        self.events.closed.trigger(obj=self)
        for f in list(self.events.closed.connected):
            self.events.closed.disconnect(f)
        if self._draw_event_cid is not None:
            get_backend().disconnect_event(self.figure, self._draw_event_cid)
            self._draw_event_cid = None
        self.figure = None
        _logger.debug("`BlittedFigure` closed.")

    def close(self):
        _logger.debug("`close` `BlittedFigure` called.")
        fig = self.get_mpl_figure()
        self._on_close()  # Needs to trigger serially for a well defined state
        get_backend().close_figure(fig)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        # Wrap the title so that each line is not longer than 60 characters.
        self._title = textwrap.fill(value, 60)

    def render_figure(self):
        backend = get_backend()
        if backend.supports_blit(self.figure) and self._background is not None:
            self._update_animated()
        else:
            backend.draw_idle(self.figure)


class AbstractSignal1DFigure(BlittedFigure, ABC):
    """Abstract interface every 1-D signal figure manager must satisfy.

    Backends return a concrete subclass from
    :meth:`~hyperspy.drawing.backends._protocol.PlottingBackend.create_signal1d_figure`.
    The MPL implementation is
    :class:`~hyperspy.drawing.signal1d.Signal1DFigure`.
    """

    @abstractmethod
    def add_line(self, line, ax="left", connect_navigation=False):
        """Attach a line object to the figure."""

    @abstractmethod
    def plot(self, **kwargs):
        """Render all lines; call once configuration is complete."""

    @abstractmethod
    def update(self):
        """Redraw all lines at the current navigation index."""


class AbstractImageFigure(BlittedFigure, ABC):
    """Abstract interface every 2-D image figure manager must satisfy.

    Backends return a concrete subclass from
    :meth:`~hyperspy.drawing.backends._protocol.PlottingBackend.create_image_figure`.
    The MPL implementation is
    :class:`~hyperspy.drawing.image.ImagePlot`.
    """

    @abstractmethod
    def plot(self, **kwargs):
        """Render the image; call once configuration is complete."""

    @abstractmethod
    def update(self, data_changed=True, **kwargs):
        """Redraw the image at the current navigation index."""
