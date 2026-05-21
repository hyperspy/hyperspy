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

"""Backend-agnostic HyperExplorer base."""

import logging
from functools import partial

from hyperspy.events import Event, Events

_logger = logging.getLogger(__name__)


class HyperExplorer:
    """Orchestrates signal + navigator plotting for any backend.

    Subclasses must implement:
      - plot_signal(**kwargs)
      - assign_pointer()
      - _connect_pointer(pointer, figure) -> None
      - _connect_key_nav(figure) -> None
    """

    def __init__(self):
        self.signal_data_function = None
        self.navigator_data_function = None
        # args to pass to `__call__`
        self.signal_data_function_kwargs = {}
        self.axes_manager = None
        self.signal_title = ""
        self.navigator_title = ""
        self.quantity_label = ""
        self.signal_plot = None
        self.navigator_plot = None
        self.axis = None
        self.pointer = None
        self._pointer_nav_dim = None

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

    def plot_signal(self, **kwargs):
        # This method should be implemented by the subclasses.
        # Doing nothing is good enough for signal_dimension==0 though.
        if self.axes_manager.signal_dimension == 0:
            return
        if self.signal_data_function_kwargs.get("fft_shift", False):
            self.axes_manager = self.axes_manager.deepcopy()
            for axis in self.axes_manager.signal_axes:
                axis.offset = -axis.high_value / 2.0

    def plot_navigator(self, title=None, **kwargs):
        """
        Parameters
        ----------
        title : str, optional
            Title of the navigator. The default is None.
        **kwargs : dict
            The kwargs are passed to plot method of
            :meth:`hyperspy.drawing.image.ImagePlot` or
            :meth:`hyperspy.drawing.signal1d.Signal1DLine`.

        """
        if self.axes_manager.navigation_dimension == 0:
            return
        if self.navigator_data_function is None:
            return
        if self.navigator_data_function == "slider":
            self._get_navigation_sliders()
            return
        title = title or self.signal_title + " Navigator" if self.signal_title else ""

        if len(self.navigator_data_function().shape) == 1:
            fig = self._create_1d_nav_figure(title, **kwargs)
            self.navigator_plot = fig
            if self.pointer is not None:
                self._connect_pointer(self.pointer, fig)
            if self.axes_manager.navigation_dimension > 1:
                self._get_navigation_sliders()
                for ax in self.axes_manager.navigation_axes[:-2]:
                    ax.events.index_changed.connect(fig.update, [])
                    self.events.closed.connect(
                        partial(ax.events.index_changed.disconnect, fig.update), []
                    )
        elif len(self.navigator_data_function().shape) >= 2:
            fig = self._create_2d_nav_figure(title, **kwargs)
            self.navigator_plot = fig
            if self.pointer is not None:
                self._connect_pointer(self.pointer, fig)
            if self.axes_manager.navigation_dimension > 2:
                self._get_navigation_sliders()
                for ax in self.axes_manager.navigation_axes[2:]:
                    ax.events.index_changed.connect(fig.update, [])
                    self.events.closed.connect(
                        partial(ax.events.index_changed.disconnect, fig.update), []
                    )

    def _create_1d_nav_figure(self, title, **kwargs):
        raise NotImplementedError

    def _create_2d_nav_figure(self, title, **kwargs):
        raise NotImplementedError

    def _connect_pointer(self, pointer, figure):
        raise NotImplementedError

    def _connect_key_nav(self, figure):
        raise NotImplementedError

    def _get_navigation_sliders(self):
        try:
            self.axes_manager.gui_navigation_sliders(
                title=self.signal_title + " navigation sliders"
            )
        except (ValueError, ImportError) as e:
            _logger.warning("Navigation sliders not available. " + str(e))

    def close_navigator_plot(self):
        if self.navigator_plot:
            self.navigator_plot.close()

    def assign_pointer(self):
        raise NotImplementedError

    @property
    def is_active(self):
        """A plot is active when it has the figure open meaning that it has
        either one of 'signal_plot' or 'navigation_plot' is not None and it
        has a attribute 'figure' which is not None.
        """
        if self.signal_plot and self.signal_plot.figure:
            return True
        elif self.navigator_plot and self.navigator_plot.figure:
            return True
        else:
            return False

    def close(self):
        """
        Close the plot of the signals.

        This function is called programmatically or on matplotlib close
        callback.
        When closing, it does the following:
        1. trigger a closed event
        2. disconnect the closed event
        3. run the close method of the signal_plot and navigator_plot
        4. reset the attribute
        """
        self.events.closed.trigger(obj=self)
        for f in self.events.closed.connected:
            self.events.closed.disconnect(f)

        for p in [self.signal_plot, self.navigator_plot]:
            if p is not None:
                p.close()

        self.navigator_plot = None
        self.signal_plot = None
