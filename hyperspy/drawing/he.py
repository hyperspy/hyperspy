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

from traits.api import Undefined

from hyperspy.events import Event, Events

_logger = logging.getLogger(__name__)


class HyperExplorer:
    """Orchestrates signal + navigator plotting for any backend.

    All common explorer logic lives here.  Backend-specific subclasses
    override only what differs (e.g. ``_display`` for ipympl layout,
    ``_add_right_line`` for MPL-specific twin-y axes).
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
        self._key_nav_cids = []  # list of (fig, cid) pairs for cleanup

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

    def plot(self, **kwargs):
        for key in ["power_spectrum", "fft_shift"]:
            if key in kwargs:
                self.signal_data_function_kwargs[key] = kwargs.pop(key)
        plot_style = kwargs.pop("plot_style", None)
        self._display(plot_style=plot_style, **kwargs)

    def _display(self, plot_style=None, **kwargs):
        """Set up pointer, call plot_navigator then plot_signal."""
        navigator_kwds = kwargs.pop("navigator_kwds", {})
        if self.pointer is None:
            pointer_cls = self.assign_pointer()
            if pointer_cls is not None:
                self.pointer = pointer_cls(self.axes_manager)
                self.pointer.is_pointer = True
                self.pointer.color = "red"
                self.pointer.connect_navigate()
                self.events.closed.connect(self.pointer.disconnect, [])
            self.plot_navigator(**navigator_kwds)
        self.plot_signal(**kwargs)

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

    # ── Pointer assignment ────────────────────────────────────────────────

    def assign_pointer(self):
        from hyperspy.drawing import widgets

        if self.navigator_data_function is None:
            nav_dim = 0
        elif self.navigator_data_function == "slider":
            nav_dim = 0
        else:
            nav_dim = len(self.navigator_data_function().shape)

        if nav_dim == 2:
            if self.axes_manager.navigation_dimension > 1:
                Pointer = widgets.SquareWidget
            else:
                Pointer = widgets.HorizontalLineWidget
        elif nav_dim == 1:
            Pointer = widgets.VerticalLineWidget
        else:
            Pointer = None
        self._pointer_nav_dim = nav_dim
        return Pointer

    def _connect_pointer(self, pointer, figure):
        pointer.set_ax(figure.ax)

    # ── Navigator figure creation ─────────────────────────────────────────

    def _create_1d_nav_figure(self, title, **kwargs):
        from hyperspy.drawing import signal1d

        fig = kwargs.pop("fig", None)
        sf = signal1d.Signal1DFigure(
            title=title,
            _on_figure_window_close=self.close,
            fig=fig,
        )
        axis = self.axes_manager.navigation_axes[0]
        sf.xlabel = "%s" % str(axis)
        if axis.units is not Undefined:
            sf.xlabel += " (%s)" % axis.units
        sf.ylabel = r"$\Sigma\mathrm{data\,over\,all\,other\,axes}$"
        sf.axis = axis
        sf.axes_manager = self.axes_manager

        sl = signal1d.Signal1DLine()
        sl.data_function = self.navigator_data_function
        for key in list(kwargs.keys()):
            if hasattr(sl, key):
                setattr(sl, key, kwargs.pop(key))
        sl.set_line_properties(color="blue", type="step" if axis.is_uniform else "line")
        sf.add_line(sl)
        sf.plot()
        return sf

    def _create_2d_nav_figure(self, title, **kwargs):
        from hyperspy.defaults_parser import preferences
        from hyperspy.drawing import image

        imf = image.ImagePlot(title=title)
        imf.data_function = self.navigator_data_function

        for key, value in list(kwargs.items()):
            if hasattr(imf, key):
                setattr(imf, key, kwargs.pop(key))

        if self.axes_manager.navigation_dimension == 1:
            imf.yaxis = self.axes_manager.navigation_axes[0]
            imf.xaxis = self.axes_manager.signal_axes[0]
        elif self.axes_manager.navigation_dimension >= 2:
            imf.yaxis = self.axes_manager.navigation_axes[1]
            imf.xaxis = self.axes_manager.navigation_axes[0]

        if "cmap" not in kwargs or kwargs.get("cmap") is None:
            kwargs["cmap"] = preferences.Plot.cmap_navigator
        imf.plot(_on_figure_window_close=self.close, **kwargs)
        return imf

    # ── Key navigation ────────────────────────────────────────────────────

    def _connect_key_nav(self, figure):
        from hyperspy.drawing.backends import get_backend

        if figure.figure is not None and self.axes_manager.navigation_axes:
            cid = get_backend().connect_key_press(
                figure.figure, self.axes_manager.key_navigator
            )
            if cid is not None:
                self._key_nav_cids.append((figure.figure, cid))

    # ── Utility ───────────────────────────────────────────────────────────

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
        1. disconnect key navigation event handlers
        2. trigger a closed event
        3. disconnect the closed event
        4. run the close method of the signal_plot and navigator_plot
        5. reset the attribute
        """
        from hyperspy.drawing.backends import get_backend

        backend = get_backend()
        for fig, cid in self._key_nav_cids:
            backend.disconnect_event(fig, cid)
        self._key_nav_cids.clear()

        self.events.closed.trigger(obj=self)
        for f in self.events.closed.connected:
            self.events.closed.disconnect(f)

        for p in [self.signal_plot, self.navigator_plot]:
            if p is not None:
                p.close()

        self.navigator_plot = None
        self.signal_plot = None
