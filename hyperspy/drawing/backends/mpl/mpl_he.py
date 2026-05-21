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

import warnings
from threading import Lock

import matplotlib
from traits.api import Undefined

from hyperspy.defaults_parser import preferences
from hyperspy.drawing import image, signal1d, widgets
from hyperspy.drawing.he import HyperExplorer

_lock = Lock()


def _is_widget_backend():
    backend = matplotlib.get_backend()
    # in ipympl 0.9.4/ipython 8.24, the backend name changed from ipympl to widget
    return backend.lower() in ["ipympl", "widget", "module://ipympl.backend_nbagg"]


class MPL_HyperExplorer(HyperExplorer):
    """ """

    def assign_pointer(self):
        if self.navigator_data_function is None:
            nav_dim = 0
        elif self.navigator_data_function == "slider":
            nav_dim = 0
        else:
            nav_dim = len(self.navigator_data_function().shape)

        if nav_dim == 2:  # It is an image
            if self.axes_manager.navigation_dimension > 1:
                Pointer = widgets.SquareWidget
            else:  # It is the image of a "spectrum stack"
                Pointer = widgets.HorizontalLineWidget
        elif nav_dim == 1:  # It is a spectrum
            Pointer = widgets.VerticalLineWidget
        else:
            Pointer = None
        self._pointer_nav_dim = nav_dim
        return Pointer

    def _create_1d_nav_figure(self, title, **kwargs):
        fig = kwargs.pop("fig", None)
        sf = signal1d.Signal1DFigure(
            title=title,
            # Passed to figure creation
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

        # Create a line to the left axis with the default indices
        sl = signal1d.Signal1DLine()
        sl.data_function = self.navigator_data_function

        # Set all kwargs value to the signal figure before passing the rest
        # of the kwargs to plot method of the signal figure
        for key in list(kwargs.keys()):
            if hasattr(sl, key):
                setattr(sl, key, kwargs.pop(key))
        sl.set_line_properties(color="blue", type="step" if axis.is_uniform else "line")
        # Add the line to the figure
        sf.add_line(sl)
        sf.plot()
        return sf

    def _create_2d_nav_figure(self, title, **kwargs):
        imf = image.ImagePlot(
            title=title,
        )
        imf.data_function = self.navigator_data_function

        # Set all kwargs value to the image figure before passing the rest
        # of the kwargs to plot method of the image figure
        for key, value in list(kwargs.items()):
            if hasattr(imf, key):
                setattr(imf, key, kwargs.pop(key))

        # Navigator labels
        if self.axes_manager.navigation_dimension == 1:
            imf.yaxis = self.axes_manager.navigation_axes[0]
            imf.xaxis = self.axes_manager.signal_axes[0]
        elif self.axes_manager.navigation_dimension >= 2:
            imf.yaxis = self.axes_manager.navigation_axes[1]
            imf.xaxis = self.axes_manager.navigation_axes[0]

        if "cmap" not in kwargs.keys() or kwargs["cmap"] is None:
            kwargs["cmap"] = preferences.Plot.cmap_navigator
        imf.plot(
            # Passed to figure creation
            _on_figure_window_close=self.close,
            # Other kwargs
            **kwargs,
        )
        return imf

    def _connect_pointer(self, pointer, figure):
        pointer.set_mpl_ax(figure.ax)

    def _connect_key_nav(self, figure):
        if figure.figure is not None and self.axes_manager.navigation_axes:
            figure.figure.canvas.mpl_connect(
                "key_press_event", self.axes_manager.key_navigator
            )

    def _display(self, plot_style=None, **kwargs):
        if not _is_widget_backend():
            if plot_style is not None:
                warnings.warn(
                    "The `plot_style` keyword is only used with the ipympl backend."
                )
            super()._display(plot_style=plot_style, **kwargs)
            return

        def _do_plot():
            super(MPL_HyperExplorer, self)._display(plot_style=plot_style, **kwargs)

        with matplotlib.pyplot.ioff():
            _do_plot()

        if "fig" not in kwargs:
            from IPython.display import display
            from ipywidgets.widgets import HBox, VBox

            if plot_style not in ["vertical", "horizontal", None]:
                raise ValueError(
                    "plot_style must be one of ['vertical', 'horizontal', None]"
                )
            if plot_style is None:
                plot_style = preferences.Plot.widget_plot_style

            # lock to use IPython.display.display with kernel subshells
            # See https://github.com/matplotlib/ipympl/pull/603
            with _lock:
                sp = self.signal_plot
                np_ = self.navigator_plot
                if sp is None and np_ is not None:
                    display(np_.figure.canvas)
                elif np_ is None:
                    display(sp.figure.canvas)
                elif plot_style == "horizontal":
                    display(HBox([np_.figure.canvas, sp.figure.canvas]))
                else:
                    display(VBox([np_.figure.canvas, sp.figure.canvas]))
