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
import warnings
from functools import partial

import matplotlib as mpl
from traits.api import Undefined

from hyperspy.defaults_parser import preferences
from hyperspy.drawing import image, signal1d, widgets

_logger = logging.getLogger(__name__)


class MPL_HyperExplorer(object):
    """ """

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
            # Create the figure
            sf = signal1d.Signal1DFigure(title=title)
            axis = self.axes_manager.navigation_axes[0]
            sf.xlabel = "%s" % str(axis)
            if axis.units is not Undefined:
                sf.xlabel += " (%s)" % axis.units
            sf.ylabel = r"$\Sigma\mathrm{data\,over\,all\,other\,axes}$"
            sf.axis = axis
            sf.axes_manager = self.axes_manager
            self.navigator_plot = sf

            # Create a line to the left axis with the default indices
            sl = signal1d.Signal1DLine()
            sl.data_function = self.navigator_data_function

            # Set all kwargs value to the image figure before passing the rest
            # of the kwargs to plot method of the image figure
            for key in list(kwargs.keys()):
                if hasattr(sl, key):
                    setattr(sl, key, kwargs.pop(key))
            sl.set_line_properties(
                color="blue", type="step" if axis.is_uniform else "line"
            )
            # Add the line to the figure
            sf.add_line(sl)
            sf.plot()
            self.pointer.set_mpl_ax(sf.ax)
            if self.axes_manager.navigation_dimension > 1:
                self._get_navigation_sliders()
                for axis in self.axes_manager.navigation_axes[:-2]:
                    axis.events.index_changed.connect(sf.update, [])
                    sf.events.closed.connect(
                        partial(axis.events.index_changed.disconnect, sf.update), []
                    )
            self.navigator_plot = sf
        elif len(self.navigator_data_function().shape) >= 2:
            # Create the figure
            imf = image.ImagePlot(title=title)
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
                if self.axes_manager.navigation_dimension > 2:
                    self._get_navigation_sliders()
                    for axis in self.axes_manager.navigation_axes[2:]:
                        axis.events.index_changed.connect(imf.update, [])
                        imf.events.closed.connect(
                            partial(axis.events.index_changed.disconnect, imf.update),
                            [],
                        )

            if "cmap" not in kwargs.keys() or kwargs["cmap"] is None:
                kwargs["cmap"] = preferences.Plot.cmap_navigator
            imf.plot(**kwargs)
            self.pointer.set_mpl_ax(imf.ax)
            self.navigator_plot = imf

        if self.navigator_plot is not None:
            self.navigator_plot.events.closed.connect(
                self._on_navigator_plot_closing, []
            )

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

    def plot(self, **kwargs):
        # Parse the kwargs for plotting complex data
        for key in ["power_spectrum", "fft_shift"]:
            if key in kwargs:
                self.signal_data_function_kwargs[key] = kwargs.pop(key)
        backend = mpl.get_backend()
        if "ipympl" not in backend and "plot_style" in kwargs:
            warnings.warn(
                "The `plot_style` keyword is only used when the `ipympl` or `widget`"
                "plotting backends are used."
            )
        plot_style = kwargs.pop("plot_style", None)

        # matplotlib plotting backend
        def plot_sig_and_nav(plot_style):
            if self.pointer is None:
                pointer = self.assign_pointer()
                if pointer is not None:
                    self.pointer = pointer(self.axes_manager)
                    self.pointer.is_pointer = True
                    self.pointer.color = "red"
                    self.pointer.connect_navigate()
                self.plot_navigator(**kwargs.pop("navigator_kwds", {}))
                if pointer is not None:
                    self.navigator_plot.events.closed.connect(
                        self.pointer.disconnect, []
                    )
            self.plot_signal(**kwargs)
            if "ipympl" in backend:
                if plot_style not in ["vertical", "horizontal", None]:
                    raise ValueError(
                        "plot_style must be one of ['vertical', 'horizontal', None]"
                    )
                if plot_style is None:
                    plot_style = preferences.Plot.widget_plot_style
                # If widgets do not already exist, we will `display` them at the end
                from IPython.display import display
                from ipywidgets.widgets import HBox, VBox

                if self.signal_plot is None and self.navigator_plot is not None:
                    # in case the signal is navigation only
                    display(self.navigator_plot.figure.canvas)
                elif self.navigator_plot is None:
                    # in case the signal is signal  only
                    display(self.signal_plot.figure.canvas)
                elif plot_style == "horizontal":
                    display(
                        HBox(
                            [
                                self.navigator_plot.figure.canvas,
                                self.signal_plot.figure.canvas,
                            ]
                        )
                    )
                else:  # plot_style == "vertical":
                    display(
                        VBox(
                            [
                                self.navigator_plot.figure.canvas,
                                self.signal_plot.figure.canvas,
                            ]
                        )
                    )

        if "ipympl" in backend:
            import matplotlib.pyplot as plt

            with plt.ioff():
                plot_sig_and_nav(plot_style)
        else:
            plot_sig_and_nav(plot_style)

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

    def _on_navigator_plot_closing(self):
        self.navigator_plot = None

    def _on_signal_plot_closing(self):
        self.signal_plot = None

    def close(self):
        """When closing, we make sure:
        - close the matplotlib figure
        - drawing events are disconnected
        - the attribute 'signal_plot' and 'navigation_plot' are set to None
        """
        if self.is_active:
            if self.signal_plot:
                self.signal_plot.close()
            self.close_navigator_plot()
