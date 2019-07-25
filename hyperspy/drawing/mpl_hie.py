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

from hyperspy.drawing import image
from hyperspy.drawing.mpl_he import MPL_HyperExplorer
from hyperspy.docstrings.plot import PLOT2D_DOCSTRING, KWARGS_DOCSTRING


class MPL_HyperImage_Explorer(MPL_HyperExplorer):

    def plot_signal(self,
                    colorbar=True,
                    scalebar=True,
                    scalebar_color="white",
                    axes_ticks=None,
                    axes_off=False,
                    saturated_pixels=None,
                    vmin=None,
                    vmax=None,
                    no_nans=False,
                    centre_colormap="auto",
                    norm="auto",
                    min_aspect=0.1,
                    gamma=1.0,
                    linthresh=0.01,
                    linscale=0.1,
                    **kwargs
                    ):
        """Plot image.

        Parameters
        ----------
        %s
        %s

        """
        if self.signal_plot is not None:
            self.signal_plot.plot(**kwargs)
            return
        super().plot_signal()
        imf = image.ImagePlot()
        imf.axes_manager = self.axes_manager
        imf.data_function = self.signal_data_function
        imf.title = self.signal_title + " Signal"
        imf.xaxis, imf.yaxis = self.axes_manager.signal_axes
        imf.colorbar = colorbar
        imf.quantity_label = self.quantity_label
        imf.scalebar = scalebar
        imf.axes_ticks = axes_ticks
        imf.axes_off = axes_off
        imf.vmin, imf.vmax = vmin, vmax
        imf.saturated_pixels = saturated_pixels
        imf.no_nans = no_nans
        imf.scalebar_color = scalebar_color
        imf.centre_colormap = centre_colormap
        imf.min_aspect = min_aspect
        imf.norm = norm
        imf.gamma = gamma
        imf.linthresh = linthresh
        imf.linscale = linscale
        kwargs['data_function_kwargs'] = self.signal_data_function_kwargs
        imf.plot(**kwargs)
        self.signal_plot = imf

        if imf.figure is not None:
            if self.axes_manager.navigation_axes:
                self.signal_plot.figure.canvas.mpl_connect(
                    'key_press_event', self.axes_manager.key_navigator)
            if self.navigator_plot is not None:
                self.navigator_plot.figure.canvas.mpl_connect(
                    'key_press_event', self.axes_manager.key_navigator)
                self.navigator_plot.events.closed.connect(
                    self._on_navigator_plot_closing, [])
                imf.events.closed.connect(self.close_navigator_plot, [])

    plot_signal.__doc__ %= (PLOT2D_DOCSTRING, KWARGS_DOCSTRING)
