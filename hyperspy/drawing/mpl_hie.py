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

from hyperspy.drawing import image, utils
from hyperspy.drawing.mpl_he import MPL_HyperExplorer


class MPL_HyperImage_Explorer(MPL_HyperExplorer):

    def plot_signal(self,
                    colorbar=True,
                    scalebar=True,
                    scalebar_color="white",
                    axes_ticks=None,
                    saturated_pixels=0,
                    vmin=None,
                    vmax=None,
                    no_nans=False,
                    centre_colormap="auto",
                    **kwargs
                    ):
        """Plot image.

        Parameters
        ----------
        colorbar : bool, optional
             If true, a colorbar is plotted for non-RGB images.
        scalebar : bool, optional
            If True and the units and scale of the x and y axes are the same a
            scale bar is plotted.
        scalebar_color : str, optional
            A valid MPL color string; will be used as the scalebar color.
        axes_ticks : {None, bool}, optional
            If True, plot the axes ticks. If None axes_ticks are only
            plotted when the scale bar is not plotted. If False the axes ticks
            are never plotted.
        saturated_pixels: scalar
            The percentage of pixels that are left out of the bounds. For
            example, the low and high bounds of a value of 1 are the
            0.5% and 99.5% percentiles. It must be in the [0, 100] range.
        vmin, vmax : scalar, optional
            `vmin` and `vmax` are used to normalize luminance data.
        no_nans : bool, optional
            If True, set nans to zero for plotting.
        **kwargs, optional
            Additional key word arguments passed to matplotlib.imshow()

        """
        if self.signal_plot is not None:
            self.signal_plot.plot(**kwargs)
            return
        imf = image.ImagePlot()
        imf.axes_manager = self.axes_manager
        imf.data_function = self.signal_data_function
        imf.title = self.signal_title + " Signal"
        imf.xaxis, imf.yaxis = self.axes_manager.signal_axes
        imf.colorbar = colorbar
        imf.quantity_label = self.quantity_label
        imf.scalebar = scalebar
        imf.axes_ticks = axes_ticks
        imf.vmin, imf.vmax = vmin, vmax
        imf.saturated_pixels = saturated_pixels
        imf.no_nans = no_nans
        imf.scalebar_color = scalebar_color
        imf.centre_colormap = centre_colormap
        imf.plot(**kwargs)
        self.signal_plot = imf

        if self.navigator_plot is not None and imf.figure is not None:
            self.navigator_plot.events.closed.connect(
                self._on_navigator_plot_closing, [])
            imf.events.closed.connect(self.close_navigator_plot, [])
            self.signal_plot.figure.canvas.mpl_connect(
                'key_press_event', self.axes_manager.key_navigator)
            self.navigator_plot.figure.canvas.mpl_connect(
                'key_press_event', self.axes_manager.key_navigator)
