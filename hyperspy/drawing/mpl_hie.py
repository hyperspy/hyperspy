# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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
                    auto_contrast=True,
                    percentile=0.1,
                    vmin=None,
                    vmax=None,
                    no_nans=False,
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
        auto_contrast : bool, optional
            If True, the contrast is stretched for each image using the 
            percentile value.
        percentile : float
            The percentile to be used for contrast stretching. It should be a
            scalar in the 0 to 1 range.
        vmin, vmax : scalar, optional
            `vmin` and `vmax` are used to normalize luminance data. If
            `auto_contrast` is True these values are ignore.
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
        imf.scalebar = scalebar
        imf.axes_ticks = axes_ticks
        imf.vmin, imf.vmax = vmin, vmax
        imf.perc = percentile
        imf.no_nans = no_nans
        imf.scalebar_color = scalebar_color
        imf.auto_contrast = auto_contrast
        imf.plot(**kwargs)
        self.signal_plot = imf

        if self.navigator_plot is not None and imf.figure is not None:
            utils.on_figure_window_close(self.navigator_plot.figure,
                                         self.close_navigator_plot)
            utils.on_figure_window_close(
                imf.figure, self.close_navigator_plot)
            self._key_nav_cid = \
                self.signal_plot.figure.canvas.mpl_connect(
                    'key_press_event', self.axes_manager.key_navigator)
            self._key_nav_cid = \
                self.navigator_plot.figure.canvas.mpl_connect(
                    'key_press_event', self.axes_manager.key_navigator)
