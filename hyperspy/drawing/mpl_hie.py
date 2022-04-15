# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

from hyperspy.drawing import image
from hyperspy.drawing.mpl_he import MPL_HyperExplorer
from hyperspy.defaults_parser import preferences


class MPL_HyperImage_Explorer(MPL_HyperExplorer):

    def plot_signal(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : dict
            The kwargs are passed to plot method of the image figure.

        """
        super().plot_signal()
        imf = image.ImagePlot()
        imf.axes_manager = self.axes_manager
        imf.data_function = self.signal_data_function
        imf.title = self.signal_title + " Signal"
        imf.xaxis, imf.yaxis = self.axes_manager.signal_axes

        # Set all kwargs value to the image figure before passing the rest
        # of the kwargs to plot method of the image figure
        for key, value in list(kwargs.items()):
            if hasattr(imf, key):
                setattr(imf, key, kwargs.pop(key))

        imf.quantity_label = self.quantity_label

        kwargs['data_function_kwargs'] = self.signal_data_function_kwargs
        if "cmap" not in kwargs.keys() or kwargs['cmap'] is None:
            kwargs["cmap"] = preferences.Plot.cmap_signal
        imf.plot(**kwargs)
        self.signal_plot = imf

        if imf.figure is not None:
            if self.axes_manager.navigation_axes:
                self.signal_plot.figure.canvas.mpl_connect(
                    'key_press_event', self.axes_manager.key_navigator)
            if self.navigator_plot is not None:
                self.navigator_plot.figure.canvas.mpl_connect(
                    'key_press_event', self.axes_manager.key_navigator)
                imf.events.closed.connect(self.close_navigator_plot, [])
            imf.events.closed.connect(self._on_signal_plot_closing, [])
