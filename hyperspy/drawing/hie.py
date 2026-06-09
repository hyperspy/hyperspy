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

"""Backend-agnostic HyperImage_Explorer base."""

from hyperspy.drawing.he import HyperExplorer


class HyperImage_Explorer(HyperExplorer):
    """Base for 2-D image explorers."""

    def plot_signal(self, **kwargs):
        super().plot_signal()
        imf = self._make_image_figure(**kwargs)
        self.signal_plot = imf
        self._connect_key_nav(imf)
        if self.navigator_plot is not None:
            self._connect_key_nav(self.navigator_plot)

    def _make_image_figure(self, **kwargs):
        from hyperspy.defaults_parser import preferences
        from hyperspy.drawing import image

        imf = image.ImagePlot()
        imf.axes_manager = self.axes_manager
        imf.data_function = self.signal_data_function
        imf.title = self.signal_title + " Signal"
        imf.xaxis, imf.yaxis = self.axes_manager.signal_axes

        for key, value in list(kwargs.items()):
            if hasattr(imf, key):
                setattr(imf, key, kwargs.pop(key))

        imf.quantity_label = self.quantity_label
        kwargs["data_function_kwargs"] = self.signal_data_function_kwargs
        if "cmap" not in kwargs or kwargs.get("cmap") is None:
            kwargs["cmap"] = preferences.Plot.cmap_signal
        imf.plot(_on_figure_window_close=self.close, **kwargs)
        return imf
