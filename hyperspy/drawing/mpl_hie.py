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

    def plot_signal(self):
        if self.signal_plot is not None:
            self.signal_plot.plot()
            return
        imf = image.ImagePlot()
        imf.axes_manager = self.axes_manager
        imf.data_function = self.signal_data_function
        imf.title = self.signal_title + " Signal"
        imf.xaxis, imf.yaxis = self.axes_manager.signal_axes
        imf.plot_colorbar = True
        imf.plot()
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
