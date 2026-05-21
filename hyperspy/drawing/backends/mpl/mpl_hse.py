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

import numpy as np

from hyperspy.drawing import signal1d
from hyperspy.drawing.backends.mpl.mpl_he import MPL_HyperExplorer
from hyperspy.drawing.hse import HyperSignal1D_Explorer


class MPL_HyperSignal1D_Explorer(HyperSignal1D_Explorer, MPL_HyperExplorer):
    """Plots the current spectrum to the screen and a map with a cursor
    to explore the SI.

    """

    def _make_signal_figure(self, **kwargs):
        fig = kwargs.pop("fig", None)
        sf = signal1d.Signal1DFigure(
            title=self.signal_title + " Signal",
            # Passed to figure creation
            _on_figure_window_close=self.close,
            fig=fig,
        )
        sf.axis = self.axis
        if sf.ax is None:
            sf.create_axis()
        sf.axes_manager = self.axes_manager
        sf.xlabel = self.xlabel
        sf.ylabel = self.ylabel

        # Create a line to the left axis with the default indices
        sl = signal1d.Signal1DLine()
        is_complex = np.iscomplexobj(self.signal_data_function())
        sl.data_function = self.signal_data_function
        kwargs["data_function_kwargs"] = self.signal_data_function_kwargs
        sl.plot_indices = True
        if self.pointer is not None:
            color = self.pointer.color
        else:
            color = "red"
        sl.set_line_properties(color=color, type="step")
        # Add the line to the figure:
        sf.add_line(sl)
        # If the data is complex create a line in the left axis with the
        # default coordinates
        if is_complex:
            sl = signal1d.Signal1DLine()
            sl.data_function = self.signal_data_function
            sl.plot_coordinates = True
            sl._plot_imag = True
            sl.set_line_properties(color="blue", type="step")
            # Add extra line to the figure
            sf.add_line(sl)

        sf.plot(**kwargs)
        return sf

    def _connect_key_handler(self, figure, fn):
        if figure.figure is not None:
            canvas = getattr(figure.figure, "canvas", None)
            if canvas is not None:
                canvas.mpl_connect("key_press_event", fn)

    def _add_right_line(self, **kwargs):
        rl = signal1d.Signal1DLine()
        rl.data_function = self.signal_data_function
        rl.set_line_properties(color=self.right_pointer.color, type="step")
        self.signal_plot.create_right_axis()
        self.signal_plot.add_line(rl, ax="right", connect_navigation=True)
        rl.plot_indices = True
        rl.text_position = (
            1.0,
            1.05,
        )
        rl.plot(**kwargs)

    def _redraw_signal_figure(self):
        # because we added the right axis, we need to redraw the canvas to
        # update the background
        self.signal_plot.figure.canvas.draw_idle()
