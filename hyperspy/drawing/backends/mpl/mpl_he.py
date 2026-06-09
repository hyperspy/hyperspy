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
import matplotlib.pyplot as plt

from hyperspy.defaults_parser import preferences
from hyperspy.drawing.he import HyperExplorer

_lock = Lock()


def _is_widget_backend():
    backend = matplotlib.get_backend()
    # in ipympl 0.9.4/ipython 8.24, the backend name changed from ipympl to widget
    return backend.lower() in ["ipympl", "widget", "module://ipympl.backend_nbagg"]


class MPL_HyperExplorer(HyperExplorer):
    """ """

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

        with plt.ioff():
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
