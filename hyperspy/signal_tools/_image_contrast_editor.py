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

import copy

import matplotlib
import numpy as np
import traits.api as t

from hyperspy import drawing
from hyperspy.axes import UniformDataAxis
from hyperspy.docstrings.signal import HISTOGRAM_MAX_BIN_ARGS
from hyperspy.drawing._widgets.range import SpanSelector
from hyperspy.drawing.signal1d import Signal1DFigure
from hyperspy.misc.array_tools import numba_histogram
from hyperspy.ui_registry import add_gui_method


@add_gui_method(toolkey="hyperspy.Signal1D.contrast_editor")
class ImageContrastEditor(t.HasTraits):
    mpl_help = "See the matplotlib SymLogNorm for more information."
    ss_left_value = t.Float()
    ss_right_value = t.Float()
    bins = t.Int(
        100,
        desc="Number of bins used for the histogram.",
        auto_set=False,
        enter_set=True,
    )
    gamma = t.Range(0.1, 3.0, 1.0)
    percentile_range = t.Range(0.0, 100.0)
    vmin_percentile = t.Float(0.0)
    vmax_percentile = t.Float(100.0)

    norm = t.Enum("Linear", "Power", "Log", "Symlog", default="Linear")
    linthresh = t.Range(
        0.0,
        1.0,
        0.01,
        exclude_low=True,
        exclude_high=False,
        desc="Range of value closed to zero, which are "
        f"linearly extrapolated. {mpl_help}",
    )
    linscale = t.Range(
        0.0,
        10.0,
        0.1,
        exclude_low=False,
        exclude_high=False,
        desc=f"Number of decades to use for each half of the linear range. {mpl_help}",
    )
    auto = t.Bool(
        True,
        desc="Adjust automatically the display when changing "
        "navigator indices. Unselect to keep the same display.",
    )

    def __init__(self, image):
        super().__init__()
        self.image = image

        self._init_plot()

        # self._vmin and self._vmax are used to compute the histogram
        # by default, the image display uses these, except when there is a
        # span selector on the histogram. This is implemented in the
        # `_get_current_range` method.
        self._vmin, self._vmax = self.image._vmin, self.image._vmax
        self.gamma = self.image.gamma
        self.linthresh = self.image.linthresh
        self.linscale = self.image.linscale
        if self.image._vmin_percentile is not None:
            self.vmin_percentile = float(self.image._vmin_percentile.split("th")[0])
        if self.image._vmax_percentile is not None:
            self.vmax_percentile = float(self.image._vmax_percentile.split("th")[0])

        # Copy the original value to be used when resetting the display
        self.vmin_original = self._vmin
        self.vmax_original = self._vmax
        self.gamma_original = self.gamma
        self.linthresh_original = self.linthresh
        self.linscale_original = self.linscale
        self.vmin_percentile_original = self.vmin_percentile
        self.vmax_percentile_original = self.vmax_percentile

        if self.image.norm == "auto":
            self.norm = "Linear"
        else:
            self.norm = self.image.norm.capitalize()
        self.norm_original = copy.deepcopy(self.norm)

        self.span_selector = SpanSelector(
            self.ax,
            onselect=self._update_image_contrast,
            onmove_callback=self._update_image_contrast,
            direction="horizontal",
            interactive=True,
            ignore_event_outside=False,
            drag_from_anywhere=True,
            props={"alpha": 0.25, "color": "r"},
            handle_props={"alpha": 0.5, "color": "r"},
            useblit=self.ax.figure.canvas.supports_blit,
        )

        self.plot_histogram()

        if self.image.axes_manager is not None:
            self.image.axes_manager.events.indices_changed.connect(self._reset, [])
            self.hspy_fig.events.closed.connect(
                lambda: self.image.axes_manager.events.indices_changed.disconnect(
                    self._reset
                ),
                [],
            )

            # Disconnect update image to avoid image flickering and reconnect
            # it when necessary in the close method.
            self.image.disconnect()

    def _init_plot(self):
        figsize = matplotlib.rcParamsDefault.get("figure.figsize")
        figsize = figsize[0], figsize[1] / 3
        self.hspy_fig = Signal1DFigure(figsize=figsize)
        self.ax = self.hspy_fig.ax
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.figure.subplots_adjust(0, 0, 1, 1)

    def _gamma_changed(self, old, new):
        if self._vmin == self._vmax:
            return
        self.image.gamma = new
        self._reset(auto=False, indices_changed=False, update_histogram=False)
        self.update_line()

    def _vmin_percentile_changed(self, old, new):
        if isinstance(new, str):
            new = float(new.split("th")[0])
        self.image.vmin = f"{new}th"
        self._reset(auto=True, indices_changed=False)
        self._clear_span_selector()

    def _vmax_percentile_changed(self, old, new):
        if isinstance(new, str):
            new = float(new.split("th")[0])
        self.image.vmax = f"{new}th"
        self._reset(auto=True, indices_changed=False)
        self._clear_span_selector()

    def _auto_changed(self, old, new):
        # Do something only if auto is ticked
        if new:
            self._reset(indices_changed=False, update_histogram=False)
            self._clear_span_selector()

    def _bins_changed(self, old, new):
        if old != new:
            self.update_histogram(clear_selector=False)

    def _norm_changed(self, old, new):
        self.image.norm = new.lower()
        self._reset(auto=False, indices_changed=False, update_histogram=False)
        self.update_line()

    def _linthresh_changed(self, old, new):
        self.image.linthresh = new
        self._reset(auto=False, indices_changed=False, update_histogram=False)

    def _linscale_changed(self, old, new):
        self.image.linscale = new
        self._reset(auto=False, indices_changed=False, update_histogram=False)

    def update_span_selector_traits(self, *args, **kwargs):
        self.ss_left_value, self.ss_right_value = sorted(self._get_current_range())
        self.update_line()

    def _update_image_contrast(self, *args, **kwargs):
        self.update_span_selector_traits(*args, **kwargs)
        self._reset(auto=False, indices_changed=False, update_histogram=False)

    def _get_data(self):
        return self.image._current_data

    def _get_histogram(self, data):
        return numba_histogram(data, bins=self.bins, ranges=(self._vmin, self._vmax))

    def plot_histogram(self, max_num_bins=250):
        """Plot a histogram of the data.

        Parameters
        ----------
        %s

        Returns
        -------
        None

        """
        if self._vmin == self._vmax:
            return
        data = self._get_data()
        # masked data outside vmin/vmax
        data = np.ma.masked_outside(data, self._vmin, self._vmax).compressed()

        # Sturges rule
        sturges_bin_width = np.ptp(data) / (np.log2(data.size) + 1.0)

        iqr = np.subtract(*np.percentile(data, [75, 25]))
        fd_bin_width = 2.0 * iqr * data.size ** (-1.0 / 3.0)

        if fd_bin_width > 0:
            bin_width = min(fd_bin_width, sturges_bin_width)
        else:
            # limited variance: fd_bin_width may be zero
            bin_width = sturges_bin_width

        self.bins = min(int(np.ceil(np.ptp(data) / bin_width)), max_num_bins)
        self.update_histogram()
        self._setup_line()

    plot_histogram.__doc__ %= HISTOGRAM_MAX_BIN_ARGS

    def update_histogram(self, clear_selector=True):
        if self._vmin == self._vmax:
            return

        if hasattr(self, "hist"):
            self.hist.remove()

        self.xaxis = UniformDataAxis(
            scale=(self._vmax - self._vmin) / self.bins,
            offset=self._vmin,
            size=self.bins,
        )
        self.hist_data = self._get_histogram(self._get_data())

        # We don't use blitting for the histogram because it will be part
        # included in the background
        self.hist = self.ax.fill_between(
            self.xaxis.axis,
            self.hist_data,
            step="mid",
            color="C0",
        )

        self.ax.set_xlim(self._vmin, self._vmax)
        if self.hist_data.max() != 0:
            self.ax.set_ylim(0, self.hist_data.max())

        if self.auto and self._is_selector_visible and clear_selector:
            # in auto mode, the displayed contrast cover the full range
            # and we need to reset the span selector
            # no need to clear the line, it will updated
            self.span_selector.clear()

        self.update_line()

        self.ax.figure.canvas.draw()

    def _setup_line(self):
        self.hspy_fig.axis = self.xaxis
        self.line = drawing.signal1d.Signal1DLine()
        self.line.data_function = self._get_data_function
        self.line.set_line_properties(color="C1", type="line")
        # Add the line to the figure
        self.hspy_fig.add_line(self.line)
        self.line.plot()

    def _set_xaxis_line(self):
        cmin, cmax = self._get_current_range()
        self.line.axis = np.linspace(cmin, cmax, self.bins)

    def _get_data_function(self, *args, **kwargs):
        xaxis = self.xaxis.axis
        cmin, cmax = xaxis[0], xaxis[-1]
        max_hist = self.hist_data.max()
        if self.image.norm == "linear":
            values = ((xaxis - cmin) / (cmax - cmin)) * max_hist
        elif self.image.norm == "symlog":
            v = self._sym_log_transform(xaxis)
            values = (v - v[0]) / (v[-1] - v[0]) * max_hist
        elif self.image.norm == "log":
            v = np.log(xaxis)
            values = (v - v[0]) / (v[-1] - v[0]) * max_hist
        else:
            # if "auto" or "power" use the self.gamma value
            values = ((xaxis - cmin) / (cmax - cmin)) ** self.gamma * max_hist

        return values

    def _sym_log_transform(self, arr):
        # adapted from matploltib.colors.SymLogNorm
        arr = arr.copy()
        _linscale_adj = self.linscale / (1.0 - np.e**-1)
        with np.errstate(invalid="ignore"):
            masked = abs(arr) > self.linthresh
        sign = np.sign(arr[masked])
        log = _linscale_adj + np.log(abs(arr[masked]) / self.linthresh)
        log *= sign * self.linthresh
        arr[masked] = log
        arr[~masked] *= _linscale_adj

        return arr

    def update_line(self):
        if not hasattr(self, "line") or self._vmin == self._vmax:
            return
        self._set_xaxis_line()
        self.line.update(render_figure=True)
        if not self.line.line.get_visible():
            # when the selector have been cleared, line is not visible anymore
            self.line.line.set_visible(True)

    def apply(self):
        if self.ss_left_value == self.ss_right_value:
            # No span selector, so we use the default vim and vmax values
            self._reset(auto=True, indices_changed=False)
        else:
            # When we apply the selected range and update the xaxis
            self._vmin, self._vmax = self._get_current_range()
            # Remove the span selector and set the new one ready to use
            self._clear_span_selector()
            self._reset(auto=False, indices_changed=False)

    def reset(self):
        # Reset the display as original
        self._reset_original_settings()
        self._clear_span_selector()
        self._reset(indices_changed=False)

    def _reset_original_settings(self):
        if self.vmin_percentile_original is not None:
            self.vmin_percentile = self.vmin_percentile_original
        if self.vmax_percentile_original is not None:
            self.vmax_percentile = self.vmax_percentile_original
        self._vmin = self.vmin_original
        self._vmax = self.vmax_original
        self.norm = self.norm_original.capitalize()
        self.gamma = self.gamma_original
        self.linthresh = self.linthresh_original
        self.linscale = self.linscale_original

    @property
    def _is_selector_visible(self):
        if hasattr(self, "span_selector"):
            return self.span_selector.artists[0].get_visible()

    def _get_current_range(self):
        # Get the range from the span selector if it is displayed otherwise
        # fallback to the _vmin/_vmax cache values
        if self._is_selector_visible and np.diff(self.span_selector.extents) > 0:
            # if we have a span selector, use it to set the display
            return self.span_selector.extents
        else:
            return self._vmin, self._vmax

    def close(self):
        # And reconnect the image if we close the ImageContrastEditor
        if self.image is not None:
            if self.auto:
                self.image.vmin = f"{self.vmin_percentile}th"
                self.image.vmax = f"{self.vmax_percentile}th"
            else:
                self.image.vmin, self.image.vmax = self._get_current_range()
            self.image.connect()
        self.hspy_fig.close()

    def _reset(self, auto=None, indices_changed=True, update_histogram=True):
        # indices_changed is used for the connection to the indices_changed
        # event of the axes_manager, which will require to update the displayed
        # image
        self.image.norm = self.norm.lower()
        if auto is None:
            auto = self.auto

        if auto:
            # Update the image display, which calculates the _vmin/_vmax
            self.image.update(data_changed=indices_changed, auto_contrast=auto)
            self._vmin, self._vmax = self.image._vmin, self.image._vmax
        else:
            vmin, vmax = self._get_current_range()
            self.image.update(
                data_changed=indices_changed, auto_contrast=auto, vmin=vmin, vmax=vmax
            )

        if update_histogram and hasattr(self, "hist"):
            self.update_histogram()
            self.update_span_selector_traits()

    def _clear_span_selector(self):
        if hasattr(self, "span_selector"):
            self.span_selector.clear()
        if hasattr(self, "line"):
            self.line.line.set_visible(False)
            self.hspy_fig.render_figure()

    def _show_help_fired(self):
        from pyface.message_dialog import information

        _help = _IMAGE_CONTRAST_EDITOR_HELP.replace("PERCENTILE", _PERCENTILE_TRAITSUI)
        _ = (information(None, _help, title="Help"),)


_IMAGE_CONTRAST_EDITOR_HELP = """
<h2>Image contrast editor</h2>
<p>This tool provides controls to adjust the contrast of the image.</p>

<h3>Basic parameters</h3>

<p><b>Auto</b>: If selected, adjust automatically the contrast when changing
nagivation axis by taking into account others parameters.</p>

PERCENTILE

<p><b>Bins</b>: Number of bins used in the histogram calculation</p>

<p><b>Norm</b>: Normalisation used to display the image.</p>

<p><b>Gamma</b>: Paramater of the power law transform, also known as gamma
correction. <i>Only available with the 'power' norm</i>.</p>


<h3>Advanced parameters</h3>

<p><b>Linear threshold</b>: Since the values close to zero tend toward infinity,
there is a need to have a range around zero that is linear.
This allows the user to specify the size of this range around zero.
<i>Only with the 'log' norm and when values <= 0 are displayed</i>.</p>

<p><b>Linear scale</b>: Since the values close to zero tend toward infinity,
there is a need to have a range around zero that is linear.
This allows the user to specify the size of this range around zero.
<i>Only with the 'log' norm and when values <= 0 are displayed</i>.</p>

<h3>Buttons</h3>

<p><b>Apply</b>: Calculate the histogram using the selected range defined by
the range selector.</p>

<p><b>Reset</b>: Reset the settings to their initial values.</p>

<p><b>OK</b>: Close this tool.</p>

"""

_PERCENTILE_TRAITSUI = """<p><b>vmin percentile</b>: The percentile value defining the number of
pixels left out of the lower bounds.</p>

<p><b>vmax percentile</b>: The percentile value defining the number of
pixels left out of the upper bounds.</p>"""

_PERCENTILE_IPYWIDGETS = """<p><b>vmin/vmax percentile</b>: The percentile values defining the number of
pixels left out of the lower and upper bounds.</p>"""

IMAGE_CONTRAST_EDITOR_HELP_IPYWIDGETS = _IMAGE_CONTRAST_EDITOR_HELP.replace(
    "PERCENTILE", _PERCENTILE_IPYWIDGETS
)
