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

import functools
import logging

import matplotlib
import numpy as np
import scipy
import traits.api as t

import hyperspy
from hyperspy.ui_registry import add_gui_method

_logger = logging.getLogger(__name__)


class Smoothing(t.HasTraits):
    line_color_ipy = t.Str("blue")
    differential_order = t.Int(0)

    @property
    def line_color_rgb(self):
        if hasattr(self, "line_color"):
            try:
                # PyQt and WX
                return np.array(self.line_color.Get()) / 255.0
            except AttributeError:
                try:
                    # PySide
                    return np.array(self.line_color.getRgb()) / 255.0
                except BaseException:
                    return matplotlib.colors.to_rgb(self.line_color_ipy)
        else:
            return matplotlib.colors.to_rgb(self.line_color_ipy)

    def __init__(self, signal):
        self.ax = None
        self.data_line = None
        self.smooth_line = None
        self.signal = signal
        self.single_spectrum = self.signal.get_current_signal().deepcopy()
        self.axis = self.signal.axes_manager.signal_axes[0].axis
        self.plot()

    def plot(self):
        if self.signal._plot is None or not self.signal._plot.is_active:
            self.signal.plot()
        hse = self.signal._plot
        l1 = hse.signal_plot.ax_lines[0]
        self.original_color = l1.line.get_color()
        l1.set_line_properties(color=self.original_color, type="scatter")

        l2 = hyperspy.drawing.signal1d.Signal1DLine()
        l2.data_function = self.model2plot

        l2.set_line_properties(color=self.line_color_rgb, type="line")
        # Add the line to the figure
        hse.signal_plot.add_line(l2)
        l2.plot()

        self.data_line = l1
        self.smooth_line = l2
        self.smooth_diff_line = None

    def update_lines(self):
        self.smooth_line.update()
        if self.smooth_diff_line is not None:
            self.smooth_diff_line.update()

    def turn_diff_line_on(self, diff_order):
        self.signal._plot.signal_plot.create_right_axis()
        self.smooth_diff_line = hyperspy.drawing.signal1d.Signal1DLine()
        self.smooth_diff_line.axes_manager = self.signal.axes_manager
        self.smooth_diff_line.data_function = self.diff_model2plot
        self.smooth_diff_line.set_line_properties(
            color=self.line_color_rgb, type="line"
        )
        self.signal._plot.signal_plot.add_line(self.smooth_diff_line, ax="right")

    def _line_color_ipy_changed(self):
        if hasattr(self, "line_color"):
            self.line_color = str(self.line_color_ipy)
        else:
            self._line_color_changed(None, None)

    def turn_diff_line_off(self):
        if self.smooth_diff_line is None:
            return
        self.smooth_diff_line.close()
        self.smooth_diff_line = None

    def _differential_order_changed(self, old, new):
        if new == 0:
            self.turn_diff_line_off()
            return
        if old == 0:
            self.turn_diff_line_on(new)
            self.smooth_diff_line.plot()
        else:
            self.smooth_diff_line.update(force_replot=False)

    def _line_color_changed(self, old, new):
        self.smooth_line.line_properties = {"color": self.line_color_rgb}
        if self.smooth_diff_line is not None:
            self.smooth_diff_line.line_properties = {"color": self.line_color_rgb}
        try:
            # it seems that changing the properties can be done before the
            # first rendering event, which can cause issue with blitting
            self.update_lines()
        except AttributeError:
            pass

    def diff_model2plot(self, axes_manager=None):
        n = self.differential_order
        smoothed = self.model2plot(axes_manager)
        while n:
            smoothed = np.gradient(smoothed, self.axis)
            n -= 1
        return smoothed

    def close(self):
        if self.signal._plot.is_active:
            if self.differential_order != 0:
                self.turn_diff_line_off()
            self.smooth_line.close()
            self.data_line.set_line_properties(color=self.original_color, type="line")


@add_gui_method(toolkey="hyperspy.Signal1D.smooth_savitzky_golay")
class SmoothingSavitzkyGolay(Smoothing):
    polynomial_order = t.Int(
        3,
        desc="The order of the polynomial used to fit the samples."
        "`polyorder` must be less than `window_length`.",
    )

    window_length = t.Int(5, desc="`window_length` must be a positive odd integer.")

    increase_window_length = t.Button(orientation="horizontal", label="+")
    decrease_window_length = t.Button(orientation="horizontal", label="-")

    def _increase_window_length_fired(self):
        if self.window_length % 2:
            nwl = self.window_length + 2
        else:
            nwl = self.window_length + 1
        if nwl < self.signal.axes_manager[2j].size:
            self.window_length = nwl

    def _decrease_window_length_fired(self):
        if self.window_length % 2:
            nwl = self.window_length - 2
        else:
            nwl = self.window_length - 1
        if nwl > self.polynomial_order:
            self.window_length = nwl
        else:
            _logger.warning(
                "The window length must be greater than the polynomial order"
            )

    def _polynomial_order_changed(self, old, new):
        if self.window_length <= new:
            self.window_length = new + 2 if new % 2 else new + 1
            _logger.warning(
                "Polynomial order must be < window length. Window length set to %i.",
                self.window_length,
            )
        self.update_lines()

    def _window_length_changed(self, old, new):
        self.update_lines()

    def _differential_order_changed(self, old, new):
        if new > self.polynomial_order:
            self.polynomial_order += 1
            _logger.warning(
                "Differential order must be <= polynomial order. "
                "Polynomial order set to %i.",
                self.polynomial_order,
            )
        super()._differential_order_changed(old, new)

    def diff_model2plot(self, axes_manager=None):
        self.single_spectrum.data = self.signal._get_current_data().copy()
        self.single_spectrum.smooth_savitzky_golay(
            polynomial_order=self.polynomial_order,
            window_length=self.window_length,
            differential_order=self.differential_order,
        )
        return self.single_spectrum.data

    def model2plot(self, axes_manager=None):
        self.single_spectrum.data = self.signal._get_current_data().copy()
        self.single_spectrum.smooth_savitzky_golay(
            polynomial_order=self.polynomial_order,
            window_length=self.window_length,
            differential_order=0,
        )
        return self.single_spectrum.data

    def apply(self):
        self.signal.smooth_savitzky_golay(
            polynomial_order=self.polynomial_order,
            window_length=self.window_length,
            differential_order=self.differential_order,
        )
        self.signal._replot()


@add_gui_method(toolkey="hyperspy.Signal1D.smooth_lowess")
class SmoothingLowess(Smoothing):
    smoothing_parameter = t.Range(
        low=0.001,
        high=0.99,
        value=0.1,
    )
    number_of_iterations = t.Range(low=1, value=1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _smoothing_parameter_changed(self, old, new):
        if new == 0:
            self.smoothing_parameter = old
        else:
            self.update_lines()

    def _number_of_iterations_changed(self, old, new):
        self.update_lines()

    def model2plot(self, axes_manager=None):
        self.single_spectrum.data = self.signal._get_current_data().copy()
        self.single_spectrum.smooth_lowess(
            smoothing_parameter=self.smoothing_parameter,
            number_of_iterations=self.number_of_iterations,
            show_progressbar=False,
        )

        return self.single_spectrum.data

    def apply(self):
        self.signal.smooth_lowess(
            smoothing_parameter=self.smoothing_parameter,
            number_of_iterations=self.number_of_iterations,
        )
        self.signal._replot()


@add_gui_method(toolkey="hyperspy.Signal1D.smooth_total_variation")
class SmoothingTV(Smoothing):
    smoothing_parameter = t.Float(200)

    def _smoothing_parameter_changed(self, old, new):
        self.update_lines()

    def model2plot(self, axes_manager=None):
        self.single_spectrum.data = self.signal._get_current_data().copy()
        self.single_spectrum.smooth_tv(
            smoothing_parameter=self.smoothing_parameter, show_progressbar=False
        )

        return self.single_spectrum.data

    def apply(self):
        self.signal.smooth_tv(smoothing_parameter=self.smoothing_parameter)
        self.signal._replot()


@add_gui_method(toolkey="hyperspy.Signal1D.smooth_butterworth")
class ButterworthFilter(Smoothing):
    cutoff_frequency_ratio = t.Range(0.01, 1.0, 0.01)
    type = t.Enum("low", "high")
    order = t.Int(2)

    def _cutoff_frequency_ratio_changed(self, old, new):
        self.update_lines()

    def _type_changed(self, old, new):
        self.update_lines()

    def _order_changed(self, old, new):
        self.update_lines()

    def model2plot(self, axes_manager=None):
        b, a = scipy.signal.butter(self.order, self.cutoff_frequency_ratio, self.type)
        smoothed = scipy.signal.filtfilt(b, a, self.signal._get_current_data())
        return smoothed

    def apply(self):
        b, a = scipy.signal.butter(self.order, self.cutoff_frequency_ratio, self.type)
        f = functools.partial(scipy.signal.filtfilt, b, a)
        self.signal.map(f)
