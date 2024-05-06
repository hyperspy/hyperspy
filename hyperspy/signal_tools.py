# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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
import functools
import logging

import matplotlib
import matplotlib.colors
import matplotlib.text as mpl_text
import numpy as np
import traits.api as t
from scipy import interpolate
from scipy import signal as sp_signal

from hyperspy import components1d, drawing
from hyperspy.axes import AxesManager, UniformDataAxis
from hyperspy.component import Component
from hyperspy.docstrings.signal import HISTOGRAM_MAX_BIN_ARGS
from hyperspy.drawing._markers.circles import Circles
from hyperspy.drawing._widgets.range import SpanSelector
from hyperspy.drawing.markers import convert_positions
from hyperspy.drawing.signal1d import Signal1DFigure
from hyperspy.drawing.widgets import Line2DWidget, VerticalLineWidget
from hyperspy.exceptions import SignalDimensionError
from hyperspy.misc.array_tools import numba_histogram
from hyperspy.misc.math_tools import check_random_state
from hyperspy.ui_registry import add_gui_method

_logger = logging.getLogger(__name__)


class LineInSignal2D(t.HasTraits):
    """
    Adds a vertical draggable line to a spectrum that reports its
    position to the position attribute of the class.

    Attributes
    ----------
    x0, y0, x1, y1 : floats
        Position of the line in scaled units.
    length : float
        Length of the line in scaled units.
    on : bool
        Turns on and off the line
    color : wx.Colour
        The color of the line. It automatically redraws the line.

    """

    x0, y0, x1, y1 = t.Float(0.0), t.Float(0.0), t.Float(1.0), t.Float(1.0)
    length = t.Float(1.0)
    is_ok = t.Bool(False)
    on = t.Bool(False)
    # The following is disabled because as of traits 4.6 the Color trait
    # imports traitsui (!)
    # try:
    #     color = t.Color("black")
    # except ModuleNotFoundError:  # traitsui is not installed
    #     pass
    color_str = t.Str("black")

    def __init__(self, signal):
        if signal.axes_manager.signal_dimension != 2:
            raise SignalDimensionError(signal.axes_manager.signal_dimension, 2)

        self.signal = signal
        if (self.signal._plot is None) or (not self.signal._plot.is_active):
            self.signal.plot()
        axis_dict0 = signal.axes_manager.signal_axes[0].get_axis_dictionary()
        axis_dict1 = signal.axes_manager.signal_axes[1].get_axis_dictionary()
        am = AxesManager([axis_dict1, axis_dict0])
        am._axes[0].navigate = True
        am._axes[1].navigate = True
        self.axes_manager = am
        self.on_trait_change(self.switch_on_off, "on")

    def draw(self):
        self.signal._plot.signal_plot.figure.canvas.draw_idle()

    def _get_initial_position(self):
        am = self.axes_manager
        d0 = (am[0].high_value - am[0].low_value) / 10
        d1 = (am[1].high_value - am[1].low_value) / 10
        position = (
            (am[0].low_value + d0, am[1].low_value + d1),
            (am[0].high_value - d0, am[1].high_value - d1),
        )
        return position

    def switch_on_off(self, obj, trait_name, old, new):
        if not self.signal._plot.is_active:
            return

        if new is True and old is False:
            self._line = Line2DWidget(self.axes_manager)
            self._line.position = self._get_initial_position()
            self._line.set_mpl_ax(self.signal._plot.signal_plot.ax)
            self._line.linewidth = 1
            self._color_changed("black", "black")
            self.update_position()
            self._line.events.changed.connect(self.update_position)
            # There is not need to call draw because setting the
            # color calls it.

        elif new is False and old is True:
            self._line.close()
            self._line = None
            self.draw()

    def update_position(self, *args, **kwargs):
        if not self.signal._plot.is_active:
            return
        pos = self._line.position
        (self.x0, self.y0), (self.x1, self.y1) = pos
        self.length = np.linalg.norm(np.diff(pos, axis=0), axis=1)[0]

    def _color_changed(self, old, new):
        if self.on is False:
            return
        self.draw()


@add_gui_method(toolkey="hyperspy.Signal2D.calibrate")
class Signal2DCalibration(LineInSignal2D):
    new_length = t.Float(t.Undefined, label="New length")
    scale = t.Float()
    units = t.Unicode()

    def __init__(self, signal):
        super(Signal2DCalibration, self).__init__(signal)
        if signal.axes_manager.signal_dimension != 2:
            raise SignalDimensionError(signal.axes_manager.signal_dimension, 2)
        self.units = self.signal.axes_manager.signal_axes[0].units
        self.scale = self.signal.axes_manager.signal_axes[0].scale
        self.on = True

    def _new_length_changed(self, old, new):
        # If the line position is invalid or the new length is not defined do
        # nothing
        if (
            np.isnan(self.x0)
            or np.isnan(self.y0)
            or np.isnan(self.x1)
            or np.isnan(self.y1)
            or self.new_length is t.Undefined
        ):
            return
        self.scale = self.signal._get_signal2d_scale(
            self.x0, self.y0, self.x1, self.y1, self.new_length
        )

    def _length_changed(self, old, new):
        # If the line position is invalid or the new length is not defined do
        # nothing
        if (
            np.isnan(self.x0)
            or np.isnan(self.y0)
            or np.isnan(self.x1)
            or np.isnan(self.y1)
            or self.new_length is t.Undefined
        ):
            return
        self.scale = self.signal._get_signal2d_scale(
            self.x0, self.y0, self.x1, self.y1, self.new_length
        )

    def apply(self):
        if self.new_length is t.Undefined:
            _logger.warning("Input a new length before pressing apply.")
            return
        x0, y0, x1, y1 = self.x0, self.y0, self.x1, self.y1
        if np.isnan(x0) or np.isnan(y0) or np.isnan(x1) or np.isnan(y1):
            _logger.warning("Line position is not valid")
            return
        self.signal._calibrate(
            x0=x0, y0=y0, x1=x1, y1=y1, new_length=self.new_length, units=self.units
        )
        self.signal._replot()


class SpanSelectorInSignal1D(t.HasTraits):
    ss_left_value = t.Float(np.nan)
    ss_right_value = t.Float(np.nan)
    is_ok = t.Bool(False)

    def __init__(self, signal):
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(signal.axes_manager.signal_dimension, 1)

        # Plot the signal (or model) if it is not already plotted
        if signal._plot is None or not signal._plot.is_active:
            signal.plot()

        from hyperspy.model import BaseModel

        if isinstance(signal, BaseModel):
            signal = signal.signal

        self.signal = signal
        self.axis = self.signal.axes_manager.signal_axes[0]
        self.span_selector = None

        self.span_selector_switch(on=True)

        self.signal._plot.signal_plot.events.closed.connect(self.disconnect, [])

    def on_disabling_span_selector(self):
        self.disconnect()

    def span_selector_switch(self, on):
        if not self.signal._plot.is_active:
            return

        if on is True:
            if self.span_selector is None:
                ax = self.signal._plot.signal_plot.ax
                self.span_selector = SpanSelector(
                    ax=ax,
                    onselect=lambda *args, **kwargs: None,
                    onmove_callback=self.span_selector_changed,
                    direction="horizontal",
                    interactive=True,
                    ignore_event_outside=True,
                    drag_from_anywhere=True,
                    props={"alpha": 0.25, "color": "r"},
                    handle_props={"alpha": 0.5, "color": "r"},
                    useblit=ax.figure.canvas.supports_blit,
                )
                self.connect()

        elif self.span_selector is not None:
            self.on_disabling_span_selector()
            self.span_selector.disconnect_events()
            self.span_selector.clear()
            self.span_selector = None

    def span_selector_changed(self, *args, **kwargs):
        if not self.signal._plot.is_active:
            return

        x0, x1 = sorted(self.span_selector.extents)

        # typically during initialisation
        if x0 == x1:
            return

        # range of span selector invalid
        if x0 < self.axis.low_value:
            x0 = self.axis.low_value
        if x1 > self.axis.high_value or x1 < self.axis.low_value:
            x1 = self.axis.high_value

        if np.diff(self.axis.value2index(np.array([x0, x1]))) == 0:
            return

        self.ss_left_value, self.ss_right_value = x0, x1

    def reset_span_selector(self):
        self.span_selector_switch(False)
        self.ss_left_value = np.nan
        self.ss_right_value = np.nan
        self.span_selector_switch(True)

    @property
    def _is_valid_range(self):
        return (
            self.span_selector is not None
            and not np.isnan([self.ss_left_value, self.ss_right_value]).any()
        )

    def _reset_span_selector_background(self):
        if self.span_selector is not None:
            # For matplotlib backend supporting blit, we need to reset the
            # background when the data displayed on the figure is changed,
            # otherwise, when the span selector is updated, old background is
            # restore
            self.span_selector.background = None
            # Trigger callback
            self.span_selector_changed()

    def connect(self):
        for event in [
            self.signal.events.data_changed,
            self.signal.axes_manager.events.indices_changed,
        ]:
            event.connect(self._reset_span_selector_background, [])

    def disconnect(self):
        function = self._reset_span_selector_background
        for event in [
            self.signal.events.data_changed,
            self.signal.axes_manager.events.indices_changed,
        ]:
            if function in event.connected:
                event.disconnect(function)


class LineInSignal1D(t.HasTraits):
    """Adds a vertical draggable line to a spectrum that reports its
    position to the position attribute of the class.

    Attributes
    ----------
    position : float
        The position of the vertical line in the one dimensional signal. Moving
        the line changes the position but the reverse is not true.
    on : bool
        Turns on and off the line
    color : wx.Colour
        The color of the line. It automatically redraws the line.

    """

    position = t.Float()
    is_ok = t.Bool(False)
    on = t.Bool(False)
    # The following is disabled because as of traits 4.6 the Color trait
    # imports traitsui (!)
    # try:
    #     color = t.Color("black")
    # except ModuleNotFoundError:  # traitsui is not installed
    #     pass
    color_str = t.Str("black")

    def __init__(self, signal):
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(signal.axes_manager.signal_dimension, 1)

        self.signal = signal
        self.signal.plot()
        axis_dict = signal.axes_manager.signal_axes[0].get_axis_dictionary()
        am = AxesManager(
            [
                axis_dict,
            ]
        )
        am._axes[0].navigate = True
        # Set the position of the line in the middle of the spectral
        # range by default
        am._axes[0].index = int(round(am._axes[0].size / 2))
        self.axes_manager = am
        self.axes_manager.events.indices_changed.connect(self.update_position, [])
        self.on_trait_change(self.switch_on_off, "on")

    def draw(self):
        self.signal._plot.signal_plot.figure.canvas.draw_idle()

    def switch_on_off(self, obj, trait_name, old, new):
        if not self.signal._plot.is_active:
            return

        if new is True and old is False:
            self._line = VerticalLineWidget(self.axes_manager)
            self._line.set_mpl_ax(self.signal._plot.signal_plot.ax)
            self._line.patch.set_linewidth(2)
            self._color_changed("black", "black")
            # There is not need to call draw because setting the
            # color calls it.

        elif new is False and old is True:
            self._line.close()
            self._line = None
            self.draw()

    def update_position(self, *args, **kwargs):
        if not self.signal._plot.is_active:
            return
        self.position = self.axes_manager.coordinates[0]

    def _color_changed(self, old, new):
        if self.on is False:
            return

        self._line.patch.set_color(
            (
                self.color.Red() / 255.0,
                self.color.Green() / 255.0,
                self.color.Blue() / 255.0,
            )
        )
        self.draw()


@add_gui_method(toolkey="hyperspy.Signal1D.calibrate")
class Signal1DCalibration(SpanSelectorInSignal1D):
    left_value = t.Float(t.Undefined, label="New left value")
    right_value = t.Float(t.Undefined, label="New right value")
    offset = t.Float()
    scale = t.Float()
    units = t.Unicode()

    def __init__(self, signal):
        super().__init__(signal)
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(signal.axes_manager.signal_dimension, 1)
        if not isinstance(self.axis, UniformDataAxis):
            raise NotImplementedError(
                "The calibration tool supports only uniform axes."
            )
        self.units = self.axis.units
        self.scale = self.axis.scale
        self.offset = self.axis.offset
        self.last_calibration_stored = True
        self.span_selector.snap_values = self.axis.axis

    def _left_value_changed(self, old, new):
        if self._is_valid_range and self.right_value is not t.Undefined:
            self._update_calibration()

    def _right_value_changed(self, old, new):
        if self._is_valid_range and self.left_value is not t.Undefined:
            self._update_calibration()

    def _update_calibration(self, *args, **kwargs):
        # If the span selector or the new range values are not defined do
        # nothing
        if not self._is_valid_range or self.signal._plot.signal_plot is None:
            return
        lc = self.axis.value2index(self.ss_left_value)
        rc = self.axis.value2index(self.ss_right_value)
        self.offset, self.scale = self.axis.calibrate(
            (self.left_value, self.right_value), (lc, rc), modify_calibration=False
        )

    def apply(self):
        if not self._is_valid_range:
            _logger.warning(
                "Select a range by clicking on the signal figure "
                "and dragging before pressing Apply."
            )
            return
        elif self.left_value is t.Undefined or self.right_value is t.Undefined:
            _logger.warning(
                "Select the new left and right values before " "pressing apply."
            )
            return
        axis = self.axis
        axis.scale = self.scale
        axis.offset = self.offset
        axis.units = self.units
        self.span_selector_switch(on=False)
        self.signal._replot()
        self.span_selector_switch(on=True)
        self.last_calibration_stored = True


class Signal1DRangeSelector(SpanSelectorInSignal1D):
    on_close = t.List()


class Smoothing(t.HasTraits):
    # The following is disabled because as of traits 4.6 the Color trait
    # imports traitsui (!)
    # try:
    #     line_color = t.Color("blue")
    # except ModuleNotFoundError:
    #     # traitsui is required to define this trait so it is not defined when
    #     # traitsui is not installed.
    #     pass
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

        l2 = drawing.signal1d.Signal1DLine()
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
        self.smooth_diff_line = drawing.signal1d.Signal1DLine()
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
                "Polynomial order must be < window length. " "Window length set to %i.",
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
        b, a = sp_signal.butter(self.order, self.cutoff_frequency_ratio, self.type)
        smoothed = sp_signal.filtfilt(b, a, self.signal._get_current_data())
        return smoothed

    def apply(self):
        b, a = sp_signal.butter(self.order, self.cutoff_frequency_ratio, self.type)
        f = functools.partial(sp_signal.filtfilt, b, a)
        self.signal.map(f)


class Load(t.HasTraits):
    filename = t.File
    lazy = t.Bool(False)


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
        desc="Number of decades to use for each half of "
        f"the linear range. {mpl_help}",
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
        sturges_bin_width = data.ptp() / (np.log2(data.size) + 1.0)

        iqr = np.subtract(*np.percentile(data, [75, 25]))
        fd_bin_width = 2.0 * iqr * data.size ** (-1.0 / 3.0)

        if fd_bin_width > 0:
            bin_width = min(fd_bin_width, sturges_bin_width)
        else:
            # limited variance: fd_bin_width may be zero
            bin_width = sturges_bin_width

        self.bins = min(int(np.ceil(data.ptp() / bin_width)), max_num_bins)
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


@add_gui_method(toolkey="hyperspy.Signal1D.remove_background")
class BackgroundRemoval(SpanSelectorInSignal1D):
    background_type = t.Enum(
        "Doniach",
        "Exponential",
        "Gaussian",
        "Lorentzian",
        "Offset",
        "Polynomial",
        "Power law",
        "Skew normal",
        "Split Voigt",
        "Voigt",
        default="Power law",
    )
    polynomial_order = t.Range(1, 10)
    fast = t.Bool(
        True,
        desc=(
            "Perform a fast (analytic, but possibly less accurate)"
            " estimation of the background. Otherwise use "
            "non-linear least squares."
        ),
    )
    zero_fill = t.Bool(
        False,
        desc=(
            "Set all spectral channels lower than the lower \n"
            "bound of the fitting range to zero (this is the \n"
            "default behavior of Gatan's DigitalMicrograph). \n"
            "Otherwise leave the pre-fitting region as-is \n"
            "(useful for inspecting quality of background fit)."
        ),
    )
    background_estimator = t.Instance(Component)
    bg_line_range = t.Enum("from_left_range", "full", "ss_range", default="full")
    red_chisq = t.Float(np.nan)

    def __init__(
        self,
        signal,
        background_type="Power law",
        polynomial_order=2,
        fast=True,
        plot_remainder=True,
        zero_fill=False,
        show_progressbar=None,
        model=None,
    ):
        super().__init__(signal)
        # setting the polynomial order will change the backgroud_type to
        # polynomial, so we set it before setting the background type
        self.bg_line = None
        self.rm_line = None
        self.background_estimator = None
        self.fast = fast
        self.plot_remainder = plot_remainder
        if plot_remainder:
            # When plotting the remainder on the right hand side axis, we
            # adjust the layout here to avoid doing it later to avoid
            # corrupting the background when using blitting
            figure = signal._plot.signal_plot.figure
            figure.tight_layout(rect=[0, 0, 0.95, 1])
        if model is None:
            from hyperspy.models.model1d import Model1D

            model = Model1D(signal)
        self.model = model
        self.polynomial_order = polynomial_order
        if background_type in ["Power Law", "PowerLaw"]:
            background_type = "Power law"
        if background_type in ["Skew Normal", "SkewNormal"]:
            background_type = "Skew normal"
        if background_type in ["Split voigt", "SplitVoigt"]:
            background_type = "Split Voigt"
        self.background_type = background_type
        self.zero_fill = zero_fill
        self.show_progressbar = show_progressbar
        self.set_background_estimator()

    def on_disabling_span_selector(self):
        # Disconnect event
        super().on_disabling_span_selector()
        if self.bg_line is not None:
            self.bg_line.close()
            self.bg_line = None
        if self.rm_line is not None:
            self.rm_line.close()
            self.rm_line = None
            self.signal._plot.signal_plot.close_right_axis()

    def set_background_estimator(self):
        if self.model is not None:
            for component in self.model:
                self.model.remove(component)
        self.background_estimator, self.bg_line_range = _get_background_estimator(
            self.background_type, self.polynomial_order
        )
        if self.model is not None and len(self.model) == 0:
            self.model.append(self.background_estimator)
        if not self.fast and self._is_valid_range:
            self.background_estimator.estimate_parameters(
                self.signal, self.ss_left_value, self.ss_right_value, only_current=True
            )

    def _polynomial_order_changed(self, old, new):
        self.set_background_estimator()
        self.span_selector_changed()

    def _background_type_changed(self, old, new):
        self.set_background_estimator()
        self.span_selector_changed()

    def _fast_changed(self, old, new):
        if not self._is_valid_range:
            return
        self._fit()
        self._update_line()

    def create_background_line(self):
        self.bg_line = drawing.signal1d.Signal1DLine()
        self.bg_line.data_function = self.bg_to_plot
        self.bg_line.set_line_properties(color="blue", type="line", scaley=False)
        self.signal._plot.signal_plot.add_line(self.bg_line)
        self.bg_line.autoscale = ""
        self.bg_line.plot()

    def create_remainder_line(self):
        self.rm_line = drawing.signal1d.Signal1DLine()
        self.rm_line.data_function = self.rm_to_plot
        self.rm_line.set_line_properties(color="green", type="line", scaley=False)
        self.signal._plot.signal_plot.create_right_axis(
            color="green", adjust_layout=False
        )
        self.signal._plot.signal_plot.add_line(self.rm_line, ax="right")
        self.rm_line.plot()

    def bg_to_plot(self, axes_manager=None, fill_with=np.nan):
        if self.bg_line_range == "from_left_range":
            bg_array = np.zeros(self.axis.axis.shape)
            bg_array[:] = fill_with
            from_index = self.axis.value2index(self.ss_left_value)
            bg_array[from_index:] = self.background_estimator.function(
                self.axis.axis[from_index:]
            )
            to_return = bg_array
        elif self.bg_line_range == "full":
            to_return = self.background_estimator.function(self.axis.axis)
        elif self.bg_line_range == "ss_range":
            bg_array = np.zeros(self.axis.axis.shape)
            bg_array[:] = fill_with
            from_index = self.axis.value2index(self.ss_left_value)
            to_index = self.axis.value2index(self.ss_right_value)
            bg_array[from_index:] = self.background_estimator.function(
                self.axis.axis[from_index:to_index]
            )
            to_return = bg_array

        if self.axis.is_binned:
            if self.axis.is_uniform:
                to_return *= self.axis.scale
            else:
                to_return *= np.gradient(self.axis.axis)
        return to_return

    def rm_to_plot(self, axes_manager=None, fill_with=np.nan):
        return self.signal._get_current_data() - self.bg_line.line.get_ydata()

    def span_selector_changed(self, *args, **kwargs):
        super().span_selector_changed()
        if not self._is_valid_range:
            return
        try:
            self._fit()
            self._update_line()
        except Exception:
            pass

    def _fit(self):
        if not self._is_valid_range:
            return
        # Set signal range here to set correctly the _channel_switches for
        # the chisq calculation when using fast
        self.model.set_signal_range(self.ss_left_value, self.ss_right_value)
        if self.fast:
            self.background_estimator.estimate_parameters(
                self.signal, self.ss_left_value, self.ss_right_value, only_current=True
            )
            # Calculate chisq
            self.model._calculate_chisq()
        else:
            self.model.fit()
        self.red_chisq = self.model.red_chisq.data[self.model.axes_manager.indices][0]

    def _update_line(self):
        if self.bg_line is None:
            self.create_background_line()
        else:
            self.bg_line.update(
                render_figure=not self.plot_remainder, update_ylimits=False
            )
        if self.plot_remainder:
            if self.rm_line is None:
                self.create_remainder_line()
            else:
                self.rm_line.update(render_figure=True, update_ylimits=True)

    def apply(self):
        if not self._is_valid_range:
            return
        return_model = self.model is not None
        result = self.signal._remove_background_cli(
            signal_range=(self.ss_left_value, self.ss_right_value),
            background_estimator=self.background_estimator,
            fast=self.fast,
            zero_fill=self.zero_fill,
            show_progressbar=self.show_progressbar,
            model=self.model,
            return_model=return_model,
        )
        new_spectra = result[0] if return_model else result
        self.signal.data = new_spectra.data
        self.signal.events.data_changed.trigger(self)

    def disconnect(self):
        super().disconnect()
        axes_manager = self.signal.axes_manager
        for f in [self._fit, self.model._on_navigating]:
            if f in axes_manager.events.indices_changed.connected:
                axes_manager.events.indices_changed.disconnect(f)


def _get_background_estimator(background_type, polynomial_order=1):
    """
    Assign 1D component to specified background type.

    Parameters
    ----------
    background_type : str
        The name of the component to model the background.
    polynomial_order : int, optional
        The polynomial order used in the polynomial component

    Raises
    ------
    ValueError
        When the background type is not a valid string.

    Returns
    -------
    background_estimator : Component1D
        The component mdeling the background.
    bg_line_range : 'full' or 'from_left_range'
        The range to draw the component (used in the BackgroundRemoval tool)

    """
    background_type = background_type.lower().replace(" ", "")
    if background_type == "doniach":
        background_estimator = components1d.Doniach()
        bg_line_range = "full"
    elif background_type == "gaussian":
        background_estimator = components1d.Gaussian()
        bg_line_range = "full"
    elif background_type == "lorentzian":
        background_estimator = components1d.Lorentzian()
        bg_line_range = "full"
    elif background_type == "offset":
        background_estimator = components1d.Offset()
        bg_line_range = "full"
    elif background_type == "polynomial":
        background_estimator = components1d.Polynomial(order=polynomial_order)
        bg_line_range = "full"
    elif background_type == "powerlaw":
        background_estimator = components1d.PowerLaw()
        bg_line_range = "from_left_range"
    elif background_type == "exponential":
        background_estimator = components1d.Exponential()
        bg_line_range = "from_left_range"
    elif background_type == "skewnormal":
        background_estimator = components1d.SkewNormal()
        bg_line_range = "full"
    elif background_type == "splitvoigt":
        background_estimator = components1d.SplitVoigt()
        bg_line_range = "full"
    elif background_type == "voigt":
        background_estimator = components1d.Voigt()
        bg_line_range = "full"
    else:
        raise ValueError(f"Background type '{background_type}' not recognized.")

    return background_estimator, bg_line_range


SPIKES_REMOVAL_INSTRUCTIONS = (
    "To remove spikes from the data:\n\n"
    '   1. Click "Show derivative histogram" to '
    "determine at what magnitude the spikes are present.\n"
    "   2. Enter a suitable threshold (lower than the "
    "lowest magnitude outlier in the histogram) in the "
    '"Threshold" box, which will be the magnitude '
    "from which to search. \n"
    '   3. Click "Find next" to find the first spike.\n'
    "   4. If desired, the width and position of the "
    "boundaries used to replace the spike can be "
    "adjusted by clicking and dragging on the displayed "
    "plot.\n "
    "   5. View the spike (and the replacement data that "
    'will be added) and click "Remove spike" in order '
    "to alter the data as shown. The tool will "
    "automatically find the next spike to replace.\n"
    "   6. Repeat this process for each spike throughout "
    "the dataset, until the end of the dataset is "
    "reached.\n"
    '   7. Click "OK" when finished to close the spikes '
    "removal tool.\n\n"
    "Note: Various settings can be configured in "
    'the "Advanced settings" section. Hover the '
    "mouse over each parameter for a description of what "
    "it does."
    "\n"
)


@add_gui_method(toolkey="hyperspy.SimpleMessage")
class SimpleMessage(t.HasTraits):
    text = t.Str

    def __init__(self, text=""):
        self.text = text


class SpikesRemoval:
    def __init__(
        self,
        signal,
        navigation_mask=None,
        signal_mask=None,
        threshold="auto",
        default_spike_width=5,
        add_noise=True,
        max_num_bins=1000,
        random_state=None,
    ):
        self.ss_left_value = np.nan
        self.ss_right_value = np.nan
        self.default_spike_width = default_spike_width
        self.add_noise = add_noise
        self.signal_mask = signal_mask
        self.navigation_mask = navigation_mask
        self.interpolated_line = None
        self.coordinates = [
            coordinate
            for coordinate in signal.axes_manager._am_indices_generator()
            if (navigation_mask is None or not navigation_mask[coordinate[::-1]])
        ]
        self.signal = signal
        self.axis = self.signal.axes_manager.signal_axes[0]
        if len(self.coordinates) > 1:
            signal.axes_manager.indices = self.coordinates[0]
        if threshold == "auto":
            # Find the first zero of the spikes diagnosis plot
            hist = signal._spikes_diagnosis(
                signal_mask=signal_mask,
                navigation_mask=navigation_mask,
                max_num_bins=max_num_bins,
                show_plot=False,
                use_gui=False,
            )
            zero_index = np.where(hist.data == 0)[0]
            if zero_index.shape[0] > 0:
                index = zero_index[0]
            else:
                index = hist.data.shape[0] - 1
            threshold = np.ceil(hist.axes_manager[0].index2value(index))
            _logger.info(f"Threshold value: {threshold}")
        self.argmax = None
        self.derivmax = None
        self.spline_order = 1
        self._temp_mask = np.zeros(self.signal._get_current_data().shape, dtype="bool")
        self.index = 0
        self.threshold = threshold
        md = self.signal.metadata
        from hyperspy.signal import BaseSignal

        self._rng = check_random_state(random_state)

        if "Signal.Noise_properties" in md:
            if "Signal.Noise_properties.variance" in md:
                self.noise_variance = md.Signal.Noise_properties.variance
                if isinstance(md.Signal.Noise_properties.variance, BaseSignal):
                    self.noise_type = "heteroscedastic"
                else:
                    self.noise_type = "white"
            else:
                self.noise_type = "shot noise"
        else:
            self.noise_type = "shot noise"

    def detect_spike(self):
        axis = self.signal.axes_manager.signal_axes[-1].axis
        derivative = np.gradient(self.signal._get_current_data(), axis)
        if self.signal_mask is not None:
            derivative[self.signal_mask] = 0
        if self.argmax is not None:
            left, right = self.get_interpolation_range()
            # Don't search for spikes in the are where one has
            # been found next time `find` is called.
            self._temp_mask[left : right + 1] = True
            derivative[self._temp_mask] = 0
        if abs(derivative.max()) >= self.threshold:
            self.argmax = derivative.argmax()
            self.derivmax = abs(derivative.max())
            return True
        else:
            return False

    def find(self, back=False):
        ncoordinates = len(self.coordinates)
        spike = self.detect_spike()
        with self.signal.axes_manager.events.indices_changed.suppress():
            while not spike and (
                (self.index < ncoordinates - 1 and back is False)
                or (self.index > 0 and back is True)
            ):
                if back is False:
                    self.index += 1
                else:
                    self.index -= 1
                self._index_changed(self.index, self.index)
                spike = self.detect_spike()

        return spike

    def _index_changed(self, old, new):
        self.signal.axes_manager.indices = self.coordinates[new]
        self.argmax = None
        self._temp_mask[:] = False

    def get_interpolation_range(self):
        axis = self.signal.axes_manager.signal_axes[0]
        if hasattr(self, "span_selector") and self._is_valid_range:
            left = axis.value2index(self.ss_left_value)
            right = axis.value2index(self.ss_right_value)
        else:
            left = self.argmax - self.default_spike_width
            right = self.argmax + self.default_spike_width

        # Clip to the axis dimensions
        nchannels = self.signal.axes_manager.signal_shape[0]
        left = left if left >= 0 else 0
        right = right if right < nchannels else nchannels - 1

        return left, right

    def get_interpolated_spectrum(self, axes_manager=None):
        data = self.signal._get_current_data().copy()
        axis = self.signal.axes_manager.signal_axes[0]
        left, right = self.get_interpolation_range()
        pad = self.spline_order
        ileft = left - pad
        iright = right + pad
        ileft = np.clip(ileft, 0, len(data))
        iright = np.clip(iright, 0, len(data))
        left = int(np.clip(left, 0, len(data)))
        right = int(np.clip(right, 0, len(data)))
        if ileft == 0:
            # Extrapolate to the left
            if right == iright:
                right -= 1
            data[:right] = data[right:iright].mean()

        elif iright == len(data):
            # Extrapolate to the right
            if left == ileft:
                left += 1
            data[left:] = data[ileft:left].mean()

        else:
            # Interpolate
            x = np.hstack((axis.axis[ileft:left], axis.axis[right:iright]))
            y = np.hstack((data[ileft:left], data[right:iright]))
            intp = interpolate.make_interp_spline(x, y, k=self.spline_order)
            data[left:right] = intp(axis.axis[left:right])

        # Add noise
        if self.add_noise is True:
            if self.noise_type == "white":
                data[left:right] += self._rng.normal(
                    scale=np.sqrt(self.noise_variance), size=right - left
                )
            elif self.noise_type == "heteroscedastic":
                noise_variance = self.noise_variance(
                    axes_manager=self.signal.axes_manager
                )[left:right]
                noise = [
                    self._rng.normal(scale=np.sqrt(item)) for item in noise_variance
                ]
                data[left:right] += noise
            else:
                data[left:right] = self._rng.poisson(
                    np.clip(data[left:right], 0, np.inf)
                )

        return data

    def remove_all_spikes(self):
        spike = self.find()
        while spike:
            self.signal._get_current_data()[:] = self.get_interpolated_spectrum()
            spike = self.find()


@add_gui_method(toolkey="hyperspy.Signal1D.spikes_removal_tool")
class SpikesRemovalInteractive(SpikesRemoval, SpanSelectorInSignal1D):
    threshold = t.Float(
        400, desc="the derivative magnitude threshold above\n" "which to find spikes"
    )
    click_to_show_instructions = t.Button()
    show_derivative_histogram = t.Button()
    spline_order = t.Range(
        1,
        10,
        1,
        desc="the order of the spline used to\n" "connect the reconstructed data",
    )
    interpolator = None
    default_spike_width = t.Int(
        5,
        desc="the width over which to do the interpolation\n"
        "when removing a spike (this can be "
        "adjusted for each\nspike by clicking "
        "and dragging on the display during\n"
        "spike replacement)",
    )
    index = t.Int(0)
    add_noise = t.Bool(
        True,
        desc="whether to add noise to the interpolated\nportion"
        "of the spectrum. The noise properties defined\n"
        "in the Signal metadata are used if present,"
        "otherwise\nshot noise is used as a default",
    )

    def __init__(self, signal, max_num_bins=1000, **kwargs):
        SpanSelectorInSignal1D.__init__(self, signal=signal)
        signal._plot.auto_update_plot = False
        self.line = signal._plot.signal_plot.ax_lines[0]
        self.ax = signal._plot.signal_plot.ax
        SpikesRemoval.__init__(self, signal=signal, **kwargs)
        self.update_signal_mask()
        self.max_num_bins = max_num_bins

    def _threshold_changed(self, old, new):
        self.index = 0
        self.update_plot()

    def _click_to_show_instructions_fired(self):
        from pyface.message_dialog import information

        _ = (information(None, SPIKES_REMOVAL_INSTRUCTIONS, title="Instructions"),)

    def _show_derivative_histogram_fired(self):
        self.signal._spikes_diagnosis(
            signal_mask=self.signal_mask,
            navigation_mask=self.navigation_mask,
            max_num_bins=self.max_num_bins,
            show_plot=True,
            use_gui=True,
        )

    def _reset_line(self):
        if self.interpolated_line is not None:
            self.interpolated_line.close()
            self.interpolated_line = None
            self.reset_span_selector()

    def find(self, back=False):
        self._reset_line()
        spike = super().find(back=back)

        if spike is False:
            m = SimpleMessage()
            m.text = "End of dataset reached"
            try:
                m.gui()
            except (NotImplementedError, ImportError):
                # This is only available for traitsui, ipywidgets has a
                # progress bar instead.
                pass
            except ValueError as error:
                _logger.warning(error)
            self.index = 0
            self._reset_line()
            return
        else:
            minimum = max(0, self.argmax - 50)
            maximum = min(len(self.signal._get_current_data()) - 1, self.argmax + 50)
            thresh_label = DerivativeTextParameters(
                text=r"$\mathsf{\delta}_\mathsf{max}=$", color="black"
            )
            self.ax.legend(
                [thresh_label],
                [repr(int(self.derivmax))],
                handler_map={DerivativeTextParameters: DerivativeTextHandler()},
                loc="best",
            )
            self.ax.set_xlim(
                self.signal.axes_manager.signal_axes[0].index2value(minimum),
                self.signal.axes_manager.signal_axes[0].index2value(maximum),
            )
            if self.signal._plot.navigator_plot is not None:
                self.signal._plot.pointer._set_indices(self.coordinates[self.index])
            self.update_plot()
            self.create_interpolation_line()

    def update_plot(self):
        if self.interpolated_line is not None:
            self.interpolated_line.close()
            self.interpolated_line = None
        self.reset_span_selector()
        self.update_spectrum_line()
        self.update_signal_mask()
        if len(self.coordinates) > 1:
            self.signal._plot.pointer._on_navigate(self.signal.axes_manager)

    def update_signal_mask(self):
        if hasattr(self, "mask_filling"):
            self.mask_filling.remove()
        if self.signal_mask is not None:
            self.mask_filling = self.ax.fill_between(
                self.axis.axis,
                self.signal._get_current_data(),
                0,
                where=self.signal_mask,
                facecolor="blue",
                alpha=0.5,
            )

    def update_spectrum_line(self):
        self.line.auto_update = True
        self.line.update()
        self.line.auto_update = False

    def on_disabling_span_selector(self):
        super().on_disabling_span_selector()
        if self.interpolated_line is not None:
            self.interpolated_line.close()
            self.interpolated_line = None

    def _spline_order_changed(self, old, new):
        if new != old:
            self.spline_order = new
            self.span_selector_changed()

    def _add_noise_changed(self, old, new):
        self.span_selector_changed()

    def create_interpolation_line(self):
        self.interpolated_line = drawing.signal1d.Signal1DLine()
        self.interpolated_line.data_function = self.get_interpolated_spectrum
        self.interpolated_line.set_line_properties(color="blue", type="line")
        self.signal._plot.signal_plot.add_line(self.interpolated_line)
        self.interpolated_line.auto_update = False
        self.interpolated_line.autoscale = ""
        self.interpolated_line.plot()

    def span_selector_changed(self, *args, **kwargs):
        super().span_selector_changed()
        if self.interpolated_line is None:
            return
        else:
            self.interpolated_line.update()

    def apply(self):
        if not self.interpolated_line:  # No spike selected
            return
        self.signal._get_current_data()[:] = self.get_interpolated_spectrum()
        self.signal.events.data_changed.trigger(obj=self.signal)
        self.update_spectrum_line()
        self.interpolated_line.close()
        self.interpolated_line = None
        self.reset_span_selector()
        self.find()


@add_gui_method(toolkey="hyperspy.Signal2D.find_peaks")
class PeaksFinder2D(t.HasTraits):
    method = t.Enum(
        "Local max",
        "Max",
        "Minmax",
        "Zaefferer",
        "Stat",
        "Laplacian of Gaussian",
        "Difference of Gaussian",
        "Template matching",
        default="Local Max",
    )
    # For "Local max" method
    local_max_distance = t.Range(1, 20, value=3)
    local_max_threshold = t.Range(0, 20.0, value=10)
    # For "Max" method
    max_alpha = t.Range(0, 6.0, value=3)
    max_distance = t.Range(1, 20, value=10)
    # For "Minmax" method
    minmax_distance = t.Range(0, 6.0, value=3)
    minmax_threshold = t.Range(0, 20.0, value=10)
    # For "Zaefferer" method
    zaefferer_grad_threshold = t.Range(0, 0.2, value=0.1)
    zaefferer_window_size = t.Range(2, 80, value=40)
    zaefferer_distance_cutoff = t.Range(1, 100.0, value=50)
    # For "Stat" method
    stat_alpha = t.Range(0, 2.0, value=1)
    stat_window_radius = t.Range(5, 20, value=10)
    stat_convergence_ratio = t.Range(0, 0.1, value=0.05)
    # For "Laplacian of Gaussian" method
    log_min_sigma = t.Range(0, 2.0, value=1)
    log_max_sigma = t.Range(0, 100.0, value=50)
    log_num_sigma = t.Range(0, 20, value=10)
    log_threshold = t.Range(0, 0.4, value=0.2)
    log_overlap = t.Range(0, 1.0, value=0.5)
    log_log_scale = t.Bool(False)
    # For "Difference of Gaussian" method
    dog_min_sigma = t.Range(0, 2.0, value=1)
    dog_max_sigma = t.Range(0, 100.0, value=50)
    dog_sigma_ratio = t.Range(0, 3.2, value=1.6)
    dog_threshold = t.Range(0, 0.4, value=0.2)
    dog_overlap = t.Range(0, 1.0, value=0.5)
    # For "Cross correlation" method
    xc_template = None
    xc_distance = t.Range(0, 100.0, value=5.0)
    xc_threshold = t.Range(0, 10.0, value=0.5)

    random_navigation_position = t.Button()
    compute_over_navigation_axes = t.Button()

    show_navigation_sliders = t.Bool(False)

    def __init__(self, signal, method, peaks=None, **kwargs):
        self._attribute_argument_mapping_local_max = {
            "local_max_distance": "min_distance",
            "local_max_threshold": "threshold_abs",
        }
        self._attribute_argument_mapping_max = {
            "max_alpha": "alpha",
            "max_distance": "distance",
        }
        self._attribute_argument_mapping_local_minmax = {
            "minmax_distance": "distance",
            "minmax_threshold": "threshold",
        }
        self._attribute_argument_mapping_local_zaefferer = {
            "zaefferer_grad_threshold": "grad_threshold",
            "zaefferer_window_size": "window_size",
            "zaefferer_distance_cutoff": "distance_cutoff",
        }
        self._attribute_argument_mapping_local_stat = {
            "stat_alpha": "alpha",
            "stat_window_radius": "window_radius",
            "stat_convergence_ratio": "convergence_ratio",
        }
        self._attribute_argument_mapping_local_log = {
            "log_min_sigma": "min_sigma",
            "log_max_sigma": "max_sigma",
            "log_num_sigma": "num_sigma",
            "log_threshold": "threshold",
            "log_overlap": "overlap",
            "log_log_scale": "log_scale",
        }
        self._attribute_argument_mapping_local_dog = {
            "dog_min_sigma": "min_sigma",
            "dog_max_sigma": "max_sigma",
            "dog_sigma_ratio": "sigma_ratio",
            "dog_threshold": "threshold",
            "dog_overlap": "overlap",
        }
        self._attribute_argument_mapping_local_xc = {
            "xc_template": "template",
            "xc_distance": "distance",
            "xc_threshold": "threshold",
        }

        self._attribute_argument_mapping_dict = {
            "local_max": self._attribute_argument_mapping_local_max,
            "max": self._attribute_argument_mapping_max,
            "minmax": self._attribute_argument_mapping_local_minmax,
            "zaefferer": self._attribute_argument_mapping_local_zaefferer,
            "stat": self._attribute_argument_mapping_local_stat,
            "laplacian_of_gaussian": self._attribute_argument_mapping_local_log,
            "difference_of_gaussian": self._attribute_argument_mapping_local_dog,
            "template_matching": self._attribute_argument_mapping_local_xc,
        }

        if signal.axes_manager.signal_dimension != 2:
            raise SignalDimensionError(signal.axes.signal_dimension, 2)

        self._set_parameters_observer()
        self.on_trait_change(
            self.set_random_navigation_position, "random_navigation_position"
        )

        self.signal = signal
        self.peaks = peaks
        self.markers = None
        if self.signal._plot is None or not self.signal._plot.is_active:
            self.signal.plot()
        if self.signal.axes_manager.navigation_size > 0:
            self.show_navigation_sliders = True
            self.signal.axes_manager.events.indices_changed.connect(
                self._update_peak_finding, []
            )
            self.signal._plot.signal_plot.events.closed.connect(self.disconnect, [])
        # Set initial parameters:
        # As a convenience, if the template argument is provided, we keep it
        # even if the method is different, to be able to use it later.
        if "template" in kwargs.keys():
            self.xc_template = kwargs["template"]
        if method is not None:
            method_dict = {
                "local_max": "Local max",
                "max": "Max",
                "minmax": "Minmax",
                "zaefferer": "Zaefferer",
                "stat": "Stat",
                "laplacian_of_gaussian": "Laplacian of Gaussian",
                "difference_of_gaussian": "Difference of Gaussian",
                "template_matching": "Template matching",
            }
            self.method = method_dict[method]
        self._parse_paramaters_initial_values(**kwargs)
        self._update_peak_finding()

    def _parse_paramaters_initial_values(self, **kwargs):
        # Get the attribute to argument mapping for the current method
        arg_mapping = self._attribute_argument_mapping_dict[
            self._normalise_method_name(self.method)
        ]
        for attr, arg in arg_mapping.items():
            if arg in kwargs.keys():
                setattr(self, attr, kwargs[arg])

    def _update_peak_finding(self, method=None):
        if method is None:
            method = self.method
        self._find_peaks_current_index(method=method)
        self._plot_markers()

    def _method_changed(self, old, new):
        if new == "Template matching" and self.xc_template is None:
            raise RuntimeError('The "template" argument is required.')
        self._update_peak_finding(method=new)

    def _parameter_changed(self, old, new):
        self._update_peak_finding()

    def _set_parameters_observer(self):
        for parameters_mapping in self._attribute_argument_mapping_dict.values():
            for parameter in list(parameters_mapping.keys()):
                self.on_trait_change(self._parameter_changed, parameter)

    def _get_parameters(self, method):
        # Get the attribute to argument mapping for the given method
        arg_mapping = self._attribute_argument_mapping_dict[method]
        # return argument and values as kwargs
        return {arg: getattr(self, attr) for attr, arg in arg_mapping.items()}

    def _normalise_method_name(self, method):
        return method.lower().replace(" ", "_")

    def _find_peaks_current_index(self, method):
        method = self._normalise_method_name(method)
        self.peaks.data = self.signal.find_peaks(
            method,
            current_index=True,
            interactive=False,
            **self._get_parameters(method),
        )

    def _plot_markers(self):
        offsets = self.peaks.data
        offsets = convert_positions(offsets, self.signal.axes_manager.signal_axes)
        if self.markers is None:
            self.markers = Circles(
                offsets=offsets,
                edgecolor="red",
                facecolors="none",
                sizes=20,
                units="points",
            )
        else:
            self.markers.offsets = offsets

    def compute_navigation(self):
        method = self._normalise_method_name(self.method)
        with self.signal.axes_manager.events.indices_changed.suppress():
            self.peaks.data = self.signal.find_peaks(
                method,
                interactive=False,
                current_index=False,
                **self._get_parameters(method),
            )

    def close(self):
        # remove markers
        if self.signal._plot is not None and self.signal._plot.is_active:
            self.signal._plot.signal_plot.remove_markers(render_figure=True)
        self.disconnect()

    def disconnect(self):
        # disconnect event
        am = self.signal.axes_manager
        if self._update_peak_finding in am.events.indices_changed.connected:
            am.events.indices_changed.disconnect(self._update_peak_finding)

    def set_random_navigation_position(self):
        index = self._rng.integers(0, self.signal.axes_manager._max_index)
        self.signal.axes_manager.indices = np.unravel_index(
            index, tuple(self.signal.axes_manager._navigation_shape_in_array)
        )[::-1]


# For creating a text handler in legend (to label derivative magnitude)
class DerivativeTextParameters(object):
    def __init__(self, text, color):
        self.my_text = text
        self.my_color = color


class DerivativeTextHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        patch = mpl_text.Text(text=orig_handle.my_text, color=orig_handle.my_color)
        handlebox.add_artist(patch)
        return patch
