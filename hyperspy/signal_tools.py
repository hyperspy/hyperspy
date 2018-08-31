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

import logging
import functools

import numpy as np
import scipy as sp
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.text as mpl_text
import traits.api as t

from hyperspy import drawing
from hyperspy.exceptions import SignalDimensionError
from hyperspy.axes import AxesManager
from hyperspy.drawing.widgets import VerticalLineWidget
from hyperspy import components1d
from hyperspy.component import Component
from hyperspy.ui_registry import add_gui_method

_logger = logging.getLogger(__name__)


class SpanSelectorInSignal1D(t.HasTraits):
    ss_left_value = t.Float(np.nan)
    ss_right_value = t.Float(np.nan)
    is_ok = t.Bool(False)

    def __init__(self, signal):
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(
                signal.axes_manager.signal_dimension, 1)

        self.signal = signal
        self.axis = self.signal.axes_manager.signal_axes[0]
        self.span_selector = None
        self.signal.plot()
        self.span_selector_switch(on=True)

    def on_disabling_span_selector(self):
        pass

    def span_selector_switch(self, on):
        if not self.signal._plot.is_active:
            return

        if on is True:
            self.span_selector = \
                drawing.widgets.ModifiableSpanSelector(
                    self.signal._plot.signal_plot.ax,
                    onselect=self.update_span_selector_traits,
                    onmove_callback=self.update_span_selector_traits,)

        elif self.span_selector is not None:
            self.on_disabling_span_selector()
            self.span_selector.turn_off()
            self.span_selector = None

    def update_span_selector_traits(self, *args, **kwargs):
        if not self.signal._plot.is_active:
            return
        x0 = self.span_selector.rect.get_x()
        if x0 < self.axis.low_value:
            x0 = self.axis.low_value
        self.ss_left_value = x0
        x1 = self.ss_left_value + self.span_selector.rect.get_width()
        if x1 > self.axis.high_value:
            x1 = self.axis.high_value
        self.ss_right_value = x1

    def reset_span_selector(self):
        self.span_selector_switch(False)
        self.ss_left_value = np.nan
        self.ss_right_value = np.nan
        self.span_selector_switch(True)


class LineInSignal1D(t.HasTraits):

    """Adds a vertical draggable line to a spectrum that reports its
    position to the position attribute of the class.

    Attributes:
    -----------
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
            raise SignalDimensionError(
                signal.axes_manager.signal_dimension, 1)

        self.signal = signal
        self.signal.plot()
        axis_dict = signal.axes_manager.signal_axes[0].get_axis_dictionary()
        am = AxesManager([axis_dict, ])
        am._axes[0].navigate = True
        # Set the position of the line in the middle of the spectral
        # range by default
        am._axes[0].index = int(round(am._axes[0].size / 2))
        self.axes_manager = am
        self.axes_manager.events.indices_changed.connect(
            self.update_position, [])
        self.on_trait_change(self.switch_on_off, 'on')

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

        self._line.patch.set_color((self.color.Red() / 255.,
                                    self.color.Green() / 255.,
                                    self.color.Blue() / 255.,))
        self.draw()


@add_gui_method(toolkey="Signal1D.calibrate")
class Signal1DCalibration(SpanSelectorInSignal1D):
    left_value = t.Float(t.Undefined, label='New left value')
    right_value = t.Float(t.Undefined, label='New right value')
    offset = t.Float()
    scale = t.Float()
    units = t.Unicode()

    def __init__(self, signal):
        super(Signal1DCalibration, self).__init__(signal)
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(
                signal.axes_manager.signal_dimension, 1)
        self.units = self.axis.units
        self.scale = self.axis.scale
        self.offset = self.axis.offset
        self.last_calibration_stored = True

    def _left_value_changed(self, old, new):
        if self.span_selector is not None and \
                self.span_selector.range is None:
            return
        else:
            self._update_calibration()

    def _right_value_changed(self, old, new):
        if self.span_selector.range is None:
            return
        else:
            self._update_calibration()

    def _update_calibration(self, *args, **kwargs):
        # If the span selector or the new range values are not defined do
        # nothing
        if np.isnan(self.ss_left_value) or np.isnan(self.ss_right_value) or\
                t.Undefined in (self.left_value, self.right_value):
            return
        lc = self.axis.value2index(self.ss_left_value)
        rc = self.axis.value2index(self.ss_right_value)
        self.offset, self.scale = self.axis.calibrate(
            (self.left_value, self.right_value), (lc, rc),
            modify_calibration=False)

    def apply(self):
        if np.isnan(self.ss_left_value) or np.isnan(self.ss_right_value):
            _logger.warning("Select a range by clicking on the signal figure "
                            "and dragging before pressing Apply.")
            return
        elif self.left_value is t.Undefined or self.right_value is t.Undefined:
            _logger.warning("Select the new left and right values before "
                            "pressing apply.")
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
                return np.array(self.line_color.Get()) / 255.
            except AttributeError:
                try:
                    # PySide
                    return np.array(self.line_color.getRgb()) / 255.
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
        l1.set_line_properties(color=self.original_color,
                               type='scatter')
        l2 = drawing.signal1d.Signal1DLine()
        l2.data_function = self.model2plot

        l2.set_line_properties(
            color=self.line_color_rgb,
            type='line')
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
            color=self.line_color_rgb,
            type='line')
        self.signal._plot.signal_plot.add_line(self.smooth_diff_line,
                                               ax='right')

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
        self.smooth_line.line_properties = {
            'color': self.line_color_rgb}
        if self.smooth_diff_line is not None:
            self.smooth_diff_line.line_properties = {
                'color': self.line_color_rgb}
        try:
            # it seems that changing the properties can be done before the
            # first rendering event, which can cause issue with blitting
            self.update_lines()
        except AttributeError:
            pass

    def diff_model2plot(self, axes_manager=None):
        smoothed = np.diff(self.model2plot(axes_manager),
                           self.differential_order)
        return smoothed

    def close(self):
        if self.signal._plot.is_active:
            if self.differential_order != 0:
                self.turn_diff_line_off()
            self.smooth_line.close()
            self.data_line.set_line_properties(
                color=self.original_color,
                type='line')


@add_gui_method(toolkey="Signal1D.smooth_savitzky_golay")
class SmoothingSavitzkyGolay(Smoothing):

    polynomial_order = t.Int(
        3,
        desc="The order of the polynomial used to fit the samples."
             "`polyorder` must be less than `window_length`.")

    window_length = t.Int(
        5,
        desc="`window_length` must be a positive odd integer.")

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
                "The window length must be greater than the polynomial order")

    def _polynomial_order_changed(self, old, new):
        if self.window_length <= new:
            self.window_length = new + 2 if new % 2 else new + 1
            _logger.warning(
                "Polynomial order must be < window length. "
                "Window length set to %i.", self.window_length)
        self.update_lines()

    def _window_length_changed(self, old, new):
        self.update_lines()

    def _differential_order_changed(self, old, new):
        if new > self.polynomial_order:
            self.polynomial_order += 1
            _logger.warning(
                "Differential order must be <= polynomial order. "
                "Polynomial order set to %i.", self.polynomial_order)
        super(
            SmoothingSavitzkyGolay,
            self)._differential_order_changed(
            old,
            new)

    def diff_model2plot(self, axes_manager=None):
        self.single_spectrum.data = self.signal().copy()
        self.single_spectrum.smooth_savitzky_golay(
            polynomial_order=self.polynomial_order,
            window_length=self.window_length,
            differential_order=self.differential_order)
        return self.single_spectrum.data

    def model2plot(self, axes_manager=None):
        self.single_spectrum.data = self.signal().copy()
        self.single_spectrum.smooth_savitzky_golay(
            polynomial_order=self.polynomial_order,
            window_length=self.window_length,
            differential_order=0)
        return self.single_spectrum.data

    def apply(self):
        self.signal.smooth_savitzky_golay(
            polynomial_order=self.polynomial_order,
            window_length=self.window_length,
            differential_order=self.differential_order)
        self.signal._replot()


@add_gui_method(toolkey="Signal1D.smooth_lowess")
class SmoothingLowess(Smoothing):
    smoothing_parameter = t.Range(low=0.,
                                  high=1.,
                                  value=0.5,
                                  )
    number_of_iterations = t.Range(low=1,
                                   value=1)

    def __init__(self, *args, **kwargs):
        super(SmoothingLowess, self).__init__(*args, **kwargs)

    def _smoothing_parameter_changed(self, old, new):
        if new == 0:
            self.smoothing_parameter = old
        else:
            self.update_lines()

    def _number_of_iterations_changed(self, old, new):
        self.update_lines()

    def model2plot(self, axes_manager=None):
        self.single_spectrum.data = self.signal().copy()
        self.single_spectrum.smooth_lowess(
            smoothing_parameter=self.smoothing_parameter,
            number_of_iterations=self.number_of_iterations,
            show_progressbar=False)

        return self.single_spectrum.data

    def apply(self):
        self.signal.smooth_lowess(
            smoothing_parameter=self.smoothing_parameter,
            number_of_iterations=self.number_of_iterations)
        self.signal._replot()


@add_gui_method(toolkey="Signal1D.smooth_total_variation")
class SmoothingTV(Smoothing):
    smoothing_parameter = t.Float(200)

    def _smoothing_parameter_changed(self, old, new):
        self.update_lines()

    def model2plot(self, axes_manager=None):
        self.single_spectrum.data = self.signal().copy()
        self.single_spectrum.smooth_tv(
            smoothing_parameter=self.smoothing_parameter,
            show_progressbar=False)

        return self.single_spectrum.data

    def apply(self):
        self.signal.smooth_tv(
            smoothing_parameter=self.smoothing_parameter)
        self.signal._replot()


@add_gui_method(toolkey="Signal1D.smooth_butterworth")
class ButterworthFilter(Smoothing):
    cutoff_frequency_ratio = t.Range(0.01, 1., 0.01)
    type = t.Enum('low', 'high')
    order = t.Int(2)

    def _cutoff_frequency_ratio_changed(self, old, new):
        self.update_lines()

    def _type_changed(self, old, new):
        self.update_lines()

    def _order_changed(self, old, new):
        self.update_lines()

    def model2plot(self, axes_manager=None):
        b, a = sp.signal.butter(self.order, self.cutoff_frequency_ratio,
                                self.type)
        smoothed = sp.signal.filtfilt(b, a, self.signal())
        return smoothed

    def apply(self):
        b, a = sp.signal.butter(self.order, self.cutoff_frequency_ratio,
                                self.type)
        f = functools.partial(sp.signal.filtfilt, b, a)
        self.signal.map(f)


class Load(t.HasTraits):
    filename = t.File
    lazy = t.Bool(False)


@add_gui_method(toolkey="Signal1D.contrast_editor")
class ImageContrastEditor(t.HasTraits):
    ss_left_value = t.Float()
    ss_right_value = t.Float()

    def __init__(self, image):
        super(ImageContrastEditor, self).__init__()
        self.image = image
        f = plt.figure()
        self.ax = f.add_subplot(111)
        self.plot_histogram()

        self.span_selector = None
        self.span_selector_switch(on=True)

    def on_disabling_span_selector(self):
        pass

    def span_selector_switch(self, on):
        if on is True:
            self.span_selector = \
                drawing.widgets.ModifiableSpanSelector(
                    self.ax,
                    onselect=self.update_span_selector_traits,
                    onmove_callback=self.update_span_selector_traits)

        elif self.span_selector is not None:
            self.on_disabling_span_selector()
            self.span_selector.turn_off()
            self.span_selector = None

    def update_span_selector_traits(self, *args, **kwargs):
        self.ss_left_value = self.span_selector.rect.get_x()
        self.ss_right_value = self.ss_left_value + \
            self.span_selector.rect.get_width()

    def plot_histogram(self):
        vmin, vmax = self.image.vmin, self.image.vmax
        pad = (vmax - vmin) * 0.05
        vmin -= pad
        vmax += pad
        data = self.image.data_function().ravel()
        self.patches = self.ax.hist(data, 100, range=(vmin, vmax),
                                    color='blue')[2]
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(vmin, vmax)
        self.ax.figure.canvas.draw_idle()

    def reset(self):
        data = self.image.data_function().ravel()
        self.image.vmin, self.image.vmax = np.nanmin(data), np.nanmax(data)
        self.image.update()
        self.update_histogram()

    def update_histogram(self):
        for patch in self.patches:
            self.ax.patches.remove(patch)
        self.plot_histogram()

    def apply(self):
        if self.ss_left_value == self.ss_right_value:
            return
        self.image.vmin = self.ss_left_value
        self.image.vmax = self.ss_right_value
        self.image.update()
        self.update_histogram()

    def close(self):
        plt.close(self.ax.figure)


@add_gui_method(toolkey="Signal1D.integrate_in_range")
class IntegrateArea(SpanSelectorInSignal1D):
    integrate = t.Button()

    def __init__(self, signal, signal_range=None):
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(
                signal.axes.signal_dimension, 1)

        self.signal = signal
        self.axis = self.signal.axes_manager.signal_axes[0]
        self.span_selector = None
        if (not hasattr(self.signal, '_plot') or self.signal._plot is None or
                not self.signal._plot.is_active):
            self.signal.plot()
        self.span_selector_switch(on=True)

    def apply(self):
        integrated_spectrum = self.signal._integrate_in_range_commandline(
            signal_range=(
                self.ss_left_value,
                self.ss_right_value)
        )
        # Replaces the original signal inplace with the new integrated spectrum
        plot = False
        if self.signal._plot and integrated_spectrum.axes_manager.shape != ():
            self.signal._plot.close()
            plot = True
        self.signal.__init__(**integrated_spectrum._to_dictionary())
        self.signal._assign_subclass()
        self.signal.axes_manager.set_signal_dimension(0)

        if plot is True:
            self.signal.plot()


@add_gui_method(toolkey="Signal1D.remove_background")
class BackgroundRemoval(SpanSelectorInSignal1D):
    background_type = t.Enum(
        'Power Law',
        'Gaussian',
        'Offset',
        'Polynomial',
        default='Power Law')
    polynomial_order = t.Range(1, 10)
    fast = t.Bool(True,
                  desc=("Perform a fast (analytic, but possibly less accurate)"
                        " estimation of the background. Otherwise use "
                        "use non-linear least squares."))
    zero_fill = t.Bool(
        False,
        desc=("Set all spectral channels lower than the lower \n"
              "bound of the fitting range to zero (this is the \n"
              "default behavior of Gatan's DigitalMicrograph). \n"
              "Otherwise leave the pre-fitting region as-is \n"
              "(useful for inspecting quality of background fit)."))
    background_estimator = t.Instance(Component)
    bg_line_range = t.Enum('from_left_range',
                           'full',
                           'ss_range',
                           default='full')
    hi = t.Int(0)

    def __init__(self, signal, background_type='Power Law', polynomial_order=2,
                 fast=True, plot_remainder=True, zero_fill=False,
                 show_progressbar=None):
        super(BackgroundRemoval, self).__init__(signal)
        # setting the polynomial order will change the backgroud_type to
        # polynomial, so we set it before setting the background type
        self.polynomial_order = polynomial_order
        self.background_type = background_type
        self.set_background_estimator()
        self.fast = fast
        self.plot_remainder = plot_remainder
        self.zero_fill = zero_fill
        self.show_progressbar = show_progressbar
        self.bg_line = None
        self.rm_line = None

    def on_disabling_span_selector(self):
        if self.bg_line is not None:
            self.bg_line.close()
            self.bg_line = None
        if self.rm_line is not None:
            self.rm_line.close()
            self.rm_line = None

    def set_background_estimator(self):
        if self.background_type == 'Power Law':
            self.background_estimator = components1d.PowerLaw()
            self.bg_line_range = 'from_left_range'
        elif self.background_type == 'Gaussian':
            self.background_estimator = components1d.Gaussian()
            self.bg_line_range = 'full'
        elif self.background_type == 'Offset':
            self.background_estimator = components1d.Offset()
            self.bg_line_range = 'full'
        elif self.background_type == 'Polynomial':
            self.background_estimator = components1d.Polynomial(
                self.polynomial_order)
            self.bg_line_range = 'full'

    def _polynomial_order_changed(self, old, new):
        self.background_estimator = components1d.Polynomial(new)
        self.span_selector_changed()

    def _background_type_changed(self, old, new):
        self.set_background_estimator()
        self.span_selector_changed()

    def _ss_left_value_changed(self, old, new):
        if not (np.isnan(self.ss_right_value) or np.isnan(self.ss_left_value)):
            self.span_selector_changed()

    def _ss_right_value_changed(self, old, new):
        if not (np.isnan(self.ss_right_value) or np.isnan(self.ss_left_value)):
            self.span_selector_changed()

    def create_background_line(self):
        self.bg_line = drawing.signal1d.Signal1DLine()
        self.bg_line.data_function = self.bg_to_plot
        self.bg_line.set_line_properties(
            color='blue',
            type='line',
            scaley=False)
        self.signal._plot.signal_plot.add_line(self.bg_line)
        self.bg_line.autoscale = False
        self.bg_line.plot()

    def create_remainder_line(self):
        self.rm_line = drawing.signal1d.Signal1DLine()
        self.rm_line.data_function = self.rm_to_plot
        self.rm_line.set_line_properties(
            color='green',
            type='line',
            scaley=False)
        self.signal._plot.signal_plot.add_line(self.rm_line)
        self.rm_line.autoscale = False
        self.rm_line.plot()

    def bg_to_plot(self, axes_manager=None, fill_with=np.nan):
        # First try to update the estimation
        self.background_estimator.estimate_parameters(
            self.signal, self.ss_left_value, self.ss_right_value,
            only_current=True)

        if self.bg_line_range == 'from_left_range':
            bg_array = np.zeros(self.axis.axis.shape)
            bg_array[:] = fill_with
            from_index = self.axis.value2index(self.ss_left_value)
            bg_array[from_index:] = self.background_estimator.function(
                self.axis.axis[from_index:])
            to_return = bg_array
        elif self.bg_line_range == 'full':
            to_return = self.background_estimator.function(self.axis.axis)
        elif self.bg_line_range == 'ss_range':
            bg_array = np.zeros(self.axis.axis.shape)
            bg_array[:] = fill_with
            from_index = self.axis.value2index(self.ss_left_value)
            to_index = self.axis.value2index(self.ss_right_value)
            bg_array[from_index:] = self.background_estimator.function(
                self.axis.axis[from_index:to_index])
            to_return = bg_array

        if self.signal.metadata.Signal.binned is True:
            to_return *= self.axis.scale
        return to_return

    def rm_to_plot(self, axes_manager=None, fill_with=np.nan):
        return self.signal() - self.bg_line.line.get_ydata()

    def span_selector_changed(self):
        if self.ss_left_value is np.nan or self.ss_right_value is np.nan or\
                self.ss_right_value <= self.ss_left_value:
            return
        if self.background_estimator is None:
            return
        res = self.background_estimator.estimate_parameters(
            self.signal, self.ss_left_value,
            self.ss_right_value,
            only_current=True)
        if self.bg_line is None:
            if res:
                self.create_background_line()
        else:
            self.bg_line.update()
        if self.plot_remainder:
            if self.rm_line is None:
                if res:
                    self.create_remainder_line()
            else:
                self.rm_line.update()

    def apply(self):
        if self.signal._plot:
            self.signal._plot.close()
            plot = True
        else:
            plot = False
        background_type = ("PowerLaw" if self.background_type == "Power Law"
                           else self.background_type)
        new_spectra = self.signal.remove_background(
            signal_range=(self.ss_left_value, self.ss_right_value),
            background_type=background_type,
            fast=self.fast,
            zero_fill=self.zero_fill,
            polynomial_order=self.polynomial_order,
            show_progressbar=self.show_progressbar)
        self.signal.data = new_spectra.data
        self.signal.events.data_changed.trigger(self)
        if plot:
            self.signal.plot()


SPIKES_REMOVAL_INSTRUCTIONS = (
    "To remove spikes from the data:\n\n"

    "   1. Click \"Show derivative histogram\" to "
    "determine at what magnitude the spikes are present.\n"
    "   2. Enter a suitable threshold (lower than the "
    "lowest magnitude outlier in the histogram) in the "
    "\"Threshold\" box, which will be the magnitude "
    "from which to search. \n"
    "   3. Click \"Find next\" to find the first spike.\n"
    "   4. If desired, the width and position of the "
    "boundaries used to replace the spike can be "
    "adjusted by clicking and dragging on the displayed "
    "plot.\n "
    "   5. View the spike (and the replacement data that "
    "will be added) and click \"Remove spike\" in order "
    "to alter the data as shown. The tool will "
    "automatically find the next spike to replace.\n"
    "   6. Repeat this process for each spike throughout "
    "the dataset, until the end of the dataset is "
    "reached.\n"
    "   7. Click \"OK\" when finished to close the spikes "
    "removal tool.\n\n"

    "Note: Various settings can be configured in "
    "the \"Advanced settings\" section. Hover the "
    "mouse over each parameter for a description of what "
    "it does."

    "\n")


@add_gui_method(toolkey="SimpleMessage")
class SimpleMessage(t.HasTraits):
    text = t.Str

    def __init__(self, text=""):
        self.text = text


@add_gui_method(toolkey="Signal1D.spikes_removal_tool")
class SpikesRemoval(SpanSelectorInSignal1D):
    interpolator_kind = t.Enum(
        'Linear',
        'Spline',
        default='Linear',
        desc="the type of interpolation to use when\n"
             "replacing the signal where a spike has been replaced")
    threshold = t.Float(400, desc="the derivative magnitude threshold above\n"
                        "which to find spikes")
    click_to_show_instructions = t.Button()
    show_derivative_histogram = t.Button()
    spline_order = t.Range(1, 10, 3,
                           desc="the order of the spline used to\n"
                           "connect the reconstructed data")
    interpolator = None
    default_spike_width = t.Int(5,
                                desc="the width over which to do the interpolation\n"
                                "when removing a spike (this can be "
                                "adjusted for each\nspike by clicking "
                                     "and dragging on the display during\n"
                                     "spike replacement)")
    index = t.Int(0)
    add_noise = t.Bool(True,
                       desc="whether to add noise to the interpolated\nportion"
                       "of the spectrum. The noise properties defined\n"
                       "in the Signal metadata are used if present,"
                            "otherwise\nshot noise is used as a default")

    def __init__(self, signal, navigation_mask=None, signal_mask=None):
        super(SpikesRemoval, self).__init__(signal)
        self.interpolated_line = None
        self.coordinates = [coordinate for coordinate in
                            signal.axes_manager._am_indices_generator()
                            if (navigation_mask is None or not
                                navigation_mask[coordinate[::-1]])]
        self.signal = signal
        self.line = signal._plot.signal_plot.ax_lines[0]
        self.ax = signal._plot.signal_plot.ax
        signal._plot.auto_update_plot = False
        if len(self.coordinates) > 1:
            signal.axes_manager.indices = self.coordinates[0]
        self.index = 0
        self.argmax = None
        self.derivmax = None
        self.kind = "linear"
        self._temp_mask = np.zeros(self.signal().shape, dtype='bool')
        self.signal_mask = signal_mask
        self.navigation_mask = navigation_mask
        md = self.signal.metadata
        from hyperspy.signal import BaseSignal

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

    def _threshold_changed(self, old, new):
        self.index = 0
        self.update_plot()

    def _click_to_show_instructions_fired(self):
        from pyface.message_dialog import information
        m = information(None, SPIKES_REMOVAL_INSTRUCTIONS,
                        title="Instructions"),

    def _show_derivative_histogram_fired(self):
        self.signal._spikes_diagnosis(signal_mask=self.signal_mask,
                                      navigation_mask=self.navigation_mask)

    def detect_spike(self):
        derivative = np.diff(self.signal())
        if self.signal_mask is not None:
            derivative[self.signal_mask[:-1]] = 0
        if self.argmax is not None:
            left, right = self.get_interpolation_range()
            self._temp_mask[left:right] = True
            derivative[self._temp_mask[:-1]] = 0
        if abs(derivative.max()) >= self.threshold:
            self.argmax = derivative.argmax()
            self.derivmax = abs(derivative.max())
            return True
        else:
            return False

    def _reset_line(self):
        if self.interpolated_line is not None:
            self.interpolated_line.close()
            self.interpolated_line = None
            self.reset_span_selector()

    def find(self, back=False):
        self._reset_line()
        ncoordinates = len(self.coordinates)
        spike = self.detect_spike()
        with self.signal.axes_manager.events.indices_changed.suppress():
            while not spike and (
                    (self.index < ncoordinates - 1 and back is False) or
                    (self.index > 0 and back is True)):
                if back is False:
                    self.index += 1
                else:
                    self.index -= 1
                spike = self.detect_spike()

        if spike is False:
            m = SimpleMessage()
            m.text = 'End of dataset reached'
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
            maximum = min(len(self.signal()) - 1, self.argmax + 50)
            thresh_label = DerivativeTextParameters(
                text=r"$\mathsf{\delta}_\mathsf{max}=$",
                color="black")
            self.ax.legend([thresh_label], [repr(int(self.derivmax))],
                           handler_map={DerivativeTextParameters:
                                        DerivativeTextHandler()},
                           loc='best')
            self.ax.set_xlim(
                self.signal.axes_manager.signal_axes[0].index2value(
                    minimum),
                self.signal.axes_manager.signal_axes[0].index2value(
                    maximum))
            self.update_plot()
            self.create_interpolation_line()

    def update_plot(self):
        if self.interpolated_line is not None:
            self.interpolated_line.close()
            self.interpolated_line = None
        self.reset_span_selector()
        self.update_spectrum_line()
        if len(self.coordinates) > 1:
            self.signal._plot.pointer._on_navigate(self.signal.axes_manager)

    def update_spectrum_line(self):
        self.line.auto_update = True
        self.line.update()
        self.line.auto_update = False

    def _index_changed(self, old, new):
        self.signal.axes_manager.indices = self.coordinates[new]
        self.argmax = None
        self._temp_mask[:] = False

    def on_disabling_span_selector(self):
        if self.interpolated_line is not None:
            self.interpolated_line.close()
            self.interpolated_line = None

    def _spline_order_changed(self, old, new):
        self.kind = self.spline_order
        self.span_selector_changed()

    def _add_noise_changed(self, old, new):
        self.span_selector_changed()

    def _interpolator_kind_changed(self, old, new):
        if new == 'linear':
            self.kind = new
        else:
            self.kind = self.spline_order
        self.span_selector_changed()

    def _ss_left_value_changed(self, old, new):
        if not (np.isnan(self.ss_right_value) or np.isnan(self.ss_left_value)):
            self.span_selector_changed()

    def _ss_right_value_changed(self, old, new):
        if not (np.isnan(self.ss_right_value) or np.isnan(self.ss_left_value)):
            self.span_selector_changed()

    def create_interpolation_line(self):
        self.interpolated_line = drawing.signal1d.Signal1DLine()
        self.interpolated_line.data_function = self.get_interpolated_spectrum
        self.interpolated_line.set_line_properties(
            color='blue',
            type='line')
        self.signal._plot.signal_plot.add_line(self.interpolated_line)
        self.interpolated_line.auto_update = False
        self.interpolated_line.autoscale = False
        self.interpolated_line.plot()

    def get_interpolation_range(self):
        axis = self.signal.axes_manager.signal_axes[0]
        if np.isnan(self.ss_left_value) or np.isnan(self.ss_right_value):
            left = self.argmax - self.default_spike_width
            right = self.argmax + self.default_spike_width
        else:
            left = axis.value2index(self.ss_left_value)
            right = axis.value2index(self.ss_right_value)

        # Clip to the axis dimensions
        nchannels = self.signal.axes_manager.signal_shape[0]
        left = left if left >= 0 else 0
        right = right if right < nchannels else nchannels - 1

        return left, right

    def get_interpolated_spectrum(self, axes_manager=None):
        data = self.signal().copy()
        axis = self.signal.axes_manager.signal_axes[0]
        left, right = self.get_interpolation_range()
        if self.kind == 'linear':
            pad = 1
        else:
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
            intp = sp.interpolate.interp1d(x, y, kind=self.kind)
            data[left:right] = intp(axis.axis[left:right])

        # Add noise
        if self.add_noise is True:
            if self.noise_type == "white":
                data[left:right] += np.random.normal(
                    scale=np.sqrt(self.noise_variance),
                    size=right - left)
            elif self.noise_type == "heteroscedastic":
                noise_variance = self.noise_variance(
                    axes_manager=self.signal.axes_manager)[left:right]
                noise = [np.random.normal(scale=np.sqrt(item))
                         for item in noise_variance]
                data[left:right] += noise
            else:
                data[left:right] = np.random.poisson(
                    np.clip(data[left:right], 0, np.inf))

        return data

    def span_selector_changed(self):
        if self.interpolated_line is None:
            return
        else:
            self.interpolated_line.update()

    def apply(self):
        if not self.interpolated_line:  # No spike selected
            return
        self.signal()[:] = self.get_interpolated_spectrum()
        self.signal.events.data_changed.trigger(obj=self.signal)
        self.update_spectrum_line()
        self.interpolated_line.close()
        self.interpolated_line = None
        self.reset_span_selector()
        self.find()


# For creating a text handler in legend (to label derivative magnitude)
class DerivativeTextParameters(object):

    def __init__(self, text, color):
        self.my_text = text
        self.my_color = color


class DerivativeTextHandler(object):

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpl_text.Text(
            text=orig_handle.my_text,
            color=orig_handle.my_color)
        handlebox.add_artist(patch)
        return patch
