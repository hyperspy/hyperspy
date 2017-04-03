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

import numpy as np
import scipy as sp
import matplotlib.colors
import matplotlib.pyplot as plt
import traits.api as t

from hyperspy import drawing
from hyperspy.exceptions import SignalDimensionError
from hyperspy.axes import AxesManager
from hyperspy.drawing.widgets import VerticalLineWidget
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
        if not self.signal._plot.is_active():
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
        if not self.signal._plot.is_active():
            return
        self.ss_left_value = self.span_selector.rect.get_x()
        self.ss_right_value = self.ss_left_value + \
            self.span_selector.rect.get_width()

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
    color = t.Color("black")

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
        self.signal._plot.signal_plot.figure.canvas.draw()

    def switch_on_off(self, obj, trait_name, old, new):
        if not self.signal._plot.is_active():
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
        if not self.signal._plot.is_active():
            return
        self.position = self.axes_manager.coordinates[0]

    def _color_changed(self, old, new):
        if self.on is False:
            return

        self._line.patch.set_color((self.color.Red() / 255.,
                                    self.color.Green() / 255.,
                                    self.color.Blue() / 255.,))
        self.draw()


class Signal1DCalibration(SpanSelectorInSignal1D):
    left_value = t.Float(label='New left value')
    right_value = t.Float(label='New right value')
    offset = t.Float()
    scale = t.Float()
    units = t.Unicode()

    def __init__(self, signal):
        super(Signal1DCalibration, self).__init__(signal)
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(
                signal.axes_manager.signal_dimension, 1)
        self.units = self.axis.units
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
        if self.left_value == self.right_value:
            return
        lc = self.axis.value2index(self.ss_left_value)
        rc = self.axis.value2index(self.ss_right_value)
        self.offset, self.scale = self.axis.calibrate(
            (self.left_value, self.right_value), (lc, rc),
            modify_calibration=False)


class Signal1DRangeSelector(SpanSelectorInSignal1D):
    on_close = t.List()


class Smoothing(t.HasTraits):
    line_color = t.Color("blue")
    line_color_ipy = t.Str("blue")
    differential_order = t.Int(0)

    @property
    def line_color_rgb(self):
        try:
            # PyQt and WX
            return np.array(self.line_color.Get()) / 255.
        except AttributeError:
            try:
                # PySide
                return np.array(self.line_color.getRgb()) / 255.
            except:
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
        if self.signal._plot is None or not \
                self.signal._plot.is_active():
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
        self.smooth_diff_line.data_function = self.diff_model2plot
        self.smooth_diff_line.set_line_properties(
            color=self.line_color_rgb,
            type='line')
        self.signal._plot.signal_plot.add_line(self.smooth_diff_line,
                                               ax='right')
        self.smooth_diff_line.axes_manager = self.signal.axes_manager

    def _line_color_ipy_changed(self):
        self.line_color = str(self.line_color_ipy)

    def turn_diff_line_off(self):
        if self.smooth_diff_line is None:
            return
        self.smooth_diff_line.close()
        self.smooth_diff_line = None

    def _differential_order_changed(self, old, new):
        if old == 0:
            self.turn_diff_line_on(new)
            self.smooth_diff_line.plot()
        if new == 0:
            self.turn_diff_line_off()
            return
        self.smooth_diff_line.update(force_replot=False)

    def _line_color_changed(self, old, new):
        self.smooth_line.line_properties = {
            'color': self.line_color_rgb}
        if self.smooth_diff_line is not None:
            self.smooth_diff_line.line_properties = {
                'color': self.line_color_rgb}
        self.update_lines()

    def diff_model2plot(self, axes_manager=None):
        smoothed = np.diff(self.model2plot(axes_manager),
                           self.differential_order)
        return smoothed

    def close(self):
        if self.signal._plot.is_active():
            if self.differential_order != 0:
                self.turn_diff_line_off()
            self.smooth_line.close()
            self.data_line.set_line_properties(
                color=self.original_color,
                type='line')


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
            _logger.warn(
                "The window length must be greater than the polynomial order")

    def _polynomial_order_changed(self, old, new):
        if self.window_length <= new:
            self.window_length = new + 2 if new % 2 else new + 1
            _logger.warn(
                "Polynomial order must be < window length. "
                "Window length set to %i.", self.window_length)
        self.update_lines()

    def _window_length_changed(self, old, new):
        self.update_lines()

    def _differential_order_changed(self, old, new):
        if new > self.polynomial_order:
            self.polynomial_order += 1
            _logger.warn(
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


class ButterworthFilter(Smoothing):
    cutoff_frequency_ratio = t.Range(0., 1., 0.05)
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
        self.ax.figure.canvas.draw()

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


class IntegrateArea(SpanSelectorInSignal1D):
    integrate = t.Button()

    def __init__(self, signal, signal_range=None):
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(
                signal.axes.signal_dimension, 1)

        self.signal = signal
        self.axis = self.signal.axes_manager.signal_axes[0]
        self.span_selector = None
        if not hasattr(self.signal, '_plot'):
            self.signal.plot()
        elif self.signal._plot is None:
            self.signal.plot()
        elif self.signal._plot.is_active() is False:
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
