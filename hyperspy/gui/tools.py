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
import matplotlib.pyplot as plt
import traits.api as t
import traitsui.api as tu
from traitsui.menu import (OKButton, CancelButton, OKCancelButtons)

from hyperspy import drawing
from hyperspy.exceptions import SignalDimensionError
from hyperspy.gui import messages
from hyperspy.axes import AxesManager
from hyperspy.drawing.widgets import VerticalLineWidget

_logger = logging.getLogger(__name__)


OurOKButton = tu.Action(name="OK",
                        action="OK",)

OurApplyButton = tu.Action(name="Apply",
                           action="apply")

OurResetButton = tu.Action(name="Reset",
                           action="reset")

OurCloseButton = tu.Action(name="Close",
                           action="close_directly")

OurFindButton = tu.Action(name="Find next",
                          action="find",)

OurPreviousButton = tu.Action(name="Find previous",
                              action="back",)


class SmoothingHandler(tu.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        if is_ok is True:
            info.object.apply()
        else:
            info.object.close()
        return True


class SpanSelectorInSignal1DHandler(tu.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        info.object.span_selector_switch(False)
        if is_ok is True:
            self.apply(info)

        return True

    def close_directly(self, info):
        if (info.ui.owner is not None) and self.close(info, False):
            info.ui.owner.close()

    def apply(self, info, *args, **kwargs):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object
        obj.is_ok = True
        if hasattr(obj, 'apply'):
            obj.apply()

        return

    def next(self, info, *args, **kwargs):
        """Handles the **Next** button being clicked.

        """
        obj = info.object
        obj.is_ok = True
        if hasattr(obj, 'next'):
            next(obj)
        return


class Signal1DRangeSelectorHandler(tu.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        info.object.span_selector_switch(False)
        if is_ok is True:
            self.apply(info)
        return True

    def apply(self, info, *args, **kwargs):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object
        if obj.ss_left_value != obj.ss_right_value:
            info.object.span_selector_switch(False)
            for method, cls in obj.on_close:
                method(cls, obj.ss_left_value, obj.ss_right_value)
            info.object.span_selector_switch(True)

        obj.is_ok = True

        return


class CalibrationHandler(SpanSelectorInSignal1DHandler):

    def apply(self, info, *args, **kwargs):
        """Handles the **Apply** button being clicked.
        """
        if info.object.signal is None:
            return
        axis = info.object.axis
        axis.scale = info.object.scale
        axis.offset = info.object.offset
        axis.units = info.object.units
        info.object.span_selector_switch(on=False)
        info.object.signal._replot()
        info.object.span_selector_switch(on=True)
        info.object.last_calibration_stored = True
        return


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
    view = tu.View(
        tu.Group(
            'left_value',
            'right_value',
            tu.Item('ss_left_value',
                    label='Left',
                    style='readonly'),
            tu.Item('ss_right_value',
                    label='Right',
                    style='readonly'),
            tu.Item(name='offset',
                    style='readonly'),
            tu.Item(name='scale',
                    style='readonly'),
            'units',),
        handler=CalibrationHandler,
        buttons=[OKButton, OurApplyButton, CancelButton],
        kind='live',
        title='Calibration parameters')

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
            messages.information(
                'Please select a range in the spectrum figure'
                'by dragging the mouse over it')
            return
        else:
            self._update_calibration()

    def _right_value_changed(self, old, new):
        if self.span_selector.range is None:
            messages.information(
                'Please select a range in the spectrum figure'
                'by dragging the mouse over it')
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

    view = tu.View(
        tu.Item('ss_left_value', label='Left', style='readonly'),
        tu.Item('ss_right_value', label='Right', style='readonly'),
        handler=Signal1DRangeSelectorHandler,
        buttons=[OKButton, OurApplyButton, CancelButton],)


class Smoothing(t.HasTraits):
    line_color = t.Color('blue')
    differential_order = t.Int(0)

    @property
    def line_color_rgb(self):
        try:
            # PyQt and WX
            return self.line_color.Get()
        except AttributeError:
            try:
                # PySide
                return self.line_color.getRgb()
            except:
                raise

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
            color=np.array(self.line_color_rgb) / 255.,
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
            color=np.array(self.line_color_rgb) / 255.,
            type='line')
        self.signal._plot.signal_plot.add_line(self.smooth_diff_line,
                                               ax='right')
        self.smooth_diff_line.axes_manager = self.signal.axes_manager

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
            'color': np.array(self.line_color_rgb) / 255.}
        if self.smooth_diff_line is not None:
            self.smooth_diff_line.line_properties = {
                'color': np.array(self.line_color_rgb) / 255.}
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

    view = tu.View(
        tu.Group(
            tu.Group(
                'window_length',
                tu.Item(
                    'decrease_window_length',
                    show_label=False),
                tu.Item(
                    'increase_window_length',
                    show_label=False),
                orientation="horizontal"),
            'polynomial_order',
            tu.Item(
                name='differential_order',
                tooltip='The order of the derivative to compute. This must '
                        'be a nonnegative integer. The default is 0, which '
                        'means to filter the data without differentiating.',
            ),
            'line_color'),
        kind='live',
        handler=SmoothingHandler,
        buttons=OKCancelButtons,
        title='Savitzky-Golay Smoothing',
    )

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
    view = tu.View(
        tu.Group(
            'smoothing_parameter',
            'number_of_iterations',
            'line_color'),
        kind='live',
        handler=SmoothingHandler,
        buttons=OKCancelButtons,
        title='Lowess Smoothing',)

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

    view = tu.View(
        tu.Group(
            'smoothing_parameter',
            'line_color'),
        kind='live',
        handler=SmoothingHandler,
        buttons=OKCancelButtons,
        title='Total Variation Smoothing',)

    def _smoothing_parameter_changed(self, old, new):
        self.update_lines()

    def _number_of_iterations_changed(self, old, new):
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

    view = tu.View(
        tu.Group(
            'cutoff_frequency_ratio',
            'order',
            'type'),
        kind='live',
        handler=SmoothingHandler,
        buttons=OKCancelButtons,
        title='Butterworth filter',)

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
    traits_view = tu.View(
        tu.Group('filename'),
        kind='livemodal',
        buttons=[OKButton, CancelButton],
        title='Load file')


class ImageContrastHandler(tu.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        #        info.object.span_selector_switch(False)
        #        if is_ok is True:
        #            self.apply(info)
        if is_ok is False:
            info.object.image.update()
        info.object.close()
        return True

    def apply(self, info):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object
        obj.apply()

        return

    @staticmethod
    def reset(info):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object
        obj.reset()
        return

    @staticmethod
    def our_help(info):
        """Handles the **Apply** button being clicked.

        """
        info.object._help()


class ImageContrastEditor(t.HasTraits):
    ss_left_value = t.Float()
    ss_right_value = t.Float()

    view = tu.View(tu.Item('ss_left_value',
                           label='vmin',
                           show_label=True,
                           style='readonly',),
                   tu.Item('ss_right_value',
                           label='vmax',
                           show_label=True,
                           style='readonly'),
                   handler=ImageContrastHandler,
                   buttons=[OKButton,
                            OurApplyButton,
                            OurResetButton,
                            CancelButton, ],
                   title='Constrast adjustment tool',
                   )

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


class ComponentFit(SpanSelectorInSignal1D):
    fit = t.Button()
    only_current = t.List(t.Bool(True))

    view = tu.View(
        tu.Item('fit', show_label=False),
        tu.Item('only_current', show_label=False, style='custom',
                editor=tu.CheckListEditor(values=[(True, 'Only current')])),
        buttons=[OurCloseButton],
        title='Fit single component',
        handler=SpanSelectorInSignal1DHandler,
    )

    def __init__(self, model, component, signal_range=None,
                 estimate_parameters=True, fit_independent=False,
                 only_current=True, **kwargs):
        if model.signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(
                model.signal.axes_manager.signal_dimension, 1)

        self.signal = model.signal
        self.axis = self.signal.axes_manager.signal_axes[0]
        self.span_selector = None
        self.only_current = [True] if only_current else []  # CheckListEditor
        self.model = model
        self.component = component
        self.signal_range = signal_range
        self.estimate_parameters = estimate_parameters
        self.fit_independent = fit_independent
        self.fit_kwargs = kwargs
        if signal_range == "interactive":
            if not hasattr(self.model, '_plot'):
                self.model.plot()
            elif self.model._plot is None:
                self.model.plot()
            elif self.model._plot.is_active() is False:
                self.model.plot()
            self.span_selector_switch(on=True)

    def _fit_fired(self):
        if (self.signal_range != "interactive" and
                self.signal_range is not None):
            self.model.set_signal_range(*self.signal_range)
        elif self.signal_range == "interactive":
            self.model.set_signal_range(self.ss_left_value,
                                        self.ss_right_value)

        # Backup "free state" of the parameters and fix all but those
        # of the chosen component
        if self.fit_independent:
            active_state = []
            for component_ in self.model:
                active_state.append(component_.active)
                if component_ is not self.component:
                    component_.active = False
                else:
                    component_.active = True
        else:
            free_state = []
            for component_ in self.model:
                for parameter in component_.parameters:
                    free_state.append(parameter.free)
                    if component_ is not self.component:
                        parameter.free = False

        # Setting reasonable initial value for parameters through
        # the components estimate_parameters function (if it has one)
        only_current = len(self.only_current) > 0   # CheckListEditor
        if self.estimate_parameters:
            if hasattr(self.component, 'estimate_parameters'):
                if (self.signal_range != "interactive" and
                        self.signal_range is not None):
                    self.component.estimate_parameters(
                        self.signal,
                        self.signal_range[0],
                        self.signal_range[1],
                        only_current=only_current)
                elif self.signal_range == "interactive":
                    self.component.estimate_parameters(
                        self.signal,
                        self.ss_left_value,
                        self.ss_right_value,
                        only_current=only_current)

        if only_current:
            self.model.fit(**self.fit_kwargs)
        else:
            self.model.multifit(**self.fit_kwargs)

        # Restore the signal range
        if self.signal_range is not None:
            self.model.channel_switches = (
                self.model.backup_channel_switches.copy())

        self.model.update_plot()

        if self.fit_independent:
            for component_ in self.model:
                component_.active = active_state.pop(0)
        else:
            # Restore the "free state" of the components
            for component_ in self.model:
                for parameter in component_.parameters:
                    parameter.free = free_state.pop(0)

    def apply(self):
        self._fit_fired()


class IntegrateArea(SpanSelectorInSignal1D):
    integrate = t.Button()

    view = tu.View(
        buttons=[OKButton, CancelButton],
        title='Integrate in range',
        handler=SpanSelectorInSignal1DHandler,
    )

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
