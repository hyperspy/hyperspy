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

import numpy as np
import scipy as sp
import traits.api as t
import traitsui.api as tu
from traitsui.menu import OKButton, CancelButton

from hyperspy import components
from hyperspy.component import Component
from hyperspy import drawing
from hyperspy.gui.tools import (SpanSelectorInSpectrum,
                                SpanSelectorInSpectrumHandler,
                                OurFindButton,
                                OurPreviousButton,
                                OurApplyButton)
import hyperspy.gui.messages as messages


class BackgroundRemoval(SpanSelectorInSpectrum):
    background_type = t.Enum(
        'Power Law',
        'Gaussian',
        'Offset',
        'Polynomial',
        default='Power Law')
    polynomial_order = t.Range(1, 10)
    background_estimator = t.Instance(Component)
    bg_line_range = t.Enum('from_left_range',
                           'full',
                           'ss_range',
                           default='full')
    hi = t.Int(0)
    view = tu.View(
        tu.Group(
            'background_type',
            tu.Group(
                'polynomial_order',
                visible_when='background_type == \'Polynomial\''),),
            buttons=[OKButton, CancelButton],
            handler=SpanSelectorInSpectrumHandler,
            title='Background removal tool')

    def __init__(self, signal):
        super(BackgroundRemoval, self).__init__(signal)
        self.set_background_estimator()
        self.bg_line = None

    def on_disabling_span_selector(self):
        if self.bg_line is not None:
            self.bg_line.close()
            self.bg_line = None

    def set_background_estimator(self):

        if self.background_type == 'Power Law':
            self.background_estimator = components.PowerLaw()
            self.bg_line_range = 'from_left_range'
        elif self.background_type == 'Gaussian':
            self.background_estimator = components.Gaussian()
            self.bg_line_range = 'full'
        elif self.background_type == 'Offset':
            self.background_estimator = components.Offset()
            self.bg_line_range = 'full'
        elif self.background_type == 'Polynomial':
            self.background_estimator = \
                components.Polynomial(self.polynomial_order)
            self.bg_line_range = 'full'

    def _polynomial_order_changed(self, old, new):
        self.background_estimator = components.Polynomial(new)
        self.span_selector_changed()

    def _background_type_changed(self, old, new):
        self.set_background_estimator()
        self.span_selector_changed()

    def _ss_left_value_changed(self, old, new):
        self.span_selector_changed()

    def _ss_right_value_changed(self, old, new):
        self.span_selector_changed()

    def create_background_line(self):
        self.bg_line = drawing.spectrum.SpectrumLine()
        self.bg_line.data_function = self.bg_to_plot
        self.bg_line.set_line_properties(
            color='blue',
            type='line')
        self.signal._plot.signal_plot.add_line(self.bg_line)
        self.bg_line.autoscale = False
        self.bg_line.plot()

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

    def span_selector_changed(self):
        if (self.ss_left_value is np.nan) or (self.ss_right_value is np.nan):
            return
        if self.background_estimator is None:
            print("No bg estimator")
            return
        if self.bg_line is None and \
            self.background_estimator.estimate_parameters(
                self.signal, self.ss_left_value, self.ss_right_value,
                only_current=True) is True:
            self.create_background_line()
        else:
            self.bg_line.update()

    def apply(self):
        self.signal._plot.auto_update_plot = False
        new_spectra = self.signal._remove_background_cli(
            (self.ss_left_value, self.ss_right_value),
            self.background_estimator)
        self.signal.data = new_spectra.data
        self.signal._replot()
        self.signal._plot.auto_update_plot = True


class SpikesRemovalHandler(tu.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        info.object.span_selector_switch(False)
        return True

    def apply(self, info, *args, **kwargs):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object
        obj.is_ok = True
        if hasattr(obj, 'apply'):
            obj.apply()

        return

    def find(self, info, *args, **kwargs):
        """Handles the **Next** button being clicked.

        """
        obj = info.object
        obj.is_ok = True
        if hasattr(obj, 'find'):
            obj.find()
        return

    def back(self, info, *args, **kwargs):
        """Handles the **Next** button being clicked.

        """
        obj = info.object
        obj.is_ok = True
        if hasattr(obj, 'find'):
            obj.find(back=True)
        return


class SpikesRemoval(SpanSelectorInSpectrum):
    interpolator_kind = t.Enum(
        'Linear',
        'Spline',
        default='Linear')
    threshold = t.Float()
    show_derivative_histogram = t.Button()
    spline_order = t.Range(1, 10, 3)
    interpolator = None
    default_spike_width = t.Int(5)
    index = t.Int(0)
    add_noise = t.Bool(True,
                       desc="Add noise to the healed portion of the "
                       "spectrum. Use the noise properties "
                       "defined in metadata if present, otherwise "
                       "it defaults to shot noise.")
    view = tu.View(tu.Group(
        tu.Group(
            tu.Item('show_derivative_histogram', show_label=False),
            'threshold',
            show_border=True,),
        tu.Group(
            'add_noise',
            'interpolator_kind',
            'default_spike_width',
            tu.Group(
                'spline_order',
                visible_when='interpolator_kind == \'Spline\''),
            show_border=True,
            label='Advanced settings'),
    ),
        buttons=[OKButton,
                 OurPreviousButton,
                 OurFindButton,
                 OurApplyButton, ],
        handler=SpikesRemovalHandler,
        title='Spikes removal tool')

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
        self.threshold = 400
        self.index = 0
        self.argmax = None
        self.kind = "linear"
        self._temp_mask = np.zeros(self.signal().shape, dtype='bool')
        self.signal_mask = signal_mask
        self.navigation_mask = navigation_mask
        md = self.signal.metadata
        from hyperspy.signal import Signal
        if "Signal.Noise_properties" in md:
            if "Signal.Noise_properties.variance" in md:
                self.noise_variance = md.Signal.Noise_properties.variance
                if isinstance(md.Signal.Noise_properties.variance, Signal):
                    self.noise_type = "heteroscedastic"
                else:
                    self.noise_type = "white"
        else:
            self.noise_type = "shot noise"

    def _threshold_changed(self, old, new):
        self.index = 0
        self.update_plot()

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
        while not spike and (
                (self.index < ncoordinates - 1 and back is False) or
                (self.index > 0 and back is True)):
            if back is False:
                self.index += 1
            else:
                self.index -= 1
            spike = self.detect_spike()

        if spike is False:
            messages.information('End of dataset reached')
            self.index = 0
            self._reset_line()
            return
        else:
            minimum = max(0, self.argmax - 50)
            maximum = min(len(self.signal()) - 1, self.argmax + 50)
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
            self.signal._plot.pointer.update_patch_position()

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
        self.span_selector_changed()

    def _ss_right_value_changed(self, old, new):
        self.span_selector_changed()

    def create_interpolation_line(self):
        self.interpolated_line = drawing.spectrum.SpectrumLine()
        self.interpolated_line.data_function = \
            self.get_interpolated_spectrum
        self.interpolated_line.set_line_properties(
            color='blue',
            type='line')
        self.signal._plot.signal_plot.add_line(self.interpolated_line)
        self.interpolated_line.autoscale = False
        self.interpolated_line.plot()

    def get_interpolation_range(self):
        axis = self.signal.axes_manager.signal_axes[0]
        if self.ss_left_value == self.ss_right_value:
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
            pad = 10
        ileft = left - pad
        iright = right + pad
        ileft = np.clip(ileft, 0, len(data))
        iright = np.clip(iright, 0, len(data))
        left = int(np.clip(left, 0, len(data)))
        right = int(np.clip(right, 0, len(data)))
        x = np.hstack((axis.axis[ileft:left], axis.axis[right:iright]))
        y = np.hstack((data[ileft:left], data[right:iright]))
        if ileft == 0:
            # Extrapolate to the left
            data[left:right] = data[right + 1]

        elif iright == (len(data) - 1):
            # Extrapolate to the right
            data[left:right] = data[left - 1]

        else:
            # Interpolate
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
        self.signal()[:] = self.get_interpolated_spectrum()
        self.update_spectrum_line()
        self.interpolated_line.close()
        self.interpolated_line = None
        self.reset_span_selector()
        self.find()
