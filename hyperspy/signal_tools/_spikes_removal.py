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

import logging

import matplotlib
import numpy as np
import scipy
import traits.api as t

from hyperspy import drawing, signal_tools
from hyperspy.misc.math_tools import check_random_state
from hyperspy.ui_registry import add_gui_method

_logger = logging.getLogger(__name__)


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
            intp = scipy.interpolate.make_interp_spline(x, y, k=self.spline_order)
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
class SpikesRemovalInteractive(SpikesRemoval, signal_tools.SpanSelectorInSignal1D):
    threshold = t.Float(
        400, desc="the derivative magnitude threshold above\nwhich to find spikes"
    )
    click_to_show_instructions = t.Button()
    show_derivative_histogram = t.Button()
    spline_order = t.Range(
        1,
        10,
        1,
        desc="the order of the spline used to\nconnect the reconstructed data",
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
        signal_tools.SpanSelectorInSignal1D.__init__(self, signal=signal)
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
            m = signal_tools.SimpleMessage()
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


# For creating a text handler in legend (to label derivative magnitude)
class DerivativeTextParameters(object):
    def __init__(self, text, color):
        self.my_text = text
        self.my_color = color


class DerivativeTextHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        patch = matplotlib.text.Text(
            text=orig_handle.my_text, color=orig_handle.my_color
        )
        handlebox.add_artist(patch)
        return patch


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
