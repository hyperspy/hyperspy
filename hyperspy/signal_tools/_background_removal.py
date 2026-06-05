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
import traits.api as t

from hyperspy import components1d, drawing
from hyperspy.component import Component
from hyperspy.signal_tools import SpanSelectorInSignal1D
from hyperspy.ui_registry import add_gui_method


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
        # for navigation dimension 0, use (0, ) instead of ()
        # to avoid numpy array to float conversion
        indices = self.model.axes_manager.indices or (0,)
        self.red_chisq = float(self.model.red_chisq.data[indices])

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
