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

import numpy as np
import traits.api as t

from hyperspy.axes import UniformDataAxis
from hyperspy.exceptions import SignalDimensionError
from hyperspy.signal_tools import LineInSignal2D, SpanSelectorInSignal1D
from hyperspy.ui_registry import add_gui_method

_logger = logging.getLogger(__name__)


@add_gui_method(toolkey="hyperspy.Signal2D.calibrate")
class Signal2DCalibration(LineInSignal2D):
    new_length = t.Float(t.Undefined, label="New length")
    scale = t.Float()
    units = t.Unicode()

    def __init__(self, signal, **kwargs):
        super().__init__(signal, **kwargs)
        self.units = self.signal.axes_manager.signal_axes[0].units
        self.scale = self.signal.axes_manager.signal_axes[0].scale

    def _new_length_changed(self, old, new):
        if old != new and self._line is not None:
            self._calculate_scale()

    def _length_changed(self, old, new):
        if old != new and self._line is not None:
            self._calculate_scale()

    def _calculate_scale(self):
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
        self.close()
        self.signal._replot()


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
                "Select the new left and right values before pressing apply."
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
