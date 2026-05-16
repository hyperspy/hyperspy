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

from hyperspy.drawing._widgets.range import SpanSelector
from hyperspy.exceptions import SignalDimensionError


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


class Signal1DRangeSelector(SpanSelectorInSignal1D):
    on_close = t.List()
