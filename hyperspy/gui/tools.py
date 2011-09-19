# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

import enthought.traits.api as t
import enthought.traits.ui.api as tu
from enthought.traits.ui.menu import OKButton, ApplyButton, CancelButton, ModalButtons

from hyperspy.misc import utils
from hyperspy import drawing
from hyperspy.misc.interactive_ns import interactive_ns
from hyperspy.exceptions import SignalOutputDimensionError
from hyperspy.gui import messages

import sys

class CalibrationHandler(tu.Handler):
    def close(self, info, is_ok):
        # Removes the span selector from the plot
        if is_ok is True:
            self.apply(info)
        info.object.roi_selection = False
        return True

    def apply(self, info, *args, **kwargs):
        """Handles the **Apply** button being clicked.
        """
        obj = info.object
        if obj.signal is None: return
        axis = obj.axis
        axis.scale = obj.scale
        axis.offset = obj.offset
        axis.units = obj.units

        obj.last_calibration_stored = True
        obj.roi_selection = False
        obj.signal._replot()
        obj.roi_selection = True
        return

class Calibration(t.HasTraits):
    roi_selection = t.Bool(False)
    left_value = t.Float()
    right_value = t.Float()
    offset = t.Float()
    scale = t.Float()
    units = t.Unicode()
    ok = t.Button()   
    view = tu.View(
        tu.Group(
            'left_value',
            'right_value',
            'offset',
            'scale',
            'units',),
            handler = CalibrationHandler,
            buttons = [OKButton, ApplyButton, CancelButton])
            
    def __init__(self, signal):
        if signal.axes_manager.signal_dimension != 1:
         raise SignalOutputDimensionError(signal.axes.signal_dimension, 1)
            
        self.signal = signal
        self.axis = self.signal.axes_manager._slicing_axes[0]
        self.units = self.axis.units
        self.bg_span_selector = None
        self.signal.plot()
        self.roi_selection = True
        self.last_calibration_stored = True
            
    def _roi_selection_changed(self, old, new):
        if new is True:
            self.bg_span_selector = \
            drawing.widgets.ModifiableSpanSelector(
            self.signal._plot.spectrum_plot.left_ax,
            onselect = self._update_calibration,
            onmove_callback = self._update_calibration)
        elif self.bg_span_selector is not None:
            self.bg_span_selector.turn_off()
            self.bg_span_selector = None

    def _left_value_changed(self, old, new):
        if self.bg_span_selector.range is None:
            messages.information('Please select a range in the spectrum '
            'figure by dragging the mouse over it')
            return
        else:
            self._update_calibration()
            
    def _left_value_changed(self, old, new):
        if self.bg_span_selector is not None and \
        self.bg_span_selector.range is None:
            messages.information('Please select a range in the spectrum figure' 
            'by dragging the mouse over it')
            return
        else:
            self._update_calibration()
    
    def _right_value_changed(self, old, new):
        if self.bg_span_selector.range is None:
            messages.information('Please select a range in the spectrum figure' 
            'by dragging the mouse over it')
            return
        else:
            self._update_calibration()
            
    def _update_calibration(self, *args, **kwargs):
        if self.left_value == self.right_value:
            return
            
        left = self.bg_span_selector.rect.get_x()
        right = left + self.bg_span_selector.rect.get_width()
        lc = self.axis.value2index(left)
        rc = self.axis.value2index(right)
        self.offset, self.scale = self.axis.calibrate(
            (self.left_value, self.right_value), (lc,rc),
            modify_calibration = False)

class Smoothing(t.HasTraits):

    def __init__(self, signal):
        self.ax = None
        self.data_line = None
        self.smooth_line = None
        self.signal = signal
        self.axis = self.signal.axes_manager._slicing_axes[0].axis
        self.plot()
                   
    def plot(self):
        self.signal.plot()
        hse = self.signal._plot
        l1 = hse.spectrum_plot.left_ax_lines[0]
        color = l1.line.get_color()
        l1.line_properties_helper(color, 'scatter')
        l1.set_properties()
        
        l2 = drawing.spectrum.SpectrumLine()
        l2.data_function = self.model2plot
        l2.line_properties_helper('blue', 'line')        
        # Add the line to the figure
          
        hse.spectrum_plot.add_line(l2)
        l2.plot()
        self.data_line = l1
        self.smooth_line = l2
    

class SmoothingSavitzkyGolay(Smoothing):
    polynomial_order = t.Int(3)
    number_of_points = t.Int(5)
    differential_order = t.Int(0)
    view = tu.View(
        tu.Group(
            'polynomial_order',
            'number_of_points',
            'differential_order',),
            kind = 'nonmodal',
            buttons= ModalButtons)

    def _polynomial_order_changed(self, old, new):
        self.smooth_line.update()
        
    def _number_of_points_changed(self, old, new):
        self.smooth_line.update()
        
    def _differential_order_changed(self, old, new):
        self.smooth_line.update()
            
    def model2plot(self, axes_manager = None):
        smoothed = utils.sg(self.signal(), self.number_of_points, 
                            self.polynomial_order, self.differential_order)
        return smoothed
            
class SmoothingLowess(Smoothing):
    smoothing_parameter = t.Float(2/3.)
    number_of_iterations = t.Int(3)
    view = tu.View(
        tu.Group(
            'smoothing_parameter',
            'number_of_iterations',),
            kind = 'nonmodal',
            buttons= ModalButtons,)
            
    def _smoothing_parameter_changed(self, old, new):
        self.smooth_line.update()
        
    def _number_of_iterations_changed(self, old, new):
        self.smooth_line.update()
            
    def model2plot(self, axes_manager = None):
        smoothed = utils.lowess(self.axis, self.signal(), 
                                self.smoothing_parameter, 
                                self.number_of_iterations)
        return smoothed
