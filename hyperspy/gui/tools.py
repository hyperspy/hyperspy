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

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import traits.api as t
import traitsui.api as tu
from traitsui.menu import (OKButton, ApplyButton, CancelButton, ModalButtons,
    OKCancelButtons)

from hyperspy.misc import utils
from hyperspy import drawing
from hyperspy.misc.interactive_ns import interactive_ns
from hyperspy.exceptions import SignalOutputDimensionError
from hyperspy.gui import messages
from hyperspy.misc.progressbar import progressbar
from hyperspy.misc.tv_denoise import _tv_denoise_1d
from hyperspy.drawing.utils import does_figure_object_exists
from hyperspy.gui.mpl_traits_editor import MPLFigureEditor

import sys

OurApplyButton = tu.Action(name = "Apply",
                           action = "apply")
                           
OurResetButton = tu.Action(name = "Reset",
                           action = "reset")
                           
OurFindButton = tu.Action(name = "Find",
                           action = "find")
                           
OurPreviousButton = tu.Action(name = "Previous",
                           action = "back")
                           
                
class SmoothingHandler(tu.Handler):
    def close(self, info, is_ok):
        # Removes the span selector from the plot
        if is_ok is True:
            info.object.apply()
        else:
            info.object.close()
        return True


class SpanSelectorInSpectrumHandler(tu.Handler):
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
            obj.next()
        return

class SpectrumRangeSelectorHandler(tu.Handler):
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


class CalibrationHandler(SpanSelectorInSpectrumHandler):

    def apply(self, info, *args, **kwargs):
        """Handles the **Apply** button being clicked.
        """
        if info.object.signal is None: return
        axis = info.object.axis
        axis.scale = info.object.scale
        axis.offset = info.object.offset
        axis.units = info.object.units
        info.object.span_selector_switch(on = False)
        info.object.signal._replot()
        info.object.span_selector_switch(on = True)
        info.object.last_calibration_stored = True
        return
        
class SpanSelectorInSpectrum(t.HasTraits):
    ss_left_value = t.Float()
    ss_right_value = t.Float()
    is_ok = t.Bool(False)
            
    def __init__(self, signal):
        if signal.axes_manager.signal_dimension != 1:
         raise SignalOutputDimensionError(signal.axes.signal_dimension, 1)
        
        self.signal = signal
        self.axis = self.signal.axes_manager.signal_axes[0]
        self.span_selector = None
        self.signal.plot()
        self.span_selector_switch(on = True)
        
    def on_disabling_span_selector(self):
        pass
            
    def span_selector_switch(self, on):
        if not self.signal._plot.is_active(): return
        
        if on is True:
            self.span_selector = \
            drawing.widgets.ModifiableSpanSelector(
            self.signal._plot.signal_plot.ax,
            onselect = self.update_span_selector_traits,
            onmove_callback = self.update_span_selector_traits)

        elif self.span_selector is not None:
            self.on_disabling_span_selector()
            self.span_selector.turn_off()
            self.span_selector = None

    def update_span_selector_traits(self, *args, **kwargs):
        if not self.signal._plot.is_active(): return
        self.ss_left_value = self.span_selector.rect.get_x()
        self.ss_right_value = self.ss_left_value + \
            self.span_selector.rect.get_width()
            
    def reset_span_selector(self):
        self.span_selector_switch(False)
        self.ss_left_value = 0
        self.ss_right_value = 0
        self.span_selector_switch(True)
        

class SpectrumCalibration(SpanSelectorInSpectrum):
    left_value = t.Float(label = 'New left value')
    right_value = t.Float(label = 'New right value')
    offset = t.Float()
    scale = t.Float()
    units = t.Unicode()
    view = tu.View(
        tu.Group(
            'left_value',
            'right_value',
            tu.Item('ss_left_value', label = 'Left', style = 'readonly'),
            tu.Item('ss_right_value', label = 'Right', style = 'readonly'),
            tu.Item(name = 'offset', style = 'readonly'),
            tu.Item(name = 'scale', style = 'readonly'),
            'units',),
        handler = CalibrationHandler,
        buttons = [OKButton, OurApplyButton, CancelButton],
        kind = 'live',
        title = 'Calibration parameters')
            
    def __init__(self, signal):
        super(SpectrumCalibration, self).__init__(signal)
        if signal.axes_manager.signal_dimension != 1:
            raise SignalOutputDimensionError(signal.axes.signal_dimension, 1)
        self.units = self.axis.units
        self.last_calibration_stored = True
            
    def _left_value_changed(self, old, new):
        if self.span_selector is not None and \
        self.span_selector.range is None:
            messages.information('Please select a range in the spectrum figure' 
            'by dragging the mouse over it')
            return
        else:
            self._update_calibration()
    
    def _right_value_changed(self, old, new):
        if self.span_selector.range is None:
            messages.information('Please select a range in the spectrum figure' 
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
            (self.left_value, self.right_value), (lc,rc),
            modify_calibration = False)
            
class SpectrumRangeSelector(SpanSelectorInSpectrum):
    on_close = t.List()
        
    view = tu.View(
        tu.Item('ss_left_value', label = 'Left', style = 'readonly'),
        tu.Item('ss_right_value', label = 'Right', style = 'readonly'),
        handler = SpectrumRangeSelectorHandler,
        buttons = [OKButton, OurApplyButton, CancelButton],)
            

class Smoothing(t.HasTraits):
    line_color = t.Color('blue')
    differential_order = t.Int(0)
    crop_diff_axis = True
    
    def __init__(self, signal):
        self.ax = None
        self.data_line = None
        self.smooth_line = None
        self.signal = signal
        self.axis = self.signal.axes_manager.signal_axes[0].axis
        self.plot()
                   
    def plot(self):
        if self.signal._plot is None or not \
            does_figure_object_exists(self.signal._plot.signal_plot.figure):
            self.signal.plot()
        hse = self.signal._plot
        l1 = hse.signal_plot.ax_lines[0]
        self.original_color = l1.line.get_color()
        l1.line_properties_helper(self.original_color, 'scatter')
        l1.set_properties()
        
        l2 = drawing.spectrum.SpectrumLine()
        l2.data_function = self.model2plot
        l2.line_properties_helper(np.array(
            self.line_color.Get())/255., 'line')   
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
        self.smooth_diff_line = drawing.spectrum.SpectrumLine()
        self.smooth_diff_line.data_function = self.diff_model2plot
        self.smooth_diff_line.line_properties_helper(np.array(
            self.line_color.Get())/255., 'line')   
        self.signal._plot.signal_plot.add_line(self.smooth_diff_line,
                                                 ax = 'right')
        self.smooth_diff_line.axes_manager = self.signal.axes_manager
        
    def turn_diff_line_off(self):
        if self.smooth_diff_line is None: return
        self.smooth_diff_line.close()
        self.smooth_diff_line = None
        
    def _differential_order_changed(self, old, new):
        if old == 0:
            self.turn_diff_line_on(new)
        if new == 0:
            self.turn_diff_line_off()
            return
        if self.crop_diff_axis is True:
            self.smooth_diff_line.axis =\
                self.axis[:-new] + (self.axis[1] - self.axis[0]) * new
        if old == 0:
            self.smooth_diff_line.plot()
        self.smooth_diff_line.update(force_replot = True)    
        
    def _line_color_changed(self, old, new):
        self.smooth_line.line_properties['color'] = np.array(
            self.line_color.Get())/255.
        self.smooth_line.set_properties()
        if self.smooth_diff_line is not None:
            self.smooth_diff_line.line_properties['color'] = np.array(
            self.line_color.Get())/255.
            self.smooth_diff_line.set_properties()
        self.update_lines()
            
    def diff_model2plot(self, axes_manager = None):
        smoothed = np.diff(self.model2plot(axes_manager),
            self.differential_order)
        return smoothed
        
    def apply(self):
        self.signal._plot.auto_update_plot = False
        pbar = progressbar(
        maxval = (np.cumprod(self.signal.axes_manager.navigation_shape)[-1]))
        up_to = None
        if self.differential_order == 0:
            f = self.model2plot
        else:
            f = self.diff_model2plot
            if self.crop_diff_axis is True:
                up_to = -self.differential_order
        i = 0
        for index in np.ndindex(
        tuple(self.signal.axes_manager.navigation_shape)):
            self.signal.axes_manager.set_not_slicing_indexes(index)
            self.signal.data[
            self.signal.axes_manager._getitem_tuple][:up_to]\
                 = f()
            i += 1
            pbar.update(i)
        pbar.finish()
        if self.differential_order > 0:
            self.signal.axes_manager.signal_axes[0].offset = \
                self.smooth_diff_line.axis[0]
            self.signal.crop_in_pixels(-1,0,-self.differential_order)
        self.signal._replot()
        self.signal._plot.auto_update_plot = True
        
    def close(self):
        if self.signal._plot.is_active():
            if self.differential_order != 0:
                self.turn_diff_line_off()
            self.smooth_line.close()
            self.data_line.line_properties_helper(self.original_color, 'line')
            self.data_line.set_properties()
        

class SmoothingSavitzkyGolay(Smoothing):
    polynomial_order = t.Int(3)
    number_of_points = t.Int(5)
    crop_diff_axis = False
    view = tu.View(
        tu.Group(
            'polynomial_order',
            'number_of_points',
            'differential_order',
            'line_color'),
            kind = 'live',
            handler = SmoothingHandler,
            buttons= OKCancelButtons,
            title = 'Savitzky-Golay Smoothing',)

    def _polynomial_order_changed(self, old, new):
        self.update_lines()
        
    def _number_of_points_changed(self, old, new):
        self.update_lines()
    def _differential_order(self, old, new):
        self.update_lines()
        
    def diff_model2plot(self, axes_manager = None):
        smoothed = utils.sg(self.signal(), self.number_of_points, 
                            self.polynomial_order, self.differential_order)
        return smoothed
                                        
    def model2plot(self, axes_manager = None):
        smoothed = utils.sg(self.signal(), self.number_of_points, 
                            self.polynomial_order, 0)
        return smoothed
            
class SmoothingLowess(Smoothing):
    smoothing_parameter = t.Float(2/3.)
    number_of_iterations = t.Int(3)
    differential_order = t.Int(0)
    view = tu.View(
        tu.Group(
            'smoothing_parameter',
            'number_of_iterations',
            'differential_order',
            'line_color'),
            kind = 'live',
            handler = SmoothingHandler,
            buttons= OKCancelButtons,
            title = 'Lowess Smoothing',)
            
    def _smoothing_parameter_changed(self, old, new):
        self.update_lines()
        
    def _number_of_iterations_changed(self, old, new):
        self.update_lines()
            
    def model2plot(self, axes_manager = None):
        smoothed = utils.lowess(self.axis, self.signal(), 
                                self.smoothing_parameter, 
                                self.number_of_iterations)
                            
        return smoothed

class SmoothingTV(Smoothing):
    smoothing_parameter = t.Float(200)

    view = tu.View(
        tu.Group(
            'smoothing_parameter',
            'differential_order',
            'line_color'),
            kind = 'live',
            handler = SmoothingHandler,
            buttons= OKCancelButtons,
            title = 'Total Variation Smoothing',)
            
    def _smoothing_parameter_changed(self, old, new):
        self.update_lines()
        
    def _number_of_iterations_changed(self, old, new):
        self.update_lines()
            
    def model2plot(self, axes_manager = None):
        smoothed = _tv_denoise_1d(self.signal(), 
                                weight = self.smoothing_parameter,)
        return smoothed
        
class ButterworthFilter(Smoothing):
    cutoff_frequency_ratio = t.Range(0.,1.,0.05)
    type = t.Enum('low', 'high')
    order = t.Int(2)
    
    view = tu.View(
        tu.Group(
            'cutoff_frequency_ratio',
            'order',
            'type'),
            kind = 'live',
            handler = SmoothingHandler,
            buttons= OKCancelButtons,
            title = 'Butterworth filter',)
            
    def _cutoff_frequency_ratio_changed(self, old, new):
        self.update_lines()
        
    def _type_changed(self, old, new):
        self.update_lines()
        
    def _order_changed(self, old, new):
        self.update_lines()
            
    def model2plot(self, axes_manager = None):
        b, a = sp.signal.butter(self.order, self.cutoff_frequency_ratio,
                                self.type)
        smoothed = sp.signal.filtfilt(b, a, self.signal())
        return smoothed

        
class Load(t.HasTraits):
    filename = t.File
    traits_view = tu.View(
        tu.Group('filename'),
        kind = 'livemodal',
        buttons = [OKButton, CancelButton],
        title = 'Load file')
        
class ImageContrastHandler(tu.Handler):
    def close(self, info, is_ok):
#        # Removes the span selector from the plot
#        info.object.span_selector_switch(False)
#        if is_ok is True:
#            self.apply(info)
        if is_ok is False:
            info.object.image.update_image(auto_contrast=True)
        info.object.close()
        return True

    def apply(self, info, *args, **kwargs):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object
        obj.apply()
        
        return
        
    def reset(self, info, *args, **kwargs):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object
        obj.reset()
        return

    def our_help(self, info, *args, **kwargs):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object._help()
    
                
class ImageContrastEditor(t.HasTraits):
    ss_left_value = t.Float()
    ss_right_value = t.Float()

    view = tu.View( tu.Item('ss_left_value',
                            label = 'vmin',
                            show_label=True,
                            style = 'readonly',),
                    tu.Item('ss_right_value',
                            label = 'vmax',
                            show_label=True,
                            style = 'readonly'),

#                    resizable=True,
                    handler = ImageContrastHandler,
                    buttons = [OKButton,
                               OurApplyButton,
                               OurResetButton,
                               CancelButton,],
                    title = 'Constrast adjustment tool',
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
            onselect = self.update_span_selector_traits,
            onmove_callback = self.update_span_selector_traits)

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
        vmin = vmin - pad
        vmax = vmax + pad
        data = self.image.data_function().ravel()
        self.patches = self.ax.hist(data,100, range = (vmin, vmax),
                                    color = 'blue')[2]
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(vmin, vmax)
        self.ax.figure.canvas.draw()

    def reset(self):
        data = self.image.data_function().ravel()
        self.image.vmin, self.image.vmax = np.nanmin(data),np.nanmax(data)
        self.image.update_image(auto_contrast=False)
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
        self.image.update_image(auto_contrast=False)
        self.update_histogram()
        
    def close(self):
        plt.close(self.ax.figure)
        


        
