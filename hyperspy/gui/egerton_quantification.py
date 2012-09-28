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

import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import traits.api as t
import traitsui.api as tu
from traitsui.menu import OKButton, ApplyButton, CancelButton

from hyperspy import components
from hyperspy.component import Component
from hyperspy.misc import utils
from hyperspy import drawing
from hyperspy.misc.interactive_ns import interactive_ns
from hyperspy.gui.tools import (SpanSelectorInSpectrum, 
    SpanSelectorInSpectrumHandler,OurFindButton, OurPreviousButton,
    OurApplyButton)
from hyperspy.misc.progressbar import progressbar
import hyperspy.gui.messages as messages


class BackgroundRemoval(SpanSelectorInSpectrum):
    background_type = t.Enum(
        'Power Law',
        'Gaussian',
        'Offset',
        'Polynomial',
        default = 'Power Law')
    polynomial_order = t.Range(1,10)
    background_estimator = t.Instance(Component)
    bg_line_range = t.Enum('from_left_range',
                           'full',
                           'ss_range',
                           default = 'full')
    hi = t.Int(0)
    view = tu.View(
        tu.Group(
            'background_type',
            tu.Group(
                'polynomial_order', 
                visible_when = 'background_type == \'Polynomial\''),),
            buttons= [OKButton, CancelButton],
            handler = SpanSelectorInSpectrumHandler,
            title = 'Background removal tool')
                 
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
        self.bg_line.line_properties_helper('blue', 'line')
        self.signal._plot.signal_plot.add_line(self.bg_line)
        self.bg_line.autoscale = False
        self.bg_line.plot()
        
    def bg_to_plot(self, axes_manager=None, fill_with=np.nan):
        # First try to update the estimation
        self.background_estimator.estimate_parameters(
            self.signal, self.ss_left_value, self.ss_right_value, 
            only_current = True) 
            
        if self.bg_line_range == 'from_left_range':
            bg_array = np.zeros(self.axis.axis.shape)
            bg_array[:] = fill_with
            from_index = self.axis.value2index(self.ss_left_value)
            bg_array[from_index:] = self.background_estimator.function(
                self.axis.axis[from_index:])
            return bg_array
        elif self.bg_line_range == 'full':
            return self.background_estimator.function(self.axis.axis)
        elif self.bg_line_range == 'ss_range':
            bg_array = np.zeros(self.axis.axis.shape)
            bg_array[:] = fill_with
            from_index = self.axis.value2index(self.ss_left_value)
            to_index = self.axis.value2index(self.ss_right_value)
            bg_array[from_index:] = self.background_estimator.function(
                self.axis.axis[from_index:to_index])
                      
    def span_selector_changed(self):
        if self.background_estimator is None:
            print("No bg estimator")
            return
        if self.bg_line is None and \
            self.background_estimator.estimate_parameters(
                self.signal, self.ss_left_value, self.ss_right_value, 
                only_current = True) is True:
            self.create_background_line()
        else:
            self.bg_line.update()
            
    def apply(self):
        self.signal._plot.auto_update_plot = False
        maxval = self.signal.axes_manager.navigation_size
        if maxval > 0:
            pbar = progressbar(maxval=maxval)
        i = 0
        self.bg_line_range = 'full'
        for s in self.signal:
            s.data[:] -= \
            np.nan_to_num(self.bg_to_plot(self.signal.axes_manager,
                                          0))
            if self.background_type == 'Power Law':
                s.data[:self.axis.value2index(self.ss_right_value)] = 0
                
            i+=1
            if maxval > 0:
                pbar.update(i)
        if maxval > 0:
            pbar.finish()
            
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
        default = 'Linear')
    threshold = t.Float()
    show_derivative_histogram = t.Button()
    spline_order = t.Range(1,10, 3)
    interpolator = None
    default_spike_width = t.Int(5)
    index = t.Int(0)
    view = tu.View(tu.Group(
        tu.Group(
                 tu.Item('show_derivative_histogram', show_label=False),
                 'threshold',
                 show_border=True,),
        tu.Group(
            'interpolator_kind',
            'default_spike_width',
            tu.Group(
                'spline_order', 
                visible_when = 'interpolator_kind == \'Spline\''),
            show_border=True,
            label='Advanced settings'),
            ),
            buttons= [OKButton,
                      OurPreviousButton,
                      OurFindButton,
                      OurApplyButton,],
            handler = SpikesRemovalHandler,
            title = 'Spikes removal tool')
                 
    def __init__(self, signal,navigation_mask=None, signal_mask=None):
        super(SpikesRemoval, self).__init__(signal)
        self.interpolated_line = None
        self.coordinates = [coordinate for coordinate in np.ndindex(
                            tuple(signal.axes_manager.navigation_shape))
                            if (navigation_mask is None or not 
                                navigation_mask[coordinate])]
        self.signal = signal
        sys.setrecursionlimit(np.cumprod(self.signal.data.shape)[-1])
        self.line = signal._plot.signal_plot.ax_lines[0]
        self.ax = signal._plot.signal_plot.ax
        signal._plot.auto_update_plot = False
        signal.axes_manager.coordinates = self.coordinates[0]
        self.threshold = 400
        self.index = 0
        self.argmax = None
        self.kind = "linear"
        self._temp_mask = np.zeros(self.signal().shape, dtype='bool')
        self.signal_mask = signal_mask
        self.navigation_mask = navigation_mask
        
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

    def find(self, back=False):
        if ((self.index == len(self.coordinates) - 1 and back is False)
        or (back is True and self.index == 0)):
            messages.information('End of dataset reached')
            return
        if self.interpolated_line is not None:
            self.interpolated_line.close()
            self.interpolated_line = None
            self.reset_span_selector()
        
        if self.detect_spike() is False:
            if back is False:
                self.index += 1
            else:
                self.index -= 1
            self.find(back=back)
        else:
            minimum = max(0,self.argmax - 50)
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
        self.signal._plot.pointer.update_patch_position()
        
    def update_spectrum_line(self):
        self.line.auto_update = True
        self.line.update()
        self.line.auto_update = False
        
    def _index_changed(self, old, new):
        self.signal.axes_manager.coordinates = self.coordinates[new]
        self.argmax = None
        self._temp_mask[:] = False
        
    def on_disabling_span_selector(self):
        if self.interpolated_line is not None:
            self.interpolated_line.close()
            self.interpolated_line = None
           
    def _spline_order_changed(self, old, new):
        self.kind = self.spline_order
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
        self.interpolated_line.line_properties_helper('blue', 'line')
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
            
        return left,right
        
        
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
        left = np.clip(left, 0, len(data))
        right = np.clip(right, 0, len(data))
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
        data = np.random.poisson(np.clip(data, 0, np.inf))
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
        
    
#class EgertonPanel(t.HasTraits):
#    define_background_window = t.Bool(False)
#    bg_window_size_variation = t.Button()
#    background_substracted_spectrum_name = t.Str('signal')
#    extract_background = t.Button()    
#    define_signal_window = t.Bool(False)
#    signal_window_size_variation = t.Button()
#    signal_name = t.Str('signal')
#    extract_signal = t.Button()
#    view = tu.View(tu.Group(
#        tu.Group('define_background_window',
#                 tu.Item('bg_window_size_variation', 
#                         label = 'window size effect', show_label=False),
#                 tu.Item('background_substracted_spectrum_name'),
#                 tu.Item('extract_background', show_label=False),
#                 ),
#        tu.Group('define_signal_window',
#                 tu.Item('signal_window_size_variation', 
#                         label = 'window size effect', show_label=False),
#                 tu.Item('signal_name', show_label=True),
#                 tu.Item('extract_signal', show_label=False)),))
#                 
#    def __init__(self, signal):
#        
#        self.signal = signal
#        
#        # Background
#        self.span_selector = None
#        self.background_estimator = components.PowerLaw()
#        self.bg_line = None
#        self.bg_cube = None
#                
#        # Signal
#        self.signal_span_selector = None
#        self.signal_line = None
#        self.signal_map = None
#        self.map_ax = None
#    
#    def store_current_spectrum_bg_parameters(self, *args, **kwards):
#        if self.define_background_window is False or \
#        self.span_selector.range is None: return
#        pars = utils.two_area_powerlaw_estimation(
#        self.signal, *self.span_selector.range,only_current_spectrum = True)
#        self.background_estimator.r.value = pars['r']
#        self.background_estimator.A.value = pars['A']
#                     
#        if self.define_signal_window is True and \
#        self.signal_span_selector.range is not None:
#            self.background_estimatorot_signal_map()
#                     
#    def _define_background_window_changed(self, old, new):
#        if new is True:
#            self.span_selector = \
#            drawing.widgets.ModifiableSpanSelector(
#            self.signal.hse.signal_plot.ax,
#            onselect = self.store_current_spectrum_bg_parameters,
#            onmove_callback = self.background_estimatorot_bg_removed_spectrum)
#        elif self.span_selector is not None:
#            if self.bg_line is not None:
#                self.span_selector.ax.lines.remove(self.bg_line)
#                self.bg_line = None
#            if self.signal_line is not None:
#                self.span_selector.ax.lines.remove(self.signal_line)
#                self.signal_line = None
#            self.span_selector.turn_off()
#            self.span_selector = None
#                      
#    def _bg_window_size_variation_fired(self):
#        if self.define_background_window is False: return
#        left = self.span_selector.rect.get_x()
#        right = left + self.span_selector.rect.get_width()
#        energy_window_dependency(self.signal, left, right, min_width = 10)
#        
#    def _extract_background_fired(self):
#        if self.background_estimator is None: return
#        signal = self.signal() - self.background_estimator.function(self.signal.energy_axis)
#        i = self.signal.energy2index(self.span_selector.range[1])
#        signal[:i] = 0.
#        s = Spectrum({'calibration' : {'data_cube' : signal}})
#        s.get_calibration_from(self.signal)
#        interactive_ns[self.background_substracted_spectrum_name] = s       
#        
#    def _define_signal_window_changed(self, old, new):
#        if new is True:
#            self.signal_span_selector = \
#            drawing.widgets.ModifiableSpanSelector(
#            self.signal.hse.signal_plot.ax, 
#            onselect = self.store_current_spectrum_bg_parameters,
#            onmove_callback = self.background_estimatorot_signal_map)
#            self.signal_span_selector.rect.set_color('blue')
#        elif self.signal_span_selector is not None:
#            self.signal_span_selector.turn_off()
#            self.signal_span_selector = None
#            
#    def plot_bg_removed_spectrum(self, *args, **kwards):
#        if self.span_selector.range is None: return
#        self.store_current_spectrum_bg_parameters()
#        ileft = self.signal.energy2index(self.span_selector.range[0])
#        iright = self.signal.energy2index(self.span_selector.range[1])
#        ea = self.signal.energy_axis[ileft:]
#        if self.bg_line is not None:
#            self.span_selector.ax.lines.remove(self.bg_line)
#            self.span_selector.ax.lines.remove(self.signal_line)
#        self.bg_line, = self.signal.hse.signal_plot.ax.plot(
#        ea, self.background_estimator.function(ea), color = 'black')
#        self.signal_line, = self.signal.hse.signal_plot.ax.plot(
#        self.signal.energy_axis[iright:], self.signal()[iright:] - 
#        self.background_estimator.function(self.signal.energy_axis[iright:]), color = 'black')
#        self.signal.hse.signal_plot.ax.figure.canvas.draw()

#        
#    def plot_signal_map(self, *args, **kwargs):
#        if self.define_signal_window is True and \
#        self.signal_span_selector.range is not None:
#            ileft = self.signal.energy2index(self.signal_span_selector.range[0])
#            iright = self.signal.energy2index(self.signal_span_selector.range[1])
#            signal_sp = self.signal.data_cube[ileft:iright,...].squeeze().copy()
#            if self.define_background_window is True:
#                pars = utils.two_area_powerlaw_estimation(
#                self.signal, *self.span_selector.range, only_current_spectrum = False)
#                x = self.signal.energy_axis[ileft:iright, np.newaxis, np.newaxis]
#                A = pars['A'][np.newaxis,...]
#                r = pars['r'][np.newaxis,...]
#                self.bg_sp = (A*x**(-r)).squeeze()
#                signal_sp -= self.bg_sp
#            self.signal_map = signal_sp.sum(0)
#            if self.map_ax is None:
#                f = plt.figure()
#                self.map_ax = f.add_subplot(111)
#                if len(self.signal_map.squeeze().shape) == 2:
#                    self.map = self.map_ax.imshow(self.signal_map.T, 
#                                                  interpolation = 'nearest')
#                else:
#                    self.map, = self.map_ax.plot(self.signal_map.squeeze())
#            if len(self.signal_map.squeeze().shape) == 2:
#                    self.map.set_data(self.signal_map.T)
#                    self.map.autoscale()
#                    
#            else:
#                self.map.set_ydata(self.signal_map.squeeze())
#            self.map_ax.figure.canvas.draw()
#            
#    def _extract_signal_fired(self):
#        if self.signal_map is None: return
#        if len(self.signal_map.squeeze().shape) == 2:
#            s = Image(
#            {'calibration' : {'data_cube' : self.signal_map.squeeze()}})
#            s.xscale = self.signal.xscale
#            s.yscale = self.signal.yscale
#            s.xunits = self.signal.xunits
#            s.yunits = self.signal.yunits
#            interactive_ns[self.signal_name] = s
#        else:
#            s = Spectrum(
#            {'calibration' : {'data_cube' : self.signal_map.squeeze()}})
#            s.energyscale = self.signal.xscale
#            s.energyunits = self.signal.xunits
#            interactive_ns[self.signal_name] = s
#    

#def energy_window_dependency(s, left, right, min_width = 10):
#    ins = s.energy2index(left)
#    ine = s.energy2index(right)
#    energies = s.energy_axis[ins:ine - min_width]
#    rs = []
#    As = []
#    for E in energies:
#        di = utils.two_area_powerlaw_estimation(s, E, ine)
#        rs.append(di['r'].mean())
#        As.append(di['A'].mean())
#    f = plt.figure()
#    ax1  = f.add_subplot(211)
#    ax1.plot(s.energy_axis[ins:ine - min_width], rs)
#    ax1.set_title('Rs')
#    ax1.set_xlabel('Energy')
#    ax2  = f.add_subplot(212, sharex = ax1)
#    ax2.plot(s.energy_axis[ins:ine - min_width], As)
#    ax2.set_title('As')
#    ax2.set_xlabel('Energy')
#    return rs, As
