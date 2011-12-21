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

import matplotlib.pyplot as plt

import numpy as np
import traits.api as t
import traitsui.api as tu
from traitsui.menu import OKButton, ApplyButton, CancelButton

from hyperspy import components
from hyperspy.component import Component
from hyperspy.misc import utils
from hyperspy import drawing
from hyperspy.misc.interactive_ns import interactive_ns
from hyperspy.gui.tools import (SpanSelectorInSpectrum, 
    SpanSelectorInSpectrumHandler)
from hyperspy.misc.progressbar import progressbar


class BackgroundRemoval(SpanSelectorInSpectrum):
    background_type = t.Enum('Power Law', 'Gaussian', 'Offset',
    'Polynomial', default = 'Power Law')
    polynomial_order = t.Range(1,10)
    background_estimator = t.Instance(Component)
    bg_line_range = t.Enum('from_left_range', 'full', 'ss_range', 
        default = 'full')
    hi = t.Int(0)
    view = tu.View(
        tu.Group(
            'background_type',
            tu.Group(
                'polynomial_order', 
                visible_when = 'background_type == \'Polynomial\''),),
            buttons= [OKButton, CancelButton],
            handler = SpanSelectorInSpectrumHandler)
                 
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
        self.signal._plot.spectrum_plot.add_line(self.bg_line)
        self.bg_line.autoscale = False
        self.bg_line.plot()
        
    def bg_to_plot(self, axes_manager = None, fill_with = np.nan):
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
        if self.signal.axes_manager.navigation_dimension != 0:
            pbar = progressbar(
            maxval = (np.cumprod(self.signal.axes_manager.navigation_shape)[-1]))
            i = 0
            self.bg_line_range = 'full'
            indexes = np.ndindex(
            tuple(self.signal.axes_manager.navigation_shape))
            for index in indexes:
                self.signal.axes_manager.set_not_slicing_indexes(index)
                self.signal.data[
                self.signal.axes_manager._getitem_tuple] -= \
                np.nan_to_num(self.bg_to_plot(self.signal.axes_manager, 0))
                i+=1
                pbar.update(i)
            pbar.finish()
        else:
            self.signal.data[self.signal.axes_manager._getitem_tuple] -= \
                np.nan_to_num(self.bg_to_plot(self.signal.axes_manager, 0))
            
        self.signal._replot()
        self.signal._plot.auto_update_plot = True
    

       
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
#            self.signal.hse.spectrum_plot.ax,
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
#            self.signal.hse.spectrum_plot.ax, 
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
#        self.bg_line, = self.signal.hse.spectrum_plot.ax.plot(
#        ea, self.background_estimator.function(ea), color = 'black')
#        self.signal_line, = self.signal.hse.spectrum_plot.ax.plot(
#        self.signal.energy_axis[iright:], self.signal()[iright:] - 
#        self.background_estimator.function(self.signal.energy_axis[iright:]), color = 'black')
#        self.signal.hse.spectrum_plot.ax.figure.canvas.draw()

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
