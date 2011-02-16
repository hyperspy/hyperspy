# -*- coding: utf-8 -*-
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
from matplotlib.widgets import SpanSelector, MultiCursor, Cursor

import numpy as np
import enthought.traits.api as t
import enthought.traits.ui.api as tu

from .. import file_io
from .. import components
from .. import utils
from .. import drawing
from ..spectrum import Spectrum
from ..interactive_ns import interactive_ns



class SavitzkyGolay(t.HasTraits):
    input_signal_name = t.Str
    polynomial_order = t.Range(1,10,3)
    number_of_points = t.Int(5)
    differential_order = t.Int(0)
    signal_name = t.Str('signal')
    extract_signal = t.Button()    
    view = tu.View(
        tu.Group(
            'input_signal_name',
            'polynomial_order',
            'number_of_points',
            'differential_order',
            'signal_name',
            tu.Item('extract_signal', show_label=False),),)
            
    def _polynomial_order_changed(self, old, new):
        self.smooth_line.update()
        
    def _number_of_points_changed(self, old, new):
        self.smooth_line.update()
        
    def _differential_order_changed(self, old, new):
        self.smooth_line.update()
            
    def init(self):
        self.ax = None
        self.data_line = None
        self.smooth_line = None
        self.signal = None
        
    def _input_signal_name_changed(self, old, new):
        if interactive_ns.has_key(new):
            self.signal = interactive_ns[new]
            self.plot()
            
    def model2plot(self, coordinates = None):
        smoothed = utils.sg(self.signal(), self.number_of_points, 
                            self.polynomial_order, self.differential_order)
        return smoothed
        
            
    def plot(self):
        self.signal.plot()
        hse = self.signal.hse
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
        
    def _extract_signal_fired(self):
        s = Spectrum({'calibration' : {'data_cube' : self.model2plot()}})
        s.get_calibration_from(self.signal)
        interactive_ns[self.signal_name] = s
        
class Lowess(t.HasTraits):
    input_signal_name = t.Str
    smoothing_parameter = t.Float(2/3.)
    number_of_iterations = t.Int(3)
    signal_name = t.Str('signal')
    extract_signal = t.Button()    
    view = tu.View(
        tu.Group(
            'input_signal_name',
            'smoothing_parameter',
            'number_of_iterations',
            'signal_name',
            tu.Item('extract_signal', show_label=False),),)
            
    def _smoothing_parameter_changed(self, old, new):
        self.smooth_line.update()
        
    def _number_of_iterations_changed(self, old, new):
        self.smooth_line.update()
        
    def init(self):
        self.ax = None
        self.data_line = None
        self.smooth_line = None
        self.signal = None
        
    def _input_signal_name_changed(self, old, new):
        if interactive_ns.has_key(new):
            self.signal = interactive_ns[new]
            self.plot()
            
    def model2plot(self, coordinates = None):
        smoothed = utils.lowess(self.signal.energy_axis, self.signal(), 
                                self.smoothing_parameter, 
                                self.number_of_iterations)
        return smoothed
        
    def _extract_signal_fired(self):
        s = Spectrum({'calibration' : {'data_cube' : self.model2plot()}})
        s.get_calibration_from(self.signal)
        interactive_ns[self.signal_name] = s
        
            
    def plot(self):
        self.signal.plot()
        hse = self.signal.hse
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
