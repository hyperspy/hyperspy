import matplotlib.pyplot as plt

import numpy as np
import enthought.traits.api as t
import enthought.traits.ui.api as tu

from eelslab import components
from eelslab import utils
from eelslab import drawing
from eelslab.spectrum import Spectrum
from eelslab.image import Image
from eelslab.interactive_ns import interactive_ns


class EgertonPanel(t.HasTraits):
    define_background_window = t.Bool(False)
    bg_window_size_variation = t.Button()
    background_substracted_spectrum_name = t.Str('signal')
    extract_background = t.Button()    
    define_signal_window = t.Bool(False)
    signal_window_size_variation = t.Button()
    signal_name = t.Str('signal')
    extract_signal = t.Button()
    view = tu.View(tu.Group(
        tu.Group('define_background_window',
                 tu.Item('bg_window_size_variation', 
                         label = 'window size effect', show_label=False),
                 tu.Item('background_substracted_spectrum_name'),
                 tu.Item('extract_background', show_label=False),
                 ),
        tu.Group('define_signal_window',
                 tu.Item('signal_window_size_variation', 
                         label = 'window size effect', show_label=False),
                 tu.Item('signal_name', show_label=True),
                 tu.Item('extract_signal', show_label=False)),))
                 
    def __init__(self, SI):
        
        self.SI = SI
        
        # Background
        self.bg_span_selector = None
        self.pl = components.PowerLaw()
        self.bg_line = None
        self.bg_cube = None
                
        # Signal
        self.signal_span_selector = None
        self.signal_line = None
        self.signal_map = None
        self.map_ax = None
    
    def store_current_spectrum_bg_parameters(self, *args, **kwards):
        if self.define_background_window is False or \
        self.bg_span_selector.range is None: return
        pars = utils.two_area_powerlaw_estimation(
        self.SI, *self.bg_span_selector.range,only_current_spectrum = True)
        self.pl.r.value = pars['r']
        self.pl.A.value = pars['A']
                     
        if self.define_signal_window is True and \
        self.signal_span_selector.range is not None:
            self.plot_signal_map()
                     
    def _define_background_window_changed(self, old, new):
        if new is True:
            self.bg_span_selector = \
            drawing.widgets.ModifiableSpanSelector(
            self.SI.hse.spectrum_plot.left_ax,
            onselect = self.store_current_spectrum_bg_parameters,
            onmove_callback = self.plot_bg_removed_spectrum)
        elif self.bg_span_selector is not None:
            if self.bg_line is not None:
                self.bg_span_selector.ax.lines.remove(self.bg_line)
                self.bg_line = None
            if self.signal_line is not None:
                self.bg_span_selector.ax.lines.remove(self.signal_line)
                self.signal_line = None
            self.bg_span_selector.turn_off()
            self.bg_span_selector = None
                      
    def _bg_window_size_variation_fired(self):
        if self.define_background_window is False: return
        left = self.bg_span_selector.rect.get_x()
        right = left + self.bg_span_selector.rect.get_width()
        energy_window_dependency(self.SI, left, right, min_width = 10)
        
    def _extract_background_fired(self):
        if self.pl is None: return
        signal = self.SI() - self.pl.function(self.SI.energy_axis)
        i = self.SI.energy2index(self.bg_span_selector.range[1])
        signal[:i] = 0.
        s = Spectrum({'calibration' : {'data_cube' : signal}})
        s.get_calibration_from(self.SI)
        interactive_ns[self.background_substracted_spectrum_name] = s       
        
    def _define_signal_window_changed(self, old, new):
        if new is True:
            self.signal_span_selector = \
            drawing.widgets.ModifiableSpanSelector(
            self.SI.hse.spectrum_plot.left_ax, 
            onselect = self.store_current_spectrum_bg_parameters,
            onmove_callback = self.plot_signal_map)
            self.signal_span_selector.rect.set_color('blue')
        elif self.signal_span_selector is not None:
            self.signal_span_selector.turn_off()
            self.signal_span_selector = None
            
    def plot_bg_removed_spectrum(self, *args, **kwards):
        if self.bg_span_selector.range is None: return
        self.store_current_spectrum_bg_parameters()
        ileft = self.SI.energy2index(self.bg_span_selector.range[0])
        iright = self.SI.energy2index(self.bg_span_selector.range[1])
        ea = self.SI.energy_axis[ileft:]
        if self.bg_line is not None:
            self.bg_span_selector.ax.lines.remove(self.bg_line)
            self.bg_span_selector.ax.lines.remove(self.signal_line)
        self.bg_line, = self.SI.hse.spectrum_plot.left_ax.plot(
        ea, self.pl.function(ea), color = 'black')
        self.signal_line, = self.SI.hse.spectrum_plot.left_ax.plot(
        self.SI.energy_axis[iright:], self.SI()[iright:] - 
        self.pl.function(self.SI.energy_axis[iright:]), color = 'black')
        self.SI.hse.spectrum_plot.left_ax.figure.canvas.draw()

        
    def plot_signal_map(self, *args, **kwargs):
        if self.define_signal_window is True and \
        self.signal_span_selector.range is not None:
            ileft = self.SI.energy2index(self.signal_span_selector.range[0])
            iright = self.SI.energy2index(self.signal_span_selector.range[1])
            signal_sp = self.SI.data_cube[ileft:iright,...].squeeze().copy()
            if self.define_background_window is True:
                pars = utils.two_area_powerlaw_estimation(
                self.SI, *self.bg_span_selector.range, only_current_spectrum = False)
                x = self.SI.energy_axis[ileft:iright, np.newaxis, np.newaxis]
                A = pars['A'][np.newaxis,...]
                r = pars['r'][np.newaxis,...]
                self.bg_sp = (A*x**(-r)).squeeze()
                signal_sp -= self.bg_sp
            self.signal_map = signal_sp.sum(0)
            if self.map_ax is None:
                f = plt.figure()
                self.map_ax = f.add_subplot(111)
                if len(self.signal_map.squeeze().shape) == 2:
                    self.map = self.map_ax.imshow(self.signal_map.T, 
                                                  interpolation = 'nearest')
                else:
                    self.map, = self.map_ax.plot(self.signal_map.squeeze())
            if len(self.signal_map.squeeze().shape) == 2:
                    self.map.set_data(self.signal_map.T)
                    self.map.autoscale()
                    
            else:
                self.map.set_ydata(self.signal_map.squeeze())
            self.map_ax.figure.canvas.draw()
            
    def _extract_signal_fired(self):
        if self.signal_map is None: return
        if len(self.signal_map.squeeze().shape) == 2:
            s = Image(
            {'calibration' : {'data_cube' : self.signal_map.squeeze()}})
            s.xscale = self.SI.xscale
            s.yscale = self.SI.yscale
            s.xunits = self.SI.xunits
            s.yunits = self.SI.yunits
            interactive_ns[self.signal_name] = s
        else:
            s = Spectrum(
            {'calibration' : {'data_cube' : self.signal_map.squeeze()}})
            s.energyscale = self.SI.xscale
            s.energyunits = self.SI.xunits
            interactive_ns[self.signal_name] = s
    

def energy_window_dependency(s, left, right, min_width = 10):
    ins = s.energy2index(left)
    ine = s.energy2index(right)
    energies = s.energy_axis[ins:ine - min_width]
    rs = []
    As = []
    for E in energies:
        di = utils.two_area_powerlaw_estimation(s, E, ine)
        rs.append(di['r'].mean())
        As.append(di['A'].mean())
    f = plt.figure()
    ax1  = f.add_subplot(211)
    ax1.plot(s.energy_axis[ins:ine - min_width], rs)
    ax1.set_title('Rs')
    ax1.set_xlabel('Energy')
    ax2  = f.add_subplot(212, sharex = ax1)
    ax2.plot(s.energy_axis[ins:ine - min_width], As)
    ax2.set_title('As')
    ax2.set_xlabel('Energy')
    return rs, As

plt.show()
