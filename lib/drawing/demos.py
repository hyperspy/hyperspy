#!/usr/bin/env python

from eelslab.signal import Signal
import numpy as np

def three_d_image():
    s = Signal({'data' : np.random.random((16,32,32))})
    s.axes_manager.axes[1].slice_bool = True
    s.plot()

def four_d_image():
    s = Signal({'data' : np.random.random((16,16,32,32))})
    s.axes_manager.axes[2].slice_bool = True
    s.plot()

def line_spectrum():
    s = Signal({'data' : np.random.random((100,1024))})
    s.plot()

def spectrum_image():
    s = Signal({'data' : np.random.random((64,64,1024))})
    s.plot()


