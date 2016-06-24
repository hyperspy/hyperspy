"""
The Signal class and its specilized subclasses:

    Signal
        For generic data with arbitrary signal_dimension. All other signal
        classes inherit from this one. It should only be used with none of
        the others is appropriated.
    Signal1D
        For generic data with signal_dimension equal 1, i.e. spectral data of
        n-dimensions. The signal is unbinned by default.
    Signal2D
        For generic data with signal_dimension equal 2, i.e. image data of
        n-dimensions. The signal is unbinned by default.
    Signal1D
        Deprecated in favour of Signal1D from version 1.0.0
    Signal2D
        Deprecated in favour of Signal2D from version 1.0.0
    Simulation
        For generic simulated data with arbitrary signal_dimension. All other
        simulation signal classes inherit from this one. It should only be used
        with none of the others is appropriated.
    EELSSpectrum
        For electron energy-loss data with signal_dimension equal 1, i.e.
        spectral data of n-dimensions. The signal is binned by default.
    EELSSpectrumSimulation, SpectrumSimulation, ImageSimulation
        Simulation versions of EELSSpectrum, Signal1D and Signal2D.
    EDSTEMSpectrum
        For electron energy-dispersive X-rays data acquired in a transmission
        electron microscopy with signal_dimension equal 1, i.e.
        spectral data of n-dimensions. The signal is binned by default.
    EDSSEMSpectrum
        For electron energy-dispersive X-rays data acquired in a scanning
        electron microscopy with signal_dimension equal 1, i.e.
        spectral data of n-dimensions. The signal is binned by default.
    DielectricFunction
        For dielectric function data with signal_dimension equal 1. The signal
        is unbinned by default.
    WaveImage
        For complex 2D-images. Phase, amplitude, real and imaginary parts are
        easily accessible and can be manipulated.

"""

# -*- coding: utf-8 -*-
from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D
from hyperspy._signals.complex_signal import ComplexSignal, ComplexSignal1D, ComplexSignal2D
from hyperspy._signals.eels import EELSSpectrum
from hyperspy._signals.eds_sem import EDSSEMSpectrum
from hyperspy._signals.eds_tem import EDSTEMSpectrum
from hyperspy._signals.dielectric_function import DielectricFunction
from hyperspy._signals.simulation import Simulation
from hyperspy._signals.image_simulation import ImageSimulation
from hyperspy._signals.spectrum_simulation import SpectrumSimulation
from hyperspy._signals.eels_spectrum_simulation import (
    EELSSpectrumSimulation)
from hyperspy._signals.electron_wave_image import ElectronWaveImage
from hyperspy.signal import BaseSignal
