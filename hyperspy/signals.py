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
    Spectrum
        Deprecated in favour of Signal1D from version 1.0.0
    Image
        Deprecated in favour of Signal2D from version 1.0.0
    Simulation
        For generic simulated data with arbitrary signal_dimension. All other
        simulation signal classes inherit from this one. It should only be used
        with none of the others is appropriated.
    EELSSpectrum
        For electron energy-loss data with signal_dimension equal 1, i.e.
        spectral data of n-dimensions. The signal is binned by default.
    EELSSpectrumSimulation, SpectrumSimulation, ImageSimulation
        Simulation versions of EELSSpectrum, Spectrum and Image.
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

"""

# -*- coding: utf-8 -*-

from hyperspy._signals.signal1D import Signal1D, Signal1DTools
from hyperspy._signals.signal2D import Signal2D, Signal2DTools
from hyperspy.misc.hspy_warnings import VisibleDeprecationWarning


class Spectrum(Signal1D,
               Signal2DTools,):
    VisibleDeprecationWarning('The Spectrum class will be deprecated from\
                              version 1.0.0 and replaced with Signal1D')


class Image(Signal2D,
            Signal1DTools,):
    VisibleDeprecationWarning('The Image class will be deprecated from\
                              version 1.0.0 and replaced with Signal2D')


from hyperspy._signals.eels import EELSSpectrum
from hyperspy._signals.eds_sem import EDSSEMSpectrum
from hyperspy._signals.eds_tem import EDSTEMSpectrum
from hyperspy._signals.dielectric_function import DielectricFunction
from hyperspy._signals.simulation import Simulation
from hyperspy._signals.image_simulation import ImageSimulation
from hyperspy._signals.spectrum_simulation import SpectrumSimulation
from hyperspy._signals.eels_spectrum_simulation import (
    EELSSpectrumSimulation)
from hyperspy.signal import BaseSignal


class Signal(BaseSignal,
             Signal1DTools,
             Signal2DTools,):
    VisibleDeprecationWarning('The Signal class will be deprecated from\
                              version 1.0.0 and replaced with BaseSignal')
