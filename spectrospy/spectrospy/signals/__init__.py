
""" Signals to be operated on. The basic unit of data"
    EELSSpectrum
        For electron energy-loss data with signal_dimension equal 1, i.e.
        spectral data of n-dimensions. The signal is binned by default.
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
from .dielectric_function import DielectricFunction, LazyDielectricFunction
from .eds import EDSSpectrum, LazyEDSSpectrum
from .eds_sem import EDSSEMSpectrum, LazyEDSSEMSpectrum
from .eds_tem import EDSTEMSpectrum, LazyEDSTEMSpectrum
from .eels import EELSSpectrum, LazyEELSSpectrum
__all__ = ["DielectricFunction",
            "LazyDielectricFunction",
            "EDSSpectrum",
            "LazyEDSSpectrum",
            "EDSTEMSpectrum",
            "LazyEDSTEMSpectrum",
            "EELSSpectrum",
            "LazyEELSSpectrum",
            "EDSSEMSpectrum",
            "LazyEDSSEMSpectrum",
            ]