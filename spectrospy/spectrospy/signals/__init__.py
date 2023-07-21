"""Signals to be operated on. The basic unit of data"""

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