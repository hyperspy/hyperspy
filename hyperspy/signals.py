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
import warnings
from hyperspy._signals.signal1d import Signal1D, Signal1DTools
from hyperspy._signals.signal2d import Signal2D, Signal2DTools
from hyperspy.misc.hspy_warnings import VisibleDeprecationWarning


class Spectrum(Signal1D,
               Signal2DTools,):

    def __init__(self, *args, **kwargs):
        warnings.warn("The Spectrum class will be deprecated from version 1.0.0"
                      " and replaced with Signal1D",
                      VisibleDeprecationWarning)
        Signal1D.__init__(self, *args, **kwargs)

    def to_image(self):
        """Returns the spectrum as an image.

        See Also
        --------
        as_image : a method for the same purpose with more options.
        signals.Image.to_spectrum : performs the inverse operation on images.

        Raises
        ------
        DataDimensionError: when data.ndim < 2

        """
        warnings.warn("The to_image method will be deprecated from version"
                      " 1.0.0 and replaced with to_signal2D",
                      VisibleDeprecationWarning)
        if self.data.ndim < 2:
            raise DataDimensionError(
                "A Signal dimension must be >= 2 to be converted to Signal2D")
        im = self.rollaxis(-1 + 3j, 0 + 3j)
        im.metadata.Signal.record_by = "image"
        im._assign_subclass()
        return im


class Image(Signal2D,
            Signal1DTools,):

    def __init__(self, *args, **kwargs):
        warnings.warn("The Image class will be deprecated from version 1.0.0"
                      " and replaced with Signal2D",
                      VisibleDeprecationWarning)
        Signal2D.__init__(self, *args, **kwargs)

    def to_spectrum(self):
        """Returns the image as a spectrum.

        See Also
        --------
        as_spectrum : a method for the same purpose with more options.
        signals.Spectrum.to_image : performs the inverse operation on spectra.

        Raises
        ------
        DataDimensionError: when data.ndim < 2

        """
        warnings.warn("The to_spectrum method will be deprecated from version"
                      " 1.0.0 and replaced with to_signal1D",
                      VisibleDeprecationWarning)
        s = self.to_signal1D()
        return s


from hyperspy._signals.eels import EELSSpectrum
from hyperspy._signals.eds_sem import EDSSEMSpectrum
from hyperspy._signals.eds_tem import EDSTEMSpectrum
from hyperspy._signals.dielectric_function import DielectricFunction
from hyperspy._signals.simulation import Simulation
from hyperspy._signals.image_simulation import ImageSimulation
from hyperspy._signals.spectrum_simulation import SpectrumSimulation
from hyperspy._signals.eels_spectrum_simulation import (
    EELSSpectrumSimulation)
from hyperspy.exceptions import DataDimensionError

from hyperspy.signal import BaseSignal


class Signal(BaseSignal,
             Signal1DTools,
             Signal2DTools,):

    def __init__(self, *args, **kwargs):
        warnings.warn("The Signal class will be deprecated from version 1.0.0"
                      " and replaced with BaseSignal",
                      VisibleDeprecationWarning)
        BaseSignal.__init__(self, *args, **kwargs)
