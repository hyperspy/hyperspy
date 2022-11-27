Other Signal Classes
--------------------

API of signal classes, which are not part of the :mod:`hyperspy.api` namespace but are inherited in
HyperSpy signals classes. These classes are not expected to be instantiated by users but their methods, 
which are used by other classes, are documented here.

Common Signals
""""""""""""""

.. autosummary::
   :nosignatures:

   hyperspy._signals.common_signal1d.CommonSignal1D
   hyperspy._signals.common_signal2d.CommonSignal2D   
   hyperspy._signals.eds.EDSSpectrum

Lazy Signals
""""""""""""

.. autosummary::
   :nosignatures:

   hyperspy._lazy_signals.LazyComplexSignal
   hyperspy._lazy_signals.LazyComplexSignal1D
   hyperspy._lazy_signals.LazyComplexSignal2D
   hyperspy._lazy_signals.LazyDielectricFunction
   hyperspy._signals.eds.LazyEDSSpectrum
   hyperspy._lazy_signals.LazyEDSSEMSpectrum
   hyperspy._lazy_signals.LazyEDSTEMSpectrum
   hyperspy._lazy_signals.LazyEELSSpectrum
   hyperspy._lazy_signals.LazyHologramImage
   hyperspy._signals.lazy.LazySignal
   hyperspy._lazy_signals.LazySignal1D
   hyperspy._lazy_signals.LazySignal2D

.. toctree::
   :maxdepth: 2
   :hidden:

   CommonSignal1D
   CommonSignal2D
   EDSSpectrum
   LazyComplexSignal
   LazyComplexSignal1D
   LazyComplexSignal2D
   LazyDielectricFunction
   LazyEDSSpectrum
   LazyEDSSEMSpectrum
   LazyEDSTEMSpectrum
   LazyEELSSpectrum
   LazyHologramImage
   LazySignal
   LazySignal1D
   LazySignal2D
