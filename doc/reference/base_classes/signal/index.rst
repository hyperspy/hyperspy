Signal
------

API of signal classes, which are not part of the :mod:`hyperspy.api` namespace but are inherited in
HyperSpy signals classes. These classes are not expected to be instantiated by users but their methods, 
which are used by other classes, are documented here.

Common Signals
""""""""""""""
.. currentmodule:: hyperspy._signals.common_signal1d

.. autosummary::
   :nosignatures:

   CommonSignal1D

.. currentmodule:: hyperspy._signals.common_signal2d

.. autosummary::
   :nosignatures:

   CommonSignal2D


Lazy Signals
""""""""""""
.. currentmodule:: hyperspy._signals.lazy

.. autosummary::
   :nosignatures:

   LazySignal

.. currentmodule:: hyperspy._lazy_signals

.. autosummary::
   :nosignatures:

   LazyComplexSignal
   LazyComplexSignal1D
   LazyComplexSignal2D
   LazySignal1D
   LazySignal2D

.. toctree::
   :maxdepth: 2
   :hidden:

   CommonSignal1D
   CommonSignal2D
   LazyComplexSignal
   LazyComplexSignal1D
   LazyComplexSignal2D
   LazySignal
   LazySignal1D
   LazySignal2D
