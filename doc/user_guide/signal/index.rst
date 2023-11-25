.. _signal-label:

The Signal class
****************

.. WARNING::
    This subsection can be a bit confusing for beginners.
    Do not worry if you do not understand it all.


HyperSpy stores the data in the :class:`~.api.signals.BaseSignal` class, that is
the object that you get when e.g. you load a single file using
:func:`~.api.load`. Most of the data analysis functions are also contained in
this class or its specialized subclasses. The :class:`~.api.signals.BaseSignal`
class contains general functionality that is available to all the subclasses.
The subclasses provide functionality that is normally specific to a particular
type of data, e.g. the :class:`~.api.signals.Signal1D` class provides
common functionality to deal with one-dimensional (e.g. spectral) data and
:class:`exspy.signals.EELSSpectrum` (which is a subclass of
:class:`~.api.signals.Signal1D`) adds extra functionality to the
:class:`~.api.signals.Signal1D` class for electron energy-loss
spectroscopy data analysis.

A signal store other objects in what are called attributes. For
examples, the data is stored in a numpy array in the
:attr:`~.api.signals.BaseSignal.data` attribute, the original parameters in the
:attr:`~.api.signals.BaseSignal.original_metadata` attribute, the mapped parameters
in the :attr:`~.signals.BaseSignal.metadata` attribute and the axes
information (including calibration) can be accessed (and modified) in the
:class:`~.axes.AxesManager` attribute.


.. toctree::
    :maxdepth: 2

    signal_basics.rst
    ragged.rst
    binned_signals.rst
    indexing.rst
    generic_tools.rst
    basic_statistical_analysis.rst
    setting_noise_properties.rst
    speeding_up_operation.rst
    complex_datatype.rst
    gpu.rst
