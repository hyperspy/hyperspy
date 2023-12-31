.. _signal.binned:

Binned and unbinned signals
---------------------------

Signals that are a histogram of a probability density function (pdf) should
have the ``is_binned`` attribute of the signal axis set to ``True``. The reason
is that some methods operate differently on signals that are *binned*. An
example of *binned* signals are EDS spectra, where the multichannel analyzer
integrates the signal counts in every channel (=bin).
Note that for 2D signals each signal axis has an ``is_binned``
attribute that can be set independently. For example, for the first signal
axis: ``signal.axes_manager.signal_axes[0].is_binned``.

The default value of the ``is_binned`` attribute is shown in the
following table:

.. table:: Binned default values for the different subclasses.


    +----------------------------------------+--------+----------+
    |     BaseSignal subclass                | binned | Library  |
    +========================================+========+==========+
    | :class:`~.api.signals.BaseSignal`      | False  | hyperspy |
    +----------------------------------------+--------+----------+
    | :class:`~.api.signals.Signal1D`        | False  | hyperspy |
    +----------------------------------------+--------+----------+
    | :class:`exspy.signals.EELSSpectrum`    | True   |  exSpy   |
    +----------------------------------------+--------+----------+
    | :class:`exspy.signals.EDSSEMSpectrum`  | True   |  exSpy   |
    +----------------------------------------+--------+----------+
    | :class:`exspy.signals.EDSTEMSpectrum`  | True   |  exSpy   |
    +----------------------------------------+--------+----------+
    | :class:`~.api.signals.Signal2D`        | False  | hyperspy |
    +----------------------------------------+--------+----------+
    | :class:`~.api.signals.ComplexSignal`   | False  | hyperspy |
    +----------------------------------------+--------+----------+
    | :class:`~.api.signals.ComplexSignal1D` | False  | hyperspy |
    +----------------------------------------+--------+----------+
    | :class:`~.api.signals.ComplexSignal2D` | False  | hyperspy |
    +----------------------------------------+--------+----------+



To change the default value:

.. code-block:: python

    >>> s.axes_manager[-1].is_binned = True # doctest: +SKIP

.. versionchanged:: 1.7 The ``binned`` attribute from the metadata has been
    replaced by the axis attributes ``is_binned``.

Integration of binned signals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For binned axes, the detector already provides the per-channel integration of
the signal. Therefore, in this case, :meth:`~.api.signals.BaseSignal.integrate1D`
performs a simple summation along the given axis. In contrast, for unbinned
axes, :meth:`~.api.signals.BaseSignal.integrate1D` calls the
:meth:`~.api.signals.BaseSignal.integrate_simpson` method.
