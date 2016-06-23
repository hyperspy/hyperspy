Electron Holography
*******************

HyperSpy provides the user with two classes which can be used to process electron holography data:

* :py:class:`~._signals.hologram_image.HologramImage`
* :py:class:`~._signals.electron_wave_image.ElectronWaveImage`

The usage of both classes is explained in the following sections.



The ElectronWaveImage class
===========================

The :py:class:`~._signals.electron_wave_image.ElectronWaveImage` class can hold information about
the complex electron wave. It inherits from :py:class:`~._signals.signal2d.Signal2D` and
:py:class:`~._signals.complex_signal.ComplexSignal` and as such relevant properties like the
`amplitude`, `phase` and the `real` and `imag` part of the complex data can be directly accessed
and return appropriate :py:class:`~._signals.signal2d.Signal2D` signals.

To transform a :py:class:`~._signals.signal2d.Signa2D` (or subclass) into a
:py:class:`~._signals.electron_wave_image.ElectronWaveImage` use:

.. code-block:: python

    >>> im.set_signal_type('complex')


Add a linear phase ramp
-----------------------

A linear phase ramp can be added to the signal via the :py:func:`~._signals.signal2d.Signal2D.add_phase_ramp`
method. The parameters `ramp_x` and `ramp_y` dictate the slope of the ramp in `x`- and `y` direction,
while the offset is determined by the `offset` parameter. The fulcrum of the linear ramp is at the origin
and the slopes are given in units of the axis with the according scale taken into account.
Both are available via the :py:class:`~.axes.AxesManager` of the signal.
