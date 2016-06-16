Electron Holography
*******************

HyperSpy provides the user with two classes which can be used to process electron holography data:
 
* :py:class:`~._signals.hologram_image.HologramImage`
* :py:class:`~._signals.wave_image.WaveImage`

Both inherit directly from the :py:class:`~._signals.signal2d.Signal2D` class and thus can use all of its
functionality. The usage of both classes is explained in the following sections.



The WaveImage class
===================

The :py:class:`~._signals.wave_image.WaveImage` class can hold information about the complex electron
wave. As such, relevant properties like the `amplitude`, `phase` and the `real` and `imag` part can be
directly accessed and return appropriate :py:class:`~._signals.signal2d.Signal2D` signals.

To transform a :py:class:`~._signals.signal2d.Signa2D` (or subclass) into a 
:py:class:`~._signals.wave_image.WaveImage` use:

.. code-block:: python

    >>> im.set_signal_type('wave')


Unwrap the phase
----------------

With the :py:func:`~._signals.wave_image.WaveImage.get_unwrapped_phase` method the phase can be
unwrapped and returned as an :class:`~hyperspy._signals.signal2d.Signal2D`. The underlying method is
:py:func:`~skimage.restoration.unwrap`.


Add a linear ramp
-----------------

A linear ramp can be added to the wave via the :py:func:`~._signals.wave_image.WaveImage.add_phase_ramp`
method. The parameters `ramp_x` and `ramp_y` dictate the slope of the ramp in `x`- and `y` direction,
while the offset is determined by the `offset` parameter. The fulcrum of the linear ramp is at the origin
and the slopes are given in units of the axis with the according scale taken into account.
Both are available via the :py:class:`~.axes.AxesManager` of the signal.
