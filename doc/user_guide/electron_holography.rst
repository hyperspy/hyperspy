Electron Holography
*******************

HyperSpy provides the user with two classes which can be used to process electron holography data:
 
* :py:class:`~._signals.hologram_image.HologramImage`
* :py:class:`~._signals.wave_image.WaveImage`

Both inherit directly from the :py:class:`~._signals.image.Image` class and thus can use all of its
functionality. The usage of both classes is explained in the following sections.


The HologramImage class
=======================

The :py:class:`~._signals.hologram_image.HologramImage` class is designed to hold images acquired via
electron holography. The complex electron wave can be reconstructed from the signal via  the
:py:func:`~._signals.hologram_image.HologramImage.reconstruct_wave_image` method which will return
a :py:class:`~._signals.wave_image.WaveImage` class, which can then be further analysed.

To transform a :py:class:`~._signals.image.Image` (or subclass) into a 
:py:class:`~._signals.hologram_image.HologramImage` use:

.. code-block:: python

    >>> im.set_signal_type('HOLOGRAM')


The WaveImage class
===================

The :py:class:`~._signals.wave_image.WaveImage` class can hold information about the complex electron
wave. As such, relevant properties like the `amplitude`, `phase` and the `real` and `imag` part can be
directly accessed and return appropriate :py:class:`~._signals.image.Image` signals.


Unwrap the phase
----------------

With the :py:func:`~._signals.wave_image.WaveImage.get_unwrapped_phase` method the phase can be
unwrapped and returned as an :class:`~hyperspy._signals.Image`. The underlying method is
:py:func:`~skimage.restoration.unwrap`.


Normalize the wave
------------------

The :py:func:`~._signals.wave_image.WaveImage.normalize` method can be used to normalize the wave.
If the normalization factor is a complex number, the amplitude will be divided and the phase shifted
according to:

.. math::

    wave_normalized = \frac{A}{A_0}\cdot\exp{\phi-\phi_0}
   
If the input is an array instead, the mean value will be calculated before normalization.


Subtract a reference wave
-------------------------
With the :py:func:`~._signals.wave_image.WaveImage.subtract_reference` method a reference wave can
be subtracted element wise. The reference wave also has to be a :py:class:`~_signals.wave_image.WaveImage`.


Add a linear ramp
-----------------

A linear ramp can be added to the wave via the :py:func:`~._signals.wave_image.WaveImage.add_phase_ramp`
method. The parameters `ramp_x` and `ramp_y` dictate the slope of the ramp in `x`- and `y` direction,
while the offset is determined by the `offset` parameter. The fulcrum of the linear ramp is at the origin
and the slopes are given in units of the axis with the according scale taken into account.
Both are available via the :py:class:`~.axes.AxesManager` of the signal.
