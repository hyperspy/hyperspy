Electron Holography
*******************

HyperSpy provides the user with two classes which can be used to process electron holography data:
 
* :py:class:`~._signals.hologram_image.HologramImage`
* :py:class:`~._signals.wave_image.WaveImage`

Both inherit directly from the :py:class:`~._signals.signal1d.SignalD` class and thus can use all of
its functionality. The usage of both classes is explained in the following sections.


The HologramImage class
=======================

The :py:class:`~._signals.hologram_image.HologramImage` class is designed to hold images acquired via
electron holography. The complex electron wave can be reconstructed from the signal via the
:py:func:`~._signals.hologram_image.HologramImage.reconstruct_wave_image` method which will return
a :py:class:`~._signals.wave_image.WaveImage` class, which can then be further analysed.

To transform a :py:class:`~._signals.image.Image` (or subclass) into a 
:py:class:`~._signals.hologram_image.HologramImage` use:

.. code-block:: python

    >>> im.set_signal_type('hologram')
	