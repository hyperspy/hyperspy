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

To transform a :py:class:`~._signals.signal2d.Signal2D` (or subclass) into a
:py:class:`~._signals.wave_image.WaveImage` use:

.. code-block:: python

    >>> im.set_signal_type('wave')
