Electron Holography
*******************

HyperSpy provides the user with two classes which can be used to process electron holography data:
 
* :py:class:`~._signals.hologram_image.HologramImage`
* :py:class:`~._signals.wave_image.ComplexSignal2D`

Both inherit directly from the :py:class:`~._signals.signal2d.Signal2D` class and thus can use all of
its functionality. The usage of both classes is explained in the following sections.


The HologramImage class
=======================

The :py:class:`~._signals.hologram_image.HologramImage` class is designed to hold images acquired via
electron holography.

To transform a :py:class:`~._signals.signal2d.Signa2D` (or subclass) into a
:py:class:`~._signals.hologram_image.HologramImage` use:

.. code-block:: python

    >>> im.set_signal_type('hologram')


Reconstruction of holograms
---------------------------
The detailed description of electron holography and reconstruction of holograms can be found in literature
`[Gabor1948] <http://www.nature.com/doifinder/10.1038/161777a0>`_,
`[Tonomura1999] <http://www.springer.com/us/book/9783540645559>`_,
`[McCartney2007] <http://dx.doi.org/10.1146/annurev.matsci.37.052506.084219>`_,
`[Joy1993] <http://dx.doi.org/10.1016/0304-3991(93)90130-P>`_. Fourier based reconstruction of off-axis holograms
(includes finding a side band in FFT, isolating and filtering it, recenter and calculate inverse Fourier transform)
can be performed using :py:func:`~._signals.hologram_image.HologramImage.reconstruct_phase` method
which returns a :py:class:`~._signals.wave_image.ComplexSignal2D` class, containing the reconstructed electron wave:

.. code-block:: python

    >>> wave_image = im.reconstruct_phase()

To reconstruct the hologram with a reference wave it should be provided to the method either as Hyperspy's
:py:class:`~._signals.signal2d.Signa2D` (or subclass) or as a nparray:

.. code-block:: python

    >>> wave_image = im.reconstruct_phase(ref_image)

The reconstruction parameters can be extracted from metadata of wave_image as follows:

.. code-block:: python

    >>> reconstruction_param = wave_image.rec_param

This option can be used for batch processing as the reconstruction_param can be provided to the reconstruction method as follows:

.. code-block:: python

    >>> wave_image1 = im1.reconstruct_phase(rec_param=reconstruction_param)

Further analyses of the reconstructed wave (phase) can be done using :py:class:`~._signals.wave_image.ComplexSignal2D` class
functionality (see bellow).


The ComplexSignal and ComplexSignal2D classes
=============================================

The :py:class:`~._signals.wave_image.ComplexSignal2D` class can hold information about the complex electron
wave. As such, relevant properties like the `amplitude`, `phase` and the `real` and `imag` part can be
directly accessed and return appropriate :py:class:`~._signals.signal2d.Signal2D` signals.

To transform a :py:class:`~._signals.signal2d.Signa2D` (or subclass) into a 
:py:class:`~._signals.wave_image.ComplexSignal2D` use:

.. code-block:: python

    >>> im.set_signal_type('ComplexSignal2D')


Unwrap the phase
----------------

With the :py:func:`~._signals.wave_image.ComplexSignal2D.unwrapped_phase` method the phase can be
unwrapped and returned as an :class:`~hyperspy._signals.signal2d.Signal2D`. The underlying method is
:py:func:`~skimage.restoration.unwrap`.


Add a linear ramp
-----------------

A linear ramp can be added to the wave via the :py:func:`~._signals.wave_image.ComplexSignal2D.add_phase_ramp`
method. The parameters `ramp_x` and `ramp_y` dictate the slope of the ramp in `x`- and `y` direction,
while the offset is determined by the `offset` parameter. The fulcrum of the linear ramp is at the origin
and the slopes are given in units of the axis with the according scale taken into account.
Both are available via the :py:class:`~.axes.AxesManager` of the signal.
