Electron Holography
*******************

HyperSpy provides the user with a class which can be used to process electron holography data:
 
* :py:class:`~._signals.hologram_image.HologramImage`

It inherits directly from the :py:class:`~._signals.signal2d.Signal2D` class and thus can use all of
its functionality. The usage is explained in the following sections.


The HologramImage class
=======================

The :py:class:`~._signals.hologram_image.HologramImage` class is designed to hold images acquired via
electron holography.

To transform a :py:class:`~._signals.signal2d.Signal2D` (or subclass) into a
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
:py:class:`~._signals.signal2d.Signal2D` (or subclass) or as a nparray:

.. code-block:: python

    >>> wave_image = im.reconstruct_phase(ref_image)

The reconstruction parameters can be extracted from metadata of wave_image as follows:

.. code-block:: python

    >>> reconstruction_param = wave_image.rec_param

This option can be used for batch processing as the reconstruction_param can be provided to the reconstruction method as follows:

.. code-block:: python

    >>> wave_image1 = im1.reconstruct_phase(rec_param=reconstruction_param)

Further analyses of the reconstructed wave (phase) can be done using the :py:class:`~._signals.wave_image.ComplexSignal2D` class
functionality (see dedicated documentation).