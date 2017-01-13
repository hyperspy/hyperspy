Electron Holography
*******************

HyperSpy provides the user with a signal class which can be used to process electron holography data:
 
* :py:class:`~._signals.hologram_image.HologramImage`

It inherits from :py:class:`~._signals.signal2d.Signal2D` class and thus can use all of its functionality.
The usage of the class is explained in the following sections.


The HologramImage class
=======================

The :py:class:`~._signals.hologram_image.HologramImage` class is designed to contain images acquired via
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
which returns a :py:class:`~._signals.electron_wave_image.ComplexImage2D` class and in future releases
:py:class:`~._signals.electron_wave_image.ElectronWaveImage` class, containing the reconstructed
electron wave. The `reconstruct_phase` method
takes sideband position and size as parameters:

.. code-block:: python

    >>> wave_image = im.reconstruct_phase(sb_position=(<y>, <x>), sb_size=sb_radius)

The parameters can be found automatically by calling following methods:

.. code-block:: python

    >>> sb_position = im.estimate_sideband_position(ap_cb_radius=None, sb='lower')
    >>> sb_size = im.estimate_sideband_size(sb_position)

`estimate_sideband_position` method searches for maximum of intensity in upper or lower part of FFT pattern (parameter `sb`)
excluding the middle area defined by `ap_cb_radius`. `estimate_sideband_size` method calculates the radius of the sideband
filter as half of the distance to the central band which is commonly used for strong phase objects. Alternatively,
the sideband filter radius can be recalculate as 1/3 of the distance (often used for weak phase objects) for example:

.. code-block:: python

    >>> sb_size = sb_size * 2 / 3


To reconstruct the hologram with a vacuum reference wave, the reference hologram should be provided to the method either as Hyperspy's
:py:class:`~._signals.hologram_image.HologramImage` or as a nparray:

.. code-block:: python

    >>> wave_image = im.reconstruct_phase(reference_hologram, sb_position=sb_position, sb_size=sb_sb_size)


Additionally, it is possible to change the smoothness of the sideband filter edge (which is by default set to 5% of the
filter radius) using parameter `sb_smoothness`.

Both `sb_size` and `sb_smoothness` can be provided in desired units rather than pixels (by default) by setting `sb_unit`
value either to `mrad` or `nm` for milliradians or inverse nanometers respectively. For example:

.. code-block:: python

    >>> wave_image = im.reconstruct_phase(reference_hologram, sb_position=sb_position, sb_size=30,
                                          sb_smoothness=0.05*30,sb_unit='mrad')

Also the `reconstruct_phase` method can output wave images with desired size (shape). By default the shape of the
original hologram is preserved. Though this leads to oversampling of the output wave images, since the information is
limited by the size of the sideband filter. To avoid oversampling the the output shape can be set to the diameter of the
sideband as follows:

.. code-block:: python

    >>> wave_image = im.reconstruct_phase(reference_hologram, sb_position=sb_position,
                                          sb_size=sb_sb_size, output_shape=(2*sb_size, 2*sb_size))

Note that the `reconstruct_phase` method can be called without parameters, which will cause their automatic assignment
by `estimate_sideband_position` and `estimate_sideband_size` methods. This, however, is not recommended for not experienced
users.
