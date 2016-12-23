Electron Holography
*******************

HyperSpy provides the user with two classes which can be used to process electron holography data:
 
* :py:class:`~._signals.hologram_image.HologramImage`
* :py:class:`~._signals.electron_wave_image.ElectronWaveImage`

The classes inherit from :py:class:`~._signals.signal2d.Signal2D` and :py:class:`~._signals.electron_wave_image.ComplexSignal2D`
classes respectively and thus can use all of its functionality. The usage of both classes is explained in the following sections.


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
which returns a :py:class:`~._signals.electron_wave_image.ElectronWaveImage` class, containing the reconstructed
electron wave. At the moment the reconstruction can be done using square images only! The `reconstruct_phase` method
takes sideband position and size as parameters:

.. code-block:: python

    >>> wave_image = im.reconstruct_phase(sb_position=(<x>, <y>), sb_size=sb_radius)

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

After the reconstruction the reconstruction parameters can be viewed as follows:

.. code-block:: python

    >>> wave_image.display_reconstruction_parameters


This option can be used for example to check the sideband size value in px after reconstruction with `sb_size` given in
mrad or nm.

Further analyses of the reconstructed wave (phase) can be done using :py:class:`~._signals.electron_wave_image.ElectronWaveImage` class
functionality (see bellow).


The ElectronWaveImage class
===========================

The :py:class:`~._signals.electron_wave_image.ElectronWaveImage` class can hold information about the complex electron
wave. As such, relevant properties like the `amplitude`, `phase` and the `real` and `imag` part can be
directly accessed and return appropriate :py:class:`~._signals.signal2d.Signal2D` signals.

To transform a :py:class:`~._signals.complex_signal2d.ComplexSignal2D` (or subclass) into a
:py:class:`~._signals.electron_wave_image.ElectronWaveImage` use:

.. code-block:: python

    >>> im.set_signal_type('electron_wave')


Unwrap the phase
----------------

With the :py:func:`~._signals.electron_wave_image.ElectronWaveImage.get_unwrapped_phase` method the phase can be
unwrapped and returned as an :class:`~hyperspy._signals.signal2d.Signal2D`. The underlying method is
:py:func:`~skimage.restoration.unwrap`.


Add a linear ramp
-----------------

A linear ramp can be added to the wave via the :py:func:`~._signals.electron_wave_image.ElectronWaveImage.add_phase_ramp`
method. The parameters `ramp_x` and `ramp_y` dictate the slope of the ramp in `x`- and `y` direction,
while the offset is determined by the `offset` parameter. The fulcrum of the linear ramp is at the origin
and the slopes are given in units of the axis with the according scale taken into account.
Both are available via the :py:class:`~.axes.AxesManager` of the signal.
