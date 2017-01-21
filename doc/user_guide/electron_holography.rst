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
[Gabor1948]_, [Tonomura1999]_, [McCartney2007]_ and [Joy1993]_. Fourier based
reconstruction of off-axis holograms (includes finding a side band in FFT,
isolating and filtering it, recenter and calculate inverse Fourier transform)
can be performed using
:meth:`~._signals.hologram_image.HologramImage.reconstruct_phase` method
which returns a :py:class:`~._signals.complex_signal2d.ComplexSignal2D` class, containing
the reconstructed electron wave. The :meth:`~._signals.hologram_image.HologramImage.reconstruct_phase` method takes sideband
position and size as parameters:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> im =  hs.datasets.example_signals.object_hologram()
    >>> wave_image = im.reconstruct_phase(sb_position=(<y>, <x>), sb_size=sb_radius)

The parameters can be found automatically by calling following methods:

.. code-block:: python

    >>> sb_position = im.estimate_sideband_position(ap_cb_radius=None, sb='lower')
    >>> sb_size = im.estimate_sideband_size(sb_position)

:meth:`~._signals.hologram_image.HologramImage.estimate_sideband_position` method searches for maximum of intensity in upper or lower part of FFT pattern (parameter ``sb``)
excluding the middle area defined by ``ap_cb_radius``. :meth:`~._signals.hologram_image.HologramImage.estimate_sideband_size` method calculates the radius of the sideband
filter as half of the distance to the central band which is commonly used for strong phase objects. Alternatively,
the sideband filter radius can be recalculate as 1/3 of the distance (often used for weak phase objects) for example:

.. code-block:: python

    >>> sb_size = sb_size * 2 / 3


To reconstruct the hologram with a vacuum reference wave, the reference hologram should be provided to the method either as Hyperspy's
:py:class:`~._signals.hologram_image.HologramImage` or as a nparray:

.. code-block:: python

    >>> reference_hologram = hs.datasets.example_signals.reference_hologram()
    >>> wave_image = im.reconstruct_phase(reference_hologram, sb_position=sb_position, sb_size=sb_sb_size)

Using reconstructed wave one can access its amplitude and phase (also unwrapped phase) using :meth:`~._signals.complex_signal2d.ComplexSignal2D.amplitude` and
 :meth:`~._signals.complex_signal2d.ComplexSignal2D.phase` properties (also :meth:`~._signals.complex_signal2d.ComplexSignal2D.unwrapped_phase` method):

.. code-block:: python

    >>> wave_image.unwrapped_phase().plot()

.. figure:: images/holography_unwrapped_phase.png
    :align: center

Preferences user interface.

Additionally, it is possible to change the smoothness of the sideband filter edge (which is by default set to 5% of the
filter radius) using parameter `sb_smoothness`.

Both ``sb_size`` and ``sb_smoothness`` can be provided in desired units rather than pixels (by default) by setting ``sb_unit``
value either to ``mrad`` or ``nm`` for milliradians or inverse nanometers respectively. For example:

.. code-block:: python

    >>> wave_image = im.reconstruct_phase(reference_hologram, sb_position=sb_position, sb_size=30,
                                          sb_smoothness=0.05*30,sb_unit='mrad')

Also the :meth:`~._signals.hologram_image.HologramImage.reconstruct_phase`
method can output wave images with desired size (shape). By default the shape
of the original hologram is preserved. Though this leads to oversampling of the
output wave images, since the information is limited by the size of the
sideband filter. To avoid oversampling the output shape can be set to the
diameter of the sideband as follows:

.. code-block:: python

    >>> wave_image = im.reconstruct_phase(reference_hologram, sb_position=sb_position,
                                          sb_size=sb_sb_size, output_shape=(2*sb_size, 2*sb_size))

Note that the :meth:`~._signals.hologram_image.HologramImage.reconstruct_phase` method can be called without parameters, which will cause their automatic assignment
by :meth:`~._signals.hologram_image.HologramImage.estimate_sideband_position`
and :meth:`~._signals.hologram_image.HologramImage.estimate_sideband_size`
methods. This, however, is not recommended for not experienced users.
