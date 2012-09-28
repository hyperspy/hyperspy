
Tools
*****

The Signal class and its subclasses
-----------------------------------

.. WARNING::
    This subsection can be a bit confusing for beginners. Do not worry if you do not understand it all.
    

Hyperspy stores hyperspectra in the :py:class:`~.signal.Signal` class, that is the object that you get when e.g. you load a single file using :py:func:`~.io.load`. Most of the data analysis functions are also contained in this class or its specialized subclasses. The :py:class:`~.signal.Signal` class contains general functionality that is available to all the subclasses. The subclasses provide functionality that is normally specific to a particular type of data, e.g. the :py:class:`~.signals.spectrum.Spectrum` class provides common functionality to deal with spectral data and :py:class:`~.signals.eels.EELSSpectrum` (which is a subclass of :py:class:`~.signals.spectrum.Spectrum`) adds extra functionality to the :py:class:`~.signals.spectrum.Spectrum` class for electron energy-loss spectroscopy data analysis.

Currently the following signal subclasses are available:

* :py:class:`~.signals.spectrum.Spectrum`
* :py:class:`~.signals.image.Image`
* :py:class:`~.signals.eels.EELSSpectrum`
* :py:class:`~.signals.spectrum_simulation.SpectrumSimulation`

The :py:mod:`~.signals` module is imported in the user namespace when
loading hyperspy.

The different signals store other objects in what are called attributes. For examples, the hyperspectral data is stored in the :py:attr:`~.signal.Signal.data` attribute, the original parameters in the :py:attr:`~.signal.Signal.original_parameters` attribute, the mapped parameters in the :py:attr:`~.signal.Signal.mapped_parameters` attribute and the axes information (including calibration) can be accessed (and modified) in the :py:attr:`~.signal.Signal.axes_manager` attribute.


Transforming between signal subclasses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
It is possible to transform between signal subclasses, e.g.:

.. code-block:: python
    
    >>> s = load('EELS Spectrum Image (high-loss).dm3')

	Title: EELS Spectrum Image (high-loss).dm3
	Signal type: EELS
	Data dimensions: (21, 42, 2048)
	Data representation: spectrum

    
    # We check the type of object that loading the file has created:
    >>> s
    <EELSSpectrum, title: EELS Spectrum Image (high-loss).dm3, dimensions: (21, 42, 2048)>
    
    # We convert it into an Image object
    >>> im = s.to_image()
    >>> im
    <Image, title: EELS Spectrum Image (high-loss).dm3, dimensions: (2048, 21, 42)>
    # And now we turn it into a Spectrum
    s2 = im.to_spectrum()
    >>> s2
    <Spectrum, title: EELS Spectrum Image (high-loss).dm3, dimensions: (21, 42, 2048)>
    # And now back to EELSSpectrum
    >>> s3 = s2.to_EELS()
    >>> s3
    <EELSSpectrum, title: EELS Spectrum Image (high-loss).dm3, dimensions: (21, 42, 2048)>
    

When transforming between spectrum and image classes the order in which the data array is stored in memory is modified to improve performance and several functions, e.g. plotting or decomposing, will behave differently.

Below we briefly introduce some of the most commonly used tools (methods). For more details about a particular method click on its name. For a detailed list of all the methods available see the :py:class:`~.signal.Signal` documentation.

Generic tools
-------------

These are the methods which are available to all the signals.

Cropping
^^^^^^^^

The following methods are available to crop a given axis:

* :py:meth:`~.signal.Signal.crop_in_pixels`
* :py:meth:`~.signal.Signal.crop_in_units`

Rebinning
^^^^^^^^^

The :py:meth:`~.signal.Signal.rebin` method rebins data in place down to a size determined by the user.

Folding and unfolding
^^^^^^^^^^^^^^^^^^^^^

When dealing with multidimensional datasets it is sometimes useful to transform the data into a two dimensional dataset. This can be accomplished using the following two methods:

* :py:meth:`~.signal.Signal.fold`
* :py:meth:`~.signal.Signal.unfold`

It is also possible to unfold only the navigation or only the signal space:

* :py:meth:`~.signal.Signal.unfold_navigation_space`
* :py:meth:`~.signal.Signal.unfold_signal_space`

Sum or average over one axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signal.Signal.sum`
* :py:meth:`~.signal.Signal.mean`

Changing the data type
^^^^^^^^^^^^^^^^^^^^^^

Even if the original data is recorded with a limited dynamic range, it is often desirable to perform the analysis operations with a higher precision. Conversely, if space is limited, storing in a shorter data type can decrease the file size. The :py:meth:`~.signal.Signal.change_dtype` changes the data type in place, e.g.:

.. code-block:: python

    >>> s = load('EELS Spectrum Image (high-loss).dm3')
        Title: EELS Spectrum Image (high-loss).dm3
        Signal type: EELS
        Data dimensions: (21, 42, 2048)
        Data representation: spectrum
        Data type: float32
    >>> s.change_dtype('float64')
    >>> print(s)
        Title: EELS Spectrum Image (high-loss).dm3
        Signal type: EELS
        Data dimensions: (21, 42, 2048)
        Data representation: spectrum
        Data type: float64



Spectrum tools
--------------

These methods are only available for the following signals:

* :py:class:`~.signals.spectrum.Spectrum`
* :py:class:`~.signals.eels.EELSSpectrum`
* :py:class:`~.signals.spectrum_simulation.SpectrumSimulation`


Cropping
^^^^^^^^

The :py:meth:`~.signals.spectrum.Spectrum.crop_spectrum`, method is used to crop the spectral energy range. If no parameter is passed, a user interface appears in which to crop the spectrum.

Background removal
^^^^^^^^^^^^^^^^^^

The :py:meth:`~.signals.spectrum.Spectrum.remove_background` method provides a user interface to remove some background functions.

Calibration
^^^^^^^^^^^
The :py:meth:`~.signals.spectrum.Spectrum.calibrate` method provides a user interface to calibrate the spectral axis.

Aligning
^^^^^^^^

The following methods use sub-pixel cross-correlation or user-provided shifts to align spectra. They support applying the same transformation to multiple files.

* :py:meth:`~.signals.spectrum.Spectrum.align_1D`
* :py:meth:`~.signals.spectrum.Spectrum.align_with_array_1D`


Data smoothing
^^^^^^^^^^^^^^

The following methods (that include user interfaces when no arguments are passed) can perform data smoothing with different algorithms:

* :py:meth:`~.signals.spectrum.Spectrum.smooth_lowess`
* :py:meth:`~.signals.spectrum.Spectrum.smooth_tv`
* :py:meth:`~.signals.spectrum.Spectrum.smooth_savitzky_golay`

Other methods
^^^^^^^^^^^^^^

* :py:meth:`~.signals.spectrum.Spectrum.hanning_taper`



Image tools
-----------

* :py:meth:`~.signals.image.Image.crop_image`


Image registration (alignment)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 0.5

The :py:meth:`~.signals.image.Image.align2D` method provides advanced image alignment functionality, including subpixel alignment.




EELS tools
----------

These methods are only available for the following signals:

* :py:class:`~.signals.eels.EELSSpectrum`

Spikes removal
^^^^^^^^^^^^^^
.. versionadded:: 0.5
    The :py:meth:`~.signals.eels.EELSSpectrum.spikes_removal_tool` replaces the old :py:meth:`~.signals.eels.EELSSpectrum.remove_spikes`.


:py:meth:`~.signals.eels.EELSSpectrum.spikes_removal_tool` provides an user interface to remove spikes from spectra.


.. figure::  images/spikes_removal_tool.png
   :align:   center
   :width:   500    

   Spikes removal tool


Define the elemental composition of the sample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It can be useful to define the composition of the sample for archiving purposes or for some other process (e.g. curve fitting) that may use this information. The elemental composition of the sample can be defined using :py:meth:`~.signals.eels.EELSSpectrum.add_elements`. The information is stored in the :py:attr:`~.signal.Signal.mapped_parameters` attribute (see :ref:`mapped_parameters_structure`)

Estimate the FWHM of a peak
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signals.eels.EELSSpectrum.estimate_FWHM`

Estimate the thickness
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~.signals.eels.EELSSpectrum.estimate_thickness` can estimate the thickness from a low-loss EELS spectrum.

Estimate zero loss peak centre
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signals.eels.EELSSpectrum.estimate_zero_loss_peak_centre`

Deconvolutions
^^^^^^^^^^^^^^

* :py:meth:`~.signals.eels.EELSSpectrum.fourier_log_deconvolution`
* :py:meth:`~.signals.eels.EELSSpectrum.fourier_ratio_deconvolution`
* :py:meth:`~.signals.eels.EELSSpectrum.richardson_lucy_deconvolution`

 




