
Tools
*****

The Signal class and its subclasses
-----------------------------------

In Hyperspy there are a number of different data types that can be handled. In programming terms we would say they are all contained in the :py:class:`~.signal.Signal` class which has several (currently 4) subclasses; most of the data analysis functions are also contained in these classes. The :py:class:`~.signal.Signal` class contains general functionality that is available to all the subclasses. The subclasses provide functionality that is normally specific to a particular type of data, e.g. the :py:class:`~.signals.spectrum.Spectrum` class provides common functionality to deal with spectral data and :py:class:`~.signals.eels.EELSSpectrum` (which is a subclass of :py:class:`~.signals.spectrum.Spectrum`) adds extra functionality to the :py:class:`~.signals.spectrum.Spectrum` class for electron energy-loss spectroscopy data analysis.

Currently the following signal subclasses are available:

* :py:class:`~.signals.spectrum.Spectrum`
* :py:class:`~.signals.image.Image`
* :py:class:`~.signals.eels.EELSSpectrum`
* :py:class:`~.signals.spectrum_simulation.SpectrumSimulation`

Datasets (and other types of objects handled in Hyperspy) store their different components in "attributes".The data is stored in the :py:attr:`~.signal.Signal.data` attribute, the original parameters in the :py:attr:`~.signal.Signal.original_parameters` attribute, the mapped parameters in the :py:attr:`~.signal.Signal.mapped_parameters` attribute and the axes information (including calibration) can be accessed (and modified) in the :py:attr:`~.signal.Signal.axes_manager` attribute.


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
    

When transforming between spectrum and image classes the order in which the data array is stored in memory is modified and several functions, e.g. plotting or decomposing, will behave differently.

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


Image tools
-----------

Currently there are no methods unique to the image class. 

EELS tools
----------

These methods are only available for the following signals:

* :py:class:`~.signals.eels.EELSSpectrum`

Spikes removal
^^^^^^^^^^^^^^

The following methods can assist in removing spikes by interpolation:

* :py:meth:`~.signals.eels.EELSSpectrum.spikes_diagnosis`
* :py:meth:`~.signals.eels.EELSSpectrum.plot_spikes`
* :py:meth:`~.signals.eels.EELSSpectrum.remove_spikes`

The workflow is as follows:

* Use :py:meth:`~.signals.eels.EELSSpectrum.spikes_diagnosis` to display a histogram which can be used to select a threshold above which all points will be indentified as spikes.

* When the optimal threshold has been found, use it as a parameter in :py:meth:`~.signals.eels.EELSSpectrum.remove_spikes` to remove the spikes by interpolation. This method also accepts a list of coordinates and other parameters.

* Use :py:meth:`~.signals.eels.EELSSpectrum.plot_spikes` to plot all the spectra containing points above the threshold (which is passed as a parameter). This method returns a list of spikes that can be edited and passed to :py:meth:`~.signals.eels.EELSSpectrum.remove_spikes` if the automatic detection is too coarse.

Define the elemental composition of the sample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It can be useful to define the composition of the sample for archiving purposes or for some other process (e.g. curve fitting) that may use this information. The elemental composition of the sample can be defined using :py:meth:`~.signals.eels.EELSSpectrum.add_elements`. The information is stored in the :py:attr:`~.signal.Signal.mapped_parameters` attribute (see :ref:`mapped_parameters_structure`)

Estimate the FWHM of a peak
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signals.eels.EELSSpectrum.calculate_FWHM`

Estimate the thickness
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~.signals.eels.EELSSpectrum.calculate_thickness` can estimate the thickness from a low-loss EELS spectrum.

Estimate zero loss peak centre
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signals.eels.EELSSpectrum.find_low_loss_centre`

Fourier-log deconvolution
^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signals.eels.EELSSpectrum.fourier_log_deconvolution`
 




