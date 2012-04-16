
Tools
*****

The Signal class and its subclasses
-----------------------------------

In Hyperspy the data and most of the data analysis functions are contained in a :py:class:`~.signal.Signal` class or subclass. The :py:class:`~.signal.Signal` class contains general functionality that is available to all the subclasses. The subclasses provide functionality that is normally specific to a particular type of data, e.g. the :py:class:`~.signals.spectrum.Spectrum` class provides common functionality to deal with spectral data and :py:class:`~.signals.eels.EELSSpectrum` (that is a subclass of :py:class:`~.signals.spectrum.Spectrum`) adds extra functionality to the :py:class:`~.signals.spectrum.Spectrum` class for electron energy-loss spectroscopy data analysis.

Currently the following signal subclasses are available:

* :py:class:`~.signals.spectrum.Spectrum`
* :py:class:`~.signals.image.Image`
* :py:class:`~.signals.eels.EELSSpectrum`
* :py:class:`~.signals.spectrum_simulation.SpectrumSimulation`

The data is stores in the :py:attr:`~.signal.Signal.data` attribute, the original parameters in the :py:attr:`~.signal.Signal.original_parameters` attribute, the mapped parameters in the :py:attr:`~.signal.Signal.mapped_parameters` attribute and the axes information (incluiding calibration) can be accessed (and modified) in the :py:attr:`~.signal.Signal.axes_manager` attribute. All the methods of the class provides functionality.


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
    
    # We convert it in an Image object
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
 


Generic tools
-------------

These are the tools (methods) which are available to all the signals. In this section only a small data set is discussed, for a detailed list of all the methods available see the :py:class:`~.signal.Signal` documentation.

Cropping
^^^^^^^^

The following methods are available to crop a given axis:

* :py:meth:`~.signal.Signal.crop_in_pixels`
* :py:meth:`~.signal.Signal.crop_in_units`

Rebinning
^^^^^^^^^

The :py:meth:`~.signal.Signal.rebin` method rebins that data in place.

Folding and unfolding
^^^^^^^^^^^^^^^^^^^^^

When dealing with multidimensional datasets it is sometimes useful to transform the data in a two dimensional dataset. This feature is provided by the following two method:

* :py:meth:`~.signal.Signal.fold`
* :py:meth:`~.signal.Signal.unfold`

It is also possible to unfold only the navigation or signal space:

* :py:meth:`~.signal.Signal.unfold_navigation_space`
* :py:meth:`~.signal.Signal.unfold_signal_space`

Sum or average over one axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signal.Signal.sum`
* :py:meth:`~.signal.Signal.mean`

Changing the data type
^^^^^^^^^^^^^^^^^^^^^^

Even if the original data is recorded with a limited dynamic range, it is often desirable to perform the analysis operations with a higher precision. Conversely, storing in a shorter data type can decrease the file size. The :py:meth:`~.signal.Signal.change_dtype` changes the data type in place, e.g.:

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

These methods are only available in the following signals:

* :py:class:`~.signals.spectrum.Spectrum`
* :py:class:`~.signals.image.Image`
* :py:class:`~.signals.eels.EELSSpectrum`
* :py:class:`~.signals.spectrum_simulation.SpectrumSimulation`


Cropping
^^^^^^^^

The :py:meth:`~.signals.spectrum.Spectrum.crop_spectrum`, if not parameter is passed, provides a user interface to crop the spectrum.

Background removal
^^^^^^^^^^^^^^^^^^

The :py:meth:`~.signals.spectrum.Spectrum.remove_background` provides a user interface to remove some background funtions.

Calibration
^^^^^^^^^^^
The :py:meth:`~.signals.spectrum.Spectrum.calibrate` provides a user interface to calibrate the spectral axis.

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

Currently there are not methods unique to the image class. 

EELS tools
----------

These methods are only available in the following signals:

* :py:class:`~.signals.eels.EELSSpectrum`

Spikes removal
^^^^^^^^^^^^^^

The following methods can assist in removing spikes by interpolation:

* :py:meth:`~.signals.eels.EELSSpectrum.spikes_diagnosis`
* :py:meth:`~.signals.eels.EELSSpectrum.plot_spikes`
* :py:meth:`~.signals.eels.EELSSpectrum.remove_spikes`

The workflow is as follows:

* Use :py:meth:`~.signals.eels.EELSSpectrum.spikes_diagnosis` to select a threshold in the histogram. All the points above the threshold will be indentified as spikes.
* Use :py:meth:`~.signals.eels.EELSSpectrum.plot_spikes` to plot the spikes detected using the given threshold. This method returns a list of spikes that can be edited and passes to :py:meth:`~.signals.eels.EELSSpectrum.remove_spikes` if the automatic detection is too coarse.
* When the optimal threshold have been found, use :py:meth:`~.signals.eels.EELSSpectrum.remove_spikes` to remove the spikes by interpolation. This method also accepts a list of coordinates and other parameters.

Define the elemental composition of the sample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It can be useful to define the composition of the sample for archiving purposes or for some features (e.g. curve fitting) that may use this information. The elemental composition of the sample can be defined using :py:meth:`~.signals.eels.EELSSpectrum.remove_spikes`. The information is stored in the :py:attr:`~.signal.Signal.mapped_parameters` attribute (see :ref:`mapped_parameters_structure`)

Estimate the FWHM of a peak
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signals.eels.EELSSpectrum.calculate_FWHM`

Estimate the thickness
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~.signals.eels.EELSSpectrum.calculate_thickness` can estimate the thickness from a low loss EELS spectrum.

Estimate zero loss peak centre
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signals.eels.EELSSpectrum.find_low_loss_centre`

Fourier-log deconvolution
^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signals.eels.EELSSpectrum.fourier_log_deconvolution`
 




