Energy-Dispersive X-Rays Spectrometry (EDS)
******************************************

.. versionadded:: 0.7

The methods described here are specific to the following signals:

* :py:class:`~._signals.eds_tem.EDSTEMSpectrum`
* :py:class:`~._signals.eds_sem.EDSSEMSpectrum`

This chapter described step by step a qualitative analysis of an EDS 
spectrum (SEM or TEM). 

Spectrum loading and parameters
-------------------------------

Loading
^^^^^^^^

All data loadings are done with the :py:func:`~.io.load` function as decribed in details in 
:ref:`Loading files<loading_files>`. Hyperspy is able to import different formats,
among them ".msa" and ".rpl" (the raw format of Oxford Instrument and Brucker). 

Here is three example for files exported by Oxford Instrument software (INCA).
For a single spectrum:

.. code-block:: python

    >>> spec = load("spectrum.msa")    
    
For a spectrum_image (The .rpl file is recorded as an image in this example,
The option record_by='spectrum' set it back to a spectrum):

.. code-block:: python

    >>> spec_img = load("spectrum_image.rpl",record_by="spectrum")   
    
For a stack of spectrum_images (The "*" replace all chains of string, in this
example 01, 02, 03,...):

.. code-block:: python

    >>> spec_img_3D = load("spectrum_image_*.rpl",stack=True)  
    >>> spec_img_3D = spec_img_3D.as_spectrum(0) 
    
Microscope and detector parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, the type of microscope ("EDS_TEM" or "EDS_SEM") needs to be set with the 
:py:meth:`~.signal.Signal.set_signal_type` method. The class of the
object is thus assigned, and specific EDS methods become available.

.. code-block:: python

    >>> spec.set_signal_type("EDS_TEM")
    
or as an argument of the :py:func:`~.io.load` function:
    
.. code-block:: python
    
    >>> spec = load("spectrum.msa",signal_type="EDS_TEM")
    
The main values for the energy axis and the microscope parameters are 
automatically imported from the file, if existing. The microscope and 
detector parameters are stored in stored in the 
:py:attr:`~.signal.Signal.mapped_parameters` 
attribute (see :ref:`mapped_parameters_structure`). These parameters can be displayed
as follow:
    
.. code-block:: python

    >>> spec.mapped_parameters.TEM
    ├── EDS
    │   ├── azimuth_angle = 0.0
    │   ├── elevation_angle = 37.0
    │   ├── energy_resolution_MnKa = 130.0
    │   ├── live_time = 30.0
    │   └── real_time = 55.0
    ├── beam_current = 0.0
    ├── beam_energy = 300
    └── tilt_stage = 36.0


They can be set directly:

.. code-block:: python

    >>> spec.mapped_parameters.TEM.beam_energy = 300

or with the  
:py:meth:`~._signals.eds_tem.EDSTEMSpectrum.set_microscope_parameters` method:

.. code-block:: python

    >>> spec.set_microscope_parameters(beam_energy = 300)
    
or raising the gui:
    
.. code-block:: python

    >>> spec.set_microscope_parameters()
    
.. figure::  images/EDS_microscope_parameters_gui.png
   :align:   center
   :width:   350  
   
If the microcsope parameters are not written in the original file, some 
of them are set by default. The default value can be changed in the 
:py:class:`~.defaults_parser.Preferences` class (see :ref:`preferences
<configuring-hyperspy-label>`).

.. code-block:: python

    >>> preferences.EDS.eds_detector_elevation = 37
    
or raising the gui:

.. code-block:: python

    >>> preferences.gui()
    
.. figure::  images/EDS_preferences_gui.png
   :align:   center
   :width:   400 

Energy axis
^^^^^^^^^^^

The properties of the energy axis can be set manually with the :py:class:`~.axes.AxesManager`.
(see :ref:`Axis properties<Setting_axis_properties>` for more info):

.. code-block:: python

    >>> spec.axes_manager[-1].name = 'E'
    >>> spec.axes_manager['E'].units = 'kV'
    >>> spec.axes_manager['E'].scale = 0.01
    >>> spec.axes_manager['E'].offset = -0.1

or with the :py:meth:`~.axes.AxesManager.gui` method:

.. code-block:: python

    >>> spec.axes_manager.gui()
    
.. figure::  images/EDS_energy_axis_gui.png
   :align:   center
   :width:   280 
   
Related method
^^^^^^^^^^^^^^

All the above parameters can be copy from one spectrum to the one other
with the :py:meth:`~._signals.eds_tem.EDSTEMSpectrum.get_calibration_from`.

.. code-block:: python

    >>> # Load spectrum.msa which contains the parameters
    >>> spec = load("spectrum.msa",signal_type="EDS_TEM")
    >>> # load spectrum_image.rpl which contains no parameters
    >>> spec_img = load("spectrum_image.rpl",record_by="spectrum",signal_type="EDS_TEM")
    >>> # Set all the properties of spec to spec_img
    >>> spec_img.get_calibration_from(s)
    
    
Describing the sample
---------------------


Set  and add elements
^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~._signals.eds.EDSSpectrum.set_elements` method is used 
to define a set of elements and corresponding X-ray lines
that will be used in other process (e.g. X-ray intensity mapping).
The information is stored in the :py:attr:`~.signal.Signal.mapped_parameters` 
attribute (see :ref:`mapped_parameters_structure`)


When the set_elements method erases all previously defined elements, 
the :py:meth:`~._signals.eds.EDSSpectrum.add_elements` method adds a new
set of elements to the previous set.

Plotting the spectrum
--------------------


Get lines intensity
^^^^^^^^^^^^^^^^^^^

With the :py:meth:`~._signals.eds.EDSSpectrum.get_lines_intensity`, the 
intensity of X-ray lines is used to generate a map. The number of counts
under the selected peaks is used.


