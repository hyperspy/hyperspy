
Electron Energy Loss Spectroscopy
*********************************

Tools
-----

These methods are only available for the following signals:

* :py:class:`~._signals.eels.EELSSpectrum`

Define the elemental composition of the sample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It can be useful to define the composition of the sample for archiving purposes
or for some other process (e.g. curve fitting) that may use this information.
The elemental composition of the sample can be defined using
:py:meth:`~._signals.eels.EELSSpectrum.add_elements`. The information is stored
in the :py:attr:`~.signal.Signal.metadata` attribute (see
:ref:`metadata_structure`)

Estimate the thickness
^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~._signals.eels.EELSSpectrum.estimate_thickness` can estimate the
thickness from a low-loss EELS spectrum.

Estimate zero loss peak centre
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~._signals.eels.EELSSpectrum.estimate_zero_loss_peak_centre`

Deconvolutions
^^^^^^^^^^^^^^

* :py:meth:`~._signals.eels.EELSSpectrum.fourier_log_deconvolution`
* :py:meth:`~._signals.eels.EELSSpectrum.fourier_ratio_deconvolution`
* :py:meth:`~._signals.eels.EELSSpectrum.richardson_lucy_deconvolution`

Estimate elastic scattering threshold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use
:py:meth:`~._signals.eels.EELSSpectrum.estimate_elastic_scattering_threshold` to
calculate separation point between elastic and inelastic scattering on some
EELS low-loss spectra. This algorithm calculates the derivative of the signal
and assigns the inflexion point to the first point below a certain tolerance.
This tolerance value can be set using the tol keyword.

Currently, the method uses smoothing to reduce the impact of the noise in the
measure. The number of points used for the smoothing window can be specified by
the npoints keyword. 

Estimate elastic scattering intensity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use
:py:meth:`~._signals.eels.EELSSpectrum.estimate_elastic_scattering_intensity`
to calculate the integral below the zero loss peak (elastic intensity) from
EELS low-loss spectra containing the zero loss peak. This integral can use the
threshold image calculated by the
:py:meth:`~._signals.eels.EELSSpectrum.estimate_elastic_scattering_threshold`
as end energy for the integration at each spectra or use the same energy value
for all spectra. Also, if no threshold is specified, the routine will perform a
rough estimation of the inflexion values at each spectrum.


.. _eels.kk:

Kramers-Kronig Analysis
^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 0.7

The single-scattering EEL spectrum is approximately related to the complex
permittivity of the sample and can be estimated by Kramers-Kronig analysis.
The :py:meth:`~._signals.eels.EELSSpectrum.kramers_kronig_analysis` method
inplements the Kramers-Kronig FFT method as in [Egerton2011]_ to estimate the
complex dielectric funtion from a low-loss EELS spectrum. In addition, it can
estimate the thickness if the refractive index is known and approximately
correct for surface plasmon excitations in layers.

.. _eels_tools-label:



EELS curve fitting
------------------

HyperSpy makes it really easy to quantify EELS core-loss spectra by curve
fitting as it is shown in the next example of quantification of a boron nitride
EELS spectrum from the `The EELS Data Base
<http://pc-web.cemes.fr/eelsdb/index.php?page=home.php>`_. 

Load the core-loss and low-loss spectra


.. code-block:: python
       
    >>> s = load("BN_(hex)_B_K_Giovanni_Bertoni_100.msa")
    >>> ll = load("BN_(hex)_LowLoss_Giovanni_Bertoni_96.msa")


Set some important experimental information that is missing from the original
core-loss file

.. code-block:: python
       
    >>> s.set_microscope_parameters(beam_energy=100, convergence_angle=0.2, collection_angle=2.55)
    
    
Define the chemical composition of the sample

.. code-block:: python
       
    >>> s.add_elements(('B', 'N'))
    
    
We pass the low-loss spectrum to :py:func:`~.hspy.create_model` to include the
effect of multiple scattering by Fourier-ratio convolution.

.. code-block:: python
       
    >>> m = create_model(s, ll=ll)


HyperSpy has created the model and configured it automatically:

.. code-block:: python
       
    >>> m
    [<background (PowerLaw component)>,
    <N_K (EELSCLEdge component)>,
    <B_K (EELSCLEdge component)>]


Furthermore, the components are available in the user namespace

.. code-block:: python

    >>> N_K
    <N_K (EELSCLEdge component)>
    >>> B_K
    <B_K (EELSCLEdge component)>
    >>> background
    <background (PowerLaw component)>


Conveniently, variables named as the element symbol contain all the eels
core-loss components of the element to facilitate applying some methods to all
of them at once. Although in this example the list contains just one component
this is not generally the case.

.. code-block:: python
       
    >>> N
    [<N_K (EELSCLEdge component)>]


By default the fine structure features are disabled (although the default value
can be configured (see :ref:`configuring-hyperspy-label`). We must enable them
to accurately fit this spectrum.

.. code-block:: python
       
    >>> m.enable_fine_structure()


We use smart_fit instead of standard fit method because smart_fit is optimized
to fit EELS core-loss spectra

.. code-block:: python
       
    >>> m.smart_fit()

Print the result of the fit 

.. code-block:: python

    >>> m.quantify()
    Absolute quantification:
    Elem.	Intensity
    B	0.045648
    N	0.048061


Visualize the result

.. code-block:: python

    >>> m.plot()
    

.. figure::  images/curve_fitting_BN.png
   :align:   center
   :width:   500    

   Curve fitting quantification of a boron nitride EELS core-loss spectrum from
   `The EELS Data Base
   <http://pc-web.cemes.fr/eelsdb/index.php?page=home.php>`_
   

There are several methods that are only available in
:py:class:`~.models.eelsmodel.EELSModel`:

* :py:meth:`~.models.eelsmodel.EELSModel.smart_fit` is a fit method that is 
  more robust than the standard routine when fitting EELS data.
* :py:meth:`~.models.eelsmodel.EELSModel.quantify` prints the intensity at 
  the current locations of all the EELS ionisation edges in the model.
* :py:meth:`~.models.eelsmodel.EELSModel.remove_fine_structure_data` removes 
  the fine structure spectral data range (as defined by the 
  :py:attr:`~._components.eels_cl_edge.EELSCLEdge.fine_structure_width)` 
  ionisation edge components. It is specially useful when fitting without 
  convolving with a zero-loss peak.

The following methods permit to easily enable/disable background and ionisation
edges components:

* :py:meth:`~.models.eelsmodel.EELSModel.enable_edges`
* :py:meth:`~.models.eelsmodel.EELSModel.enable_background`
* :py:meth:`~.models.eelsmodel.EELSModel.disable_background`
* :py:meth:`~.models.eelsmodel.EELSModel.enable_fine_structure`
* :py:meth:`~.models.eelsmodel.EELSModel.disable_fine_structure`

The following methods permit to easily enable/disable several ionisation 
edge functionalities:

* :py:meth:`~.models.eelsmodel.EELSModel.set_all_edges_intensities_positive`
* :py:meth:`~.models.eelsmodel.EELSModel.unset_all_edges_intensities_positive`
* :py:meth:`~.models.eelsmodel.EELSModel.enable_free_onset_energy`
* :py:meth:`~.models.eelsmodel.EELSModel.disable_free_onset_energy`
* :py:meth:`~.models.eelsmodel.EELSModel.fix_edges`
* :py:meth:`~.models.eelsmodel.EELSModel.free_edges`
* :py:meth:`~.models.eelsmodel.EELSModel.fix_fine_structure`
* :py:meth:`~.models.eelsmodel.EELSModel.free_fine_structure`
