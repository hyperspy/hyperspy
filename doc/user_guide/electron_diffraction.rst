.. _ed-label:

Electron Diffraction
********************

Methods to analyse electron diffraction data acquired conventional or scanning
mode of a scanning transmission electron microscope (STEM) are described in
this chapter. These methods are specific to the signals in the
:py:class:`~._signals.electron_diffraction.ElectronDiffraction` class.

.. NOTE::

    See also the `SED tutorials <http://nbviewer.ipython.org/github/hyperspy/hyperspy-demos/blob/master/electron_microscopy/SED/>`_ .

The data used throughout the following examples can be downloaded as follows:

.. code-block:: python

    >>> from urllib import urlretrieve
    >>> url = 'http://cook.msm.cam.ac.uk//~hyperspy//SED_tutorial//'
    >>> urlretrieve(url + 'diffraction_example.tif', 'sed_nanowire.hdf5')

Loading electron diffraction data & setting microscope parameters
-----------------------------------------------------------------

Loading
^^^^^^^^

All data are loaded with the :py:func:`~.io.load` function, as described in
detail in :ref:`Loading files<loading_files>`. HyperSpy is able to import
different formats, among them ".blo" (the format used by NanoMEGAS). Below are
examples for loading a single diffraction pattern and loading a stack of
diffraction patterns.

For a single diffraction pattern:

.. code-block:: python

    >>> dp = hs.load("diffraction_example.tif")
    >>> dp
    <ElectronDiffraction, title: , dimensions: (|144, 144)>

For a scanning electron diffraction dataset stored as a stack of images:

.. code-block:: python

    >>> dp = hs.load("sed_nanowire.hdf5", signal_type="electron_diffraction")
    >>> dp
    <ElectronDiffraction, title: , dimensions: (100, 30|144, 144)>


Microscope and detector parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Microscope and detector parameters are stored in the
:py:attr:`~.signal.Signal.metadata` attribute (see :ref:`metadata_structure`).
These parameters can be displayed as follows:

.. code-block:: python

    >>> dp = hs.load("sed_nanowire.hdf5", signal_type="electron_diffraction")
    >>> dp.metadata
    ├── Detector
    │   └── Diffraction
    │       ├── camera_length = 0.006855
    │       └── exposure_time = 0.0
    ├── beam_energy = 300.0
    ├── beam_energy = 15.0
    └── scan_rotation = 38.0

Parameters can be specified directly:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.hdf5", signal_type="electron_diffraction")
    >>> dp.metadata.Acquisition_instrument.SED.convergence_angle = 5.

or with the
:py:meth:`~._signals.electron_diffraction.ElectronDiffraction.set_microscope_parameters` method:

.. code-block:: python

    >>> dp = hs.load("sed_nanowire.hdf5", signal_type="electron_diffraction")
    >>> dp.set_microscope_parameters(convergence_angle=5.)

or raising the gui:

.. code-block:: python

    >>> dp = hs.load("sed_nanowire.hdf5", signal_type="electron_diffraction")
    >>> dp.set_microscope_parameters()

.. figure::  images/SED_microscope_parameters_gui.png
   :align:   center
   :width:   400

   SED microscope parameters window.

If the microscope and detector parameters are not written in the original file,
some of them are set by default. The default values can be changed in the
:py:class:`~.defaults_parser.Preferences` class (see :ref:`preferences
<configuring-hyperspy-label>`).

.. code-block:: python

    >>> hs.preferences.SED.precession_angle = 36.

or raising the gui:

.. code-block:: python

    >>> hs.preferences.gui()

.. figure::  images/SED_preferences_gui.png
   :align:   center
   :width:   400

   SED preferences window.


Pattern alignment
-----------------

Alignment is based on determining the direct beam position, which is assumed to
move only small amounts on the detector throughout the dataset. The position of
the direct beam is estimated using the approach described by Zaeferrer
[Zaeferrer2000]_. Briefly, the patterns in the stack are first summed and it is
assumed that the direct beam is reinforced such that the maximum in the summed
pattern is a reasonable estimate of the direct beam position. The pixels within
a user specified circular region around the initial guess are then searched for
higher intensity values until the local maximum is found. If numerous pixels
have the same value, due to beam saturation, the average index is taken.

.. code-block:: python

    >>> dp = hs.load("sed_nanowire.hdf5", signal_type="electron_diffraction")
    >>> centers = dp.get_direct_beam_position()

The shifts required to center each pattern are calculated from the direct beam
positions. These shifts can then be applied to the data to align the stack using
the py:meth:`~._signals.signal2d.align2D()` method.

.. code-block:: python

    >>> dp = hs.load("sed_nanowire.hdf5", signal_type="electron_diffraction")
    >>> shifts = dp.get_direct_beam_shifts()
    >>> dp.align2D(shifts=shifts)


Radial profile calculation
--------------------------

The radial average profile of each electron diffraction pattern about a given
center can be calculated and returned as a
:py:class:`~._signals.signal1d.Signal1D` class object. If no center is specified
by the user it is assumed that the center should be the direct beam position,
which is estimated using the py:meth:`~._signals.electron_diffraction.get_direct_beam_position()`
method.

.. code-block:: python

    >>> dp = hs.load("sed_nanowire.hdf5", signal_type="electron_diffraction")
    >>> rp = dp.get_radial_profile()
    >>> rp.plot()

.. figure:: images/diffraction_radial_profile.png
   :align: center
   :width: 400

   Automatically generated direct beam mask.


Direct beam masking
-------------------

A signal mask that excludes pixels in the SED patterns containing the direct
beam can be generated automatically using the
py:meth:`~._signals.electron_diffraction.get_direct_beam_mask()` method. This can be useful for
visualisation if the direct beam is much more intense than diffracted beams and
can alleviate issues associated with saturation of the direct beam that may
affect further analysis.

The py:meth:`~._signals.electron_diffraction.get_direct_beam_mask()` method estimates the direct
beam position in each SED pattern using the
py:meth:`~._signals.electron_diffraction.get_direct_beam_position()` method and masks a
circular region around that position with a user specified radius, as follows:

.. code-block:: python

    >>> dp = hs.load("sed_nanowire.hdf5", signal_type="electron_diffraction")
    >>> dpmask = dp * dp.get_direct_beam_mask(radius=6)
    >>> dpmask.plot()

.. figure:: images/direct_beam_mask.png
   :align: center
   :width: 400

   Automatically generated direct beam mask.


Vacuum masking
--------------

A navigation mask to exclude electron diffraction patterns acquired in vacuum
from further analysis can be generated using the
py:meth:`~._signals.electron_diffraction.get_vacuum_mask()` method. Ignoring
these patterns, which do not contain useful information, in later analysis is
efficient in terms of computation time and can improve machine learning results.
The method crudely determines whether a SED pattern was acquired in vacuum by
assessing whether or not any diffraction peaks exist in the region excluding the
direct beam. This is based on a user defined threshold for the maximum value
after the patterns have been masked using the
py:meth:`~._signals.electron_diffraction.get_direct_beam_mask()` method.

.. code-block:: python

    >>> dp = hs.load("sed_nanowire.hdf5", signal_type="electron_diffraction")
    >>> dp.get_vacuum_mask()

.. figure:: images/diffraction_vacuum_mask.png
   :align: center
   :width: 400

   Automatically generated mask excluding SED patterns acquired in vacuum.


'Virtual' diffraction imaging
-----------------------------

'Virtual' diffraction imaging involves plotting the intensity of a sub-set of
pixels in each electron diffraction pattern comprising a scanning electron
diffraction dataset, as a function of probe position. In this way, variations in
the diffraction condition are mapped. Forming such 'virtual' diffraction images
in HyperSpy is easy using the 'interactive' and 'ROI' functionality of the
signal class as follows:

.. code-block:: python

    >>> dp = hs.load("sed_nanowire.hdf5", signal_type="electron_diffraction")
    >>>

.. figure:: images/sed_diffraction_imaging.png
   :align: center
   :width: 400


Machine learning SED data
-------------------------

Machine learning decomposition approaches can be applied to scanning electron
diffraction data [Eggeman2015]_. When applying these methods it may be useful to
mask the direct beam or regions of vacuum from the data. A specialised method is
therefore implemented to provide easy access to these options. If float values
are passed as the signal_mask and/or navigation_mask then masks are generated
and applied using the py:meth:`~._signals.electron_diffraction.get_direct_beam_mask()` and
py:meth:`~._signals.electron_diffraction.get_vacuum_mask()` methods respectively.

.. code-block:: python

    >>> dp = hs.load("sed_nanowire.hdf5", signal_type="electron_diffraction")
    >>> dp.decomposition(signal_mask=5., navigation_mask=75.)
