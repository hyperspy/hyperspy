.. _sed-label:

Scanning Electron Diffraction (SED)
***********************************

Methods to analyse electron diffraction data acquired conventional or scanning
mode of a scanning transmission electron microscope (STEM) are described in
this chapter. These methods are specific to the signals in the
:py:class:`~._signals.sed.SEDPattern` class.

.. NOTE::

    See also the `SED tutorials <http://nbviewer.ipython.org/github/hyperspy/hyperspy-demos/blob/master/electron_microscopy/SED/>`_ .

The data used throughout the following examples can be downloaded as follows:

.. code-block:: python

    >>> from urllib import urlretrieve
    >>> url = 'http://cook.msm.cam.ac.uk//~hyperspy//SED_tutorial//'
    >>> urlretrieve(url + 'GaAs_001.tif', 'GaAs_nanowire.hdf5')

Loading SED data & setting microscope parameters
------------------------------------------------

Loading
^^^^^^^^

All data are loaded with the :py:func:`~.io.load` function, as described in
detail in :ref:`Loading files<loading_files>`. HyperSpy is able to import
different formats, among them ".blo" (the format used by NanoMEGAS). Below are
examples for loading a single diffraction pattern and loading a stack of
diffraction patterns.

For a single diffraction pattern:

.. code-block:: python

    >>> dp = hs.load("GaAs_001.tif")
    >>> dp
    <Image, title: Image, dimensions: (|144, 144)>

For a SED dataset here stored as a stack of images:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.hdf5", signal_type="SED_Pattern")
    >>> dp
    <Image, title: , dimensions: (100, 30|144, 144)>


Microscope and detector parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Microscope and detector parameters are stored in the
:py:attr:`~.signal.Signal.metadata` attribute (see :ref:`metadata_structure`).
These parameters can be displayed as follows:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.hdf5", signal_type="SED_Pattern")
    >>> dp.metadata


Parameters can be specified directly:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.hdf5", signal_type="SED_Pattern")
    >>> dp.metadata.Acquisition_instrument.SED.convergence_angle = 5.

or with the
:py:meth:`~._signals.sed.SEDPattern.set_microscope_parameters` method:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.rpl", signal_type="SED_Pattern")
    >>> dp.set_microscope_parameters(convergence_angle = 5.)

or raising the gui:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.rpl", signal_type="SED_Pattern")
    >>> dp.set_microscope_parameters()

.. figure::  images/SED_microscope_parameters_gui.png
   :align:   center
   :width:   400

   SED microscope parameters preferences window.

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


Pre-processing
--------------

Basic pre-processing of SED datasets involves aligning the recorded electron
diffraction patterns such that all have a common center, performing background
subtraction, and masking the direct beam. Methods to achieve perform these
pre-processing steps are available in HyperSpy as described below.


Pattern alignment
^^^^^^^^^^^^^^^^^

Alignment is based on determining the direct beam position, which should be
invariant throughout a stack of electron diffraction patterns. The position of
the direct beam is estimated using the approach described by Zaeferrer
[Ref, Zaeferrer 2000] and White [Ref, White Thesis] as follows:

.. code-block:: python

    >>>
    >>> dp.get_direct_beam_position()


In brief, the py:meth:`~._signals.sed.estimate_direct_beam_position()` method first sums all diffraction patterns in the stack


Applies these shifts using the align2D() method.


Direct beam masking
^^^^^^^^^^^^^^^^^^^

A signal mask that excludes pixels in the SED patterns containing the direct
beam can be generated automatically using the py:meth:`~._signals.sed.direct_beam_mask()`
method. This can be useful for visualisation if the direct beam is much more
intense than diffracted beams and can alleviate issues associated with
saturation of the direct beam that may affect further analysis.

The py:meth:`~._signals.sed.direct_beam_mask()` method estimates the direct beam
position in each SED pattern using the py:meth:`~._signals.sed.estimate_direct_beam_position()`
method and masks a circular region around that position with a user specified
radius, as follows:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.rpl", signal_type="SED_Pattern")
    >>> dpmask = dp * dp.direct_beam_mask(radius=6)
    >>> dpmask.plot()

.. figure:: images/SED_direct_beam_mask.png
   :align: center
   :width: 400

   Automatically generated direct beam mask.


Vacuum masking
^^^^^^^^^^^^^^

A navigation mask to exclude SED patterns acquired in vacuum from further
analysis can be generated automatically using the vacuum_mask() method. Ignoring
these patterns, which do not contain useful information, in later analysis is
efficient in terms of computation time and can improve results from statistical
methods that use all of the selected data.

The vacuum_mask() method automatically determines whether a SED pattern was
acquired in vacuum by assessing whether or not any diffraction peaks exist in
the region that does not contain the direct beam.

The method is applied as follows:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.rpl", signal_type="SED_Pattern")
    >>>

.. figure:: images/SED_vacuum_mask.png
   :align: center
   :width: 400

   Automatically generated mask excluding SED patterns acquired in vacuum.


'Virtual' diffraction imaging
-----------------------------

'Virtual' diffraction imaging involves plotting the intensity of a sub-set of
pixels in each electron diffraction pattern comprising a SED dataset, as a
function of probe position. In this way, variations in the diffraction condition
are mapped. Forming such 'virtual' diffraction images in HyperSpy is easy using
the 'interactive' and 'ROI' functionality of the signal class as follows:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.rpl", signal_type="SED_Pattern")
    >>>

.. figure:: images/SED_vacuum_mask.png
   :align: center
   :width: 400

   Automatically generated mask excluding SED patterns acquired in vacuum.


Machine learning SED data
-------------------------

Machine learning decomposition approaches can be applied to SED data
[Ref, Eggeman et al].
