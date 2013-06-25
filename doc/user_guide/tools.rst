
Tools
*****

The Signal class and its subclasses
-----------------------------------

.. WARNING::
    This subsection can be a bit confusing for beginners.
    Do not worry if you do not understand it all.
    

Hyperspy stores hyperspectra in the :py:class:`~.signal.Signal` class, that is
the object that you get when e.g. you load a single file using
:py:func:`~.io.load`. Most of the data analysis functions are also contained in
this class or its specialized subclasses. The :py:class:`~.signal.Signal` class
contains general functionality that is available to all the subclasses. The
subclasses provide functionality that is normally specific to a particular type
of data, e.g. the :py:class:`~.signals.spectrum.Spectrum` class provides common
functionality to deal with spectral data and
:py:class:`~.signals.eels.EELSSpectrum` (which is a subclass of
:py:class:`~.signals.spectrum.Spectrum`) adds extra functionality to the
:py:class:`~.signals.spectrum.Spectrum` class for electron energy-loss
spectroscopy data analysis.

Currently the following signal subclasses are available:

* :py:class:`~.signals.spectrum.Spectrum`
* :py:class:`~.signals.image.Image`
* :py:class:`~.signals.eels.EELSSpectrum`
* :py:class:`~.signals.eds_tem.EDSTEMSpectrum`
* :py:class:`~.signals.eds_sem.EDSSEMSpectrum`
* :py:class:`~.signals.spectrum_simulation.SpectrumSimulation`
* :py:class:`~.signals.image_simulation.ImageSimulation`

The :py:mod:`~.signals` module, which contains all available signal subclasses,
is imported in the user namespace when loading hyperspy. In the following
example we create an Image instance from a 2D numpy array:

.. code-block:: python
    
    >>> im = signals.Image(np.random.random((64,64)))
    

The different signals store other objects in what are called attributes. For
examples, the hyperspectral data is stored in the
:py:attr:`~.signal.Signal.data` attribute, the original parameters in the
:py:attr:`~.signal.Signal.original_parameters` attribute, the mapped parameters
in the :py:attr:`~.signal.Signal.mapped_parameters` attribute and the axes
information (including calibration) can be accessed (and modified) in the
:py:attr:`~.signal.Signal.axes_manager` attribute.

.. _transforming.signal:

Transforming between signal subclasses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The different subclasses are characterized by three
:py:attr:`~.signal.Signal.mapped_parameters` attributes (see the table below):

`record_by`
    Can be "spectrum", "image" or "", the latter meaning undefined.
    It describes the way the data is arranged in memory.
    It is possible to transform any :py:class:`~.signal.Signal` subclass in a 
    :py:class:`~.signals.spectrum.Spectrum` or :py:class:`~.signals.image.Image` 
    subclass using the following :py:class:`~.signal.Signal` methods: 
    :py:meth:`~.signal.Signal.as_image`, * :py:meth:`~.signal.Signal.as_spectrum`.
    In addition :py:class:`~.signals.spectrum.Spectrum` instances can be 
    transformed in images using :py:meth:`~.signals.spectrum.Spectrum.to_image` 
    and image instances in spectrum instances using 
    :py:meth:`~.signals.image.Image.to_spectrum`. When transforming between 
    spectrum and image classes the order in which the
    data array is stored in memory is modified to improve performance and several
    functions, e.g. plotting or decomposing, will behave differently.
    
`signal_type`
    Describes the nature of the signal. It can be any string, normally the 
    acronym associated with a
    particular signal. In certain cases Hyperspy provides features that are 
    only available for a 
    particular signal type through :py:class:`~.signal.Signal` subclasses.
    The :py:class:`~.signal.Signal` method 
    :py:meth:`~.signal.Signal.set_signal_type`
    changes the signal_type in place, what may result in a 
    :py:class:`~.signal.Signal`
    subclass transformation.
    
`signal_origin`
    Describes the origin of the signal and can be "simulation" or 
    "experiment" or "",
    the latter meaning undefined. In certain cases Hyperspy provides features 
    that are only available for a 
    particular signal origin. The :py:class:`~.signal.Signal` method 
    :py:meth:`~.signal.Signal.set_signal_origin`
    changes the signal_origin in place, what may result in a 
    :py:class:`~.signal.Signal`
    subclass transformation.
    
.. table:: Signal subclass :py:attr:`~.signal.Signal.mapped_parameters` attributes.

    +---------------------------------------------------------------+------------+--------------+---------------+
    |                       Signal subclass                         | record_by  | signal_type  | signal_origin |
    +===============================================================+============+==============+===============+
    |                 :py:class:`~.signal.Signal`                   |     -      |      -       |       -       |
    +---------------------------------------------------------------+------------+--------------+---------------+
    |           :py:class:`~.signals.spectrum.Spectrum`             | spectrum   |      -       |       -       |
    +---------------------------------------------------------------+------------+--------------+---------------+
    | :py:class:`~.signals.spectrum_simulation.SpectrumSimulation`  | spectrum   |      -       |  simulation   |
    +---------------------------------------------------------------+------------+--------------+---------------+
    |           :py:class:`~.signals.eels.EELSSpectrum`             | spectrum   |    EELS      |       -       |
    +---------------------------------------------------------------+------------+--------------+---------------+
    |              :py:class:`~.signals.image.Image`                |   image    |      -       |       -       |
    +---------------------------------------------------------------+------------+--------------+---------------+
    |    :py:class:`~.signals.image_simulation.ImageSimulation`     |   image    |      -       |  simulation   |
    +---------------------------------------------------------------+------------+--------------+---------------+


The following example shows how to transform between different subclasses.

.. code-block:: python
    
    >>> s = signals.Spectrum(np.random.random((10,20,100)))
    >>> s
    <Spectrum, title: , dimensions: (20, 10, 100)>
    >>> s.mapped_parameters
    ├── record_by = spectrum
    └── title = 
    
    >>> im = s.to_image()
    >>> im
    <Image, title: , dimensions: (20, 10, 100)>
    >>> im.ma
    im.mapped_parameters  im.max                
    >>> im.mapped_parameters
    ├── record_by = image
    └── title = 
    
    >>> s.set_signal_type("EELS")
    >>> s
    <EELSSpectrum, title: , dimensions: (20, 10, 100)>
    >>> s.mapped_parameters
    ├── record_by = spectrum
    ├── signal_type = EELS
    └── title = 
    
    >>> s.set_signal_origin("simulation")
    >>> s
    <EELSSpectrumSimulation, title: , dimensions: (20, 10, 100)>
    >>> s.mapped_parameters
    ├── record_by = spectrum
    ├── signal_origin = simulation
    ├── signal_type = EELS
    └── title = 



The navigation and signal dimensions
------------------------------------

Hyperspy can deal with data of arbitrary dimensions. Each dimension is internally
classified as either "navigation" or "signal" and the 
way this classification is done determines the behaviour of the signal.

The concept is probably best understood with 
an example: let's imagine a three dimensional dataset. This dataset 
could be an spectrum image acquired by scanning over a sample in two 
dimensions. In Hyperspy's terminology the spectrum dimension would be 
the signal dimension and the two other dimensions would be the navigation 
dimensions. We could see the same dataset as an image stack instead. 
Actually it could has been acquired by capturing two
dimensional images at different wavelenghts. Then it would be natural 
to identify the two spatial dimensions as the signal dimensions and 
the wavelenght dimension as the navigation dimension. 
However, for data analysis purposes, one may like to operate with an image stack 
as if it was a set of spectra or viceversa. One can easily switch between these 
two alternative ways of classifiying the dimensions of a three-dimensional 
dataset by 
:ref:`transforming between Spectrum and Image subclasses <transforming.signal>`.

.. NOTE::
    Although each dimension can be arbitrarily classified as "navigation dimension"
    or "signal dimension", for most common tasks there is no need to modify 
    Hyperspy's default choice.


Generic tools
-------------

Below we briefly introduce some of the most commonly used tools (methods). For
more details about a particular method click on its name. For a detailed list
of all the methods available see the :py:class:`~.signal.Signal` documentation.

The methods of this section are available to all the signals. In the subsections
we describe methods that are only available in specialized subclasses.

.. _signal.indexing:

Indexing
^^^^^^^^
.. versionadded:: 0.6

Indexing the :py:class:`~.signal.Signal`  provides a
powerful, convenient and Pythonic way to access and modify its data.
It is a concept that might take some time to grasp but, once 
mastered, it can greatly simplify many common
signal processing tasks.
 
Indexing refers to any use of the square brackets ([]) to index the
data stored in a :py:class:`~.signal.Signal`. The result of indexing 
a :py:class:`~.signal.Signal` is another :py:class:`~.signal.Signal` 
that shares a subset of the data of the original :py:class:`~.signal.Signal`.
 
 
Hyperspy's Signal indexing is similar to numpy array indexing and, therefore,
rather that explaining this feature in detail we will just give some examples
of usage here. The interested reader is encouraged to read the `numpy
documentation on the subject  <http://ipython.org/>`_ for a detailed
explanation of the concept. When doing so it is worth to keep in mind the
following main differences:

* Hyperspy (unlike numpy) does not support:

    * Indexing using arrays.  * Adding new axes using the newaxis object.
    
* Hyperspy (unlike numpy):

    * Supports indexing with decimal numbers.  * Uses the natural order when
      indexing i.e. [x, y, z,...] (hyperspy) vs [...,z,y,x] (numpy)
    
Lets start by indexing a single spectrum:


.. code-block:: python
    
    >>> s = signals.Spectrum(np.arange(10))
    >>> s
    <Spectrum, title: , dimensions: (10,)>
    >>> s.data
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> s[0]
    <Spectrum, title: , dimensions: (1,)>
    >>> s[0].data
    array([0])
    >>> s[9].data
    array([9])
    >>> s[-1].data
    array([9])
    >>> s[:5]
    <Spectrum, title: , dimensions: (5,)>
    >>> s[:5].data
    array([0, 1, 2, 3, 4])
    >>> s[5::-1]
    <Spectrum, title: , dimensions: (6,)>
    >>> s[5::-1]
    array([5, 4, 3, 2, 1, 0])
    >>> s[5::2]
    <Spectrum, title: , dimensions: (3,)>
    >>> s[5::2].data
    array([5, 7, 9])   
    

Unlike numpy, Hyperspy supports indexing using decimal numbers, in which case
Hyperspy indexes using the axis scales instead of the indices.
 
.. code-block:: python

    >>> s = signals.Spectrum(np.arange(10))
    >>> s
    <Spectrum, title: , dimensions: (10,)>
    >>> s.data
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> s.axes_manager[0].scale = 0.5
    >>> s.axes_manager[0].axis
    array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5])
    >>> s[0.5:4.].data
    array([1, 2, 3, 4, 5, 6, 7])
    >>> s[0.5:4].data
    array([1, 2, 3])
    >>> s[0.5:4:2].data
    array([1, 3])


Importantly the original :py:class:`~.signal.Signal` and its "indexed self"
share their data and, therefore, modifying the value of the data in one
modifies the same value in the other.

.. code-block:: python

    >>> s = signals.Spectrum(np.arange(10))
    >>> s
    <Spectrum, title: , dimensions: (10,)>
    >>> s.data
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> si = s[::2]
    >>> si.data
    array([0, 2, 4, 6, 8])
    >>> si.data[:] = 10
    >>> si.data
    array([10, 10, 10, 10, 10])
    >>> s.data
    array([10,  1, 10,  3, 10,  5, 10,  7, 10,  9])
    >>> s.data[:] = 0
    >>> si.data
    array([0, 0, 0, 0, 0])
    

Of course it is also possible to use the same syntax to index multidimensional
data.  The first indexes are always the navigation indices in "natural order"
i.e. x,y,z...  and the following indexes are the signal indices also in natural
order.
    
.. code-block:: python
    
    >>> s = signals.Spectrum(np.arange(2*3*4).reshape((2,3,4)))
    >>> s
    <Spectrum, title: , dimensions: (10, 10, 10)>
    >>> s.data
    array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
    >>> s.axes_manager[0].name = 'x'
    >>> s.axes_manager[1].name = 'y'
    >>> s.axes_manager[2].name = 't'
    >>> s.axes_manager.signal_axes
    (<t axis, size: 4>,)
    >>> s.axes_manager.navigation_axes
    (<x axis, size: 3, index: 0>, <y axis, size: 2, index: 0>)
    >>> s[0,0].data
    array([0, 1, 2, 3])
    >>> s[0,0].axes_manager
    <Axes manager, axes: (<t axis, size: 4>,)>
    >>> s[0,0,::-1].data
    array([3, 2, 1, 0])
    >>> s[...,0]
    <Spectrum, title: , dimensions: (2, 3)>
    >>> s[...,0].axes_manager
    <Axes manager, axes: (<x axis, size: 3, index: 0>, <y axis, size: 2, index: 0>)>
    >>> s[...,0].data
    array([[ 0,  4,  8],
       [12, 16, 20]])
       
For convenience and clarity it is possible to index the signal and navigation
dimensions independently:

.. code-block:: python
    
    >>> s = signals.Spectrum(np.arange(2*3*4).reshape((2,3,4)))
    >>> s
    <Spectrum, title: , dimensions: (10, 10, 10)>
    >>> s.data
    array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
    >>> s.axes_manager[0].name = 'x'
    >>> s.axes_manager[1].name = 'y'
    >>> s.axes_manager[2].name = 't'
    >>> s.axes_manager.signal_axes
    (<t axis, size: 4>,)
    >>> s.axes_manager.navigation_axes
    (<x axis, size: 3, index: 0>, <y axis, size: 2, index: 0>)
    >>> s.navigation_indexer[0,0].data
    array([0, 1, 2, 3])
    >>> s.navigation_indexer[0,0].axes_manager
    <Axes manager, axes: (<t axis, size: 4>,)>
    >>> s.signal_indexer[0]
    <Spectrum, title: , dimensions: (2, 3)>
    >>> s.signal_indexer[0].axes_manager
    <Axes manager, axes: (<x axis, size: 3, index: 0>, <y axis, size: 2, index: 0>)>
    >>> s.signal_indexer[0].data
    array([[ 0,  4,  8],
       [12, 16, 20]])
       

The same syntax can be used to set the data values:

.. code-block:: python
    
    >>> s = signals.Spectrum(np.arange(2*3*4).reshape((2,3,4)))
    >>> s
    <Spectrum, title: , dimensions: (10, 10, 10)>
    >>> s.data
    array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
    >>> s.navigation_indexer[0,0].data
    array([0, 1, 2, 3])
    >>> s.navigation_indexer[0,0] = 1
    >>> s.navigation_indexer[0,0].data
    array([1, 1, 1, 1])
    >>> s.navigation_indexer[0,0] = s[1,1]
    >>> s.navigation_indexer[0,0].data
    array([16, 17, 18, 19])


       
.. _signal.operations:
       
Signal operations
^^^^^^^^^^^^^^^^^
.. versionadded:: 0.6

:py:class:`~.signal.Signal` supports all the Python binary arithmetic
opearations (+, -, *, //, %, divmod(), pow(), **, <<, >>, &, ^, |),
augmented binary assignments (+=, -=, *=, /=, //=, %=, **=, <<=, >>=, 
&=, ^=, |=), unary operations (-, +, abs() and ~) and rich comparisons 
operations (<, <=, ==, x!=y, <>, >, >=).

These operations are performed element-wise. When the dimensions of the signals
are not equal `numpy broadcasting rules apply
<http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ *first*. In
addition Hyperspy extend numpy's broadcasting rules to the following cases:



+------------+----------------------+------------------+
| **Signal** | **NavigationShape**  | **SignalShape**  |
+============+======================+==================+
|   s1       |        a             |      b           |
+------------+----------------------+------------------+
|   s2       |       (0,)           |      a           |
+------------+----------------------+------------------+
|   s1 + s2  |       a              |      b           |
+------------+----------------------+------------------+
|   s2 + s1  |       a              |      b           |
+------------+----------------------+------------------+


+------------+----------------------+------------------+
| **Signal** | **NavigationShape**  | **SignalShape**  |
+============+======================+==================+
|   s1       |        a             |      b           |
+------------+----------------------+------------------+
|   s2       |       (0,)           |      b           |
+------------+----------------------+------------------+
|   s1 + s2  |       a              |      b           |
+------------+----------------------+------------------+
|   s2 + s1  |       a              |      b           |
+------------+----------------------+------------------+


+------------+----------------------+------------------+
| **Signal** | **NavigationShape**  | **SignalShape**  |
+============+======================+==================+
|   s1       |       (0,)           |      a           |
+------------+----------------------+------------------+
|   s2       |       (0,)           |      b           |
+------------+----------------------+------------------+
|   s1 + s2  |       b              |      a           |
+------------+----------------------+------------------+
|   s2 + s1  |       a              |      b           |
+------------+----------------------+------------------+


Cropping
^^^^^^^^

Cropping can be performed in a very compact and powerful way using 
:ref:`signal.indexing` . In addition it can be performed using the 
following method or GUIs if cropping :ref:`spectra <>` or 
:ref:`images <>`

* :py:meth:`~.signal.Signal.crop`

Rebinning
^^^^^^^^^

The :py:meth:`~.signal.Signal.rebin` method rebins data in place down to a size
determined by the user.

Folding and unfolding
^^^^^^^^^^^^^^^^^^^^^

When dealing with multidimensional datasets it is sometimes useful to transform
the data into a two dimensional dataset. This can be accomplished using the
following two methods:

* :py:meth:`~.signal.Signal.fold`
* :py:meth:`~.signal.Signal.unfold`

It is also possible to unfold only the navigation or only the signal space:

* :py:meth:`~.signal.Signal.unfold_navigation_space`
* :py:meth:`~.signal.Signal.unfold_signal_space`

Simple operations over one axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signal.Signal.sum`
* :py:meth:`~.signal.Signal.mean`
* :py:meth:`~.signal.Signal.max`
* :py:meth:`~.signal.Signal.min`
* :py:meth:`~.signal.Signal.std`
* :py:meth:`~.signal.Signal.var`
* :py:meth:`~.signal.Signal.diff`

Changing the data type
^^^^^^^^^^^^^^^^^^^^^^

Even if the original data is recorded with a limited dynamic range, it is often
desirable to perform the analysis operations with a higher precision.
Conversely, if space is limited, storing in a shorter data type can decrease
the file size. The :py:meth:`~.signal.Signal.change_dtype` changes the data
type in place, e.g.:

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

These methods are only available for Signal object with signal_dimension 
equal to one.

.. _spectrum.crop:

Cropping
^^^^^^^^

In addition to cropping using the powerful and compact 
:ref:`Signal indexing <signal.indexing>` syntax
the following method is available to crop spectra using a GUI:

The :py:meth:`~.signal.Signal1DTools.crop_spectrum`, method is used to crop the
spectral energy range. If no parameter is passed, a user interface appears in
which to crop the spectrum.

Background removal
^^^^^^^^^^^^^^^^^^

The :py:meth:`~.signal.Signal1DTools.remove_background` method provides a user
interface to remove some background functions.

Calibration
^^^^^^^^^^^
The :py:meth:`~.signal.Signal1DTools.calibrate` method provides a user
interface to calibrate the spectral axis.

Aligning
^^^^^^^^

The following methods use sub-pixel cross-correlation or user-provided shifts
to align spectra. They support applying the same transformation to multiple
files.

* :py:meth:`~.signal.Signal1DTools.align1D`
* :py:meth:`~.signal.Signal1DTools.shift1D`


Data smoothing
^^^^^^^^^^^^^^

The following methods (that include user interfaces when no arguments are
passed) can perform data smoothing with different algorithms:

* :py:meth:`~.signal.Signal1DTools.smooth_lowess`
* :py:meth:`~.signal.Signal1DTools.smooth_tv`
* :py:meth:`~.signal.Signal1DTools.smooth_savitzky_golay`

Other methods
^^^^^^^^^^^^^^


* Apply a hanning taper to the spectra 
  :py:meth:`~.signal.Signal1DTools.hanning_taper`
* Find peaks in spectra 
  :py:meth:`~.signal.Signal1DTools.find_peaks1D_ohaver`
* Interpolate the spectra in between two positions 
  :py:meth:`~.signal.Signal1DTools.interpolate_in_between`
* Convolve the spectra with a gaussian 
  :py:meth:`~.signal.Signal1DTools.gaussian_filter`



Image tools
-----------

These methods are only available for Signal object with signal_dimension 
equal to two.

Image registration (alignment)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 0.5

The :py:meth:`~.signal.Signal2DTools.align2D` method provides advanced image
alignment functionality, including subpixel alignment.

.. image.crop:

Cropping an image
^^^^^^^^^^^^^^^^^
In addition to cropping using the powerful and compact :ref:`signal.indexing`
the following method is available to crop spectra the familiar 
top, bottom, left, right syntax.

* :py:meth:`~.signal.Signal2DTools.crop_image`


EELS tools
----------

These methods are only available for the following signals:

* :py:class:`~.signals.eels.EELSSpectrum`

Spikes removal
^^^^^^^^^^^^^^
.. versionadded:: 0.5
    The :py:meth:`~.signals.eels.EELSSpectrum.spikes_removal_tool` replaces the
    old :py:meth:`~.signals.eels.EELSSpectrum.remove_spikes`.


:py:meth:`~.signals.eels.EELSSpectrum.spikes_removal_tool` provides an user
interface to remove spikes from spectra.


.. figure::  images/spikes_removal_tool.png
   :align:   center
   :width:   500    

   Spikes removal tool


Define the elemental composition of the sample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It can be useful to define the composition of the sample for archiving purposes
or for some other process (e.g. curve fitting) that may use this information.
The elemental composition of the sample can be defined using
:py:meth:`~.signals.eels.EELSSpectrum.add_elements`. The information is stored
in the :py:attr:`~.signal.Signal.mapped_parameters` attribute (see
:ref:`mapped_parameters_structure`)

Estimate the FWHM of a peak
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signals.eels.EELSSpectrum.estimate_FWHM`

Estimate the thickness
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~.signals.eels.EELSSpectrum.estimate_thickness` can estimate the
thickness from a low-loss EELS spectrum.

Estimate zero loss peak centre
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signals.eels.EELSSpectrum.estimate_zero_loss_peak_centre`

Deconvolutions
^^^^^^^^^^^^^^

* :py:meth:`~.signals.eels.EELSSpectrum.fourier_log_deconvolution`
* :py:meth:`~.signals.eels.EELSSpectrum.fourier_ratio_deconvolution`
* :py:meth:`~.signals.eels.EELSSpectrum.richardson_lucy_deconvolution`

Estimate elastic scattering threshold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use
:py:meth:`~.signals.eels.EELSSpectrum.estimate_elastic_scattering_threshold` to
calculate separation point between elastic and inelastic scattering on some
EELS low-loss spectra. This algorithm calculates the derivative of the signal
and assigns the inflexion point to the first point below a certain tolerance.
This tolerance value can be set using the tol keyword.

Currently, the method uses smoothing to reduce the impact of the noise in the
measure. The number of points used for the smoothing window can be specified by
the npoints keyword. 

Estimate elastic scattering intensity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :py:meth:`estimate_elastic_scattering_intensity` to calculate the integral
below the zero loss peak (elastic intensity) from EELS low-loss spectra
containing the zero loss peak. This integral can use the threshold image
calculated by the
:py:meth:`~.signals.eels.EELSSpectrum.estimate_elastic_scattering_threshold` as
end energy for the integration at each spectra or use the same energy value for
all spectra. Also, if no threshold is specified, the routine will perform a
rough estimation of the inflexion values at each spectrum.

Splice zero loss peak
^^^^^^^^^^^^^^^^^^^^^
Once :py:meth:`~.signals.eels.EELSSpectrum.estimate_elastic_scattering_threshold` has determined the elastic scattering threshold value(s), this tool can be used to separate the zero loss peak from the eels spectra. Use :py:meth:`~.signals.eels.EELSSpectrum.splice_zero_loss_peak` in order to obtain a ZLP suitable for Fourier-Log deconvolution from your EELS low-loss spectra by setting the "smooth" option, that will apply the hanning window to the righ end of the data.

EDS tools
----------

These methods are only available for the following signals:

* :py:class:`~.signals.eds_tem.EDSTEMSpectrum`
* :py:class:`~.signals.eds_sem.EDSSEMSpectrum`


Set elements
^^^^^^^^^^^^

The :py:meth:`~.signals.eds.EDSSpectrum.set_elements` method is used 
to define a set of elements and corresponding X-ray lines
that will be used in other process (e.g. X-ray intensity mapping).
The information is stored in the :py:attr:`~.signal.Signal.mapped_parameters` attribute (see :ref:`mapped_parameters_structure`)


Add elements
^^^^^^^^^^^^

When the set_elements method erases all previously defined elements, 
the :py:meth:`~.signals.eds.EDSSpectrum.add_elements` method adds a new
set of elements to the previous set.


Get intensity map
^^^^^^^^^^^^^^^^^

With the :py:meth:`~.signals.eds.EDSSpectrum.get_intensity_map`, the 
intensity of X-ray lines is used to generate a map. The number of counts
under the selected peaks is used.

Set microscope parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~.signals.eds_tem.EDSTEMSpectrum.set_microscope_parameters` method provides an user 
interface to calibrate the paramters if the microscope and the EDS detector.

Get the calibration from another spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signals.eds_tem.EDSTEMSpectrum.get_calibration_from`
