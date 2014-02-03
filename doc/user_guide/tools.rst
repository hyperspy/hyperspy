
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
of data, e.g. the :py:class:`~._signals.spectrum.Spectrum` class provides common
functionality to deal with spectral data and
:py:class:`~._signals.eels.EELSSpectrum` (which is a subclass of
:py:class:`~._signals.spectrum.Spectrum`) adds extra functionality to the
:py:class:`~._signals.spectrum.Spectrum` class for electron energy-loss
spectroscopy data analysis.

Currently the following signal subclasses are available:

* :py:class:`~._signals.spectrum.Spectrum`
* :py:class:`~._signals.image.Image`
* :py:class:`~._signals.eels.EELSSpectrum`
* :py:class:`~._signals.eds_tem.EDSTEMSpectrum`
* :py:class:`~._signals.eds_sem.EDSSEMSpectrum`
* :py:class:`~._signals.spectrum_simulation.SpectrumSimulation`
* :py:class:`~._signals.image_simulation.ImageSimulation`

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
    :py:class:`~._signals.spectrum.Spectrum` or :py:class:`~._signals.image.Image` 
    subclass using the following :py:class:`~.signal.Signal` methods: 
    :py:meth:`~.signal.Signal.as_image`, * :py:meth:`~.signal.Signal.as_spectrum`.
    In addition :py:class:`~._signals.spectrum.Spectrum` instances can be 
    transformed in images using :py:meth:`~._signals.spectrum.Spectrum.to_image` 
    and image instances in spectrum instances using 
    :py:meth:`~._signals.image.Image.to_spectrum`. When transforming between 
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

    +---------------------------------------------------------------+-----------+-------------+---------------+
    |                       Signal subclass                         | record_by | signal_type | signal_origin |
    +===============================================================+===========+=============+===============+
    |                 :py:class:`~.signal.Signal`                   |     -     |      -      |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |           :py:class:`~._signals.spectrum.Spectrum`            | spectrum  |      -      |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    | :py:class:`~._signals.spectrum_simulation.SpectrumSimulation` | spectrum  |      -      |  simulation   |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |           :py:class:`~._signals.eels.EELSSpectrum`            | spectrum  |    EELS     |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |           :py:class:`~._signals.eds_sem.EDSSEMSpectrum`       | spectrum  |   EDS_SEM   |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |           :py:class:`~._signals.eds_tem.EDSTEMSpectrum`       | spectrum  |   EDS_TEM   |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |              :py:class:`~._signals.image.Image`               |   image   |      -      |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |    :py:class:`~._signals.image_simulation.ImageSimulation`    |   image   |      -      |  simulation   |
    +---------------------------------------------------------------+-----------+-------------+---------------+


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
    >>> s.inav[0,0].data
    array([0, 1, 2, 3])
    >>> s.inav[0,0].axes_manager
    <Axes manager, axes: (<t axis, size: 4>,)>
    >>> s.isig[0]
    <Spectrum, title: , dimensions: (2, 3)>
    >>> s.isig[0].axes_manager
    <Axes manager, axes: (<x axis, size: 3, index: 0>, <y axis, size: 2, index: 0>)>
    >>> s.isig[0].data
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
    >>> s.inav[0,0].data
    array([0, 1, 2, 3])
    >>> s.inav[0,0] = 1
    >>> s.inav[0,0].data
    array([1, 1, 1, 1])
    >>> s.inav[0,0] = s[1,1]
    >>> s.inav[0,0].data
    array([16, 17, 18, 19])


       
.. _signal.operations:
       
Signal operations
^^^^^^^^^^^^^^^^^
.. versionadded:: 0.6

:py:class:`~.signal.Signal` supports all the Python binary arithmetic
opearations (+, -, \*, //, %, divmod(), pow(), \*\*, <<, >>, &, ^, \|),
augmented binary assignments (+=, -=, \*=, /=, //=, %=, \*\*=, <<=, >>=, 
&=, ^=, \|=), unary operations (-, +, abs() and ~) and rich comparisons 
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

Iterating over the navigation axes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signal instances are iterables over the navigation axes. For example, the 
following code creates a stack of 10 images and saves them in separate "png"
files by iterating over the signal instance:

.. code-block:: python

    >>> image_stack = signals.Image(np.random.random((2, 5, 64,64)))
    >>> for single_image in image_stack:
    ...    single_image.save("image %s.png" % str(image_stack.axes_manager.indices))
    The "image (0, 0).png" file was created.
    The "image (1, 0).png" file was created.
    The "image (2, 0).png" file was created.
    The "image (3, 0).png" file was created.
    The "image (4, 0).png" file was created.
    The "image (0, 1).png" file was created.
    The "image (1, 1).png" file was created.
    The "image (2, 1).png" file was created.
    The "image (3, 1).png" file was created.
    The "image (4, 1).png" file was created.

The data of the signal instance that is returned at each iteration is a view of
the original data, a property that we can use to perform operations on the
data.  For example, the following code rotates the image at each coordinate  by
a given angle and uses the :py:func:`~.utils.stack` function in combination
with `list comprehensions
<http://docs.python.org/2/tutorial/datastructures.html#list-comprehensions>`_
to make a horizontal "collage" of the image stack:

.. code-block:: python

    >>> import scipy.ndimage
    >>> image_stack = signals.Image(np.array([scipy.misc.lena()]
    >>> image_stack.axes_manager[1].name = "x"
    >>> image_stack.axes_manager[2].name = "y"))
    >>> for image, angle in zip(image_stack, (0, 45, 90, 135, 180)):
    ...    image.data[:] = scipy.ndimage.rotate(image.data, angle=angle,
    ...    reshape=False)
    >>> collage = utils.stack([image for image in image_stack], axis=0)
    >>> collage.plot()

.. figure::  images/rotate_lena.png
  :align:   center
  :width:   500    




.. versionadded:: 0.7


Transforming the data at each coordinate as in the previous example using an
external function can be more easily accomplished using the
:py:meth:`~.signal.Signal.apply_function` method:

.. code-block:: python

    >>> import scipy.ndimage
    >>> image_stack = signals.Image(np.array([scipy.misc.lena()]*4))
    >>> image_stack.axes_manager[1].name = "x"
    >>> image_stack.axes_manager[2].name = "y"
    >>> image_stack.apply_function(scipy.ndimage.rotate,
    ...                            angle=45,
    ...                            reshape=False)
    >>> collage = utils.stack([image for image in image_stack], axis=0)
    >>> collage.plot()

.. figure::  images/rotate_lena_apply_simple.png
  :align:   center
  :width:   500    

The :py:meth:`~.signal.Signal.apply_function` method can also take variable 
arguments as in the following example.

.. code-block:: python

    >>> import scipy.ndimage
    >>> image_stack = signals.Image(np.array([scipy.misc.lena()]*4))
    >>> image_stack.axes_manager[1].name = "x"
    >>> image_stack.axes_manager[2].name = "y"
    >>> angles = signals.Signal(np.array([0, 45, 90, 135]))
    >>> angles.axes_manager.set_signal_dimension(0)
    >>> modes = signals.Signal(np.array(['constant', 'nearest', 'reflect', 'wrap']))
    >>> modes.axes_manager.set_signal_dimension(0)
    >>> image_stack.apply_function(scipy.ndimage.rotate,
    ...                            angle=angles,
    ...                            reshape=False,
    ...                            mode=modes)
    calculating 100% |#############################################| ETA:  00:00:00Cropping

.. figure::  images/rotate_lena_apply_ndkwargs.png
  :align:   center
  :width:   500    

Cropping
^^^^^^^^

Cropping can be performed in a very compact and powerful way using
:ref:`signal.indexing` . In addition it can be performed using the following
method or GUIs if cropping :ref:`spectra <spectrum.crop>` or :ref:`images
<image.crop>`. There is also a general :py:meth:`~.signal.Signal.crop`
method that operates *in place*.

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
* :py:meth:`~.signal.Signal.integrate_simpson`

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


Basic statistical analysis
--------------------------
.. versionadded:: 0.7

:py:meth:`~.signal.Signal.get_histogram` computes the histogram and
conveniently returns it as signal instance. It provides methods to
calculate the bins. :py:meth:`~.signal.Signal.print_summary_statistics` prints
the five-number summary statistics of the data. 

These two methods can be combined with
:py:meth:`~.signal.Signal.get_current_signal` to compute the histogram or
print the summary stastics of the signal at the current coordinates, e.g:
.. code-block:: python

    >>> s = signals.EELSSpectrum(np.random.normal(size=(10,100)))
    >>> s.print_summary_statistics()
    Summary statistics
    ------------------
    mean:	0.021
    std:	0.957
    min:	-3.991
    Q1:	-0.608
    median:	0.013
    Q3:	0.652
    max:	2.751
     
    >>> s.get_current_signal().print_summary_statistics()
    Summary statistics
    ------------------
    mean:   -0.019
    std:    0.855
    min:    -2.803
    Q1: -0.451
    median: -0.038
    Q3: 0.484
    max:    1.992


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

The :py:meth:`~.signal.Signal1DTools.remove_background` method provides
background removal capabilities through both a CLI and a GUI. Current
background type supported are power law, offset, polynomial and gaussian.

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

.. _integrate_1D-label:

Integration
-----------
The :py:meth:`~.signal.Signal1DTools.integrate_in_range` method provides a GUI
and a CLI to integrate the 1D signal dimension in a given range using the
Simpson's rule. The GUI operates in-place while the CLI opearation is
not-in-place. 

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

.. _image.crop:

Cropping an image
^^^^^^^^^^^^^^^^^
In addition to cropping using the powerful and compact :ref:`signal.indexing`
the following method is available to crop spectra the familiar 
top, bottom, left, right syntax.

* :py:meth:`~.signal.Signal2DTools.crop_image`


EELS tools
----------

These methods are only available for the following signals:

* :py:class:`~._signals.eels.EELSSpectrum`

Spikes removal
^^^^^^^^^^^^^^
.. versionadded:: 0.5
    The :py:meth:`~._signals.eels.EELSSpectrum.spikes_removal_tool` replaces the
    old :py:meth:`~._signals.eels.EELSSpectrum.remove_spikes`.


:py:meth:`~._signals.eels.EELSSpectrum.spikes_removal_tool` provides an user
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
:py:meth:`~._signals.eels.EELSSpectrum.add_elements`. The information is stored
in the :py:attr:`~.signal.Signal.mapped_parameters` attribute (see
:ref:`mapped_parameters_structure`)

Estimate the FWHM of a peak
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~._signals.eels.EELSSpectrum.estimate_FWHM`

Estimate the thickness
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. _eds_tools-label:

EDS tools
---------

.. versionadded:: 0.7

These methods are only available for the following signals:

* :py:class:`~._signals.eds_tem.EDSTEMSpectrum`
* :py:class:`~._signals.eds_sem.EDSSEMSpectrum`


Set elements
^^^^^^^^^^^^

The :py:meth:`~._signals.eds.EDSSpectrum.set_elements` method is used 
to define a set of elements and corresponding X-ray lines
that will be used in other process (e.g. X-ray intensity mapping).
The information is stored in the :py:attr:`~.signal.Signal.mapped_parameters` attribute (see :ref:`mapped_parameters_structure`)


Add elements
^^^^^^^^^^^^

When the set_elements method erases all previously defined elements, 
the :py:meth:`~._signals.eds.EDSSpectrum.add_elements` method adds a new
set of elements to the previous set.


Get intensity map
^^^^^^^^^^^^^^^^^

With the :py:meth:`~._signals.eds.EDSSpectrum.get_lines_intensity`, the 
intensity of X-ray lines is used to generate a map. The number of counts
under the selected peaks is used.

Set microscope parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~._signals.eds_tem.EDSTEMSpectrum.set_microscope_parameters` method provides an user 
interface to calibrate the parameters if the microscope and the EDS detector.

Get the calibration from another spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~._signals.eds_tem.EDSTEMSpectrum.get_calibration_from`

Dielectric function tools
-------------------------

.. versionadded:: 0.7

Number of effective electrons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 0.7

The Bethe f-sum rule gives rise to two definitions of the effective number (see
[Egerton2011]_):

.. math::

   n_{\mathrm{eff1}}\left(-\Im\left(\epsilon^{-1}\right)\right)=\frac{2\epsilon_{0}m_{0}}{\pi\hbar^{2}e^{2}n_{a}}\int_{0}^{E}E'\Im\left(\frac{-1}{\epsilon}\right)dE'

   n_{\mathrm{eff2}}\left(\epsilon_{2}\right)=\frac{2\epsilon_{0}m_{0}}{\pi\hbar^{2}e^{2}n_{a}}\int_{0}^{E}E'\epsilon_{2}\left(E'\right)dE'
 
where :math:`n_a` is the number of atoms (or molecules) per unit volume of the
sample, :math:`\epsilon_0` is the vacuum permittivity, :math:`m_0` is the
elecron mass and :math:`e` is the electron charge.

The
:py:meth:`~._signals.dielectric_function.DielectricFunction.get_number_of_effective_electrons`
method computes both.

Compute the electron energy-loss signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 0.7

The
:py:meth:`~._signals.dielectric_function.DielectricFunction.get_electron_energy_loss_spectrum`
"naively" computes the single-scattering electron-energy loss spectrum from the
dielectric function given the zero-loss peak (or its integral) and the sample
thickness using:

.. math::

    S\left(E\right)=\frac{2I_{0}t}{\pi
    a_{0}m_{0}v^{2}}\ln\left[1+\left(\frac{\beta}{\theta(E)}\right)^{2}\right]\Im\left[\frac{-1}{\epsilon\left(E\right)}\right]
     
where :math:`I_0` is the zero-loss peak integral, :math:`t` the sample
thickness, :math:`\beta` the collection angle and :math:`\theta(E)` the
characteristic scattering angle.

Electron and X-ray range
^^^^^^^^^^^^^^^^^^^^^^^^

The electron and X-ray range in a bulk material can be estimated with 
:py:meth:`~.utils.eds.electron_range` and :py:meth:`~.utils.eds.xray_range`
