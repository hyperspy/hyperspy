
Tools: the Signal class
***********************

The Signal class and its subclasses
-----------------------------------

.. WARNING::
    This subsection can be a bit confusing for beginners.
    Do not worry if you do not understand it all.


HyperSpy stores the data in the :py:class:`~.signal.BaseSignal` class, that is
the object that you get when e.g. you load a single file using
:py:func:`~.io.load`. Most of the data analysis functions are also contained in
this class or its specialized subclasses. The :py:class:`~.signal.BaseSignal`
class contains general functionality that is available to all the subclasses.
The subclasses provide functionality that is normally specific to a particular
type of data, e.g. the :py:class:`~._signals.signal1d.Signal1D` class provides
common functionality to deal with one-dimensional (e.g. spectral) data and
:py:class:`~._signals.eels.EELSSpectrum` (which is a subclass of
:py:class:`~._signals.signal1d.Signal1D`) adds extra functionality to the
:py:class:`~._signals.signal1d.Signal1D` class for electron energy-loss
spectroscopy data analysis.

.. versionchanged:: 0.8.5

Currently the following signal subclasses are available:

* :py:class:`~._signals.signal1d.Signal1D`
* :py:class:`~._signals.signal2d.Signal2D`
* :py:class:`~._signals.eels.EELSSpectrum`
* :py:class:`~._signals.eds_tem.EDSTEMSpectrum`
* :py:class:`~._signals.eds_sem.EDSSEMSpectrum`
* :py:class:`~._signals.spectrum_simulation.SpectrumSimulation`
* :py:class:`~._signals.image_simulation.ImageSimulation`

.. versionchanged:: 0.8.5

Note that in 0.8.5 the :py:class:`~._signals.signal1d.Signal1D` and
:py:class:`~._signals.signal2d.Signal2D` classes were created to deprecate the
old :py:class:`~._signals.spectrum.Spectrum` and
:py:class:`~._signals.image.Image` classes.


The :py:mod:`~.signals` module, which contains all available signal subclasses,
is imported in the user namespace when loading hyperspy. In the following
example we create a Signal2D instance from a 2D numpy array:

.. code-block:: python

    >>> im = hs.signals.Signal2D(np.random.random((64,64)))


The different signals store other objects in what are called attributes. For
examples, the data is stored in a numpy array in the
:py:attr:`~.signal.BaseSignal.data` attribute, the original parameters in the
:py:attr:`~.signal.BaseSignal.original_metadata` attribute, the mapped parameters
in the :py:attr:`~.signal.BaseSignal.metadata` attribute and the axes
information (including calibration) can be accessed (and modified) in the
:py:attr:`~.signal.BaseSignal.axes_manager` attribute.


.. _transforming.signal:

Transforming between signal subclasses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The different subclasses are characterized by three
:py:attr:`~.signal.BaseSignal.metadata` attributes (see the table below):

`record_by`
    Can be "spectrum", "image" or "", the latter meaning undefined and describes
    the way the data is arranged in memory. It is possible to transform any
    :py:class:`~.signal.BaseSignal` subclass to a :py:class:`~._signals.signal1d.Signal1D`
    or :py:class:`~._signals.signal2d.Signal2D` subclass using the following
    :py:class:`~.signal.BaseSignal` methods: :py:meth:`~.signal.BaseSignal.as_signal2D`
    and :py:meth:`~.signal.BaseSignal.as_signal1D`. In addition
    :py:class:`~._signals.signal1d.Signal1D` instances can be transformed into
    two-dimensional signals using :py:meth:`~._signals.signal1d.Signal1D.to_signal2D`
    and two-dimensional instances transformed into one dimensional instances using
    :py:meth:`~._signals.signal2d.Signal2D.to_signal1D`. When transforming between
    one and two dimensinoal signal classes the order in which the data array is stored
    in memory is modified to improve performance. Also, some functions, e.g. plotting
    or decomposing, will behave differently.

`signal_type`
    Describes the nature of the signal. It can be any string, normally the
    acronym associated with a particular signal. In certain cases HyperSpy provides
    features that are only available for a particular signal type through
    :py:class:`~.signal.BaseSignal` subclasses. The :py:class:`~.signal.BaseSignal` method
    :py:meth:`~.signal.BaseSignal.set_signal_type` changes the signal_type in place, which
    may result in a :py:class:`~.signal.BaseSignal` subclass transformation.

`signal_origin`
    Describes the origin of the signal and can be "simulation" or "experiment" or "", the
    latter meaning undefined. In certain cases HyperSpy provides features that are only
    available for a particular signal origin. The :py:class:`~.signal.BaseSignal` method
    :py:meth:`~.signal.BaseSignal.set_signal_origin` changes the signal_origin in place,
    which may result in a :py:class:`~.signal.BaseSignal` subclass transformation.

.. table:: BaseSignal subclass :py:attr:`~.signal.BaseSignal.metadata` attributes.

    +---------------------------------------------------------------+-----------+-------------+---------------+
    |                      BaseSignal subclass                      | record_by | signal_type | signal_origin |
    +===============================================================+===========+=============+===============+
    |                 :py:class:`~.signal.BaseSignal`               |     -     |      -      |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |           :py:class:`~._signals.signal1d.Signal1D`            | spectrum  |      -      |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    | :py:class:`~._signals.spectrum_simulation.SpectrumSimulation` | spectrum  |      -      |  simulation   |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |           :py:class:`~._signals.eels.EELSSpectrum`            | spectrum  |    EELS     |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |           :py:class:`~._signals.eds_sem.EDSSEMSpectrum`       | spectrum  |   EDS_SEM   |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |           :py:class:`~._signals.eds_tem.EDSTEMSpectrum`       | spectrum  |   EDS_TEM   |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |              :py:class:`~._signals.signal2d.Signal2D`         |   image   |      -      |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |    :py:class:`~._signals.image_simulation.ImageSimulation`    |   image   |      -      |  simulation   |
    +---------------------------------------------------------------+-----------+-------------+---------------+


The following example shows how to transform between different subclasses.

   .. code-block:: python

       >>> s = hs.signals.Signal1D(np.random.random((10,20,100)))
       >>> s
       <Signal1D, title: , dimensions: (20, 10|100)>
       >>> s.metadata
       ├── record_by = spectrum
       ├── signal_origin =
       ├── signal_type =
       └── title =
       >>> im = s.to_signal2D()
       >>> im
       <Signal2D, title: , dimensions: (100|20, 10)>
       >>> im.metadata
       ├── record_by = image
       ├── signal_origin =
       ├── signal_type =
       └── title =
       >>> s.set_signal_type("EELS")
       >>> s
       <EELSSpectrum, title: , dimensions: (20, 10|100)>
       >>> s.set_signal_origin("simulation")
       >>> s
       <EELSSpectrumSimulation, title: , dimensions: (20, 10|100)>


The navigation and signal dimensions
------------------------------------

HyperSpy can deal with data of arbitrary dimensions. Each dimension is
internally classified as either "navigation" or "signal" and the way this
classification is done determines the behaviour of the signal.

The concept is probably best understood with an example: let's imagine a three
dimensional dataset. This dataset could be an spectrum image acquired by
scanning over a sample in two dimensions. In HyperSpy's terminology the
spectrum dimension would be the signal dimension and the two other dimensions
would be the navigation dimensions. We could see the same dataset as an image
stack instead.  Actually it could has been acquired by capturing two
dimensional images at different wavelengths. Then it would be natural to
identify the two spatial dimensions as the signal dimensions and the wavelength
dimension as the navigation dimension.  However, for data analysis purposes,
one may like to operate with an image stack as if it was a set of spectra or
viceversa. One can easily switch between these two alternative ways of
classifiying the dimensions of a three-dimensional dataset by
:ref:`transforming between BaseSignal subclasses
<transforming.signal>`.

.. NOTE::

    Although each dimension can be arbitrarily classified as "navigation
    dimension" or "signal dimension", for most common tasks there is no need to
    modify HyperSpy's default choice.


.. _signal.binned:

Binned and unbinned signals
---------------------------

.. versionadded:: 0.7

Signals that are a histogram of a probability density function (pdf) should
have the ``signal.metadata.Signal.binned`` attribute set to
``True``. This is because some methods operate differently in signals that are
*binned*.

The default value of the ``binned`` attribute is shown in the
following table:

.. table:: Binned default values for the different subclasses.


    +---------------------------------------------------------------+--------+
    |                       BaseSignal subclass                     | binned |
    +===============================================================+========+
    |                 :py:class:`~.signal.BaseSignal`               | False  |
    +---------------------------------------------------------------+--------+
    |           :py:class:`~._signals.signal1d.Signal1D`            | False  |
    +---------------------------------------------------------------+--------+
    | :py:class:`~._signals.spectrum_simulation.SpectrumSimulation` | False  |
    +---------------------------------------------------------------+--------+
    |           :py:class:`~._signals.eels.EELSSpectrum`            | True   |
    +---------------------------------------------------------------+--------+
    |           :py:class:`~._signals.eds_sem.EDSSEMSpectrum`       | True   |
    +---------------------------------------------------------------+--------+
    |           :py:class:`~._signals.eds_tem.EDSTEMSpectrum`       | True   |
    +---------------------------------------------------------------+--------+
    |              :py:class:`~._signals.signal2d.Signal2D`         | False  |
    +---------------------------------------------------------------+--------+
    |    :py:class:`~._signals.image_simulation.ImageSimulation`    | False  |
    +---------------------------------------------------------------+--------+





To change the default value:

.. code-block:: python

    >>> s.metadata.Signal.binned = True

Generic tools
-------------

Below we briefly introduce some of the most commonly used tools (methods). For
more details about a particular method click on its name. For a detailed list
of all the methods available see the :py:class:`~.signal.BaseSignal` documentation.

The methods of this section are available to all the signals. In other chapters
methods that are only available in specialized
subclasses.

.. _signal.indexing:

Indexing
^^^^^^^^
.. versionadded:: 0.6
.. versionchanged:: 0.8.1

Indexing a :py:class:`~.signal.BaseSignal`  provides a powerful, convenient and
Pythonic way to access and modify its data. In HyperSpy indexing is achieved
using ``isig`` and ``inav``, which allow the navigation and signal dimensions
to be indexed independently. The idea is essentially to specify a subset of the
data based on its position in the array and it is therefore essential to know
the convention adopted for specifying that position, which is described here.

Those new to Python may find indexing a somewhat esoteric concept but once
mastered it is one of the most powerful features of Python based code and
greatly simplifies many common tasks. HyperSpy's Signal indexing is similar
to numpy array indexing and those new to Python are encouraged to read the
associated `numpy documentation on the subject  <http://ipython.org/>`_.


Key features of indexing in HyperSpy are as follows (note that some of these
features differ from numpy):

* HyperSpy indexing does:

  + Allow independent indexing of signal and navigation dimensions
  + Support indexing with decimal numbers.
  + Use the image order for indexing i.e. [x, y, z,...] (hyperspy) vs
    [...,z,y,x] (numpy)

* HyperSpy indexing does not:

  + Support indexing using arrays.
  + Allow the addition of new axes using the newaxis object.

The examples below illustrate a range of common indexing tasks.

First consider indexing a single spectrum, which has only one signal dimension
(and no navigation dimensions) so we use ``isig``:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(10))
    >>> s
    <Signal1D, title: , dimensions: (|10)>
    >>> s.data
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> s.isig[0]
    <Signal1D, title: , dimensions: (|1)>
    >>> s.isig[0].data
    array([0])
    >>> s.isig[9].data
    array([9])
    >>> s.isig[-1].data
    array([9])
    >>> s.isig[:5]
    <Signal1D, title: , dimensions: (|5)>
    >>> s.isig[:5].data
    array([0, 1, 2, 3, 4])
    >>> s.isig[5::-1]
    <Signal1D, title: , dimensions: (|6)>
    >>> s.isig[5::-1]
    <Signal1D, title: , dimensions: (|6)>
    >>> s.isig[5::2]
    <Signal1D, title: , dimensions: (|3)>
    >>> s.isig[5::2].data
    array([5, 7, 9])


Unlike numpy, HyperSpy supports indexing using decimal numbers, in which case
HyperSpy indexes using the axis scales instead of the indices.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(10))
    >>> s
    <Signal1D, title: , dimensions: (|10)>
    >>> s.data
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> s.axes_manager[0].scale = 0.5
    >>> s.axes_manager[0].axis
    array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5])
    >>> s.isig[0.5:4.].data
    array([1, 2, 3, 4, 5, 6, 7])
    >>> s.isig[0.5:4].data
    array([1, 2, 3])
    >>> s.isig[0.5:4:2].data
    array([1, 3])


Importantly the original :py:class:`~.signal.BaseSignal` and its "indexed self"
share their data and, therefore, modifying the value of the data in one
modifies the same value in the other. Note also that in the example below
s.data is used to access the data as a numpy array directly and this array is
then indexed using numpy indexing.

.. code-block:: python

    >>> s = hs.signals.Spectrum(np.arange(10))
    >>> s
    <Spectrum, title: , dimensions: (10,)>
    >>> s.data
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> si = s.isig[::2]
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
data treating navigation axes using ``inav`` and signal axes using ``isig``.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(2*3*4).reshape((2,3,4)))
    >>> s
    <Signal1D, title: , dimensions: (10, 10, 10)>
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
    >>> s.inav[0,0].isig[::-1].data
    array([3, 2, 1, 0])
    >>> s.isig[0]
    <Signal1D, title: , dimensions: (2, 3)>
    >>> s.isig[0].axes_manager
    <Axes manager, axes: (<x axis, size: 3, index: 0>, <y axis, size: 2, index: 0>)>
    >>> s.isig[0].data
    array([[ 0,  4,  8],
       [12, 16, 20]])

Independent indexation of the signal and navigation dimensions is demonstrated
further in the following:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(2*3*4).reshape((2,3,4)))
    >>> s
    <Signal1D, title: , dimensions: (10, 10, 10)>
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
    <Signal1D, title: , dimensions: (2, 3)>
    >>> s.isig[0].axes_manager
    <Axes manager, axes: (<x axis, size: 3, index: 0>, <y axis, size: 2, index: 0>)>
    >>> s.isig[0].data
    array([[ 0,  4,  8],
       [12, 16, 20]])


The same syntax can be used to set the data values in signal and navigation
dimensions respectively:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(2*3*4).reshape((2,3,4)))
    >>> s
    <Signal1D, title: , dimensions: (10, 10, 10)>
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

.. versionadded:: 0.8.3

:py:class:`~.signal.BaseSignal` supports all the Python binary arithmetic
opearations (+, -, \*, //, %, divmod(), pow(), \*\*, <<, >>, &, ^, \|),
augmented binary assignments (+=, -=, \*=, /=, //=, %=, \*\*=, <<=, >>=, &=,
^=, \|=), unary operations (-, +, abs() and ~) and rich comparisons operations
(<, <=, ==, x!=y, <>, >, >=).

These operations are performed element-wise. When the dimensions of the signals
are not equal `numpy broadcasting rules apply
<http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ independently
for the navigation and signal axes.

In the following example `s2` has only one navigation axis while `s` has two.
However, because the size of their first navigation axis is the same, their
dimensions are compatible and `s2` is
broacasted to match `s`'s dimensions.

.. code-block:: python

    >>> s = hs.signals.Signal2D(np.ones((3,2,5,4)))
    >>> s2 = hs.signals.Signal2D(np.ones((2,5,4)))
    >>> s
    <Signal2D, title: , dimensions: (2, 3|4, 5)>
    >>> s2
    <Signal2D, title: , dimensions: (2|4, 5)>
    >>> s + s2
    <Signal2D, title: , dimensions: (2, 3|4, 5)>

In the following example the dimensions are not compatible and an exception
is raised.

.. code-block:: python

    >>> s = hs.signals.Signal2D(np.ones((3,2,5,4)))
    >>> s2 = hs.signals.Signal2D(np.ones((3,5,4)))
    >>> s
    <Signal2D, title: , dimensions: (2, 3|4, 5)>
    >>> s2
    <Signal2D, title: , dimensions: (3|4, 5)>
    >>> s + s2
    Traceback (most recent call last):
      File "<ipython-input-55-044bb11a0bd9>", line 1, in <module>
        s + s2
      File "<string>", line 2, in __add__
      File "/home/fjd29/Python/hyperspy/hyperspy/signal.py", line 2686, in _binary_operator_ruler
        raise ValueError(exception_message)
    ValueError: Invalid dimensions for this operation

Broacasting operates exactly in the same way for the signal axes:

.. code-block:: python

    >>> s = hs.signals.Signal2D(np.ones((3,2,5,4)))
    >>> s2 = hs.signals.Signal1D(np.ones((3, 2, 4)))
    >>> s
    <Signal2D, title: , dimensions: (2, 3|4, 5)>
    >>> s2
    <Signal1D, title: , dimensions: (2, 3|4)>
    >>> s + s2
    <Signal2D, title: , dimensions: (2, 3|4, 5)>

In-place operators also support broadcasting, but only when broadcasting would
not change the left most signal dimensions:

.. code-block:: python

    >>> s += s2
    >>> s
    <Signal2D, title: , dimensions: (2, 3|4, 5)>
    >>> s2 += s
    Traceback (most recent call last):
      File "<ipython-input-64-fdb9d3a69771>", line 1, in <module>
        s2 += s
      File "<string>", line 2, in __iadd__
      File "/home/fjd29/Python/hyperspy/hyperspy/signal.py", line 2737, in _binary_operator_ruler
        self.data = getattr(sdata, op_name)(odata)
    ValueError: non-broadcastable output operand with shape (3,2,1,4) doesn't match the broadcast shape (3,2,5,4)


.. _signal.iterator:

Iterating over the navigation axes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signal instances are iterables over the navigation axes. For example, the
following code creates a stack of 10 images and saves them in separate "png"
files by iterating over the signal instance:

.. code-block:: python

    >>> image_stack = hs.signals.Signal2D(np.random.random((2, 5, 64,64)))
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
    >>> image_stack = hs.signals.Signal2D(np.array([scipy.misc.lena()]*5))
    >>> image_stack.axes_manager[1].name = "x"
    >>> image_stack.axes_manager[2].name = "y"
    >>> for image, angle in zip(image_stack, (0, 45, 90, 135, 180)):
    ...    image.data[:] = scipy.ndimage.rotate(image.data, angle=angle,
    ...    reshape=False)
    >>> collage = hs.stack([image for image in image_stack], axis=0)
    >>> collage.plot()

.. figure::  images/rotate_lena.png
  :align:   center
  :width:   500

  Rotation of images by iteration.

.. versionadded:: 0.7


Iterating external functions with the map method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Performing an operation on the data at each coordinate, as in the previous example,
using an external function can be more easily accomplished using the
:py:meth:`~.signal.BaseSignal.map` method:

.. code-block:: python

    >>> import scipy.ndimage
    >>> image_stack = hs.signals.Signal2D(np.array([scipy.misc.lena()]*4))
    >>> image_stack.axes_manager[1].name = "x"
    >>> image_stack.axes_manager[2].name = "y"
    >>> image_stack.map(scipy.ndimage.rotate,
    ...                            angle=45,
    ...                            reshape=False)
    >>> collage = hs.stack([image for image in image_stack], axis=0)
    >>> collage.plot()

.. figure::  images/rotate_lena_apply_simple.png
  :align:   center
  :width:   500

  Rotation of images by the same amount using :py:meth:`~.signal.BaseSignal.map`.

The :py:meth:`~.signal.BaseSignal.map` method can also take variable
arguments as in the following example.

.. code-block:: python

    >>> import scipy.ndimage
    >>> image_stack = hs.signals.Signal2D(np.array([scipy.misc.lena()]*4))
    >>> image_stack.axes_manager[1].name = "x"
    >>> image_stack.axes_manager[2].name = "y"
    >>> angles = hs.signals.Signal(np.array([0, 45, 90, 135]))
    >>> angles.axes_manager.set_signal_dimension(0)
    >>> modes = hs.signals.Signal(np.array(['constant', 'nearest', 'reflect', 'wrap']))
    >>> modes.axes_manager.set_signal_dimension(0)
    >>> image_stack.map(scipy.ndimage.rotate,
    ...                            angle=angles,
    ...                            reshape=False,
    ...                            mode=modes)
    calculating 100% |#############################################| ETA:  00:00:00Cropping

.. figure::  images/rotate_lena_apply_ndkwargs.png
  :align:   center
  :width:   500

  Rotation of images using :py:meth:`~.signal.BaseSignal.map` with different
  arguments for each image in the stack.

Cropping
^^^^^^^^

Cropping can be performed in a very compact and powerful way using
:ref:`signal.indexing` . In addition it can be performed using the following
method or GUIs if cropping :ref:`signal1D <signal1D.crop>` or :ref:`signal2D
<signal2D.crop>`. There is also a general :py:meth:`~.signal.BaseSignal.crop`
method that operates *in place*.

Rebinning
^^^^^^^^^

The :py:meth:`~.signal.BaseSignal.rebin` method rebins data in place down to a size
determined by the user.

Folding and unfolding
^^^^^^^^^^^^^^^^^^^^^

When dealing with multidimensional datasets it is sometimes useful to transform
the data into a two dimensional dataset. This can be accomplished using the
following two methods:

* :py:meth:`~.signal.BaseSignal.fold`
* :py:meth:`~.signal.BaseSignal.unfold`

It is also possible to unfold only the navigation or only the signal space:

* :py:meth:`~.signal.BaseSignal.unfold_navigation_space`
* :py:meth:`~.signal.BaseSignal.unfold_signal_space`


.. _signal.stack_split:

Splitting and stacking
^^^^^^^^^^^^^^^^^^^^^^

Several objects can be stacked together over an existing axis or over a
new axis using the :py:func:`~.utils.stack` function, if they share axis
with same dimension.

.. code-block:: python

    >>> image = hs.signals.Signal2D(scipy.misc.lena())
    >>> image = hs.stack([hs.stack([image]*3,axis=0)]*3,axis=1)
    >>> image.plot()

.. figure::  images/stack_lena_3_3.png
  :align:   center
  :width:   500

  Stacking example.

An object can be splitted into several objects
with the :py:meth:`~.signal.BaseSignal.split` method. This function can be used
to reverse the :py:func:`~.utils.stack` function:

.. code-block:: python

    >>> image = image.split()[0].split()[0]
    >>> image.plot()

.. figure::  images/split_lena_3_3.png
  :align:   center
  :width:   400

  Splitting example.


Simple operations over one axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~.signal.BaseSignal.sum`
* :py:meth:`~.signal.BaseSignal.mean`
* :py:meth:`~.signal.BaseSignal.max`
* :py:meth:`~.signal.BaseSignal.min`
* :py:meth:`~.signal.BaseSignal.std`
* :py:meth:`~.signal.BaseSignal.var`
* :py:meth:`~.signal.BaseSignal.diff`
* :py:meth:`~.signal.BaseSignal.derivative`
* :py:meth:`~.signal.BaseSignal.integrate_simpson`

.. _signal.change_dtype:

Changing the data type
^^^^^^^^^^^^^^^^^^^^^^

Even if the original data is recorded with a limited dynamic range, it is often
desirable to perform the analysis operations with a higher precision.
Conversely, if space is limited, storing in a shorter data type can decrease
the file size. The :py:meth:`~.signal.BaseSignal.change_dtype` changes the data
type in place, e.g.:

.. code-block:: python

    >>> s = hs.load('EELS Spectrum Image (high-loss).dm3')
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


.. versionadded:: 0.7

    In addition to all standard numpy dtypes HyperSpy supports four extra
    dtypes for RGB images: rgb8, rgba8, rgb16 and rgba16. Changing
    from and to any rgbx dtype is more constrained than most other dtype
    conversions. To change to a rgbx dtype the signal `record_by` must be
    "spectrum", `signal_dimension` must be 3(4) for rgb(rgba) dtypes and the
    dtype must be uint8(uint16) for rgbx8(rgbx16).  After conversion
    `record_by` becomes `image` and the spectra dimension is removed. The dtype
    of images of dtype rgbx8(rgbx16) can only be changed to uint8(uint16) and
    the `record_by` becomes "spectrum".

    In the following example we create

   .. code-block:: python

        >>> rgb_test = np.zeros((1024, 1024, 3))
        >>> ly, lx = rgb_test.shape[:2]
        >>> offset_factor = 0.16
        >>> size_factor = 3
        >>> Y, X = np.ogrid[0:lx, 0:ly]
        >>> rgb_test[:,:,0] = (X - lx / 2 - lx*offset_factor) ** 2 + (Y - ly / 2 - ly*offset_factor) ** 2 < lx * ly / size_factor **2
        >>> rgb_test[:,:,1] = (X - lx / 2 + lx*offset_factor) ** 2 + (Y - ly / 2 - ly*offset_factor) ** 2 < lx * ly / size_factor **2
        >>> rgb_test[:,:,2] = (X - lx / 2) ** 2 + (Y - ly / 2 + ly*offset_factor) ** 2 < lx * ly / size_factor **2
        >>> rgb_test *= 2**16 - 1
        >>> s = hs.signals.Signal1D(rgb_test)
        >>> s.change_dtype("uint16")
        >>> s
        <Signal1D, title: , dimensions: (1024, 1024|3)>
        >>> s.change_dtype("rgb16")
        >>> s
        <Signal2D, title: , dimensions: (|1024, 1024)>
        >>> s.plot()


   .. figure::  images/rgb_example.png
      :align:   center
      :width:   500

      RGB data type example.



Basic statistical analysis
--------------------------
.. versionadded:: 0.7

:py:meth:`~.signal.BaseSignal.get_histogram` computes the histogram and
conveniently returns it as signal instance. It provides methods to
calculate the bins. :py:meth:`~.signal.BaseSignal.print_summary_statistics` prints
the five-number summary statistics of the data.

These two methods can be combined with
:py:meth:`~.signal.BaseSignal.get_current_signal` to compute the histogram or
print the summary stastics of the signal at the current coordinates, e.g:
.. code-block:: python

    >>> s = hs.signals.EELSSpectrum(np.random.normal(size=(10,100)))
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

Histogram of different objects can be compared with the functions
:py:func:`~.drawing.utils.plot_histograms` (see
:ref:`visualisation <plot_spectra>` for the plotting options). For example,
with histograms of several random chi-square distributions:


.. code-block:: python

    >>> img = hs.signals.Signal2D([np.random.chisquare(i+1,[100,100]) for i in range(5)])
    >>> hs.plot.plot_histograms(img,legend='auto')

.. figure::  images/plot_histograms_chisquare.png
   :align:   center
   :width:   500

   Comparing histograms.


.. _signal.noise_properties:

Setting the noise properties
----------------------------

Some data operations require the data variance. Those methods use the
``metadata.Signal.Noise_properties.variance`` attribute if it exists. You can
set this attribute as in the following example where we set the variance to be
10:

.. code-block:: python

    s.metadata.Signal.set_item("Noise_properties.variance", 10)

For heterocedastic noise the ``variance`` attribute must be a
:class:`~.signal_base.BaseSignal`.  Poissonian noise is a common case  of
heterocedastic noise where the variance is equal to the expected value. The
:meth:`~.signal_base.BaseSignal.estimate_poissonian_noise_variance`
:class:`~.signal_base.BaseSignal` method can help setting the variance of data with
semi-poissonian noise. With the default arguments, this method simply sets the
variance attribute to the given ``expected_value``. However, more generally
(although then noise is not strictly poissonian), the variance may be proportional
to the expected value. Moreover, when the noise is a mixture of white
(gaussian) and poissonian noise, the variance is described by the following
linear model:

    .. math::

        \mathrm{Var}[X] = (a * \mathrm{E}[X] + b) * c

Where `a` is the ``gain_factor``, `b` is the ``gain_offset`` (the gaussian
noise variance) and `c` the ``correlation_factor``. The correlation
factor accounts for correlation of adjacent signal elements that can
be modeled as a convolution with a gaussian point spread function.
:meth:`~.signal.BaseSignal.estimate_poissonian_noise_variance` can be used to set
the noise properties when the variance can be described by this linear model,
for example:


.. code-block:: python

  >>> s = hs.signals.SpectrumSimulation(np.ones(100))
  >>> s.add_poissonian_noise()
  >>> s.metadata
  ├── General
  │   └── title =
  └── Signal
      ├── binned = False
      ├── record_by = spectrum
      ├── signal_origin = simulation
      └── signal_type =

  >>> s.estimate_poissonian_noise_variance()
  >>> s.metadata
  ├── General
  │   └── title =
  └── Signal
      ├── Noise_properties
      │   ├── Variance_linear_model
      │   │   ├── correlation_factor = 1
      │   │   ├── gain_factor = 1
      │   │   └── gain_offset = 0
      │   └── variance = <SpectrumSimulation, title: Variance of , dimensions: (|100)>
      ├── binned = False
      ├── record_by = spectrum
      ├── signal_origin = simulation
      └── signal_type =
