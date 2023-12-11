Basics of signals
-----------------

.. _signal_initialization:

Signal initialization
^^^^^^^^^^^^^^^^^^^^^

Many of the values in the :class:`~.axes.AxesManager` can be
set when making the :class:`~.api.signals.BaseSignal` object.

.. code-block:: python

    >>> dict0 = {'size': 10, 'name':'Axis0', 'units':'A', 'scale':0.2, 'offset':1}
    >>> dict1 = {'size': 20, 'name':'Axis1', 'units':'B', 'scale':0.1, 'offset':2}
    >>> s = hs.signals.BaseSignal(np.random.random((10,20)), axes=[dict0, dict1])
    >>> s.axes_manager
    <Axes manager, axes: (|20, 10)>
                Name |   size |  index |  offset |   scale |  units
    ================ | ====== | ====== | ======= | ======= | ======
    ---------------- | ------ | ------ | ------- | ------- | ------
               Axis1 |     20 |      0 |       2 |     0.1 |      B
               Axis0 |     10 |      0 |       1 |     0.2 |      A

This also applies to the :attr:`~.signals.BaseSignal.metadata`.

.. code-block:: python

    >>> metadata_dict = {'General':{'name':'A BaseSignal'}}
    >>> metadata_dict['General']['title'] = 'A BaseSignal title'
    >>> s = hs.signals.BaseSignal(np.arange(10), metadata=metadata_dict)
    >>> s.metadata
    ├── General
    │   ├── name = A BaseSignal
    │   └── title = A BaseSignal title
    └── Signal
	    └── signal_type =

Instead of using a list of *axes dictionaries* ``[dict0, dict1]`` during signal
initialization, you can also pass a list of *axes objects*: ``[axis0, axis1]``.

The navigation and signal dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy can deal with data of arbitrary dimensions. Each dimension is
internally classified as either "navigation" or "signal" and the way this
classification is done determines the behaviour of the signal.

The concept is probably best understood with an example: let's imagine a three
dimensional dataset e.g. a numpy array with dimensions `(10, 20, 30)`. This
dataset could be an spectrum image acquired by scanning over a sample in two
dimensions. As in this case the signal is one-dimensional we use a
:class:`~.api.signals.Signal1D` subclass for this data e.g.:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.random.random((10, 20, 30)))
    >>> s
    <Signal1D, title: , dimensions: (20, 10|30)>

In HyperSpy's terminology, the *signal dimension* of this dataset is 30 and
the navigation dimensions (20, 10). Notice the separator `|` between the
navigation and signal dimensions.


However, the same dataset could also be interpreted as an image
stack instead.  Actually it could has been acquired by capturing two
dimensional images at different wavelengths. Then it would be natural to
identify the two spatial dimensions as the signal dimensions and the wavelength
dimension as the navigation dimension. To view the data in this way we could
have used a :class:`~.api.signals.Signal2D` instead e.g.:

.. code-block:: python

    >>> im = hs.signals.Signal2D(np.random.random((10, 20, 30)))
    >>> im
    <Signal2D, title: , dimensions: (10|30, 20)>

Indeed, for data analysis purposes,
one may like to operate with an image stack as if it was a set of spectra or
viceversa. One can easily switch between these two alternative ways of
classifying the dimensions of a three-dimensional dataset by
:ref:`transforming between BaseSignal subclasses
<transforming_signal-label>`.

The same dataset could be seen as a three-dimensional signal:

.. code-block:: python

    >>> td = hs.signals.BaseSignal(np.random.random((10, 20, 30)))
    >>> td
    <BaseSignal, title: , dimensions: (|30, 20, 10)>

Notice that with use :class:`~.api.signals.BaseSignal` because there is
no specialised subclass for three-dimensional data. Also note that by default
:class:`~.api.signals.BaseSignal` interprets all dimensions as signal dimensions.
We could also configure it to operate on the dataset as a three-dimensional
array of scalars by changing the default *view* of
:class:`~.api.signals.BaseSignal` by taking the transpose of it:

.. code-block:: python

    >>> scalar = td.T
    >>> scalar
    <BaseSignal, title: , dimensions: (30, 20, 10|)>

For more examples of manipulating signal axes in the "signal-navigation" space
can be found in :ref:`signal.transpose`.

.. NOTE::

    Although each dimension can be arbitrarily classified as "navigation
    dimension" or "signal dimension", for most common tasks there is no need to
    modify HyperSpy's default choice.


.. _signal-subclasses:

Signal subclasses
^^^^^^^^^^^^^^^^^

The :mod:`~.api.signals` module, which contains all available signal subclasses,
is imported in the user namespace when loading HyperSpy. In the following
example we create a Signal2D instance from a 2D numpy array:

.. code-block:: python

    >>> im = hs.signals.Signal2D(np.random.random((64,64)))
    >>> im
    <Signal2D, title: , dimensions: (|64, 64)>

The :ref:`table below <signal_subclasses_table-label>` summarises all the
:class:`~.api.signals.BaseSignal` subclasses currently distributed
with HyperSpy. From HyperSpy 2.0, all domain specific signal
subclasses, characterized by the ``signal_type`` metadata attribute, are
provided by dedicated :ref:`extension packages <hyperspy_extensions-label>`.

The generic subclasses provided by HyperSpy are characterized by the the data
``dtype`` and the signal dimension. In particular, there are specialised signal
subclasses to handle complex data. See the table and diagram below. Where
appropriate, functionalities are restricted to certain
:class:`~.api.signals.BaseSignal` subclasses.

.. _signal_overview_figure-label:

.. figure::  ../images/HyperSpySignalOverview.png
  :align:   center
  :width:   500

  Diagram showing the inheritance structure of the different subclasses. The
  upper part contains the generic classes shipped with HyperSpy. The lower
  part contains examples of domain specific subclasses provided by some of the
  :ref:`hyperspy_extensions-label`.

.. _signal_subclasses_table-label:

.. table:: BaseSignal subclass characteristics.

    +----------------------------------------+------------------+-------------+---------+
    | BaseSignal subclass                    | signal_dimension | signal_type |  dtype  |
    +========================================+==================+=============+=========+
    | :class:`~.api.signals.BaseSignal`      |        -         |      -      |  real   |
    +----------------------------------------+------------------+-------------+---------+
    | :class:`~.api.signals.Signal1D`        |        1         |      -      |  real   |
    +----------------------------------------+------------------+-------------+---------+
    | :class:`~.api.signals.Signal2D`        |        2         |      -      |  real   |
    +----------------------------------------+------------------+-------------+---------+
    | :class:`~.api.signals.ComplexSignal`   |        -         |      -      | complex |
    +----------------------------------------+------------------+-------------+---------+
    | :class:`~.api.signals.ComplexSignal1D` |        1         |      -      | complex |
    +----------------------------------------+------------------+-------------+---------+
    | :class:`~.api.signals.ComplexSignal2D` |        2         |      -      | complex |
    +----------------------------------------+------------------+-------------+---------+

.. versionchanged:: 1.0
    The subclasses ``Simulation``, ``SpectrumSimulation`` and ``ImageSimulation``
    were removed.

.. versionadded:: 1.5
    External packages can register extra :class:`~.api.signals.BaseSignal`
    subclasses.

.. versionchanged:: 2.0
    The subclasses ``EELS``, ``EDS_SEM``, ``EDS_TEM`` and
    ``DielectricFunction`` have been moved to the extension package
    ``EleXSpy`` and the subclass ``hologram`` has been
    moved to the extension package ``HoloSpy``.

.. _hyperspy_extensions-label:

HyperSpy extensions
^^^^^^^^^^^^^^^^^^^

Domain specific functionalities for specific types of data are provided through
a number of dedicated python packages that qualify as `HyperSpy extensions`. These
packages provide subclasses of the generic signal classes listed above, depending
on the dimensionality and type of the data. Some examples are included in the
:ref:`diagram above <signal_overview_figure-label>`.
If an extension package is installed on your system, the provided signal
subclasses are registered with HyperSpy and these classes are directly
available when loading the ``hyperspy.api`` into the namespace. A `list of packages
that extend HyperSpy <https://github.com/hyperspy/hyperspy-extensions-list>`_
is curated in a dedicated repository.

The metadata attribute ``signal_type`` describes the nature of the signal. It can
be any string, normally the acronym associated with a particular signal. To print
all :class:`~.api.signals.BaseSignal` subclasses available in your system call
the function :func:`~.api.print_known_signal_types` as in the following
example:

.. code-block:: python

    >>> hs.print_known_signal_types() # doctest: +SKIP
    +--------------------+---------------------+--------------------+----------+
    |    signal_type     |       aliases       |     class name     | package  |
    +--------------------+---------------------+--------------------+----------+
    | DielectricFunction | dielectric function | DielectricFunction |  exspy   |
    |      EDS_SEM       |                     |   EDSSEMSpectrum   |  exspy   |
    |      EDS_TEM       |                     |   EDSTEMSpectrum   |  exspy   |
    |        EELS        |       TEM EELS      |    EELSSpectrum    |  exspy   |
    |      hologram      |                     |   HologramImage    | holospy  |
    +--------------------+---------------------+--------------------+----------+

When :ref:`loading data <loading_files>`, the ``signal_type`` will be
set automatically by the file reader, as defined in ``rosettasciio``. If the
extension providing the corresponding signal subclass is installed,
:func:`~.api.load` will return the subclass from the hyperspy extension,
otherwise a warning will be raised to explain that
no registered signal class can be assigned to the given ``signal_type``.

Since the :func:`~.api.load` can return domain specific signal objects (e.g.
``EDSSEMSpectrum`` from ``EleXSpy``) provided by extensions, the corresponding
functionalities (so-called `method` of `object` in object-oriented programming,
e.g. ``EDSSEMSpectrum.get_lines_intensity()``) implemented in signal classes of
the extension can be accessed directly. To use additional functionalities
implemented in extensions, but not as method of the signal class, the extensions
need to be imported explicitly (e.g. ``import elexspy``). Check the user guides
of the respective `HyperSpy extensions
<https://github.com/hyperspy/hyperspy-extensions-list>`_ for details on the
provided methods and functions.

For details on how to write and register extensions see
:ref:`writing_extensions-label`.

.. _transforming_signal-label:

Transforming between signal subclasses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~.api.signals.BaseSignal` method
:meth:`~.api.signals.BaseSignal.set_signal_type` changes the ``signal_type``
in place, which may result in a :class:`~.api.signals.BaseSignal` subclass
transformation.

The following example shows how to change the signal dimensionality and how
to transform between different subclasses:

   .. code-block:: python

       >>> s = hs.signals.Signal1D(np.random.random((10,20,100)))
       >>> s
       <Signal1D, title: , dimensions: (20, 10|100)>
       >>> s.metadata
       ├── General
       │   └── title = 
       └── Signal
           └── signal_type = 
       >>> im = s.to_signal2D()
       >>> im
       <Signal2D, title: , dimensions: (100|20, 10)>
       >>> im.metadata
       ├── General
       │   └── title = 
       └── Signal
           └── signal_type = 
       >>> s.set_signal_type("EELS")
       >>> s
       <EELSSpectrum, title: , dimensions: (20, 10|100)>
       >>> s.metadata
       ├── General
       │   └── title = 
       └── Signal
           └── signal_type = EELS
       >>> s.change_dtype("complex")
       >>> s
       <ComplexSignal1D, title: , dimensions: (20, 10|100)>
