Basics of signals
-----------------

.. _signal_initialization:

Signal initialization
^^^^^^^^^^^^^^^^^^^^^

Many of the values in the :py:class:`~.axes.AxesManager` can be
set when making the :py:class:`~.api.signals.BaseSignal` object.

.. code-block:: python

    >>> dict0 = {'size': 10, 'name':'Axis0', 'units':'A', 'scale':0.2, 'offset':1}
    >>> dict1 = {'size': 20, 'name':'Axis1', 'units':'B', 'scale':0.1, 'offset':2}
    >>> s = hs.signals.BaseSignal(np.random.random((10,20)), axes=[dict0, dict1])
    >>> s.axes_manager
    <Axes manager, axes: (|20, 10)>
		        Name |   size |  index |  offset |   scale |  units
    ================ | ====== | ====== | ======= | ======= | ======
    ---------------- | ------ | ------ | ------- | ------- | ------
	           Axis1 |     20 |        |       2 |     0.1 |      B
	           Axis0 |     10 |        |       1 |     0.2 |      A

This also applies to the :py:attr:`~.signals.BaseSignal.metadata`.

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
:py:class:`~.api.signals.Signal1D` subclass for this data e.g.:

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
have used a :py:class:`~.api.signals.Signal2D` instead e.g.:

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

Notice that with use :py:class:`~.api.signals.BaseSignal` because there is
no specialised subclass for three-dimensional data. Also note that by default
:py:class:`~.api.signals.BaseSignal` interprets all dimensions as signal dimensions.
We could also configure it to operate on the dataset as a three-dimensional
array of scalars by changing the default *view* of
:py:class:`~.api.signals.BaseSignal` by taking the transpose of it:

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


.. _transforming_signal-label:

Transforming between signal subclasses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`table below <signal_subclasses_table-label>` summarises all the
specialised :py:class:`~.api.signals.BaseSignal` subclasses currently distributed
with HyperSpy.

The :py:mod:`~.api.signals` module, which contains all available signal subclasses,
is imported in the user namespace when loading HyperSpy. In the following
example we create a Signal2D instance from a 2D numpy array:

.. code-block:: python

    >>> im = hs.signals.Signal2D(np.random.random((64,64)))
    >>> im
    <Signal2D, title: , dimensions: (|64, 64)>

The different subclasses are characterized by the `signal_type` metadata attribute,
the data `dtype` and the signal dimension. See the table and diagram below.
`signal_type` describes the nature of the signal. It can be any string, normally the
acronym associated with a particular signal. In certain cases HyperSpy provides
features that are only available for a particular signal type through
:py:class:`~.api.signals.BaseSignal` subclasses. The :py:class:`~.api.signals.BaseSignal` method
:py:meth:`~.api.signals.BaseSignal.set_signal_type` changes the signal_type in place, which
may result in a :py:class:`~.api.signals.BaseSignal` subclass transformation.


Furthermore, the `dtype` of the signal data also affects the subclass assignment. There are
e.g. specialised signal subclasses to handle complex data (see the following diagram).


.. figure::  ../images/HyperSpySignalOverview.png
  :align:   center
  :width:   500

  Diagram showing the inheritance structure of the different subclasses

.. _signal_subclasses_table-label:


.. table:: BaseSignal subclass :py:attr:`~.api.signals.BaseSignal.metadata` attributes.

    +-----------------------------------------------------+------------------+-----------------------+----------+
    |            BaseSignal subclass                      | signal_dimension |  signal_type          |  dtype   |
    +=====================================================+==================+=======================+==========+
    |      :py:class:`~.api.signals.BaseSignal`           |        -         |       -               |  real    |
    +-----------------------------------------------------+------------------+-----------------------+----------+
    |      :py:class:`~.api.signals.Signal1D`             |        1         |       -               |  real    |
    +-----------------------------------------------------+------------------+-----------------------+----------+
    |      :py:class:`~.api.signals.EELSSpectrum`         |        1         |     EELS              |  real    |
    +-----------------------------------------------------+------------------+-----------------------+----------+
    |      :py:class:`~.api.signals.EDSSEMSpectrum`       |        1         |    EDS_SEM            |  real    |
    +-----------------------------------------------------+------------------+-----------------------+----------+
    |      :py:class:`~.api.signals.EDSTEMSpectrum`       |        1         |    EDS_TEM            |  real    |
    +-----------------------------------------------------+------------------+-----------------------+----------+
    |      :py:class:`~.api.signals.Signal2D`             |        2         |       -               |  real    |
    +-----------------------------------------------------+------------------+-----------------------+----------+
    |      :py:class:`~.api.signals.HologramImage`        |        2         |      hologram         |  real    |
    +-----------------------------------------------------+------------------+-----------------------+----------+
    |      :py:class:`~.api.signals.DielectricFunction`   |        1         |  DielectricFunction   |  complex |
    +-----------------------------------------------------+------------------+-----------------------+----------+
    |      :py:class:`~.api.signals.ComplexSignal`        |        -         |       -               | complex  |
    +-----------------------------------------------------+------------------+-----------------------+----------+
    |      :py:class:`~.api.signals.ComplexSignal1D`      |        1         |       -               | complex  |
    +-----------------------------------------------------+------------------+-----------------------+----------+
    |      :py:class:`~.api.signals.ComplexSignal2D`      |        2         |       -               | complex  |
    +-----------------------------------------------------+------------------+-----------------------+----------+

.. versionchanged:: 1.0 ``Simulation``, ``SpectrumSimulation`` and ``ImageSimulation``
   classes removed.

.. versionadded:: 1.5
    External packages can register extra :py:class:`~.api.signals.BaseSignal`
    subclasses.

Note that, if you have :ref:`packages that extend HyperSpy
<hyperspy_extensions-label>` installed in your system, there may
be more specialised signals available to you. To print all available specialised
:py:class:`~.api.signals.BaseSignal` subclasses installed in your system call the
:py:func:`~.api.print_known_signal_types`
function as in the following example:

.. code-block:: python

    >>> hs.print_known_signal_types()
    +--------------------+---------------------+--------------------+----------+
    |    signal_type     |       aliases       |     class name     | package  |
    +--------------------+---------------------+--------------------+----------+
    | DielectricFunction | dielectric function | DielectricFunction | hyperspy |
    |      EDS_SEM       |                     |   EDSSEMSpectrum   | hyperspy |
    |      EDS_TEM       |                     |   EDSTEMSpectrum   | hyperspy |
    |        EELS        |       TEM EELS      |    EELSSpectrum    | hyperspy |
    |      hologram      |                     |   HologramImage    | hyperspy |
    |      MySignal      |                     |      MySignal      | hspy_ext |
    +--------------------+---------------------+--------------------+----------+

.. warning::
    From version 2.0 HyperSpy will no longer ship
    :py:class:`~.api.signals.BaseSignal` subclasses that are specific to a
    particular type of data (i.e. with non-empty ``signal_type``). All those
    signals currently distributed with HyperSpy will be moved to new
    packages.

The following example shows how to transform between different subclasses.

   .. code-block:: python

       >>> s = hs.signals.Signal1D(np.random.random((10,20,100)))
       >>> s
       <Signal1D, title: , dimensions: (20, 10|100)>
       >>> s.metadata
       ├── signal_type =
       └── title =
       >>> im = s.to_signal2D()
       >>> im
       <Signal2D, title: , dimensions: (100|20, 10)>
       >>> im.metadata
       ├── signal_type =
       └── title =
       >>> s.set_signal_type("EELS")
       >>> s
       <EELSSpectrum, title: , dimensions: (20, 10|100)>
       >>> s.change_dtype("complex")
       >>> s
       <ComplexSignal1D, title: , dimensions: (20, 10|100)>
