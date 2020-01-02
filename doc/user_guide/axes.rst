Axes handling
*************


The navigation and signal dimensions
------------------------------------

In HyperSpy, the data is interpreted as a signal array and, therefore, the data
axes are not equivalent. HyperSpy distinguishes between *signal* and
*navigation* axes and most functions operate on the *signal* axes and
iterate over the *navigation* axes. For example, an EELS spectrum image (i.e.
a 2D array of spectra) has three dimensions: X, Y and energy-loss. In
HyperSpy, X and Y are the *navigation* dimensions and the energy-loss is the
*signal* dimension. To make this distinction more explicit, the
representation of the object includes a separator ``|`` between the
navigation and signal dimensions.

For example: A spectrum image has signal dimension 1 and navigation dimension 2
and is stored in the Signal1D subclass.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.zeros((10, 20, 30)))
    >>> s
    <Signal1D, title: , dimensions: (20, 10|30)>


An image stack has signal dimension 2 and navigation dimension 1 and is stored
in the Signal2D subclass.

.. code-block:: python

    >>> im = hs.signals.Signal2D(np.zeros((30, 10, 20)))
    >>> im
    <Signal2D, title: , dimensions: (30|20, 10)>


Axes storage and ordering
^^^^^^^^^^^^^^^^^^^^^^^^^

Note that HyperSpy rearranges the axes when compared to the array order. The
following few paragraphs explain how and why.

Depending on how the array is arranged, some axes are faster to iterate than
others. Consider an example of a book as the dataset in question. It is
trivially simple to look at letters in a line, and then lines down the page,
and finally pages in the whole book.  However, if your words are written
vertically, it can be inconvenient to read top-down (the lines are still
horizontal, it's just the meaning that's vertical!). It is very time-consuming
if every letter is on a different page, and for every word you have to turn 5-6
pages. Exactly the same idea applies here - in order to iterate through the
data (most often for plotting, but for any other operation as well), you
want to keep it ordered for "fast access".

In Python (more explicitly `numpy`), the "fast axes order" is `C order` (also
called row-major order). This means that the **last** axis of a numpy array is
fastest to iterate over (i.e. the lines in the book). An alternative ordering
convention is `F order` (column-major), where it is the other way round: the
first axis of an array is the fastest to iterate over. In both cases, the
further an axis is from the `fast axis` the slower it is to iterate over this
axis. In the book analogy, you could think about reading the first lines of
all pages, then the second and so on.

When data is acquired sequentially, it is usually stored in acquisition order.
When a dataset is loaded, HyperSpy generally stores it in memory in the same
order, which is good for the computer. However, HyperSpy will reorder and
classify the axes to make it easier for humans. Let's imagine a single numpy
array that contains pictures of a scene acquired with different exposure times
on different days. In numpy, the array dimensions are  ``(D, E, Y, X)``. This
order makes it fast to iterate over the images in the order in which they were
acquired. From a human point of view, this dataset is just a collection of
images, so HyperSpy first classifies the image axes (``X`` and ``Y``) as
`signal axes` and the remaining axes the `navigation axes`. Then it reverses
the order of each set of axes because many humans are used to get the ``X``
axis first and, more generally, the axes in acquisition order from left to
right. So, the same axes in HyperSpy are displayed like this: ``(E, D | X,
Y)``.

Extending this to arbitrary dimensions, by default, we reverse the numpy axes,
chop them into two chunks (signal and navigation), and then swap those chunks,
at least when printing. As an example:

.. code-block:: bash
    (a1, a2, a3, a4, a5, a6) # original (numpy)
    (a6, a5, a4, a3, a2, a1) # reverse
    (a6, a5) (a4, a3, a2, a1) # chop
    (a4, a3, a2, a1) (a6, a5) # swap (HyperSpy)

In the background, HyperSpy also takes care of storing the data in memory in
a "machine-friendly" way, so that iterating over the navigation axes is always
fast.


.. _Setting_axis_properties:

Setting axis properties
-----------------------

The axes are managed and stored by the :py:class:`~.axes.AxesManager` class
that is stored in the :py:attr:`~.signal.BaseSignal.axes_manager` attribute of
the signal class. The individual axes can be accessed by indexing the
AxesManager, e.g.:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.random.random((10, 20 , 100)))
    >>> s
    <Signal1D, title: , dimensions: (20, 10|100)>
    >>> s.axes_manager
    <Axes manager, axes: (<Unnamed 0th axis, size: 20, index: 0>, <Unnamed 1st
    axis, size: 10, index: 0>|<Unnamed 2nd axis, size: 100>)>
    >>> s.axes_manager[0]
    <Unnamed 0th axis, size: 20, index: 0>

The navigation axes come first, followed by the signal axes. Alternatively, 
it is possible to selectively access the navigation or signal dimensions:

.. code-block:: python

    >>> s.axes_manager.navigation_axes[1]
    <Unnamed 1st axis, size: 10, index: 0>
    >>> s.axes_manager.signal_axes[0]
    <Unnamed 2nd axis, size: 100>

For the given example of two navigation and one signal dimensions, all the
following commands will access the same axis:

.. code-block:: python

    >>> s.axes_manager[2]
    >>> s.axes_manager[-1]
    >>> s.axes_manager.signal_axes[0]

The axis properties can be set by setting the :py:class:`~.axes.BaseDataAxis`
attributes, e.g.:

.. code-block:: python

    >>> s.axes_manager[0].name = "X"
    >>> s.axes_manager[0]
    <X axis, size: 20, index: 0>


Once the name of an axis has been defined it is possible to request it by its
name e.g.:

.. code-block:: python

    >>> s.axes_manager["X"]
    <X axis, size: 20, index: 0>
    >>> s.axes_manager["X"].scale = 0.2
    >>> s.axes_manager["X"].units = "nm"
    >>> s.axes_manager["X"].offset = 100


It is also possible to set the axes properties using a GUI by calling the
:py:meth:`~.axes.AxesManager.gui` method of the :py:class:`~.axes.AxesManager`

.. code-block:: python

    >>> s.axes_manager.gui()

.. _axes_manager_gui_image:

.. figure::  images/axes_manager_gui_ipywidgets.png
   :align:   center

   AxesManager ipywidgets GUI.

or, for a specific axis, the respective method of e.g.
:py:class:`~.axes.LinearDataAxis`:

.. code-block:: python

    >>> s.axes_manager["X"].gui()

.. _data_axis_gui_image:

.. figure::  images/data_axis_gui_ipywidgets.png
   :align:   center

   LinearDataAxis ipywidgets GUI.

To simply change the "current position" (i.e. the indices of the navigation
axes) you could use the navigation sliders:

.. code-block:: python

    >>> s.axes_manager.gui_navigation_sliders()

.. _navigation_sliders_image:

.. figure::  images/axes_manager_navigation_sliders_ipywidgets.png
   :align:   center

   Navigation sliders ipywidgets GUI.

Alternatively, the "current position" can be changed programmatically by
directly accessing the ``indices`` attribute of a signal's
:py:class:`~.axes.AxesManager` or the ``index`` attribute of an individual
axis. This is particularly useful when trying to set
a specific location at which to initialize a model's parameters to
sensible values before performing a fit over an entire spectrum image. The
``indices`` must be provided as a tuple, with the same length as the number of
navigation dimensions:

.. code-block:: python

    >>> s.axes_manager.indices = (5, 4)


Summary of axis properties
^^^^^^^^^^^^^^^^^^^^^

* ``name`` (str) and ``units`` (str) are basic parameters describing an axis
  used in plotting. The latter enables the :ref:`conversion of units
  <quantity_and_converting_units>`.
* ``navigate`` (bool) determines, whether it is a navigation axis.
* ``size`` (int) gives the number of elements in an axis.
* ``index`` (int) determines the "current position for a navigation axis and
  ``value`` (float) returns the value at this position.
* ``low_index`` (int) and ``high_index`` (int) are the first and last index.
* ``low_value`` (int) and ``high_value`` (int) are the smallest and largest
  value.
* The ``axis`` (array) vector stores the values of the axis points. However,
  depending on the type of axis, this vector may be updated from the **defining
  attributes** as discussed in the following section.


.. _Axes_types:

Types of data axes
------------------

HyperSpy supports different *data axis types*, which differ in how the axis is
defined: 

* :py:class:`~.axes.DataAxis` defined by a vector ``axis``, 
* :py:class:`~.axes.FunctionalDataAxis` defined by a function ``expression`` or 
* :py:class:`~.axes.LinearDataAxis` defined by the initial value ``offset``
and spacing ``scale``.

The main disambiguation is whether the
axis is **linear**, where the data points are equidistantly spaced, or
**non linear**, where the spacing may vary. The latter can become important
when, e.g., a spectrum recorded over a *wavelength* axis is converted to a
*wavenumber* or *energy* scale, where the conversion is based on a ``1/x``
dependence so that the axis spacing of the new axis varies along the length
of the axis. Whether an axis is linear or not can be queried through the 
property ``is_linear`` (bool) of the axis.

Every axis of a signal object may be of a different type. For example, it will
be common that the *navigation* axes are *linear*, while the *signal* axis is
*non linear*.

When an axis is created, the type is automatically determined by the attributes
passed to the generator. The three different axis types are summarized in the
following table.

.. table:: BaseDataAxis subclasses.

    +-------------------------------------------------------------------+------------------------+-------------+
    |                   BaseDataAxis subclass                           |  defining attributes   |  is_linear  |
    +===================================================================+========================+=============+
    |                :py:class:`~.axes.DataAxis`                        |         axis           |  False      |
    +-------------------------------------------------------------------+------------------------+-------------+
    |           :py:class:`~.axes.FunctionalDataAxis`                   |      expression        |  False      |
    +-------------------------------------------------------------------+------------------------+-------------+
    |             :py:class:`~.axes.LinearDataAxis`                     |    offset, scale       |  True       |
    +-------------------------------------------------------------------+------------------------+-------------+    

.. NOTE::

    Certain functionalities require the ``offest`` and ``scale`` parameters of
    a ``LinearDataAxis`` and thus may not support the non linear axis types.


Linear data axis
^^^^^^^^^^^^^^^^

The most common case is the :py:class:`~.axes.LinearDataAxis`. Here, the axis
is defined by the ``offset`` and ``scale`` parameters, which determine the
`initial value` and `spacing`, respectively. The actual ``axis`` vector is
automatically calculated from these two values. The ``LinearDataAxis`` is a
special case of the ``FunctionalDataAxis`` defined by the function
``scale * x + offset``.

Sample dictionary for a :py:class:`~.axes.LinearDataAxis`:

.. code-block:: python

    >>> dict0 = {'offset': 300, 'scale': 1, 'size': 500}
    >>> s = hs.signals.Signal1D(np.ones(500), axes=[dict0])
    >>> s.axes_manager[0].get_axis_dictionary()
    {'name': <undefined>,
    'units': <undefined>,
    'navigate': False,
    'size': 500,
    'scale': 1,
    'offset': 300}

Corresponding output of :py:class:`~.axes.AxesManager`:

.. code-block:: python

    >>> s.axes_manager
    < Axes manager, axes: (|1000) >
                Name |   size |  index |  offset |   scale |  units
    ================ | ====== | ====== | ======= | ======= | ======
    ---------------- | ------ | ------ | ------- | ------- | ------
                     |    500 |        |     300 |       1 |       


Functional data axis
^^^^^^^^^^^^^^^^^^^^

Alternatively, a :py:class:`~.axes.FunctionalDataAxis` is defined based on an
``expression`` that is evaluated to yield the axis points. The `expression`
is a function defined as a ``string`` using the
`SymPy <https://docs.sympy.org/latest/tutorial/intro.html>`_ text expression
format. An example would be ``expression = a / x + b``. Any variables in the
expression, in this case ``a`` and ``b`` must be defined as additional
attributes of the axis. The property ``is_linear`` is automatically set to
``False``.

By default, the axis is built using a vector ``x = np.arange(size)``. However,
the expression can also reference a vector ``x0`` that contains an array of 
`x-values` at which to evaluate `expression`. For example: ``expression = '1240
/ x0', x0 = np.arange(300,400,0.5)``

Sample dictionary for a :py:class:`~.axes.FunctionalDataAxis`:

.. code-block:: python

    >>> dict0 = {'expression': 'a / (x + 1) + b', 'a': 100, 'b': 10, 'size': 500}
    >>> s = hs.signals.Signal1D(np.ones(500), axes=[dict0])
    >>> s.axes_manager[0].get_axis_dictionary()
    {'name': <undefined>,
    'units': <undefined>,
    'navigate': False,
    'expression': 'a / (x +1) + b',
    'size': 500,
    'a': 100,
    'b': 10}

Corresponding output of :py:class:`~.axes.AxesManager`:

.. code-block:: python

    >>> s.axes_manager
    < Axes manager, axes: (|1000) >
                Name |   size |  index |          offset |           scale |  units
    ================ | ====== | ====== | =============== | =============== | ======
    ---------------- | ------ | ------ | --------------- | --------------- | ------
                     |    500 |        | non linear axis | non linear axis |       


(Non linear) Data axis
^^^^^^^^^^^^^^^^^^^^^^

A :py:class:`~.axes.DataAxis` is the most flexible type of axis. The axis
points are directly given by a vector named ``axis``. As this can be any
vector, the property ``is_linear`` is automatically set to ``False``.


Sample dictionary for a :py:class:`~.axes.DataAxis`:

.. code-block:: python

    >>> dict0 = {'axis': np.arange(12)**2}
    >>> s = hs.signals.Signal1D(np.ones(12), axes=[dict0])
    >>> s.axes_manager[0].get_axis_dictionary()
    {'name': <undefined>,
    'units': <undefined>,
    'navigate': False,
    'axis': array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121])}

Corresponding output of :py:class:`~.axes.AxesManager`:

.. code-block:: python

    >>> s.axes_manager
    < Axes manager, axes: (|1000) >
                Name |   size |  index |  offset |   scale |  units
    ================ | ====== | ====== | ======= | ======= | ======
    ---------------- | ------ | ------ | ------- | ------- | ------
                     |     12 |        |     300 |       1 |       


Defining a new axis
-------------------

An axis object can be created through the ``axes.create_axis()`` method, which
automatically determines the type of axis by the given attributes:

.. code-block:: python

    >>> axis = axes.create_axis(offset=10,scale=0.5,size=20)
    >>> axis
    <Unnamed axis, size: 20>
    
Alternatively, the creator of the different types of axes can be called
directly:

.. code-block:: python

    >>> axis = axes.LinearDataAxis(offset=10,scale=0.5,size=20)
    >>> axis
    <Unnamed axis, size: 20>
    
The dictionary defining the axis is returned by the ``get_axis_dictionary()``
method:

.. code-block:: python

    >>> axis.get_axis_dictionary()
    {'name': <undefined>,
    'units': <undefined>,
    'navigate': <undefined>,
    'size': 20,
    'scale': 0.5,
    'offset': 10.0}

This dictionary can be used, for example, in the :ref:`initilization of a new
signal<signal_initialization>`.


.. _quantity_and_converting_units:

Using quantity and converting units
-----------------------------------

The ``scale`` and the ``offset`` of each :py:class:`~.axes.LinearDataAxis` axis
can be set and retrieved as quantity.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(10))
    >>> s.axes_manager[0].scale_as_quantity
    1.0 dimensionless
    >>> s.axes_manager[0].scale_as_quantity = '2.5 µm'
    >>> s.axes_manager
    <Axes manager, axes: (|10)>
                Name |   size |  index |  offset |   scale |  units 
    ================ | ====== | ====== | ======= | ======= | ====== 
    ---------------- | ------ | ------ | ------- | ------- | ------ 
         <undefined> |     10 |        |       0 |     2.5 |     µm
    >>> s.axes_manager[0].offset_as_quantity = '2.5 nm'
    <Axes manager, axes: (|10)>
                Name |   size |  index |  offset |   scale |  units 
    ================ | ====== | ====== | ======= | ======= | ====== 
    ---------------- | ------ | ------ | ------- | ------- | ------ 
         <undefined> |     10 |        |     2.5 | 2.5e+03 |     nm


Internally, HyperSpy uses the `pint <http://pint.readthedocs.io>`_ library to
manage the scale and offset quantities. The ``scale_as_quantity`` and
``offset_as_quantity`` attributes return pint object:

.. code-block:: python

    >>> q = s.axes_manager[0].offset_as_quantity
    >>> type(q) # q is a pint quantity object
    pint.quantity.build_quantity_class.<locals>.Quantity
    >>> q
    2.5 nanometer


The ``convert_units`` method of the :py:class:`~.axes.AxesManager` converts
units, which by default (no parameters provided) converts all axis units to an
optimal unit to avoid using too large or small numbers.

Each axis can also be converted individually using the ``convert_to_units``
method of the :py:class:`~.axes.LinearDataAxis`:

.. code-block:: python

    >>> axis = hs.hyperspy.axes.DataAxis(size=10, scale=0.1, offset=10, units='mm')
    >>> axis.scale_as_quantity
    0.1 millimeter
    >>> axis.convert_to_units('µm')
    >>> axis.scale_as_quantity
    100.0 micrometer

