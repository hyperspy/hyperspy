.. _signal.indexing:

Indexing
--------

Indexing a :class:`~.api.signals.BaseSignal`  provides a powerful, convenient and
Pythonic way to access and modify its data. In HyperSpy indexing is achieved
using :attr:`~.api.signals.BaseSignal.isig` and :attr:`~.api.signals.BaseSignal.inav`,
which allow the navigation and signal dimensions to be indexed independently.
The idea is essentially to specify a subset of the data based on its position
in the array and it is therefore essential to know the convention adopted for
specifying that position, which is described here.

Those new to Python may find indexing a somewhat esoteric concept but once
mastered it is one of the most powerful features of Python based code and
greatly simplifies many common tasks. HyperSpy's Signal indexing is similar
to numpy array indexing and those new to Python are encouraged to read the
associated `numpy documentation on the subject <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_.


Key features of indexing in HyperSpy are as follows (note that some of these
features differ from numpy):

* HyperSpy indexing does:

  + Allow independent indexing of signal and navigation dimensions
  + Support indexing with decimal numbers.
  + Support indexing with units.
  + Support indexing with relative coordinates i.e. 'rel0.5'
  + Use the image order for indexing i.e. [x, y, z,...] (HyperSpy) vs
    [..., z, y, x] (numpy)

* HyperSpy indexing does not:

  + Support indexing using arrays.
  + Allow the addition of new axes using the newaxis object.

The examples below illustrate a range of common indexing tasks.

First consider indexing a single spectrum, which has only one signal dimension
(and no navigation dimensions) so we use :attr:`~.api.signals.BaseSignal.isig`:

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

Unlike numpy, HyperSpy supports indexing using decimal numbers or strings
(containing a decimal number and units), in which case
HyperSpy indexes using the axis scales instead of the indices. Additionally,
one can index using relative coordinates, for example ``'rel0.5'`` to index the
middle of the axis.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(10))
    >>> s
    <Signal1D, title: , dimensions: (|10)>
    >>> s.data
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> s.axes_manager[0].scale = 0.5
    >>> s.axes_manager[0].axis
    array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])
    >>> s.isig[0.5:4.].data
    array([1, 2, 3, 4, 5, 6, 7])
    >>> s.isig[0.5:4].data
    array([1, 2, 3])
    >>> s.isig[0.5:4:2].data
    array([1, 3])
    >>> s.axes_manager[0].units = 'µm'
    >>> s.isig[:'2000 nm'].data
    array([0, 1, 2, 3])
    >>> s.isig[:'rel0.5'].data
    array([0, 1, 2, 3])

Importantly the original :class:`~.api.signals.BaseSignal` and its "indexed self"
share their data and, therefore, modifying the value of the data in one
modifies the same value in the other. Note also that in the example below
s.data is used to access the data as a numpy array directly and this array is
then indexed using numpy indexing.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(10))
    >>> s
    <Signal1D, title: , dimensions: (|10)>
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
data treating navigation axes using :attr:`~.api.signals.BaseSignal.inav`
and signal axes using :attr:`~.api.signals.BaseSignal.isig`.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(2*3*4).reshape((2,3,4)))
    >>> s
    <Signal1D, title: , dimensions: (3, 2|4)>
    >>> s.data # doctest: +SKIP
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
    <Axes manager, axes: (|4)>
                Name |   size |  index |  offset |   scale |  units
    ================ | ====== | ====== | ======= | ======= | ======
    ---------------- | ------ | ------ | ------- | ------- | ------
                   t |      4 |      0 |       0 |       1 | <undefined>
    >>> s.inav[0,0].isig[::-1].data
    array([3, 2, 1, 0])
    >>> s.isig[0]
    <BaseSignal, title: , dimensions: (3, 2|)>
    >>> s.isig[0].axes_manager
    <Axes manager, axes: (3, 2|)>
                Name |   size |  index |  offset |   scale |  units
    ================ | ====== | ====== | ======= | ======= | ======
                   x |      3 |      0 |       0 |       1 | <undefined>
                   y |      2 |      0 |       0 |       1 | <undefined>
    ---------------- | ------ | ------ | ------- | ------- | ------
    >>> s.isig[0].data
    array([[ 0,  4,  8],
       [12, 16, 20]])

Independent indexation of the signal and navigation dimensions is demonstrated
further in the following:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(2*3*4).reshape((2,3,4)))
    >>> s
    <Signal1D, title: , dimensions: (3, 2|4)>
    >>> s.data # doctest: +SKIP
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
    <Axes manager, axes: (|4)>
                Name |   size |  index |  offset |   scale |  units
    ================ | ====== | ====== | ======= | ======= | ======
    ---------------- | ------ | ------ | ------- | ------- | ------
                   t |      4 |      0 |       0 |       1 | <undefined>
    >>> s.isig[0]
    <BaseSignal, title: , dimensions: (3, 2|)>
    >>> s.isig[0].axes_manager
    <Axes manager, axes: (3, 2|)>
                Name |   size |  index |  offset |   scale |  units
    ================ | ====== | ====== | ======= | ======= | ======
                   x |      3 |      0 |       0 |       1 | <undefined>
                   y |      2 |      0 |       0 |       1 | <undefined>
    ---------------- | ------ | ------ | ------- | ------- | ------
    >>> s.isig[0].data
    array([[ 0,  4,  8],
       [12, 16, 20]])


The same syntax can be used to set the data values in signal and navigation
dimensions respectively:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(2*3*4).reshape((2,3,4)))
    >>> s
    <Signal1D, title: , dimensions: (3, 2|4)>
    >>> s.data # doctest: +SKIP
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
    >>> s.inav[0,0] = s.inav[1,1]
    >>> s.inav[0,0].data
    array([16, 17, 18, 19])
