Slicing Tools
**************

Hyperspy attempts to copy the `numpy fancy indexing`_ approach in most cases but separates the
handling of the signal and navigation axes.

In addition to the numpy indexing/slicing hyperspy also supports indexing or slicing using real units
as long as the axis is ordered, meaning the axis is constantly increasing/decreasing.
For axes which are not ordered, slicing with real units is ambiguous but indexing with
real unit values is still possible.

In the case of slicing with a bool array, or an array of indexes a DataAxis is always returned.

For example:

.. code-block:: python

    >>> s = hs.signals.Signal2D(np.random.rand((10,10,20,10)))
    >>> print(type(s.axes_manager[2]))
    <class 'hyperspy.axes.UniformDataAxis'>
    >>> sliced_s = s.isig[[0,3,6,5,4]]
    >>> print(type(sliced_s.axes_manager[2]))
    <class 'hyperspy.axes.DataAxis'>
    >>>print(sliced_s.axes_manager[2].axis)
    [0. 3. 6. 5. 4.]

    >>> bool_ind = np.zeros(10, dtype=bool)
    >>> bool_ind[0,3,6,5,4] =True
    >>>sliced_s = s.isig[bool_ind]
    >>> print(type(sliced_s.axes_manager[2]))
    <class 'hyperspy.axes.DataAxis'>
    >>>print(sliced_s.axes_manager[2].axis)
    [0. 3. 4. 5. 6.]
    ...

What is happening here is that once the data is no longer uniform it will convert to a DataAxis which can
handle any arrangement of values. In this way no information is lost when slicing or indexing the data. In the
top case you can see the resulting axis is unordered while the bottom case the axis is ordered.

Due to the flexibility of the DataAxis we can also have labeled axes. This is useful for defining vectors,
or other quantities within the framework of a signal.

We can use this quite effectively in the case of trying to return multiple values.
For example lets say we have a set of 2-D images and we want to find the center
of mass as well as the intensity at the Center.

.. code-block:: python

    >>> print(s)
    <Signal2D, title: , dimensions: (10, 10|256, 256)>
    >>> s.map(find_center_of_mass)
    >>> print(s)
    <Signal1D, title: , dimensions: (10, 10|3)>
    >>> s.axes_manager.signal_axes[0].convert_to_non_uniform_axis
    >>> s.axes_manager.signal_axes[0].axis = ["cx","cy", "int"]

    >>> s.isig[["cx", "cy"]] # get only the centers
    ...

This is quite useful for developers, making it easier to represent complex results inside the framework of
some signal as well as for users to better save and convey their results.


.. _numpy fancy indexing: https://numpy.org/doc/stable/user/basics.indexing.html#basics-indexing
