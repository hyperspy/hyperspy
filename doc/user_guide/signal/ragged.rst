.. _signal.ragged:

Ragged signals
--------------

A ragged array (also called jagged array) is an array created with
sequences-of-sequences, where the nested sequences don't have the same length.
For example, a numpy ragged array can be created as follow:

.. code-block:: python

    >>> arr = np.array([[1, 2, 3], [1]], dtype=object)
    >>> arr
    array([list([1, 2, 3]), list([1])], dtype=object)

Note that the array shape is (2, ):

.. code-block:: python

    >>> arr.shape
    (2,)


Numpy ragged array must have python ``object`` type to allow the variable length of
the nested sequences - here ``[1, 2, 3]`` and ``[1]``. As explained in
`NEP-34 <https://numpy.org/neps/nep-0034-infer-dtype-is-object.html>`_,
``dtype=object`` needs to be specified when creating the array to avoid ambiguity
about the shape of the array.

HyperSpy supports the use of ragged array with the following conditions:

- The signal must be explicitly defined as being :attr:`~.api.signals.BaseSignal.ragged`, either when creating
  the signal or by changing the ragged attribute of the signal
- The signal dimension is the variable length dimension of the array
- The :attr:`~.api.signals.BaseSignal.isig` syntax is not supported
- Signal with ragged array can't be transposed
- Signal with ragged array can't be plotted

To create a hyperspy signal of a numpy ragged array:

.. code-block:: python

    >>> s = hs.signals.BaseSignal(arr, ragged=True)
    >>> s
    <BaseSignal, title: , dimensions: (2|ragged)>

    >>> s.ragged
    True

    >>> s.axes_manager
    <Axes manager, axes: (2|ragged)>
                Name |   size |  index |  offset |   scale |  units
    ================ | ====== | ====== | ======= | ======= | ======
        <undefined> |      2 |      0 |       0 |       1 | <undefined>
    ---------------- | ------ | ------ | ------- | ------- | ------
        Ragged axis |               Variable length

.. note::
    When possible, numpy will cast sequences-of-sequences to "non-ragged" array:

    .. code-block:: python

        >>> arr = np.array([np.array([1, 2]), np.array([1, 2])], dtype=object)
        >>> arr
        array([[1, 2],
                [1, 2]], dtype=object)


    Unlike in the previous example, here the array is not ragged, because
    the length of the nested sequences are equal (2) and numpy will create
    an array of shape (2, 2) instead of (2, ) as in the previous example of
    ragged array

    .. code-block:: python

        >>> arr.shape
        (2, 2)

In addition to the use of the keyword ``ragged`` when creating an hyperspy
signal, the :attr:`~.api.signals.BaseSignal.ragged` attribute can also
be set to specify whether the signal contains a ragged array or not.

In the following example, an hyperspy signal is created without specifying that
the array is ragged. In this case, the signal dimension is 2, which *can be*
misleading, because each item contains a list of numbers. To provide a unambiguous
representation of the fact that the signal contains a ragged array, the
:attr:`~.api.signals.BaseSignal.ragged` attribute can be set to ``True``.
By doing so, the signal space will be described as "ragged" and the navigation shape
will become the same as the shape of the ragged array:

.. code-block:: python

    >>> arr = np.array([[1, 2, 3], [1]], dtype=object)
    >>> s = hs.signals.BaseSignal(arr)
    >>> s
    <BaseSignal, title: , dimensions: (|2)>

    >>> s.ragged = True
    >>> s
    <BaseSignal, title: , dimensions: (2|ragged)>
