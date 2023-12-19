.. _interactive-label:

Interactive Operations
**********************

The function :func:`~.api.interactive` simplifies the definition of
operations that are automatically updated when an event is triggered. By
default the operation is recomputed when the data or the axes of the original
signal is changed.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(10.))
    >>> ssum = hs.interactive(s.sum, axis=0)
    >>> ssum.data
    array([45.])
    >>> s.data /= 10
    >>> s.events.data_changed.trigger(s)
    >>> ssum.data
    array([4.5])

Interactive operations can be performed in a chain.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(2 * 3 * 4).reshape((2, 3, 4)))
    >>> ssum = hs.interactive(s.sum, axis=0)
    >>> ssum_mean = hs.interactive(ssum.mean, axis=0)
    >>> ssum_mean.data
    array([30.,  33.,  36.,  39.])
    >>> s.data # doctest: +SKIP
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],

           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> s.data *= 10
    >>> s.events.data_changed.trigger(obj=s)
    >>> ssum_mean.data
    array([300.,  330.,  360.,  390.])
