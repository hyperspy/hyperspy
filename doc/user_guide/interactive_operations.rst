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

.. _interactive_cleanup-label:

Cleaning up
===========

When an interactive operation is no longer needed, call the
``Interactive.close`` method to disconnect all
event handlers and release internal references. This prevents the
operation from continuing to respond to changes in the original signal
and avoids leaking memory through lingering event callbacks.

.. code-block:: python

    >>> from hyperspy.interactive import Interactive
    >>> s = hs.signals.Signal1D(np.arange(10.))
    >>> ssum = Interactive(s.sum, axis=0)
    >>> ssum.out.data
    array([45.])
    >>> s.data /= 10
    >>> ssum.close()
    >>> s.events.data_changed.trigger(s)
    >>> ssum.out.data  # no change after close
    array([45.])
