Speeding up operations
----------------------

Reusing a Signal for output
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many signal methods create and return a new signal. For fast operations, the
new signal creation time is non-negligible. Also, when the operation is
repeated many times, for example in a loop, the cumulative creation time can
become significant. Therefore, many operations on
:class:`~.api.signals.BaseSignal` accept an optional argument `out`. If an
existing signal is passed to `out`, the function output will be placed into
that signal, instead of being returned in a new signal.  The following example
shows how to use this feature to slice a :class:`~.api.signals.BaseSignal`. It is
important to know that the :class:`~.api.signals.BaseSignal` instance passed in
the `out` argument must be well-suited for the purpose. Often this means that
it must have the same axes and data shape as the
:class:`~.api.signals.BaseSignal` that would normally be returned by the
operation.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(10))
    >>> s_sum = s.sum(0)
    >>> s_sum.data
    array([45])
    >>> s.isig[:5].sum(0, out=s_sum)
    >>> s_sum.data
    array([10])
    >>> s_roi = s.isig[:3]
    >>> s_roi
    <Signal1D, title: , dimensions: (|3)>
    >>> s.isig.__getitem__(slice(None, 5), out=s_roi)
    >>> s_roi
    <Signal1D, title: , dimensions: (|5)>
