.. _gpu_processing:

GPU support
-----------

.. versionadded:: 1.7

GPU processing is supported thanks to the numpy dispatch mechanism of array functions
- read `NEP-18 <https://numpy.org/neps/nep-0018-array-function-protocol.html>`_
and `NEP-35 <https://numpy.org/neps/nep-0035-array-creation-dispatch-with-array-function.html>`_
for more information. It means that most HyperSpy functions will work on a GPU
if the data is a :class:`cupy.ndarray` and the required functions are
implemented in ``cupy``.

.. note::
    GPU processing with hyperspy requires numpy>=1.20 and dask>=2021.3.0, to be
    able to use NEP-18 and NEP-35.

.. code-block:: python

    >>> import cupy as cp # doctest: +SKIP
    >>> # Create a cupy array (on GPU device)
    >>> data = cp.random.random(size=(20, 20, 100, 100)) # doctest: +SKIP
    >>> s = hs.signals.Signal2D(data) # doctest: +SKIP
    >>> type(s.data) # doctest: +SKIP
    ... cupy._core.core.ndarray

Two convenience methods are available to transfer data between the host and
the (GPU) device memory:

- :meth:`~.api.signals.BaseSignal.to_host`
- :meth:`~.api.signals.BaseSignal.to_device`

For lazy processing, see the :ref:`corresponding section<big_data.gpu>`.
