.. _big-data-label:

Working with big data
*********************

.. versionadded:: 1.2

HyperSpy makes it possible to analyse data larger than the available memory by
providing "lazy" versions of most of its signals and functions. In most cases
the syntax remains the same. This chapter describes how to work with data
larger than memory using the :py:class:`~._signals.lazy.LazySignal` class and
its derivatives.


Creating Lazy Signals
---------------------

Lazy Signals from external data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the data is large and not loaded by HyperSpy (for example a ``hdf5.Dataset``
or similar), first wrap it in ``dask.array.Array`` as shown `here
<https://dask.readthedocs.io/en/latest/array-creation.html>`_ and then pass it
as normal and call ``as_lazy()``:

.. code-block:: python

    >>> import h5py
    >>> f = h5py.File("myfile.hdf5")           # Load the file
    >>> data = f['/data/path']                 # Get the data
    >>> import dask.array as da                # Import dask to wrap
    >>> chunks = (1000,100)                    # Chunk as appropriate
    >>> x = da.from_array(data, chunks=chunks) # Wrap the data in dask
    >>> s = hs.signals.Signal1D(x).as_lazy() # Create the lazy signal


Loading lazily
^^^^^^^^^^^^^^

To load the data lazily, pass the keyword ``lazy=True``.  As an example,
loading a 34.9 GB ``.blo`` file on a regular laptop might look like:

.. code-block:: python

    >>> s = hs.load("shish26.02-6.blo", lazy=True)
    >>> s
    <LazySignal2D, title: , dimensions: (400, 333|512, 512)>
    >>> s.data
    dask.array<array-e..., shape=(333, 400, 512, 512), dtype=uint8, chunksize=(20, 12, 512, 512)>
    >>> print(s.data.dtype, s.data.nbytes / 1e9)
    uint8 34.9175808
    >>> s.change_dtype("float") # To be able to perform decomposition, etc.
    >>> print(s.data.dtype, s.data.nbytes / 1e9)
    float64 279.3406464

Loading the dataset in the original unsigned integer format would require
around 35GB of memory. To store it in a floating-point format one would need
almost 280GB of memory. However, with the lazy processing both of these steps
are near-instantaneous and require very little computational resources.

Lazy stacking
^^^^^^^^^^^^^

Occasionally the full dataset consists of many smaller files. To combine them
into a one large ``LazySignal``, we can :ref:`stack<signal.stack_split>` them
lazily (both when loading or afterwards):

.. code-block:: python

    >>> siglist = hs.load("*.hdf5")
    >>> s = hs.stack(siglist, lazy=True)
    >>> # Or load lazily and stack afterwards:
    >>> siglist = hs.load("*.hdf5", lazy=True)
    >>> s = hs.stack(siglist) # no need to pass 'lazy', as signals already lazy
    >>> # Or do everything in one go:
    >>> s = hs.load("*.hdf5", lazy=True, stack=True)

Casting signals as lazy
^^^^^^^^^^^^^^^^^^^^^^^

To convert a regular HyperSpy signal to a lazy one such that any future
operations are only performed lazily, use the
:py:meth:`~.signal.BaseSignal.as_lazy` method:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(150.).reshape((3, 50)))
    >>> s
    <Signal1D, title: , dimensions: (3|50)>
    >>> sl = s.as_lazy()
    >>> sl
    <LazySignal1D, title: , dimensions: (3|50)>

Practical tips
--------------

Despite the limitations detailed below, most HyperSpy operations can be
performed lazily. Importand points of note are:

Computing lazy signals
^^^^^^^^^^^^^^^^^^^^^^

Upon saving lazy signals, the result of computations is stored on disk.

In order to store the lazy signal in memory (i.e. make it a normal HyperSpy
signal) it has a :py:meth:`~._signals.lazy.LazySignal.compute` method:

.. code-block:: python

    >>> s
    <LazySignal2D, title: , dimensions: (|512, 512)>
    >>> s.compute()
    [########################################] | 100% Completed |  0.1s
    >>> s
    <Signal2D, title: , dimensions: (|512, 512)>


Navigator plot
^^^^^^^^^^^^^^

The default signal navigator is the sum of the signal across all signal
dimensions and all but 1 or 2 navigation dimensions. If the dataset is large,
this can take a significant amount of time to perform with every plot. A more
convenient alternative is to calculate the summed navigation signal manually
once, and only pass it for all other plots. Pay attention to the transpose
(``.T``):

.. code-block:: python

    >>> s
    <LazySignal2D, title: , dimensions: (200, 200|512, 512)>
    >>> # for fastest results, just pick one signal space pixel
    >>> nav = s.isig[256, 256].T
    >>> # Alternatively, sum as per default behaviour
    >>> nav = s.sum(s.axes_manager.signal_axes).T
    >>> nav
    <LazySignal2D, title: , dimensions: (|200, 200)>
    >>> # Compute the result
    >>> nav.compute()
    [########################################] | 100% Completed | 13.1s
    >>> s.plot(navigator=nav)

Alternatively, it is possible to not have a navigator, and use sliders
instead:

.. code-block:: python

    >>> s
    <LazySignal2D, title: , dimensions: (200, 200|512, 512)>
    >>> s.plot(navigator='slider')



Limitations
-----------

Most operations can be performed lazily. However, lazy operations come with
a few limitations and constraints that we detail below.

Immutable signals
^^^^^^^^^^^^^^^^^

An important limitation when using ``LazySignal`` is the inability to modify
existing data (immutability). This is a logical consequence of the DAG (tree
structure, explained in :ref:`lazy_details`), where a complete history of the
processing has to be stored to traverse later.

In fact, lazy evaluation removes the need for such operation, since only
additional tree branches are added, requiring very little resources. In
practical terms the following fails with lazy signals:

.. code-block:: python

    >>> s = hs.signals.BaseSignal([0]).as_lazy()
    >>> s += 1
    Traceback (most recent call last):
      File "<ipython-input-6-1bd1db4187be>", line 1, in <module>
        s += 1
      File "<string>", line 2, in __iadd__
      File "/home/fjd29/Python/hyperspy3/hyperspy/signal.py", line 1591, in _binary_operator_ruler
        getattr(self.data, op_name)(other)
    AttributeError: 'Array' object has no attribute '__iadd__'

However, when operating lazily there is no clear benefit to using in-place
operations. So, the operation above could be rewritten as follows:

.. code-block:: python

    >>> s = hs.signals.BaseSignal([0]).as_lazy()
    >>> s = s + 1

Or even better:

.. code-block:: python

    >>> s = hs.signals.BaseSignal([0]).as_lazy()
    >>> s1 = s + 1

Machine learning (decomposition)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`decomposition` algorithms often performs large matrix manipulations,
requiring significantly more memory than the data size. To perform
decomposition operation lazily HyperSpy provides several "online" algorithms.
These algorithms perform the decomposition by operating serially on chunks of
data, enabling the lazy decomposition of large datasets.

In line with the standard HyperSpy signals,
:py:meth:`~._signals.lazy.LazySignal.decomposition` offers  the following
implementations:

* **PCA** (``algorithm='PCA'``): performs `IncrementalPCA <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA>`_
  from ``scikit-learn``.
* **ORPCA** (``algorithm='ORPCA'``): performs Online Robust PCA. (It is also available
  for regular signals.)
* **NMF** (``algorithm='ONMF'``): performs Online Robust NMF, as per "OPGD"
  algorithm in [Zhao2016]_.

Other minor differences
^^^^^^^^^^^^^^^^^^^^^^^

* **Histograms** for a ``LazySignal`` do not support ``knuth`` and ``blocks``
  binning algorithms.
* **CircleROI** sets the elements outside the ROI to ``np.nan`` instead of
  using a masked array, because ``dask`` does not support masking. As a
  convenience, ``nansum``, ``nanmean`` and other ``nan*`` signal methods were
  added to mimic the workflow as closely as possible.


.. _lazy_details:

Behind the scenes --technical details
-------------------------------------

Standard HyperSpy signals load the data into memory for fast access and
processing. While this behaviour gives good performance in terms of speed, it
obviously requires at least as much computer memory as the dataset, and often
twice that to store the results of subsequent computations. This can become a
significant problem when processing very large datasets on consumer-oriented
hardware.

HyperSpy offers a solution for this problem by including
:py:class:`~._signals.lazy.LazySignal` and its derivatives. The main idea of
these classes is to perform any operation (as the name suggests)
`lazily <https://en.wikipedia.org/wiki/Lazy_evaluation>`_ (delaying the
execution until the result is requested (e.g. saved, plotted)) and in a
`blocked fashion <https://en.wikipedia.org/wiki/Block_matrix>`_. This is
achieved by building a "history tree" (formally called a Directed Acyclic Graph
(DAG)) of the computations, where the original data is at the root, and any
further operations branch from it. Only when a certain branch result is
requested, the way to the root is found and evaluated in the correct sequence
on the correct blocks.

The "magic" is performed by (for the sake of simplicity) storing the data not
as ``numpy.ndarray``, but ``dask.array.Array`` (more information `here
<https://dask.readthedocs.io/en/latest/>`_). ``dask`` offers a couple of
advantages:

* **Arbitrary-sized data processing is possible**. By only loading a couple of
  chunks at a time, theoretically any signal can be processed, albeit slower.
  In practice, this may be limited: (i) some operations may require certain
  chunking pattern, which may still saturate memory; (ii) many chunks should
  fit into the computer memory comfortably at the same time.
* **Loading only the required data**. If a certain part (chunk) of the data is
  not required for the final result, it will not be loaded at all, saving time
  and resources.
* **Able to extend to a distributed computing environment (clusters)**.
  ``dask.distributed`` (documentation `here
  <https://distributed.readthedocs.io/en/latest/>`_) offers a straightforward
  way to expand the effective memory for computations to that of a cluster,
  which allows performing the operations significantly faster than on a single
  machine.
