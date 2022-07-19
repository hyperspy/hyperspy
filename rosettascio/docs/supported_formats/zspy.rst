.. _zspy-format:

ZSpy - HyperSpy's Zarr Specification
------------------------------------

Similarly to the :ref:`hspy format <hspy-format>`, the zspy format guarantees that no
information will be lost in the writing process and that supports saving data
of arbitrary dimensions. It is based on the `Zarr project <https://zarr.readthedocs.io/en/stable/index.html>`_. Which exists as a drop in
replacement for hdf5 with the intention to fix some of the speed and scaling
issues with the hdf5 format and is therefore suitable for saving 
:external+hyperspy:ref:`big data <big_data.saving>`.


.. code-block:: python

    >>> s = hs.signals.BaseSignal([0])
    >>> s.save('test.zspy') # will save in nested directory
    >>> hs.load('test.zspy') # loads the directory


When saving to `zspy <https://zarr.readthedocs.io/en/stable/index.html>`_, all supported objects in the signal's
:py:attr:`~.signal.BaseSignal.metadata` is stored. This includes lists, tuples and signals.
Please note that in order to increase saving efficiency and speed, if possible,
the inner-most structures are converted to numpy arrays when saved. This
procedure homogenizes any types of the objects inside, most notably casting
numbers as strings if any other strings are present:

By default, a :py:class:`zarr.storage.NestedDirectoryStore` is used, but other
zarr store can be used by providing a `zarr store <https://zarr.readthedocs.io/en/stable/api/storage.html>`_
instead as argument to the :py:meth:`~.signal.BaseSignal.save` or the
:py:func:`~.io.load` function. If a zspy file has been saved with a different
store, it would need to be loaded by passing a store of the same type:

.. code-block:: python

    >>> import zarr
    >>> filename = 'test.zspy'
    >>> store = zarr.LMDBStore(filename)
    >>> signal.save(store) # saved to LMDB

To load this file again

.. code-block:: python

    >>> import zarr
    >>> filename = 'test.zspy'
    >>> store = zarr.LMDBStore(filename)
    >>> s = hs.load(store) # load from LMDB

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``compressor``: `Numcodecs codec <https://numcodecs.readthedocs.io/en/stable/index.html?>`_,
  a compressor can be passed to the save function to compress the data efficiently. The default
  is to call a Blosc compressor object.

    .. code-block:: python

        >>> from numcodecs import Blosc
        >>> compressor=Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE) # Used by default
        >>> s.save('test.zspy', compressor = compressor) # will save with Blosc compression

    .. note::

        Lazy operations are often i-o bound, reading and writing the data creates a bottle neck in processes
        due to the slow read write speed of many hard disks. In these cases, compressing your data is often
        beneficial to the speed of some operations. Compression speeds up the process as there is less to
        read/write with the trade off of slightly more computational work on the CPU.


- ``chunks``: tuple of integer or None. Define the chunking used for saving
  the dataset. If None, calculates chunks for the signal, with preferably at
  least one chunk per signal space.
- ``close_file``: only relevant for some zarr store (``ZipStore``, ``DBMStore``)
  requiring store to flush data to disk. If ``False``, doesn't close the file
  after writing. The file should not be closed if the data need to be accessed
  lazily after saving.
  Default is ``True``.
- ``write_dataset``: if ``False``, doesn't write the dataset when writing the file.
  This can be useful to overwrite signal attributes only (for example ``axes_manager``)
  without having to write the whole dataset, which can take time. Default is ``True``.
