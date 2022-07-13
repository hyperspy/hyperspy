.. _hspy-format:

HSpy - HyperSpy's HDF5 Specification
------------------------------------

This is the default format and it is the only one that guarantees that no
information will be lost in the writing process and that supports saving data
of arbitrary dimensions. It is based on the `HDF5 open standard
<http://www.hdfgroup.org/HDF5/>`_. The HDF5 file format is supported by `many
applications
<http://www.hdfgroup.org/products/hdf5_tools/SWSummarybyName.htm>`_.
Part of the specification is documented in :ref:`metadata_structure`.

.. versionadded:: 1.2
    Enable saving HSpy files with the ``.hspy`` extension. Previously only the
    ``.hdf5`` extension was recognised.

.. versionchanged:: 1.3
    The default extension for the HyperSpy HDF5 specification is now ``.hspy``.
    The option to change the default is no longer present in ``preferences``.

Only loading of HDF5 files following the HyperSpy specification are supported.
Usually their extension is ``.hspy`` extension, but older versions of HyperSpy
would save them with the ``.hdf5`` extension. Both extensions are recognised
by HyperSpy since version 1.2. However, HyperSpy versions older than 1.2
won't recognise the ``.hspy`` extension. To
workaround the issue when using old HyperSpy installations simply change the
extension manually to ``.hdf5`` or
save directly the file using this extension by explicitly adding it to the
filename e.g.:

.. code-block:: python

    >>> s = hs.signals.BaseSignal([0])
    >>> s.save('test.hdf5')


When saving to ``hspy``, all supported objects in the signal's
:py:attr:`~.signal.BaseSignal.metadata` is stored. This includes lists, tuples and signals.
Please note that in order to increase saving efficiency and speed, if possible,
the inner-most structures are converted to numpy arrays when saved. This
procedure homogenizes any types of the objects inside, most notably casting
numbers as strings if any other strings are present:

.. code-block:: python

    >>> # before saving:
    >>> somelist
    [1, 2.0, 'a name']
    >>> # after saving:
    ['1', '2.0', 'a name']

The change of type is done using numpy "safe" rules, so no information is lost,
as numbers are represented to full machine precision.

This feature is particularly useful when using
:py:meth:`~hyperspy._signals.eds.EDSSpectrum.get_lines_intensity`:

.. code-block:: python

    >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
    >>> s.metadata.Sample.intensities = s.get_lines_intensity()
    >>> s.save('EDS_spectrum.hspy')

    >>> s_new = hs.load('EDS_spectrum.hspy')
    >>> s_new.metadata.Sample.intensities
    [<BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: Al_Ka at 1.49 keV, dimensions: (|)>,
     <BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: C_Ka at 0.28 keV, dimensions: (|)>,
     <BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: Cu_La at 0.93 keV, dimensions: (|)>,
     <BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: Mn_La at 0.63 keV, dimensions: (|)>,
     <BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: Zr_La at 2.04 keV, dimensions: (|)>]

.. versionadded:: 1.3.1
    ``chunks`` keyword argument

The hyperspy HDF5 format supports chunking the data into smaller pieces to make it possible to load only part
of a dataset at a time. By default, the data is saved in chunks that are optimised to contain at least one
full signal.  It is possible to
customise the chunk shape using the ``chunks`` keyword.
For example, to save the data with ``(20, 20, 256)`` chunks instead of the default ``(7, 7, 2048)`` chunks
for this signal:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.random.random((100, 100, 2048)))
    >>> s.save("test_chunks", chunks=(20, 20, 256))

Note that currently it is not possible to pass different customised chunk shapes to all signals and
arrays contained in a signal and its metadata. Therefore, the value of ``chunks`` provided on saving
will be applied to all arrays contained in the signal.

By passing ``True`` to ``chunks`` the chunk shape is guessed using ``h5py``'s ``guess_chunk`` function
what, for large signal spaces usually leads to smaller chunks as ``guess_chunk`` does not impose the
constrain of storing at least one signal per chunks. For example, for the signal in the example above
passing ``chunks=True`` results in ``(7, 7, 256)`` chunks.

Choosing the correct chunk-size can significantly affect the speed of reading, writing and performance of many HyperSpy algorithms.
See the :ref:`chunking section <big_data.chunking>` for more information.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``compression``: One of ``None``, ``'gzip'``, ``'szip'``, ``'lzf'`` (default is ``'gzip'``).
  ``'szip'`` may be unavailable as it depends on the HDF5 installation including it.

    .. note::

        HyperSpy uses h5py for reading and writing HDF5 files and, therefore, it
        supports all `compression filters supported by h5py <https://docs.h5py.org/en/stable/high/dataset.html#dataset-compression>`_.
        The default is ``'gzip'``. It is possible to enable other compression filters
        such as ``blosc`` by installing e.g. `hdf5plugin <https://github.com/silx-kit/hdf5plugin>`_.
        However, be aware that loading those files will require installing the package
        providing the compression filter. If not available an error will be raised.

        Compression can significantly increase the saving speed. If file size is not
        an issue, it can be disabled by setting ``compression=None``. Notice that only
        ``compression=None`` and ``compression='gzip'`` are available in all platforms,
        see the `h5py documentation <https://docs.h5py.org/en/stable/faq.html#what-compression-processing-filters-are-supported>`_
        for more details. Therefore, if you choose any other compression filter for
        saving a file, be aware that it may not be possible to load it in some platforms.

- ``chunks``: tuple of integer or None. Define the chunking used for saving
  the dataset. If None, calculates chunks for the signal, with preferably at
  least one chunk per signal space.
- ``close_file``: if ``False``, doesn't close the file after writing. The file
  should not be closed if the data need to be accessed lazily after saving.
  Default is ``True``.
- ``write_dataset``: if ``False``, doesn't write the dataset when writing the file.
  This can be useful to overwrite signal attributes only (for example ``axes_manager``)
  without having to write the whole dataset, which can take time. Default is ``True``.
