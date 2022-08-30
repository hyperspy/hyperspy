.. _io:

***********************
Loading and saving data
***********************

.. contents::
   :depth: 3

.. _loading_files:

Loading files: the load function
================================

HyperSpy can read and write to multiple formats (see :ref:`supported-formats`).
To load data use the :py:func:`~.load` command. For example, to load the
image spam.jpg you can type:

.. code-block:: python

    >>> s = hs.load("spam.jpg")

If loading was successful, the variable ``s`` contains a HyperSpy signal or any
type of signal defined in on of the :ref:`HyperSpy extensions <hyperspy_extensions-label>`
- see available :ref:`signal subclasses <transforming_signal-label>` for more
information. To list the signal types available on your local installation use:

.. code-block:: python

    >>> hs.print_known_signal_types()

HyperSpy will try to guess the most likely data type for the corresponding
file. However, you can force it to read the data as a particular data type by
providing the ``signal_type`` keyword, which has to correspond to one of the
available subclasses of signal, e.g.:

.. code-block:: python

    >>> s = hs.load("filename", signal_type="EELS")

If the loaded file contains several datasets, the :py:func:`~.io.load`
functions will return a list of the corresponding signals:

.. code-block:: python

    >>> s = hs.load("spameggsandham.hspy")
    >>> s
    [<Signal1D, title: spam, dimensions: (32,32|1024)>,
    <Signal1D, title: eggs, dimensions: (32,32|1024)>,
    <Signal1D, title: ham, dimensions: (32,32|1024)>]

.. note::

    Note for python programmers: the data is stored in a numpy array
    in the :py:attr:`~.signal.BaseSignal.data` attribute, but you will not
    normally need to access it there.

HyperSpy will attempt to infer the appropriate file reader to use based on
the file extension (for example. ``.hspy``, ``.emd`` and so on). You can
override this using the ``reader`` keyword:

.. code-block:: python

    # Load a .hspy file with an unknown extension
    >>> s = hs.load("filename.some_extension", reader="hspy")

Some file formats store some extra information about the data (metadata) and
HyperSpy reads most of them and stores them in the
:py:attr:`~.signal.BaseSignal.original_metadata` attribute. Also, depending on
the file format, a part of this information will be mapped by HyperSpy to the
:py:attr:`~.signal.BaseSignal.metadata` attribute, where it can be used by
e.g. routines operating on the signal. See :ref:`metadata structure
<metadata_structure>` for details.

.. note::

    Extensive metadata can slow down loading and processing, and
    loading the :py:attr:`~.signal.BaseSignal.original_metadata` can be disabled
    using the ``load_original_metadata`` argument of the :py:func:`~.load`
    function; in this case, the :py:attr:`~.signal.BaseSignal.metadata` will
    still be populated.

To print the content of the attributes simply use:

.. code-block:: python

    >>> s.original_metadata
    >>> s.metadata

The :py:attr:`~.signal.BaseSignal.original_metadata` and
:py:attr:`~.signal.BaseSignal.metadata` can be exported to text files
using the :py:meth:`~.misc.utils.DictionaryTreeBrowser.export` method, e.g.:

.. code-block:: python

    >>> s.original_metadata.export('parameters')

.. _load_to_memory-label:

.. deprecated:: 1.2
   ``memmap_dir`` and ``load_to_memory`` :py:func:`~.io.load` keyword
   arguments. Use ``lazy`` instead of ``load_to_memory``. ``lazy`` makes
   ``memmap_dir`` unnecessary.

.. versionadd: 1.2
   ``lazy`` keyword argument.

Almost all file readers support `lazy` loading, which means accessing the data
without loading it to memory (see :ref:`supported-formats` for a list). This
feature can be useful when analysing large files. To use this feature set
``lazy`` to ``True`` e.g.:

.. code-block:: python

    >>> s = hs.load("filename.hspy", lazy=True)

More details on lazy evaluation support in :ref:`big-data-label`.

The units of the navigation and signal axes can be converted automatically
during loading using the ``convert_units`` parameter. If `True`, the
``convert_to_units`` method of the ``axes_manager`` will be used for the conversion
and if set to `False`, the units will not be converted (default).

.. _load-multiple-label:

Loading multiple files
----------------------

Rather than loading files individually, several files can be loaded with a
single command. This can be done by passing a list of filenames to the load
functions, e.g.:

.. code-block:: python

    >>> s = hs.load(["file1.hspy", "file2.hspy"])

or by using `shell-style wildcards <https://docs.python.org/library/glob.html>`_:

.. code-block:: python

    >>> s = hs.load("file*.hspy")

Alternatively, regular expression type character classes can be used such as
``[a-z]`` for lowercase letters or ``[0-9]`` for one digit integers:

.. code-block:: python

    >>> s = hs.load('file[0-9].hspy')

.. note::

    Wildcards are implemented using ``glob.glob()``, which treats ``*``, ``[``
    and ``]`` as special characters for pattern matching. If your filename or
    path contains square brackets, you may want to set
    ``escape_square_brackets=True``:

    .. code-block:: python

        >>> # Say there are two files like this:
        >>> # /home/data/afile[1x1].hspy
        >>> # /home/data/afile[1x2].hspy

        >>> s = hs.load("/home/data/afile[*].hspy", escape_square_brackets=True)

HyperSpy also supports ```pathlib.Path`` <https://docs.python.org/3/library/pathlib.html>`_
objects, for example:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> from pathlib import Path

    >>> # Use pathlib.Path
    >>> p = Path("/path/to/a/file.hspy")
    >>> s = hs.load(p)

    >>> # Use pathlib.Path.glob
    >>> p = Path("/path/to/some/files/").glob("*.hspy")
    >>> s = hs.load(p)

By default HyperSpy will return a list of all the files loaded. Alternatively,
by setting ``stack=True``, HyperSpy can be instructed to stack the data - given
that the files contain data with exactly the same
dimensions. If this is not the case, an error is raised. If each file contains
multiple (N) signals, N stacks will be created. Here, the number of signals
per file must also match, or an error will be raised.

.. code-block:: python

    >>> ls
    CL1.raw  CL1.rpl  CL2.raw  CL2.rpl  CL3.raw  CL3.rpl  CL4.raw  CL4.rpl
    LL3.raw  LL3.rpl  shift_map-SI3.npy  hdf5/
    >>> s = hs.load('*.rpl')
    >>> s
    [<EELSSpectrum, title: CL1, dimensions: (64, 64, 1024)>,
    <EELSSpectrum, title: CL2, dimensions: (64, 64, 1024)>,
    <EELSSpectrum, title: CL3, dimensions: (64, 64, 1024)>,
    <EELSSpectrum, title: CL4, dimensions: (64, 64, 1024)>,
    <EELSSpectrum, title: LL3, dimensions: (64, 64, 1024)>]
    >>> s = hs.load('*.rpl', stack=True)
    >>> s
    <EELSSpectrum, title: mva, dimensions: (5, 64, 64, 1024)>


.. _saving_files:

Saving data to files
====================

To save data to a file use the :py:meth:`~.signal.BaseSignal.save` method. The
first argument is the filename and the format is defined by the filename
extension. If the filename does not contain the extension, the default format
(:ref:`hspy-format`) is used. For example, if the :py:const:`s` variable
contains the :py:class:`~.signal.BaseSignal` that you want to write to a file,
the following will write the data to a file called :file:`spectrum.hspy` in the
default :ref:`hspy-format` format:

.. code-block:: python

    >>> s.save('spectrum')

If you want to save to the :ref:`ripple format <ripple-format>` instead, write:

.. code-block:: python

    >>> s.save('spectrum.rpl')

Some formats take extra arguments. See the relevant subsections of
:ref:`supported-formats` for more information.


.. _supported-formats:

Supported formats
=================

Here is a summary of the different formats that are currently supported by
HyperSpy. The "lazy" column specifies if lazy evaluation is supported.


.. table:: Supported file formats

    +-----------------------------------+--------+--------+--------+
    | Format                            | Read   | Write  | lazy   |
    +===================================+========+========+========+
    | Gatan's dm3                       |    Yes |    No  |    Yes |
    +-----------------------------------+--------+--------+--------+
    | Gatan's dm4                       |    Yes |    No  |    Yes |
    +-----------------------------------+--------+--------+--------+
    | FEI's emi and ser                 |    Yes |    No  |    Yes |
    +-----------------------------------+--------+--------+--------+
    | hspy                              |    Yes |    Yes |    Yes |
    +-----------------------------------+--------+--------+--------+
    | zspy                              |    Yes |    Yes |    Yes |
    +-----------------------------------+--------+--------+--------+
    | Image: e.g. jpg, png, tif, ...    |    Yes |    Yes |    Yes |
    +-----------------------------------+--------+--------+--------+
    | TIFF                              |    Yes |    Yes |    Yes |
    +-----------------------------------+--------+--------+--------+
    | MRC                               |    Yes |    No  |    Yes |
    +-----------------------------------+--------+--------+--------+
    | MRCZ                              |    Yes |    Yes |    Yes |
    +-----------------------------------+--------+--------+--------+
    | EMSA/MSA                          |    Yes |    Yes |    No  |
    +-----------------------------------+--------+--------+--------+
    | NetCDF                            |    Yes |    No  |    No  |
    +-----------------------------------+--------+--------+--------+
    | Ripple                            |    Yes |    Yes |    Yes |
    +-----------------------------------+--------+--------+--------+
    | SEMPER unf                        |    Yes |    Yes |    Yes |
    +-----------------------------------+--------+--------+--------+
    | Blockfile                         |    Yes |    Yes |    Yes |
    +-----------------------------------+--------+--------+--------+
    | DENSsolutions' Impulse log        |    Yes |    No  |    No  |
    +-----------------------------------+--------+--------+--------+
    | DENSsolutions' Digiheater log     |    Yes |    No  |    No  |
    +-----------------------------------+--------+--------+--------+
    | Bruker's bcf                      |    Yes |    No  |    Yes |
    +-----------------------------------+--------+--------+--------+
    | Bruker's spx                      |    Yes |    No  |    No  |
    +-----------------------------------+--------+--------+--------+
    | EMD (NCEM)                        |    Yes |    Yes |    Yes |
    +-----------------------------------+--------+--------+--------+
    | EMD (Velox)                       |    Yes |    No  |    Yes |
    +-----------------------------------+--------+--------+--------+
    | Protochips log                    |    Yes |    No  |    No  |
    +-----------------------------------+--------+--------+--------+
    | EDAX spc and spd                  |    Yes |    No  |    Yes |
    +-----------------------------------+--------+--------+--------+
    | h5USID h5                         |    Yes |   Yes  |   Yes  |
    +-----------------------------------+--------+--------+--------+
    | Phenom elid                       |    Yes |    No  |    No  |
    +-----------------------------------+--------+--------+--------+
    | DigitalSurf's sur and pro         |    Yes |    No  |    No  |
    +-----------------------------------+--------+--------+--------+
    | Nexus nxs                         |    Yes |   Yes  |   Yes  |
    +-----------------------------------+--------+--------+--------+
    | EMPAD xml                         |    Yes |    No  |   Yes  |
    +-----------------------------------+--------+--------+--------+
    | JEOL asw, map, img, pts, eds      |    Yes |    No  |    No  |
    +-----------------------------------+--------+--------+--------+
    | TVIPS .tvips                      |    Yes |    Yes |   Yes  |
    +-----------------------------------+--------+--------+--------+

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


.. _zspy-format:

ZSpy - HyperSpy's Zarr Specification
------------------------------------

Similarly to the :ref:`hspy format <hspy-format>`, the zspy format guarantees that no
information will be lost in the writing process and that supports saving data
of arbitrary dimensions. It is based on the `Zarr project <https://zarr.readthedocs.io/en/stable/index.html>`_. Which exists as a drop in
replacement for hdf5 with the intention to fix some of the speed and scaling
issues with the hdf5 format and is therefore suitable for saving :ref:`big data <big_data.saving>`.


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


.. _netcdf-format:

NetCDF
------

This was the default format in HyperSpy's predecessor, EELSLab, but it has been
superseded by :ref:`hspy-format` in HyperSpy. We provide only reading capabilities
but we do not support writing to this format.

Note that only NetCDF files written by EELSLab are supported.

To use this format a python netcdf interface must be installed manually because
it is not installed by default when using the automatic installers.


.. _mrc-format:

MRC
---

This is a format widely used for tomographic data. Our implementation is based
on `this specification
<https://www2.mrc-lmb.cam.ac.uk/research/locally-developed-software/image-processing-software/>`_. We also
partly support FEI's custom header. We do not provide writing features for this
format, but, as it is an open format, we may implement this feature in the
future on demand.

For mrc files ``load`` takes the ``mmap_mode`` keyword argument enabling
loading the file using a different mode (default is copy-on-write) . However,
note that lazy loading does not support in-place writing (i.e lazy loading and
the "r+" mode are incompatible).

.. _mrcz-format:

MRCZ
----

MRCZ is an extension of the CCP-EM MRC2014 file format. `CCP-EM MRC2014
<http://www.ccpem.ac.uk/mrc_format/mrc2014.php>`_ file format.  It uses the
`blosc` meta-compression library to bitshuffle and compress files in a blocked,
multi-threaded environment. The supported data types are:

[`float32`,`int8`,`uint16`,`int16`,`complex64`]

It supports arbitrary meta-data, which is serialized into JSON.

MRCZ also supports asynchronous reads and writes.

Repository: https://github.com/em-MRCZ
PyPI:       https://pypi.python.org/pypi/mrcz
Citation:   Submitted.
Preprint:   http://www.biorxiv.org/content/early/2017/03/13/116533

Support for this format is not enabled by default. In order to enable it
install the `mrcz` and optionally the `blosc` Python packages.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``do_async``: currently supported within HyperSpy for writing only, this will
  save  the file in a background thread and return immediately. Defaults
  to `False`.

.. Warning::

    There is no method currently implemented within Hyperspy to tell if an
    asychronous write has finished.


- ``compressor``: The compression codec, one of [`None`,`'zlib`',`'zstd'`, `'lz4'`].
  Defaults to `None`.
- ``clevel``: The compression level, an `int` from 1 to 9. Defaults to 1.
- ``n_threads``: The number of threads to use for 'blosc' compression. Defaults to
  the maximum number of virtual cores (including Intel Hyperthreading)
  on your system, which is recommended for best performance. If \
  ``do_async = True`` you may wish to leave one thread free for the
  Python GIL.

The recommended compression codec is 'zstd' (zStandard) with `clevel=1` for
general use. If speed is critical, use 'lz4' (LZ4) with `clevel=9`. Integer data
compresses more redably than floating-point data, and in general the histogram
of values in the data reflects how compressible it is.

To save files that are compatible with other programs that can use MRC such as
GMS, IMOD, Relion, MotionCorr, etc. save with `compressor=None`, extension `.mrc`.
JSON metadata will not be recognized by other MRC-supporting software but should
not cause crashes.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

    >>> s.save('file.mrcz', do_async=True, compressor='zstd', clevel=1)

    >>> new_signal = hs.load('file.mrcz')


.. _msa-format:

EMSA/MSA
--------

This `open standard format
<https://www.microscopy.org/resources/scientific_data/index.cfm>`__
is widely used to exchange single spectrum data, but it does not support
multidimensional data. It can be used to exchange single spectra with Gatan's
Digital Micrograph.

.. WARNING::
    If several spectra are loaded and stacked (``hs.load('pattern', stack_signals=True``)
    the calibration read from the first spectrum and applied to all other spectra.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

For the MSA format the ``format`` argument is used to specify whether the
energy axis should also be saved with the data.  The default, 'Y' omits the
energy axis in the file.  The alternative, 'XY', saves a second column with the
calibrated energy data. It  is possible to personalise the separator with the
`separator` keyword.

.. Warning::

    However, if a different separator is chosen the resulting file will not
    comply with the MSA/EMSA standard and HyperSpy and other software may not
    be able to read it.

The default encoding is `latin-1`. It is possible to set a different encoding
using the `encoding` argument, e.g.:

.. code-block:: python

    >>> s.save('file.msa', encoding = 'utf8')


.. _ripple-format:

Ripple
------

This *open standard format* developed at NIST as native format for
`Lispix <http://www.nist.gov/lispix/>`_ is widely used to exchange
multidimensional data. However, it only supports data of up to three
dimensions. It can also be used to exchange data with Bruker and used in
combination with the :ref:`import-rpl` it is very useful for exporting data
to Gatan's Digital Micrograph.

The default encoding is latin-1. It is possible to set a different encoding
using the encoding argument, e.g.:

.. code-block:: python

    >>> s.save('file.rpl', encoding = 'utf8')


For mrc files ``load`` takes the ``mmap_mode`` keyword argument enabling
loading the file using a different mode (default is copy-on-write) . However,
note that lazy loading does not support in-place writing (i.e lazy loading and
the "r+" mode are incompatible).

.. _image-format:

Images
------

HyperSpy can read and write data to `all the image formats
<https://imageio.readthedocs.io/en/stable/formats.html>`_ supported by
`imageio`, which uses the Python Image Library  (PIL/pillow).
This includes png, pdf, gif, etc.
It is important to note that these image formats only support 8-bit files, and
therefore have an insufficient dynamic range for most scientific applications.
It is therefore highly discouraged to use any general image format (with the
exception of :ref:`tiff-format` which uses another library) to store data for
analysis purposes.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``scalebar`` (bool, optional): Export the image with a scalebar. Default
  is False.
- ``scalebar_kwds`` (dict, optional): dictionary of keyword arguments for the
  scalebar. Useful to set formattiong, location, etc. of the scalebar. See the
  `matplotlib-scalebar <https://pypi.org/project/matplotlib-scalebar/>`_
  documentation for more information.
- ``output_size`` : (int, tuple of length 2 or None, optional): the output size
  of the image in pixels:

  * if ``int``, defines the width of the image, the height is
    determined from the aspect ratio of the image.
  * if ``tuple`` of length 2, defines the width and height of the
    image. Padding with white pixels is used to maintain the aspect
    ratio of the image.
  * if ``None``, the size of the data is used.

  For output sizes larger than the data size, "nearest" interpolation is
  used by default and this behaviour can be changed through the
  ``imshow_kwds`` dictionary.

- ``imshow_kwds`` (dict, optional):  Keyword arguments dictionary for
  :py:func:`~.matplotlib.pyplot.imshow`.
- ``**kwds`` : keyword arguments supported by the individual file
  writers as documented at
  https://imageio.readthedocs.io/en/stable/formats.html when exporting
  an image without scalebar. When exporting with a scalebar, the keyword
  arguments are passed to the `pil_kwargs` dictionary of
  :py:func:`matplotlib.pyplot.savefig`


When saving an image, a scalebar can be added to the image and the formatting,
location, etc. of the scalebar can be set using the ``scalebar_kwds``
arguments:

.. code-block:: python

    >>> s.save('file.jpg', scalebar=True)
    >>> s.save('file.jpg', scalebar=True, scalebar_kwds={'location':'lower right'})

In the example above, the image is created using
:py:func:`~.matplotlib.pyplot.imshow`, and additional keyword arguments can be
passed to this function using ``imshow_kwds``. For example, this can be used
to save an image displayed using a matplotlib colormap:

.. code-block:: python

    >>> s.save('file.jpg', imshow_kwds=dict(cmap='viridis'))


The resolution of the exported image can be adjusted:

.. code-block:: python

    >>> s.save('file.jpg', output_size=512)


.. _tiff-format:

TIFF
----

HyperSpy can read and write 2D and 3D TIFF files using using
Christoph Gohlke's ``tifffile`` library. In particular, it supports reading and
writing of TIFF, BigTIFF, OME-TIFF, STK, LSM, NIH, and FluoView files. Most of
these are uncompressed or losslessly compressed 2**(0 to 6) bit integer, 16, 32
and 64-bit float, grayscale and RGB(A) images, which are commonly used in
bio-scientific imaging. See `the library webpage
<http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html>`_ for more details.

.. versionadded: 1.0
   Add support for writing/reading scale and unit to tif files to be read with
   ImageJ or DigitalMicrograph

Currently HyperSpy has limited support for reading and saving the TIFF tags.
However, the way that HyperSpy reads and saves the scale and the units of TIFF
files is compatible with ImageJ/Fiji and Gatan Digital Micrograph software.
HyperSpy can also import the scale and the units from TIFF files saved using
FEI, Zeiss SEM, Olympus SIS, Jeol SightX and Hamamatsu HPD-TA (streak camera)
software.

Extra loading arguments
^^^^^^^^^^^^^^^^^^^^^^^

- ``force_read_resolution`` (bool, optional): Force read image resolution using
  the x_resolution, y_resolution and resolution_unit tags of the TIFF. Beware:
  most software don't (properly) use these tags when saving TIFF files. Default
  is ``False``.
- ``hamamatsu_streak_axis_type`` (str, optional): decide the type of the
  time axis for hamamatsu streak files:

  * if ``uniform``, the best-fit linear axis is used, inducing a (small)
    linearisation error. Initialise a UniformDataAxis.
  * if ``data``, the raw time axis parsed from the metadata is used. Initialise
    a DataAxis.
  * if ``functional``, the best-fit 3rd-order polynomial axis is used, avoiding
    linearisation error. Initialise a FunctionalDataAxis.

By default, ``uniform`` is used but a warning of the linearisation error is issued.
Explicitly passing ``hamamatsu_streak_axis_type='uniform'`` suppresses the warning.
In all cases, the original axis values are stored in the ``original_metadata`` of the
signal object.

.. code-block:: python

    >>> # Force read image resolution using the x_resolution, y_resolution and
    >>> # the resolution_unit of the TIFF tags.
    >>> s = hs.load('file.tif', force_read_resolution=True)
    >>> # Load a non-uniform axis from a hamamatsu streak file:
    >>> s = hs.load('file.tif', hamamatsu_streak_axis_type='data')


Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``extratags`` (tuple, optional): save custom tags through the
   ``tifffile`` library. Must conform to a specific convention
   (see ``tifffile`` documentation and example below).

.. code-block:: python

    >>> # Saving the string 'Random metadata' in a custom tag (ID 65000)
    >>> extratag = [(65000, 's', 1, "Random metadata", False)]
    >>> s.save('file.tif', extratags=extratag)

    >>> # Saving the string 'Random metadata' from a custom tag (ID 65000)
    >>> s2 = hs.load('file.tif')
    >>> s2.original_metadata['Number_65000']
    b'Random metadata'

.. warning::

    The file will be saved with the same bit depth as the signal. Since
    most processing operations in HyperSpy and numpy will result in 64-bit
    floats, this can result in 64-bit ``.tiff`` files, which are not always
    compatible with other imaging software.

    You can first change the dtype of the signal before saving:

    .. code-block:: python

        >>> s.data.dtype
        dtype('float64')
        >>> s.change_dtype('float32')
        >>> s.data.dtype
        dtype('float32')
        >>> s.save('file.tif')

.. _dm3-format:

Gatan Digital Micrograph
------------------------

HyperSpy can read both dm3 and dm4 files but the reading features are not
complete (and probably they will remain so unless Gatan releases the
specifications of the format). That said, we understand that this is an
important feature and if loading a particular Digital Micrograph file fails for
you, please report it as an issue in the `issues tracker
<https://github.com/hyperspy/hyperspy/issues>`__ to make
us aware of the problem.

Some of the tags in the DM-files are added to the metadata of the signal
object. This includes, microscope information and certain parameters for EELS,
EDS and CL signals.

Extra loading arguments
^^^^^^^^^^^^^^^^^^^^^^^

- ``optimize``: bool, default is True. During loading, the data is replaced by its
  :ref:`optimized copy <signal.transpose_optimize>` to speed up operations,
  e. g. iteration over navigation axes. The cost of this speed improvement is to
  double the memory requirement during data loading.

.. warning::

    It has been reported that in some versions of Gatan Digital Micrograph,
    any binned data stores the _averages_ of the binned channels or pixels,
    rather than the _sum_, which would be required for proper statistical
    analysis. We therefore strongly recommend that all binning is performed
    using Hyperspy where possible.

    See the original `bug report here <https://github.com/hyperspy/hyperspy/issues/1624>`_.


.. _edax-format:

EDAX TEAM/Genesis SPD and SPC
-----------------------------

HyperSpy can read both ``.spd`` (spectrum image) and ``.spc`` (single spectra)
files from the EDAX TEAM software and its predecessor EDAX Genesis.
If reading an ``.spd`` file, the calibration of the
spectrum image is loaded from the corresponding ``.ipr`` and ``.spc`` files
stored in the same directory, or from specific files indicated by the user.
If these calibration files are not available, the data from the ``.spd``
file will still be loaded, but with no spatial or energy calibration.
If elemental information has been defined in the spectrum image, those
elements will automatically be added to the signal loaded by HyperSpy.

Currently, loading an EDAX TEAM spectrum or spectrum image will load an
``EDSSEMSpectrum`` Signal. If support for TEM EDS data is needed, please
open an issue in the `issues tracker <https://github.com/hyperspy/hyperspy/issues>`__ to
alert the developers of the need.

For further reference, file specifications for the formats are
available publicly available from EDAX and are on Github
(`.spc <https://github.com/hyperspy/hyperspy/files/29506/SPECTRUM-V70.pdf>`_,
`.spd <https://github.com/hyperspy/hyperspy/files/29505/
SpcMap-spd.file.format.pdf>`_, and
`.ipr <https://github.com/hyperspy/hyperspy/files/29507/ImageIPR.pdf>`_).

Extra loading arguments for SPD file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``spc_fname``: {None, str}, name of file from which to read the spectral calibration. If data was exported fully from EDAX TEAM software, an .spc file with the same name as the .spd should be present. If `None`, the default filename will be searched for. Otherwise, the name of the ``.spc`` file to use for calibration can be explicitly given as a string.
- ``ipr_fname``: {None, str}, name of file from which to read the spatial calibration. If data was exported fully from EDAX TEAM software, an ``.ipr`` file with the same name as the ``.spd`` (plus a "_Img" suffix) should be present.  If `None`, the default filename will be searched for. Otherwise, the name of the ``.ipr`` file to use for spatial calibration can be explicitly given as a string.
- ``**kwargs``: remaining arguments are passed to the Numpy ``memmap`` function.

Extra loading arguments for SPD and SPC files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``load_all_spc`` : bool, switch to control if all of the ``.spc`` header is
  read, or just the important parts for import into HyperSpy.


.. _fei-format:

FEI TIA SER and EMI
-------------------

HyperSpy can read ``ser`` and ``emi`` files but the reading features are not
complete (and probably they will be unless FEI releases the specifications of
the format). That said we know that this is an important feature and if loading
a particular ser or emi file fails for you, please report it as an issue in the
`issues tracker <https://github.com/hyperspy/hyperspy/issues>`__ to make us
aware of the problem.

HyperSpy (unlike TIA) can read data directly from the ``.ser`` files. However,
by doing so, the information that is stored in the emi file is lost.
Therefore strongly recommend to load using the ``.emi`` file instead.

When reading an ``.emi`` file if there are several ``.ser`` files associated
with it, all of them will be read and returned as a list.


Extra loading arguments
^^^^^^^^^^^^^^^^^^^^^^^

- ``only_valid_data`` : bool, in case of series or linescan data with the
  acquisition stopped before the end: if True, load only the acquired data.
  If False, the empty data are filled with zeros. The default is False and this
  default value will change to True in version 2.0.

.. _unf-format:

SEMPER UNF binary format
------------------------

SEMPER is a fully portable system of programs for image processing, particularly
suitable for applications in electron microscopy developed by Owen Saxton (see
DOI: 10.1016/S0304-3991(79)80044-3 for more information). The unf format is a
binary format with an extensive header for up to 3 dimensional data.
HyperSpy can read and write unf-files and will try to convert the data into a
fitting BaseSignal subclass, based on the information stored in the label.
Currently version 7 of the format should be fully supported.

.. _blockfile-format:

Blockfile
---------

HyperSpy can read and write the blockfile format from NanoMegas ASTAR software.
It is used to store a series of diffraction patterns from scanning precession
electron diffraction (SPED) measurements, with a limited set of metadata. The
header of the blockfile contains information about centering and distortions
of the diffraction patterns, but is not applied to the signal during reading.
Blockfiles only support data values of type
`np.uint8 <http://docs.scipy.org/doc/numpy/user/basics.types.html>`_ (integers
in range 0-255).

.. warning::

   While Blockfiles are supported, it is a proprietary format, and future
   versions of the format might therefore not be readable. Complete
   interoperability with the official software can neither be guaranteed.

Blockfiles are by default loaded in a "copy-on-write" manner using
`numpy.memmap
<http://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html>`_ .
For blockfiles ``load`` takes the ``mmap_mode`` keyword argument enabling
loading the file using a different mode. However, note that lazy loading
does not support in-place writing (i.e lazy loading and the "r+" mode
are incompatible).

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``intensity_scaling`` : in case the dataset that needs to be saved does not
  have the `np.uint8` data type, casting to this datatype without intensity
  rescaling results in overflow errors (default behavior). This option allows
  you to perform linear intensity scaling of the images prior to saving the
  data. The options are:

  - `'dtype'`: the limits of the datatype of the dataset, e.g. 0-65535 for
    `np.uint16`, are mapped onto 0-255 respectively. Does not work for `float`
    data types.
  - `'minmax'`: the minimum and maximum in the dataset are mapped to 0-255.
  - `'crop'`: everything below 0 and above 255 is set to 0 and 255 respectively
  - 2-tuple of `floats` or `ints`: the intensities between these values are
    scaled between 0-255, everything below is 0 and everything above is 255.
- ``navigator_signal``: the BLO file also stores a virtual bright field (VBF) image which
  behaves like a navigation signal in the ASTAR software. By default this is
  set to `'navigator'`, which results in the default :py:attr:`navigator` signal to be used.
  If this signal was not calculated before (e.g. by calling :py:meth:`~.signal.BaseSignal.plot`), it is
  calculated when :py:meth:`~.signal.BaseSignal.save` is called, which can be time consuming.
  Alternatively, setting the argument to `None` will result in a correctly sized
  zero array to be used. Finally, a custom ``Signal2D`` object can be passed,
  but the shape must match the navigation dimensions.

.. _dens-format:

DENSsolutions formats
---------------------
HyperSpy can read any logfile from DENSsolutions' new Impulse software as well as the legacy heating software DigiHeater.

DENSsolutions Impulse logfile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Impulse logfiles are stored in csv format. All metadata linked to the experiment is stored in a separate metadata.log file.
This metadata file contains crucial information about the experiment and should be included in the same folder with the csv file when reading data into Hyperspy.

To read Impulse logfiles, use the reader argument to define the correct file reader:

.. code-block:: python

    >>> hs.load("filename.csv", reader="impulse")


DENSsolutions DigiHeater logfile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy can read the heater log format from the DENSsolutions’ DigiHeater software. The format stores all the captured data for each timestamp, together with a small header in a plain-text format. The reader extracts the measured temperature along the time axis, as well as the date and calibration constants stored in the header.


Bruker's formats
----------------
Bruker's Esprit(TM) software and hardware allows to acquire and save the data
in different kind of formats. Hyperspy can read two main basic formats: bcf
and spx.

.. _bcf-format:

Bruker composite file
^^^^^^^^^^^^^^^^^^^^^

HyperSpy can read "hypermaps" saved with Bruker's Esprit v1.x or v2.x in bcf
hybrid (virtual file system/container with xml and binary data, optionally
compressed) format. Most bcf import functionality is implemented. Both
high-resolution 16-bit SEM images and hyperspectral EDX data can be retrieved
simultaneously.

BCF can look as all inclusive format, however it does not save some key EDX
parameters: any of dead/live/real times, FWHM at Mn_Ka line. However, real time
for whole map is calculated from pixelAverage, lineAverage, pixelTime,
lineCounter and map height parameters.

Note that Bruker Esprit uses a similar format for EBSD data, but it is not
currently supported by HyperSpy.

Extra loading arguments
+++++++++++++++++++++++

- ``select_type`` : one of (None, 'spectrum', 'image'). If specified, only the
  corresponding type of data, either spectrum or image, is returned.
  By default (None), all data are loaded.
- ``index`` : one of (None, int, "all"). Allow to select the index of the dataset
  in the bcf file, which can contains several datasets. Default None value
  result in loading the first dataset. When set to 'all', all available datasets
  will be loaded and returned as separate signals.
- ``downsample`` : the downsample ratio of hyperspectral array (height and width
  only), can be integer >=1, where '1' results in no downsampling (default 1).
  The underlying method of downsampling is unchangeable: sum. Differently than
  ``block_reduce`` from skimage.measure it is memory efficient (does not creates
  intermediate arrays, works inplace).
- ``cutoff_at_kV`` : if set (can be None, int, float (kV), one of 'zealous'
  or 'auto') can be used either to crop or enlarge energy (or number of
  channels) range at max values. It can be used to conserve memory or enlarge
  the range if needed to mach the size of other file. Default value is None
  (which does not influence size). Numerical values should be in kV.
  'zealous' truncates to the last non zero channel (this option
  should not be used for stacks, as low beam current EDS can have different
  last non zero channel per slice). 'auto' truncates channels to SEM/TEM
  acceleration voltage or energy at last channel, depending which is smaller.
  In case the hv info is not there or hv is off (0 kV) then it fallbacks to
  full channel range.

Example of loading reduced (downsampled, and with energy range cropped)
"spectrum only" data from bcf (original shape: 80keV EDS range (4096 channels),
100x75 pixels; SEM acceleration voltage: 20kV):

.. code-block:: python

    >>> hs.load("sample80kv.bcf", select_type='spectrum', downsample=2, cutoff_at_kV=10)
    <EDSSEMSpectrum, title: EDX, dimensions: (50, 38|595)>

load the same file with limiting array size to SEM acceleration voltage:

.. code-block:: python

    >>> hs.load("sample80kv.bcf", cutoff_at_kV='auto')
    [<Signal2D, title: BSE, dimensions: (|100, 75)>,
    <Signal2D, title: SE, dimensions: (|100, 75)>,
    <EDSSEMSpectrum, title: EDX, dimensions: (100, 75|1024)>]

The loaded array energy dimension can by forced to be larger than the data
recorded by setting the 'cutoff_at_kV' kwarg to higher value:

.. code-block:: python

    >>> hs.load("sample80kv.bcf", cutoff_at_kV=60)
    [<Signal2D, title: BSE, dimensions: (|100, 75)>,
    <Signal2D, title: SE, dimensions: (|100, 75)>,
    <EDSSEMSpectrum, title: EDX, dimensions: (100, 75|3072)>]

loading without setting cutoff_at_kV value would return data with all 4096
channels. Note that setting downsample to >1 currently locks out using SEM
images for navigation in the plotting.

.. _spx-format:

SPX format
^^^^^^^^^^

Hyperspy can read Bruker's spx format (single spectra format based on XML).
The format contains extensive list of details and parameters of EDS analyses
which are mapped in hyperspy to metadata and original_metadata dictionaries.

.. _emd-format:

EMD
---

EMD stands for “Electron Microscopy Dataset.” It is a subset of the open source
HDF5 wrapper format. N-dimensional data arrays of any standard type can be
stored in an HDF5 file, as well as tags and other metadata.

EMD (NCEM)
^^^^^^^^^^

This `EMD format <https://emdatasets.com>`_ was developed by Colin Ophus at the
National Center for Electron Microscopy (NCEM).
This format is used by the `prismatic software <https://prism-em.com/docs-outputs/>`_
to save the simulation outputs.

Extra loading arguments
+++++++++++++++++++++++

- ``dataset_path`` : None, str or list of str. Path of the dataset. If None,
  load all supported datasets, otherwise the specified dataset(s).
- ``stack_group`` : bool, default is True. Stack datasets of groups with common
  path. Relevant for emd file version >= 0.5 where groups can be named
  'group0000', 'group0001', etc.
- ``chunks`` : None, True or tuple. Determine the chunking of the dataset to save.
  See the ``chunks`` arguments of the ``hspy`` file format for more details.


For files containing several datasets, the `dataset_name` argument can be
used to select a specific one:

.. code-block:: python

    >>> s = hs.load("adatafile.emd", dataset_name="/experimental/science_data_1/data")


Or several by using a list:

.. code-block:: python

    >>> s = hs.load("adatafile.emd",
    ...             dataset_name=[
    ...                 "/experimental/science_data_1/data",
    ...                 "/experimental/science_data_2/data"])


.. _emd_fei-format:

EMD (Velox)
^^^^^^^^^^^

This is a non-compliant variant of the standard EMD format developed by
Thermo-Fisher (former FEI). HyperSpy supports importing images, EDS spectrum and EDS
spectrum streams (spectrum images stored in a sparse format). For spectrum
streams, there are several loading options (described below) to control the frames
and detectors to load and if to sum them on loading.  The default is
to import the sum over all frames and over all detectors in order to decrease
the data size in memory.


.. note::

    Pruned Velox EMD files only contain the spectrum image in a proprietary
    format that HyperSpy cannot read. Therefore, don't prune Velox EMD files
    if you intend to read them with HyperSpy.

.. code-block:: python

    >>> hs.load("sample.emd")
    [<Signal2D, title: HAADF, dimensions: (|179, 161)>,
    <EDSSEMSpectrum, title: EDS, dimensions: (179, 161|4096)>]

.. note::

    FFTs made in Velox are loaded in as-is as a HyperSpy ComplexSignal2D object.
    The FFT is not centered and only positive frequencies are stored in the file.
    Making FFTs with HyperSpy from the respective image datasets is recommended.

.. note::

    DPC data is loaded in as a HyperSpy ComplexSignal2D object.

.. note::

    Currently only lazy uncompression rather than lazy loading is implemented.
    This means that it is not currently possible to read EDS SI Velox EMD files
    with size bigger than the available memory.


.. warning::

   This format is still not stable and files generated with the most recent
   version of Velox may not be supported. If you experience issues loading
   a file, please report it  to the HyperSpy developers so that they can
   add support for newer versions of the format.


.. _Extra-loading-arguments-fei-emd:

Extra loading arguments
+++++++++++++++++++++++

- ``select_type`` : one of {None, 'image', 'single_spectrum', 'spectrum_image'} (default is None).
- ``first_frame`` : integer (default is 0).
- ``last_frame`` : integer (default is None)
- ``sum_frames`` : boolean (default is True)
- ``sum_EDS_detectors`` : boolean (default is True)
- ``rebin_energy`` : integer (default is 1)
- ``SI_dtype`` : numpy dtype (default is None)
- ``load_SI_image_stack`` : boolean (default is False)

The ``select_type`` parameter specifies the type of data to load: if `image` is selected,
only images (including EDS maps) are loaded, if `single_spectrum` is selected, only
single spectra are loaded and if `spectrum_image` is selected, only the spectrum
image will be loaded. The ``first_frame`` and ``last_frame`` parameters can be used
to select the frame range of the EDS spectrum image to load. To load each individual
EDS frame, use ``sum_frames=False`` and the EDS spectrum image will be loaded
with an extra navigation dimension corresponding to the frame index
(time axis). Use the ``sum_EDS_detectors=True`` parameter to load the signal of
each individual EDS detector. In such a case, a corresponding number of distinct
EDS signal is returned. The default is ``sum_EDS_detectors=True``, which loads the
EDS signal as a sum over the signals from each EDS detectors.  The ``rebin_energy``
and ``SI_dtype`` parameters are particularly useful in combination with
``sum_frames=False`` to reduce the data size when one want to read the
individual frames of the spectrum image. If ``SI_dtype=None`` (default), the dtype
of the data in the emd file is used. The ``load_SI_image_stack`` parameter allows
loading the stack of STEM images acquired simultaneously as the EDS spectrum image.
This can be useful to monitor any specimen changes during the acquisition or to
correct the spatial drift in the spectrum image by using the STEM images.

.. code-block:: python

    >>> hs.load("sample.emd", sum_EDS_detectors=False)
    [<Signal2D, title: HAADF, dimensions: (|179, 161)>,
    <EDSSEMSpectrum, title: EDS - SuperXG21, dimensions: (179, 161|4096)>,
    <EDSSEMSpectrum, title: EDS - SuperXG22, dimensions: (179, 161|4096)>,
    <EDSSEMSpectrum, title: EDS - SuperXG23, dimensions: (179, 161|4096)>,
    <EDSSEMSpectrum, title: EDS - SuperXG24, dimensions: (179, 161|4096)>]

    >>> hs.load("sample.emd", sum_frames=False, load_SI_image_stack=True, SI_dtype=np.int8, rebin_energy=4)
    [<Signal2D, title: HAADF, dimensions: (50|179, 161)>,
    <EDSSEMSpectrum, title: EDS, dimensions: (50, 179, 161|1024)>]



.. _protochips-format:

Protochips log
--------------

HyperSpy can read heater, biasing and gas cell log files for Protochips holder.
The format stores all the captured data together with a small header in a csv
file. The reader extracts the measured quantity (e. g. temperature, pressure,
current, voltage) along the time axis, as well as the notes saved during the
experiment. The reader returns a list of signal with each signal corresponding
to a quantity. Since there is a small fluctuation in the step of the time axis,
the reader assumes that the step is constant and takes its mean, which is a
good approximation. Further release of HyperSpy will read the time axis more
precisely by supporting non-uniform axis.

To read Protochips logfiles, use the reader argument to define the correct file reader:

.. code-block:: python

    >>> hs.load("filename.csv", reader="protochips")




.. _usid-format:

USID
----

Background
^^^^^^^^^^

`Universal Spectroscopy and Imaging Data <https://pycroscopy.github.io/USID/about.html>`_
(USID) is an open, community-driven, self-describing, and standardized schema for
representing imaging and spectroscopy data of any size, dimensionality, precision,
instrument of origin, or modality. USID data is typically stored in
Hierarchical Data Format Files (HDF5) and the combination of USID within HDF5 files is
referred to as h5USID.

`pyUSID <https://pycroscopy.github.io/pyUSID/about.html>`_
provides a convenient interface to I/O operations on such h5USID files. USID
(via pyUSID) forms the foundation for other materials microscopy scientific
python package called `pycroscopy <https://pycroscopy.github.io/pycroscopy/about.html>`_.
If you have any questions regarding this module, please consider
`contacting <https://pycroscopy.github.io/pyUSID/contact.html>`_
the developers of pyUSID.

Requirements
^^^^^^^^^^^^

1. Reading and writing h5USID files require the
   `installation of pyUSID <https://pycroscopy.github.io/pyUSID/install.html>`_.
2. Files must use the ``.h5`` file extension in order to use this io plugin.
   Using the ``.hdf5`` extension will default to HyperSpy's own plugin.

Reading
^^^^^^^

h5USID files can contain multiple USID datasets within the same file.
HyperSpy supports reading in one or more USID datasets.

Extra loading arguments
+++++++++++++++++++++++

- ``dataset_path``: str. Absolute path of USID Main HDF5 dataset.
  (default is ``None`` - all USID Main Datasets will be read)
- ``ignore_non_linear_dims``: bool, default is True. If True, parameters that
  were varied non-linearly in the desired dataset will result in Exceptions.
  Else, all such non-linearly varied parameters will be treated as
  linearly varied parameters and a Signal object will be generated.


Reading the sole dataset within a h5USID file:

.. code-block:: python

    >>> hs.load("sample.h5")
    <Signal2D, title: HAADF, dimensions: (|128, 128)>

If multiple datasets are present within the h5USID file and you try the same command again,
**all** available datasets will be loaded.

.. note::

    Given that HDF5 files can accommodate very large datasets, setting ``lazy=True``
    is strongly recommended if the contents of the HDF5 file are not known apriori.
    This prevents issues with regard to loading datasets far larger than memory.

    Also note that setting ``lazy=True`` leaves the file handle to the HDF5 file open.
    If it is important that the files be closed after reading, set ``lazy=False``.

.. code-block:: python

    >>> hs.load("sample.h5")
    [<Signal2D, title: HAADF, dimensions: (|128, 128)>,
    <Signal1D, title: EELS, dimensions: (|64, 64, 1024)>]

We can load a specific dataset using the ``dataset_path`` keyword argument. Setting it to the
absolute path of the desired dataset will cause the single dataset to be loaded.

.. code-block:: python

    >>> # Loading a specific dataset
    >>> hs.load("sample.h5", dataset_path='/Measurement_004/Channel_003/Main_Data')
    <Signal2D, title: HAADF, dimensions: (|128, 128)>

h5USID files support the storage of HDF5 dataset with
`compound data types <https://pycroscopy.github.io/USID/usid_model.html#compound-datasets>`_.
As an (*oversimplified*) example, one could store a color image using a compound data type that allows
each color channel to be accessed by name rather than an index.
Naturally, reading in such a compound dataset into HyperSpy will result in a separate
signal for each named component in the dataset:

.. code-block:: python

    >>> hs.load("file_with_a_compound_dataset.h5")
    [<Signal2D, title: red, dimensions: (|128, 128)>,
    Signal2D, title: blue, dimensions: (|128, 128)>,
    Signal2D, title: green, dimensions: (|128, 128)>]

h5USID files also support parameters or dimensions that have been varied non-uniformly.
This capability is important in several spectroscopy techniques where the bias is varied as a
`bi-polar triangular waveform <https://pycroscopy.github.io/pyUSID/auto_examples/beginner/plot_usi_dataset.html#values-for-each-dimension>`_
rather than uniformly from the minimum value to the maximum value.
Since HyperSpy Signals expect uniform variation of parameters / axes, such non-uniform information
would be lost in the axes manager. The USID plugin will default to a warning
when it encounters a parameter that has been varied non-uniformly:

.. code-block:: python

    >>> hs.load("sample.h5")
    UserWarning: Ignoring non-uniformity of dimension: Bias
    <BaseSignal, title: , dimensions: (|7, 3, 5, 2)>

Obviously, the
In order to prevent accidental misinterpretation of information downstream, the keyword argument
``ignore_non_uniform_dims`` can be set to ``False`` which will result in a ``ValueError`` instead.

.. code-block:: python

    >>> hs.load("sample.h5")
    ValueError: Cannot load provided dataset. Parameter: Bias was varied non-uniformly.
    Supply keyword argument "ignore_non_uniform_dims=True" to ignore this error

Writing
^^^^^^^

Signals can be written to new h5USID files using the standard :py:meth:`~.signal.BaseSignal.save` function.
Setting the ``overwrite`` keyword argument to ``True`` will append to the specified
HDF5 file. All other keyword arguments will be passed to
`pyUSID.hdf_utils.write_main_dataset() <https://pycroscopy.github.io/pyUSID/_autosummary/_autosummary/pyUSID.io.hdf_utils.html#pyUSID.io.hdf_utils.write_main_dataset>`_

.. code-block:: python

    >>> sig.save("USID.h5")

Note that the model and other secondary data artifacts linked to the signal are not
written to the file but these can be implemented at a later stage.

.. _nexus-format:

Nexus
-----

Background
^^^^^^^^^^

`NeXus <https://www.nexusformat.org>`_ is a common data format originally
developed by the neutron and x-ray science x-ray communities. It is still being
developed as an international standard by scientists and programmers
representing major scientific facilities in order to facilitate greater
cooperation in the analysis and visualization of data.
Nexus uses a variety of classes to record data, values,
units and other experimental metadata associated with an experiment.
For specific types of experiments an Application Definition may exist, which
defines an agreed common layout that facilities can adhere to.

Nexus metadata and data are stored in Hierarchical Data Format Files (HDF5) with
a .nxs extension although standards HDF5 extensions are sometimes used.
Files must use the ``.nxs`` file extension in order to use this io plugin.
Using the ``.nxs`` extension will default to the Nexus loader. If your file has
an HDF5 extension, you can also explicitly set the Nexus file reader:

.. code-block:: python

    # Load a NeXus file with a .h5 extension
    >>> s = hs.load("filename.h5", reader="nxs")

The loader will follow version 3 of the
`Nexus data rules <https://manual.nexusformat.org/datarules.html#version-3>`_.
The signal type, Signal1D or Signal2D, will be inferred by the ``interpretation``
attribute, if this is set to ``spectrum`` or ``image``, in the ``NXdata``
description. If the `interpretation
<https://manual.nexusformat.org/design.html#design-attributes>`_ attribute is
not set, the loader will return a ``BaseSignal``, which must then be converted
to the appropriate signal type. Following the Nexus data rules, if a ``default``
dataset is not defined, the loader will load NXdata
and HDF datasets according to the keyword options in the reader.
A number of the `Nexus examples <https://github.com/nexusformat/exampledata>`_
from large facilties do not use NXdata or use older versions of the Nexus
implementation. Data can still be loaded from these files but information or
associations may be missing. However, this missing information can be recovered
from within the  ``original_metadata`` which contains the overall structure of
the entry.

As the Nexus format uses the HDF5 format and needs to read both data and
metadata structured in different ways, the loader is written to be quite
flexible and can also be used to inspect any hdf5 based file.


Differences with respect to hspy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy metadata structure stores arrays as hdf datasets without attributes
and stores floats, ints and strings as attributes.
Nexus formats typically use hdf datasets attributes to store additional
information such as an indication of the units for an axis or the NX_class which
the dataset structure follows. The metadata, hyperspy  or original_metadata,
therefore needs to be able to indicate the values and attributes of a dataset.
To implement this structure the ``value`` and ``attrs`` of a dataset can also be
defined. The value of a dataset is set using a ``value`` key.
The attributes of a dataset are defined by an ``attrs`` key.

For example, to store an array called ``axis_x``, with a units attribute within
original_metadata, the following structure would be used:

::

    ├──original_metadata
    │   ├── axis_x
    │   │   ├── value : array([1.0,2.0,3.0,4.0,5.0])
    │   │   ├── attrs
    │   │   │   ├── units : mm


.. code-block:: python

    >>> original_metadata.set_item(axis_x.value,[1.0,2.0,3.0,4.0,5.0])
    >>> original_metadata.set_item(axis_x.attrs.units,"mm")

To access the axis information:

.. code-block:: python

    >>> original_metadata.axis_x.value
    >>> original_metadata.axis_x.attrs.units

To modify the axis information:

.. code-block:: python

    >>> original_metadata.axis_x.value = [2.0,3.0,4.0,5.0,6.0]
    >>> original_metadata.axis_x.attrs.units = "um"

To store data in a Nexus monochromator format, ``value``
and ``attrs`` keys can define additional attributes:

::

    ├── monochromator
    │   ├── energy
    │   │   ├── value : 12.0
    │   │   ├── attrs
    │   │   │   ├── units : keV
    │   │   │   ├── NXclass : NXmonochromator


The ``attrs`` key can also be used to define Nexus structures for the definition
of structures and relationships between data:

::

    ├── mydata
    │   ├── attrs
    │   │   ├── NX_class : "NXdata"
    │   │   ├── axes : ["x","."]
    │   ├── data
    │   │   ├──value : [[30,23...110]
    │   ├── x
    │   │   ├──value : [1,2.....100]
    │   │   ├── attrs
    │   │   │   ├── unit : "mm"


The use of ``attrs`` or ``value`` to set values within the metadata is optional
and metadata values can also be set, read or modified in the normal way.


.. code-block:: python

    >>> original_metadata.monochromator.energy = 12.5

HyperSpy metadata is stored within the Nexus file and should be automatically
restored when a signal is loaded from a previously saved Nexus file.

.. note::

    Altering the standard metadata structure of a signal
    using ``attrs`` or ``value`` keywords is not recommended.

Reading
^^^^^^^
Nexus files can contain multiple datasets within the same file, but the
ordering of datasets can vary depending on the setup of an experiment or
processing step when the data was collected.
For example, in one experiment Fe, Ca, P, Pb were collected but in the next experiment
Ca, P, K, Fe, Pb were collected. HyperSpy supports reading in one or more datasets
and returns a list of signals but in this example case the indexing is different.
To control which data or metadata is loaded and in what order
some additional loading arguments are provided.

Extra loading arguments
+++++++++++++++++++++++

- ``dataset_key``: ``None``, ``str`` or ``list`` of strings - Default is ``None`` . String(s) to search for in the path to find one or more datasets.
- ``dataset_path``: ``None``, ``str`` or ``list`` of strings - Default is ``None`` . Absolute path(s) to search for in the path to find one or more datasets.
- ``metadata_key``: ``None``, ``str`` or ``list`` of strings - Default is ``None`` . Absolute path(s) or string(s) to search for in the path to find metadata.
- ``skip_array_metadata``: ``bool`` - Default is False. Option to skip loading metadata that are arrays to avoid duplicating loading of data.
- ``nxdata_only``: ``bool`` - Default is False. Option to only convert NXdata formatted data to signals.
- ``hardlinks_only``: ``bool`` - Default is False. Option to ignore soft or External links in the file.
- ``use_default``: ``bool`` - Default is False. Only load the ``default`` dataset, if defined, from the file. Otherwise load according to the other keyword options.

.. note::

    Given that HDF5 files can accommodate very large datasets, setting ``lazy=True``
    is strongly recommended if the content of the HDF5 file is not known apriori.
    This prevents issues with regard to loading datasets far larger than memory.

    Also note that setting ``lazy=True`` leaves the file handle to the HDF5 file open
    and it can be closed with :py:meth:`~._signals.lazy.LazySignal.close_file`
    or when using :py:meth:`~._signals.lazy.LazySignal.compute` with ``close_file=True``.


Reading a Nexus file (a single Nexus dataset):

.. code-block:: python

    >>> sig = hs.load("sample.nxs")

By default, the loader will look for stored NXdata objects.
If there are hdf datasets which are not stored as NXdata, but which
should be loaded as signals, set the ``nxdata_only`` keyword to False and all
hdf datasets will be returned as signals:

.. code-block:: python

    >>> sig = hs.load("sample.nxs", nxdata_only=False)

We can load a specific dataset using the ``dataset_path`` keyword argument.
Setting it to the absolute path of the desired dataset will cause
the single dataset to be loaded:

.. code-block:: python

    >>> # Loading a specific dataset
    >>> hs.load("sample.nxs", dataset_path="/entry/experiment/EDS/data")

We can also choose to load datasets based on a search key using the
``dataset_key`` keyword argument. This can also be used to load NXdata not
outside of the ``default`` version 3 rules. Instead of providing an absolute
path, a string can be provided as well, and datasets with this key will be
returned. The previous example could also be written as:

.. code-block:: python

    >>> # Loading datasets containing the string "EDS"
    >>> hs.load("sample.nxs", dataset_key="EDS")

The difference between ``dataset_path`` and ``dataset_key`` is illustrated
here:

.. code-block:: python

    >>> # Only the dataset /entry/experiment/EDS/data will be loaded
    >>> hs.load("sample.nxs", dataset_path="/entry/experiment/EDS/data")
    >>> # All datasets contain the entire string "/entry/experiment/EDS/data" will be loaded
    >>> hs.load("sample.nxs", dataset_key="/entry/experiment/EDS/data")

Multiple datasets can be loaded by providing a number of keys:

.. code-block:: python

    >>> # Loading a specific dataset
    >>> hs.load("sample.nxs", dataset_key=["EDS", "Fe", "Ca"])

Metadata can also be filtered in the same way using ``metadata_key``:

.. code-block:: python

    >>> # Load data with metadata matching metadata_key
    >>> hs.load("sample.nxs", metadata_key="entry/instrument")

.. note::

    The Nexus loader removes any NXdata blocks from the metadata.

Metadata that are arrays can be skipped by using ``skip_array_metadata``:

.. code-block:: python

    >>> # Load data while skipping metadata that are arrays
    >>> hs.load("sample.nxs", skip_array_metadata=True)

Nexus files also support parameters or dimensions that have been varied
non-linearly. Since HyperSpy Signals expect linear variation of parameters /
axes, such non-linear information would be lost in the axes manager and
replaced with indices.
Nexus and HDF can result in large metadata structures with large datasets within the loaded
original_metadata. If lazy loading is used this may not be a concern but care must be taken
when saving the data. To control whether large datasets are loaded or saved,
use the ``metadata_key`` to load only the most relevant information. Alternatively,
set ``skip_array_metadata`` to ``True`` to avoid loading those large datasets in original_metadata.


Writing
^^^^^^^
Signals can be written to new Nexus files using the standard :py:meth:`~.signal.BaseSignal.save`
function.

Extra saving arguments
++++++++++++++++++++++
- ``save_original_metadata``: ``bool`` - Default is True, option to save the original_metadata when storing to file.
- ``skip_metadata_key``: ``bool`` - ``None``, ``str`` or ``list`` of strings - Default is ``None``. Option to skip certain metadata keys when storing to file.
- ``use_default``: ``bool`` - Default is False. Set the ``default`` attribute for the Nexus file.

.. code-block:: python

    >>> sig.save("output.nxs")

Using the save method will store the nexus file with the following structure:

::

    ├── entry1
    │   ├── signal_name
    │   │   ├── auxiliary
    │   │   │   ├── original_metadata
    │   │   │   ├── hyperspy_metadata
    │   │   │   ├── learning_results
    │   │   ├── signal_data
    │   │   │   ├── data and axes (NXdata format)


The original_metadata can include hdf datasets which you may not wish to store.
The original_metadata can be omitted using ``save_original_metadata``.

.. code-block:: python

    >>> sig.save("output.nxs", save_original_metadata=False)

If only certain metadata are to be ignored, use ``skip_metadata_key``:

.. code-block:: python

    >>> sig.save("output.nxs", skip_metadata_key=['xsp3', 'solstice_scan'])

To save multiple signals, the file_writer method can be called directly.

.. code-block:: python

    >>> from hyperspy.io_plugins.nexus import file_writer
    >>> file_writer("test.nxs",[signal1,signal2])

When saving multiple signals, a default signal can be defined. This can be used when storing
associated data or processing steps along with a final result. All signals can be saved but
a single signal can be marked as the default for easier loading in HyperSpy or plotting with Nexus tools.
The default signal is selected as the first signal in the list:

.. code-block:: python

    >>> from hyperspy.io_plugins.nexus import file_writer
    >>> import hyperspy.api as hs
    >>> file_writer("test.nxs", [signal1, signal2], use_default = True)
    >>> hs.load("test.nxs", use_default = True)

The output will be arranged by signal name:

::

    ├── entry1 (NXentry)
    │   ├── signal_name (NXentry)
    │   │   ├── auxiliary (NXentry)
    │   │   │   ├── original_metadata (NXcollection)
    │   │   │   ├── hyperspy_metadata (NXcollection)
    │   │   │   ├── learning_results  (NXcollection)
    │   │   ├── signal_data (NXdata format)
    │   │   │   ├── data and axes
    ├── entry2 (NXentry)
    │   ├── signal_name (NXentry)
    │   │   ├── auxiliary (NXentry)
    │   │   │   ├── original_metadata (NXcollection)
    │   │   │   ├── hyperspy_metadata (NXcollection)
    │   │   │   ├── learning_results (NXcollection)
    │   │   ├── signal_data (NXdata)
    │   │   │   ├── data and axes


.. note::

    Signals saved as nxs by this plugin can be loaded normally and the
    original_metadata, signal data, axes, metadata and learning_results
    will be restored. Model information is not currently stored.
    Nexus does not store how the data should be displayed.
    To preserve the signal details an additional navigation attribute
    is added to each axis to indicate if it is a navigation axis.


Inspecting
^^^^^^^^^^
Looking in a Nexus or HDF file for specific metadata is often useful - e.g. to find
what position a specific stage was at. The methods ``read_metadata_from_file``
and ``list_datasets_in_file`` can be used to load the file contents or
list the hdf datasets contained in a file. The inspection methods use the same ``metadata_key`` or ``dataset_key`` as when loading.
For example to search for metadata in a file:

    >>> from hyperspy.io_plugins.nexus import read_metadata_from_file
    >>> read_metadata_from_file("sample.hdf5",metadata_key=["stage1_z"])
    {'entry': {'instrument': {'scannables': {'stage1': {'stage1_z': {'value': -9.871700000000002,
    'attrs': {'gda_field_name': 'stage1_z',
    'local_name': 'stage1.stage1_z',
    'target': '/entry/instrument/scannables/stage1/stage1_z',
    'units': 'mm'}}}}}}}

To list the datasets stored in the file:

    >>> from hyperspy.io_plugins.nexus import read_datasets_from_file
    >>> list_datasets_in_file("sample.nxs")
    NXdata found
    /entry/xsp3_addetector
    /entry/xsp3_addetector_total
    HDF datasets found
    /entry/solstice_scan/keys/uniqueKeys
    /entry/solstice_scan/scan_shape
    Out[3]:
    (['/entry/xsp3_addetector', '/entry/xsp3_addetector_total'],
     ['/entry/solstice_scan/keys/uniqueKeys', '/entry/solstice_scan/scan_shape'])


.. _sur-format:

SUR and PRO format
------------------

This is a format developed by the digitalsurf company to handle various types of
scientific measurements data such as profilometer, SEM, AFM, RGB(A) images, multilayer
surfaces and profiles. Even though it is essentially a surfaces format, 1D signals
are supported for spectra and spectral maps. Specifically, this file format is used
by Attolight SA for its scanning electron microscope cathodoluminescence
(SEM-CL) hyperspectral maps. Metadata parsing is supported, including user-specific
metadata, as well as the loading of files containing multiple objects packed together.

The plugin was developed based on the MountainsMap software documentation, which
contains a description of the binary format.

.. _empad-format:

EMPAD format
------------

This is the file format used by the Electron Microscope Pixel Array
Detector (EMPAD). It is used to store a series of diffraction patterns from
scanning transmission electron diffraction measurements, with a limited set of
metadata. Similarly, to the :ref:`ripple format <ripple-format>`, the raw data
and metadata are saved in two different files and for the EMPAD reader, these
are saved in the ``raw`` and ``xml`` files, respectively. To read EMPAD data,
use the ``xml`` file:

.. code-block:: python

    >>> sig = hs.load("file.xml")


which will automatically read the raw data from the ``raw`` file too. The
filename of the ``raw`` file is defined in the ``xml`` file, which implies
changing the file name of the ``raw`` file will break reading the file.


.. _elid_format-label:

Phenom ELID format
------------------

This is the file format used by the software package Element Identification for the Thermo
Fisher Scientific Phenom desktop SEM. It is a proprietary binary format which can contain
images, single EDS spectra, 1D line scan EDS spectra and 2D EDS spectrum maps. The reader
will convert all signals and its metadata into hyperspy signals.

The current implementation supports ELID files created with Element Identification version
3.8.0 and later. You can convert older ELID files by loading the file into a recent Element
Identification release and then save the ELID file into the newer file format.

.. _jeol_format-label:

JEOL ASW format
---------------

This is the file format used by the `JEOL Analysist Station software` for which
hyperspy can read the ``asw``, ``pts``, ``map`` and ``eds`` format. To read the
calibration, it is required to load the ``asw`` file, which will load all others
files automatically.

Extra loading arguments
^^^^^^^^^^^^^^^^^^^^^^^

- ``rebin_energy`` : int, default 1.
  Factor used to rebin the energy dimension. It must be a
  factor of the number of channels, typically 4096.
- ``sum_frames`` : bool, default True.
  If False, each individual frame (sweep in JEOL software jargon)
  is loaded. Be aware that loading each individual will use a lot of memory,
  however, it can be used in combination with ``rebin_energy``, ``cutoff_at_kV``
  and ``downsample`` to reduce memory usage.
- ``SI_dtype`` : dtype, default np.uint8.
  set dtype of the eds dataset. Useful to adjust memory usage
  and maximum number of X-rays per channel.
- ``cutoff_at_kV`` : int, float, or None, default None.
  if set (>= 0), use to crop the energy range up the specified energy.
  If ``None``, the whole energy range is loaded.
  Useful to reduce memory usage.
- ``downsample`` : int, default 1.
  the downsample ratio of the navigation dimension of EDS
  dataset, it can be integer or a tuple of length 2 to define ``x`` and ``y``
  separetely and it must be a mutiple of the size of the navigation dimension.
- ``only_valid_data`` : bool, default True.
  for ``pts`` file only, ignore incomplete and partly
  acquired last frame, which typically occurs when the acquisition was
  interrupted. When loading incomplete data (``only_valid_data=False``),
  the missing data are filled with zeros. If ``sum_frames=True``, this argument
  will be ignored to enforce consistent sum over the mapped area.
- ``read_em_image`` : bool, default False.
  for ``pts`` file only, If ``read_em_image=True``,
  read SEM/STEM image from pts file if available. In this case, both
  spectrum Image and SEM/STEM Image will be returned as list.
- ``frame_list`` : list of integer or None, default None
  for ``pts`` file only, frames in frame_list will be loaded.
  for example, ``frame_list=[1,3]`` means second and forth frame will be loaded.
  If ``None``, all frames are loaded.
- ``frame_shifts`` : list of [int, int], list of [int, int, int], or None, default None
  for ``pts`` file only, each frame will be loaded with offset of
  [dy, dx (, and optionary dEnergy)]. Units are pixels/channels.
  The result of estimate_shift2D() can be used as a parameter of frame_shifts.
  This is useful for express drift correction. Not suitable for accurate analysis.
- ``lazy`` : bool, default False
  for ``pts`` file only, spectrum image is loaded as a dask.array if lazy == true.
  This is useful to reduce memory usage, with cost of cpu time for calculation.


Example of loading data downsampled, and with energy range cropped with the
original navigation dimension 512 x 512 and the EDS range 40 keV over 4096
channels:

.. code-block:: python

    >>> hs.load("sample40kv.asw", downsample=8, cutoff_at_kV=10)
    [<Signal2D, title: IMG1, dimensions: (|512, 512)>,
     <Signal2D, title: C K, dimensions: (|512, 512)>,
     <Signal2D, title: O K, dimensions: (|512, 512)>,
     <EDSTEMSpectrum, title: EDX, dimensions: (64, 64|1096)>]

load the same file without extra arguments:

.. code-block:: python

    >>> hs.load("sample40kv.asw")
    [<Signal2D, title: IMG1, dimensions: (|512, 512)>,
     <Signal2D, title: C K, dimensions: (|512, 512)>,
     <Signal2D, title: O K, dimensions: (|512, 512)>,
     <EDSTEMSpectrum, title: EDX, dimensions: (512, 512|4096)>]

.. _tvips_format-label:

TVIPS format
------------

The TVIPS format is the default format for image series collected by pixelated
cameras from the TVIPS company. Typically individual images captured by these
cameras are stored in the :ref:`TIFF format<tiff-format>` which can also be
loaded by Hyperspy. This format instead serves to store image streams from
in-situ and 4D-STEM experiments. During collection, the maximum file size is
typically capped meaning the dataset is typically split over multiple files
ending in `_xyz.tvips`. The `_000.tvips` will contain the main header and
it is essential for loading the data. If a filename is provided for loading
or saving without a `_000` suffix, this will automatically be added. Loading
will not work if no such file is found.

.. warning::

   While TVIPS files are supported, it is a proprietary format, and future
   versions of the format might therefore not be readable. Complete
   interoperability with the official software can neither be guaranteed.

.. warning::

   The TVIPS format currently stores very limited amount of metadata about
   scanning experiments. To reconstruct scan data, e.g. 4D-STEM datasets,
   parameters like the shape and scales of the scan dimensions should be
   manually recorded.

Extra loading arguments
^^^^^^^^^^^^^^^^^^^^^^^
- ``scan_shape``: a tuple of integers to indicate the shape of the navigation
  axes. For example, `(3, 4)` will have 3 scan points in the y direction and 4
  in the x direction. If this argument is not provided, the data will be loaded
  as a 1D stack of images. `auto` is also an option which aims to reconstruct
  the scan based on the `rotidx` indices in frame headers. Since this only
  works for square scan grids and is prone to failure, this option is not
  recommended.
- ``scan_start_frame``: index of the first frame of the dataset to consider,
  mainly relevant for 4D-STEM datasets. If `scan_shape="auto"` this parameter
  is ignored.
- ``winding_scan_axis``: if the acquisition software collected data without
  beam flyback but with a winding "snake" scan, then every second scan row
  or column needs to be reversed to make sense of the data. This can be
  indicated with values `"x"` or `"y"`, depending on whether winding happened
  along the primary or secondary axis. By default, flyback scan without winding
  is assumed with `x` the fast scan and `y` the slow scan direction.
- ``hysteresis``: if winding scan was active it is likely there is an overshoot
  of a few pixels (2-5) every second scan row. This parameter shifts every
  second row by the indicated number of scan points to align even and odd scan
  rows. Default is 0, no hysteresis.
- ``rechunking``: only relevant when using lazy loading. If set to `False`
  each TVIPS file is loaded as a single chunk. For a better experience, with the
  default setting of `auto` rechunking is performed such that the navigation axes
  are optimally chunked and the signal axes are not chunked. If set to anything else, the
  value will be passed to the `chunks` argument in `dask.array.rechunk`.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``max_file_size``: approximate maximum size of individual files in bytes.
  In this way a dataset can be split into multiple files. A file needs to be
  at least the size of the main header in the first file plus one frame and its
  frame header. By default there is no maximum and the entire dataset is saved
  to one file.
- ``version``: TVIPS file format version, defaults to version 2. Only version
  1 and 2 are currently supported.
- ``frame_header_extra_bytes``: the number of bytes to pad the frame headers
  with. By default this is 0.
- ``mode``: `1` for imaging, `2` for diffraction. By default the mode is
  guessed from the signal type and signal unites.

Reading data generated by HyperSpy using other software packages
================================================================

The following scripts may help reading data generated by HyperSpy using
other software packages.


.. _import-rpl:

ImportRPL Digital Micrograph plugin
-----------------------------------

This Digital Micrograph plugin is designed to import Ripple files into Digital Micrograph.
It is used to ease data transit between DigitalMicrograph and HyperSpy without losing
the calibration using the extra keywords that HyperSpy adds to the standard format.

When executed it will ask for 2 files:

#. The riple file with the data  format and calibrations
#. The data itself in raw format.

If a file with the same name and path as the riple file exits
with raw or bin extension it is opened directly without prompting.
ImportRPL was written by Luiz Fernando Zagonel.

`Download ImportRPL <https://github.com/downloads/hyperspy/ImportRPL/ImportRPL.s>`_


HDF5 reader plugin for Digital Micrograph
-----------------------------------------

This Digital Micrograph plugin is designed to import HDF5 files and like the
`ImportRPL` script above, it can be used to easily transfer data from HyperSpy to
Digital Micrograph by using the HDF5 hyperspy format (``hspy`` extension).

Download ``gms_plugin_hdf5`` from its `Github repository <https://github.com/niermann/gms_plugin_hdf5>`__.


.. _hyperspy-matlab:

readHyperSpyH5 MATLAB Plugin
----------------------------

This MATLAB script is designed to import HyperSpy's saved HDF5 files (``.hspy`` extension).
Like the Digital Micrograph script above, it is used to easily transfer data
from HyperSpy to MATLAB, while retaining spatial calibration information.

Download ``readHyperSpyH5`` from its `Github repository <https://github.com/jat255/readHyperSpyH5>`__.
