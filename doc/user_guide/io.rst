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
To load data use the :py:func:`~.io.load` command. For example, to load the
image ascent.jpg you can type:

.. code-block:: python

    >>> s = hs.load("ascent.jpg")

If loading was successful, the variable ``s`` contains a HyperSpy signal
or a signal of the :ref:`HyperSpy extensions <hyperspy_extensions-label>`
- see available :ref:`signal subclasses <transforming_signal-label>` for more
information.
If the loaded file contains several datasets, the :py:func:`~.io.load`
functions will return a list of the corresponding signal.

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

HyperSpy will try to guess the most likely data type for the corresponding
file. However, you can force it to read the data as a particular data type by
providing the ``signal`` keyword, which has to be one of: ``spectrum``,
``image`` or ``EELS``, e.g.:

.. code-block:: python

    >>> s = hs.load("filename", signal_type="EELS")

Some file formats store some extra information about the data, which can be
stored in "attributes". If HyperSpy manages to read some extra information
about the data it stores it in the
:py:attr:`~.signal.BaseSignal.original_metadata` attribute. Also, it is
possible that other information will be mapped by HyperSpy to a standard
location where it can be used by some standard routines, the
:py:attr:`~.signal.BaseSignal.metadata` attribute.

To print the content of the parameters simply:

.. code-block:: python

    >>> s.metadata

The :py:attr:`~.signal.BaseSignal.original_metadata` and
:py:attr:`~.signal.BaseSignal.metadata` can be exported to  text files
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

Almost all file readers support accessing the data without reading it to memory
(see :ref:`supported-formats` for a list). This feature can be useful when
analysing large files. To load a file without loading it to memory simply set
``lazy`` to ``True`` e.g.:

The units of the navigation and signal axes can be converted automatically
during loading using the ``convert_units`` parameter. If `True`, the
``convert_to_units`` method of the ``axes_manager`` will be used for the conversion
and if set to `False`, the units will not be converted. The default is `False`.

.. code-block:: python

    >>> s = hs.load("filename.hspy", lazy=True)

More details on lazy evaluation support in :ref:`big-data-label`.

.. _load-multiple-label:

Loading multiple files
----------------------

Rather than loading files individually, several files can be loaded with a
single command. This can be done by passing a list of filenames to the load
functions, e.g.:

.. code-block:: python

    >>> s = hs.load(["file1.hspy", "file2.hspy"])

or by using `shell-style wildcards <http://docs.python.org/library/glob.html>`_:

.. code-block:: python

    >>> s = hs.load("file*.hspy")

.. note::

    Wildcards are implemented using ``glob.glob()``, which treats ``*``, ``[``
    and ``]`` as special characters for pattern matching. If your filename or
    path contains square brackets, you may want to escape these characters first.

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
HyperSpy can stack the data of the files contain data with exactly the same
dimensions. If this is not the case an error is raised. If each file contains
multiple (N) signals, N stacks will be created. Here, the numbers of signals
per file must also match, or an error will be raised.

It is also possible to load multiple files with a single command without
stacking them by passing the `stack=False` argument to the load function, in
which case the function will return a list of objects, e.g.:

.. code-block:: python

    >>> ls
    CL1.raw  CL1.rpl~  CL2.rpl  CL3.rpl  CL4.rpl  LL3.raw  shift_map-          SI3.npy
    CL1.rpl  CL2.raw   CL3.raw  CL4.raw  hdf5/    LL3.rpl
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
extension. If the filename does not contain the extension the default format
(:ref:`hspy-format`) is used. For example, if the :py:const:`s` variable
contains the :py:class:`~.signal.BaseSignal` that you want to write to a file,
the following will write the data to a file called :file:`spectrum.hspy` in the
default :ref:`hspy-format` format:

.. code-block:: python

    >>> s.save('spectrum')

If you want to save in the :ref:`ripple format <ripple-format>` write
instead:

.. code-block:: python

    >>> s.save('spectrum.rpl')

Some formats take extra arguments. See the relevant subsection of
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
    | Image: jpg                        |    Yes |    Yes |    Yes |
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
    | DENS heater log                   |    Yes |    No  |    No  |
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
    | EDAX .spc and .spd                |    Yes |    No  |    Yes |
    +-----------------------------------+--------+--------+--------+
    | h5USID .h5                        |    Yes |   Yes  |   Yes  |
    +-----------------------------------+--------+--------+--------+
    | Phenom .elid                      |    Yes |    No  |    No  |
    +-----------------------------------+--------+--------+--------+
    | DigitalSurf's .sur and .pro       |    Yes |    No  |    No  |
    +-----------------------------------+--------+--------+--------+
    | Nexus .nxs                        |    Yes |   Yes  |   Yes  |
    +-----------------------------------+--------+--------+--------+
    | EMPAD .xml                        |    Yes |    No  |   Yes  |
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
:py:meth:`~hyperspy._signals.eds.EDS_mixin.get_lines_intensity`:

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
full signal shape. It is possible to customise the chunk shape using the ``chunks`` keyword. 
For example, to save the data with ``(20, 20, 256)`` chunks instead of the default ``(7, 7, 2048)`` chunks
for this signal:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.random.random((100, 100, 2048)))
    >>> s.save("test_chunks", chunks=(20, 20, 256))

Note that currently it is not possible to pass different customised chunk shapes to all signals and
arrays contained in a signal and its metadata. Therefore, the value of ``chunks`` provided on saving
will be applied to all arrays contained in the signal.

By passing ``True`` to ``chunks`` the chunk shape is guessed using ``h5py``'s ``guess_chunks`` function
what, for large signal spaces usually leads to smaller chunks as ``guess_chunks`` does not impose the
constrain of storing at least one signal per chunks. For example, for the signal in the example above
passing ``chunks=True`` results in ``(7, 7, 256)`` chunks.

Choosing the correct chunk-size can significantly affect the speed of reading, writing and performance of many hyperspy algorithms.
See the `chunking section <big_data.html#Chunking>`__ under `Working with big data <big_data.html>`__ for more information.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^^
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

MRCZ also supports asychronous reads and writes.

Repository: https://github.com/em-MRCZ
PyPI:       https://pypi.python.org/pypi/mrcz
Citation:   Submitted.
Preprint:   http://www.biorxiv.org/content/early/2017/03/13/116533

Support for this format is not enabled by default. In order to enable it
install the `mrcz` and optionally the `blosc` Python packages.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``do_async``: currently supported within Hyperspy for writing only, this will
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
<http://www.amc.anl.gov/ANLSoftwareLibrary/02-MMSLib/XEDS/EMMFF/EMMFF.IBM/Emmff.Total>`__
is widely used to exchange single spectrum data, but it does not support
multidimensional data. It can be used to exchange single spectra with Gatan's
Digital Micrograph.

.. WARNING::
    If several spectra are loaded and stacked (``hs.load('pattern', stack_signals=True``)
    the calibration read from the first spectrum and applied to all other spectra.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^^

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

This `open standard format
<http://www.nist.gov/lispix/doc/image-file-formats/raw-file-format.htm>`__ is
widely used to exchange multidimensional data. However, it only supports data of
up to three dimensions. It can be used to exchange data with Bruker and `Lispix
<http://www.nist.gov/lispix/>`_. Used in combination with the :ref:`import-rpl`
it is very useful for exporting data to Gatan's Digital Micrograph.

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

HyperSpy is able to read and write data too all the image formats supported by
`the Python Image Library <http://www.pythonware.com/products/pil/>`_ (PIL).
This includes png, pdf, gif etc.

It is important to note that these image formats only support 8-bit files, and
therefore have an insufficient dynamic range for most scientific applications.
It is therefore highly discouraged to use any general image format (with the
exception of :ref:`tiff-format` which uses another library) to store data for
analysis purposes.

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
FEI, Zeiss SEM and Olympus SIS software.

.. code-block:: python

    >>> # Force read image resolution using the x_resolution, y_resolution and
    >>> # the resolution_unit of the TIFF tags. Be aware, that most of the
    >>> # software doesn't (properly) use these tags when saving TIFF files.
    >>> s = hs.load('file.tif', force_read_resolution=True)

HyperSpy can also read and save custom tags through the ``tifffile``
library.

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
complete (and probably they will be unless Gatan releases the specifications of
the format). That said, we understand that this is an important feature and if
loading a particular Digital Micrograph file fails for you, please report it as
an issue in the `issues tracker <https://github.com/hyperspy/hyperspy/issues>`__ to make
us aware of the problem.

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

EDAX TEAM SPD and SPC
---------------------

HyperSpy can read both ``.spd`` (spectrum image) and ``.spc`` (single spectra)
files from the EDAX TEAM software.
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. _dens-format:

DENS heater log
---------------

HyperSpy can read heater log format for DENS solution's heating holder. The
format stores all the captured data for each timestamp, together with a small
header in a plain-text format. The reader extracts the measured temperature
along the time axis, as well as the date and calibration constants stored in
the header.

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
- ``cutoff_at_kV`` : if set (can be int or float >= 0) can be used either to crop
  or enlarge energy (or channels) range at max values (default None).

Example of loading reduced (downsampled, and with energy range cropped)
"spectrum only" data from bcf (original shape: 80keV EDS range (4096 channels),
100x75 pixels):

.. code-block:: python

    >>> hs.load("sample80kv.bcf", select_type='spectrum', downsample=2, cutoff_at_kV=10)
    <EDSSEMSpectrum, title: EDX, dimensions: (50, 38|595)>

load the same file without extra arguments:

.. code-block:: python

    >>> hs.load("sample80kv.bcf")
    [<Signal2D, title: BSE, dimensions: (|100, 75)>,
    <Signal2D, title: SE, dimensions: (|100, 75)>,
    <EDSSEMSpectrum, title: EDX, dimensions: (100, 75|1095)>]

The loaded array energy dimension can by forced to be larger than the data
recorded by setting the 'cutoff_at_kV' kwarg to higher value:

.. code-block:: python

    >>> hs.load("sample80kv.bcf", cutoff_at_kV=80)
    [<Signal2D, title: BSE, dimensions: (|100, 75)>,
    <Signal2D, title: SE, dimensions: (|100, 75)>,
    <EDSSEMSpectrum, title: EDX, dimensions: (100, 75|4096)>]

Note that setting downsample to >1 currently locks out using SEM imagery
as navigator in the plotting.

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
    Lazy reading of these datasets is not supported. Making FFTs with HyperSpy
    from the respective image datasets is recommended.

.. note::
    
    DPC data is loaded in as a HyperSpy ComplexSignal2D object. Lazy reading of these
    datasets is not supported.

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
with an an extra navigation dimension corresponding to the frame index
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
precisely by supporting non-linear axis.


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

We can load a specific dataset using the ``dataset_path`` keyword argument. setting it to the
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

h5USID files also support parameters or dimensions that have been varied non-linearly.
This capability is important in several spectroscopy techniques where the bias is varied as a
`bi-polar triangular waveform <https://pycroscopy.github.io/pyUSID/auto_examples/beginner/plot_usi_dataset.html#values-for-each-dimension>`_
rather than linearly from the minimum value to the maximum value.
Since HyperSpy Signals expect linear variation of parameters / axes, such non-linear information
would be lost in the axes manager. The USID plugin will default to a warning
when it encounters a parameter that has been varied non-linearly:

.. code-block:: python

    >>> hs.load("sample.h5")
    UserWarning: Ignoring non-linearity of dimension: Bias
    <BaseSignal, title: , dimensions: (|7, 3, 5, 2)>

Obviously, the
In order to prevent accidental misinterpretation of information downstream, the keyword argument
``ignore_non_linear_dims`` can be set to ``False`` which will result in a ``ValueError`` instead.

.. code-block:: python

    >>> hs.load("sample.h5")
    ValueError: Cannot load provided dataset. Parameter: Bias was varied non-linearly.
    Supply keyword argument "ignore_non_linear_dims=True" to ignore this error

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
`NeXus <https://www.nexusformat.org>`_ is a common data format orginally
developed by the neutron, x-ray communities. It is still being developed as
an international standard by scientists and programmers representing major
scientific facilities in order to facilitate greater cooperation in the analysis
and visualization of data.
Nexus uses a variety of classes to record data, values,
units and other experimental metadata associated with an experiment.
For specific types of experiments an Application Definition may exist which
defines an agreed common layout that facilities can adhere to.

Nexus metadata and data are stored in Hierarchical Data Format Files (HDF5) with
a .nxs extension although standards HDF5 extensions are sometimes used.
Files must use the ``.nxs`` file extension in order to use this io plugin.
Using the ``.nxs`` extension will default to the Nexus loader. If your file has
a HDF5 extension, you can also explicitly set the Nexus file reader:

.. code-block:: python

    # Load a NeXus file with a .h5 extension
    >>> s = hs.load("filename.h5", reader="nxs")

The loader will follow version 3 of the
`Nexus data rules <https://manual.nexusformat.org/datarules.html#version-3>`_.
The signal type, Signal1D or Signal2D, will be inferred by the ``interpretation`` attribute,
if this set to ``spectrum`` or ``image``, in the ``NXdata`` description. If the
`interpretation <https://manual.nexusformat.org/design.html#design-attributes>`_
attribute is not set the loader will return a ``BaseSignal`` which must then be
converted to the appropriate signal type.
Following the Nexus data rules if a  ``default`` dataset is not defined the loader will load NXdata
and HDF datasets according to the keyword options in the reader.
A number of the `Nexus examples <https://github.com/nexusformat/exampledata>`_ from large facilties
don't use NXdata or use older versions of the Nexus implementation.
Data can still be loaded from these files but information or associations may be missing.
This missing information can however be recovered from
within the ``original_metadata`` which contains the overall structure of the entry.

As the Nexus format uses HDF5 and needs to read data and metadata structured
in different ways the loader is written to quite flexible and can also be used
to inspect any hdf5 based file.


Differences with respect to hspy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Hyperspy metadata structure stores arrays as hdf datasets without attributes
and stores floats,ints and strings as attributes.
Nexus formats typcial use hdf datasets attributes to store additional
information such as an indication of the units for an axis or the NX_class which
the dataset structure follows. The metadata, hyperspy  or original_metadata,
therefore needs to be able to indicate the values and attributes of a dataset.
To implement this structure the ``value`` and ``attrs`` of a dataset can also be
defined. The value of a dataset is set using a ``value`` key.
The attributes of a dataset are defined by an ``attrs`` key.

For example to store an array, called axis_x, with a units attribute within
original_metadata the following structure would be used.

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

To store data in a Nexus monochromator format the ``value``
and ``attrs``  can define additional attributes.

::

    ├── monochromator
    │   ├── energy
    │   │   ├── value : 12.0
    │   │   ├── attrs
    │   │   │   ├── units : keV
    │   │   │   ├── NXclass : NXmonochromator


The ``attrs`` key can also to define Nexus structures to define
structures and relationships between data.

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

Hyperspy metadata is stored within the Nexus file and should be automatically
restored when a signal is loaded from a previously saved Nexus file.

.. note::

    Altering the standard metadata structure of a signal
    using ``attrs`` or ``value`` keywords is not recommended.

Reading
^^^^^^^
Nexus files can contain multiple datasets within the same file but the
ordering of datasets can vary depending on the setup of an experiment or
processing step when the data was collected.
For example in one experiment Fe, Ca, P, Pb were collected but in the next experiment
Ca, P, K, Fe, Pb were collected. HyperSpy supports reading in one or more datasets
and returns a list of signals but in this example case the indexing is different.
To control which data or metadata is loaded and in what order
some additional loading arguments are provided.

Extra loading arguments
+++++++++++++++++++++++
- ``dataset_keys``: ``None``, ``str`` or ``list`` of strings - Default is ``None`` . Absolute path(s) or string(s) to search for in the path to find one or more datasets.
- ``metadata_keys``: ``None``, ``str`` or ``list`` of strings - Default is ``None`` . Absolute path(s) or string(s) to search for in the path to find metadata.
- ``nxdata_only``: ``bool`` - Default is False. Option to only convert NXdata formatted data to signals.
- ``hardlinks_only``: ``bool`` - Default is False. Option to ignore soft or External links in the file.
- ``use_default``: ``bool`` - Default is False. Only load the ``default`` dataset, if defined, from the file. Otherwise load according to the other keyword options.

.. note::

    Given that HDF5 files can accommodate very large datasets, setting ``lazy=True``
    is strongly recommended if the contents of the HDF5 file are not known apriori.
    This prevents issues with regard to loading datasets far larger than memory.

    Also note that setting ``lazy=True`` leaves the file handle to the HDF5 file open
    and it can be closed with :py:meth:`~._signals.lazy.LazySignal.close_file`
    or when using :py:meth:`~._signals.lazy.LazySignal.compute` with ``close_file=True``.


Reading a Nexus file a single Nexus dataset:

.. code-block:: python

    >>> sig = hs.load("sample.nxs")

By default the loader will look for stored NXdata objects.
If there are hdf datasets which are not stored as NXdata but which
should be loaded as signals set the ``nxdata_only`` keyword to False and all
hdf datasets will be returned as signals.

.. code-block:: python

    >>> sig = hs.load("sample.nxs",nxdata_only=False)

We can load a specific datasets using the ``dataset_keys`` keyword argument.
Setting it to the absolute path of the desired dataset will cause
the single dataset to be loaded.

.. code-block:: python

    >>> # Loading a specific dataset
    >>> hs.load("sample.nxs", dataset_keys='/entry/experiment/EDS/data')

We can also choose to load datasets based on a search key using the
``dataset_keys`` keyword argument. This can also be used to load NXdata not
outside of the ``default`` version 3 rules. Instead of providing an absolute path
a strings to can be provided and datasets with this key will be returned.
The previous example could also be written as:

.. code-block:: python

    >>> # Loading a specific dataset
    >>> hs.load("sample.nxs", dataset_keys="EDS")

Multiple datasets can be loaded by providing a number of keys:

.. code-block:: python

    >>> # Loading a specific dataset
    >>> hs.load("sample.nxs", dataset_keys=["EDS", "Fe", "Ca"])

Metadata can also be filtered in the same way using ``metadata_keys``

.. code-block:: python

    >>> # Load data with metadata matching metadata_keys
    >>> hs.load("sample.nxs", metadata_keys="entry/instrument")

.. note::

    The Nexus loader removes any NXdata blocks from the metadata.


Nexus files also support parameters or dimensions that have been varied
non-linearly.
Since HyperSpy Signals expect linear variation of parameters / axes, such
non-linear information would be lost in the axes manager and replaced with
indices.
Nexus and HDF can result in large metadata structures with large datasets within the loaded
original_metadata. If lazy loading is used this may not be a concern but care must be taken
when saving the data.
To control whether large datasets are loaded or saved  the
use the ``metadata_keys`` to load only the most relevant information.


Writing
^^^^^^^
Signals can be written to new Nexus files using the standard :py:meth:`~.signal.BaseSignal.save`
function.

Extra saving arguments
++++++++++++++++++++++
- ``save_original_metadata``: ``bool`` - Default is True, Option to save the original_metadata when storing to file.
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

To save multiple signals the file_writer method can be called directly.

.. code-block:: python

    >>> from hyperspy.io_plugins.nexus import file_writer
    >>> file_writer("test.nxs",[signal1,signal2])

When saving multiple signals a default signal can be defined. This can be used when storing
associated data or processing steps along with a final result. All signals can be saved but
a single signal can be marked as the default for easier loading in hyperspy or plotting with Nexus tools.
The default signal is selected as the first signal in the list.

.. code-block:: python

    >>> from hyperspy.io_plugins.nexus import file_writer
    >>> import hyperspy.api as hs
    >>> file_writer("test.nxs", [signal1,signal2], use_default = True)
    >>> hs.load("test.nxs", use_default = True)

The output will be arranged by signal name.

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
    is added to each axis to indicate if is a navigation axis.


Inspecting
^^^^^^^^^^
Looking in a Nexus or HDF file for specific metadata is often useful - .e.g to find
what position a specific stage was at. The methods ``read_metadata_from_file``
and ``list_datasets_in_file`` can be used to load the file contents or
list the hdf datasets contained in a file. The inspection methods use the same ``metadata_keys`` or ``dataset_keys`` as when loading.
For example to search for metadata in a file:

    >>> from hyperspy.io_plugins.nexus import read_metadata_from_file
    >>> read_metadata_from_file("sample.hdf5",metadata_keys=["stage1_z"])
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
scientific measurements data such as profilometer,SEM,AFM,RGB(A) images, multilayer
surfaces and profiles. Even though it is essentially a surfaces format, 1D signals
are supported for spectra and spectral maps. Specifically, this file format is used
by Attolight SA for the its Scanning Electron Microscope Cathodoluminescence
(SEM-CL) hyperspectral maps. Metadata parsing is supported, including user-specific
metadata, as well as the loading of files containing multiple objects packed together.

The plugin was developed based on the MountainsMap software documentation which
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
`ImportRPL` script above, it can used to easily transfer data from HyperSpy to
Digital Micrograph by using the HDF5 hyperspy format (``hspy`` extension).

Download ``gms_plugin_hdf5`` from its `Github repository <https://github.com/niermann/gms_plugin_hdf5>`_.


.. _hyperspy-matlab:

readHyperSpyH5 MATLAB Plugin
----------------------------

This MATLAB script is designed to import HyperSpy's saved HDF5 files (``.hspy`` extension).
Like the Digital Micrograph script above, it is used to easily transfer data
from HyperSpy to MATLAB, while retaining spatial calibration information.

Download ``readHyperSpyH5`` from its `Github repository <https://github.com/jat255/readHyperSpyH5>`_.
