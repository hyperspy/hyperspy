.. _io:

***********************
Loading and saving data
***********************

.. contents::

.. _loading_files:

Loading files: the load function
================================

HyperSpy can read and write to multiple formats (see :ref:`supported-formats`).
To load data use the :py:func:`~.io.load` command. For example, to load the
image lena.jpg you can type:

.. code-block:: python

    >>> s = hs.load("lena.jpg")

If the loading was successful, the variable :guilabel:`s` contains a generic
:py:class:`~.signal.BaseSignal`, a :py:class:`~._signals.signal1d.Signal1D` or
an :py:class:`~._signals.signal2d.Signal2D`.

.. NOTE::
    Note for python programmers: the data is stored in a numpy array
    in the :py:attr:`~.signal.BaseSignal.data` attribute, but you will not
    normally need to access it there.)


HyperSpy will try to guess the most likely data type for the corresponding
file. However, you can force it to read the data as a particular data type by
providing the ``signal`` keyword, which has to be one of: ``spectrum``,
``image`` or ``EELS``, e.g.:

.. code-block:: python

    >>> s = hs.load("filename", signal = "EELS")

Some file formats store some extra information about the data, which can be
stored in "attributes". If HyperSpy manages to read some extra information
about the data it stores it in `~.signal.BaseSignal.original_metadata`
attribute. Also, it is possible that other information will be mapped by
HyperSpy to a standard location where it can be used by some standard routines,
the :py:attr:`~.signal.BaseSignal.metadata` attribute.

To print the content of the parameters simply:

.. code-block:: python

    >>> s.metadata

::
Th :py:attr:`~.signal.BaseSignal.original_metadata` and
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

.. code-block:: python

    >>> s = hs.load("filename.hdf5", lazy=True)

More details on lazy evaluation support in :ref:`big-data-label`.


Loading multiple files
----------------------

Rather than loading files individually, several files can be loaded with a
single command. This can be done by passing a list of filenames to the load
functions, e.g.:

.. code-block:: python

    >>> s = hs.load(["file1.hdf5", "file2.hdf5"])

or by using `shell-style wildcards <http://docs.python.org/library/glob.html>`_


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
(:ref:`hdf5-format`) is used. For example, if the :py:const:`s` variable
contains the :py:class:`~.signal.BaseSignal` that you want to write to a file,
the following will write the data to a file called :file:`spectrum.hdf5` in the
default :ref:`hdf5-format` format:

.. code-block:: python

    >>> s.save('spectrum')

If instead you want to save in the :ref:`ripple-format` write instead:

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

    +--------------------+--------+--------+--------+
    | Format             | Read   | Write  | lazy   |
    +====================+========+========+========+
    | Gatan's dm3        |    Yes |    No  |    Yes |
    +--------------------+--------+--------+--------+
    | Gatan's dm4        |    Yes |    No  |    Yes |
    +--------------------+--------+--------+--------+
    | FEI's emi and ser  |    Yes |    No  |    Yes |
    +--------------------+--------+--------+--------+
    | HDF5               |    Yes |    Yes |    Yes |
    +--------------------+--------+--------+--------+
    | Image: jpg..       |    Yes |    Yes |    Yes |
    +--------------------+--------+--------+--------+
    | TIFF               |    Yes |    Yes |    Yes |
    +--------------------+--------+--------+--------+
    | MRC                |    Yes |    No  |    Yes |
    +--------------------+--------+--------+--------+
    | EMSA/MSA           |    Yes |    Yes |    No  |
    +--------------------+--------+--------+--------+
    | NetCDF             |    Yes |    No  |    No  |
    +--------------------+--------+--------+--------+
    | Ripple             |    Yes |    Yes |    Yes |
    +--------------------+--------+--------+--------+
    | SEMPER unf         |    Yes |    Yes |    Yes |
    +--------------------+--------+--------+--------+
    | Blockfile          |    Yes |    Yes |    Yes |
    +--------------------+--------+--------+--------+
    | DENS heater log    |    Yes |    No  |    No  |
    +--------------------+--------+--------+--------+
    | Bruker's bcf       |    Yes |    No  |    Yes |
    +--------------------+--------+--------+--------+
    | EMD (Berkley Labs) |    Yes |    Yes |    Yes |
    +--------------------+--------+--------+--------+
    | Protochips log     |    Yes |    No  |    No  |
    +--------------------+--------+--------+--------+
    | EDAX .spc and .spd |    Yes |    No  |    Yes |
    +--------------------+--------+--------+--------+

.. _hdf5-format:

HDF5
----

This is the default format and it is the only one that guarantees that no
information will be lost in the writing process and that supports saving data
of arbitrary dimensions. It is based on the `HDF5 open standard
<http://www.hdfgroup.org/HDF5/>`_. The HDF5 file format is supported by `many
applications
<http://www.hdfgroup.org/products/hdf5_tools/SWSummarybyName.htm>`_.

Note that only HDF5 files written by HyperSpy are supported

.. versionadded:: 0.8

It is also possible to save more complex structures (i.e. lists, tuples and
signals) in :py:attr:`~.metadata` of the signal. Please note that in order to
increase saving efficiency and speed, if possible, the inner-most structures
are converted to numpy arrays when saved. This procedure homogenizes any types
of the objects inside, most notably casting numbers as strings if any other
strings are present:

.. code-block:: python

    >>> # before saving:
    >>> somelist
    [1, 2.0, 'a name']
    >>> # after saving:
    ['1', '2.0', 'a name']

The change of type is done using numpy "safe" rules, so no information is lost,
as numbers are represented to full machine precision.

This feature is particularly useful when using
:py:meth:`~._signals.EDSSEMSpectrum.get_lines_intensity` (see :ref:`get lines
intensity<get_lines_intensity>`):

.. code-block:: python

    >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
    >>> s.metadata.Sample.intensities = s.get_lines_intensity()
    >>> s.save('EDS_spectrum.hdf5')

    >>> s_new = hs.load('EDS_spectrum.hdf5')
    >>> s_new.metadata.Sample.intensities
    [<BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: Al_Ka at 1.49 keV, dimensions: (|)>,
     <BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: C_Ka at 0.28 keV, dimensions: (|)>,
     <BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: Cu_La at 0.93 keV, dimensions: (|)>,
     <BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: Mn_La at 0.63 keV, dimensions: (|)>,
     <BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: Zr_La at 2.04 keV, dimensions: (|)>]



Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^^
compression: One of None, 'gzip', 'szip', 'lzf'.

'gzip' is the default


.. _netcdf-format:

NetCDF
------

This was the default format in HyperSpy's predecessor, EELSLab, but it has been
superseeded by :ref:`HDF5` in HyperSpy. We provide only reading capabilities
but we do not support writing to this format.

Note that only NetCDF files written by EELSLab are supported.

To use this format a python netcdf interface must be installed manually because
it is not installed by default when using the automatic installers.


.. _mrc-format:

MRC
---

This is a format widely used for tomographic data. Our implementation is based
on `this specification
<http://ami.scripps.edu/software/mrctools/mrc_specification.php>`_. We also
partly support FEI's custom header. We do not provide writing features for this
format, but, as it is an open format, we may implement this feature in the
future on demand.

For mrc files ``load`` takes the ``mmap_mode`` keyword argument enabling
loading the file using a different mode (default is copy-on-write) . However,
note that lazy loading does not support in-place writing (i.e lazy loading and
the "r+" mode are incompatible).

.. _msa-format:

EMSA/MSA
--------

This `open standard format
<http://www.amc.anl.gov/ANLSoftwareLibrary/02-MMSLib/XEDS/EMMFF/EMMFF.IBM/Emmff.Total>`_
is widely used to exchange single spectrum data, but it does not support
multidimensional data. It can be used to exchange single spectra with Gatan's
Digital Micrograph.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^^

For the MSA format the msa_format argument is used to specify whether the
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
<http://www.nist.gov/lispix/doc/image-file-formats/raw-file-format.htm>`_ is
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
Christoph Gohlke's tifffile library. In particular it supports reading and
writing of TIFF, BigTIFF, OME-TIFF, STK, LSM, NIH, and FluoView files. Most of
these are uncompressed or losslessly compressed 2**(0 to 6) bit integer,16, 32
and 64-bit float, grayscale and RGB(A) images, which are commonly used in
bio-scientific imaging. See `the library webpage
<http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html>`_ for more details.

.. versionadded: 1.0
   Add support for writing/reading scale and unit to tif files to be read with
   ImageJ or DigitalMicrograph

Currently HyperSpy has limited support for reading and saving the TIFF tags.
However, the way that HyperSpy reads and saves the scale and the units of tiff
files is compatible with ImageJ/Fiji and Gatan Digital Micrograph softwares.
HyperSpy can also import the scale and the units from tiff files saved using
FEI and Zeiss SEM softwares.

.. code-block:: python

    >>> # Force read image resolution using the x_resolution, y_resolution and
    >>> # the resolution_unit of the tiff tags. Be aware, that most of the
    >>> # software doesn't (properly) use these tags when saving tiff files.
    >>> s = hs.load('file.tif', force_read_resolution=True)

HyperSpy can also read and save custom tags through Christoph Gohlke's tifffile
library. See `the library webpage
<http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html>`_ for more details.

.. code-block:: python

    >>> # Saving the string 'Random metadata' in a custom tag (ID 65000)
    >>> extratag = [(65000, 's', 1, "Random metadata", False)]
    >>> s.save('file.tif', extratags=extratag)

    >>> # Saving the string 'Random metadata' from a custom tag (ID 65000)
    >>> s2 = hs.load('file.tif')
    >>> s2.original_metadata['Number_65000']
    b'Random metadata'

.. _dm3-format:

Gatan Digital Micrograph
------------------------

HyperSpy can read both dm3 and dm4 files but the reading features are not
complete (and probably they will be unless Gatan releases the specifications of
the format). That said, we understand that this is an important feature and if
loading a particular Digital Micrograph file fails for you, please report it as
an issue in the `issues tracker <github.com/hyperspy/hyperspy/issues>`_ to make
us aware of the problem.

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
open an issue in the `issues tracker <github.com/hyperspy/hyperspy/issues>`_ to
alert the developers of the need.

For further reference, file specifications for the formats are
available publicly available from EDAX and are on Github
(`.spc <https://github.com/hyperspy/hyperspy/files/29506/SPECTRUM-V70.pdf>`_,
`.spd <https://github.com/hyperspy/hyperspy/files/29505/
SpcMap-spd.file.format.pdf>`_, and
`.ipr <https://github.com/hyperspy/hyperspy/files/29507/ImageIPR.pdf>`_).

.. _fei-format:

FEI TIA ser and emi
-------------------

HyperSpy can read ``ser`` and ``emi`` files but the reading features are not
complete (and probably they will be unless FEI releases the specifications of
the format). That said we know that this is an important feature and if loading
a particular ser or emi file fails for you, please report it as an issue in the
`issues tracker <github.com/hyperspy/hyperspy/issues>`_ to make us aware of the
problem.

HyperSpy (unlike TIA) can read data directly from the ``.ser`` files. However,
by doing so, the information that is stored in the emi file is lost.
Therefore strongly recommend to load using the ``.emi`` file instead.

When reading an ``.emi`` file if there are several ``.ser`` files associated
with it, all of them will be read and returned as a list.

.. _unf-format:

SEMPER unf binary format
------------------------

SEMPER is a fully portable system of programs for image processing, particularly
suitable for applications in electron microscopy developed by Owen Saxton (see
DOI: 10.1016/S0304-3991(79)80044-3 for more information).The unf format is a
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
<http://docs.scipy.org/doc/numpy/referen-ce/generated/numpy.memmap.html>`_ .
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


.. _bcf-format:

Bruker composite file
----------------

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
^^^^^^^^^^^^^^^^^^^^^^^
select_type: One of ('spectrum', 'image'). If specified just selected type of
data is returned. (default None)

index: index of dataset in bcf v2 files, which can hold few datasets (delaut 0)

downsample: the downsample ratio of hyperspectral array (hight and width only),
can be integer >=1, where '1' results in no downsampling (default 1). The
underlying method of downsampling is unchangeable: sum. Differently than
block_reduce from skimage.measure it is memory efficient (does not creates
intermediate arrays, works inplace).

cutoff_at_kV: if set (can be int of float >= 0) can be used either to crop or
enlarge energy (or channels) range at max values. (default None)

Example of loading reduced (downsampled, and with energy range cropped)
"spectrum only" data from bcf (original shape: 80keV EDS range (4096 channels),
100x75 pixels):

.. code-block:: python

    >>> hs.load("sample80kv.bcf", select_type='spectrum', downsample=2, cutoff_at_kV=10)
    <EDSSEMSpectrum, title: EDX, dimensions: (50, 38|595)>

load the same file without extra arguments:

.. code-block:: python

    >>> hs.load("sample80kv.bcf")
    [<Image, title: BSE, dimensions: (|100, 75)>,
    <Image, title: SE, dimensions: (|100, 75)>,
    <EDSSEMSpectrum, title: EDX, dimensions: (100, 75|1095)>]

The loaded array energy dimension can by forced to be larger than the data
recorded by setting the 'cutoff_at_kV' kwarg to higher value:

.. code-block:: python

    >>> hs.load("sample80kv.bcf", cutoff_at_kV=80)
    [<Image, title: BSE, dimensions: (|100, 75)>,
    <Image, title: SE, dimensions: (|100, 75)>,
    <EDSSEMSpectrum, title: EDX, dimensions: (100, 75|4096)>]

Note that setting downsample to >1 currently locks out using SEM imagery
as navigator in the plotting.


.. _emd-format:

EMD Electron Microscopy Datasets (HDF5)
---------------------------------------

EMD stands for “Electron Microscopy Dataset.” It is a subset of the open source
HDF5 wrapper format. N-dimensional data arrays of any standard type can be
stored in an HDF5 file, as well as tags and other metadata. The EMD format was
developed at Lawrence Berkeley National Lab (see http://emdatasets.lbl.gov/ for
more information). NOT to be confused with the FEI EMD format which was
developed later and has a different structure.

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
