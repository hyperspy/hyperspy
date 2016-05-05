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
:py:class:`~.signal.BaseSignal`, a :py:class:`~._signals.signal1d.Signal1D` or an
:py:class:`~._signals.signal2d.Signal2D`.

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
about the data it stores it in :py:attr:`~.signal.BaseSignal.original_metadata`
attribute. Also, it is possible that other information will be mapped by
HyperSpy to a standard location where it can be used by some standard routines,
the :py:attr:`~.signal.BaseSignal.metadata` attribute.

To print the content of the parameters simply:

.. code-block:: python

    >>> s.metadata


The :py:attr:`~.signal.BaseSignal.original_metadata` and
:py:attr:`~.signal.BaseSignal.metadata` can be exported to  text files
using the :py:meth:`~.misc.utils.DictionaryTreeBrowser.export` method, e.g.:

.. code-block:: python

    >>> s.original_metadata.export('parameters')

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
dimensions. If this is not the case an error is raised.

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
contains the :py:class:`~.signal.BaseSignal` that you want to write to a file, the
following will write the data to a file called :file:`spectrum.hdf5` in the
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
HyperSpy.


.. table:: Supported file formats

    +--------------------+-----------+----------+
    | Format             | Read      | Write    |
    +====================+===========+==========+
    | Gatan's dm3        |    Yes    |    No    |
    +--------------------+-----------+----------+
    | Gatan's dm4        |    Yes    |    No    |
    +--------------------+-----------+----------+
    | FEI's emi and ser  |    Yes    |    No    |
    +--------------------+-----------+----------+
    | HDF5               |    Yes    |    Yes   |
    +--------------------+-----------+----------+
    | Image: jpg..       |    Yes    |    Yes   |
    +--------------------+-----------+----------+
    | TIFF               |    Yes    |    Yes   |
    +--------------------+-----------+----------+
    | MRC                |    Yes    |    No    |
    +--------------------+-----------+----------+
    | EMSA/MSA           |    Yes    |    Yes   |
    +--------------------+-----------+----------+
    | NetCDF             |    Yes    |    No    |
    +--------------------+-----------+----------+
    | Ripple             |    Yes    |    Yes   |
    +--------------------+-----------+----------+

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

It is also possible to save more complex structures (i.e. lists, tuples and signals) in
:py:attr:`~.metadata` of the signal. Please note that in order to increase
saving efficiency and speed, if possible, the inner-most structures are
converted to numpy arrays when saved. This procedure homogenizes any types of
the objects inside, most notably casting numbers as strings if any other
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
    [<Signal, title: X-ray line intensity of EDS SEM Spectrum: Al_Ka at 1.49 keV, dimensions: (|)>,
     <Signal, title: X-ray line intensity of EDS SEM Spectrum: C_Ka at 0.28 keV, dimensions: (|)>,
     <Signal, title: X-ray line intensity of EDS SEM Spectrum: Cu_La at 0.93 keV, dimensions: (|)>,
     <Signal, title: X-ray line intensity of EDS SEM Spectrum: Mn_La at 0.63 keV, dimensions: (|)>,
     <Signal, title: X-ray line intensity of EDS SEM Spectrum: Zr_La at 2.04 keV, dimensions: (|)>]



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
format, but, as it is an an open format, we may implement this feature in the
future on demand.

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

Since version 4.1 HyperSpy can read and write 2D and 3D TIFF files using using
Christoph Gohlke's tifffile library. In particular it supports reading and
writing of TIFF, BigTIFF, OME-TIFF, STK, LSM, NIH, and FluoView files. Most of
these are uncompressed or losslessly compressed 2**(0 to 6) bit integer,16, 32
and 64-bit float, grayscale and RGB(A) images, which are commonly used in
bio-scientific imaging. See `the library webpage
<http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html>`_ for more details.

Currently HyperSpy cannot read the TIFF tags.


.. _dm3-format:

Gatan Digital Micrograph
------------------------

HyperSpy can read both dm3 and dm4 files but the reading features are not
complete (and probably they will be unless Gatan releases the specifications of
the format). That said, we understand that this is an important feature and if
loading a particular Digital Micrograph file fails for you, please report it as
an issue in the `issues tracker <github.com/hyperspy/hyperspy/issues>`_ to make
us aware of the problem.

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
Therefore strongly reccommend to load using the ``.emi`` file instead.

When reading an ``.emi`` file if there are several ``.ser`` files associated
with it, all of them will be read and returned as a list.
