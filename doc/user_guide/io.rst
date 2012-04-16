.. _io:

***********************
Loading and saving data
***********************

.. contents::

.. _loading_files:

Loading files: the load function
================================

Hyperspy can read and write to multiple formats (see :ref:`supported-formats`). To load data use the :py:func:`~.io.load` that it is available by default in the main namespace. For example, to load a ``jpg`` image from :file:`lena` you can type:

.. code-block:: python

    >>> s = load("lena.jpg")
    
If the loading was successful, the variable :guilabel:`s` contains a generic :py:class:`~.signal.Signal`, a :py:class:`~.signals.spectrum.Spectrum` or a :py:class:`~.signals.image.Image`. In any case, the data is stored in a numpy array in the :py:attr:`~.signal.Signal.data` attribute, but you will not normally need to access it there.

Hyperspy will try to guess the most convenient object for the corresponding file. However, you can force it to read the data as a particular data type by providing the ``signal`` keyword that has to be one of: ``spectrum``, ``image`` or ``EELS``, e.g.:

.. code-block:: python

    >>> s = load("filename", signal = "EELS")

Some file formats store some extra information about the data. If Hyperspy was able to read some extra information it stores it in :py:attr:`~.signal.Signal.original_parameters` attribute. Also, it is possible that some information was mapped by Hyperspy to a standard location where it can be used by some standard routines, the :py:attr:`~.signal.Signal.mapped_parameters` attribute.

To print the content of the parameters simply:

.. code-block:: python

    >>> s.mapped_parameters


The :py:attr:`~.signal.Signal.original_parameters` and :py:attr:`~.signal.Signal.mapped_parameters` can be exported to a text files using the :py:meth:`~.misc.utils.DictionaryBrowser.export` method, e.g.:

.. code-block:: python
    
    # The following command stores the original parameters in the parameters.txt file
    >>> s.original_parameters.export('parameters')

Loading multiple files
----------------------

Rather than loading files individually, several files
can be loaded with a single command. This can be done by passing a list of filenames to the load functions, e.g.:

.. code-block:: python

    >>> s = load(["file1.hdf5", "file2.hdf5"])
    
or by using `shell-style wildcards <http://docs.python.org/library/glob.html>`_

.. code-block:: python

    >>> s = load("file*.hdf5",)
    
By default Hyperspy will try to stack all the files in a single file, but for this to work all the files need to contain data with exactly the same dimensions. If this is not the case an error is raised.

It is also possible to load multiple files with a single command without stacking them by passing the stack=False argument to the load function, in which case the function will return a list of objects, e.g.:

.. code-block:: python

    >>> s = load("file*.hdf5", stack = False)

.. _saving_files:

Saving data to files
====================

To save data to a file use the :py:meth:`~.signal.Signal.save` method of the :py:class:`~.signal.Signal` class or its subclasses. The first argument is the filename and the format is defined by the filename extension. If the filename does not contain the extension the default format (:ref:`hdf5-format`) is used. For example, if the :py:const:`s` variable contains the :py:class:`~.signal.Signal` that you want to write to a file, the following will write the data to a file called :file:`spectrum.hdf5` in the default :ref:`hdf5-format` format::

.. code-block:: python

    >>> s.save('spectrum')
    
If instead you want to save in the :ref:`ripple-format` write instead:

.. code-block:: python

    >>> s.save('spectrum.rpl')

Some formats take extra arguments. See the relevant subsection of :ref:`supported-formats` for more information.


.. _supported-formats:

Supported formats
=================

In :ref:`supported-file-formats-table` we summarise the different formats that are currently supported by Hyperspy.

.. _supported-file-formats-table:

.. table:: Supported file formats

    +--------------------+-----------+----------+
    | Format             | Read      | Write    |
    +====================+===========+==========+
    | Gatan's dm3        | Partial   | -        |
    +--------------------+-----------+----------+
    | FEI's emi and ser  | Partial   | -        |
    +--------------------+-----------+----------+
    | HDF5               | Complete  | Complete |
    +--------------------+-----------+----------+
    | Image: jpg..       | Complete  | Complete |
    +--------------------+-----------+----------+
    | TIFF               | Complete  | Complete |
    +--------------------+-----------+----------+
    | MRC                | Complete  | -        |
    +--------------------+-----------+----------+
    | EMSA/MSA           | Complete  | Complete |
    +--------------------+-----------+----------+
    | NetCDF             | Complete  | -        |
    +--------------------+-----------+----------+
    | Ripple             | Complete  | Complete |
    +--------------------+-----------+----------+

.. _hdf5-format:

HDF5
----

It is the default format and it is the only one that guarantees that no information will be lost in the writing process and that support saving data of arbitrary dimensions. It is based in the `HDF5 open standard <http://www.hdfgroup.org/HDF5/>`_. The HDF5 file format is supported by `many applications <http://www.hdfgroup.org/products/hdf5_tools/SWSummarybyName.htm>`_.

Note that only HDF5 files written by Hyperspy are supported.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^^
compression: One of None, 'gzip', 'szip', 'lzf'.

'gzip' is the default


.. _netcdf-format:

NetCDF
------

It was the default format in EELSLab but it has been superseeded by :ref:`HDF5` in Hyperspy. We provide only reading capabilities but we do not support writing to this format.

Note that only NetCDF files written by EELSLab are supported.

To use this format a python netcdf interface must be installed manually because it is not installed by default when using the automatic installers.


.. _mrc-format:

MRC
---

It is a format widely used for tomographic data. Our implementation is based on 
`this specification <http://ami.scripps.edu/software/mrctools/mrc_specification.php>`_. We also partly support FEI's custom header. We do not provide writing features for this format, but, being an open format, we may implement this feature in the future on demand.

.. _msa-format:

EMSA/MSA
--------

This `open standard format <http://www.amc.anl.gov/ANLSoftwareLibrary/02-MMSLib/XEDS/EMMFF/EMMFF.IBM/Emmff.Total>`_ is widely used to exchange single spectrum data, but it does not support multidimensional data. It can be used to exchange single spectrum with Gatan Digital Micrograph.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^^
For the MSA format the msa_format argument is used to specify whether the energy axis should also be saved with the data.  The default, 'Y' omits the energy axis in the file.  The alternative, 'XY', saves a second column with the calibrated energy data. It  is possible to personalise the separator with the `separator` keyword. 

.. Warning::

    However, if a different separator is chosen the resulting file will not comply with the MSA/EMSA standard and Hyperspy and other software may not be able to read it.
    
The default encoding is `latin-1`. It is possible to set a different encoding using the `encoding` argument, e.g.:

.. code-block:: python

    >>> s.save('file.msa', encoding = 'utf8')



.. _ripple-format:

Ripple
------

This `open standard format <http://www.nist.gov/lispix/doc/image-file-formats/raw-file-format.htm>`_ is widely used to exchange hyperspectra data. However, it only support data of up to three dimensions. It can be used to exchange data with Bruker and `Lispix <http://www.nist.gov/lispix/>`. Installing the :ref:`import-rpl` it is very useful to export data to Gatan Digital Micrograph.

The default encoding is latin-1. It is possible to set a different encoding using the encoding argument, e.g.:

.. code-block:: python

    >>> s.save('file.rpl', encoding = 'utf8')

.. _image-format:

Image
-----

Hyperspy is able to read and write data too all the image formats supported by `the Python Image Library <http://www.pythonware.com/products/pil/>`_ (PIL). This includes png, pdf, gif etc.

It is important to note these image formats only support 8-bit files, what may incur in dynamic range loss in most cases. It is therefore highly discouraged to use any image format (with the exception of :ref:`tiff-format` that uses another library) to store data for analysis purposes.

.. _tiff-format:
    
TIFF
----

Since version 4.1 Hyperspy can read and write 2D and 3D TIFF files using using Christoph Gohlke's tifffile library. In particular it supports can be reading and writing TIFF, BigTIFF, OME-TIFF, STK, LSM, NIH,
and FluoView files,mainly uncompressed and losslessly compressed 2**(0 to 6) bit integer,16, 32 and 64-bit float, grayscale and RGB(A) images, which are commonly
used in bio-scientific imaging. See `the library webpage <http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html>`_ for more details.

Currently Hyperspy cannot read the TIFF tags.


 
.. _dm3-format:

Gatan Digital Micrograph
------------------------

Hyperspy supports reading dm3 files but the reading features are not complete (and probably they will never be because it is not an open standard format). That said we know that this is an important feature and if loading a particular dm3 file fails for you, please report an issue in the `issues tracker <github.com/hyperspy/hyperspy/issues>`_ to make us aware of the problem. 

.. _fei-format:

FEI TIA ser and emi
-------------------

Hyperspy supports reading ``ser`` and ``emi`` files but the reading features are not complete (and probably they will never be because it is not an open standard format). That said we know that this is an important feature and if loading a particular ser or emi file fails for you, please report an issue in the `issues tracker <github.com/hyperspy/hyperspy/issues>`_ to make us aware of the problem.

Hyperspy (unlike in TIA) can read data directly from the ``.ser`` files. However, by doing so, the experiment information that is stored in the emi file is lost. Therefore it is reccommend to load using the ``.emi`` file.

When reading an ``.emi`` file if there are several ``.ser`` files associated to it, all of them will be read and returned as a list.


