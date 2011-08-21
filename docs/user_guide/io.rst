.. _io:

***********************
Loading and saving data
***********************

.. contents::

.. _loading_files:

Loading files: the load function
================================

Hyperspy can read and write to multiple formats (see :ref:`supported-formats`). To load data use the :py:func:`~.io.load` that it is available by default in the main namespace. For example, to load a ``jpg`` image from :file:`lena` you can type

.. code-block:python
    s = load('lena.jpg')
    
If the loading was successful, the variable :guilabel:`s` contains a generic :py:class:`~.signal.Signal, a :py:class:`~.signals.spectrum.Spectrum or a :py:class:`~.signals.image.Image. In any case, the data is stored in a numpy array in the :py:attr:`~.signal.Signal.data` attribute, but you will not normally need to access it there.

Hyperspy will try to guess the 
most convenient object for the corresponding file. However, you can force it to read the data as a particular data type by providing the ``signal`` keyword that has to be one of: ``spectrum``, ``image`` or ``EELS``, e.g.:

.. code-block:: python

    s = load('filename', signal = 'EELS')

Some file formats store some extra information about the data. If Hyperspy was able to read some extra information it stores it in :py:attr:`~.signal.Signal.original_parameters` attribute. Also, it is possible that some information was mapped by Hyperspy to a standard location where it can be used by some standard routines, the :py:attr:`~.signal.Signal.mapped_parameters` attribute.

In addition Hyperspy supports reading collections of files, see :ref:`aggregate`

.. _saving_files:

Saving data to files
====================

To save data to a file use the :py:meth:`~.signal.Signal.save` method of the :py:class:`~.signal.Signal` class or its subclasses. The first argument is the filename and the format is defined by the filename extension. If the filename does not contain the extension the default format (:ref:`hdf5-format`) is used. For example, if the :py:const:`s` variable contains the :py:class:`~.signal.Signal` that you want to write to a file, the following will write the data to a file called :file:`spectrum.hdf5` in the default :ref:`hdf5-format` format::
    
    s.save('spectrum')
    
If instead you want to save in the :ref:`ripple-format` write instead::

    s.save('spectrum.rpl')

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
    | Image              | Complete  | Complete |
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

.. _netcdf-format:

NetCDF
------

It was the default format in EELSLab but it has been superseeded by :ref:`HDF5` in Hyperspy. We provide only reading capabilities but we do not support writing to this format.

Note that only NetCDF files written by EELSLab are supported.


.. _mrc-format:

MRC
---

It is a format widely used for tomographic data. Our implementation is based on 
`this specification <http://ami.scripps.edu/software/mrctools/mrc_specification.php>`_. We also partly support FEI's custom header. We do not provide writing features for this format, but, being an open format, we may implement this feature in the future if it is demanded by the users.

.. _msa-format:

EMSA/MSA
--------

This `open standard format <http://www.amc.anl.gov/ANLSoftwareLibrary/02-MMSLib/XEDS/EMMFF/EMMFF.IBM/Emmff.Total>`_ is widely used to exchange single spectrum data, but it does not support multidimensional data. It can be used to exchange single spectrum with Gatan Digital Micrograph.

Extra arguments
^^^^^^^^^^^^^^^
For the MSA format the msa_format argument is used to specify whether the energy axis should also be saved with the data.  The default, 'Y' omits the energy axis in the file.  The alternative, 'XY', saves a second column with the calibrated energy data. Also it  is possible to personalise the separator with the `separator` keyword. 

.. Warning::

    However, if a different separator is chosen the resulting file will not comply with the MSA/EMSA standard and Hyperspy and other software may not be able to read it.

.. _ripple-format:

Ripple
------

This `open standard format <http://www.nist.gov/lispix/doc/image-file-formats/raw-file-format.htm>`_ is widely used to exchange hyperspectra data. However, it only support data of up to three dimensions. It can be used to exchange data with Bruker and Lispix. Installing the :ref:`import-rpl` it is very useful to export data to Gatan Digital Micrograph.

.. _image-format:

Image
-----

Hyperspy is able to read and write data too all the image formats supported by `the Python Image Library <http://www.pythonware.com/products/pil/>`_ (PIL). 

It is important to note that all the image formats only support 8-bit files, what may incur in dynamic range loss in most cases.

It is possible (and strongly reccommend if saving to an image format is required) to read and write 16-bit images in the TIFF format by installing `mahotas <http://pypi.python.org/pypi/mahotas>`_ and the `freeimage library <http://freeimage.sourceforge.net/>`_.
 
.. _dm3-format:

Gatan Digital Micrograph
------------------------

Hyperspy only support reading dm3 files and the reading features are not complete (and probably they will never be because it is not an open standard format). That said we know that this is an important feature and if loading a particular dm3 file fails for you, please report an issue in the `issues tracker <github.com/hyperspy/hyperspy/issues>`_ to make us aware of the problem. 

.. _fei-format:

FEI TIA ser and emi
-------------------

Hyperspy only support reading ``ser`` and ``emi`` files and the reading features are not complete (and probably they will never be because it is not an open standard format). That said we know that this is an important feature and if loading a particular ser or emi file fails for you, please report an issue in the `issues tracker <github.com/hyperspy/hyperspy/issues>`_ to make us aware of the problem.

In Hyperspy (and unlike in TIA) it is possible to read the data directly from the ``.ser`` files. However, by doing so, the experiment information that is stored in the emi file is lost. Therefore it is reccommend to load using the ``.emi`` file.

.. _aggregate:

Aggregating data
================

Loading Aggregate Files
------------------------

Rather than loading and working with files individually, several files
can be loaded simultaneously, creating an aggregate object.  In
aggregate objects, all of the data from each set is collected into a
single data set.  Any analysis you do runs on the aggregate set, and
any results are split into separate files afterwards.

Here's an example of loading multiple files:

.. code-block:: python

    d=load('file1.ext', 'file2.ext')

If you just want to load, say, all tif files in a folder into an
aggregate, you can do something like this:

.. code-block:: python

    from glob import glob
    d=load(*glob('*.tif'))

glob('*.tif') returns a python list of the files.  Preceding the
function with the * transforms that list into a comma-separated
collection of the files - effectively imitating the previous code
block of comma-separated filenames.

Files can be added to aggregates in one of two ways, both using the
append function on any existing Aggregate object.

Adding files that are not yet loaded (passing filenames):

.. code-block:: python

    d.append('file3.ext')


Adding files that are already loaded (passing objects):

.. code-block:: python

    d2=load('file3.ext')
    d.append(d2)

Of course, the object types must match - you cannot aggregate spectrum
images with normal images.

Notes:
Presently, aggregate spectra are not checked for any energy
alignment.  You must have similar energy ranges, with similar numbers
of channels on all files you wish to aggregate.

Images are stacked along the 3rd dimension.  Any images you aggregate must
have similar dimensions in terms of pixel size.  The aggregator does
not check for calibrated size.  It does not physically make sense to
aggregate images with differing fields of view.

For the future of aggregate spectra, the goal is that each file must
share at least some part of their energy range.  The aggregate energy
range will be automatically truncated to include only the union of all
energy ranges.  Interpolation will be used in case of any channel mismatch
between data sets.

Saving Aggregate files
-------------------------

Aggregate files are saved similarly to other Signal based classes,
however, depending on the file format, several files will be created.
HDF5, the preferred format, will save one file containing the entire
hierarchy of the aggregate.  Other formats will create folder
structures, placing files of the desired format in folders according
to their place in the aggregate hierarchy.

Loading Saved Aggregate Files
--------------------------------

Please, please use the HDF5 file format.  It will make your life
easier.  To load an hdf5 aggregate data set, use the simple load
command:

.. code-block:: python

    d=load('filename.hdf5')

For all other formats, the folder hierarchy created when the aggregate
was saved must remain exactly the same, or the aggregate will no
longer load properly.  Do not delete, move, or edit files from the
automatically created folders.  When saved, a file consisting of a
table of contents of the aggregate is created.  To load the aggregate, 
provide this file to the load function:

.. code-block:: python

    d=load('filename_agg_contents.txt')
