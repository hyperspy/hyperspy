Aggregating data
******************************

Loading Aggregate Files
---------------------------

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
