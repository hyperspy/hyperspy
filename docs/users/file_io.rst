File Input and Output
+++++++++++++++++++++

Loading Data
==============

EELSlab has a unified loading command, regardless of what format your
data is in.  Once you have started EELSlab, load your data into a
variable using a command like this:

.. code-block:: python

d=load('your file name.ext')


Loading Aggregate Files
---------------------------
Additionally, several files can be loaded simultaneously, creating an
aggregate object.  In aggregate objects, all of the data from each set
is collected into a single data set.  Any analysis you do runs on the
aggregate set, and any results are split into separate files
afterwards.

Here's an example of loading multiple files:

.. code-block:: python

d=load('file1.ext', 'file2.ext')

If you just want to load, say, all tif files in a folder into an
aggregate, you can do something like this:

.. code-block:: python

from glob import glob
d=load(*glob('*.tif'))

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
THIS IS THE GOAL, NOT HOW THINGS CURRENTLY OPERATE!
Spectrum images are aggregated spectrally.  They must share at least
some part of their energy range.  The aggregate energy range will be
automatically truncated to include only the union of all energy
ranges.  

Images are stacked along the 3rd dimension.  Any images you aggregate must
have similar dimensions in terms of pixel size.  The aggregator does
not check for calibrated size.  It does not physically make sense to
aggregate images with differing fields of view.
