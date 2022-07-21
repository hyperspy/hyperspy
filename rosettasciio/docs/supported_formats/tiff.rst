.. _tiff-format:

Tagges image file format (TIFF)
-------------------------------

RosettaSciIO can read and write 2D and 3D ``.tiff`` files using using
Christoph Gohlke's `tifffile <https://github.com/cgohlke/tifffile/>`_ library.
In particular, it supports reading and
writing of TIFF, BigTIFF, OME-TIFF, STK, LSM, NIH, and FluoView files. Most of
these are uncompressed or losslessly compressed 2**(0 to 6) bit integer, 16, 32
and 64-bit float, grayscale and RGB(A) images, which are commonly used in
bio-scientific imaging. See `the library webpage
<http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html>`_ for more details.

.. versionadded: 1.0
   Add support for writing/reading scale and unit to tif files to be read with
   ImageJ or DigitalMicrograph

Currently RosettaSciIO has limited support for reading and saving the TIFF tags.
However, the way that RosettaSciIO reads and saves the scale and the units of ``.tiff``
files is compatible with ImageJ/Fiji and Gatan Digital Micrograph software.
RosettaSciIO can also import the scale and the units from ``.tiff`` files saved using
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
